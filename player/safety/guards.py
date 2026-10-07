"""The guard chain every dispatch passes through.

Guards are evaluated **at dispatch time**, not sampled into a flag that a worker reads
later. The window between "guard tripped" and "next keypress" has to be one dispatch,
not one poll interval — otherwise the kill switch has a latency measured in whatever the
loop period happens to be.

A tripped guard stops dispatch and flushes the timeline. It does not queue work for
later: a plan built for a world state that has since expired is not worth executing when
the guard clears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

from ..clock import RateLimiter, now
from ..state import WorldState


@dataclass(frozen=True, slots=True)
class GuardStatus:
    name: str
    blocking: bool
    reason: str = ""


@runtime_checkable
class Guard(Protocol):
    name: str

    def check(self, state: WorldState | None, at: float) -> GuardStatus: ...


@dataclass(slots=True)
class KillSwitchGuard:
    """The human override. Must always win, and must latch."""

    killswitch: object  # KillSwitch; typed loosely to keep safety free of act imports
    name: str = "killswitch"

    def check(self, state: WorldState | None, at: float) -> GuardStatus:
        ks = self.killswitch
        if getattr(ks, "tripped", False):
            return GuardStatus(self.name, True, "kill switch tripped")
        if getattr(ks, "paused", False):
            return GuardStatus(self.name, True, "paused")
        return GuardStatus(self.name, False)


@dataclass(slots=True)
class ForegroundGuard:
    """Refuses to dispatch unless the game window has focus.

    Both a safety and a correctness property. Safety: the keys go to whatever the human
    just alt-tabbed to otherwise. Correctness: the game ignores input it does not have
    focus for, so dispatching anyway produces a rotation with silent holes in it.
    """

    tracker: object  # WindowTracker
    name: str = "foreground"
    required: bool = True

    def check(self, state: WorldState | None, at: float) -> GuardStatus:
        if not self.required:
            return GuardStatus(self.name, False)
        if not getattr(self.tracker, "available", False):
            return GuardStatus(self.name, True, "window tracking unavailable")
        if not self.tracker.is_foreground():
            return GuardStatus(self.name, True, "target window not focused")
        return GuardStatus(self.name, False)


@dataclass(slots=True)
class StalenessGuard:
    """Refuses to act on a world state older than `max_age_ms`.

    Acting on a stale world is acting blind. This is what turns "capture stalled" from a
    silent behaviour change into an explicit stop.
    """

    max_age_ms: float = 250.0
    name: str = "staleness"

    def check(self, state: WorldState | None, at: float) -> GuardStatus:
        if state is None:
            return GuardStatus(self.name, True, "no world state yet")
        age = state.age_ms(at)
        if age > self.max_age_ms:
            return GuardStatus(self.name, True, f"state {age:.0f}ms old")
        return GuardStatus(self.name, False)


@dataclass(slots=True)
class ConfidenceGuard:
    """Refuses to act when fields the policy depends on are not trustworthy.

    A misread health bar is worse than an unread one: the first produces a confident
    wrong decision, the second produces no decision.
    """

    required_fields: tuple[str, ...] = ()
    min_confidence: float = 0.5
    name: str = "confidence"

    def check(self, state: WorldState | None, at: float) -> GuardStatus:
        if state is None:
            return GuardStatus(self.name, True, "no world state yet")
        if not self.required_fields:
            return GuardStatus(self.name, False)
        bad = state.untrusted(self.required_fields, self.min_confidence)
        if bad:
            shown = ", ".join(bad[:3]) + ("..." if len(bad) > 3 else "")
            return GuardStatus(self.name, True, f"untrusted: {shown}")
        return GuardStatus(self.name, False)


@dataclass(slots=True)
class RateGuard:
    """Hard ceiling on dispatch rate. A backstop, not a pacing mechanism.

    Pacing belongs in the rotation planner, which knows what the game's cooldowns are.
    This exists to bound the damage when a bug turns the decision loop into a key-spam
    generator.
    """

    max_per_sec: int = 30
    name: str = "rate"
    _limiter: RateLimiter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.max_per_sec)

    def check(self, state: WorldState | None, at: float) -> GuardStatus:
        # Probe only: `consume` is what actually spends budget, called by the
        # dispatcher once it commits to sending. The probe must let the window
        # slide, or a full window would block every later consume and never empty.
        if self._limiter.count(at) >= self.max_per_sec:
            return GuardStatus(self.name, True, f"> {self.max_per_sec}/s")
        return GuardStatus(self.name, False)

    def consume(self, at: float | None = None) -> bool:
        return self._limiter.allow(at)

    def reset(self) -> None:
        self._limiter.reset()


@dataclass(slots=True)
class TakeoverGuard:
    """Yields to the human.

    If a key was pressed that the dispatcher did not send, back off. This is mostly a
    usability property — it is what lets you grab the keyboard without racing the player
    — but it also prevents the two of you from fighting over a rotation.

    Depends on the kill switch's hook to distinguish human input from our own; without
    that hook installed this guard is inert rather than permanently blocking, because a
    guard that always blocks is a guard nobody keeps enabled.
    """

    killswitch: object
    cooldown_ms: float = 1500.0
    name: str = "takeover"
    enabled: bool = True

    def check(self, state: WorldState | None, at: float) -> GuardStatus:
        if not self.enabled:
            return GuardStatus(self.name, False)
        last = getattr(self.killswitch, "last_human_key_at", 0.0)
        if last <= 0.0:
            return GuardStatus(self.name, False)
        elapsed_ms = (at - last) * 1000.0
        if elapsed_ms < self.cooldown_ms:
            return GuardStatus(self.name, True, f"human input {elapsed_ms:.0f}ms ago")
        return GuardStatus(self.name, False)


class SafetyGate:
    """Evaluates every guard and reports whether dispatch may proceed.

    Order matters only for reporting: all guards are evaluated so the status line can
    show everything currently blocking rather than just the first thing. That turns "why
    is it not doing anything" from an investigation into a glance.
    """

    def __init__(self, guards: Iterable[Guard] | None = None) -> None:
        self.guards: list[Guard] = list(guards or [])
        self._last: list[GuardStatus] = []
        self.block_counts: dict[str, int] = {}
        self.was_blocked = False

    def add(self, guard: Guard) -> "SafetyGate":
        self.guards.append(guard)
        return self

    def evaluate(self, state: WorldState | None, at: float | None = None) -> list[GuardStatus]:
        at = now() if at is None else at
        statuses = []
        for guard in self.guards:
            try:
                status = guard.check(state, at)
            except Exception as exc:
                # A guard that throws must fail closed. An exception in the safety layer
                # is not a reason to start dispatching.
                status = GuardStatus(getattr(guard, "name", "?"), True, f"guard error: {exc}")
            statuses.append(status)
            if status.blocking:
                self.block_counts[status.name] = self.block_counts.get(status.name, 0) + 1
        self._last = statuses
        return statuses

    def allowed(self, state: WorldState | None, at: float | None = None) -> bool:
        blocked = any(s.blocking for s in self.evaluate(state, at))
        self.was_blocked = blocked
        return not blocked

    def blocking(self) -> list[GuardStatus]:
        return [s for s in self._last if s.blocking]

    def status(self) -> str:
        blocking = self.blocking()
        if not blocking:
            return "clear"
        return "BLOCKED: " + "; ".join(f"{s.name}({s.reason})" for s in blocking)

    def consume_rate(self, at: float | None = None) -> bool:
        """Spend one unit of the rate budget. Called once dispatch commits."""
        for guard in self.guards:
            if isinstance(guard, RateGuard):
                return guard.consume(at)
        return True
