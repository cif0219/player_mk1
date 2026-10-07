"""The deadline-scheduled input timeline.

The obvious executor is a queue plus a worker that sleeps a fixed delay after each
action. It has a fatal property: everything serialises behind the previous action's
sleep, so a reflex that decided in one millisecond still waits out whatever the rotation
queued before it. The reaction time you measured in the decision layer is not the
reaction time the game sees.

So: a priority queue keyed on absolute monotonic deadlines, with key-down and key-up as
separate scheduled events. Three properties follow, all of which this workload needs.

*Precise holds.* "Press 2 for 40ms" is two scheduled events, not a blocking sleep, so
scheduling it costs nothing and holding it blocks nothing else.

*Weave windows fall out for free.* FFXIV's off-GCD weaving means "GCD at t, animation
lock until t+600ms, off-GCD at t+620ms, next GCD at t+2500ms". That is four deadlines.
Expressing it against a sleeping worker means computing the residual sleep by hand at
every step and getting it wrong once.

*Cancellation means something.* A reflex firing with `preempt=True` flushes what is
scheduled and inserts a dodge. Against a sleeping worker there is nothing to flush — the
sleep already happened.

One invariant is absolute: **a key-up is never silently dropped.** Preemption and expiry
both convert pending key-ups into immediate ones. A stuck key in a game does more damage
than any missed action.
"""

from __future__ import annotations

import heapq
import itertools
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

from ..clock import now
from ..geometry import ScreenPoint


class Priority(IntEnum):
    """Higher wins. Equal priorities never preempt each other."""

    IDLE = 10
    ROTATION = 50
    RECOVERY = 75
    REFLEX = 100


# -- primitive input events -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeyDown:
    key: str


@dataclass(frozen=True, slots=True)
class KeyUp:
    key: str


@dataclass(frozen=True, slots=True)
class MouseMove:
    point: ScreenPoint


@dataclass(frozen=True, slots=True)
class Click:
    button: str = "left"
    point: ScreenPoint | None = None


InputEvent = KeyDown | KeyUp | MouseMove | Click


# -- plan-level actions -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Press:
    """A key held for `hold_ms`.

    Games sample key state rather than consuming keystroke messages, so a down/up pair in
    the same frame is often missed entirely. 30–50ms is the reliable range; the default
    sits in the middle of it.
    """

    key: str
    hold_ms: int = 40


Action = Press | KeyDown | KeyUp | MouseMove | Click


@dataclass(frozen=True, slots=True)
class Step:
    """One action at an offset from the plan's start."""

    at_ms: int
    action: Action


@dataclass(frozen=True, slots=True)
class Plan:
    """A relative schedule emitted by policy.

    Relative, not absolute: the timeline resolves offsets against the clock at submission
    time. That is what makes a plan comparable across a live run and a replay, and what
    makes it testable without a clock at all.
    """

    name: str
    steps: tuple[Step, ...]
    priority: Priority = Priority.ROTATION
    preempt: bool = False
    expires_in_ms: int = 2000

    @property
    def duration_ms(self) -> int:
        if not self.steps:
            return 0
        return max(
            s.at_ms + (s.action.hold_ms if isinstance(s.action, Press) else 0)
            for s in self.steps
        )

    @staticmethod
    def single(
        key: str,
        *,
        name: str | None = None,
        hold_ms: int = 40,
        priority: Priority = Priority.ROTATION,
        preempt: bool = False,
        expires_in_ms: int = 2000,
    ) -> "Plan":
        return Plan(
            name=name or f"press:{key}",
            steps=(Step(0, Press(key, hold_ms)),),
            priority=priority,
            preempt=preempt,
            expires_in_ms=expires_in_ms,
        )

    def hovered(self, point: ScreenPoint, settle_ms: int = 40) -> "Plan":
        """Prefix a mouse move for mouseover targeting.

        Distinct from `targeted` because it does not change the hard target — the boss
        stays selected, so the next damage ability still lands on it. That property is why
        FFXIV healers use mouseover rather than slot keys, and it is worth preserving in
        the model rather than treating both as "targeting".
        """
        steps = [Step(0, MouseMove(point))]
        steps.extend(Step(s.at_ms + settle_ms, s.action) for s in self.steps)
        return Plan(
            name=self.name,
            steps=tuple(steps),
            priority=self.priority,
            preempt=self.preempt,
            expires_in_ms=max(200, self.expires_in_ms - settle_ms),
        )

    def targeted(self, target_keys: tuple[str, ...], settle_ms: int = 60) -> "Plan":
        """Prefix target acquisition onto this plan, shifting everything else back.

        This is what makes a target swap cost tens of milliseconds instead of a global
        cooldown. The target key and the ability it feeds are steps in the *same*
        schedule, so they are preempted together, expire together, and cannot be
        separated by a reflex firing in between — which is exactly the failure that
        produces a heal cast on the boss.

        `settle_ms` is the gap the game needs to register the new target before the
        ability reads it. Three or four frames is reliable; zero is not, because the
        ability would be pressed on the same frame as the target change and resolve
        against the old one.
        """
        if not target_keys:
            return self
        steps = [Step(i * 20, Press(key, 30)) for i, key in enumerate(target_keys)]
        offset = (len(target_keys) - 1) * 20 + settle_ms
        steps.extend(Step(s.at_ms + offset, s.action) for s in self.steps)
        return Plan(
            name=self.name,
            steps=tuple(steps),
            priority=self.priority,
            preempt=self.preempt,
            # The window shrinks by what acquisition consumed: a plan that was already
            # marginal must not become late because it now has to target first.
            expires_in_ms=max(200, self.expires_in_ms - offset),
        )


@dataclass(order=True, slots=True)
class ScheduledEvent:
    """A primitive event with an absolute deadline.

    Ordering is `(due_at, -priority, seq)`: earliest first, higher priority first on a
    tie, then submission order. The sequence number keeps the heap total-ordered so
    equal-deadline events never compare their payloads.
    """

    due_at: float
    neg_priority: int
    seq: int
    event: InputEvent = field(compare=False)
    plan_name: str = field(compare=False, default="")
    expires_at: float = field(compare=False, default=float("inf"))
    # Links the two halves of a Press. A release is only owed if its press actually went
    # out — see `is_release` below.
    pair_id: int | None = field(compare=False, default=None)

    @property
    def priority(self) -> int:
        return -self.neg_priority

    @property
    def is_release(self) -> bool:
        """Releases get special handling everywhere: they are never silently dropped."""
        return isinstance(self.event, KeyUp)


class InputTimeline:
    """Deadline-ordered schedule of pending input events.

    Thread-safe: policy submits from the decide thread, the dispatcher drains from the
    dispatch thread.
    """

    def __init__(self) -> None:
        self._heap: list[ScheduledEvent] = []
        self._counter = itertools.count()
        self._lock = threading.Lock()
        # Presses whose key-down was dropped before dispatch. Their key-up must be
        # dropped too — sending a release for a key that was never pressed is a spurious
        # input event, and in a game that can cancel a cast or drop a held movement.
        self._orphaned: set[int] = set()
        self.submitted_plans = 0
        self.rejected_plans = 0
        self.expired_events = 0
        self.cancelled_events = 0

    # -- submission --------------------------------------------------------------

    def submit(self, plan: Plan, at: float | None = None) -> bool:
        """Schedule a plan. Returns False if it was rejected by a higher-priority hold.

        Rejection rather than queueing is deliberate: a rotation step that waits behind a
        reflex is a rotation step computed for a world state that no longer exists.
        """
        at = now() if at is None else at
        expires_at = at + plan.expires_in_ms / 1000.0

        with self._lock:
            if plan.preempt:
                # The cancelled work may have already pressed keys whose releases were
                # still pending. Those releases are not dropped — they are pulled forward
                # to now, so the dispatcher lifts them before the new plan's first event.
                # Dropping them held a movement key down for a whole fight once.
                for release in self._cancel_below_locked(plan.priority, at):
                    heapq.heappush(
                        self._heap,
                        ScheduledEvent(
                            due_at=at,
                            neg_priority=-int(plan.priority),
                            seq=next(self._counter),
                            event=release.event,
                            plan_name=release.plan_name,
                            expires_at=float("inf"),
                            pair_id=None,
                        ),
                    )
            elif self._blocked_by_higher_locked(plan.priority, at):
                self.rejected_plans += 1
                return False

            for step in plan.steps:
                base = at + step.at_ms / 1000.0
                # A Press's down and up share a pair id so the two can be reasoned about
                # together when either end is dropped.
                pair_id = next(self._counter) if isinstance(step.action, Press) else None
                for offset_s, event in _expand(step.action):
                    heapq.heappush(
                        self._heap,
                        ScheduledEvent(
                            due_at=base + offset_s,
                            neg_priority=-int(plan.priority),
                            seq=next(self._counter),
                            event=event,
                            plan_name=plan.name,
                            # A release inherits no expiry. Expiring a release is how a
                            # key gets stuck down.
                            expires_at=float("inf") if isinstance(event, KeyUp) else expires_at,
                            pair_id=pair_id,
                        ),
                    )
            self.submitted_plans += 1
            return True

    def _blocked_by_higher_locked(self, priority: int, at: float) -> bool:
        return any(e.priority > priority and e.due_at > at for e in self._heap)

    # -- draining ----------------------------------------------------------------

    def due(self, at: float | None = None) -> list[ScheduledEvent]:
        """Pop everything due at or before `at`, in schedule order.

        Expired non-release events are dropped here rather than dispatched. A late action
        is not a slow success — a dodge fired 400ms after the window closed dodges into
        the damage.
        """
        at = now() if at is None else at
        ready: list[ScheduledEvent] = []
        with self._lock:
            while self._heap and self._heap[0].due_at <= at:
                item = heapq.heappop(self._heap)

                if item.is_release:
                    # Owed only if the matching press actually went out.
                    if item.pair_id is not None and item.pair_id in self._orphaned:
                        self._orphaned.discard(item.pair_id)
                        continue
                    ready.append(item)
                    continue

                if at > item.expires_at:
                    self.expired_events += 1
                    if item.pair_id is not None:
                        self._orphaned.add(item.pair_id)
                    continue

                ready.append(item)
        return ready

    def next_deadline(self) -> float | None:
        """When the dispatcher should next wake. `None` means nothing is scheduled."""
        with self._lock:
            return self._heap[0].due_at if self._heap else None

    # -- cancellation ------------------------------------------------------------

    def cancel_below(self, priority: int, at: float | None = None) -> list[ScheduledEvent]:
        """Drop pending events below `priority`; return the releases among them.

        The caller must dispatch those releases immediately. That is the contract that
        keeps preemption from leaving keys held.
        """
        at = now() if at is None else at
        with self._lock:
            return self._cancel_below_locked(priority, at)

    def _cancel_below_locked(self, priority: int, at: float) -> list[ScheduledEvent]:
        keep: list[ScheduledEvent] = []
        dropped: list[ScheduledEvent] = []
        for item in self._heap:
            if item.priority >= priority or item.due_at <= at:
                keep.append(item)
            else:
                dropped.append(item)
                self.cancelled_events += 1
        heapq.heapify(keep)
        self._heap = keep
        return _owed_releases(dropped)

    def flush(self) -> list[ScheduledEvent]:
        """Drop everything; return the releases actually owed.

        Used when a safety guard trips. Whatever was scheduled was computed for a world
        that no longer applies, so none of it should run — but any key already held still
        has to come back up.
        """
        with self._lock:
            dropped = self._heap
            self._heap = []
            releases = _owed_releases(dropped)
            self.cancelled_events += len(dropped) - len(releases)
        return releases

    # -- inspection --------------------------------------------------------------

    def pending(self) -> int:
        with self._lock:
            return len(self._heap)

    def peek(self, limit: int = 8) -> list[ScheduledEvent]:
        with self._lock:
            return heapq.nsmallest(limit, self._heap)

    def stats(self) -> dict[str, int]:
        return {
            "pending": self.pending(),
            "submitted": self.submitted_plans,
            "rejected": self.rejected_plans,
            "expired": self.expired_events,
            "cancelled": self.cancelled_events,
        }


def _owed_releases(dropped: Iterable[ScheduledEvent]) -> list[ScheduledEvent]:
    """Of the cancelled events, which key-ups does the caller still have to send?

    A release is owed only when its press already went out. If both halves are still
    pending, the key was never pressed and releasing it would be a spurious input event —
    which in a game can cancel a cast or interrupt held movement.
    """
    dropped = list(dropped)
    pending_presses = {
        e.pair_id for e in dropped if isinstance(e.event, KeyDown) and e.pair_id is not None
    }
    return [
        e for e in dropped if e.is_release and e.pair_id not in pending_presses
    ]


def _expand(action: Action) -> Iterable[tuple[float, InputEvent]]:
    """Break a plan-level action into `(offset_seconds, primitive)` pairs."""
    if isinstance(action, Press):
        yield 0.0, KeyDown(action.key)
        yield action.hold_ms / 1000.0, KeyUp(action.key)
    else:
        yield 0.0, action


def weave(
    name: str,
    gcd_key: str,
    ogcd_keys: Iterable[str] = (),
    *,
    animation_lock_ms: int = 600,
    weave_gap_ms: int = 20,
    hold_ms: int = 40,
    priority: Priority = Priority.ROTATION,
    expires_in_ms: int = 2000,
) -> Plan:
    """Build a GCD plus its weaved off-GCDs as one plan.

    This is the shape FFXIV rotations actually take: a GCD, then up to two off-GCDs
    inside the animation-lock-to-next-GCD gap. Encoding it as one plan rather than three
    submissions means the whole weave is preempted or expires as a unit — a half-executed
    weave is worse than none, because the off-GCD lands during the next GCD's lock and
    clips it.
    """
    steps = [Step(0, Press(gcd_key, hold_ms))]
    at = animation_lock_ms + weave_gap_ms
    for key in ogcd_keys:
        steps.append(Step(at, Press(key, hold_ms)))
        at += animation_lock_ms + weave_gap_ms
    return Plan(
        name=name,
        steps=tuple(steps),
        priority=priority,
        preempt=False,
        expires_in_ms=expires_in_ms,
    )
