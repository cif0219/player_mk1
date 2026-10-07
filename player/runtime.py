"""Thread wiring and the play loop.

Four threads, chosen so no lock is ever held across an I/O boundary:

    capture     owns the capture device, writes the newest frame into a slot
    decide      perception + reflexes + rotation; emits plans onto the timeline
    dispatch    drains the timeline through the safety gate into the backend
    strategy    the Claude director; never blocks anything

Perception and decision deliberately share a thread. Together they are well under 20ms,
so merging them removes a queue hop and a lock from the hot path; splitting them would
buy parallelism this workload does not need and cost latency it cannot spare.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .act.backend import InputBackend, NullBackend
from .act.timeline import Click, InputTimeline, KeyDown, KeyUp, MouseMove, Plan, Priority
from .capture.source import CaptureSource, FrameSlot
from .clock import LatencyBudget, now
from .perceive.pipeline import PerceptionPipeline
from .policy.reflex import ReflexLayer
from .policy.rotation import RotationPlanner
from .profile import GameProfile
from .record.recorder import Recorder, RecorderConfig
from .safety.guards import SafetyGate
from .state import WorldState
from .strategy.director import Director, DirectorContext
from .strategy.directives import (
    EnableReflexGroup,
    Pause,
    SelectPlanProfile,
    SetObjective,
    SetParameter,
)


@dataclass(slots=True)
class RuntimeConfig:
    dry_run: bool = True
    capture_fps: int = 60
    dispatch_tick_s: float = 0.002  # dispatch granularity; ~2ms is well under the budget
    status_interval_s: float = 5.0
    max_runtime_s: float | None = None


class Dispatcher:
    """Drains the timeline into the backend, subject to the guard chain.

    Tracks held keys. A guard tripping mid-hold must release them, or the game keeps the
    key down with nothing left to lift it — the single worst failure this layer can
    produce.
    """

    def __init__(
        self,
        timeline: InputTimeline,
        backend: InputBackend,
        gate: SafetyGate,
        killswitch: Any | None = None,
        recorder: Recorder | None = None,
        budget: LatencyBudget | None = None,
    ) -> None:
        self.timeline = timeline
        self.backend = backend
        self.gate = gate
        self.killswitch = killswitch
        self.recorder = recorder
        self.budget = budget or LatencyBudget()
        self._held: set[str] = set()
        self.dispatched = 0
        self.blocked = 0
        self._was_blocked = False

    def pump(self, state: WorldState | None, at: float | None = None) -> int:
        at = now() if at is None else at

        if not self.gate.allowed(state, at):
            self.blocked += 1
            if not self._was_blocked:
                # Trip edge: flush what was scheduled for a world that no longer applies,
                # and lift anything we are holding.
                self._on_block()
                self._was_blocked = True
            return 0
        self._was_blocked = False

        sent = 0
        for event in self.timeline.due(at):
            if not self.gate.consume_rate(at):
                # Rate ceiling hit mid-drain. Put nothing further out; releases already
                # popped still go, because a release is never dropped.
                if isinstance(event.event, KeyUp):
                    self._send(event.event)
                    sent += 1
                continue
            self._send(event.event)
            sent += 1

        self.dispatched += sent
        return sent

    def _send(self, event) -> None:
        with self.budget.measure("dispatch"):
            if isinstance(event, KeyDown):
                self._expect(event.key)
                self.backend.key_down(event.key)
                self._held.add(event.key)
                self._record("key_down", event.key)
            elif isinstance(event, KeyUp):
                self.backend.key_up(event.key)
                self._held.discard(event.key)
                self._record("key_up", event.key)
            elif isinstance(event, MouseMove):
                self.backend.mouse_move(event.point)
                self._record("mouse_move", f"{event.point.x},{event.point.y}")
            elif isinstance(event, Click):
                self.backend.click(event.button, event.point)
                self._record("click", event.button)

    def _expect(self, key: str) -> None:
        """Tell the kill switch's hook this keystroke is ours, not the human's."""
        if self.killswitch is None:
            return
        try:
            from .act.keymap import parse_combo, resolve_key

            mods, base = parse_combo(key)
            for name in (*mods, base):
                self.killswitch.expect(resolve_key(name).code)
        except (KeyError, ValueError, AttributeError):
            pass

    def _record(self, kind: str, detail: str) -> None:
        if self.recorder is not None:
            self.recorder.dispatch(detail, kind)

    def _on_block(self) -> None:
        # Say why in the trace: a flushed timeline looks exactly like a policy bug
        # (every held key released at once) unless the guard that did it is named.
        if self.recorder is not None:
            self.recorder.note(f"dispatch blocked: {self.gate.status()}")
        for event in self.timeline.flush():
            self._send(event.event)
        self.release_all()

    def release_all(self) -> None:
        for key in sorted(self._held):
            try:
                self.backend.key_up(key)
            except Exception:
                pass
        self._held.clear()
        release = getattr(self.backend, "release_all", None)
        if callable(release):
            release()

    @property
    def held(self) -> set[str]:
        return set(self._held)


class Runtime:
    """Owns the threads and the shared objects they touch."""

    def __init__(
        self,
        profile: GameProfile,
        source: CaptureSource,
        backend: InputBackend | None = None,
        gate: SafetyGate | None = None,
        director: Director | None = None,
        recorder: Recorder | None = None,
        config: RuntimeConfig | None = None,
        killswitch: Any | None = None,
    ) -> None:
        self.profile = profile
        self.source = source
        self.config = config or RuntimeConfig()
        self.budget = LatencyBudget()

        self.backend = backend or NullBackend()
        self.gate = gate or SafetyGate()
        self.recorder = recorder or Recorder(RecorderConfig(enabled=False))
        self.killswitch = killswitch

        self.slot = FrameSlot()
        self.pipeline = PerceptionPipeline(profile.sensors, self.budget)
        self.timeline = InputTimeline()
        self.dispatcher = Dispatcher(
            self.timeline, self.backend, self.gate, killswitch, self.recorder, self.budget
        )

        self.reflexes: ReflexLayer = profile.reflexes
        rotation = profile.rotation(profile.default_rotation)
        self.planner = RotationPlanner(rotation) if rotation else None

        self.director = director
        self.objective: str = ""
        self.parameters: dict[str, Any] = dict(profile.parameters)
        self.events: list[str] = []

        self.state: WorldState | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self.started_at = 0.0
        self.ticks = 0

    # -- lifecycle ---------------------------------------------------------------

    def preflight(self) -> list[str]:
        """Everything that must be true before a single key is sent.

        Run before starting, and refuse to start live if it returns anything. Failing at
        startup is cheap; failing in the middle of a dungeon is not.
        """
        problems = list(self.profile.validate())

        if not self.config.dry_run:
            if self.killswitch is None or not getattr(self.killswitch, "available", False):
                problems.append(
                    "no kill switch available — refusing to run live. Use --dry-run."
                )
        return problems

    def start(self) -> None:
        self.started_at = now()
        self.source.open()
        self.recorder.start(
            {
                "profile": self.profile.name,
                "dry_run": self.config.dry_run,
                "backend": self.backend.name,
                "rotation": self.planner.profile.id if self.planner else None,
            }
        )
        for target, name in (
            (self._capture_loop, "capture"),
            (self._decide_loop, "decide"),
            (self._dispatch_loop, "dispatch"),
            (self._strategy_loop, "strategy"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        self.slot.wake()
        for thread in self._threads:
            thread.join(timeout=2.0)
        # Order matters: lift held keys before the backend closes underneath them.
        self.dispatcher.release_all()
        self.backend.close()
        self.source.close()
        self.recorder.stop()
        if self.killswitch is not None:
            self.killswitch.stop()

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    # -- threads -----------------------------------------------------------------

    def _capture_loop(self) -> None:
        while self.running:
            try:
                with self.budget.measure("capture"):
                    frame = self.source.grab()
            except Exception as exc:
                self.note(f"capture error: {exc}")
                self._stop.set()
                break
            if frame is None:
                if getattr(self.source, "exhausted", False):
                    self.note("capture source exhausted")
                    self._stop.set()
                    break
                self._stop.wait(0.002)
                continue
            self.slot.publish(frame)
            self.recorder.frame(frame.index, frame.image, frame.captured_at, frame.client_rect)

    def _decide_loop(self) -> None:
        while self.running:
            frame = self.slot.take(timeout=0.05)
            if frame is None:
                continue
            state = self.pipeline.process(frame)
            self.state = state
            self.ticks += 1
            self.recorder.state(state)

            with self.budget.measure("decide"):
                self._decide(state)

    def _decide(self, state: WorldState) -> None:
        at = now()

        # Reflexes first, unconditionally. A reflex exists precisely because it must
        # outrank whatever the rotation was about to do.
        fired = self.reflexes.decide(state, at)
        if fired is not None:
            reflex, plan = fired
            if self.timeline.submit(plan, at):
                self.recorder.plan(plan.name, int(plan.priority), len(plan.steps), at)
                self.note(f"reflex {reflex.id}")
            return

        if self.planner is None:
            return
        decision = self.planner.decide(state, at)
        if decision is not None and self.timeline.submit(decision.plan, at):
            self.recorder.plan(
                decision.plan.name, int(decision.plan.priority), len(decision.plan.steps), at
            )

    def _dispatch_loop(self) -> None:
        while self.running:
            deadline = self.timeline.next_deadline()
            if deadline is None:
                # Nothing scheduled, but guards still need evaluating so the status line
                # stays honest and a trip still flushes.
                self.dispatcher.pump(self.state)
                self._stop.wait(0.01)
                continue
            wait = deadline - now()
            if wait > self.config.dispatch_tick_s:
                self._stop.wait(min(wait, 0.05))
                continue
            self.dispatcher.pump(self.state)
            self._stop.wait(self.config.dispatch_tick_s)

    def _strategy_loop(self) -> None:
        if self.director is None:
            return
        while self.running:
            self._stop.wait(1.0)
            if not self.running:
                break
            state = self.state
            if state is None or not self.director.should_review():
                continue
            # Never ask for strategy on a state perception doesn't yet trust:
            # the first review of a session otherwise sees empty fields, reads
            # them as "0 HP, no target", and prudently pauses — which is the
            # right call on bad data and the wrong call about the actual game.
            if state.untrusted(self.profile.critical_fields):
                continue
            frame = self.slot.latest()
            self.director.review(
                state,
                self.director_context(),
                frame.image if frame is not None else None,
            )

    # -- director plumbing -------------------------------------------------------

    def director_context(self) -> DirectorContext:
        return DirectorContext(
            objective=self.objective,
            profile_ids=tuple(sorted(self.profile.rotations)),
            active_profile=self.planner.profile.id if self.planner else "",
            reflex_groups=tuple(sorted(self.reflexes.groups())),
            enabled_reflex_groups=tuple(sorted(self.reflexes.enabled_groups())),
            parameters=dict(self.parameters),
            state_fields=tuple(sorted(self.profile.required_fields())),
            recent_events=tuple(self.events[-8:]),
        )

    def apply_directive(self, directive: Any) -> bool:
        """Final validation before a directive reaches policy.

        Every id is re-checked against the live catalog here. The prompt tells the model
        what is legal; this is what makes it true.
        """
        self.recorder.directive(type(directive).__name__, {"reason": directive.reason})

        if isinstance(directive, SetObjective):
            self.objective = directive.objective
            self.note(f"objective: {directive.objective}")
            return True

        if isinstance(directive, SelectPlanProfile):
            rotation = self.profile.rotation(directive.profile_id)
            if rotation is None or self.planner is None:
                self.note(f"rejected unknown profile {directive.profile_id!r}")
                return False
            self.planner.set_profile(rotation)
            self.note(f"rotation -> {directive.profile_id}")
            return True

        if isinstance(directive, SetParameter):
            if directive.key not in self.parameters:
                self.note(f"rejected unknown parameter {directive.key!r}")
                return False
            self.parameters[directive.key] = _coerce(
                directive.value, self.parameters[directive.key]
            )
            self.note(f"{directive.key} = {self.parameters[directive.key]!r}")
            return True

        if isinstance(directive, EnableReflexGroup):
            if directive.group not in self.reflexes.groups():
                self.note(f"rejected unknown reflex group {directive.group!r}")
                return False
            self.reflexes.set_group_enabled(directive.group, directive.enabled)
            self.note(f"reflex[{directive.group}] = {directive.enabled}")
            return True

        if isinstance(directive, Pause):
            if self.killswitch is not None:
                if directive.paused:
                    self.killswitch.trip("director")
                else:
                    self.killswitch.resume()
            self.note("paused by director" if directive.paused else "resumed by director")
            return True

        return False

    # -- observability -----------------------------------------------------------

    def note(self, text: str) -> None:
        self.events.append(text)
        if len(self.events) > 64:
            self.events.pop(0)
        self.recorder.note(text)

    def status_line(self) -> str:
        state = self.state
        age = f"{state.age_ms():.0f}ms" if state else "-"
        parts = [
            f"t={now() - self.started_at:5.1f}s",
            f"ticks={self.ticks}",
            f"frames={self.slot.published}/drop{self.slot.dropped}",
            f"age={age}",
            f"queue={self.timeline.pending()}",
            f"sent={self.dispatcher.dispatched}",
            f"gate={self.gate.status()}",
        ]
        if self.planner is not None:
            last = self.planner.last_decision
            parts.append(f"rot={last.describe() if last else '-'}")
        if self.director is not None and self.director.config.enabled:
            parts.append(f"llm={self.director.calls}c/{self.director.output_tokens}t")
        return " | ".join(parts)

    def report(self) -> list[str]:
        lines = [
            self.profile.describe(),
            self.pipeline.describe(),
            f"timeline: {self.timeline.stats()}",
            f"reflexes: {self.reflexes.stats()}",
        ]
        if self.planner is not None:
            lines.append(f"rotation: {self.planner.stats()}")
        if self.director is not None:
            lines.append(f"director: {self.director.stats()}")
        if self.gate.block_counts:
            lines.append(f"guard blocks: {self.gate.block_counts}")
        lines.append(self.recorder.status())
        lines.extend(self.budget.report())
        return lines


def _coerce(value: str, exemplar: Any) -> Any:
    """Coerce a directive's string value to the declared parameter's type.

    Directives carry values as strings so the schema stays simple; the declared default
    is the source of truth for what type it should become.
    """
    if isinstance(exemplar, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(exemplar, int):
        try:
            return int(float(value))
        except ValueError:
            return exemplar
    if isinstance(exemplar, float):
        try:
            return float(value)
        except ValueError:
            return exemplar
    return value


__all__ = ["Dispatcher", "Plan", "Priority", "Runtime", "RuntimeConfig"]
