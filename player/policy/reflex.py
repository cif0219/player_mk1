"""Reflexes: fast conditional responses over `WorldState`.

The previous version of this project ran reflexes directly against pixels. That is the
design this one deliberately rejects — see docs/ARCHITECTURE.md. Reading `WorldState`
instead means a reflex is a pure function, so its test is three lines and no image, and a
UI change is a perception problem rather than a strategy problem.

Reflexes are checked every tick, before the rotation. A reflex that fires with
`preempt=True` flushes the timeline, which is why the timeline converts cancelled
key-ups into immediate ones: dodging while a rotation key is still held would hold that
key for the rest of the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..act.timeline import Plan, Priority
from ..clock import now
from ..state import WorldState
from .condition import Condition

PlanFactory = Callable[[WorldState], Plan | None]


@dataclass(slots=True)
class Reflex:
    """One trigger and its response.

    `group` exists so the director can enable or disable a whole category at once —
    "ground_aoe" off while in a cutscene, on in a dungeon — without needing to know
    individual rule names.

    `cooldown_ms` is not a game cooldown. It stops a reflex from re-firing while the
    condition is still true but the response is already in flight: a telegraph stays on
    screen for seconds after you have already dodged it.
    """

    id: str
    condition: Condition
    plan: Plan | PlanFactory
    group: str = "default"
    priority: Priority = Priority.REFLEX
    cooldown_ms: float = 500.0
    max_fires: int | None = None
    enabled: bool = True
    preempt: bool = True

    # -inf rather than 0.0: with a zero sentinel the first check computes an elapsed
    # time of `at`, so a reflex evaluated early in a process's life is silently held down
    # by its own cooldown before it has ever fired.
    _last_fired: float = field(default=float("-inf"), repr=False)
    _fire_count: int = field(default=0, repr=False)

    def can_fire(self, at: float) -> bool:
        if not self.enabled:
            return False
        if self.max_fires is not None and self._fire_count >= self.max_fires:
            return False
        return (at - self._last_fired) * 1000.0 >= self.cooldown_ms

    def build(self, state: WorldState) -> Plan | None:
        plan = self.plan(state) if callable(self.plan) else self.plan
        if plan is None:
            return None
        # Reflex priority and preemption are properties of the reflex, not of whatever
        # plan the factory happened to build, so they are applied here.
        return Plan(
            name=f"reflex:{self.id}",
            steps=plan.steps,
            priority=self.priority,
            preempt=self.preempt,
            expires_in_ms=plan.expires_in_ms,
        )

    def mark_fired(self, at: float) -> None:
        self._last_fired = at
        self._fire_count += 1

    def reset(self) -> None:
        self._last_fired = float("-inf")
        self._fire_count = 0


class ReflexLayer:
    """Holds reflexes and picks at most one per tick.

    One per tick, not all that match: two reflexes firing together means two preempting
    plans, and the second one flushes the first before it has run. If several conditions
    hold, the highest-priority reflex is the right answer and the others will still be
    true next tick if they still matter.
    """

    def __init__(self, reflexes: list[Reflex] | None = None) -> None:
        self._reflexes: dict[str, Reflex] = {}
        self._order: list[str] = []
        self.fires = 0
        for reflex in reflexes or []:
            self.add(reflex)

    def add(self, reflex: Reflex) -> "ReflexLayer":
        self._reflexes[reflex.id] = reflex
        if reflex.id not in self._order:
            self._order.append(reflex.id)
        self._resort()
        return self

    def remove(self, reflex_id: str) -> bool:
        if reflex_id in self._reflexes:
            del self._reflexes[reflex_id]
            self._order.remove(reflex_id)
            return True
        return False

    def _resort(self) -> None:
        self._order.sort(key=lambda rid: -int(self._reflexes[rid].priority))

    # -- group control (what the director drives) --------------------------------

    def set_group_enabled(self, group: str, enabled: bool) -> int:
        count = 0
        for reflex in self._reflexes.values():
            if reflex.group == group:
                reflex.enabled = enabled
                count += 1
        return count

    def groups(self) -> set[str]:
        return {r.group for r in self._reflexes.values()}

    def enabled_groups(self) -> set[str]:
        return {r.group for r in self._reflexes.values() if r.enabled}

    # -- evaluation --------------------------------------------------------------

    def decide(self, state: WorldState, at: float | None = None) -> tuple[Reflex, Plan] | None:
        at = now() if at is None else at
        for reflex_id in self._order:
            reflex = self._reflexes[reflex_id]
            if not reflex.can_fire(at):
                continue
            try:
                if not reflex.condition.evaluate(state):
                    continue
                plan = reflex.build(state)
            except Exception:
                # A broken reflex must not take down the loop, and must not fire.
                continue
            if plan is None:
                continue
            reflex.mark_fired(at)
            self.fires += 1
            return reflex, plan
        return None

    def required_fields(self) -> set[str]:
        out: set[str] = set()
        for reflex in self._reflexes.values():
            out |= reflex.condition.fields()
        return out

    def reset(self) -> None:
        for reflex in self._reflexes.values():
            reflex.reset()

    def __len__(self) -> int:
        return len(self._reflexes)

    def __iter__(self):
        return (self._reflexes[rid] for rid in self._order)

    def stats(self) -> dict[str, object]:
        return {
            "count": len(self._reflexes),
            "enabled": sum(1 for r in self._reflexes.values() if r.enabled),
            "fires": self.fires,
            "groups": sorted(self.enabled_groups()),
        }
