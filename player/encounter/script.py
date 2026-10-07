"""The compiled form of a raid guide.

A guide is prose describing a program. This is that program as data: what happens, how we
know it started, where to look to find out the details, where to stand, and what to press.

Two things are deliberately data rather than code:

* **The camera plan.** Where to point the camera to *see* the information a mechanic
  depends on is part of the mechanic, not an afterthought. A guide already says it — "look
  at the boss to see which arm glows" — and in a 3D game you cannot resolve what you have
  not looked at.
* **The resolution.** A closed vocabulary the compiler selects from. It never emits code
  and never emits coordinates; destinations are computed from live perception at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ..state import WorldState
from ..world.arena import ArenaFrame, ArenaPoint, WaymarkLayout


# -- triggers ---------------------------------------------------------------------


@runtime_checkable
class Trigger(Protocol):
    """How we know a mechanic has started."""

    def fires(self, state: WorldState, elapsed_s: float, at: float) -> bool: ...

    def describe(self) -> str: ...


@dataclass(frozen=True, slots=True)
class CastTrigger:
    """The boss started a named cast.

    The primary trigger, and it does double duty: a named cast is also a **resync point**
    for the encounter clock. A purely time-based script drifts — a slow kill, an HP-gated
    phase change, a stun — and drift compounds until every later mechanic fires at the
    wrong moment. Anchoring on casts means drift never accumulates past one mechanic.
    """

    name: str
    field_name: str = "boss.cast_name"

    def fires(self, state: WorldState, elapsed_s: float, at: float) -> bool:
        observed = state.text(self.field_name, "")
        return bool(observed) and observed.strip().lower() == self.name.strip().lower()

    def describe(self) -> str:
        return f'cast "{self.name}"'


@dataclass(frozen=True, slots=True)
class TimelineTrigger:
    """A fixed offset from the pull. The fallback for mechanics with no cast bar."""

    at_s: float
    window_s: float = 1.5

    def fires(self, state: WorldState, elapsed_s: float, at: float) -> bool:
        return self.at_s <= elapsed_s <= self.at_s + self.window_s

    def describe(self) -> str:
        return f"t+{self.at_s:.1f}s"


@dataclass(frozen=True, slots=True)
class DebuffTrigger:
    """We personally got something. Markers, tethers, resistance downs."""

    debuff_id: str

    def fires(self, state: WorldState, elapsed_s: float, at: float) -> bool:
        return state.flag(f"debuff.{self.debuff_id}.active")

    def describe(self) -> str:
        return f"debuff {self.debuff_id}"


# -- where to look ----------------------------------------------------------------


class LookAt(Enum):
    BOSS = "boss"
    TARGET_POINT = "point"
    WAYMARK = "waymark"
    SELF_GROUND = "self_ground"  # pitch down to see what is under you
    KEEP = "keep"  # do not move the camera; whatever is framed is enough


@dataclass(frozen=True, slots=True)
class LookPlan:
    """The camera half of a mechanic.

    Exists because perception in a 3D game is partial and steerable. A tower behind you is
    not absent, it is unobserved — so a mechanic that depends on seeing something has to
    say where to look, and how long to hold still once it gets there.
    """

    at: LookAt = LookAt.BOSS
    waymark: str = ""
    point: ArenaPoint | None = None
    # Time to hold the camera still after arriving. Detectors do badly on frames captured
    # mid-slew, so a mechanic that depends on a detection needs a moment of stillness.
    settle_s: float = 0.25
    tolerance_deg: float = 10.0

    def resolve(self, frame: ArenaFrame, boss: ArenaPoint | None) -> ArenaPoint | None:
        if self.at is LookAt.KEEP:
            return None
        if self.at is LookAt.BOSS:
            return boss
        if self.at is LookAt.WAYMARK:
            return frame.waymark(self.waymark)
        if self.at is LookAt.TARGET_POINT:
            return self.point
        if self.at is LookAt.SELF_GROUND:
            return frame.player
        return None


# -- resolutions ------------------------------------------------------------------


class ResolutionKind(Enum):
    WAYMARK = "waymark"
    AVOID_TELEGRAPHS = "avoid_telegraphs"
    RELATIVE_TO_BOSS = "relative_to_boss"
    STAY = "stay"


@dataclass(frozen=True, slots=True)
class Resolution:
    """Where to be, expressed in a closed vocabulary.

    The compiler picks a `kind` and fills parameters. It cannot emit an expression and
    cannot emit a destination — destinations are computed here, from live perception, so a
    hallucinated coordinate is not merely rejected but unrepresentable.
    """

    kind: ResolutionKind = ResolutionKind.STAY
    waymark: str = ""
    angle_deg: float = 180.0
    distance_m: float = 3.0
    clearance_m: float = 2.0
    candidates: tuple[str, ...] = ()

    def destination(
        self,
        frame: ArenaFrame,
        state: WorldState,
        boss: ArenaPoint | None = None,
        telegraphs: list[ArenaPoint] | None = None,
    ) -> ArenaPoint | None:
        if self.kind is ResolutionKind.STAY:
            return None

        if self.kind is ResolutionKind.WAYMARK:
            return frame.waymark(self.waymark)

        if self.kind is ResolutionKind.RELATIVE_TO_BOSS:
            if boss is None:
                return None
            # Angle is measured from the boss's facing when known, and from arena north
            # when it is not — an honest degradation, since "behind the boss" is
            # meaningless without knowing which way it faces.
            facing = state.num("boss.facing_deg", 0.0)
            offset = ArenaPoint(0.0, self.distance_m).rotated(-(facing + self.angle_deg))
            return boss + offset

        if self.kind is ResolutionKind.AVOID_TELEGRAPHS:
            return _safest_candidate(frame, self.candidates, self.clearance_m, telegraphs or [])

        return None

    def describe(self) -> str:
        if self.kind is ResolutionKind.WAYMARK:
            return f"go to {self.waymark}"
        if self.kind is ResolutionKind.RELATIVE_TO_BOSS:
            return f"{self.angle_deg:.0f}deg / {self.distance_m:.0f}m from boss"
        if self.kind is ResolutionKind.AVOID_TELEGRAPHS:
            return f"safe spot among {','.join(self.candidates) or 'waymarks'}"
        return "stay put"


def _safest_candidate(
    frame: ArenaFrame,
    candidates: tuple[str, ...],
    clearance_m: float,
    telegraphs: list[ArenaPoint],
) -> ArenaPoint | None:
    """Pick the candidate spot furthest from any detected telegraph.

    Candidates are waymarks rather than arbitrary points for two reasons: a guide's safe
    spots are already written as waymarks, and a discrete shortlist is checkable. A
    continuous search over the arena floor would happily return "safe" positions that are
    off the platform or inside the boss.

    Telegraph positions arrive already converted to arena metres — the runner does that
    with the ground-plane homography. Passing them in rather than reaching for them keeps
    this function pure and therefore testable without any perception at all.
    """
    options = candidates or tuple(sorted(frame.layout.known_ids()))

    best: tuple[float, ArenaPoint] | None = None
    for mark_id in options:
        point = frame.waymark(mark_id)
        if point is None:
            continue
        margin = min((point.distance_to(t) for t in telegraphs), default=float("inf"))
        if best is None or margin > best[0]:
            best = (margin, point)

    if best is None:
        return None
    # Every candidate is inside a telegraph. Report no safe spot rather than the
    # least-bad one: walking confidently into slightly less damage is not a resolution,
    # and the honest answer lets the caller fall back to something else.
    if best[0] < clearance_m:
        return None
    return best[1]


# -- mechanics --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Mechanic:
    """One thing the encounter does, and what to do about it."""

    id: str
    trigger: Trigger
    deadline_ms: int = 5000
    look: LookPlan = field(default_factory=LookPlan)
    resolve: Resolution = field(default_factory=Resolution)
    # Ability the mechanic requires — anti-knockback, a mitigation, a tower soak. Reserved
    # on the commitment board so the rotation yields its GCD rather than colliding.
    ability_id: str = ""
    ability_lead_ms: int = 800  # fire this long before the deadline
    category: str = "dodge"
    # The guide sentence this was compiled from, kept verbatim. When a mechanic fails the
    # first question is always "bad script or bad execution", and having the source line
    # next to the rule answers it in seconds instead of a trip back to the guide.
    notes: str = ""

    def describe(self) -> str:
        return f"{self.id}: on {self.trigger.describe()} -> {self.resolve.describe()}"


@dataclass(slots=True)
class EncounterScript:
    """A compiled fight."""

    id: str
    layout: WaymarkLayout
    mechanics: list[Mechanic] = field(default_factory=list)
    description: str = ""

    def by_id(self, mechanic_id: str) -> Mechanic | None:
        return next((m for m in self.mechanics if m.id == mechanic_id), None)

    def matching(self, state: WorldState, elapsed_s: float, at: float) -> Mechanic | None:
        """First mechanic whose trigger fires. Declaration order is priority order."""
        for mechanic in self.mechanics:
            try:
                if mechanic.trigger.fires(state, elapsed_s, at):
                    return mechanic
            except Exception:
                continue
        return None

    def validate(self) -> list[str]:
        problems: list[str] = []
        seen: set[str] = set()
        for mechanic in self.mechanics:
            if mechanic.id in seen:
                problems.append(f"duplicate mechanic id {mechanic.id!r}")
            seen.add(mechanic.id)

            if mechanic.deadline_ms <= 0:
                problems.append(f"{mechanic.id}: deadline must be positive")

            for mark in _referenced_marks(mechanic):
                if self.layout.get(mark) is None:
                    problems.append(
                        f"{mechanic.id}: references waymark {mark!r} not in layout "
                        f"{sorted(self.layout.known_ids())}"
                    )
        return problems


def _referenced_marks(mechanic: Mechanic) -> set[str]:
    marks: set[str] = set()
    if mechanic.resolve.waymark:
        marks.add(mechanic.resolve.waymark)
    marks.update(mechanic.resolve.candidates)
    if mechanic.look.waymark:
        marks.add(mechanic.look.waymark)
    return marks
