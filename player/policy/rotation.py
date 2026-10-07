"""The rotation planner.

An FFXIV rotation is, structurally, a priority list evaluated against resource, cooldown,
and buff state: walk the list, use the first thing that is both available and whose
condition holds. Modelling it that way — rather than as a fixed sequence — is what makes
it survive procs, resource drift, and downtime, all of which desynchronise a fixed
sequence immediately.

The part that makes it *real time* rather than a list walker is weave-window scheduling.
A GCD starts an animation lock of ~600ms, then the GCD itself does not come back for
~2500ms. The gap between those is where off-GCD abilities go, and getting it wrong in
either direction is a DPS loss: too early clips the GCD, too late drops the weave.

So the planner emits one `Plan` per GCD containing the GCD *and* its weaves. Emitting
them separately would let a preemption take the GCD and leave the off-GCD to land inside
the next GCD's lock, which is worse than not weaving at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..act.timeline import MouseMove, Plan, Press, Priority, Step
from ..clock import now
from ..geometry import ScreenPoint
from ..state import WorldState
from .commitment import CommitmentBoard, CommitmentKind
from .condition import Condition
from .targeting import TargetResolver, TargetSpec


class AbilityKind(Enum):
    GCD = "gcd"
    OGCD = "ogcd"


@dataclass(frozen=True, slots=True)
class Ability:
    """One button, and the circumstances under which to press it.

    `recast_s` lives here rather than in the cooldown probe because the probe can only
    report *what fraction* of a recast remains — turning that into seconds needs the
    duration, and the duration is a property of the ability.
    """

    id: str
    key: str
    kind: AbilityKind = AbilityKind.GCD
    recast_s: float = 2.5
    condition: Condition = field(default_factory=Condition.always)
    # Minimum spacing enforced by us, independent of the game's own cooldown. Guards
    # against double-firing when the cooldown probe flickers around zero.
    min_gap_s: float = 0.0
    hold_ms: int = 40
    # Cast time in seconds; 0 means instant. This is what lets the planner cooperate with
    # the mechanic loop rather than fight it: a hard cast started 1.8s before a scheduled
    # dodge is a cast that gets cancelled, and the damage is simply lost.
    cast_time_s: float = 0.0
    # What this points at. `CURRENT` for damage abilities, which is why a DPS profile
    # never pays for targeting; heals and utility name an ally instead.
    #
    # This one field is what unifies healing with the rotation. A healer's "rotation" is
    # a priority list whose abilities target allies and whose conditions read party HP —
    # so triage falls out of list ordering, and nothing needs a second planner.
    target: TargetSpec = field(default_factory=TargetSpec.current)
    comment: str = ""

    @property
    def instant(self) -> bool:
        return self.cast_time_s <= 0.0

    def fits_window(self, window_s: float) -> bool:
        """Can this be started with `window_s` of uninterrupted time available?"""
        return self.instant or self.cast_time_s <= window_s

    def available(self, state: WorldState) -> bool:
        """Is the game telling us this is off cooldown?

        Absence of the field is treated as available: many abilities are not on the
        hotbar slots we probe, and requiring a reading for all of them would silently
        disable most of the rotation.
        """
        ready = state.field(f"action.{self.id}.ready")
        if ready is not None and ready.value is not None:
            return bool(ready.value)
        cd = state.field(f"action.{self.id}.cooldown_s")
        if cd is not None and isinstance(cd.value, (int, float)):
            return float(cd.value) <= 0.05
        return True

    def usable(self, state: WorldState) -> bool:
        return self.available(state) and self.condition.evaluate(state)


@dataclass(slots=True)
class RotationProfile:
    """A priority-ordered ability list plus the timing constants it schedules against.

    Order *is* the priority: earlier entries win. That mirrors how rotations are written
    and reviewed by players, which matters because the person maintaining this list is
    reasoning about the game, not about the code.
    """

    id: str
    abilities: list[Ability] = field(default_factory=list)
    gcd_recast_s: float = 2.5
    animation_lock_ms: int = 600
    weave_gap_ms: int = 20
    max_weaves: int = 2
    description: str = ""

    def gcds(self) -> list[Ability]:
        return [a for a in self.abilities if a.kind is AbilityKind.GCD]

    def ogcds(self) -> list[Ability]:
        return [a for a in self.abilities if a.kind is AbilityKind.OGCD]

    def by_id(self, ability_id: str) -> Ability | None:
        return next((a for a in self.abilities if a.id == ability_id), None)

    def required_fields(self) -> set[str]:
        out: set[str] = set()
        for ability in self.abilities:
            out |= ability.condition.fields()
        return out

    def weave_capacity(self) -> int:
        """How many off-GCDs actually fit between this GCD and the next.

        Derived from the timing constants rather than hardcoded to two, so a profile with
        a shortened GCD (skill speed, haste buffs) gets the right answer instead of
        clipping.
        """
        window_ms = self.gcd_recast_s * 1000 - self.animation_lock_ms
        slot_ms = self.animation_lock_ms + self.weave_gap_ms
        return max(0, min(self.max_weaves, int(window_ms // slot_ms)))


@dataclass(slots=True)
class RotationDecision:
    plan: Plan
    gcd: Ability
    weaves: list[Ability]
    # Why this was chosen over something higher in the priority list. Populated when the
    # commitment board forced a substitution, so a session log answers "why did it cast
    # the filler there" without anyone having to reconstruct the state.
    yielded_to: str = ""
    target: str = "current"

    def describe(self) -> str:
        weaved = "+".join(a.id for a in self.weaves)
        suffix = f" <{self.yielded_to}>" if self.yielded_to else ""
        aimed = f" ->{self.target}" if self.target not in ("", "current") else ""
        return f"{self.gcd.id}" + (f" [{weaved}]" if weaved else "") + aimed + suffix


class RotationPlanner:
    """Turns a `RotationProfile` plus a `WorldState` into at most one plan per GCD.

    Commitment tracking is the subtle part. The cooldown probe reads zero for several
    consecutive frames when the GCD comes up, so a naive planner fires the same GCD three
    times in 50ms. After emitting, the planner refuses to emit again until the later of:

      * the GCD duration it just committed to, and
      * whatever the screen currently says is remaining.

    Using both means a flaky probe degrades to our own timing model rather than to
    double-firing, and an accurate probe corrects our model when a haste buff changes the
    real GCD out from under it.
    """

    def __init__(
        self,
        profile: RotationProfile,
        enabled: bool = True,
        targets: TargetResolver | None = None,
    ) -> None:
        self.profile = profile
        self.enabled = enabled
        self.targets = targets or TargetResolver()
        self._committed_until: float = 0.0
        self._last_used: dict[str, float] = {}
        self.decisions = 0
        # Both counters exist to make the two-loop interaction visible in a session
        # report. A rotation quietly clipping every cast looks identical to one that is
        # simply doing badly, unless you count.
        self.clips_avoided = 0
        self.yields = 0
        self.retargets = 0
        self.target_failures = 0
        self.last_decision: RotationDecision | None = None
        # Gap between pressing a target key and the ability that reads it. Three or four
        # frames is reliable; zero is not, because the ability would resolve against the
        # target it is replacing.
        self.target_settle_ms = 60
        # Hovering settles faster than a target change — the game reads the cursor
        # position directly rather than resolving a new target.
        self.hover_settle_ms = 40
        # How long to believe we still have the target we last asserted. The game can
        # change it under us — the target dies, we get knocked out of range — so the
        # belief expires rather than persisting forever.
        self.target_belief_s = 4.0
        self._target_slot: int | None = None
        self._target_asserted_at: float = 0.0

    def set_profile(self, profile: RotationProfile) -> None:
        """Swap the rotation. Clears commitment so the new profile starts immediately."""
        self.profile = profile
        self._committed_until = 0.0
        self._last_used.clear()

    def ready_at(self) -> float:
        return self._committed_until

    def decide(
        self,
        state: WorldState,
        at: float | None = None,
        board: "CommitmentBoard | None" = None,
    ) -> RotationDecision | None:
        at = now() if at is None else at
        if not self.enabled:
            return None
        if at < self._committed_until:
            return None

        # The screen is the authority on whether the GCD is actually back.
        gcd_remaining = state.num("player.gcd_remaining_s", 0.0)
        if gcd_remaining > 0.08:
            return None

        yielded = ""
        if board is not None:
            reserved = board.reserved_ability(at)
            if reserved is not None:
                # A mechanic has booked this GCD. Yield it — the mechanic's ability is
                # not optional and the GCD is not shared.
                ability = self.profile.by_id(reserved)
                if ability is None:
                    self.yields += 1
                    return None
                # The mechanic's ability still needs aiming — a reserved GCD is not
                # automatically a self-cast.
                target = self._resolve_target(ability, state, None, at)
                if target is None:
                    self.target_failures += 1
                    return None
                decision = RotationDecision(
                    plan=self._build_plan((ability, target), [], at),
                    gcd=ability,
                    weaves=[],
                    yielded_to="mechanic reserved",
                    target=target.describes or "current",
                )
                self._commit(at, ability, [], gcd_remaining)
                self.decisions += 1
                self.yields += 1
                self.last_decision = decision
                return decision

            if board.is_active(CommitmentKind.NO_TARGET, at):
                return None

        window = board.cast_window_s(at) if board is not None else float("inf")
        override = board.target_override(at) if board is not None else None

        selected = self._select_gcd(state, at, window, override)
        if selected is None:
            return None
        gcd, gcd_target = selected

        if window < float("inf") and not gcd.instant:
            yielded = f"{window:.1f}s window"
        elif window < float("inf") and window < 2.0 and gcd.instant:
            yielded = "instant for movement"
        if override is not None and gcd.target.is_noop:
            yielded = "add phase target"

        ogcd_blocked = board is not None and board.is_active(CommitmentKind.OGCD_RESERVED, at)
        weaves = (
            [] if ogcd_blocked else self._select_weaves(state, at, {gcd.id}, override)
        )
        plan = self._build_plan((gcd, gcd_target), weaves, at)

        self._commit(at, gcd, [a for a, _ in weaves], gcd_remaining)
        decision = RotationDecision(
            plan=plan,
            gcd=gcd,
            weaves=[a for a, _ in weaves],
            yielded_to=yielded,
            target=gcd_target.describes or "current",
        )
        self.decisions += 1
        self.last_decision = decision
        return decision

    # -- targeting ---------------------------------------------------------------

    def _resolve_target(
        self, ability: Ability, state: WorldState, override, at: float
    ) -> "TargetResolution | None":
        """Which keys, if any, this ability needs pressed first.

        Returns `None` when the ability cannot be aimed at all — which is how a healer
        priority list gets "do not heal when nobody is hurt" for free: the
        `lowest_hp_ally` spec fails to resolve and the ability is simply skipped.
        """
        from .targeting import TargetResolution, TargetSpec as _Spec

        spec = ability.target
        # An add phase redirects abilities that would otherwise hit the current target.
        # It must not redirect a heal — pointing a regen at the add is worse than useless.
        if override is not None and spec.is_noop and isinstance(override, _Spec):
            spec = override

        if spec.is_noop:
            return TargetResolution(describes="current")

        resolution = self.targets.resolve(spec, state)
        if resolution.failed:
            return None

        # Do not re-press a target key for someone we already have. A sustained heal on
        # the same ally should cost nothing after the first acquisition.
        if (
            resolution.slot is not None
            and resolution.slot == self._target_slot
            and (at - self._target_asserted_at) < self.target_belief_s
        ):
            return TargetResolution(describes=resolution.describes, slot=resolution.slot)

        return resolution

    def _select_gcd(
        self,
        state: WorldState,
        at: float,
        window_s: float = float("inf"),
        override=None,
    ) -> "tuple[Ability, TargetResolution] | None":
        """Highest-priority usable GCD that fits the time available *and* can be aimed.

        The window check turns preemption into cooperation: rather than starting the best
        ability and having it cancelled when movement begins, the planner drops down the
        list to something that will actually complete. A worse ability that resolves beats
        a better one that gets thrown away.

        The target check does the same job for healing. An unaimable ability is not a
        failure to report, it is simply not this ability's turn.
        """
        for ability in self.profile.gcds():
            if not self._respects_gap(ability, at):
                continue
            if not ability.fits_window(window_s):
                self.clips_avoided += 1
                continue
            if not ability.usable(state):
                continue
            resolution = self._resolve_target(ability, state, override, at)
            if resolution is None:
                self.target_failures += 1
                continue
            return ability, resolution
        return None

    def _select_weaves(
        self, state: WorldState, at: float, exclude: set[str], override=None
    ) -> "list[tuple[Ability, TargetResolution]]":
        capacity = self.profile.weave_capacity()
        if capacity <= 0:
            return []
        chosen: list[tuple[Ability, "TargetResolution"]] = []
        for ability in self.profile.ogcds():
            if len(chosen) >= capacity:
                break
            if ability.id in exclude:
                continue
            if not (self._respects_gap(ability, at) and ability.usable(state)):
                continue
            resolution = self._resolve_target(ability, state, override, at)
            if resolution is None:
                continue
            chosen.append((ability, resolution))
        return chosen

    def _respects_gap(self, ability: Ability, at: float) -> bool:
        if ability.min_gap_s <= 0:
            return True
        last = self._last_used.get(ability.id)
        return last is None or (at - last) >= ability.min_gap_s

    # -- plan construction -------------------------------------------------------

    def _build_plan(
        self,
        gcd: "tuple[Ability, TargetResolution]",
        weaves: "list[tuple[Ability, TargetResolution]]",
        at: float,
    ) -> Plan:
        """One plan containing target acquisition and every press it feeds.

        Target keys are steps in this plan, not a separate one. That is what keeps a swap
        to tens of milliseconds instead of a global cooldown, and — more importantly —
        keeps them atomic: a reflex firing between "press F3" and "press the heal" would
        otherwise land the heal on whatever the reflex left targeted.
        """
        ability, resolution = gcd
        steps: list[Step] = []
        offset = 0

        if resolution.point is not None:
            # Mouseover: hover, then cast. The hard target is untouched, so no belief is
            # recorded and no re-target is owed afterwards.
            steps.append(Step(0, MouseMove(ScreenPoint(*resolution.point))))
            offset = self.hover_settle_ms
            self.retargets += 1
        elif resolution.keys:
            for index, key in enumerate(resolution.keys):
                steps.append(Step(index * 20, Press(key, 30)))
            offset = (len(resolution.keys) - 1) * 20 + self.target_settle_ms
            self.retargets += 1
            self._target_slot = resolution.slot
            self._target_asserted_at = at
        elif resolution.slot is not None and not resolution.soft:
            self._target_slot = resolution.slot

        steps.append(Step(offset, Press(ability.key, ability.hold_ms)))

        weave_offset = offset + self.profile.animation_lock_ms + self.profile.weave_gap_ms
        for weave_ability, weave_target in weaves:
            for index, key in enumerate(weave_target.keys):
                steps.append(Step(weave_offset + index * 20, Press(key, 30)))
            if weave_target.keys:
                weave_offset += (len(weave_target.keys) - 1) * 20 + self.target_settle_ms
                self._target_slot = weave_target.slot
                self._target_asserted_at = at
            steps.append(Step(weave_offset, Press(weave_ability.key, weave_ability.hold_ms)))
            weave_offset += self.profile.animation_lock_ms + self.profile.weave_gap_ms

        name = ability.id + (
            "+" + "+".join(a.id for a, _ in weaves) if weaves else ""
        )
        if resolution.describes and resolution.describes != "current":
            name += f"@{resolution.describes}"

        return Plan(
            name=f"rotation:{name}",
            steps=tuple(steps),
            priority=Priority.ROTATION,
            preempt=False,
            # Expire before the next GCD. A rotation step that has not started by then is
            # answering a question about a game state that has already moved on.
            expires_in_ms=int(self.profile.gcd_recast_s * 1000 * 0.8),
        )

    def _commit(
        self, at: float, gcd: Ability, weaves: list[Ability], observed_remaining: float
    ) -> None:
        # Slightly under the nominal recast: we want to be ready the moment the screen
        # says the GCD is back, not a frame after.
        model = at + self.profile.gcd_recast_s * 0.9
        observed = at + observed_remaining
        self._committed_until = max(model, observed)
        self._last_used[gcd.id] = at
        for ability in weaves:
            self._last_used[ability.id] = at

    def stats(self) -> dict[str, object]:
        return {
            "profile": self.profile.id,
            "decisions": self.decisions,
            "clips_avoided": self.clips_avoided,
            "yields": self.yields,
            "last": self.last_decision.describe() if self.last_decision else None,
            "weave_capacity": self.profile.weave_capacity(),
        }
