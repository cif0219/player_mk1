"""Choosing what to point an ability at.

The hard constraint shapes the whole design: **a target swap has to be as fast as the next
skill use.** A healer whose tank drops to 20% cannot spend a GCD acquiring the tank and
another casting on them, and an add phase where the mob must die in eight seconds cannot
afford a swap that costs two of them.

So targeting is not a phase, a plan, or a state. It is a **prefix on the plan that needs
it** — the target key and the ability key are steps in the same schedule, tens of
milliseconds apart, and they succeed or fail together. See `Plan.targeted`.

Two kinds of target change, and they need different machinery:

* **Planned** — an add spawns on a known timeline. This is a script lookup, published as a
  `TARGET_OVERRIDE` commitment so the rotation redirects without the mechanic loop taking
  over the GCD.
* **Reactive** — who needs healing depends on how badly everyone else played, which no
  guide can predict. This is a policy over live party state, and it lives in the ability
  priority list like everything else.

FFXIV specifics worth knowing: party-list slot keys (F1–F8 by default) are the reliable
primitive. They are one keypress, deterministic, need no mouse, and do not depend on
whatever the tab-cycle happens to consider "next". Mouseover targeting is the other common
approach and is supported here, but it races the cursor against the cast and is the weaker
choice for automation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..state import WorldState


class TargetKind(Enum):
    """The closed vocabulary. A script selects from this; it never emits key names."""

    CURRENT = "current"  # whatever is already targeted — the common case, costs nothing
    SELF = "self"
    PARTY_SLOT = "party_slot"  # a fixed party-list position
    LOWEST_HP_ALLY = "lowest_hp_ally"  # resolved from party HP at press time
    # Hover the party frame instead of hard-targeting it. This is the healer's answer to
    # a real problem: a slot key *changes your target*, so the next damage ability hits
    # your ally and does nothing. Mouseover leaves the hard target on the boss, which is
    # why it is the standard among FFXIV healers rather than a curiosity.
    MOUSEOVER_ALLY = "mouseover_ally"
    ENEMY_SLOT = "enemy_slot"  # a fixed enemy-list position
    NEAREST_ENEMY = "nearest_enemy"
    MARKED_ENEMY = "marked_enemy"  # attack marker 1/2/3...
    FOCUS = "focus"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """What to target, as data."""

    kind: TargetKind = TargetKind.CURRENT
    slot: int = 0
    marker: str = ""
    # Skip allies already above this. A heal aimed at someone who recovered while we were
    # deciding is a wasted GCD, and on a healer that is the whole budget.
    hp_ceiling: float = 1.0
    # Exclude the player's own slot when scanning for the lowest ally. Some abilities
    # cannot target self; some should not.
    include_self: bool = True

    @staticmethod
    def current() -> "TargetSpec":
        return TargetSpec(kind=TargetKind.CURRENT)

    @staticmethod
    def self_() -> "TargetSpec":
        return TargetSpec(kind=TargetKind.SELF)

    @staticmethod
    def party_slot(slot: int) -> "TargetSpec":
        return TargetSpec(kind=TargetKind.PARTY_SLOT, slot=slot)

    @staticmethod
    def lowest_ally(hp_ceiling: float = 1.0, include_self: bool = True) -> "TargetSpec":
        return TargetSpec(
            kind=TargetKind.LOWEST_HP_ALLY, hp_ceiling=hp_ceiling, include_self=include_self
        )

    @staticmethod
    def mouseover_lowest_ally(
        hp_ceiling: float = 1.0, include_self: bool = True
    ) -> "TargetSpec":
        """Heal the lowest ally *without* losing the boss as the hard target."""
        return TargetSpec(
            kind=TargetKind.MOUSEOVER_ALLY, hp_ceiling=hp_ceiling, include_self=include_self
        )

    @staticmethod
    def enemy_slot(slot: int) -> "TargetSpec":
        return TargetSpec(kind=TargetKind.ENEMY_SLOT, slot=slot)

    @staticmethod
    def nearest_enemy() -> "TargetSpec":
        return TargetSpec(kind=TargetKind.NEAREST_ENEMY)

    @staticmethod
    def marked(marker: str) -> "TargetSpec":
        return TargetSpec(kind=TargetKind.MARKED_ENEMY, marker=marker)

    @property
    def is_noop(self) -> bool:
        return self.kind is TargetKind.CURRENT

    def describe(self) -> str:
        if self.kind is TargetKind.PARTY_SLOT:
            return f"party[{self.slot}]"
        if self.kind is TargetKind.ENEMY_SLOT:
            return f"enemy[{self.slot}]"
        if self.kind is TargetKind.MARKED_ENEMY:
            return f"marked({self.marker})"
        if self.kind is TargetKind.LOWEST_HP_ALLY:
            return f"lowest ally<{self.hp_ceiling:.0%}"
        return self.kind.value


@dataclass(frozen=True, slots=True)
class TargetResolution:
    """A resolved target: which keys to press, and what we believe we are targeting."""

    keys: tuple[str, ...] = ()
    describes: str = ""
    slot: int | None = None
    confidence: float = 1.0
    reason: str = ""
    # Screen position to hover for mouseover targeting. Set instead of `keys`.
    point: tuple[int, int] | None = None
    # True when this leaves the hard target alone — the mouseover case. Damage abilities
    # after it still hit the boss, so no belief needs tracking and no re-target is owed.
    soft: bool = False

    @property
    def is_noop(self) -> bool:
        return not self.keys and self.point is None

    @property
    def failed(self) -> bool:
        return self.confidence <= 0.0


@dataclass(slots=True)
class TargetBindings:
    """The user's keybinds for target acquisition.

    Party slots are a list rather than a formula because the sensible defaults (F1–F8) are
    frequently rebound, and a wrong guess here silently targets the wrong person — which
    on a healer means someone dies while the bot heals a full-HP DPS.
    """

    party_slots: tuple[str, ...] = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8")
    enemy_slots: tuple[str, ...] = ()
    self_key: str = "f1"
    nearest_enemy: str = "tab"
    focus: str = ""
    markers: dict[str, str] = field(default_factory=dict)

    def party_key(self, slot: int) -> str | None:
        return self.party_slots[slot] if 0 <= slot < len(self.party_slots) else None

    def enemy_key(self, slot: int) -> str | None:
        return self.enemy_slots[slot] if 0 <= slot < len(self.enemy_slots) else None


class TargetResolver:
    """Turns a `TargetSpec` plus live state into keys to press.

    Resolution happens at press time, not at plan time. The lowest-HP ally two seconds ago
    is not necessarily the one dying now, and a heal aimed at whoever *was* hurt is a
    wasted GCD.
    """

    def __init__(self, bindings: TargetBindings | None = None, party_size: int = 8) -> None:
        self.bindings = bindings or TargetBindings()
        self.party_size = party_size
        # Maps a party slot to the screen point to hover for mouseover targeting. Supplied
        # by the game profile, because only it knows where the party list is; refreshed as
        # geometry changes, because the window can move.
        self.slot_locator = None
        self.resolutions = 0
        self.failures = 0

    def set_slot_locator(self, locator) -> None:
        """Install (or refresh) the party-slot to screen-point mapping."""
        self.slot_locator = locator

    def resolve(self, spec: TargetSpec, state: WorldState) -> TargetResolution:
        if spec.is_noop:
            return TargetResolution(describes="current")

        self.resolutions += 1
        handler = {
            TargetKind.SELF: self._self,
            TargetKind.PARTY_SLOT: self._party_slot,
            TargetKind.LOWEST_HP_ALLY: self._lowest_ally,
            TargetKind.MOUSEOVER_ALLY: self._mouseover_ally,
            TargetKind.ENEMY_SLOT: self._enemy_slot,
            TargetKind.NEAREST_ENEMY: self._nearest_enemy,
            TargetKind.MARKED_ENEMY: self._marked,
            TargetKind.FOCUS: self._focus,
        }.get(spec.kind)

        if handler is None:
            return self._fail("unknown target kind")
        return handler(spec, state)

    # -- handlers ----------------------------------------------------------------

    def _self(self, spec: TargetSpec, state: WorldState) -> TargetResolution:
        return TargetResolution(keys=(self.bindings.self_key,), describes="self", slot=0)

    def _party_slot(self, spec: TargetSpec, state: WorldState) -> TargetResolution:
        key = self.bindings.party_key(spec.slot)
        if key is None:
            return self._fail(f"no keybind for party slot {spec.slot}")
        if not self._slot_present(state, spec.slot):
            # Targeting an empty slot in FFXIV does nothing, so the ability would fire at
            # whatever was already targeted — which for a heal is usually the boss.
            return self._fail(f"party slot {spec.slot} is empty")
        return TargetResolution(keys=(key,), describes=f"party[{spec.slot}]", slot=spec.slot)

    def _lowest_slot(self, spec: TargetSpec, state: WorldState) -> int | None:
        best_slot: int | None = None
        best_hp = spec.hp_ceiling

        for slot in range(self.party_size):
            if not spec.include_self and slot == self._self_slot(state):
                continue
            if not self._slot_present(state, slot):
                continue
            field = state.field(f"party.{slot}.hp_frac")
            if field is None or field.value is None or not field.trusted(0.5):
                # An unread party frame is not a healthy one. Skipping is the safe
                # direction: the alternative is healing someone we cannot see.
                continue
            hp = float(field.value)
            if hp < best_hp:
                best_hp, best_slot = hp, slot

        return best_slot

    def _lowest_ally(self, spec: TargetSpec, state: WorldState) -> TargetResolution:
        slot = self._lowest_slot(spec, state)
        if slot is None:
            return self._fail(f"no ally below {spec.hp_ceiling:.0%}")

        key = self.bindings.party_key(slot)
        if key is None:
            return self._fail(f"no keybind for party slot {slot}")
        hp = float(state.get(f"party.{slot}.hp_frac", 1.0))
        return TargetResolution(
            keys=(key,),
            describes=f"party[{slot}] @{hp:.0%}",
            slot=slot,
            reason=f"lowest of party at {hp:.0%}",
        )

    def _mouseover_ally(self, spec: TargetSpec, state: WorldState) -> TargetResolution:
        """Hover the hurt ally's party row, leaving the hard target on the boss.

        Falls back to slot targeting when no locator is configured. That is worse — it
        costs the boss as a target — but it is better than not healing, and the fallback
        is visible in the reason string rather than silent.
        """
        slot = self._lowest_slot(spec, state)
        if slot is None:
            return self._fail(f"no ally below {spec.hp_ceiling:.0%}")

        if self.slot_locator is None:
            fallback = self._party_slot(TargetSpec.party_slot(slot), state)
            return TargetResolution(
                keys=fallback.keys,
                describes=fallback.describes,
                slot=fallback.slot,
                confidence=fallback.confidence,
                reason="no mouseover locator; hard-targeting instead",
            )

        point = self.slot_locator(slot)
        if point is None:
            return self._fail(f"no screen position for party slot {slot}")

        return TargetResolution(
            point=point,
            describes=f"mouseover party[{slot}]",
            slot=slot,
            soft=True,
            reason="hard target preserved",
        )

    def _enemy_slot(self, spec: TargetSpec, state: WorldState) -> TargetResolution:
        key = self.bindings.enemy_key(spec.slot)
        if key is None:
            return self._fail(f"no keybind for enemy slot {spec.slot}")
        return TargetResolution(keys=(key,), describes=f"enemy[{spec.slot}]", slot=spec.slot)

    def _nearest_enemy(self, spec: TargetSpec, state: WorldState) -> TargetResolution:
        key = self.bindings.nearest_enemy
        if not key:
            return self._fail("no nearest-enemy keybind")
        # Tab-cycling is genuinely unreliable — the game's idea of "next" depends on facing
        # and range, so this can land on the wrong add. Fine for trash, not for a
        # kill-in-eight-seconds add; use an enemy-list slot there.
        return TargetResolution(
            keys=(key,), describes="nearest enemy", confidence=0.6, reason="tab-cycle ordering"
        )

    def _marked(self, spec: TargetSpec, state: WorldState) -> TargetResolution:
        key = self.bindings.markers.get(spec.marker)
        if not key:
            return self._fail(f"no keybind for marker {spec.marker!r}")
        return TargetResolution(keys=(key,), describes=f"marked({spec.marker})")

    def _focus(self, spec: TargetSpec, state: WorldState) -> TargetResolution:
        if not self.bindings.focus:
            return self._fail("no focus-target keybind")
        return TargetResolution(keys=(self.bindings.focus,), describes="focus")

    # -- helpers -----------------------------------------------------------------

    def _slot_present(self, state: WorldState, slot: int) -> bool:
        field = state.field(f"party.{slot}.present")
        if field is None or field.value is None:
            # No party sensor at all: assume the slot exists rather than refusing to
            # target anything. Solo content has no party list to read.
            return True
        return bool(field.value)

    def _self_slot(self, state: WorldState) -> int:
        value = state.get("party.self_slot")
        return int(value) if isinstance(value, (int, float)) else 0

    def _fail(self, reason: str) -> TargetResolution:
        self.failures += 1
        return TargetResolution(confidence=0.0, reason=reason)

    def stats(self) -> dict[str, object]:
        return {"resolutions": self.resolutions, "failures": self.failures}
