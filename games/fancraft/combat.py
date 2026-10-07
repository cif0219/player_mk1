"""FantCraft combat profiles.

FantCraft's combat is deliberately FF14-shaped (docs/COMBAT.md over there): a
2.5s shared GCD, oGCDs on independent recasts, combos as follow-up windows. The
rotation planner fits it natively — better than either real target, because we
wrote both sides of the contract.

Keybinds come from the game's manifest: the client publishes which ability sits
on which hotbar slot, and `keys_from_layout` turns that into ability -> key.
Rearranging the hotbar in-game rebinds the tester on the next manifest load.

**The loop, closing:** the first live run's findings became game features and
those features became fields here. The HUD now renders a gold ring on the next
combo action (finding #2) and a red wash on out-of-range slots (finding #3),
so the guardian profile plays the full iron_cleave -> bulwark_slash ->
sentinel_strike combo entirely from the screen — priority-ordered on the
`action.<id>.combo_next` fields — and every ability is gated on its own
`action.<id>.out_of_range` reading. Both gates fail open: a missing probe
(no manifest, sensor disabled) leaves the press in, which is exactly the
pre-indicator behaviour.
"""

from __future__ import annotations

from player.policy.condition import Condition
from player.policy.rotation import Ability, AbilityKind, RotationProfile

from .layout import Layout

# FantCraft constants (shared/src/constants.ts). BASE_GCD_MS = 2500.
GCD_RECAST_S = 2.5

# Abilities this profile scripts, with the recasts the derived sensor needs.
# GCD abilities share the rolling 2.5s timer; oGCDs carry their own.
# (server/src/combat/abilityData.ts is the authority.)
RECASTS: dict[str, float] = {
    "iron_cleave": GCD_RECAST_S,
    "aegis_toss": GCD_RECAST_S,
    "shield_bash": 12.0,
    "provoke": 30.0,
    "starfire": GCD_RECAST_S,
    "mending_light": GCD_RECAST_S,
    "solar_lance": GCD_RECAST_S,
}

# Fallback hotbar order per job, matching the server's ability list order,
# used only before the first manifest lands. Slots are keys "1".."9", "0".
FALLBACK_SLOTS: dict[str, list[str]] = {
    "guardian": ["iron_cleave", "bulwark_slash", "provoke", "sentinel_strike",
                 "aegis_toss", "shield_bash", "earthen_roar"],
    "luminary": ["mending_light", "starfire", "solar_lance", "renewing_grace"],
}

MOVEMENT_KEYS: dict[str, str] = {
    "forward": "w",
    "back": "s",
    "left": "a",
    "right": "d",
    "jump": "space",
    "sprint": "shift",
    "target_cycle": "tab",
    "target_clear": "escape",
}

HAS_TARGET = Condition.truthy("target.exists")


def in_range(ability_id: str) -> Condition:
    """The slot is not wearing the out-of-range wash.

    `on_missing=True` deliberately inverts the framework's fail-closed default:
    the range wash is an optimization (skip presses the server would reject),
    not a safety gate — an unreadable slot must not silence the whole rotation,
    because the server referees range regardless.
    """
    return Condition.falsy(f"action.{ability_id}.out_of_range", on_missing=True)


def combo_next(ability_id: str) -> Condition:
    """The slot is wearing the combo-next gold ring right now."""
    return Condition.truthy(f"action.{ability_id}.combo_next")


def keys_from_layout(layout: Layout, job: str) -> dict[str, str]:
    """Ability id -> hotbar key, manifest first, fallback order otherwise."""
    keys: dict[str, str] = {}
    if layout.hotbar_slots:
        for slot in layout.hotbar_slots:
            if slot.ability_id and slot.key:
                keys[slot.ability_id] = slot.key.lower()
    if not keys:
        order = FALLBACK_SLOTS.get(job, [])
        slot_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        keys = dict(zip(order, slot_keys))
    return keys


def build_pipeline(keys: dict[str, str]) -> RotationProfile:
    """One unconditional-ish button — the `genshin.attack_only` of this game.

    iron_cleave whenever a target exists. A failure here means the perceive ->
    decide -> dispatch path is broken, not that anything needs tuning.
    """
    return RotationProfile(
        id="fancraft.pipeline",
        abilities=[
            Ability(
                id="iron_cleave",
                key=keys.get("iron_cleave", "1"),
                kind=AbilityKind.GCD,
                recast_s=GCD_RECAST_S,
                condition=HAS_TARGET,
                comment="Press 1 at a mob. The whole point is proving the loop moves.",
            ),
        ],
        gcd_recast_s=GCD_RECAST_S,
        animation_lock_ms=400,
        max_weaves=1,
        description="iron_cleave only, for validating the pipeline end to end.",
    )


def build_guardian(keys: dict[str, str]) -> RotationProfile:
    """Solo Guardian: the full 1-2-3 combo, read off the HUD's combo ring.

    Priority runs finisher-first: sentinel_strike when its ring is lit, else
    bulwark_slash when its ring is lit, else iron_cleave — which both fills
    and starts the next chain. shield_bash weaves on its own recast. Every
    ability is gated on its own out-of-range wash; the approach reflex
    (reflexes.py) owns closing the distance, so the rotation never presses
    into a `range` rejection knowingly.
    """
    return RotationProfile(
        id="fancraft.guardian",
        abilities=[
            Ability(
                id="shield_bash",
                key=keys.get("shield_bash", "6"),
                kind=AbilityKind.OGCD,
                recast_s=RECASTS["shield_bash"],
                condition=Condition.all_(HAS_TARGET, in_range("shield_bash")),
                min_gap_s=12.0,
                comment="Stun on cooldown; weaves between GCDs.",
            ),
            # The 1-2-3 combo, priority-ordered by combo depth: whichever step
            # the HUD says will land is the one to press, and after a finisher
            # (or a drop) nothing glows, so iron_cleave restarts the chain.
            Ability(
                id="sentinel_strike",
                key=keys.get("sentinel_strike", "4"),
                kind=AbilityKind.GCD,
                recast_s=GCD_RECAST_S,
                condition=Condition.all_(
                    HAS_TARGET, combo_next("sentinel_strike"), in_range("sentinel_strike")
                ),
                comment="Combo finisher — pressed whenever its gold ring is lit.",
            ),
            Ability(
                id="bulwark_slash",
                key=keys.get("bulwark_slash", "2"),
                kind=AbilityKind.GCD,
                recast_s=GCD_RECAST_S,
                condition=Condition.all_(
                    HAS_TARGET, combo_next("bulwark_slash"), in_range("bulwark_slash")
                ),
                comment="Combo second hit.",
            ),
            Ability(
                id="iron_cleave",
                key=keys.get("iron_cleave", "1"),
                kind=AbilityKind.GCD,
                recast_s=GCD_RECAST_S,
                condition=Condition.all_(HAS_TARGET, in_range("iron_cleave")),
                comment="Combo starter and filler.",
            ),
        ],
        gcd_recast_s=GCD_RECAST_S,
        animation_lock_ms=400,
        max_weaves=1,
        description="Guardian solo: full 1-2-3 combo from the HUD's combo ring, shield_bash weave.",
    )


def build_luminary(keys: dict[str, str]) -> RotationProfile:
    """Solo Luminary: keep yourself alive, otherwise starfire.

    mending_light needs no target — the server self-targets heals when the
    target is absent or hostile — so triage is a pure priority-list entry: heal
    below 60% HP with mana in reserve, else damage. Starfire costs 150 mana;
    the 15% floor keeps enough back for an emergency heal.
    """
    return RotationProfile(
        id="fancraft.luminary",
        abilities=[
            Ability(
                id="mending_light",
                key=keys.get("mending_light", "1"),
                kind=AbilityKind.GCD,
                recast_s=GCD_RECAST_S,
                condition=Condition.all_(
                    Condition.hp_below(0.6),
                    Condition.field("player.mp_frac", ">=", 0.15),
                ),
                comment="Self-heal under 60% — the server self-targets untargeted heals.",
            ),
            Ability(
                id="starfire",
                key=keys.get("starfire", "2"),
                kind=AbilityKind.GCD,
                recast_s=GCD_RECAST_S,
                condition=Condition.all_(
                    HAS_TARGET,
                    in_range("starfire"),
                    Condition.field("player.mp_frac", ">=", 0.15),
                ),
                comment="Damage filler while mana holds.",
            ),
        ],
        gcd_recast_s=GCD_RECAST_S,
        animation_lock_ms=400,
        max_weaves=1,
        description="Luminary solo: heal under 60%, starfire otherwise.",
    )
