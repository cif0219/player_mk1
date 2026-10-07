"""Black Mage rotation profiles.

BLM is the right first job for this slice: its single-target rotation is a genuine
priority list over observable state (elemental gauge, MP, procs) rather than a fixed
sequence, and it is almost entirely stationary, so rotation correctness can be measured
without also solving movement.

> **Accuracy caveat.** The *structure* below — the phase loop, what gates what — is the
> shape of the job. The specific MP breakpoints and the exact ability set change with
> patches and with level. Validate against a current guide before trusting a parse, and
> treat this file as the thing to edit rather than the thing to trust. Everything here is
> data, which is the point: correcting it is editing a list, not rewriting a planner.

The gauge fields (`buff.astral_fire.active`, `buff.umbral_ice.active`, and so on) come
from status-bar and gauge probes. Until those are calibrated they read `None`, and every
condition that depends on them evaluates `False` — so an uncalibrated player does nothing
rather than doing something random. That default is deliberate.
"""

from __future__ import annotations

from player.policy.condition import Condition
from player.policy.rotation import Ability, AbilityKind, RotationProfile

# Default bindings for this job, on FFXIV's own first hotbar. Job-local rather than
# shared: a caster and a healer have disjoint ability sets, and one merged map would hide
# typos in whichever half is unused.
DEFAULT_KEYS: dict[str, str] = {
    "fire4": "1",
    "blizzard4": "2",
    "fire3": "3",
    "blizzard3": "4",
    "thunder": "5",
    "despair": "6",
    "paradox": "7",
    "manafont": "8",
    "triplecast": "9",
    "ley_lines": "0",
    "amplifier": "minus",
    "swiftcast": "equals",
}

# Hotbar slot -> logical action id. The keys themselves come from config, so a user with
# a different bar layout changes one mapping rather than every ability.
DEFAULT_SLOTS: dict[int, str] = {
    0: "fire4",
    1: "blizzard4",
    2: "fire3",
    3: "blizzard3",
    4: "thunder",
    5: "despair",
    6: "paradox",
    7: "manafont",
    8: "triplecast",
    9: "ley_lines",
    10: "amplifier",
    11: "swiftcast",
}

# Recast in seconds, used to turn cooldown-overlay progress into remaining seconds.
RECASTS: dict[str, float] = {
    "fire4": 2.5,
    "blizzard4": 2.5,
    "fire3": 2.5,
    "blizzard3": 2.5,
    "thunder": 2.5,
    "despair": 2.5,
    "paradox": 2.5,
    "manafont": 100.0,
    "triplecast": 60.0,
    "ley_lines": 120.0,
    "amplifier": 120.0,
    "swiftcast": 60.0,
}

# The GCD clock is read from one representative slot; every GCD shares it, so probing
# them all would be the same measurement twelve times.
GCD_REFERENCE = "fire4"

# Statuses the rotation reads. Declared here so `GameProfile.validate()` can confirm at
# startup that something produces them, rather than the rotation silently reading None.
STATUS_IDS: tuple[str, ...] = (
    "astral_fire",
    "umbral_ice",
    "thunder_dot",
    "firestarter",
)

IN_FIRE = Condition.buff("astral_fire")
IN_ICE = Condition.buff("umbral_ice")


def build(keys: dict[str, str], gcd_recast_s: float = 2.5) -> RotationProfile:
    """Single-target priority list.

    Read top to bottom: the first ability that is both off cooldown and whose condition
    holds is the one that fires. The phase loop falls out of the conditions rather than
    being sequenced explicitly, which is what lets it recover from an interrupted cast or
    a mistimed proc instead of desynchronising permanently.
    """
    abilities: list[Ability] = [
        # DoT first: a dropped Thunder costs more than any single filler cast.
        Ability(
            id="thunder",
            key=keys["thunder"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["thunder"],
            condition=Condition.any_(
                Condition.no_buff("thunder_dot"),
                Condition.field("buff.thunder_dot.remaining_s", "<", 4.0),
            ),
            comment="Refresh the DoT inside its pandemic window.",
        ),
        # Astral Fire: spend MP, then dump the remainder into Despair before it is wasted.
        Ability(
            id="despair",
            key=keys["despair"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["despair"],
            condition=Condition.all_(
                IN_FIRE,
                Condition.field("player.mp_frac", "<", 0.24),
                Condition.field("player.mp_frac", ">=", 0.08),
            ),
            comment="MP dump at the end of the fire phase.",
        ),
        Ability(
            id="fire4",
            key=keys["fire4"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["fire4"],
            condition=Condition.all_(IN_FIRE, Condition.field("player.mp_frac", ">=", 0.24)),
            comment="Main fire-phase filler.",
        ),
        Ability(
            id="blizzard3",
            key=keys["blizzard3"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["blizzard3"],
            condition=Condition.all_(IN_FIRE, Condition.field("player.mp_frac", "<", 0.08)),
            comment="Fire phase is out of MP; swap to ice to refill.",
        ),
        # Umbral Ice: refill MP, then swap back once full.
        Ability(
            id="fire3",
            key=keys["fire3"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["fire3"],
            condition=Condition.all_(IN_ICE, Condition.field("player.mp_frac", ">=", 0.97)),
            comment="MP is full; return to the fire phase.",
        ),
        Ability(
            id="blizzard4",
            key=keys["blizzard4"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["blizzard4"],
            condition=Condition.all_(IN_ICE, Condition.field("player.mp_frac", "<", 0.97)),
            comment="Ice-phase filler while MP refills.",
        ),
        # Neither gauge state: open with Fire III to establish Astral Fire.
        Ability(
            id="fire3_open",
            key=keys["fire3"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["fire3"],
            condition=Condition.all_(
                Condition.not_(IN_FIRE),
                Condition.not_(IN_ICE),
                Condition.truthy("target.exists"),
            ),
            min_gap_s=3.0,  # do not thrash if the gauge probe is flickering
            comment="Enter Astral Fire from neutral.",
        ),
        # Off-GCDs, weaved into the animation-lock gap after a GCD.
        Ability(
            id="manafont",
            key=keys["manafont"],
            kind=AbilityKind.OGCD,
            recast_s=RECASTS["manafont"],
            condition=Condition.all_(IN_FIRE, Condition.field("player.mp_frac", "<", 0.24)),
            comment="Extends the fire phase; only worth it inside it.",
        ),
        Ability(
            id="ley_lines",
            key=keys["ley_lines"],
            kind=AbilityKind.OGCD,
            recast_s=RECASTS["ley_lines"],
            condition=Condition.all_(Condition.truthy("player.in_combat"), IN_FIRE),
            comment="Hold for the fire phase where the extra casts land.",
        ),
        Ability(
            id="amplifier",
            key=keys["amplifier"],
            kind=AbilityKind.OGCD,
            recast_s=RECASTS["amplifier"],
            condition=Condition.truthy("player.in_combat"),
        ),
        Ability(
            id="triplecast",
            key=keys["triplecast"],
            kind=AbilityKind.OGCD,
            recast_s=RECASTS["triplecast"],
            condition=Condition.all_(Condition.truthy("player.in_combat"), IN_FIRE),
        ),
    ]

    return RotationProfile(
        id="blm.single_target",
        abilities=abilities,
        gcd_recast_s=gcd_recast_s,
        animation_lock_ms=600,
        weave_gap_ms=20,
        max_weaves=2,
        description="Black Mage single target, priority list. Validate against a current guide.",
    )


def build_dummy_safe(keys: dict[str, str], gcd_recast_s: float = 2.5) -> RotationProfile:
    """A minimal profile for first-light testing on a striking dummy.

    Two abilities, no gauge dependency, no off-GCDs. The point is to answer "does the
    perceive → decide → dispatch path work end to end" without the answer being confounded
    by whether the gauge probes are calibrated. Get this working, then switch to the real
    list.
    """
    return RotationProfile(
        id="blm.dummy_safe",
        abilities=[
            Ability(
                id="thunder",
                key=keys["thunder"],
                kind=AbilityKind.GCD,
                recast_s=RECASTS["thunder"],
                condition=Condition.truthy("target.exists"),
                min_gap_s=24.0,
                comment="Occasional DoT so the sequence is visible in a parse.",
            ),
            Ability(
                id="fire4",
                key=keys["fire4"],
                kind=AbilityKind.GCD,
                recast_s=RECASTS["fire4"],
                condition=Condition.truthy("target.exists"),
                comment="Filler. Any castable single-target spell works here.",
            ),
        ],
        gcd_recast_s=gcd_recast_s,
        description="Minimal two-button profile for validating the pipeline end to end.",
    )
