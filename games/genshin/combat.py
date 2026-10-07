"""Genshin combat profiles.

Genshin has no global cooldown, but the rotation planner's commitment model still fits:
`gcd_recast_s` becomes the pacing interval between decisions, and each decision presses
at most one button. That deliberately paces attacks slower than a human mashing the
mouse — a decision every ~0.6s is plenty to clear overworld content, and pacing from our
own clock means the planner never needs an attack-state probe that Genshin's HUD does
not offer.

The priority list is the standard shape for a single on-field character:

    burst (Q) when charged  >  skill (E) off cooldown  >  normal attack

No weaving — `max_weaves=0` — because Genshin has no off-GCD concept; everything is an
animation that locks out everything else.

> **Accuracy caveat**, same as the FFXIV jobs: cooldown durations vary per character and
> talent level. The defaults are typical mid-range values; set the real ones for your
> on-field character in the config, because a wrong recast misconverts the cooldown probe
> and the planner presses E into a cooldown (harmless, but wasted decisions).
"""

from __future__ import annotations

from player.policy.condition import Condition
from player.policy.rotation import Ability, AbilityKind, RotationProfile

# Genshin's own default bindings. Normal and charged attacks are the left mouse button
# and cannot be rebound to the keyboard — which is why the input layer understands
# "mouse1" as a pressable key.
DEFAULT_KEYS: dict[str, str] = {
    "normal_attack": "mouse1",
    "charged_attack": "mouse1",
    "skill": "e",
    "burst": "q",
    "sprint": "shift",
    "jump": "space",
    "interact": "f",
    "switch_1": "1",
    "switch_2": "2",
    "switch_3": "3",
    "switch_4": "4",
}

MOVEMENT_KEYS: dict[str, str] = {
    "forward": "w",
    "back": "s",
    "left": "a",
    "right": "d",
}

# Typical mid-range values; per-character, so expect to override in config.
DEFAULT_SKILL_COOLDOWN_S = 8.0
DEFAULT_BURST_COOLDOWN_S = 15.0

# A charged attack is the same button held past the charge threshold. 700ms covers most
# sword/claymore/catalyst charge times without tipping into claymore spin territory.
CHARGED_HOLD_MS = 700


def recasts(skill_cooldown_s: float, burst_cooldown_s: float) -> dict[str, float]:
    """Action id -> recast seconds, consumed by the derived cooldown sensor."""
    return {"skill": skill_cooldown_s, "burst": burst_cooldown_s}


def build_attack_only(keys: dict[str, str], attack_interval_s: float = 0.6) -> RotationProfile:
    """Normal attacks and nothing else — the `blm.dummy_safe` of this profile.

    One unconditional ability, so a failure means the perceive -> decide -> dispatch path
    is broken rather than that the cooldown probes need calibrating. Get this clicking,
    then switch to `genshin.solo`.
    """
    return RotationProfile(
        id="genshin.attack_only",
        abilities=[
            Ability(
                id="normal_attack",
                key=keys["normal_attack"],
                kind=AbilityKind.GCD,
                recast_s=attack_interval_s,
                hold_ms=45,
                comment="Click. The whole point is proving the pipeline moves.",
            ),
        ],
        gcd_recast_s=attack_interval_s,
        animation_lock_ms=200,
        max_weaves=0,
        description="Normal attacks only, for validating the pipeline end to end.",
    )


def build_solo(
    keys: dict[str, str],
    attack_interval_s: float = 0.6,
    skill_cooldown_s: float = DEFAULT_SKILL_COOLDOWN_S,
    burst_cooldown_s: float = DEFAULT_BURST_COOLDOWN_S,
) -> RotationProfile:
    """Single on-field character: burst > skill > normal attack.

    `min_gap_s` on burst and skill guards the same failure the FFXIV planner guards on
    the GCD: the darkness probe flickers around the ready threshold for a few frames, and
    without a floor the planner would press Q three times in 200ms. The burst gap is also
    sized to its cast cinematic, during which every input is eaten anyway.
    """
    abilities: list[Ability] = [
        Ability(
            id="burst",
            key=keys["burst"],
            kind=AbilityKind.GCD,
            recast_s=burst_cooldown_s,
            condition=Condition.ready("burst"),
            min_gap_s=4.0,
            hold_ms=50,
            comment="Fire the burst the moment energy and cooldown allow.",
        ),
        Ability(
            id="skill",
            key=keys["skill"],
            kind=AbilityKind.GCD,
            recast_s=skill_cooldown_s,
            condition=Condition.ready("skill"),
            min_gap_s=1.5,
            hold_ms=50,
            comment="Skill on cooldown; it feeds energy back into the burst.",
        ),
        Ability(
            id="normal_attack",
            key=keys["normal_attack"],
            kind=AbilityKind.GCD,
            recast_s=attack_interval_s,
            hold_ms=45,
            comment="Filler between everything else.",
        ),
    ]
    return RotationProfile(
        id="genshin.solo",
        abilities=abilities,
        gcd_recast_s=attack_interval_s,
        animation_lock_ms=200,
        max_weaves=0,
        description="Single-character priority list: burst > skill > normal attack.",
    )


def build_charged(
    keys: dict[str, str],
    attack_interval_s: float = 1.2,
    skill_cooldown_s: float = DEFAULT_SKILL_COOLDOWN_S,
    burst_cooldown_s: float = DEFAULT_BURST_COOLDOWN_S,
) -> RotationProfile:
    """`genshin.solo` with charged attacks as the filler.

    The filler is the same mouse button held past the charge threshold — which is exactly
    a `Press` with a long hold, and the reason mouse buttons went through the key path
    rather than getting a separate click primitive. Charged attacks drain stamina, which
    nothing perceives yet; the slower interval keeps the drain sustainable.
    """
    profile = build_solo(keys, attack_interval_s, skill_cooldown_s, burst_cooldown_s)
    abilities = [
        Ability(
            id="charged_attack",
            key=keys["charged_attack"],
            kind=AbilityKind.GCD,
            recast_s=attack_interval_s,
            hold_ms=CHARGED_HOLD_MS,
            comment="Held past the charge threshold; slower cadence respects stamina.",
        )
        if a.id == "normal_attack"
        else a
        for a in profile.abilities
    ]
    return RotationProfile(
        id="genshin.charged",
        abilities=abilities,
        gcd_recast_s=attack_interval_s,
        animation_lock_ms=200,
        max_weaves=0,
        description="Solo priority list with charged attacks as filler.",
    )
