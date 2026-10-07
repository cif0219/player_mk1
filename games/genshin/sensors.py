"""Assembling the Genshin sensor bundle.

Field names follow the same namespacing as the FFXIV slice (docs/CONTRACTS.md). The
Genshin-specific readings:

| Field                     | Type        | Produced by                          |
| ------------------------- | ----------- | ------------------------------------ |
| `player.hp_frac`          | float 0-1   | bar probe on the active-char HP bar  |
| `boss.exists`             | bool        | presence probe on the boss bar slot  |
| `boss.hp_frac`            | float 0-1   | bar probe, meaningful only if exists |
| `party.slot<N>.hp_frac`   | float 0-1   | bar probes under the portraits       |
| `action.skill.progress`   | float 0-1   | cooldown-sweep probe on the E icon   |
| `action.burst.progress`   | float 0-1   | cooldown/energy probe on the Q icon  |
| `action.<id>.ready`       | bool        | derived                              |
| `action.<id>.cooldown_s`  | float       | derived                              |

The burst icon carries two states in one region — recharging energy and post-cast
cooldown — and both render as a darkened icon, so one darkness probe covers "not ready"
for either reason. That conflation is fine for policy, which only ever asks "can I press
Q now".
"""

from __future__ import annotations

import numpy as np

from player.perceive.probes import BarProbe, CooldownRingProbe, ProbeReading
from player.perceive.sensor import DerivedSensor, ProbeSensor, SensorBundle
from player.state import WorldState

from .layout import BOSS_HP_WHITE, HP_GREEN, Layout


class TextOverlayBarProbe(BarProbe):
    """A `BarProbe` for bars with number text drawn over the fill.

    Genshin renders "42086 / 42086" directly on the active HP bar. The glyphs punch
    ambiguous columns into the fill profile, which the base probe's boundary-sharpness
    confidence correctly distrusts — measured live it scores a fully-readable bar at
    ~0.2, below the safety gate's threshold, so the player would refuse to act on a
    bar it is actually reading fine.

    The fix is a confidence measure that tolerates the overlay: how densely green the
    region *left* of the detected boundary is. Text costs some density but a real fill
    keeps the majority; a misaligned region or a colour-shifted (low HP) bar drops far
    below that. Density maps linearly onto confidence with 0.45 as the floor of "this
    is actually a bar".
    """

    def read(self, crop: np.ndarray) -> ProbeReading:
        base = super().read(crop)
        if base.value is None or not isinstance(base.value, float):
            return base
        if base.value <= 0.0:
            # "No green found" is indistinguishable from "the bar is not on screen"
            # (window occluded, or the bar has colour-shifted at low HP). The base
            # probe's 0.6 is above the policy threshold, which measured live meant an
            # occluded window read as zero HP and fired the retreat reflex every
            # cooldown. 0.45 keeps the reading visible but below every trust gate:
            # unreadable HP makes the player do less, never dodge phantom damage.
            return ProbeReading(0.0, min(base.confidence, 0.45))

        diff = np.abs(crop.astype(np.int16) - np.asarray(self.fill_color, dtype=np.int16))
        mask = np.all(diff <= self.tolerance, axis=2)
        boundary = max(1, int(round(base.value * mask.shape[1])))
        density = float(mask[:, :boundary].mean())
        confidence = float(np.clip((density - 0.45) / 0.35, 0.0, 1.0))
        return ProbeReading(base.value, confidence)


def build_vitals_sensor(layout: Layout) -> ProbeSensor:
    """Active-character HP and the boss bar. Every frame — these gate survival.

    Deliberately no `boss.exists` presence probe: the boss-bar region shows the 3D world
    through a transparent HUD, so a contrast test reads scenery as a boss. The colour-
    gated bar probe reads ~0 when no bar is there, which is the honest signal.
    """
    return ProbeSensor(
        "vitals",
        [
            TextOverlayBarProbe("player.hp_frac", layout.player_hp, HP_GREEN, tolerance=55),
            BarProbe("boss.hp_frac", layout.boss_hp, BOSS_HP_WHITE, tolerance=60),
        ],
        cadence=1,
    )


def build_party_sensor(layout: Layout) -> ProbeSensor:
    """Off-field party HP. Low cadence: it only changes on damage ticks and heals."""
    probes = [
        BarProbe(
            f"party.slot{i + 1}.hp_frac",
            layout.party.hp_region(i),
            HP_GREEN,
            tolerance=55,
        )
        for i in range(layout.party.slots)
    ]
    return ProbeSensor("party", probes, cadence=4)


def build_skill_sensor(layout: Layout) -> ProbeSensor:
    """Darkness fraction of the E and Q icons. Genshin sweeps a dark overlay across a
    cooling-down skill exactly the way FFXIV overlays a recast, so the same probe works.
    """
    return ProbeSensor(
        "skills",
        [
            CooldownRingProbe("action.skill.progress", layout.skill_icon),
            CooldownRingProbe("action.burst.progress", layout.burst_icon),
        ],
        cadence=2,
    )


def build_derived_sensor(recasts: dict[str, float]) -> DerivedSensor:
    """Cooldown progress -> the `ready`/`cooldown_s` fields policy reads.

    Same shape as the FFXIV derived sensor: probes report what they see (a unitless
    darkness fraction), the recast duration lives with the ability definition, and the
    conversion happens here so neither has to know about the other.
    """
    action_ids = tuple(recasts)
    provides = tuple(
        name
        for action_id in action_ids
        for name in (f"action.{action_id}.ready", f"action.{action_id}.cooldown_s")
    )

    def compute(state: WorldState) -> None:
        for action_id in action_ids:
            f = state.field(f"action.{action_id}.progress")
            if f is None or f.value is None:
                continue
            progress = float(f.value)
            remaining = max(0.0, progress * recasts[action_id])
            state.set(
                f"action.{action_id}.cooldown_s",
                round(remaining, 3),
                confidence=f.confidence,
                source="derived:cooldown",
            )
            state.set(
                f"action.{action_id}.ready",
                progress <= 0.12,
                confidence=f.confidence,
                source="derived:cooldown",
            )

    return DerivedSensor("derived", provides, compute, cadence=1)


def build_bundle(layout: Layout, recasts: dict[str, float]) -> SensorBundle:
    """Assemble the bundle in dependency order — derived last, since it reads the
    fields the skill sensor writes and `SensorBundle` runs in declaration order.
    """
    return SensorBundle(
        [
            build_vitals_sensor(layout),
            build_party_sensor(layout),
            build_skill_sensor(layout),
            build_derived_sensor(recasts),
        ]
    )
