"""Assembling the FantCraft sensor bundle.

Field names follow docs/CONTRACTS.md. The FantCraft-specific readings:

| Field                    | Type      | Produced by                             |
| ------------------------ | --------- | --------------------------------------- |
| `player.hp_frac`         | float 0-1 | overlay-tolerant bar probe (HP numbers  |
|                          |           | are drawn on the bar, "100 / 100")      |
| `player.mp_frac`         | float 0-1 | same, on the MP bar                     |
| `target.hp_frac`         | float 0-1 | bar probe on the target frame           |
| `target.exists`          | bool      | derived from the target bar reading     |
| `action.<id>.progress`   | float 0-1 | cooldown-overlay probe per hotbar slot  |
| `action.<id>.ready`      | bool      | derived                                 |
| `action.<id>.cooldown_s` | float     | derived                                 |
| `action.<id>.out_of_range`| bool     | red-wash probe per hotbar slot          |
| `action.<id>.combo_next` | bool      | gold-glow probe per hotbar slot         |

The hotbar probes are built from the game's own manifest: whatever ability the
client says is on slot N is the ability id that slot's probe reports on. A job
switch over there renames the fields here on the next manifest load — nothing
to recalibrate.

The target frame disappears from the DOM (`.hidden`) when nothing is targeted,
so its bar region shows raw 3D world. The bar probe then finds no green and
reports 0.0 at reduced confidence — same failure direction as Genshin's HP bar:
"no target" and "unreadable" both make the rotation hold, never press blind.
"""

from __future__ import annotations

import numpy as np

from player.perceive.probes import BarProbe, ProbeReading
from player.perceive.sensor import DerivedSensor, ProbeSensor, SensorBundle
from player.state import WorldState

from .layout import Layout


class TextOverlayBarProbe(BarProbe):
    """A `BarProbe` for bars with number text drawn over the fill.

    FantCraft renders "100 / 100" directly on its bars (client `.bar-text`).
    The glyphs punch ambiguous columns into the fill profile and the base
    probe's boundary-sharpness confidence correctly distrusts that — same
    problem the Genshin HP bar has, same fix: score confidence by how densely
    the fill colour covers the region *left* of the detected boundary. Text
    costs some density but a real fill keeps the majority.
    """

    def read(self, crop: np.ndarray) -> ProbeReading:
        base = super().read(crop)
        if base.value is None or not isinstance(base.value, float):
            return base
        if base.value <= 0.0:
            # "No fill found" conflates empty-bar with not-a-bar (frame hidden,
            # window occluded). Keep it visible but below every trust gate.
            return ProbeReading(0.0, min(base.confidence, 0.45))

        diff = np.abs(crop.astype(np.int16) - np.asarray(self.fill_color, dtype=np.int16))
        mask = np.all(diff <= self.tolerance, axis=2)
        boundary = max(1, int(round(base.value * mask.shape[1])))
        density = float(mask[:, :boundary].mean())
        confidence = float(np.clip((density - 0.45) / 0.35, 0.0, 1.0))
        return ProbeReading(base.value, confidence)


def build_vitals_sensor(layout: Layout) -> ProbeSensor:
    """Player HP/MP and the target bar. Every frame — these gate everything."""
    return ProbeSensor(
        "vitals",
        [
            TextOverlayBarProbe("player.hp_frac", layout.player_hp, layout.hp_color, tolerance=60),
            TextOverlayBarProbe("player.mp_frac", layout.player_mp, layout.mp_color, tolerance=60),
            TextOverlayBarProbe("target.hp_frac", layout.target_hp, layout.hp_color, tolerance=60),
        ],
        cadence=1,
    )


class AdaptiveCooldownProbe:
    """Recast progress from the radial sweep, thresholded per slot.

    The first live capture killed the fixed-threshold darkness probe: the
    slot's own idle art sat near the absolute threshold, so "ready" read as
    "cooling" (finding #5). Two things fixed it. The game now draws a
    conic sweep whose dark coverage IS the remaining fraction, over a
    brightened ready state; and this probe thresholds against the slot's own
    brightness — a running reference that rises instantly to the brightest
    thing seen (the ready state) and decays slowly, so it self-calibrates in
    the first ready moment and survives theme tweaks unchanged.

    Known conflation: the out-of-range wash also darkens the slot slightly.
    Harmless in practice — when the wash is up the rotation's range gate is
    already refusing the press, so a pessimistic cooldown reading changes
    nothing.
    """

    def __init__(self, name: str, region) -> None:
        self.name = name
        self.region = region
        self._reference: float | None = None

    def read(self, crop: np.ndarray) -> ProbeReading:
        if crop.size == 0:
            return ProbeReading(None, 0.0)
        luma = crop[..., 0] * 0.299 + crop[..., 1] * 0.587 + crop[..., 2] * 0.114
        bright = float(np.percentile(luma, 80))
        if self._reference is None:
            self._reference = bright
        else:
            # Rise instantly, decay at 0.5%/read: one glimpse of the ready
            # state calibrates the slot; a long cooldown cannot drag it down.
            self._reference = max(self._reference * 0.995, bright)

        if self._reference < 25.0:
            # Never seen anything bright enough to be a ready slot — occluded
            # window or an empty slot. Report unreadable, not "on cooldown".
            return ProbeReading(None, 0.0)

        dark = float((luma < self._reference * 0.55).mean())
        # Mid-sweep readings are the trustworthy ones; a value pinned at the
        # extremes could also be a stuck reference, so taper confidence there.
        confidence = float(np.clip(0.5 + self._reference / 100.0, 0.5, 1.0))
        return ProbeReading(round(dark, 4), confidence)


def build_hotbar_sensor(layout: Layout) -> ProbeSensor:
    """Recast progress per manifest-declared hotbar slot, via the radial
    sweep the game draws since finding #5. Slots the manifest maps to no
    ability get no probe — nothing to report on.
    """
    probes = [
        AdaptiveCooldownProbe(f"action.{slot.ability_id}.progress", slot.region)
        for slot in layout.hotbar_slots
        if slot.ability_id
    ]
    return ProbeSensor("hotbar", probes, cadence=2)


class RedTintProbe:
    """Is the slot under the out-of-range red wash?

    The client paints rgba(205,52,46,0.4) over the whole slot when the locked
    target is beyond the ability's range. Rather than matching the blended
    colour exactly (fragile against whatever art sits under the wash), this
    measures red-channel dominance: the wash lifts R far above G and B across
    most of the slot, which nothing in the normal slot art does.
    """

    def __init__(self, name: str, region, threshold: float = 0.15) -> None:
        self.name = name
        self.region = region
        self.threshold = threshold

    def read(self, crop: np.ndarray) -> ProbeReading:
        if crop.size == 0:
            return ProbeReading(None, 0.0)
        c = crop.astype(np.int16)
        red = (c[..., 0] > c[..., 1] + 30) & (c[..., 0] > c[..., 2] + 30) & (c[..., 0] > 90)
        ratio = float(red.mean())
        margin = abs(ratio - self.threshold) / self.threshold
        return ProbeReading(ratio >= self.threshold, float(np.clip(margin, 0.0, 1.0)))


class GoldGlowProbe:
    """Is the slot wearing the combo-next gold ring?

    The glow is rgba(255,200,60): high red, high green, low blue — a channel
    signature the slot art (slate blues, dark reds) never produces in volume.
    Threshold calibrated against live3 capture frames: a lit ring measured
    0.051 coverage at 48px slots, unlit slots 0.000-0.014 (neighbor spill
    included) — 0.03 splits the distributions with margin on both sides.
    """

    def __init__(self, name: str, region, threshold: float = 0.03) -> None:
        self.name = name
        self.region = region
        self.threshold = threshold

    def read(self, crop: np.ndarray) -> ProbeReading:
        if crop.size == 0:
            return ProbeReading(None, 0.0)
        c = crop.astype(np.int16)
        gold = (c[..., 0] > 150) & (c[..., 1] > 120) & (c[..., 2] < c[..., 1] - 40)
        ratio = float(gold.mean())
        margin = abs(ratio - self.threshold) / self.threshold
        return ProbeReading(ratio >= self.threshold, float(np.clip(margin, 0.0, 1.0)))


def build_slot_state_sensor(layout: Layout) -> ProbeSensor:
    """Range and combo affordances per manifest-declared hotbar slot.

    These read the indicators the game added *because* of playtest findings
    #2 and #3 — the loop closing on itself: game grows an affordance, player
    grows the probe, report scores whether both work.
    """
    probes: list = []
    for slot in layout.hotbar_slots:
        if not slot.ability_id:
            continue
        probes.append(RedTintProbe(f"action.{slot.ability_id}.out_of_range", slot.region))
        probes.append(GoldGlowProbe(f"action.{slot.ability_id}.combo_next", slot.region))
    return ProbeSensor("slot_state", probes, cadence=2)


def build_derived_sensor(recasts: dict[str, float]) -> DerivedSensor:
    """Progress -> ready/cooldown_s per ability, plus `target.exists`.

    `target.exists` is derived rather than probed: the target frame region
    shows arbitrary world pixels when hidden, so presence-by-contrast would
    hallucinate targets. A confident non-zero HP reading *is* the presence
    signal — the bar only renders when a target frame is up.
    """
    action_ids = tuple(recasts)
    # "action.*" is the established wildcard idiom (see SensorBundle
    # check_requirements): per-slot fields are manifest-dynamic, so the exact
    # set isn't knowable at declaration time. Conditions referencing a slot
    # that never materialises fall back to their on_missing behaviour.
    provides = ("target.exists", "action.*") + tuple(
        name
        for action_id in action_ids
        for name in (f"action.{action_id}.ready", f"action.{action_id}.cooldown_s")
    )

    def compute(state: WorldState) -> None:
        target = state.field("target.hp_frac")
        exists = (
            target is not None
            and isinstance(target.value, float)
            and target.value > 0.02
            and target.confidence >= 0.5
        )
        # Presence is only as sure as the bar reading, but ABSENCE is a
        # confident answer whenever the region was measured: the frame is
        # hidden between fights and the probe sees world pixels, which is
        # exactly what "no target" looks like. Without this the acquire
        # reflex (Condition.falsy) never fires and the tester idles forever.
        state.set(
            "target.exists",
            exists,
            confidence=(target.confidence if target is not None else 0.0) if exists else 0.9,
            source="derived:target",
        )

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


def build_bundle(
    layout: Layout, recasts: dict[str, float], *, hotbar_probes: bool = False
) -> SensorBundle:
    """Assemble in dependency order — derived last (SensorBundle runs in order).

    Hotbar probes default on: the game draws a radial recast sweep over a
    brightened ready state (the resolution of finding #5) and the probe
    self-calibrates per slot. `hotbar_probes=False` remains as the escape
    hatch — the planner's own GCD commitment and min_gap_s still pace
    everything when the probes are out.
    """
    sensors = [build_vitals_sensor(layout)]
    if hotbar_probes:
        sensors.append(build_hotbar_sensor(layout))
    if layout.hotbar_slots:
        sensors.append(build_slot_state_sensor(layout))
    sensors.append(build_derived_sensor(recasts if hotbar_probes else {}))
    return SensorBundle(sensors)
