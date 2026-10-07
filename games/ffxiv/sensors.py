"""Assembling the FFXIV sensor bundle.

The field names produced here are the ones documented in docs/CONTRACTS.md. Adding a
field means adding it there too — the contract is the thing policy is written against,
so an undocumented field is a field nobody will use.
"""

from __future__ import annotations

from player.perceive.detector import DetectorSensor, OnnxDetector, TelegraphSegmenter
from typing import Any

from player.perceive.probes import (
    BarProbe,
    ColorProbe,
    CooldownRingProbe,
    PresenceProbe,
    TemplateProbe,
)
from player.perceive.sensor import DerivedSensor, ProbeSensor, SensorBundle
from player.state import WorldState

from .layout import CAST_ORANGE, HP_GREEN, MP_PURPLE, TARGET_HP_YELLOW, Layout


def build_vitals_sensor(layout: Layout) -> ProbeSensor:
    """HP, MP, cast bar, combat state, target. Every frame — these change continuously."""
    return ProbeSensor(
        "vitals",
        [
            BarProbe("player.hp_frac", layout.player_hp, HP_GREEN, tolerance=45),
            BarProbe("player.mp_frac", layout.player_mp, MP_PURPLE, tolerance=45),
            BarProbe("player.cast_progress", layout.cast_bar, CAST_ORANGE, tolerance=50),
            ColorProbe(
                "player.casting",
                layout.cast_bar,
                CAST_ORANGE,
                tolerance=50,
                threshold=0.04,
            ),
            PresenceProbe("player.in_combat", layout.combat_indicator, min_std=14.0),
            PresenceProbe("target.exists", layout.target_frame, min_std=10.0),
            BarProbe("target.hp_frac", layout.target_hp, TARGET_HP_YELLOW, tolerance=50),
        ],
        cadence=1,
    )


def build_hotbar_sensor(
    layout: Layout,
    slot_actions: dict[int, str],
    recasts: dict[str, float],
    cadence: int = 1,
) -> ProbeSensor:
    """Cooldown state for each mapped hotbar slot.

    The probe reports how much of the recast overlay is drawn — a unitless 0–1 — because
    that is all the pixels can tell you. Converting to seconds needs the ability's recast
    duration, which lives in the rotation profile, so the conversion happens in the
    derived sensor rather than being baked into the probe. One probe implementation then
    serves every slot on the bar.
    """
    probes = []
    for index, action_id in sorted(slot_actions.items()):
        probes.append(
            CooldownRingProbe(
                f"action.{action_id}.progress",
                layout.hotbar.slot_region(index),
            )
        )
    return ProbeSensor("hotbar", probes, cadence=cadence)


def build_derived_sensor(
    slot_actions: dict[int, str],
    recasts: dict[str, float],
    gcd_action: str,
    gcd_recast_s: float,
) -> DerivedSensor:
    """Turns cooldown progress into the fields policy actually reads.

    Keeping this arithmetic out of both the probes and the policy is deliberate: probes
    should report only what they see, and policy should read only what it needs.
    """
    action_ids = tuple(slot_actions.values())
    provides = tuple(
        name
        for action_id in action_ids
        for name in (f"action.{action_id}.ready", f"action.{action_id}.cooldown_s")
    ) + ("player.gcd_remaining_s",)

    def compute(state: WorldState) -> None:
        for action_id in action_ids:
            f = state.field(f"action.{action_id}.progress")
            if f is None or f.value is None:
                continue
            progress = float(f.value)
            recast = recasts.get(action_id, gcd_recast_s)
            remaining = max(0.0, progress * recast)
            state.set(
                f"action.{action_id}.cooldown_s",
                round(remaining, 3),
                confidence=f.confidence,
                source="derived:cooldown",
            )
            state.set(
                f"action.{action_id}.ready",
                progress <= 0.06,
                confidence=f.confidence,
                source="derived:cooldown",
            )

        # The GCD clock is read from one representative slot. Every GCD ability shares it,
        # so probing them all would be the same measurement N times.
        gcd = state.field(f"action.{gcd_action}.cooldown_s")
        if gcd is not None and gcd.value is not None:
            state.set(
                "player.gcd_remaining_s",
                gcd.value,
                confidence=gcd.confidence,
                source="derived:gcd",
            )

    return DerivedSensor("derived", provides, compute, cadence=1)


class StatusSensor:
    """Buff and DoT state from the status bar.

    Reading these needs an icon template per status at the user's HUD scale, and those
    templates are cut from a recording during calibration. Until they exist this sensor
    declares the fields and reports them as unreadable — value `None`, confidence 0.

    That is deliberately different from omitting the sensor. Declaring the fields lets
    `GameProfile.validate()` confirm at startup that every field the rotation reads is
    accounted for, and reporting zero confidence makes every condition that depends on
    them evaluate False. The result is that an uncalibrated player stands still rather
    than casting semi-randomly — which is the correct failure, and the one that is
    obvious rather than subtle.
    """

    cadence = 3  # buffs change on GCD boundaries, not per frame

    def __init__(
        self,
        layout: Layout,
        status_ids: tuple[str, ...],
        templates: dict[str, Any] | None = None,
        name: str = "status",
    ) -> None:
        self.name = name
        self.layout = layout
        self.status_ids = status_ids
        self.templates = templates or {}
        self.provides = tuple(
            field
            for status_id in status_ids
            for field in (f"buff.{status_id}.active", f"buff.{status_id}.remaining_s")
        )

    @property
    def calibrated(self) -> bool:
        return bool(self.templates)

    def observe(self, frame, geo, state: WorldState) -> None:
        for status_id in self.status_ids:
            template = self.templates.get(status_id)
            if template is None:
                state.set(
                    f"buff.{status_id}.active",
                    None,
                    confidence=0.0,
                    source="status:uncalibrated",
                )
                state.set(
                    f"buff.{status_id}.remaining_s",
                    None,
                    confidence=0.0,
                    source="status:uncalibrated",
                )
                continue

            region = geo.region_to_frame(self.layout.status_bar)
            probe = TemplateProbe(f"buff.{status_id}.active", self.layout.status_bar, template)
            reading = probe.read(frame.crop(region))
            state.set(
                f"buff.{status_id}.active",
                reading.value,
                confidence=reading.confidence,
                source="status:template",
            )
            # Duration needs digit templates as well; left unread until those exist
            # rather than guessed at from icon presence.
            state.set(
                f"buff.{status_id}.remaining_s",
                None,
                confidence=0.0,
                source="status:no_digits",
            )


def build_status_sensor(
    layout: Layout,
    status_ids: tuple[str, ...],
    templates: dict[str, Any] | None = None,
) -> StatusSensor:
    return StatusSensor(layout, status_ids, templates)


def build_detector_sensor(
    layout: Layout,
    model_path: str | None = None,
    cadence: int = 4,
) -> DetectorSensor:
    """Telegraph detection: trained model if one exists, HSV segmentation otherwise.

    The segmenter is both the fallback and the weak-label generator that produces the
    first model's training data — see `player/perceive/detector.py`.
    """
    detector = None
    if model_path:
        candidate = OnnxDetector(model_path)
        if candidate.load():
            detector = candidate
    return DetectorSensor(
        detector=detector,
        fallback=TelegraphSegmenter(),
        cadence=cadence,
    )


def build_bundle(
    layout: Layout,
    slot_actions: dict[int, str],
    recasts: dict[str, float],
    gcd_action: str,
    gcd_recast_s: float = 2.5,
    detector_model: str | None = None,
    status_ids: tuple[str, ...] = (),
    status_templates: dict[str, Any] | None = None,
) -> tuple[SensorBundle, DetectorSensor, StatusSensor]:
    """Assemble the full bundle in dependency order.

    Order is load-bearing: the derived sensor reads fields the hotbar sensor writes, and
    `SensorBundle` runs sensors in declaration order rather than sorting them, so the
    profile controls it explicitly.
    """
    detector = build_detector_sensor(layout, detector_model)
    status = build_status_sensor(layout, status_ids, status_templates)
    bundle = SensorBundle(
        [
            build_vitals_sensor(layout),
            build_hotbar_sensor(layout, slot_actions, recasts),
            build_derived_sensor(slot_actions, recasts, gcd_action, gcd_recast_s),
            status,
            detector,
        ]
    )
    return bundle, detector, status
