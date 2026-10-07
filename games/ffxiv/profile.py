"""The FFXIV game profile: everything the runtime needs, assembled."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from player.act.keymap import validate_keymap
from player.policy.targeting import TargetBindings, TargetResolver
from player.profile import GameProfile

from .castbar import CastBarSensor, CastLibrary
from .jobs import blm, whm
from .layout import Layout
from .party import (
    PartyListLayout,
    PartyLocatorSensor,
    build_party_derived,
    build_party_sensor,
)
from .reflexes import TelegraphOverlapSensor, build_reflexes
from .sensors import build_bundle

WINDOW_TITLE = "FINAL FANTASY XIV"

# Movement and camera, shared by every job.
MOVEMENT_KEYS: dict[str, str] = {
    "forward": "w",
    "back": "s",
    "left": "a",
    "right": "d",
}


def default_keys(job: str = "blm") -> dict[str, str]:
    """Hotbar bindings for a job, on FFXIV's own default first bar (1 through =).

    Ability bindings live with the job rather than here: a healer and a caster have
    disjoint ability sets, and one shared map would either be missing half of each or be
    a union that hides typos in whichever half is unused.
    """
    module = whm if job == "whm" else blm
    return {**MOVEMENT_KEYS, **module.DEFAULT_KEYS}


# Kept for callers that just want "the usual bindings" without naming a job.
DEFAULT_KEYS: dict[str, str] = default_keys("blm")


# Party-list slot keys. FFXIV's own defaults; frequently rebound, and a wrong guess here
# silently targets the wrong person — which on a healer means someone dies while the bot
# tops up a full-HP DPS.
DEFAULT_PARTY_SLOT_KEYS: tuple[str, ...] = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8")


@dataclass(slots=True)
class FFXIVConfig:
    layout_path: str | None = None
    # Overrides only. Job defaults are merged underneath at build time, so a config need
    # only mention the binds that differ.
    keys: dict[str, str] = field(default_factory=dict)
    job: str = "blm"
    rotation: str = "blm.dummy_safe"
    gcd_recast_s: float = 2.5
    detector_model: str | None = None
    dodge_key: str = "s"
    potion_hp_threshold: float = 0.35
    party_slot_keys: tuple[str, ...] = DEFAULT_PARTY_SLOT_KEYS
    enemy_slot_keys: tuple[str, ...] = ()
    cast_library_path: str | None = None


def build(config: FFXIVConfig | None = None) -> GameProfile:
    """Assemble the profile.

    Keybinds are validated here rather than at first press. A typo'd binding otherwise
    produces a rotation that silently skips one ability, which is far harder to diagnose
    than a startup error — and keybinds come from a config file, so they are exactly the
    thing that gets typo'd.
    """
    config = config or FFXIVConfig()
    keys = {**default_keys(config.job), **config.keys}

    problems = validate_keymap(keys)
    if problems:
        raise ValueError("invalid keybinds:\n  " + "\n  ".join(problems))

    layout = Layout.load_or_default(config.layout_path)

    targets = TargetResolver(
        TargetBindings(
            party_slots=config.party_slot_keys,
            enemy_slots=config.enemy_slot_keys,
            self_key=config.party_slot_keys[0] if config.party_slot_keys else "f1",
        )
    )

    is_healer = config.job == "whm"
    job = whm if is_healer else blm
    rotations = (
        {"whm.healer": whm.build(keys, config.gcd_recast_s)}
        if is_healer
        else {
            "blm.single_target": blm.build(keys, config.gcd_recast_s),
            "blm.dummy_safe": blm.build_dummy_safe(keys, config.gcd_recast_s),
        }
    )

    bundle, detector, status = build_bundle(
        layout=layout,
        slot_actions=job.DEFAULT_SLOTS,
        recasts=job.RECASTS,
        gcd_action=job.GCD_REFERENCE,
        gcd_recast_s=config.gcd_recast_s,
        detector_model=config.detector_model,
        status_ids=job.STATUS_IDS,
    )
    # Must run after the detector, since it reads the entities the detector writes.
    bundle.add(TelegraphOverlapSensor(layout))

    # Cast-bar reading is closed-set classification against the encounter's declared
    # casts, not OCR. With no learned library it reports nothing, and `CastTrigger` falls
    # through to the timeline trigger — late but correct.
    cast_library = (
        CastLibrary.load(config.cast_library_path)
        if config.cast_library_path and Path(config.cast_library_path).exists()
        else CastLibrary()
    )
    castbar = CastBarSensor(layout.boss_cast_text, layout.boss_cast_bar, cast_library)
    bundle.add(castbar)

    # Party perception is only worth its cost when something reads it.
    party_layout = PartyListLayout()
    if is_healer:
        bundle.add(build_party_sensor(party_layout))
        bundle.add(build_party_derived(party_layout.slots, whm.PARTY_THRESHOLDS))
        bundle.add(PartyLocatorSensor(party_layout, targets))

    reflexes = build_reflexes(
        layout,
        keys,
        dodge_key=keys.get(config.dodge_key, config.dodge_key),
        potion_hp_threshold=config.potion_hp_threshold,
    )
    if is_healer:
        for reflex in whm.build_emergency_reflexes(keys, targets):
            reflexes.add(reflex)
    if detector.backend == "none":
        # A reflex whose condition can never be true is noise on the status line. Disable
        # the group and say why, rather than leaving it enabled and inert.
        reflexes.set_group_enabled("ground_aoe", False)

    # `blm.single_target` reads gauge and DoT state, which needs status templates cut
    # during calibration. Without them every gauge condition is False and the rotation
    # does nothing — so say so at build time rather than leaving the user to discover a
    # player that stands still.
    if not status.calibrated and config.rotation == "blm.single_target":
        print(
            "warning: status templates are not calibrated, so gauge conditions will "
            "never hold and blm.single_target will not cast. Use blm.dummy_safe until "
            "calibration is done."
        )

    return GameProfile(
        name="ffxiv",
        window_title=WINDOW_TITLE,
        sensors=bundle,
        reflexes=reflexes,
        rotations=rotations,
        default_rotation=config.rotation,
        targets=targets,
        # Without these the player cannot tell whether it is in combat or what it is
        # fighting, and every decision would be a guess.
        critical_fields=(
            "player.hp_frac",
            "player.gcd_remaining_s",
            "target.exists",
        ),
        parameters={
            "rotation.enabled": True,
            "reflex.dodge_enabled": detector.backend != "none",
            "potion.hp_threshold": config.potion_hp_threshold,
            "status.calibrated": status.calibrated,
            "castbar.calibrated": castbar.calibrated,
        },
        detector_backend=detector.backend,
    )


def config_from_dict(data: dict[str, Any]) -> FFXIVConfig:
    # `keys` carries overrides only; `build` merges the job's defaults underneath, so a
    # healer config does not have to restate a caster's whole hotbar to change one bind.
    return FFXIVConfig(
        layout_path=data.get("layout_path"),
        keys=dict(data.get("keys", {})),
        job=data.get("job", "blm"),
        rotation=data.get("rotation", "blm.dummy_safe"),
        gcd_recast_s=float(data.get("gcd_recast_s", 2.5)),
        detector_model=data.get("detector_model"),
        dodge_key=data.get("dodge_key", "back"),
        potion_hp_threshold=float(data.get("potion_hp_threshold", 0.35)),
        party_slot_keys=tuple(data.get("party_slot_keys", DEFAULT_PARTY_SLOT_KEYS)),
        enemy_slot_keys=tuple(data.get("enemy_slot_keys", ())),
        cast_library_path=data.get("cast_library_path"),
    )
