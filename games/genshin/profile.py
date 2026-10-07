"""The Genshin Impact game profile: everything the runtime needs, assembled.

Same ToS reality as FFXIV, stated here so it is chosen knowingly: HoYoverse's Terms of
Service prohibit third-party tools that automate gameplay, and accounts get suspended
for it. Running this against a live account is your decision and your risk. The
project-wide non-goals in docs/SAFETY.md apply unchanged — screen pixels in, synthetic
input out, nothing that touches the client process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from player.act.keymap import validate_keymap
from player.profile import GameProfile

from .combat import (
    DEFAULT_BURST_COOLDOWN_S,
    DEFAULT_KEYS,
    DEFAULT_SKILL_COOLDOWN_S,
    MOVEMENT_KEYS,
    build_attack_only,
    build_charged,
    build_solo,
    recasts,
)
from .layout import Layout
from .reflexes import build_reflexes
from .sensors import build_bundle

WINDOW_TITLE = "Genshin Impact"


def default_keys() -> dict[str, str]:
    return {**MOVEMENT_KEYS, **DEFAULT_KEYS}


@dataclass(slots=True)
class GenshinConfig:
    # Title substring to find the game window. The client is titled in its own language:
    # "Genshin Impact" (global), "原神" (Chinese). Configurable because a wrong title
    # means the tracker never finds the window and the player sees nothing.
    window_title: str = WINDOW_TITLE
    layout_path: str | None = None
    # Overrides only; the defaults above are merged underneath at build time.
    keys: dict[str, str] = field(default_factory=dict)
    rotation: str = "genshin.attack_only"
    # Pacing between decisions. There is no GCD to read off the screen, so this is our
    # own clock, and slower is safer.
    attack_interval_s: float = 0.6
    # Per-character. Wrong values misconvert the cooldown probes — harmless but wasteful.
    skill_cooldown_s: float = DEFAULT_SKILL_COOLDOWN_S
    burst_cooldown_s: float = DEFAULT_BURST_COOLDOWN_S
    dash_hp_threshold: float = 0.4
    retreat_hp_threshold: float = 0.2


def build(config: GenshinConfig | None = None) -> GameProfile:
    """Assemble the profile. Keybinds are validated here, at startup, for the same
    reason FFXIV's are: a typo'd binding otherwise surfaces as one silently missing
    ability, which is far harder to diagnose than a startup error.
    """
    config = config or GenshinConfig()
    keys = {**default_keys(), **config.keys}

    problems = validate_keymap(keys)
    if problems:
        raise ValueError("invalid keybinds:\n  " + "\n  ".join(problems))

    layout = Layout.load_or_default(config.layout_path)

    rotations = {
        "genshin.attack_only": build_attack_only(keys, config.attack_interval_s),
        "genshin.solo": build_solo(
            keys, config.attack_interval_s, config.skill_cooldown_s, config.burst_cooldown_s
        ),
        "genshin.charged": build_charged(
            keys,
            max(config.attack_interval_s, 1.0),
            config.skill_cooldown_s,
            config.burst_cooldown_s,
        ),
    }

    bundle = build_bundle(layout, recasts(config.skill_cooldown_s, config.burst_cooldown_s))
    reflexes = build_reflexes(
        keys,
        dash_hp_threshold=config.dash_hp_threshold,
        retreat_hp_threshold=config.retreat_hp_threshold,
    )

    return GameProfile(
        name="genshin",
        window_title=config.window_title,
        sensors=bundle,
        reflexes=reflexes,
        rotations=rotations,
        default_rotation=config.rotation,
        # HP is the one reading survival depends on. The skill probes degrade gracefully
        # (an unreadable icon just means the rotation falls through to normal attacks),
        # so they are deliberately not load-bearing.
        critical_fields=("player.hp_frac",),
        parameters={
            "rotation.enabled": True,
            "dash.hp_threshold": config.dash_hp_threshold,
            "retreat.hp_threshold": config.retreat_hp_threshold,
        },
        detector_backend="none",
    )


def config_from_dict(data: dict[str, Any]) -> GenshinConfig:
    return GenshinConfig(
        window_title=data.get("window_title", WINDOW_TITLE),
        layout_path=data.get("layout_path"),
        keys=dict(data.get("keys", {})),
        rotation=data.get("rotation", "genshin.attack_only"),
        attack_interval_s=float(data.get("attack_interval_s", 0.6)),
        skill_cooldown_s=float(data.get("skill_cooldown_s", DEFAULT_SKILL_COOLDOWN_S)),
        burst_cooldown_s=float(data.get("burst_cooldown_s", DEFAULT_BURST_COOLDOWN_S)),
        dash_hp_threshold=float(data.get("dash_hp_threshold", 0.4)),
        retreat_hp_threshold=float(data.get("retreat_hp_threshold", 0.2)),
    )
