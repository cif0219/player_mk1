"""Genshin Impact game profile.

Everything Genshin-specific lives here; nothing under `player/` imports from it. The
runtime reaches this package only through `build` and `config_from_dict`, dispatched by
name in `player/cli.py`.
"""

from .profile import GenshinConfig, build, config_from_dict

__all__ = ["GenshinConfig", "build", "config_from_dict"]
