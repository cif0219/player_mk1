"""Final Fantasy XIV profile.

Screen perception only: no injection, no memory reading, no packet work, no evasion. See
docs/SAFETY.md for the non-goals and the terms-of-service position, both of which are
deliberate constraints on this directory rather than unfinished work.
"""

from .layout import DEFAULT_LAYOUT, HotbarLayout, Layout
from .profile import DEFAULT_KEYS, FFXIVConfig, build, config_from_dict

__all__ = [
    "DEFAULT_KEYS",
    "DEFAULT_LAYOUT",
    "FFXIVConfig",
    "HotbarLayout",
    "Layout",
    "build",
    "config_from_dict",
]
