"""player_mk1 — screen-perception game player.

Layer order, and the only direction imports may run:

    capture -> perceive -> policy -> act
                  ^          ^
                  |          |
               strategy   safety

`games/` may import from `player/`. `player/` never imports from `games/`.
"""

from .clock import LatencyBudget, RateLimiter, ms_since, now
from .geometry import ClientPoint, FramePoint, Geometry, Rect, RelRect, ScreenPoint
from .state import Entity, Field, WorldState

__all__ = [
    "ClientPoint",
    "Entity",
    "Field",
    "FramePoint",
    "Geometry",
    "LatencyBudget",
    "RateLimiter",
    "Rect",
    "RelRect",
    "ScreenPoint",
    "WorldState",
    "ms_since",
    "now",
]

__version__ = "0.1.0"
