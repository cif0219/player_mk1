"""World space: where things are on the arena floor, and what the camera can see.

Everything in `perceive/` produces HUD readings — values at fixed screen rectangles.
This package is about the other kind of knowledge: metres, bearings, and whether a given
patch of ground is currently in frame at all.

Two ideas carry it:

* **Self-calibrating transforms.** Rather than hand-measuring a minimap scale or a camera
  model, fit the transform from correspondences between known arena positions (waymarks)
  and where they appear on screen. Two waymarks give position, scale, and camera yaw
  simultaneously; four give a full ground-plane homography.
* **Observability is a first-class fact.** "There is no telegraph" and "I could not see
  whether there is a telegraph" are different states, and a player that conflates them
  walks into things.
"""

from .arena import (
    ArenaFrame,
    ArenaPoint,
    CameraPose,
    Observability,
    WaymarkLayout,
    bearing_deg,
    normalise_deg,
    shortest_turn_deg,
)
from .localize import Localizer, LocalizationResult, Sighting
from .transform import Homography, Similarity2D, fit_homography, fit_similarity

__all__ = [
    "ArenaFrame",
    "ArenaPoint",
    "CameraPose",
    "Homography",
    "LocalizationResult",
    "Localizer",
    "Observability",
    "Sighting",
    "Similarity2D",
    "WaymarkLayout",
    "bearing_deg",
    "fit_homography",
    "fit_similarity",
    "normalise_deg",
    "shortest_turn_deg",
]
