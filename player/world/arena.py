"""Arena coordinates, camera pose, and what the camera can currently see.

A fourth coordinate space, on top of the three in `geometry.py`. The others are all
pixels; this one is **metres on the arena floor**, and it is the space a raid guide is
actually written in ("go to A", "stack 5m apart", "get behind the boss").

Axis convention: +X is arena east, +Y is arena north, matching how FFXIV's own
waymark-relative guides are drawn. Bearings are degrees clockwise from north, because that
is what a compass reads and what every guide says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def normalise_deg(deg: float) -> float:
    """Wrap to [0, 360)."""
    return deg % 360.0


def shortest_turn_deg(from_deg: float, to_deg: float) -> float:
    """Signed smallest rotation from one bearing to another, in (-180, 180].

    Used everywhere the camera controller decides which way to turn. Getting this wrong
    means taking the 350-degree route instead of the 10-degree one, which in a mechanic
    window is the difference between arriving and not.
    """
    delta = (to_deg - from_deg + 180.0) % 360.0 - 180.0
    return delta + 360.0 if delta <= -180.0 else delta


@dataclass(frozen=True, slots=True)
class ArenaPoint:
    """A position on the arena floor, in metres."""

    x: float
    y: float

    def __add__(self, other: "ArenaPoint") -> "ArenaPoint":
        return ArenaPoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "ArenaPoint") -> "ArenaPoint":
        return ArenaPoint(self.x - other.x, self.y - other.y)

    def scaled(self, factor: float) -> "ArenaPoint":
        return ArenaPoint(self.x * factor, self.y * factor)

    def distance_to(self, other: "ArenaPoint") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    @property
    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def normalised(self) -> "ArenaPoint":
        mag = self.magnitude
        return ArenaPoint(0.0, 0.0) if mag < 1e-9 else ArenaPoint(self.x / mag, self.y / mag)

    def rotated(self, deg: float) -> "ArenaPoint":
        """Rotate clockwise by `deg`, matching the bearing convention."""
        rad = math.radians(-deg)
        cos, sin = math.cos(rad), math.sin(rad)
        return ArenaPoint(self.x * cos - self.y * sin, self.x * sin + self.y * cos)


def bearing_deg(origin: ArenaPoint, target: ArenaPoint) -> float:
    """Compass bearing from `origin` to `target`: degrees clockwise from north."""
    dx = target.x - origin.x
    dy = target.y - origin.y
    return normalise_deg(math.degrees(math.atan2(dx, dy)))


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Where the camera is looking.

    `yaw_deg` is the compass bearing the camera faces. It is the single most load-bearing
    number in this package, because FFXIV movement is camera-relative — `W` means "away
    from the camera", not "north". A yaw that is wrong by 90 degrees produces a player
    that walks confidently into a wall.

    `pitch_deg` matters for projecting ground telegraphs and is harder to observe; when it
    is unknown, the ground-plane homography subsumes it entirely, which is one reason the
    homography path is preferred where waymarks are visible.
    """

    yaw_deg: float = 0.0
    pitch_deg: float = 30.0
    fov_deg: float = 78.0
    confidence: float = 0.0

    @property
    def known(self) -> bool:
        return self.confidence > 0.0

    def relative_bearing(self, absolute_bearing_deg: float) -> float:
        """Convert a compass bearing into one relative to where the camera faces.

        Zero means dead ahead. This is the conversion every movement command goes through.
        """
        return shortest_turn_deg(self.yaw_deg, absolute_bearing_deg)

    def to_camera_relative(self, offset: ArenaPoint) -> ArenaPoint:
        """Rotate an arena-space offset into camera space.

        In camera space +Y is "away from the camera" (the `W` direction) and +X is "to the
        camera's right" (the `D` direction), which is exactly the basis the movement
        controller needs.
        """
        return offset.rotated(-self.yaw_deg)


@dataclass(frozen=True, slots=True)
class Observability:
    """What the camera can currently see.

    The reason this exists as a type rather than an implicit assumption: perception in a
    3D game is **partial and steerable**. A tower behind you is not "absent" and not
    "low confidence" — it is outside the frustum, and the correct response is to look,
    not to conclude.

    A policy that treats "no detections" as "nothing there" will walk into the thing it
    did not turn around to see.
    """

    pose: CameraPose = field(default_factory=CameraPose)
    player: ArenaPoint = field(default_factory=lambda: ArenaPoint(0.0, 0.0))
    # Effective view cone, narrower than the raw FOV: things at the extreme edge are
    # foreshortened enough that the detector should not be trusted on them.
    usable_fov_deg: float = 70.0
    max_range_m: float = 40.0

    def covers(self, point: ArenaPoint) -> bool:
        """Is this arena position plausibly on screen right now?"""
        if not self.pose.known:
            return False
        offset = point - self.player
        if offset.magnitude > self.max_range_m:
            return False
        if offset.magnitude < 1e-6:
            return True  # our own feet are always in frame
        relative = abs(self.pose.relative_bearing(bearing_deg(self.player, point)))
        return relative <= self.usable_fov_deg / 2.0

    def coverage_frac(self) -> float:
        """Share of the full circle currently in view. Drives the sweep policy."""
        return 0.0 if not self.pose.known else min(1.0, self.usable_fov_deg / 360.0)

    def bearing_to_see(self, point: ArenaPoint) -> float:
        """Camera yaw that would centre `point` in frame."""
        return bearing_deg(self.player, point)


@dataclass(frozen=True, slots=True)
class WaymarkLayout:
    """Where an encounter's waymarks sit, in arena metres.

    This is the bridge between a guide and the world. Guides are written in waymark terms,
    waymarks are placed deliberately and rendered distinctly, and their known separations
    are what make the localiser self-calibrating — no hand-measured minimap scale needed.
    """

    arena_id: str
    marks: dict[str, ArenaPoint] = field(default_factory=dict)
    radius_m: float = 20.0

    def get(self, mark_id: str) -> ArenaPoint | None:
        return self.marks.get(mark_id)

    def separation(self, a: str, b: str) -> float | None:
        pa, pb = self.marks.get(a), self.marks.get(b)
        return None if pa is None or pb is None else pa.distance_to(pb)

    def known_ids(self) -> set[str]:
        return set(self.marks)


@dataclass(frozen=True, slots=True)
class ArenaFrame:
    """A resolved snapshot of where we are and what we can see."""

    player: ArenaPoint
    pose: CameraPose
    layout: WaymarkLayout
    observability: Observability
    scale_px_per_m: float = 0.0
    confidence: float = 0.0
    source: str = ""

    def waymark(self, mark_id: str) -> ArenaPoint | None:
        return self.layout.get(mark_id)

    def bearing_to(self, target: ArenaPoint) -> float:
        return bearing_deg(self.player, target)

    def distance_to(self, target: ArenaPoint) -> float:
        return self.player.distance_to(target)


# Standard FFXIV waymark ring. Real encounters override this with measured positions;
# it exists so tests and the accuracy harness have something concrete to work against.
def default_ring_layout(arena_id: str = "generic", radius_m: float = 18.0) -> WaymarkLayout:
    """A/B/C/D at the cardinals, 1/2/3/4 at the intercardinals — the common convention."""
    marks: dict[str, ArenaPoint] = {}
    cardinals = {"A": 0.0, "B": 90.0, "C": 180.0, "D": 270.0}
    intercardinals = {"1": 45.0, "2": 135.0, "3": 225.0, "4": 315.0}
    for mark_id, bearing in {**cardinals, **intercardinals}.items():
        rad = math.radians(bearing)
        marks[mark_id] = ArenaPoint(
            x=round(radius_m * math.sin(rad), 6),
            y=round(radius_m * math.cos(rad), 6),
        )
    return WaymarkLayout(arena_id=arena_id, marks=marks, radius_m=radius_m)
