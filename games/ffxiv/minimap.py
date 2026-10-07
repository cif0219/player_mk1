"""Reading the minimap.

The single highest-value sensor for anything positional. It is a fixed HUD region that
yields player position, camera yaw, party positions, and arena bounds in one read — and it
does so without needing any 3D understanding.

The design point worth keeping: **nothing here is hand-calibrated.** The minimap's
pixels-per-metre, whether it rotates with the camera or is locked north, the UI scale —
all of it falls out of fitting a similarity between the known waymark layout and where
those waymarks appear. See `player/world/localize.py`. A hand-measured constant is silently
wrong the moment someone changes a UI setting, and being silently wrong about which way
you are facing is how a player dodges into the damage.

Waymark colours below are starting points. Detection quality on a real client is an open
question, which is exactly what `tools/measure_localization.py` exists to answer before
anything is built on top of this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from player.geometry import Geometry, Rect, RelRect
from player.state import WorldState
from player.world.localize import Sighting

# FFXIV waymark colours. A/B/C/D are the lettered marks, 1/2/3/4 the numbered ones.
# Each is distinct enough to separate by hue alone, which is the whole reason waymarks
# make a better anchor than any other arena feature.
WAYMARK_COLORS: dict[str, tuple[int, int, int]] = {
    "A": (232, 72, 72),
    "B": (232, 196, 72),
    "C": (108, 196, 232),
    "D": (196, 108, 232),
    "1": (232, 72, 72),
    "2": (232, 196, 72),
    "3": (108, 196, 232),
    "4": (196, 108, 232),
}

PARTY_DOT_COLOR = (108, 232, 128)


@dataclass(slots=True)
class MinimapDetection:
    """Something spotted on the minimap, as an offset from its centre in pixels.

    `candidates` carries every mark the blob's colour is consistent with. FFXIV's two
    waymark families share colours — A and 1 are both red, B and 2 both yellow — so a
    colour-keyed detector genuinely cannot tell them apart, and pretending otherwise picks
    wrong half the time. The localiser resolves the ambiguity geometrically instead.
    """

    kind: str  # waymark | party | player
    dx: float
    dy: float  # screen convention: +y is down
    candidates: tuple[str, ...] = ()
    confidence: float = 1.0
    pixels: int = 0

    @property
    def mark_id(self) -> str:
        """Best single guess. Only meaningful when the blob was unambiguous."""
        return self.candidates[0] if self.candidates else ""

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def radius_px(self) -> float:
        return math.hypot(self.dx, self.dy)


@dataclass(slots=True)
class MinimapReading:
    detections: list[MinimapDetection] = field(default_factory=list)
    centre: tuple[float, float] = (0.0, 0.0)
    radius_px: float = 0.0

    def waymarks(self) -> list[MinimapDetection]:
        return [d for d in self.detections if d.kind == "waymark"]

    def party(self) -> list[MinimapDetection]:
        return [d for d in self.detections if d.kind == "party"]

    def to_sightings(self) -> list[Sighting]:
        return [
            Sighting(
                mark_id=d.mark_id,
                minimap_offset=(d.dx, d.dy),
                confidence=d.confidence,
                candidates=d.candidates,
            )
            for d in self.waymarks()
        ]

    @property
    def ambiguous_count(self) -> int:
        return sum(1 for d in self.waymarks() if d.ambiguous)


class MinimapReader:
    """Finds waymarks and party dots in the minimap region.

    Colour-keyed blob detection rather than template matching. Minimap icons are tiny —
    often under ten pixels — which is well below where template correlation is meaningful,
    and their defining property at that size is hue rather than shape.
    """

    def __init__(
        self,
        region: RelRect,
        *,
        waymark_colors: dict[str, tuple[int, int, int]] | None = None,
        tolerance: int = 55,
        min_blob_px: int = 4,
        expected_marks: tuple[str, ...] | None = None,
        max_blobs_per_color: int = 3,
    ) -> None:
        self.region = region
        self.waymark_colors = waymark_colors or WAYMARK_COLORS
        self.tolerance = tolerance
        self.min_blob_px = min_blob_px
        # Restrict to the marks an encounter actually places. Most use one family or the
        # other, and saying so up front removes the ambiguity before it has to be solved.
        # Leaving it `None` keeps every mark in play and lets geometry sort it out.
        self.expected_marks = expected_marks
        self.max_blobs_per_color = max_blobs_per_color

    def _colour_groups(self) -> dict[tuple[int, int, int], tuple[str, ...]]:
        """Marks grouped by shared colour — the ambiguity classes.

        In FFXIV this comes out as {red: (A, 1), yellow: (B, 2), blue: (C, 3),
        purple: (D, 4)}, which is the whole reason the localiser needs a hypothesis search.
        """
        groups: dict[tuple[int, int, int], list[str]] = {}
        for mark_id, color in self.waymark_colors.items():
            if self.expected_marks is not None and mark_id not in self.expected_marks:
                continue
            groups.setdefault(color, []).append(mark_id)
        return {color: tuple(sorted(marks)) for color, marks in groups.items()}

    def read(self, frame_image: np.ndarray, geo: Geometry) -> MinimapReading:
        rect = geo.region_to_frame(self.region)
        if rect.w <= 2 or rect.h <= 2:
            return MinimapReading()

        crop = frame_image[rect.y : rect.bottom, rect.x : rect.right]
        cx, cy = crop.shape[1] / 2.0, crop.shape[0] / 2.0
        reading = MinimapReading(centre=(cx, cy), radius_px=min(cx, cy))

        for color, marks in self._colour_groups().items():
            # Every blob of this colour is a candidate for every mark of this colour. The
            # detector's honest output is "one of these", not a guess.
            blobs = _all_blobs(
                crop, color, self.tolerance, self.min_blob_px, limit=self.max_blobs_per_color
            )
            for bx, by, pixels in blobs:
                reading.detections.append(
                    MinimapDetection(
                        kind="waymark",
                        candidates=marks,
                        dx=bx - cx,
                        dy=by - cy,
                        # More pixels means a better-resolved centroid, so confidence
                        # tracks blob size rather than being asserted.
                        confidence=min(1.0, pixels / (self.min_blob_px * 4)),
                        pixels=pixels,
                    )
                )

        for index, (px, py, pixels) in enumerate(
            _all_blobs(crop, PARTY_DOT_COLOR, self.tolerance, self.min_blob_px, limit=7)
        ):
            reading.detections.append(
                MinimapDetection(
                    kind="party",
                    candidates=(f"p{index}",),
                    dx=px - cx,
                    dy=py - cy,
                    pixels=pixels,
                )
            )

        return reading


def _mask(crop: np.ndarray, color: tuple[int, int, int], tolerance: int) -> np.ndarray:
    diff = np.abs(crop.astype(np.int16) - np.asarray(color, dtype=np.int16))
    return np.all(diff <= tolerance, axis=2)


def _largest_blob(
    crop: np.ndarray, color: tuple[int, int, int], tolerance: int, min_px: int
) -> tuple[float, float, int] | None:
    blobs = _all_blobs(crop, color, tolerance, min_px, limit=8)
    if not blobs:
        return None
    return max(blobs, key=lambda b: b[2])


def _all_blobs(
    crop: np.ndarray,
    color: tuple[int, int, int],
    tolerance: int,
    min_px: int,
    limit: int,
    reject_edge: bool = True,
) -> list[tuple[float, float, int]]:
    """Connected colour-matched regions, as `(centroid_x, centroid_y, pixel_count)`.

    Centroid rather than bounding-box centre: at these sizes an icon is a handful of
    pixels with soft edges, and the centroid is markedly more stable frame to frame. That
    stability is what the similarity fit's residual measures, so it sets the localisation
    accuracy directly.

    `reject_edge` discards blobs touching the crop boundary. A waymark at the edge of the
    minimap is *clipped*, so its visible pixels are a biased sample and its centroid is
    pulled inward by however much was cut off — several pixels, which at 3 px/m is metres
    of position error. Dropping it costs one correspondence; keeping it poisons the fit.
    """
    mask = _mask(crop, color, tolerance)
    if not mask.any():
        return []

    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    out: list[tuple[float, float, int]] = []
    ys, xs = np.nonzero(mask)

    for sy, sx in zip(ys, xs):
        if visited[sy, sx] or len(out) >= limit:
            continue
        stack = [(int(sy), int(sx))]
        visited[sy, sx] = True
        sum_x = sum_y = count = 0
        touches_edge = False
        while stack:
            y, x = stack.pop()
            sum_x += x
            sum_y += y
            count += 1
            if x == 0 or y == 0 or x == w - 1 or y == h - 1:
                touches_edge = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if count >= min_px and not (reject_edge and touches_edge):
            # +0.5 converts from pixel *index* to continuous image coordinates: pixel i
            # covers [i, i+1), so its centre is at i+0.5. Without this every centroid
            # carries a constant half-pixel bias, which the similarity fit cannot see —
            # it shifts all the marks together, so the residual stays clean while the
            # recovered position is systematically off by half a pixel's worth of metres.
            out.append((sum_x / count + 0.5, sum_y / count + 0.5, count))

    return out


class MinimapSensor:
    """Writes minimap-derived fields into `WorldState`.

    Runs every frame. Position and camera yaw feed the movement controller, which needs a
    fresh yaw each tick specifically so that camera rotation is compensated rather than
    merely tolerated.
    """

    provides = (
        "arena.player_x",
        "arena.player_y",
        "arena.camera_yaw",
        "arena.scale_px_per_m",
        "arena.localized",
        "arena.residual_px",
        "party.count",
    )
    cadence = 1

    def __init__(self, reader: MinimapReader, localizer, name: str = "minimap") -> None:
        self.name = name
        self.reader = reader
        self.localizer = localizer
        self.last_reading: MinimapReading | None = None
        self.last_result = None

    def observe(self, frame, geo: Geometry, state: WorldState) -> None:
        reading = self.reader.read(frame.image, geo)
        self.last_reading = reading

        result = self.localizer.solve(reading.to_sightings(), at=frame.captured_at)
        self.last_result = result

        state.set("party.count", len(reading.party()), confidence=1.0, source="minimap")

        if not result.usable:
            # Explicitly unlocalised rather than absent. Downstream this is the difference
            # between "I am at the origin" and "I do not know where I am", and only one of
            # those is safe to walk on.
            for name in ("arena.player_x", "arena.player_y", "arena.camera_yaw"):
                state.set(name, None, confidence=0.0, source=f"minimap:{result.reason}")
            state.set("arena.localized", False, confidence=1.0, source="minimap")
            state.set("arena.residual_px", None, confidence=0.0, source="minimap")
            return

        conf = result.confidence
        state.set("arena.player_x", round(result.player.x, 3), confidence=conf, source=result.method)
        state.set("arena.player_y", round(result.player.y, 3), confidence=conf, source=result.method)
        state.set(
            "arena.camera_yaw", round(result.pose.yaw_deg, 2), confidence=conf, source=result.method
        )
        state.set(
            "arena.scale_px_per_m",
            round(result.scale_px_per_m, 4),
            confidence=conf,
            source=result.method,
        )
        state.set("arena.localized", True, confidence=conf, source=result.method)
        state.set(
            "arena.residual_px", round(result.residual_px, 3), confidence=conf, source=result.method
        )
