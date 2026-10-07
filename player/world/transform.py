"""Fitting transforms from correspondences, so nothing has to be hand-calibrated.

Two solvers, for two situations:

* `Similarity2D` — rotation, uniform scale, translation. Fits the minimap: two identified
  waymarks are enough to recover **position, scale, and camera yaw simultaneously**, which
  is why the minimap needs no measured pixels-per-metre constant.
* `Homography` — full projective map between the arena floor and the screen. Four
  identified waymarks in the 3D view give ground-plane projection directly from
  correspondences, with no camera model, no FOV constant, and no pitch estimate. The
  camera pose is implicit in the fit.

Both take the same view: **do not measure the camera, measure the correspondences.** A
hand-calibrated constant is wrong the moment the user changes a graphics setting; a fit
from things visible in the frame is self-correcting every time it runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .arena import ArenaPoint

Point2 = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Similarity2D:
    """`observed = scale * R(rotation) * source + translation`, rotation counter-clockwise."""

    rotation_deg: float
    scale: float
    tx: float
    ty: float
    residual_rms: float = 0.0
    sample_count: int = 0

    @property
    def valid(self) -> bool:
        return self.scale > 1e-9 and math.isfinite(self.scale)

    def apply(self, point: ArenaPoint) -> ArenaPoint:
        rad = math.radians(self.rotation_deg)
        cos, sin = math.cos(rad), math.sin(rad)
        x = self.scale * (point.x * cos - point.y * sin) + self.tx
        y = self.scale * (point.x * sin + point.y * cos) + self.ty
        return ArenaPoint(x, y)

    def invert(self) -> "Similarity2D":
        """The reverse map. Used to turn "where am I on the minimap" into arena metres."""
        if not self.valid:
            raise ValueError("cannot invert a degenerate similarity")
        inv_scale = 1.0 / self.scale
        inv_rot = -self.rotation_deg
        rad = math.radians(inv_rot)
        cos, sin = math.cos(rad), math.sin(rad)
        tx = -inv_scale * (self.tx * cos - self.ty * sin)
        ty = -inv_scale * (self.tx * sin + self.ty * cos)
        return Similarity2D(
            rotation_deg=inv_rot,
            scale=inv_scale,
            tx=tx,
            ty=ty,
            residual_rms=self.residual_rms,
            sample_count=self.sample_count,
        )

    def unapply(self, point: ArenaPoint) -> ArenaPoint:
        return self.invert().apply(point)


def fit_similarity(
    source: Sequence[ArenaPoint], observed: Sequence[ArenaPoint]
) -> Similarity2D | None:
    """Least-squares similarity from `source` onto `observed`.

    Closed form (the 2D case of Umeyama), not an iterative solve: with two or three
    waymarks an iterative fit would be both slower and less predictable, and this runs
    every frame.

    Returns `None` rather than a garbage transform when the input is degenerate — fewer
    than two points, or points that coincide. A localiser that silently returns nonsense
    for collinear input is worse than one that admits it does not know.
    """
    if len(source) != len(observed) or len(source) < 2:
        return None

    src = np.array([[p.x, p.y] for p in source], dtype=np.float64)
    dst = np.array([[p.x, p.y] for p in observed], dtype=np.float64)

    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    a = src - src_centroid
    b = dst - dst_centroid

    spread = float((a**2).sum())
    if spread < 1e-12:
        return None  # every source point in the same place

    dot = float((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum())
    cross = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())

    magnitude = math.hypot(dot, cross)
    if magnitude < 1e-12:
        return None

    rotation_deg = math.degrees(math.atan2(cross, dot))
    scale = magnitude / spread
    if not math.isfinite(scale) or scale <= 1e-9:
        return None

    rad = math.radians(rotation_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    rotated_centroid = np.array(
        [
            scale * (src_centroid[0] * cos - src_centroid[1] * sin),
            scale * (src_centroid[0] * sin + src_centroid[1] * cos),
        ]
    )
    translation = dst_centroid - rotated_centroid

    fit = Similarity2D(
        rotation_deg=rotation_deg,
        scale=scale,
        tx=float(translation[0]),
        ty=float(translation[1]),
        sample_count=len(source),
    )
    return Similarity2D(
        rotation_deg=fit.rotation_deg,
        scale=fit.scale,
        tx=fit.tx,
        ty=fit.ty,
        residual_rms=_similarity_residual(fit, source, observed),
        sample_count=len(source),
    )


def _similarity_residual(
    fit: Similarity2D, source: Sequence[ArenaPoint], observed: Sequence[ArenaPoint]
) -> float:
    """RMS reprojection error, in observed units.

    This is what makes the fit self-policing: a waymark misidentified as a different one
    produces a large residual, so the localiser can reject its own answer rather than
    confidently reporting a position that is metres out.
    """
    errors = [fit.apply(s).distance_to(o) for s, o in zip(source, observed)]
    return math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0


@dataclass(frozen=True, slots=True)
class Homography:
    """Projective map between the arena ground plane and screen pixels."""

    matrix: tuple[tuple[float, float, float], ...]
    residual_rms: float = 0.0
    sample_count: int = 0

    @property
    def as_array(self) -> np.ndarray:
        return np.array(self.matrix, dtype=np.float64)

    def apply(self, point: ArenaPoint) -> Point2 | None:
        """Arena metres to screen pixels. `None` when the point maps behind the camera."""
        h = self.as_array
        vec = h @ np.array([point.x, point.y, 1.0])
        if abs(vec[2]) < 1e-9:
            return None
        return float(vec[0] / vec[2]), float(vec[1] / vec[2])

    def unapply(self, screen: Point2) -> ArenaPoint | None:
        """Screen pixels to arena metres.

        This is the one that matters: it turns a ground telegraph detected at some screen
        rectangle into "that AoE covers arena position (x, y)", which is the only form a
        resolution rule can act on.
        """
        try:
            inverse = np.linalg.inv(self.as_array)
        except np.linalg.LinAlgError:
            return None
        vec = inverse @ np.array([screen[0], screen[1], 1.0])
        if abs(vec[2]) < 1e-9:
            return None
        return ArenaPoint(float(vec[0] / vec[2]), float(vec[1] / vec[2]))

    def yaw_at(self, player: ArenaPoint, step_m: float = 1.0) -> float | None:
        """Camera yaw implied by the homography, evaluated near the player.

        Perspective means the screen direction of "arena north" varies across the frame,
        so it is measured locally with a small step rather than taken globally. Evaluating
        it at the player's own position is what makes it the number the movement
        controller actually needs.
        """
        here = self.apply(player)
        ahead = self.apply(ArenaPoint(player.x, player.y + step_m))
        if here is None or ahead is None:
            return None
        dx = ahead[0] - here[0]
        dy = ahead[1] - here[1]
        if math.hypot(dx, dy) < 1e-9:
            return None
        # Screen +y is down, so "north appears to go up the screen" is dy < 0. The camera
        # yaw is the bearing whose screen projection points up.
        return math.degrees(math.atan2(dx, -dy)) % 360.0


def fit_homography(
    arena: Sequence[ArenaPoint], screen: Sequence[Point2]
) -> Homography | None:
    """Direct Linear Transform with Hartley normalisation.

    The normalisation is not optional garnish. Raw pixel coordinates in the hundreds
    against arena metres in the tens produce a badly conditioned design matrix, and the
    resulting homography is visibly wrong at the edges of the frame — which is exactly
    where the mechanics you most need to see tend to be.
    """
    if len(arena) != len(screen) or len(arena) < 4:
        return None

    src = np.array([[p.x, p.y] for p in arena], dtype=np.float64)
    dst = np.array([[p[0], p[1]] for p in screen], dtype=np.float64)

    norm_src, t_src = _normalise(src)
    norm_dst, t_dst = _normalise(dst)
    if norm_src is None or norm_dst is None:
        return None

    rows = []
    for (x, y), (u, v) in zip(norm_src, norm_dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    design = np.array(rows, dtype=np.float64)

    try:
        _, _, vt = np.linalg.svd(design)
    except np.linalg.LinAlgError:
        return None

    normalised_h = vt[-1].reshape(3, 3)
    # Undo the conditioning transforms to get back to real pixel/metre units.
    try:
        h = np.linalg.inv(t_dst) @ normalised_h @ t_src
    except np.linalg.LinAlgError:
        return None
    if abs(h[2, 2]) < 1e-12:
        return None
    h = h / h[2, 2]

    homography = Homography(
        matrix=tuple(tuple(float(v) for v in row) for row in h),
        sample_count=len(arena),
    )
    return Homography(
        matrix=homography.matrix,
        residual_rms=_homography_residual(homography, arena, screen),
        sample_count=len(arena),
    )


def _normalise(points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Centre and scale so the mean distance from the origin is sqrt(2)."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_distance = float(np.sqrt((centred**2).sum(axis=1)).mean())
    if mean_distance < 1e-12:
        return None, None
    scale = math.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]]
    )
    return centred * scale, transform


def _homography_residual(
    fit: Homography, arena: Sequence[ArenaPoint], screen: Sequence[Point2]
) -> float:
    errors = []
    for a, s in zip(arena, screen):
        projected = fit.apply(a)
        if projected is None:
            return float("inf")
        errors.append(math.hypot(projected[0] - s[0], projected[1] - s[1]))
    return math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0
