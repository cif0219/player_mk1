"""The Phase 4 gate: how accurately can we tell where we are?

Everything in `docs/ENCOUNTERS.md` above the perception layer — navigation, mechanic
resolution, script execution — assumes the player knows its arena position and camera
yaw. This tool measures whether that assumption holds, in metres and degrees, *before*
anything is built on top of it.

Two modes:

* **Synthetic** — render minimaps with known ground truth across many positions, yaws and
  zoom levels, and measure exact error. Gives real numbers today, with no game.
* **Session** — run against recorded frames, where ground truth is unavailable. Measures
  detection rate, fit residual, and frame-to-frame self-consistency instead, which is what
  you can honestly claim without a reference.

Read the tiers, not the pass/fail. The output says which *classes of mechanic* the measured
accuracy supports, because "1.2m p95" means "fine for stacks, useless for tight
positionals" and that distinction is the actual decision being made.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

from games.ffxiv.layout import Layout
from games.ffxiv.minimap import MinimapReader
from player.geometry import Geometry, Rect
from player.world.arena import ArenaPoint, WaymarkLayout, default_ring_layout, shortest_turn_deg
from player.world.localize import Localizer
from tools.synth import render_minimap

# What each accuracy tier buys. Derived from mechanic tolerances, not from what happens
# to be achievable — the point is to find out which of these we are in, not to move the
# line until we pass.
TIERS: tuple[tuple[str, float, float, str], ...] = (
    ("tight", 0.30, 3.0, "tight positionals, exact tower soaks"),
    ("standard", 1.00, 5.0, "stacks, spreads, quadrant dodges, most tower mechanics"),
    ("coarse", 2.50, 10.0, "large-radius dodges and gross repositioning only"),
    ("unusable", math.inf, math.inf, "not enough to navigate on"),
)


@dataclass
class Measurement:
    position_errors_m: list[float] = field(default_factory=list)
    yaw_errors_deg: list[float] = field(default_factory=list)
    scale_errors_frac: list[float] = field(default_factory=list)
    residuals_px: list[float] = field(default_factory=list)
    attempts: int = 0
    localised: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    def record_failure(self, reason: str) -> None:
        key = reason.split("—")[0].strip()[:60]
        self.failures[key] = self.failures.get(key, 0) + 1

    @property
    def detection_rate(self) -> float:
        return self.localised / self.attempts if self.attempts else 0.0

    def percentile(self, values: list[float], p: float) -> float:
        if not values:
            return float("inf")
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
        return ordered[index]

    def tier(self) -> tuple[str, str]:
        pos_p95 = self.percentile(self.position_errors_m, 95)
        yaw_p95 = self.percentile(self.yaw_errors_deg, 95)
        for name, pos_limit, yaw_limit, supports in TIERS:
            if pos_p95 <= pos_limit and yaw_p95 <= yaw_limit:
                return name, supports
        return TIERS[-1][0], TIERS[-1][3]

    def report(self) -> list[str]:
        lines = [
            f"attempts:       {self.attempts}",
            f"localised:      {self.localised} ({self.detection_rate:.1%})",
        ]
        if self.position_errors_m:
            lines += [
                "",
                "position error (m)   "
                f"p50={self.percentile(self.position_errors_m, 50):.3f}  "
                f"p95={self.percentile(self.position_errors_m, 95):.3f}  "
                f"max={max(self.position_errors_m):.3f}",
                "camera yaw error (deg) "
                f"p50={self.percentile(self.yaw_errors_deg, 50):.2f}  "
                f"p95={self.percentile(self.yaw_errors_deg, 95):.2f}  "
                f"max={max(self.yaw_errors_deg):.2f}",
                "scale error (%)      "
                f"p50={self.percentile(self.scale_errors_frac, 50) * 100:.2f}  "
                f"p95={self.percentile(self.scale_errors_frac, 95) * 100:.2f}",
            ]
        if self.residuals_px:
            lines.append(
                "fit residual (px)    "
                f"p50={self.percentile(self.residuals_px, 50):.3f}  "
                f"p95={self.percentile(self.residuals_px, 95):.3f}"
            )
        if self.failures:
            lines += ["", "failures:"]
            lines += [
                f"  {count:5d}  {reason}"
                for reason, count in sorted(self.failures.items(), key=lambda kv: -kv[1])
            ]
        if self.position_errors_m:
            tier, supports = self.tier()
            lines += ["", f"TIER: {tier.upper()} — supports {supports}"]
        return lines


def measure_synthetic(
    *,
    samples: int = 400,
    arena: WaymarkLayout | None = None,
    minimap_px: int = 176,
    scales_px_per_m: tuple[float, ...] = (2.4, 3.2, 4.4),
    icon_radius_px: int = 3,
    noise_sigma: float = 0.0,
    seed: int = 7,
) -> Measurement:
    """Sweep positions, yaws, and zoom levels against known ground truth.

    Zoom levels vary because the minimap has several and the localiser is supposed to
    infer scale rather than assume it — if it only works at one zoom, that is worth
    finding out here rather than in an instance.
    """
    arena = arena or default_ring_layout()
    rng = np.random.default_rng(seed)

    # No `expected_marks` restriction: the renderer places all eight, so this exercises
    # the ambiguity search rather than sidestepping it. A/1, B/2, C/3 and D/4 share
    # colours in-game, and measuring the solver with that difficulty removed would measure
    # something we will never run.
    reader = MinimapReader(
        region=Layout().minimap,
        min_blob_px=max(2, icon_radius_px),
    )
    localizer = Localizer(arena)

    # A standalone canvas that is exactly the minimap, so the measurement isolates
    # minimap reading rather than also testing region resolution.
    region = Rect(0, 0, minimap_px, minimap_px)
    geo = Geometry(Rect(0, 0, minimap_px, minimap_px), (minimap_px, minimap_px))
    reader.region = _full_region()

    out = Measurement()
    for _ in range(samples):
        truth_player = ArenaPoint(
            float(rng.uniform(-arena.radius_m * 0.7, arena.radius_m * 0.7)),
            float(rng.uniform(-arena.radius_m * 0.7, arena.radius_m * 0.7)),
        )
        truth_yaw = float(rng.uniform(0.0, 360.0))
        truth_scale = float(rng.choice(scales_px_per_m))

        image = np.zeros((minimap_px, minimap_px, 3), dtype=np.uint8)
        render_minimap(
            image,
            region,
            arena,
            truth_player,
            truth_yaw,
            truth_scale,
            icon_radius_px=icon_radius_px,
        )
        if noise_sigma > 0:
            noisy = image.astype(np.float32) + rng.normal(0, noise_sigma, image.shape)
            image = np.clip(noisy, 0, 255).astype(np.uint8)

        out.attempts += 1
        reading = reader.read(image, geo)
        result = localizer.from_minimap(reading.to_sightings(), at=float(out.attempts))

        if not result.usable:
            out.record_failure(result.reason or "unlocalised")
            continue

        out.localised += 1
        out.position_errors_m.append(result.player.distance_to(truth_player))
        out.yaw_errors_deg.append(abs(shortest_turn_deg(result.pose.yaw_deg, truth_yaw)))
        out.scale_errors_frac.append(abs(result.scale_px_per_m - truth_scale) / truth_scale)
        out.residuals_px.append(result.residual_px)

    return out


def _full_region():
    from player.geometry import RelRect

    return RelRect(0.0, 0.0, 1.0, 1.0)


def measure_session(session_root: str, arena: WaymarkLayout | None = None) -> Measurement:
    """Measure against recorded frames.

    No ground truth exists here, so this deliberately reports *different* numbers:
    detection rate, fit residual, and frame-to-frame jump size. A large jump between
    consecutive frames means the estimate is unstable even when the residual looks fine,
    and instability is what actually breaks a movement controller.
    """
    from games.ffxiv.layout import Layout as FFLayout
    from player.record.session import Session

    arena = arena or default_ring_layout()
    session = Session(session_root)
    layout = FFLayout()
    reader = MinimapReader(region=layout.minimap)
    localizer = Localizer(arena)

    out = Measurement()
    previous: ArenaPoint | None = None
    jumps: list[float] = []

    for _index, _t, image in session.frames():
        h, w = image.shape[:2]
        geo = Geometry(Rect(0, 0, w, h), (w, h))
        out.attempts += 1
        reading = reader.read(image, geo)
        result = localizer.from_minimap(reading.to_sightings(), at=float(out.attempts))
        if not result.usable:
            out.record_failure(result.reason or "unlocalised")
            previous = None
            continue
        out.localised += 1
        out.residuals_px.append(result.residual_px)
        if previous is not None:
            jumps.append(previous.distance_to(result.player))
        previous = result.player

    if jumps:
        # Reported through the position channel so the same percentile machinery applies,
        # but it is a stability measure, not an accuracy one — labelled as such on output.
        out.position_errors_m = jumps
        out.yaw_errors_deg = [0.0] * len(jumps)
        out.scale_errors_frac = [0.0] * len(jumps)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="measure arena localisation accuracy")
    parser.add_argument("-s", "--session", default=None, help="measure against a recording")
    parser.add_argument("-n", "--samples", type=int, default=400)
    parser.add_argument("--icon-radius", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.0, help="gaussian pixel noise sigma")
    parser.add_argument("--radius", type=float, default=18.0, help="waymark ring radius in m")
    args = parser.parse_args()

    arena = default_ring_layout(radius_m=args.radius)

    if args.session:
        print(f"measuring against session {args.session}")
        print("(no ground truth available — position column is frame-to-frame JUMP, "
              "a stability measure)\n")
        measurement = measure_session(args.session, arena)
    else:
        print(
            f"synthetic sweep: {args.samples} samples, icon radius {args.icon_radius}px, "
            f"noise sigma {args.noise}\n"
        )
        measurement = measure_synthetic(
            samples=args.samples,
            arena=arena,
            icon_radius_px=args.icon_radius,
            noise_sigma=args.noise,
        )

    for line in measurement.report():
        print(line)

    print(
        "\nNote: a synthetic sweep measures the *solver*, not the game. It is a lower "
        "bound on error — real minimap icons overlap, get occluded by party dots, and "
        "sit on varying terrain. Re-run with --session against a real recording before "
        "trusting these numbers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
