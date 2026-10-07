"""Generate a synthetic session for smoke-testing the pipeline without a game.

Draws a crude FFXIV-shaped HUD — health and MP bars, a hotbar with a sweeping cooldown
overlay, a target frame — into the regions the layout declares, then records it as a
normal session. The result replays through the real pipeline, so it exercises capture,
perception, policy, the timeline, and the guard chain end to end.

This is not a substitute for calibrating against the real client. It is the thing you run
to answer "is the pipeline wired up" separately from "are the regions right", because
debugging both at once is much harder than debugging either.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from games.ffxiv.layout import HP_GREEN, MP_PURPLE, TARGET_HP_YELLOW, Layout
from games.ffxiv.minimap import PARTY_DOT_COLOR, WAYMARK_COLORS
from player.geometry import Rect
from player.record.recorder import Recorder, RecorderConfig
from player.world.arena import ArenaPoint, CameraPose, WaymarkLayout, default_ring_layout


def _fill(image: np.ndarray, rect: Rect, color: tuple[int, int, int], fraction: float = 1.0) -> None:
    w = int(rect.w * max(0.0, min(1.0, fraction)))
    if w <= 0 or rect.h <= 0:
        return
    image[rect.y : rect.bottom, rect.x : rect.x + w] = color


def render_minimap(
    image: np.ndarray,
    region: Rect,
    arena: WaymarkLayout,
    player: ArenaPoint,
    camera_yaw_deg: float,
    scale_px_per_m: float,
    *,
    icon_radius_px: int = 3,
    party: list[ArenaPoint] | None = None,
) -> None:
    """Draw a minimap with known ground truth, in place.

    Mirrors FFXIV's own behaviour: the player is always at the centre and the map rotates
    with the camera, so arena north appears rotated by the camera yaw. Reproducing exactly
    that relationship is what makes the accuracy harness meaningful — if the renderer and
    the localiser disagree about the sign of the rotation, the measurement is of nothing.
    """
    cx = region.x + region.w / 2.0
    cy = region.y + region.h / 2.0
    image[region.y : region.bottom, region.x : region.right] = 32

    # Go through CameraPose rather than rotating by hand. There is exactly one definition
    # of "arena offset to camera-relative" in the codebase, and the renderer using a
    # second one is how a sign error hides: the measurement then agrees with itself and
    # disagrees with the game.
    pose = CameraPose(yaw_deg=camera_yaw_deg, confidence=1.0)

    def project(point: ArenaPoint) -> tuple[float, float]:
        local = pose.to_camera_relative(point - player)
        # Camera-relative +y is "away from the camera", which is up the minimap; screen
        # +y is down, hence the flip.
        return cx + local.x * scale_px_per_m, cy - local.y * scale_px_per_m

    for mark_id, position in arena.marks.items():
        color = WAYMARK_COLORS.get(mark_id)
        if color is None:
            continue
        px, py = project(position)
        _dot(image, px, py, icon_radius_px, color, region)

    for member in party or []:
        px, py = project(member)
        _dot(image, px, py, max(1, icon_radius_px - 1), PARTY_DOT_COLOR, region)

    # Player marker at dead centre.
    _dot(image, cx, cy, icon_radius_px, (250, 250, 250), region)


def _dot(
    image: np.ndarray, cx: float, cy: float, radius: int, color: tuple[int, int, int], bounds: Rect
) -> None:
    """Filled circle at a **sub-pixel** centre.

    Deliberately not a snapped integer rectangle. Quantising the centre to whole pixels
    puts a floor of roughly half a pixel on any centroid recovered from it — which at
    minimap scales is ~0.2m, large enough that the accuracy harness would end up measuring
    this function rather than the localiser. Testing membership against the true float
    centre lets the rendered pixel set shift with sub-pixel motion, so the centroid carries
    the information the solver is supposed to extract.
    """
    x0 = max(bounds.x, int(math.floor(cx)) - radius - 1)
    x1 = min(bounds.right, int(math.ceil(cx)) + radius + 2)
    y0 = max(bounds.y, int(math.floor(cy)) - radius - 1)
    y1 = min(bounds.bottom, int(math.ceil(cy)) + radius + 2)
    if x1 <= x0 or y1 <= y0:
        return

    ys = np.arange(y0, y1) + 0.5
    xs = np.arange(x0, x1) + 0.5
    inside = ((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) <= radius**2
    image[y0:y1, x0:x1][inside] = color


def render_frame(
    layout: Layout,
    size: tuple[int, int],
    t: float,
    gcd_recast_s: float = 2.5,
    *,
    arena: WaymarkLayout | None = None,
    player: ArenaPoint | None = None,
    camera_yaw_deg: float = 0.0,
    minimap_scale_px_per_m: float = 3.2,
) -> np.ndarray:
    """One frame at time `t`, with the GCD sweeping on its real period."""
    w, h = size
    image = np.full((h, w, 3), 24, dtype=np.uint8)

    # Bars. HP drifts down slowly, MP cycles like a caster's fire/ice phases.
    hp = 0.95 - 0.25 * (t % 40) / 40
    mp = 0.5 + 0.5 * np.sin(t / 6.0)
    _fill(image, layout.player_hp.to_client(w, h), (30, 40, 30))
    _fill(image, layout.player_hp.to_client(w, h), HP_GREEN, hp)
    _fill(image, layout.player_mp.to_client(w, h), (35, 30, 45))
    _fill(image, layout.player_mp.to_client(w, h), MP_PURPLE, float(mp))

    # Target frame, populated so the presence probe reads true.
    target = layout.target_frame.to_client(w, h)
    image[target.y : target.bottom, target.x : target.right] = 90
    _fill(image, layout.target_hp.to_client(w, h), (40, 40, 25))
    _fill(image, layout.target_hp.to_client(w, h), TARGET_HP_YELLOW, 0.8 - 0.5 * (t % 60) / 60)

    # Combat indicator.
    combat = layout.combat_indicator.to_client(w, h)
    image[combat.y : combat.bottom, combat.x : combat.right] = ((t * 7) % 255, 200, 60)

    # Hotbar. Every slot is bright (ready) except the GCD reference, which sweeps: the
    # overlay covers a shrinking share of the icon as the recast elapses.
    phase = (t % gcd_recast_s) / gcd_recast_s
    remaining = 1.0 - phase
    for index in range(layout.hotbar.slots):
        slot = layout.hotbar.slot_region(index).to_client(w, h)
        image[slot.y : slot.bottom, slot.x : slot.right] = 190
        if index == 0:
            dark_h = int(slot.h * remaining)
            if dark_h > 0:
                image[slot.y : slot.y + dark_h, slot.x : slot.right] = 25

    # Minimap, with the player orbiting and the camera slowly panning — so a replayed
    # synthetic session exercises the localiser and the movement controller rather than
    # handing them a static scene they cannot get wrong.
    arena = arena or default_ring_layout()
    if player is None:
        angle = t * 0.4
        player = ArenaPoint(6.0 * np.cos(angle), 6.0 * np.sin(angle))
    yaw = camera_yaw_deg if camera_yaw_deg else (t * 12.0) % 360.0
    render_minimap(
        image,
        layout.minimap.to_client(w, h),
        arena,
        player,
        yaw,
        minimap_scale_px_per_m,
    )

    return image


def generate(
    out_root: str | Path,
    name: str = "synthetic",
    *,
    seconds: float = 6.0,
    fps: int = 30,
    size: tuple[int, int] = (1920, 1080),
) -> Path:
    layout = Layout()
    recorder = Recorder(
        RecorderConfig(
            enabled=True,
            root=Path(out_root),
            name=name,
            frame_every_n=1,
            jpeg_quality=88,
        )
    )
    recorder.start({"synthetic": True, "client_rect": [0, 0, size[0], size[1]]})

    frames = int(seconds * fps)
    for index in range(1, frames + 1):
        t = index / fps
        image = render_frame(layout, size, t)
        recorder.frame(index, image, recorder.t0 + t, Rect(0, 0, size[0], size[1]))
    recorder.stop()
    return recorder.root


def main() -> int:
    parser = argparse.ArgumentParser(description="generate a synthetic session")
    parser.add_argument("-o", "--out", default="sessions")
    parser.add_argument("-n", "--name", default="synthetic")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    root = generate(
        args.out,
        args.name,
        seconds=args.seconds,
        fps=args.fps,
        size=(args.width, args.height),
    )
    print(f"wrote {root}")
    print(f"replay it with:  python -m player replay --session {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
