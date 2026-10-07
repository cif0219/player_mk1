"""HUD calibration.

FFXIV's HUD is fully user-arrangeable, so the built-in layout is guaranteed wrong for any
specific installation. Calibration writes a JSON layout with regions expressed as
fractions of the client area, which is what lets one calibration survive a resolution
change at the same HUD scale.

Two modes, and the second is the one worth using:

* **Interactive** — capture one frame, print it as a coarse ASCII map, and take region
  coordinates by hand. Works with nothing installed.
* **From a recording** — auto-locate candidate bars by scanning for the HUD colours, then
  let the operator confirm. Much faster, and repeatable, because a recording does not move
  while you are measuring it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def run_calibration(game: str, out: str, session: str | None = None) -> int:
    if game == "ffxiv":
        from games.ffxiv.layout import (
            CAST_ORANGE,
            HP_GREEN,
            MP_PURPLE,
            TARGET_HP_YELLOW,
            Layout,
        )

        bars = (
            ("player_hp", HP_GREEN, 50),
            ("player_mp", MP_PURPLE, 50),
            ("cast_bar", CAST_ORANGE, 55),
            ("target_hp", TARGET_HP_YELLOW, 55),
        )
        remainder_note = (
            "Auto-location covers the coloured bars only. Hotbar geometry, the combat\n"
            "indicator, and the status bar still need checking by hand — open the JSON and\n"
            "adjust, then verify with:  python -m player replay --session <a recording>"
        )
    elif game == "genshin":
        from games.genshin.layout import BOSS_HP_WHITE, HP_GREEN, Layout

        bars = (
            ("player_hp", HP_GREEN, 55),
            ("boss_hp", BOSS_HP_WHITE, 60),
        )
        remainder_note = (
            "Auto-location covers the HP bars only, and player_hp is only found while it\n"
            "is green (healthy). Skill/burst icon regions and the party strip still need\n"
            "checking by hand — open the JSON and adjust, then verify with:\n"
            "  python -m player replay --session <a recording>"
        )
    else:
        print(f"no calibration routine for game {game!r}")
        return 2

    if session:
        image = _frame_from_session(session)
        if image is None:
            print(f"no frames in session {session}")
            return 2
    else:
        image = _grab_screen()
        if image is None:
            print("could not capture the screen; pass --session to calibrate from a recording")
            return 2

    h, w = image.shape[:2]
    print(f"calibrating against a {w}x{h} frame\n")

    layout = Layout()
    findings: dict[str, tuple[float, float, float, float] | None] = {}
    for name, color, tolerance in bars:
        found = _locate_bar(image, color, tolerance)
        findings[name] = found
        if found is None:
            print(f"  {name:12s} not found — keeping default {getattr(layout, name)}")
        else:
            x, y, rw, rh = found
            print(f"  {name:12s} -> RelRect(x={x:.4f}, y={y:.4f}, w={rw:.4f}, h={rh:.4f})")

    from player.geometry import RelRect

    for name, found in findings.items():
        if found is not None:
            setattr(layout, name, RelRect(*found))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    layout.save(out)
    print(f"\nwrote {out}")
    print()
    print(remainder_note)
    return 0


def _locate_bar(
    image: np.ndarray, color: tuple[int, int, int], tolerance: int, min_aspect: float = 6.0
) -> tuple[float, float, float, float] | None:
    """Find the widest region matching `color`, as normalised (x, y, w, h).

    Aspect ratio is the discriminator: HUD bars are long and thin, and the colour alone
    matches plenty of scenery. Requiring width to exceed height by `min_aspect` rejects
    almost everything that is not a bar.
    """
    diff = np.abs(image.astype(np.int16) - np.asarray(color, dtype=np.int16))
    mask = np.all(diff <= tolerance, axis=2)
    if not mask.any():
        return None

    h, w = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    best: tuple[int, int, int, int] | None = None
    best_width = 0

    for row in rows:
        cols = np.flatnonzero(mask[row])
        if cols.size < 2:
            continue
        run_start = cols[0]
        prev = cols[0]
        for col in cols[1:]:
            if col - prev > 3:  # tolerate a few pixels of gap within a bar
                width = prev - run_start
                if width > best_width:
                    best_width, best = width, (int(run_start), int(row), int(width), 1)
                run_start = col
            prev = col
        width = prev - run_start
        if width > best_width:
            best_width, best = width, (int(run_start), int(row), int(width), 1)

    if best is None or best_width < w * 0.02:
        return None

    x, y, bw, _ = best
    # Grow vertically from the seed row to recover the bar's real height.
    top = bottom = y
    while top > 0 and mask[top - 1, x : x + bw].mean() > 0.5:
        top -= 1
    while bottom < h - 1 and mask[bottom + 1, x : x + bw].mean() > 0.5:
        bottom += 1
    bh = max(1, bottom - top + 1)

    if bw / bh < min_aspect:
        return None
    return (x / w, top / h, bw / w, bh / h)


def _grab_screen() -> np.ndarray | None:
    try:
        import mss
    except ImportError:
        return None
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
    bgra = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
    return np.ascontiguousarray(bgra[:, :, 2::-1])


def _frame_from_session(session_root: str) -> np.ndarray | None:
    from player.record.session import Session

    session = Session(session_root)
    paths = session.frame_paths()
    if not paths:
        return None
    # A frame from the middle of a session is more likely to show combat HUD than the
    # first frame, which is often a loading screen or a menu.
    return session.load_frame(paths[len(paths) // 2])


def dump_layout(path: str) -> None:
    from games.ffxiv.layout import Layout

    print(json.dumps(Layout.load(path).to_dict(), indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="calibrate HUD regions")
    parser.add_argument("-g", "--game", default="ffxiv")
    parser.add_argument("-o", "--out", default="config/ffxiv.layout.json")
    parser.add_argument("-s", "--session", default=None)
    args = parser.parse_args()
    raise SystemExit(run_calibration(args.game, args.out, args.session))
