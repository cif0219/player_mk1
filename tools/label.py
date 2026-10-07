"""Weak-labelling telegraphs from recorded sessions.

This is how the detector gets its first training set without anyone annotating anything.
FFXIV telegraphs are strongly saturated orange/red ground decals, so HSV segmentation
finds most of them in any recording. Those labels are noisy — that is what "weak" means —
but a detector trained on them generalises to the occluded, edge-of-screen, and oddly-lit
cases segmentation misses, because the model learns what a telegraph *looks like* rather
than what colour it is.

Output is YOLO-format: one `.txt` per frame with `class cx cy w h`, all normalised.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from player.perceive.detector import TelegraphSegmenter
from player.record.session import Session


def label_session(
    session_root: str,
    out_dir: str,
    *,
    min_confidence: float = 0.25,
    max_per_frame: int = 8,
    copy_frames: bool = True,
) -> dict[str, int]:
    """Run the segmenter over every frame and write YOLO labels.

    Frames with no detections are written as empty label files rather than skipped:
    negatives matter as much as positives, and a training set of only-positives teaches a
    detector that telegraphs are always present.
    """
    session = Session(session_root)
    segmenter = TelegraphSegmenter()

    out = Path(out_dir)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if copy_frames:
        (out / "images").mkdir(parents=True, exist_ok=True)

    stats = {"frames": 0, "boxes": 0, "empty": 0}

    for path in session.frame_paths():
        image = session.load_frame(path)
        h, w = image.shape[:2]
        detections = [
            d for d in segmenter.detect(image) if d.confidence >= min_confidence
        ][:max_per_frame]

        lines = []
        for d in detections:
            cx = (d.bbox.x + d.bbox.w / 2) / w
            cy = (d.bbox.y + d.bbox.h / 2) / h
            bw = d.bbox.w / w
            bh = d.bbox.h / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out / "labels" / f"{path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if copy_frames:
            import shutil

            shutil.copy2(path, out / "images" / path.name)

        stats["frames"] += 1
        stats["boxes"] += len(lines)
        if not lines:
            stats["empty"] += 1

    (out / "classes.txt").write_text("telegraph\n", encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="weak-label telegraphs from a session")
    parser.add_argument("-s", "--session", required=True)
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--no-copy", action="store_true", help="write labels only")
    args = parser.parse_args()

    stats = label_session(
        args.session,
        args.out,
        min_confidence=args.min_confidence,
        copy_frames=not args.no_copy,
    )
    print(
        f"{stats['frames']} frames, {stats['boxes']} boxes, "
        f"{stats['empty']} negatives -> {args.out}"
    )
    if stats["boxes"] == 0:
        print(
            "\nNo detections. Either the session contains no telegraphs, or the hue "
            "ranges in TelegraphSegmenter need adjusting for your graphics settings."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
