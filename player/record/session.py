"""On-disk session format, and reading it back.

Layout:

    sessions/<name>/
        meta.json          run metadata: wall clock, geometry, profile, versions
        frames/000042.jpg  captured frames, zero-padded by frame index
        trace.jsonl        one JSON object per line, ordered by `t`

All timestamps in `trace.jsonl` are on the same monotonic clock as the frames, expressed
as seconds since the session's `t0`. That is what makes "telegraph appeared at frame N,
key dispatched at t+83ms" answerable after the fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(slots=True)
class TraceEvent:
    """One thing that happened, at a known offset from session start."""

    t: float  # seconds since session t0
    kind: str  # frame | state | plan | dispatch | directive | guard | note
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"t": round(self.t, 6), "kind": self.kind, "data": self.data})

    @classmethod
    def from_json(cls, line: str) -> "TraceEvent":
        obj = json.loads(line)
        return cls(t=obj["t"], kind=obj["kind"], data=obj.get("data", {}))


class Session:
    """Read access to a recorded session."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.meta_path = self.root / "meta.json"
        self.trace_path = self.root / "trace.jsonl"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"no meta.json under {self.root}")
        self.meta: dict[str, Any] = json.loads(self.meta_path.read_text(encoding="utf-8"))

    @property
    def client_rect(self) -> tuple[int, int, int, int]:
        r = self.meta.get("client_rect", [0, 0, 1920, 1080])
        return tuple(r)  # type: ignore[return-value]

    def frame_paths(self) -> list[Path]:
        if not self.frames_dir.exists():
            return []
        return sorted(self.frames_dir.glob("*.*"))

    def load_frame(self, path: Path) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield `(index, t, image)` in capture order.

        `t` comes from the trace rather than the filename, so a session recorded with
        frame skipping still replays at the right relative timing.
        """
        times = self.frame_times()
        for path in self.frame_paths():
            index = int(path.stem)
            yield index, times.get(index, 0.0), self.load_frame(path)

    def frame_times(self) -> dict[int, float]:
        return {
            int(ev.data["index"]): ev.t
            for ev in self.trace()
            if ev.kind == "frame" and "index" in ev.data
        }

    def trace(self, kind: str | None = None) -> Iterator[TraceEvent]:
        if not self.trace_path.exists():
            return
        with self.trace_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ev = TraceEvent.from_json(line)
                if kind is None or ev.kind == kind:
                    yield ev

    def states(self) -> Iterator[TraceEvent]:
        """Recorded `WorldState` summaries — the input to policy regression tests.

        These replay without any frames at all, which is why policy tests run in
        milliseconds while perception tests need the images.
        """
        return self.trace("state")

    def summary(self) -> str:
        counts: dict[str, int] = {}
        last_t = 0.0
        for ev in self.trace():
            counts[ev.kind] = counts.get(ev.kind, 0) + 1
            last_t = max(last_t, ev.t)
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return f"{self.root.name}: {last_t:.1f}s, {parts}"


def write_meta(root: Path, meta: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
