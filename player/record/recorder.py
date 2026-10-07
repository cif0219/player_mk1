"""Writing a session to disk while the player runs.

Recording must not perturb what it records. Frame encoding is done on a background
thread behind a bounded queue, and the queue drops rather than blocks when full — a
recorder that stalls the capture thread has changed the thing it was measuring.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..clock import now
from ..geometry import Rect
from ..state import WorldState
from .session import TraceEvent, write_meta


@dataclass(slots=True)
class RecorderConfig:
    enabled: bool = False
    root: Path = field(default_factory=lambda: Path("sessions"))
    name: str = ""
    frame_every_n: int = 2  # 30Hz of frames from a 60Hz capture is plenty for review
    jpeg_quality: int = 80
    max_queue: int = 64
    record_frames: bool = True


class Recorder:
    """Session writer. A no-op when disabled, so callers need no conditionals."""

    def __init__(self, config: RecorderConfig) -> None:
        self.config = config
        self.enabled = config.enabled
        self.t0 = now()
        self.root = config.root / (config.name or time.strftime("%Y%m%d-%H%M%S"))
        self._trace_fh = None
        self._queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(
            maxsize=config.max_queue
        )
        self._writer: threading.Thread | None = None
        self._lock = threading.Lock()
        self.frames_written = 0
        self.frames_dropped = 0
        self._frame_counter = 0

    # -- lifecycle ---------------------------------------------------------------

    def start(self, meta: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        (self.root / "frames").mkdir(parents=True, exist_ok=True)
        write_meta(
            self.root,
            {
                "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
                # Unix seconds with sub-ms precision: trace `t` offsets are
                # monotonic-relative, and joining a session against an external
                # wall-clocked log (e.g. FantCraft's playtest oracle) needs a
                # wall anchor better than the 1s resolution of started_wall.
                "started_unix": time.time(),
                "t0_monotonic": self.t0,
                **(meta or {}),
            },
        )
        self._trace_fh = (self.root / "trace.jsonl").open("w", encoding="utf-8")
        self._writer = threading.Thread(
            target=self._write_loop, name="recorder", daemon=True
        )
        self._writer.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._queue.put(None)
        if self._writer is not None:
            self._writer.join(timeout=5.0)
        with self._lock:
            if self._trace_fh is not None:
                self._trace_fh.flush()
                self._trace_fh.close()
                self._trace_fh = None

    # -- recording ---------------------------------------------------------------

    def frame(self, index: int, image: np.ndarray, captured_at: float, rect: Rect) -> None:
        if not self.enabled:
            return
        self._frame_counter += 1
        self.event("frame", captured_at, {"index": index, "rect": [rect.x, rect.y, rect.w, rect.h]})
        if not self.config.record_frames:
            return
        if self._frame_counter % max(1, self.config.frame_every_n) != 0:
            return
        try:
            # Copy: the caller's array is a live capture buffer that may be reused.
            self._queue.put_nowait((index, image.copy()))
        except queue.Full:
            self.frames_dropped += 1

    def state(self, state: WorldState) -> None:
        if not self.enabled:
            return
        self.event(
            "state",
            state.captured_at,
            {"tick": state.tick, "fields": state.to_summary()},
        )

    def plan(self, name: str, priority: int, steps: int, at: float | None = None) -> None:
        self.event("plan", at, {"name": name, "priority": priority, "steps": steps})

    def dispatch(self, key: str, action: str, at: float | None = None) -> None:
        self.event("dispatch", at, {"key": key, "action": action})

    def directive(self, kind: str, payload: dict[str, Any]) -> None:
        self.event("directive", None, {"kind": kind, **payload})

    def guard(self, name: str, blocking: bool) -> None:
        self.event("guard", None, {"name": name, "blocking": blocking})

    def note(self, text: str) -> None:
        self.event("note", None, {"text": text})

    def event(self, kind: str, at: float | None, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        ev = TraceEvent(t=(now() if at is None else at) - self.t0, kind=kind, data=data)
        line = ev.to_json()
        with self._lock:
            if self._trace_fh is not None:
                self._trace_fh.write(line + "\n")

    # -- background encode -------------------------------------------------------

    def _write_loop(self) -> None:
        from PIL import Image

        while True:
            item = self._queue.get()
            if item is None:
                break
            index, image = item
            path = self.root / "frames" / f"{index:06d}.jpg"
            try:
                Image.fromarray(image).save(path, quality=self.config.jpeg_quality)
                self.frames_written += 1
            except Exception:
                # A failed frame write must never take down a running session.
                self.frames_dropped += 1

    def status(self) -> str:
        if not self.enabled:
            return "rec:off"
        return f"rec:{self.frames_written}f drop:{self.frames_dropped}"


def load_meta(root: Path) -> dict[str, Any]:
    path = Path(root) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
