"""Replaying a recorded session as if it were a live capture.

`ReplaySource` satisfies `CaptureSource`, so everything above capture — sensors,
`WorldState` assembly, reflexes, the rotation planner — runs bit-identically against a
recording. That is the whole point: it turns "did my perception change break anything"
from a question you answer by launching a game into one you answer with `pytest`.
"""

from __future__ import annotations

from pathlib import Path

from ..clock import now
from ..geometry import Rect
from ..record.session import Session
from .source import Frame


class ReplaySource:
    """Feeds frames from a `Session`.

    Two modes:

    * `realtime=True` reproduces the original inter-frame timing, for watching a session
      play back at the speed it happened.
    * `realtime=False` (default) delivers frames as fast as they are requested, which is
      what regression tests want — a test that sleeps is a test nobody runs.
    """

    def __init__(
        self,
        session_root: str | Path,
        *,
        realtime: bool = False,
        loop: bool = False,
        speed: float = 1.0,
    ) -> None:
        self.session = Session(session_root)
        self.realtime = realtime
        self.loop = loop
        self.speed = max(0.01, speed)
        self._frames: list[tuple[int, float, Path]] = []
        self._cursor = 0
        self._started_at = 0.0
        self.exhausted = False

    def open(self) -> None:
        times = self.session.frame_times()
        self._frames = [
            (int(p.stem), times.get(int(p.stem), 0.0), p) for p in self.session.frame_paths()
        ]
        if not self._frames:
            raise ValueError(f"session {self.session.root} contains no frames")
        self._cursor = 0
        self._started_at = now()
        self.exhausted = False

    def close(self) -> None:
        self._frames = []

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def grab(self) -> Frame | None:
        if not self._frames:
            raise RuntimeError("ReplaySource.grab() before open()")

        if self._cursor >= len(self._frames):
            if not self.loop:
                self.exhausted = True
                return None
            self._cursor = 0
            self._started_at = now()

        index, t, path = self._frames[self._cursor]

        if self.realtime:
            # Hold the frame back until its original offset has elapsed. Returning None
            # rather than sleeping keeps the caller's loop responsive to shutdown.
            elapsed = (now() - self._started_at) * self.speed
            if elapsed < t:
                return None

        self._cursor += 1
        image = self.session.load_frame(path)
        h, w = image.shape[:2]
        rect = _rect_for(self.session, index, w, h)
        # `captured_at` is stamped now rather than replayed, so downstream staleness
        # checks behave the same as they would live. The original offset stays available
        # through the trace for latency forensics.
        return Frame(image=image, captured_at=now(), index=index, client_rect=rect)


def _rect_for(session: Session, index: int, w: int, h: int) -> Rect:
    """Recover the client rect that was live when this frame was captured."""
    for ev in session.trace("frame"):
        if ev.data.get("index") == index and "rect" in ev.data:
            x, y, rw, rh = ev.data["rect"]
            return Rect(x, y, rw, rh)
    return Rect(0, 0, w, h)
