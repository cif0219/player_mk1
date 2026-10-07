"""Live screen capture, scoped to the tracked window's client area."""

from __future__ import annotations

import numpy as np

from ..clock import now
from ..geometry import Rect
from .source import Frame
from .window import WindowTracker


class ScreenSource:
    """Grabs the tracked window's client area via `mss`.

    Capturing the client rect rather than the whole monitor means the frame's coordinate
    origin is the client origin, so frame space and client space differ only by the
    downscale factor. That removes an entire class of "which corner is this relative to"
    bug from every probe.

    `downscale` trades perception cost for precision. Probes read fixed regions and cope
    with 0.5 fine; the detector wants whatever its input size is anyway. Keep it at 1.0
    until a measured budget says otherwise.
    """

    def __init__(
        self,
        tracker: WindowTracker,
        downscale: float = 1.0,
        max_fps: int = 60,
    ) -> None:
        if not 0.0 < downscale <= 1.0:
            raise ValueError(f"downscale must be in (0, 1], got {downscale}")
        self.tracker = tracker
        self.downscale = downscale
        self.min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._sct = None
        self._index = 0
        self._last_grab = 0.0
        self.last_client_rect: Rect | None = None

    def open(self) -> None:
        import mss  # imported lazily so the package imports on machines with no display

        self._sct = mss.mss()

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def grab(self) -> Frame | None:
        if self._sct is None:
            raise RuntimeError("ScreenSource.grab() before open()")

        # Pace ourselves rather than spinning: capture is the one stage that will happily
        # consume a whole core producing frames nothing reads.
        elapsed = now() - self._last_grab
        if self.min_interval and elapsed < self.min_interval:
            return None

        info = self.tracker.find()
        if info is None:
            return None
        rect = info.client_rect
        self.last_client_rect = rect

        shot = self._sct.grab(
            {"left": rect.x, "top": rect.y, "width": rect.w, "height": rect.h}
        )
        captured_at = now()
        self._last_grab = captured_at

        # mss hands back BGRA; slice to RGB rather than round-tripping through PIL.
        bgra = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        image = np.ascontiguousarray(bgra[:, :, 2::-1])

        if self.downscale != 1.0:
            image = _downscale(image, self.downscale)

        self._index += 1
        return Frame(
            image=image,
            captured_at=captured_at,
            index=self._index,
            client_rect=rect,
        )


def _downscale(image: np.ndarray, factor: float) -> np.ndarray:
    """Nearest-neighbour downscale by strided slicing.

    Deliberately not an interpolating resize: probes read solid-colour HUD regions where
    averaging across a border blends the border into the reading, and nearest-neighbour
    is both faster and more faithful for that. The detector does its own resizing.
    """
    step = max(1, round(1.0 / factor))
    if step == 1:
        return image
    return np.ascontiguousarray(image[::step, ::step])
