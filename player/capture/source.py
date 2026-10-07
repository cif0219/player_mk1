"""The capture contract and the slot that carries frames between threads."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..clock import now
from ..geometry import Rect


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured image plus everything needed to interpret it.

    `client_rect` travels with the frame rather than being read from the window at
    perception time: the window can move between capture and perception, and a probe
    reprojected against the *new* rect would read the wrong pixels out of the *old*
    image.
    """

    image: np.ndarray  # (H, W, 3) uint8, RGB
    captured_at: float
    index: int
    client_rect: Rect

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    def crop(self, region: Rect) -> np.ndarray:
        """Frame-space slice. Returns a view, so probes must not write to it."""
        return self.image[region.y : region.bottom, region.x : region.right]


@runtime_checkable
class CaptureSource(Protocol):
    """Anything that can produce frames.

    `ReplaySource` implements this, which is what lets the entire pipeline above capture
    run identically against a recording and against a live game.
    """

    def open(self) -> None: ...

    def grab(self) -> Frame | None:
        """Return the next frame, or `None` if none is available right now."""
        ...

    def close(self) -> None: ...


class FrameSlot:
    """A single-slot mailbox holding only the newest frame.

    Deliberately not a ring buffer. A decision made on a frame two frames old is a worse
    decision, so there is never a reason to read anything but the latest; keeping a
    history would only create the opportunity to read the wrong one. Dropped frames are
    counted so the status line can show whether perception is keeping up with capture.
    """

    def __init__(self) -> None:
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._arrived = threading.Event()
        self.published = 0
        self.dropped = 0

    def publish(self, frame: Frame) -> None:
        with self._lock:
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self.published += 1
        self._arrived.set()

    def latest(self) -> Frame | None:
        """Peek without consuming. Used by the recorder and status line."""
        with self._lock:
            return self._frame

    def take(self, timeout: float = 0.1) -> Frame | None:
        """Block until a frame is available, then consume it.

        Consuming (rather than peeking) is what makes `dropped` meaningful: a frame that
        was replaced before anyone took it is a frame perception did not get to.
        """
        if not self._arrived.wait(timeout=timeout):
            return None
        with self._lock:
            frame = self._frame
            self._frame = None
            self._arrived.clear()
        return frame

    def wake(self) -> None:
        """Unblock a waiting `take` during shutdown."""
        self._arrived.set()


class NullSource:
    """A source that produces nothing. Used by tests that exercise the loop's plumbing."""

    def __init__(self) -> None:
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def grab(self) -> Frame | None:
        return None

    def close(self) -> None:
        self.opened = False


class StaticSource:
    """Emits one fixed image forever. Used for probe unit tests."""

    def __init__(self, image: np.ndarray, client_rect: Rect | None = None) -> None:
        self.image = image
        h, w = image.shape[:2]
        self.client_rect = client_rect or Rect(0, 0, w, h)
        self._index = 0

    def open(self) -> None:
        return None

    def grab(self) -> Frame | None:
        self._index += 1
        return Frame(
            image=self.image,
            captured_at=now(),
            index=self._index,
            client_rect=self.client_rect,
        )

    def close(self) -> None:
        return None
