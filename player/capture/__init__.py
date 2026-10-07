"""Frame acquisition."""

from .replay import ReplaySource
from .source import CaptureSource, Frame, FrameSlot
from .window import WindowInfo, WindowTracker

__all__ = [
    "CaptureSource",
    "Frame",
    "FrameSlot",
    "ReplaySource",
    "WindowInfo",
    "WindowTracker",
]
