"""Session recording and playback.

This is not a debugging nicety, it is the development loop. Without deterministic
offline replay every experiment needs the game running, in the right place, in the right
state — which is how a project like this ends up with an architecture and no gameplay.
"""

from .recorder import Recorder, RecorderConfig
from .session import Session, TraceEvent

__all__ = ["Recorder", "RecorderConfig", "Session", "TraceEvent"]
