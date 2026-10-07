"""Turning plans into input events, on a schedule.

The central object is `InputTimeline`: a deadline-ordered priority queue, not a queue
plus a sleeping worker. That distinction is the whole reason this layer exists as its own
module — see docs/ARCHITECTURE.md.
"""

from .backend import InputBackend, NullBackend, RecordingBackend
from .keymap import KEY_ALIASES, resolve_key
from .timeline import (
    Click,
    InputTimeline,
    KeyDown,
    KeyUp,
    MouseMove,
    Plan,
    Press,
    Priority,
    ScheduledEvent,
    Step,
)

__all__ = [
    "Click",
    "InputBackend",
    "InputTimeline",
    "KEY_ALIASES",
    "KeyDown",
    "KeyUp",
    "MouseMove",
    "NullBackend",
    "Plan",
    "Press",
    "Priority",
    "RecordingBackend",
    "ScheduledEvent",
    "Step",
    "resolve_key",
]
