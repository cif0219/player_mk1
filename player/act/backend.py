"""Input backends: the last hop before the OS.

Every backend implements the same three primitives. The dispatcher does not know which
one it is talking to, so `--dry-run` exercises the entire pipeline including the timeline
and the guard chain — the only difference is that nothing reaches the OS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..clock import now
from ..geometry import ScreenPoint


@runtime_checkable
class InputBackend(Protocol):
    name: str

    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def mouse_move(self, point: ScreenPoint) -> None: ...
    def click(self, button: str, point: ScreenPoint | None) -> None: ...
    def close(self) -> None: ...

    # Camera control needs two things a click-and-move backend does not provide: a button
    # held down across many frames, and *relative* motion (the camera responds to motion,
    # not to cursor position). Both are optional on the protocol so a minimal backend
    # still satisfies it; `navigate/channel.py` degrades when they are absent.
    def mouse_button(self, button: str, down: bool) -> None: ...
    def mouse_move_relative(self, dx: int, dy: int) -> None: ...


@dataclass(slots=True)
class SentEvent:
    at: float
    kind: str
    detail: str


class NullBackend:
    """Discards everything. The `--dry-run` backend.

    Keeps a bounded log so the CLI can show what *would* have been sent, which is how you
    validate a rotation before ever pointing it at a game.
    """

    name = "null"

    def __init__(self, log_limit: int = 512, on_event=None) -> None:
        self.events: list[SentEvent] = []
        self.log_limit = log_limit
        self.on_event = on_event

    def _record(self, kind: str, detail: str) -> None:
        ev = SentEvent(at=now(), kind=kind, detail=detail)
        self.events.append(ev)
        if len(self.events) > self.log_limit:
            self.events.pop(0)
        if self.on_event is not None:
            self.on_event(ev)

    def key_down(self, key: str) -> None:
        self._record("key_down", key)

    def key_up(self, key: str) -> None:
        self._record("key_up", key)

    def mouse_move(self, point: ScreenPoint) -> None:
        self._record("mouse_move", f"{point.x},{point.y}")

    def click(self, button: str, point: ScreenPoint | None) -> None:
        where = f" @{point.x},{point.y}" if point else ""
        self._record("click", f"{button}{where}")

    def mouse_button(self, button: str, down: bool) -> None:
        self._record("mouse_down" if down else "mouse_up", button)

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        self._record("mouse_rel", f"{dx},{dy}")

    def close(self) -> None:
        return None

    def keys_pressed(self) -> list[str]:
        return [e.detail for e in self.events if e.kind == "key_down"]

    def total_mouse_delta(self) -> tuple[int, int]:
        """Summed relative motion. The camera turn is the integral of these."""
        dx = dy = 0
        for event in self.events:
            if event.kind == "mouse_rel":
                a, b = event.detail.split(",")
                dx += int(a)
                dy += int(b)
        return dx, dy


@dataclass(slots=True)
class RecordingBackend:
    """Wraps another backend and records everything that passes through.

    Used by the session recorder so the trace contains what was actually dispatched, not
    what policy intended — the gap between those two is where most timing bugs live.
    """

    inner: InputBackend
    events: list[SentEvent] = field(default_factory=list)
    name: str = "recording"

    def __post_init__(self) -> None:
        self.name = f"recording:{self.inner.name}"

    def key_down(self, key: str) -> None:
        self.events.append(SentEvent(now(), "key_down", key))
        self.inner.key_down(key)

    def key_up(self, key: str) -> None:
        self.events.append(SentEvent(now(), "key_up", key))
        self.inner.key_up(key)

    def mouse_move(self, point: ScreenPoint) -> None:
        self.events.append(SentEvent(now(), "mouse_move", f"{point.x},{point.y}"))
        self.inner.mouse_move(point)

    def click(self, button: str, point: ScreenPoint | None) -> None:
        self.events.append(SentEvent(now(), "click", button))
        self.inner.click(button, point)

    def close(self) -> None:
        self.inner.close()
