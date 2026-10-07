"""Tracking the target window: where it is, and whether it has focus.

Two things depend on this. Perception needs the client rect to project HUD regions.
Safety needs foreground state, because dispatching keystrokes into whatever window the
human just alt-tabbed to is the most obvious way this project could do real damage.

Windows-only in the real implementation; every entry point degrades to a stub elsewhere
so the rest of the codebase (and the whole test suite) runs on any platform.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

from ..geometry import Rect

IS_WINDOWS = sys.platform == "win32"


@dataclass(frozen=True, slots=True)
class WindowInfo:
    handle: int
    title: str
    client_rect: Rect  # client area, in screen coordinates
    is_foreground: bool


class WindowTracker:
    """Finds a window by title substring and reports its rect and focus state.

    Polled once per capture rather than cached: the window can be moved or resized at any
    time, and a stale rect silently misreads every probe rather than failing loudly.
    """

    def __init__(self, title_contains: str) -> None:
        self.title_contains = title_contains.lower()
        self._handle: int | None = None

    @property
    def available(self) -> bool:
        return IS_WINDOWS

    def find(self) -> WindowInfo | None:
        if not IS_WINDOWS:
            return None
        handle = self._resolve_handle()
        if handle is None:
            return None
        rect = _client_rect_in_screen(handle)
        if rect is None:
            self._handle = None
            return None
        return WindowInfo(
            handle=handle,
            title=_window_title(handle),
            client_rect=rect,
            is_foreground=_foreground_handle() == handle,
        )

    def is_foreground(self) -> bool:
        """Cheap focus check for the safety gate — avoids a full window enumeration."""
        if not IS_WINDOWS:
            return False
        if self._handle is None:
            info = self.find()
            return bool(info and info.is_foreground)
        return _foreground_handle() == self._handle

    def _resolve_handle(self) -> int | None:
        # Re-validate the cached handle before trusting it: a closed-and-reopened client
        # can hand the same numeric handle to a different window.
        if self._handle is not None and _is_window(self._handle):
            if self.title_contains in _window_title(self._handle).lower():
                return self._handle
        self._handle = _find_window_by_title(self.title_contains)
        return self._handle


# -- Win32 plumbing ---------------------------------------------------------------

if IS_WINDOWS:  # pragma: no cover - exercised only on Windows
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _ENUM_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _is_window(handle: int) -> bool:
        return bool(_user32.IsWindow(wintypes.HWND(handle)))

    def _foreground_handle() -> int:
        return int(_user32.GetForegroundWindow() or 0)

    def _window_title(handle: int) -> str:
        length = _user32.GetWindowTextLengthW(wintypes.HWND(handle))
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(wintypes.HWND(handle), buf, length + 1)
        return buf.value

    def _find_window_by_title(needle: str) -> int | None:
        found: list[int] = []

        def callback(hwnd, _lparam):
            if not _user32.IsWindowVisible(hwnd):
                return True
            if needle in _window_title(hwnd).lower():
                found.append(int(hwnd))
                return False  # stop at the first match
            return True

        _user32.EnumWindows(_ENUM_PROC(callback), 0)
        return found[0] if found else None

    def _client_rect_in_screen(handle: int) -> Rect | None:
        """Client area in screen coordinates.

        GetClientRect gives a rect whose origin is always (0,0), so the top-left has to
        be mapped through ClientToScreen separately. Using the *window* rect here instead
        would include the title bar and border and shift every HUD region.
        """
        hwnd = wintypes.HWND(handle)
        rect = wintypes.RECT()
        if not _user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        origin = wintypes.POINT(0, 0)
        if not _user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return None  # minimised
        return Rect(x=origin.x, y=origin.y, w=w, h=h)

else:

    def _is_window(handle: int) -> bool:
        return False

    def _foreground_handle() -> int:
        return 0

    def _window_title(handle: int) -> str:
        return ""

    def _find_window_by_title(needle: str) -> int | None:
        return None

    def _client_rect_in_screen(handle: int) -> Rect | None:
        return None
