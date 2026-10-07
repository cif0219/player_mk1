"""The kill switch.

A low-level keyboard hook (`WH_KEYBOARD_LL`), not a focus-scoped listener. The
distinction is the whole feature: when the game has focus it swallows input, so a
focus-scoped listener never sees the key — which is precisely the situation you need the
kill switch in.

Three properties it must keep:

* **Latching.** Once tripped it stays tripped until explicitly resumed. A momentary trip
  that clears itself is not a kill switch.
* **Independent of the decision loop.** The hook runs on its own thread with its own
  message pump, so a hung perception thread cannot prevent it from stopping dispatch.
* **Checked before every dispatch**, including events already sitting on the timeline.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes

from ..clock import now
from ..act.keymap import resolve_key

IS_WINDOWS = sys.platform == "win32"

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104


class KillSwitch:
    """Global hotkey that latches the player into a stopped state.

    Also tracks the timestamp of the most recent *human* key press, which
    `TakeoverGuard` needs. Doing it here rather than in a second hook avoids installing
    two low-level hooks, which is a real cost — every keystroke on the machine passes
    through each of them.
    """

    def __init__(self, key: str = "f12", pause_key: str | None = None) -> None:
        self.key = key
        self.pause_key = pause_key
        self._scan = _try_scancode(key)
        self._pause_scan = _try_scancode(pause_key) if pause_key else None

        self._tripped = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._hook = None
        self._running = False
        self._lock = threading.Lock()

        self.last_human_key_at: float = 0.0
        self.trip_count = 0
        # Keys the dispatcher is about to send. The hook consults this to tell our own
        # synthetic input apart from a human's — see TakeoverGuard.
        self._expected: dict[int, float] = {}

    # -- state -------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return IS_WINDOWS and self._scan is not None

    @property
    def tripped(self) -> bool:
        return self._tripped.is_set()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def trip(self, reason: str = "manual") -> None:
        with self._lock:
            if not self._tripped.is_set():
                self.trip_count += 1
            self._tripped.set()
        self.last_trip_reason = reason

    def resume(self) -> None:
        """Explicit un-latch. Nothing else clears a trip."""
        self._tripped.clear()
        self._paused.clear()

    def toggle_pause(self) -> None:
        if self._paused.is_set():
            self._paused.clear()
        else:
            self._paused.set()

    def expect(self, scancode: int, window_s: float = 0.25) -> None:
        """Tell the hook that a synthetic key is about to be sent.

        Without this, our own dispatched keys look exactly like human input and the
        takeover guard blocks the player the instant it starts working.
        """
        self._expected[scancode] = now() + window_s

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> bool:
        """Install the hook. Returns False if unavailable on this platform."""
        if not IS_WINDOWS:
            return False
        if self._scan is None:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._pump, name="killswitch", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False

    def _pump(self) -> None:  # pragma: no cover - Windows-only message loop
        user32 = ctypes.WinDLL("user32", use_last_error=True)

        class KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ("vkCode", wintypes.DWORD),
                ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
            ]

        HOOKPROC = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_int, wintypes.WPARAM, ctypes.POINTER(KBDLLHOOKSTRUCT)
        )

        def handler(code, wparam, lparam):
            if code >= 0 and wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                scan = int(lparam.contents.scanCode)
                at = now()
                expiry = self._expected.get(scan)
                if expiry is not None and at <= expiry:
                    # Ours. Consume the expectation so a repeat is treated as human.
                    self._expected.pop(scan, None)
                else:
                    self.last_human_key_at = at
                    if scan == self._scan:
                        self.trip("hotkey")
                    elif self._pause_scan is not None and scan == self._pause_scan:
                        self.toggle_pause()
            return user32.CallNextHookEx(None, code, wparam, lparam)

        callback = HOOKPROC(handler)
        self._hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, callback, None, 0)
        if not self._hook:
            self._running = False
            return

        msg = wintypes.MSG()
        while self._running:
            # Bounded wait so `stop()` is observed promptly rather than at the next
            # keystroke, which might never come.
            if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            else:
                user32.MsgWaitForMultipleObjects(0, None, False, 50, 0x04FF)

        user32.UnhookWindowsHookEx(self._hook)
        self._hook = None

    def status(self) -> str:
        if not self.available:
            return "killswitch:unavailable"
        if self.tripped:
            return f"killswitch:TRIPPED({self.key})"
        return f"killswitch:armed({self.key})"


def _try_scancode(key: str | None) -> int | None:
    if not key:
        return None
    try:
        return resolve_key(key).code
    except KeyError:
        return None
