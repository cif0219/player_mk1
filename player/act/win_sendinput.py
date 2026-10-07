"""Windows `SendInput` backend, using hardware scancodes.

`KEYEVENTF_SCANCODE` is the point. Without it Windows synthesises a virtual-key event,
which many games ignore entirely because they read raw scancodes. With it the event is
indistinguishable at the input layer from one the keyboard produced, which is what makes
the game respond at all.

Mouse movement uses absolute normalised coordinates (`MOUSEEVENTF_ABSOLUTE`), scaled
against the virtual desktop rather than the primary monitor, so a multi-monitor setup
with the game on the secondary display lands where intended instead of on the wrong
screen.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from ..geometry import ScreenPoint
from .keymap import ScanCode, mouse_button, parse_combo, resolve_key

IS_WINDOWS = sys.platform == "win32"

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

_BUTTONS = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}


if IS_WINDOWS:  # pragma: no cover - exercised only on Windows

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]

    class _INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
    _user32.SendInput.restype = wintypes.UINT


class SendInputBackend:
    """Real input dispatch.

    Held keys are tracked so `release_all` can clean up on shutdown or when a guard trips.
    Leaving a key down after the process exits is the single worst failure this layer can
    produce — the game keeps holding it and there is nothing left to release it.
    """

    name = "sendinput"

    def __init__(self) -> None:
        if not IS_WINDOWS:
            raise RuntimeError(
                "SendInputBackend requires Windows. Use NullBackend (--dry-run) elsewhere."
            )
        self._held: set[str] = set()
        self._held_buttons: set[str] = set()

    # -- keyboard ----------------------------------------------------------------

    def key_down(self, key: str) -> None:
        mods, base = parse_combo(key)
        for mod in mods:
            self._send_key(resolve_key(mod), down=True)
            self._held.add(mod)
        # A mouse pseudo-key ("mouse1") routes to the button path, which tracks its own
        # held state — `release_all` then lifts it as a button, not as a scancode.
        button = mouse_button(base)
        if button is not None:
            self.mouse_button(button, down=True)
            return
        self._send_key(resolve_key(base), down=True)
        self._held.add(base)

    def key_up(self, key: str) -> None:
        mods, base = parse_combo(key)
        button = mouse_button(base)
        if button is not None:
            self.mouse_button(button, down=False)
        else:
            self._send_key(resolve_key(base), down=False)
            self._held.discard(base)
        # Release modifiers in reverse order, mirroring how a human lets go.
        for mod in reversed(mods):
            self._send_key(resolve_key(mod), down=False)
            self._held.discard(mod)

    def _send_key(self, sc: ScanCode, down: bool) -> None:
        flags = KEYEVENTF_SCANCODE
        if sc.extended:
            flags |= KEYEVENTF_EXTENDEDKEY
        if not down:
            flags |= KEYEVENTF_KEYUP
        inp = _INPUT(type=INPUT_KEYBOARD)
        inp.ki = _KEYBDINPUT(wVk=0, wScan=sc.code, dwFlags=flags, time=0, dwExtraInfo=None)
        sent = _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        if sent != 1:
            raise OSError(f"SendInput failed: {ctypes.get_last_error()}")

    def release_all(self) -> None:
        """Release every key and button we believe is held.

        Mouse buttons matter as much as keys here: a camera drag interrupted by the kill
        switch leaves the right button down, which in FFXIV locks the camera to the mouse
        and makes the desktop nearly unusable until it is released.
        """
        for key in sorted(self._held):
            try:
                self._send_key(resolve_key(key), down=False)
            except (KeyError, OSError):
                pass
        self._held.clear()
        for button in sorted(self._held_buttons):
            try:
                self._send_mouse_flag(_BUTTONS.get(button, _BUTTONS["right"])[1])
            except OSError:
                pass
        self._held_buttons.clear()

    @property
    def held(self) -> set[str]:
        return set(self._held)

    # -- mouse -------------------------------------------------------------------

    def mouse_move(self, point: ScreenPoint) -> None:
        dx, dy = _to_absolute(point)
        inp = _INPUT(type=INPUT_MOUSE)
        inp.mi = _MOUSEINPUT(
            dx=dx,
            dy=dy,
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
            time=0,
            dwExtraInfo=None,
        )
        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def click(self, button: str, point: ScreenPoint | None = None) -> None:
        if point is not None:
            self.mouse_move(point)
        down_flag, up_flag = _BUTTONS.get(button, _BUTTONS["left"])
        for flag in (down_flag, up_flag):
            self._send_mouse_flag(flag)

    def mouse_button(self, button: str, down: bool) -> None:
        """Hold or release a button. Camera drag needs the button down across many frames."""
        down_flag, up_flag = _BUTTONS.get(button, _BUTTONS["right"])
        self._send_mouse_flag(down_flag if down else up_flag)
        if down:
            self._held_buttons.add(button)
        else:
            self._held_buttons.discard(button)

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        """Relative motion — no MOUSEEVENTF_ABSOLUTE.

        This is what camera orbit consumes. Warping the cursor to an absolute position
        instead produces one jump and then nothing, because the camera responds to motion
        rather than to where the pointer ended up.
        """
        inp = _INPUT(type=INPUT_MOUSE)
        inp.mi = _MOUSEINPUT(
            dx=dx, dy=dy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=None
        )
        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def _send_mouse_flag(self, flag: int) -> None:
        inp = _INPUT(type=INPUT_MOUSE)
        inp.mi = _MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=flag, time=0, dwExtraInfo=None)
        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def close(self) -> None:
        self.release_all()


def _to_absolute(point: ScreenPoint) -> tuple[int, int]:
    """Map screen pixels to the 0–65535 absolute range over the whole virtual desktop.

    Normalising against the primary monitor instead is the classic multi-monitor bug: the
    cursor lands on the wrong display and every click misses.
    """
    origin_x = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    origin_y = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN) or 1
    height = _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN) or 1
    dx = int((point.x - origin_x) * 65535 / width)
    dy = int((point.y - origin_y) * 65535 / height)
    return max(0, min(65535, dx)), max(0, min(65535, dy))
