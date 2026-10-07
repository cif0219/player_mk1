"""Key names to hardware scancodes.

Scancodes, not virtual key codes, and this is the detail that decides whether the whole
project works. DirectInput-era and many DirectX games read raw scancodes from the
keyboard buffer and ignore synthetic virtual-key events entirely. Sending `VK_2` to such
a game does nothing at all, silently, which looks exactly like a broken decision layer
and wastes days.

Values are Set 1 make codes. Extended keys (arrows, right-side modifiers, numpad enter)
carry the 0xE0 prefix, tracked by the `extended` flag rather than baked into the code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScanCode:
    code: int
    extended: bool = False


# Set 1 make codes.
SCANCODES: dict[str, ScanCode] = {
    "escape": ScanCode(0x01),
    "1": ScanCode(0x02),
    "2": ScanCode(0x03),
    "3": ScanCode(0x04),
    "4": ScanCode(0x05),
    "5": ScanCode(0x06),
    "6": ScanCode(0x07),
    "7": ScanCode(0x08),
    "8": ScanCode(0x09),
    "9": ScanCode(0x0A),
    "0": ScanCode(0x0B),
    "minus": ScanCode(0x0C),
    "equals": ScanCode(0x0D),
    "backspace": ScanCode(0x0E),
    "tab": ScanCode(0x0F),
    "q": ScanCode(0x10),
    "w": ScanCode(0x11),
    "e": ScanCode(0x12),
    "r": ScanCode(0x13),
    "t": ScanCode(0x14),
    "y": ScanCode(0x15),
    "u": ScanCode(0x16),
    "i": ScanCode(0x17),
    "o": ScanCode(0x18),
    "p": ScanCode(0x19),
    "enter": ScanCode(0x1C),
    "ctrl": ScanCode(0x1D),
    "a": ScanCode(0x1E),
    "s": ScanCode(0x1F),
    "d": ScanCode(0x20),
    "f": ScanCode(0x21),
    "g": ScanCode(0x22),
    "h": ScanCode(0x23),
    "j": ScanCode(0x24),
    "k": ScanCode(0x25),
    "l": ScanCode(0x26),
    "shift": ScanCode(0x2A),
    "z": ScanCode(0x2C),
    "x": ScanCode(0x2D),
    "c": ScanCode(0x2E),
    "v": ScanCode(0x2F),
    "b": ScanCode(0x30),
    "n": ScanCode(0x31),
    "m": ScanCode(0x32),
    "rshift": ScanCode(0x36),
    "alt": ScanCode(0x38),
    "space": ScanCode(0x39),
    "f1": ScanCode(0x3B),
    "f2": ScanCode(0x3C),
    "f3": ScanCode(0x3D),
    "f4": ScanCode(0x3E),
    "f5": ScanCode(0x3F),
    "f6": ScanCode(0x40),
    "f7": ScanCode(0x41),
    "f8": ScanCode(0x42),
    "f9": ScanCode(0x43),
    "f10": ScanCode(0x44),
    "f11": ScanCode(0x57),
    "f12": ScanCode(0x58),
    # Extended: 0xE0-prefixed.
    "rctrl": ScanCode(0x1D, extended=True),
    "ralt": ScanCode(0x38, extended=True),
    "up": ScanCode(0x48, extended=True),
    "left": ScanCode(0x4B, extended=True),
    "right": ScanCode(0x4D, extended=True),
    "down": ScanCode(0x50, extended=True),
    "insert": ScanCode(0x52, extended=True),
    "delete": ScanCode(0x53, extended=True),
    "home": ScanCode(0x47, extended=True),
    "end": ScanCode(0x4F, extended=True),
    "pageup": ScanCode(0x49, extended=True),
    "pagedown": ScanCode(0x51, extended=True),
}

KEY_ALIASES: dict[str, str] = {
    "esc": "escape",
    "return": "enter",
    "control": "ctrl",
    "lctrl": "ctrl",
    "lshift": "shift",
    "lalt": "alt",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "del": "delete",
    "ins": "insert",
    "spacebar": "space",
    "lmb": "mouse1",
    "rmb": "mouse2",
    "mmb": "mouse3",
    "left_click": "mouse1",
    "right_click": "mouse2",
}

# Mouse buttons usable anywhere a key is. Genshin's normal and charged attacks are the
# left mouse button and cannot be rebound to the keyboard, so an ability's `key` has to
# be able to name a button — and a held `Press("mouse1", 800)` is exactly what a charged
# attack is. Pseudo-key name -> backend button name.
MOUSE_BUTTONS: dict[str, str] = {
    "mouse1": "left",
    "mouse2": "right",
    "mouse3": "middle",
}


def mouse_button(key: str) -> str | None:
    """The backend button name if `key` is a mouse pseudo-key, else None."""
    return MOUSE_BUTTONS.get(normalise(key))


def normalise(key: str) -> str:
    k = key.strip().lower()
    return KEY_ALIASES.get(k, k)


def resolve_key(key: str) -> ScanCode:
    """Look up a scancode, raising on unknown keys.

    Deliberately strict. A silently ignored keybind is a rotation that mostly works and
    is missing one ability, which is far harder to debug than a startup error — and
    keybinds come from a config file, so they are exactly the thing that gets typo'd.
    """
    name = normalise(key)
    sc = SCANCODES.get(name)
    if sc is None:
        raise KeyError(f"unknown key {key!r} (normalised to {name!r})")
    return sc


def parse_combo(combo: str) -> tuple[list[str], str]:
    """Split "ctrl+shift+3" into modifiers and the base key.

    FFXIV hotbars are commonly bound as ctrl/shift/alt + number, so combos are the normal
    case rather than an edge case.
    """
    parts = [normalise(p) for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty key combo: {combo!r}")
    *mods, base = parts
    for name in mods:
        if name in MOUSE_BUTTONS:
            raise ValueError(f"mouse button {name!r} cannot be a modifier in {combo!r}")
        resolve_key(name)  # validate now, at config load, not at first press
    if base not in MOUSE_BUTTONS:
        resolve_key(base)
    return mods, base


def validate_keymap(keys: dict[str, str]) -> list[str]:
    """Check every configured binding. Returns human-readable problems."""
    problems = []
    for action, combo in keys.items():
        try:
            parse_combo(combo)
        except (KeyError, ValueError) as exc:
            problems.append(f"{action}: {exc}")
    return problems
