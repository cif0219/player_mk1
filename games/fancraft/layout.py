"""Where the FantCraft HUD is — loaded from the game's own manifest.

FantCraft's playtest bridge measures its HUD off the live DOM and publishes a
manifest: every region as fractions of the page viewport, the bar palette as CSS
hex, and the ability on each hotbar slot. `Layout.from_manifest` consumes that,
so this layout cannot drift out of date the way a hand-calibrated one does — a
HUD redesign over there shows up here on the next session start.

Coordinate caveat, stated once: manifest fractions are of the *page viewport*,
our `RelRect`s are of the *window client area*. Run the browser fullscreen
(F11) or in app/kiosk mode and the two coincide exactly, which is the supported
configuration. In a decorated window the browser chrome (tab strip, URL bar)
sits between the two spaces, every region lands too high, and nothing here can
tell — so fullscreen is a setup requirement, not a suggestion.

The defaults below are estimates for a 1920x1080 fullscreen client with the
default HUD, good enough to smoke-test before the first manifest lands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from player.geometry import RelRect

RGB = tuple[int, int, int]

# The client's stylesheet palette (client/src/styles/main.css :root vars).
# Overridden by the manifest's measured colors when one is loaded.
HP_GREEN: RGB = (46, 204, 113)   # --clr-hp-high #2ecc71
HP_RED: RGB = (192, 57, 43)      # --clr-hp-low  #c0392b
MP_BLUE: RGB = (52, 152, 219)    # --clr-mp      #3498db


def parse_hex(color: str, fallback: RGB) -> RGB:
    """'#2ecc71' -> (46, 204, 113). Anything unparseable keeps the fallback."""
    s = color.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return fallback
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return fallback


@dataclass(slots=True)
class HotbarSlot:
    """One hotbar slot as the manifest describes it: geometry plus contents."""

    key: str
    region: RelRect
    ability_id: str | None = None
    ability_name: str | None = None


@dataclass(slots=True)
class Layout:
    """Full HUD geometry for one FantCraft session."""

    # Top-left player frame (client/index.html #player-bars).
    player_hp: RelRect = field(default_factory=lambda: RelRect(0.0085, 0.0470, 0.1150, 0.0148))
    player_mp: RelRect = field(default_factory=lambda: RelRect(0.0085, 0.0665, 0.1150, 0.0148))
    # Target frame sits below the topbar; hidden entirely when nothing is targeted.
    target_hp: RelRect = field(default_factory=lambda: RelRect(0.0085, 0.1120, 0.0990, 0.0148))
    # Bottom-centre hotbar; slots filled in from the manifest.
    hotbar_slots: list[HotbarSlot] = field(default_factory=list)

    hp_color: RGB = HP_GREEN
    hp_low_color: RGB = HP_RED
    mp_color: RGB = MP_BLUE

    # Where the manifest came from, for the startup log.
    source: str = "defaults"

    @classmethod
    def from_manifest(cls, path: str | Path) -> "Layout":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        regions = data.get("regions", {})
        colors = data.get("colors", {})

        # Browser chrome drawn inside the OS client area (Edge's app-mode
        # title bar), published by the bridge in CSS px. The manifest's
        # fractions are of the page viewport; ours are of the client area,
        # so shift and scale them by the chrome the bridge measured.
        viewport = data.get("viewport", {})
        chrome = data.get("chrome", {})
        inner_w = float(viewport.get("w", 0) or 0)
        inner_h = float(viewport.get("h", 0) or 0)
        # outerWidth - innerWidth is the two invisible resize borders, which
        # Win32 keeps OUTSIDE the client rect; outerHeight - innerHeight is
        # the title bar (inside the client rect) plus the top border. So the
        # in-client chrome is the title bar alone: top minus one border.
        border = float(chrome.get("left", 0) or 0) / 2
        top = max(0.0, float(chrome.get("top", 0) or 0) - border)
        left = 0.0
        client_w = inner_w + left
        client_h = inner_h + top

        def to_client(r: RelRect) -> RelRect:
            if inner_w <= 0 or inner_h <= 0 or (top <= 0 and left <= 0):
                return r
            return RelRect(
                x=(left + r.x * inner_w) / client_w,
                y=(top + r.y * inner_h) / client_h,
                w=r.w * inner_w / client_w,
                h=r.h * inner_h / client_h,
            )

        def rel(name: str, default: RelRect) -> RelRect:
            r = regions.get(name)
            if not r:
                return default
            return to_client(RelRect(x=float(r["x"]), y=float(r["y"]), w=float(r["w"]), h=float(r["h"])))

        layout = cls(
            player_hp=rel("playerHpBar", cls().player_hp),
            player_mp=rel("playerMpBar", cls().player_mp),
            target_hp=rel("targetHpBar", cls().target_hp),
            hp_color=parse_hex(colors.get("hpHigh", ""), HP_GREEN),
            hp_low_color=parse_hex(colors.get("hpLow", ""), HP_RED),
            mp_color=parse_hex(colors.get("mp", ""), MP_BLUE),
            source=str(path),
        )
        for slot in data.get("hotbarSlots", []):
            r = slot.get("region")
            if not r:
                continue
            layout.hotbar_slots.append(
                HotbarSlot(
                    key=str(slot.get("key", "")),
                    region=to_client(RelRect(x=float(r["x"]), y=float(r["y"]), w=float(r["w"]), h=float(r["h"]))),
                    ability_id=slot.get("abilityId"),
                    ability_name=slot.get("abilityName"),
                )
            )
        return layout

    @classmethod
    def load_or_default(cls, path: str | Path | None) -> "Layout":
        if path and Path(path).exists():
            return cls.from_manifest(path)
        return cls()

    def slot_for_ability(self, ability_id: str) -> HotbarSlot | None:
        return next((s for s in self.hotbar_slots if s.ability_id == ability_id), None)


DEFAULT_LAYOUT = Layout()
