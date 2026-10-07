"""Where the Genshin Impact HUD is.

Every region is a `RelRect` — fractions of the client area — same discipline as the
FFXIV layout. Genshin's HUD is *not* user-arrangeable, which makes these defaults far
more trustworthy than FFXIV's: at any 16:9 resolution with default UI they should land
close. They are still estimates measured off screenshots, so run
`python -m player calibrate --game genshin` and check the JSON before trusting a live
session.

One Genshin-specific wrinkle documented here because probes depend on it: the active
character's HP bar changes colour with its fill (green when healthy, shading to orange
and red as it drops). `HP_GREEN` matches the healthy state; when the bar has gone red the
green probe reads 0.0 at reduced confidence, which downstream policy should treat as
"low or unreadable" — both are reasons to back off, so the failure direction is safe.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from player.geometry import RelRect

# HP_GREEN measured off a live 3440x1440 capture (both the active bar and the party
# strip read within a few counts of it). The others remain estimates. Tolerances in the
# probes are generous because the world behind the semi-transparent HUD bleeds through.
HP_GREEN = (148, 212, 34)
BOSS_HP_WHITE = (235, 230, 215)
STAMINA_YELLOW = (235, 205, 90)


@dataclass(slots=True)
class PartyStripLayout:
    """The party roster down the right edge: up to four portraits with HP bars beneath.

    A grid — origin plus vertical pitch — for the same reason the FFXIV hotbar is one:
    recalibration is four numbers, not sixteen rectangles.
    """

    origin_x: float = 0.9060
    origin_y: float = 0.2410
    slot_w: float = 0.0620
    slot_h: float = 0.0560
    pitch_y: float = 0.0870
    hp_offset_y: float = 0.0590
    hp_h: float = 0.0050
    slots: int = 4

    def portrait_region(self, index: int) -> RelRect:
        """Zero-based party slot to its portrait region."""
        return RelRect(
            x=self.origin_x,
            y=self.origin_y + index * self.pitch_y,
            w=self.slot_w,
            h=self.slot_h,
        )

    def hp_region(self, index: int) -> RelRect:
        """The small HP bar under a portrait."""
        return RelRect(
            x=self.origin_x,
            y=self.origin_y + index * self.pitch_y + self.hp_offset_y,
            w=self.slot_w,
            h=self.hp_h,
        )


@dataclass(slots=True)
class Layout:
    """Full HUD geometry for one installation, default 16:9 UI."""

    # The active character's HP bar, bottom centre.
    player_hp: RelRect = field(default_factory=lambda: RelRect(0.3960, 0.9390, 0.2080, 0.0090))
    # Elemental skill (E) icon, bottom right. The cooldown sweep darkens it.
    skill_icon: RelRect = field(default_factory=lambda: RelRect(0.8560, 0.8420, 0.0430, 0.0760))
    # Elemental burst (Q) icon, bottom right corner, larger than the skill. Dark while
    # energy is charging, fully lit when ready.
    burst_icon: RelRect = field(default_factory=lambda: RelRect(0.9130, 0.8180, 0.0600, 0.1060))
    # Boss/elite HP bar, top centre. Present only when something with a boss bar is
    # engaged, which is the closest thing Genshin's HUD has to an in-combat indicator.
    boss_hp: RelRect = field(default_factory=lambda: RelRect(0.3350, 0.0530, 0.3300, 0.0130))
    party: PartyStripLayout = field(default_factory=PartyStripLayout)
    # The 3D viewport, HUD chrome excluded, for any future detector work.
    field_region: RelRect = field(default_factory=lambda: RelRect(0.05, 0.10, 0.90, 0.65))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Layout":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> "Layout":
        kwargs = {}
        for name, value in data.items():
            if name == "party":
                kwargs["party"] = PartyStripLayout(**value)
            elif isinstance(value, dict):
                kwargs[name] = RelRect(**value)
        return cls(**kwargs)

    @classmethod
    def load_or_default(cls, path: str | Path | None) -> "Layout":
        if path and Path(path).exists():
            return cls.load(path)
        return cls()


DEFAULT_LAYOUT = Layout()
