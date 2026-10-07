"""Where the FFXIV HUD is.

Every region is a `RelRect` — fractions of the client area — so a calibration done at one
resolution resolves correctly at another with the same in-game HUD scale. Storing
absolute pixels would break on every resolution change and on every window resize.

**The defaults below are starting points, not measurements.** FFXIV's HUD is fully
user-arrangeable, so these are certain to be wrong for any specific installation. Run
`python -m player calibrate --game ffxiv` and commit the result; the defaults exist so the
pipeline runs end-to-end before calibration, not so it reads anything correctly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from player.geometry import RelRect

# FFXIV HUD colours, sampled from the default UI theme. Tolerances in the probes are
# generous because the game applies bloom and the party-list theme shifts these slightly.
HP_GREEN = (126, 202, 108)
MP_PURPLE = (172, 118, 208)
CAST_ORANGE = (222, 175, 92)
TARGET_HP_YELLOW = (206, 186, 108)


@dataclass(slots=True)
class HotbarLayout:
    """A hotbar as an origin plus a slot grid.

    Generating per-slot regions from a grid rather than calibrating twelve rectangles by
    hand means recalibration is three numbers instead of forty-eight.
    """

    origin_x: float = 0.352
    origin_y: float = 0.905
    slot_w: float = 0.0238
    slot_h: float = 0.0424
    gap_x: float = 0.0014
    slots: int = 12

    def slot_region(self, index: int) -> RelRect:
        """Zero-based slot index to its region."""
        return RelRect(
            x=self.origin_x + index * (self.slot_w + self.gap_x),
            y=self.origin_y,
            w=self.slot_w,
            h=self.slot_h,
        )

    def regions(self) -> list[RelRect]:
        return [self.slot_region(i) for i in range(self.slots)]


@dataclass(slots=True)
class Layout:
    """Full HUD geometry for one installation."""

    player_hp: RelRect = field(default_factory=lambda: RelRect(0.7365, 0.8620, 0.0930, 0.0075))
    player_mp: RelRect = field(default_factory=lambda: RelRect(0.7365, 0.8720, 0.0930, 0.0055))
    cast_bar: RelRect = field(default_factory=lambda: RelRect(0.4400, 0.7280, 0.1200, 0.0090))
    combat_indicator: RelRect = field(
        default_factory=lambda: RelRect(0.4790, 0.0180, 0.0180, 0.0230)
    )
    target_frame: RelRect = field(default_factory=lambda: RelRect(0.4210, 0.1120, 0.1580, 0.0560))
    target_hp: RelRect = field(default_factory=lambda: RelRect(0.4260, 0.1420, 0.1480, 0.0075))
    target_cast_name: RelRect = field(
        default_factory=lambda: RelRect(0.4260, 0.1620, 0.1480, 0.0210)
    )
    status_bar: RelRect = field(default_factory=lambda: RelRect(0.7280, 0.7960, 0.2560, 0.0540))
    # The minimap is the highest-value region on the HUD for anything positional: player
    # position, camera yaw, party dots and arena bounds all come out of one read. It is
    # square, and the player sits at its exact centre.
    minimap: RelRect = field(default_factory=lambda: RelRect(0.8730, 0.0290, 0.1030, 0.1830))
    # The boss cast bar, split into the name text and the fill bar. They are read
    # differently — the name is classified against known casts, the fill is a bar probe —
    # so they are separate regions rather than one.
    boss_cast_text: RelRect = field(default_factory=lambda: RelRect(0.4260, 0.1620, 0.1480, 0.0200))
    boss_cast_bar: RelRect = field(default_factory=lambda: RelRect(0.4260, 0.1840, 0.1480, 0.0070))
    # The player's own screen position. FFXIV keeps the character near screen centre with
    # a slight downward bias from the default camera pitch; the ground-AoE reflex tests
    # telegraph overlap against this point.
    player_anchor: RelRect = field(default_factory=lambda: RelRect(0.4940, 0.5300, 0.0120, 0.0200))
    hotbar: HotbarLayout = field(default_factory=HotbarLayout)
    # Ground telegraphs only ever appear in the 3D viewport. Restricting the segmenter to
    # it keeps orange HUD chrome out of the detections.
    field_region: RelRect = field(default_factory=lambda: RelRect(0.05, 0.10, 0.90, 0.62))

    def gcd_slot(self, index: int = 0) -> RelRect:
        return self.hotbar.slot_region(index)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Layout":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Layout":
        kwargs = {}
        for name, value in data.items():
            if name == "hotbar":
                kwargs["hotbar"] = HotbarLayout(**value)
            elif isinstance(value, dict):
                kwargs[name] = RelRect(**value)
        return cls(**kwargs)

    @classmethod
    def load_or_default(cls, path: str | Path | None) -> "Layout":
        if path and Path(path).exists():
            return cls.load(path)
        return cls()


DEFAULT_LAYOUT = Layout()
