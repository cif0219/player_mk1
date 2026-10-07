"""Reading the party list.

Healing is the reactive half of targeting: who needs a heal depends on how badly everyone
else played, and no guide can predict it. That makes it a policy over live state, and this
is the state it reads.

The party list is a good sensor target for the same reason the minimap is: fixed HUD
position, uniform rows, one bar each. It is a grid, so eight slots is three calibrated
numbers rather than eight calibrated rectangles.

The derived fields at the bottom are the point. Once `party.lowest_hp_frac` and
`party.count_below_50` exist, a triage rule is an ordinary `Condition` and a healer profile
is an ordinary priority list — no second planner, no special-case code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from player.geometry import RelRect
from player.perceive.probes import BarProbe, PresenceProbe
from player.perceive.sensor import DerivedSensor, ProbeSensor
from player.state import WorldState

# FFXIV party-list HP bars. Green at full, shifting toward yellow and red as HP drops —
# which is why the tolerance is generous: the probe has to keep matching a bar whose
# colour is itself the signal.
PARTY_HP_GREEN = (94, 174, 96)


@dataclass(slots=True)
class PartyListLayout:
    """The party list as an origin plus a row pitch."""

    origin_x: float = 0.0104
    origin_y: float = 0.4074
    row_height: float = 0.0389
    bar_w: float = 0.0729
    bar_h: float = 0.0074
    # The HP bar sits below the name within each row.
    bar_offset_y: float = 0.0241
    slots: int = 8

    def row_region(self, index: int) -> RelRect:
        return RelRect(
            x=self.origin_x,
            y=self.origin_y + index * self.row_height,
            w=self.bar_w,
            h=self.row_height * 0.9,
        )

    def hp_region(self, index: int) -> RelRect:
        return RelRect(
            x=self.origin_x,
            y=self.origin_y + index * self.row_height + self.bar_offset_y,
            w=self.bar_w,
            h=self.bar_h,
        )


def build_party_sensor(layout: PartyListLayout, cadence: int = 2) -> ProbeSensor:
    """Per-slot presence and HP.

    Presence first: an empty slot's HP bar reads as zero, which is indistinguishable from
    a dead ally unless something says the slot is occupied at all. Healing an empty slot
    does nothing; believing slot 5 is at 0% when nobody is in it makes the triage policy
    permanently panic.
    """
    probes = []
    for index in range(layout.slots):
        probes.append(PresenceProbe(f"party.{index}.present", layout.row_region(index), min_std=9.0))
        probes.append(
            BarProbe(f"party.{index}.hp_frac", layout.hp_region(index), PARTY_HP_GREEN, tolerance=70)
        )
    return ProbeSensor("party", probes, cadence=cadence)


def build_party_derived(slots: int = 8, thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)) -> DerivedSensor:
    """Summaries that make triage rules ordinary conditions.

    Without these every heal rule would need custom code to scan eight slots. With them
    the rule is `Condition.field("party.lowest_hp_frac", "<", 0.6)` and the whole healer
    profile reads like the DPS one.
    """
    provides = (
        "party.lowest_hp_frac",
        "party.lowest_slot",
        "party.present_count",
        "party.mean_hp_frac",
        *(f"party.count_below_{int(t * 100)}" for t in thresholds),
    )

    def compute(state: WorldState) -> None:
        lowest = 2.0
        lowest_slot = -1
        present = 0
        total = 0.0
        counts = {t: 0 for t in thresholds}

        for index in range(slots):
            present_field = state.field(f"party.{index}.present")
            if present_field is not None and present_field.value is False:
                continue
            hp_field = state.field(f"party.{index}.hp_frac")
            if hp_field is None or hp_field.value is None or not hp_field.trusted(0.5):
                # An unread frame is not a healthy one, but it is also not evidence of
                # damage. Skipping keeps an unreadable party from either panicking the
                # policy or masking someone genuinely low.
                continue
            hp = float(hp_field.value)
            present += 1
            total += hp
            if hp < lowest:
                lowest, lowest_slot = hp, index
            for threshold in thresholds:
                if hp < threshold:
                    counts[threshold] += 1

        if present == 0:
            for name in provides:
                state.set(name, None, confidence=0.0, source="derived:party")
            return

        state.set("party.lowest_hp_frac", round(lowest, 4), source="derived:party")
        state.set("party.lowest_slot", lowest_slot, source="derived:party")
        state.set("party.present_count", present, source="derived:party")
        state.set("party.mean_hp_frac", round(total / present, 4), source="derived:party")
        for threshold, count in counts.items():
            state.set(f"party.count_below_{int(threshold * 100)}", count, source="derived:party")

    return DerivedSensor("party_derived", provides, compute, cadence=2)


class PartyLocatorSensor:
    """Keeps the mouseover locator pointing at the right pixels.

    Mouseover healing needs a *screen* position for each party row, and screen positions
    move whenever the window does. Refreshing here — inside the sensor pass, which already
    holds the current `Geometry` — means a window drag cannot leave the healer hovering
    empty desktop and casting into nothing.
    """

    provides = ()
    cadence = 4  # the window does not move sixty times a second

    def __init__(self, layout: PartyListLayout, resolver, name: str = "party_locator") -> None:
        self.name = name
        self.layout = layout
        self.resolver = resolver

    def observe(self, frame, geo, state: WorldState) -> None:
        def locate(slot: int) -> tuple[int, int] | None:
            if not 0 <= slot < self.layout.slots:
                return None
            rect = geo.region_to_screen(self.layout.row_region(slot))
            return rect.center

        self.resolver.set_slot_locator(locate)
