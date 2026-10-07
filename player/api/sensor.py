"""`SnapshotSensor` — fields from a gateway snapshot instead of pixels.

A game supplies one function: snapshot in, `{field: (value, confidence)}`
out. The sensor writes those into `WorldState` exactly as a probe sensor
would, so policy cannot tell — and must not care — which path produced them.

Confidence is still meaningful here. A value the gateway computed from an
authoritative message (HP from `entity_health`) is 1.0; one it dead-reckoned
between corrections (our own position) is a little less; a value derived
from a message that may be seconds old (a mob's last known position while it
is out of sync range) should be lower still. Guards read it.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..capture.source import Frame
from ..geometry import Geometry
from ..state import FieldValue, WorldState

Reading = tuple[FieldValue, float]
Extractor = Callable[[Mapping[str, Any]], Mapping[str, Reading]]


class SnapshotSensor:
    """Runs a game's extractor over the frame's snapshot."""

    def __init__(self, name: str, provides: tuple[str, ...], extract: Extractor, cadence: int = 1) -> None:
        self.name = name
        self.provides = provides
        self.extract = extract
        self.cadence = max(1, cadence)
        self.errors = 0

    def observe(self, frame: Frame, geo: Geometry, state: WorldState) -> None:
        snapshot = getattr(frame, "snapshot", None)
        if not snapshot:
            for name in self.provides:
                if not name.endswith("*"):
                    state.set(name, None, confidence=0.0, source=f"api:{self.name}")
            return
        try:
            readings = self.extract(snapshot)
        except Exception:
            self.errors += 1
            for name in self.provides:
                if not name.endswith("*"):
                    state.set(name, None, confidence=0.0, source=f"api:{self.name}")
            return
        for name, (value, confidence) in readings.items():
            state.set(name, value, confidence=confidence, source=f"api:{self.name}")


__all__ = ["Extractor", "Reading", "SnapshotSensor"]
