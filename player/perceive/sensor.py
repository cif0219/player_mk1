"""Sensors: the units that write fields into `WorldState`.

A sensor declares what it `provides` so the runtime can fail at startup when a policy
references a field nothing produces — rather than reading `None` sixty times a second and
behaving strangely for reasons nobody can see.

Cadence exists because not everything is worth reading every frame. Bars change
continuously; a buff-icon template match at 60Hz spends CPU to learn nothing. A sensor
with `cadence=4` runs on every fourth tick and its fields simply age between runs, which
is exactly what `Field.updated_at` is for.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..capture.source import Frame
from ..geometry import Geometry, Rect
from ..state import WorldState
from .probes import Probe


@runtime_checkable
class Sensor(Protocol):
    name: str
    provides: tuple[str, ...]
    cadence: int

    def observe(self, frame: Frame, geo: Geometry, state: WorldState) -> None: ...


class ProbeSensor:
    """Runs a set of probes and writes one field per probe.

    Region resolution is cached against the `Geometry` instance. Resolving a `RelRect` to
    a frame-space `Rect` involves two multiplications and a clamp — trivial once, wasteful
    sixty times a second across forty probes.
    """

    def __init__(
        self,
        name: str,
        probes: list[Probe],
        cadence: int = 1,
        transform: dict[str, callable] | None = None,
    ) -> None:
        self.name = name
        self.probes = probes
        self.cadence = max(1, cadence)
        self.provides = tuple(p.name for p in probes)
        # Optional post-read mapping, e.g. cooldown progress -> seconds remaining. Kept
        # out of the probe so one probe implementation serves every hotbar slot.
        self.transform = transform or {}
        self._geo_key: tuple | None = None
        self._regions: dict[str, Rect] = {}

    def _resolve(self, geo: Geometry) -> None:
        key = (geo.client_rect_screen, geo.frame_w, geo.frame_h)
        if key == self._geo_key:
            return
        self._regions = {p.name: geo.region_to_frame(p.region) for p in self.probes}
        self._geo_key = key

    def observe(self, frame: Frame, geo: Geometry, state: WorldState) -> None:
        self._resolve(geo)
        for probe in self.probes:
            region = self._regions[probe.name]
            if region.w <= 0 or region.h <= 0:
                # Region fell outside the frame — a window resize mid-flight, usually.
                state.set(probe.name, None, confidence=0.0, source=f"probe:{probe.name}")
                continue
            try:
                reading = probe.read(frame.crop(region))
            except Exception:
                # One bad probe must not blind every other sensor. Report zero confidence
                # and let the confidence guard decide whether that is fatal.
                state.set(probe.name, None, confidence=0.0, source=f"probe:{probe.name}")
                continue

            value = reading.value
            fn = self.transform.get(probe.name)
            if fn is not None and value is not None:
                value = fn(value)
            state.set(
                probe.name,
                value,
                confidence=reading.confidence,
                source=f"probe:{probe.name}",
            )


class DerivedSensor:
    """Computes fields from other fields rather than from pixels.

    Keeps arithmetic like "gcd_remaining_s = progress * recast" out of both the probes
    (which should only report what they see) and the policy (which should only read what
    it needs). Runs after the pixel sensors in bundle order.
    """

    def __init__(self, name: str, provides: tuple[str, ...], fn: callable, cadence: int = 1) -> None:
        self.name = name
        self.provides = provides
        self.fn = fn
        self.cadence = max(1, cadence)

    def observe(self, frame: Frame, geo: Geometry, state: WorldState) -> None:
        try:
            self.fn(state)
        except Exception:
            for field_name in self.provides:
                state.set(field_name, None, confidence=0.0, source=f"derived:{self.name}")


class SensorBundle:
    """An ordered set of sensors, run per tick according to their cadence.

    Order matters: `DerivedSensor` instances must come after the pixel sensors whose
    fields they consume. The bundle preserves declaration order rather than sorting, so
    the game profile controls it explicitly.
    """

    def __init__(self, sensors: list[Sensor] | None = None) -> None:
        self.sensors: list[Sensor] = list(sensors or [])

    def add(self, sensor: Sensor) -> "SensorBundle":
        self.sensors.append(sensor)
        return self

    @property
    def provides(self) -> set[str]:
        out: set[str] = set()
        for sensor in self.sensors:
            out.update(sensor.provides)
        return out

    def observe(self, tick: int, frame: Frame, geo: Geometry, state: WorldState) -> None:
        # Offset by one so every sensor runs on tick 1. Keying on `tick % cadence == 0`
        # instead means a cadence-4 sensor produces nothing for the first three frames,
        # and the player starts by acting on a world state with holes in it.
        for sensor in self.sensors:
            if (tick - 1) % sensor.cadence == 0:
                sensor.observe(frame, geo, state)

    def check_requirements(self, required: set[str]) -> list[str]:
        """Field names nothing in this bundle produces. Called once, at startup."""
        available = self.provides
        return sorted(
            name
            for name in required
            # Wildcard providers declare a prefix, e.g. "action.*", for per-slot fields.
            if name not in available
            and not any(a.endswith("*") and name.startswith(a[:-1]) for a in available)
        )
