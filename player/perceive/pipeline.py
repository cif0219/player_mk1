"""Frame in, `WorldState` out. The only place perception is assembled."""

from __future__ import annotations

from ..capture.source import Frame
from ..clock import LatencyBudget, now
from ..geometry import Geometry, Rect
from ..state import WorldState
from .sensor import SensorBundle


class PerceptionPipeline:
    """Runs a `SensorBundle` over each frame and produces one `WorldState`.

    Also owns `Geometry`, rebuilding it whenever the window rect or frame size changes.
    Doing that here rather than in each sensor means a window move reprojects every probe
    exactly once, and no sensor can be left holding a stale projection.
    """

    def __init__(
        self,
        bundle: SensorBundle,
        budget: LatencyBudget | None = None,
        carry_forward: bool = True,
    ) -> None:
        self.bundle = bundle
        self.budget = budget or LatencyBudget()
        # Fields from low-cadence sensors persist between their runs, aging naturally via
        # `Field.updated_at`. Without this a 4-tick sensor's fields would vanish for
        # three of every four frames.
        self.carry_forward = carry_forward
        self.geometry: Geometry | None = None
        self.tick = 0
        self._previous: WorldState | None = None
        self._geo_signature: tuple | None = None
        self.geometry_rebuilds = 0

    def _ensure_geometry(self, frame: Frame) -> Geometry:
        signature = (frame.client_rect, frame.size)
        if signature != self._geo_signature or self.geometry is None:
            self.geometry = Geometry(
                client_rect_screen=frame.client_rect,
                frame_size=frame.size,
            )
            self._geo_signature = signature
            self.geometry_rebuilds += 1
        return self.geometry

    def process(self, frame: Frame) -> WorldState:
        self.tick += 1
        geo = self._ensure_geometry(frame)

        state = WorldState(tick=self.tick, captured_at=frame.captured_at)
        if self.carry_forward and self._previous is not None:
            # Copy prior fields first so this tick's sensors overwrite what they refresh
            # and leave the rest to age.
            state.fields.update(self._previous.fields)
            state.entities = list(self._previous.entities)

        with self.budget.measure("perceive"):
            self.bundle.observe(self.tick, frame, geo, state)

        state.perceived_at = now()
        self.budget.record("capture_to_perceived", (state.perceived_at - frame.captured_at) * 1000)
        self._previous = state
        return state

    def reset(self) -> None:
        self._previous = None
        self.geometry = None
        self._geo_signature = None
        self.tick = 0

    def describe(self) -> str:
        rect: Rect | None = self.geometry.client_rect_screen if self.geometry else None
        return (
            f"pipeline: sensors={len(self.bundle.sensors)} "
            f"fields={len(self.bundle.provides)} "
            f"client={rect} rebuilds={self.geometry_rebuilds}"
        )
