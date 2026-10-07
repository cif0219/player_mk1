"""Deciding what to look at.

This is the active-perception scheduler, and it exists because in a 3D game **perception
competes with itself**. You want the boss in frame for its cast bar, the ground in frame
for telegraphs, and whatever is behind you in frame for the tower you have not noticed yet.
The camera can serve one of those at a time.

So the camera gets a scheduler with the same shape as any other contended resource:
prioritised requests, expiry, and a background task that fills the idle time. The
background task here is a **sweep** — the arena is divided into sectors, each remembers
when it was last in frame, and when nothing more urgent wants the camera it turns toward
whichever sector has gone unseen longest.

That sweep is what converts "I have not seen a tower" from a confident negative into an
honest "I have not looked there in nine seconds".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from ..clock import now
from ..world.arena import ArenaPoint, Observability, bearing_deg, normalise_deg


class GazePriority(IntEnum):
    """Higher wins. Equal priority keeps whatever is already committed."""

    IDLE = 0
    SWEEP = 20
    TARGET = 50
    THREAT = 90
    MECHANIC = 100


@dataclass(frozen=True, slots=True)
class GazeTarget:
    """Where to point the camera. Exactly one of `point` or `bearing_deg` is set."""

    point: ArenaPoint | None = None
    bearing_deg: float | None = None

    def resolve_bearing(self, player: ArenaPoint) -> float | None:
        if self.bearing_deg is not None:
            return normalise_deg(self.bearing_deg)
        if self.point is not None:
            return bearing_deg(player, self.point)
        return None


@dataclass(slots=True)
class GazeRequest:
    """A claim on the camera."""

    id: str
    target: GazeTarget
    priority: GazePriority = GazePriority.TARGET
    tolerance_deg: float = 10.0
    expires_at: float | None = None
    reason: str = ""
    # Hold the camera here after arriving, so a detector gets clean frames rather than
    # everything mid-slew.
    dwell_s: float = 0.0

    def expired(self, at: float) -> bool:
        return self.expires_at is not None and at > self.expires_at


@dataclass(slots=True)
class SectorMemory:
    """When each slice of the surrounding circle was last in frame.

    The whole point of tracking this: without it, "the detector found no towers" is
    indistinguishable from "the detector was pointed at a wall". With it, a policy can ask
    how stale its knowledge of a direction is and go look.
    """

    sectors: int = 12
    last_seen: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.last_seen:
            self.last_seen = [0.0] * self.sectors

    def sector_of(self, bearing: float) -> int:
        return int(normalise_deg(bearing) / (360.0 / self.sectors)) % self.sectors

    def bearing_of(self, sector: int) -> float:
        width = 360.0 / self.sectors
        return normalise_deg(sector * width + width / 2.0)

    def observe(self, observability: Observability, at: float) -> None:
        """Mark every sector currently in the view cone as freshly seen."""
        if not observability.pose.known:
            return
        half = observability.usable_fov_deg / 2.0
        yaw = observability.pose.yaw_deg
        for sector in range(self.sectors):
            centre = self.bearing_of(sector)
            delta = abs((centre - yaw + 180.0) % 360.0 - 180.0)
            if delta <= half:
                self.last_seen[sector] = at

    def stalest(self, at: float) -> tuple[int, float]:
        """The sector unseen longest, and for how long."""
        worst = min(range(self.sectors), key=lambda s: self.last_seen[s])
        return worst, at - self.last_seen[worst]

    def coverage_frac(self, at: float, within_s: float) -> float:
        """Share of the circle seen within the last `within_s` seconds."""
        fresh = sum(1 for t in self.last_seen if at - t <= within_s)
        return fresh / self.sectors

    def reset(self, at: float) -> None:
        self.last_seen = [at] * self.sectors


@dataclass(slots=True)
class GazePolicy:
    """Picks the winning gaze request, and generates sweeps when nothing else is asking."""

    memory: SectorMemory = field(default_factory=SectorMemory)
    sweep_after_s: float = 6.0
    sweep_dwell_s: float = 0.4
    # Below this the camera is left alone; a camera that is always turning is a camera
    # whose frames are always motion-blurred, which costs more than the coverage gains.
    min_switch_gain_deg: float = 15.0
    sweeps_issued: int = 0

    _requests: dict[str, GazeRequest] = field(default_factory=dict)
    _committed: str = ""

    # -- requests ----------------------------------------------------------------

    def request(self, request: GazeRequest) -> None:
        self._requests[request.id] = request

    def clear(self, request_id: str) -> None:
        self._requests.pop(request_id, None)
        if self._committed == request_id:
            self._committed = ""

    def look_at_boss(self, player: ArenaPoint, boss: ArenaPoint) -> None:
        """The default claim: the boss is where the cast bar and most telegraphs are."""
        self.request(
            GazeRequest(
                id="boss",
                target=GazeTarget(point=boss),
                priority=GazePriority.TARGET,
                reason="boss cast bar and telegraphs",
            )
        )

    def look_at_mechanic(
        self, target: ArenaPoint, deadline_at: float, reason: str = "mechanic"
    ) -> None:
        """A mechanic destination outranks everything. You cannot dodge what is off-screen."""
        self.request(
            GazeRequest(
                id="mechanic",
                target=GazeTarget(point=target),
                priority=GazePriority.MECHANIC,
                tolerance_deg=6.0,
                expires_at=deadline_at,
                reason=reason,
                dwell_s=0.3,
            )
        )

    # -- decision ----------------------------------------------------------------

    def decide(
        self,
        player: ArenaPoint | None,
        observability: Observability,
        at: float | None = None,
    ) -> GazeRequest | None:
        """The request that should own the camera right now."""
        at = now() if at is None else at
        if player is None:
            return None

        self.memory.observe(observability, at)

        for request_id in [rid for rid, r in self._requests.items() if r.expired(at)]:
            self.clear(request_id)

        candidates = list(self._requests.values())

        sweep = self._maybe_sweep(player, at)
        if sweep is not None:
            candidates.append(sweep)

        if not candidates:
            return None

        best = max(candidates, key=lambda r: (int(r.priority), r.id == self._committed))

        # Hysteresis: do not abandon what the camera is already doing for a marginal
        # improvement. Without this the camera oscillates between two similar bearings and
        # is never actually settled on either.
        current = self._requests.get(self._committed)
        if (
            current is not None
            and current is not best
            and int(current.priority) >= int(best.priority)
            and not current.expired(at)
        ):
            current_bearing = current.target.resolve_bearing(player)
            best_bearing = best.target.resolve_bearing(player)
            if current_bearing is not None and best_bearing is not None:
                gain = abs((best_bearing - current_bearing + 180.0) % 360.0 - 180.0)
                if gain < self.min_switch_gain_deg:
                    return current

        self._committed = best.id
        return best

    def _maybe_sweep(self, player: ArenaPoint, at: float) -> GazeRequest | None:
        sector, staleness = self.memory.stalest(at)
        if staleness < self.sweep_after_s:
            return None
        self.sweeps_issued += 1
        return GazeRequest(
            id="sweep",
            target=GazeTarget(bearing_deg=self.memory.bearing_of(sector)),
            priority=GazePriority.SWEEP,
            tolerance_deg=15.0,
            expires_at=at + 3.0,
            reason=f"sector unseen for {staleness:.0f}s",
            dwell_s=self.sweep_dwell_s,
        )

    # -- reporting ---------------------------------------------------------------

    def coverage(self, at: float | None = None, within_s: float = 6.0) -> float:
        return self.memory.coverage_frac(now() if at is None else at, within_s)

    def status(self, at: float | None = None) -> str:
        at = now() if at is None else at
        committed = self._requests.get(self._committed)
        what = committed.reason if committed else "free"
        return f"gaze: {what} cover={self.coverage(at):.0%} sweeps={self.sweeps_issued}"
