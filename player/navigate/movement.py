"""Walking to a place on the arena floor.

A proportional controller closed every tick, not an open-loop "press W for 800ms". The
distinction matters because the reference frame moves: the camera can rotate mid-walk, and
in FFXIV that changes what `W` means. Open-loop movement under a rotating camera curves
away and arrives somewhere else entirely.

So the loop is: read current position and camera yaw, compute the error vector in arena
space, rotate it into camera space, and reconcile the four movement keys to match. Because
yaw is re-read every tick, camera rotation is *compensated for* rather than merely
tolerated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..clock import now
from ..world.arena import ArenaPoint, CameraPose


@dataclass(slots=True)
class MovementIntent:
    """A request to be somewhere by some time."""

    target: ArenaPoint
    tolerance_m: float = 1.0
    deadline_at: float | None = None
    reason: str = ""
    # Strafing keeps the character facing forward while moving sideways, which preserves
    # caster uptime and rear positionals. Turning to face the target is faster in a
    # straight line but throws away both.
    allow_strafe: bool = True

    def expired(self, at: float) -> bool:
        return self.deadline_at is not None and at > self.deadline_at

    def time_left_s(self, at: float) -> float:
        return float("inf") if self.deadline_at is None else self.deadline_at - at


@dataclass(slots=True)
class MovementController:
    """Turns a target position into held movement keys.

    The output is a *set of keys that should currently be down*, not a sequence of
    presses. Whoever dispatches it is responsible for reconciling against what is actually
    held — pressing a key already down, or failing to release one, both strand the
    character mid-run.
    """

    keys: dict[str, str] = field(
        default_factory=lambda: {
            "forward": "w",
            "back": "s",
            "left": "a",
            "right": "d",
        }
    )
    # Below this the character is close enough and further correction just jitters.
    deadband_m: float = 0.35
    # A diagonal is two keys held together. Requiring a component to be meaningful before
    # its key goes down stops the controller shimmying between W and WA every frame.
    axis_threshold: float = 0.30

    intent: MovementIntent | None = None
    arrived: bool = False
    abandoned_reason: str = ""
    last_error_m: float = 0.0
    ticks: int = 0

    # -- intent ------------------------------------------------------------------

    def move_to(self, intent: MovementIntent) -> None:
        self.intent = intent
        self.arrived = False
        self.abandoned_reason = ""

    def stop(self, reason: str = "") -> None:
        self.intent = None
        self.abandoned_reason = reason

    @property
    def active(self) -> bool:
        return self.intent is not None and not self.arrived

    # -- control -----------------------------------------------------------------

    def desired_keys(
        self,
        player: ArenaPoint | None,
        pose: CameraPose,
        at: float | None = None,
    ) -> set[str]:
        """Keys that should be held right now. Empty means stand still."""
        at = now() if at is None else at
        self.ticks += 1

        intent = self.intent
        if intent is None:
            return set()

        # Refusing to move without a trusted position is the whole point of the
        # confidence plumbing: walking on a guessed position is worse than not walking.
        if player is None or not pose.known:
            self.abandoned_reason = "position or camera yaw unknown"
            return set()

        if intent.expired(at):
            self.stop("deadline passed")
            return set()

        offset = intent.target - player
        distance = offset.magnitude
        self.last_error_m = distance

        if distance <= max(intent.tolerance_m, self.deadband_m):
            self.arrived = True
            return set()

        # Arena space to camera space: +Y is where W takes you, +X is where D takes you.
        local = pose.to_camera_relative(offset).normalised()

        held: set[str] = set()
        if local.y > self.axis_threshold:
            held.add(self.keys["forward"])
        elif local.y < -self.axis_threshold:
            held.add(self.keys["back"])

        if intent.allow_strafe:
            if local.x > self.axis_threshold:
                held.add(self.keys["right"])
            elif local.x < -self.axis_threshold:
                held.add(self.keys["left"])
        elif not held:
            # No strafing and no forward/back component means the target is directly
            # beside us; the camera has to come round before we can walk there.
            held.add(self.keys["forward"])

        return held

    # -- reporting ---------------------------------------------------------------

    def eta_s(self, speed_m_s: float = 6.0) -> float:
        """Rough arrival estimate. 6 m/s is roughly FFXIV's uncontrolled run speed."""
        return self.last_error_m / max(speed_m_s, 1e-6)

    def will_miss_deadline(self, at: float, speed_m_s: float = 6.0) -> bool:
        """Is the deadline already unreachable?

        Worth knowing early. A mechanic that cannot be reached in time is better abandoned
        than half-run — arriving late usually means arriving *inside* the thing, having
        also given up whatever position you started from.
        """
        if self.intent is None or self.intent.deadline_at is None:
            return False
        return self.eta_s(speed_m_s) > self.intent.time_left_s(at)

    def status(self) -> str:
        if self.intent is None:
            return "move: idle" + (f" ({self.abandoned_reason})" if self.abandoned_reason else "")
        if self.arrived:
            return f"move: arrived ({self.intent.reason})"
        return (
            f"move: -> ({self.intent.target.x:+.1f},{self.intent.target.y:+.1f}) "
            f"{self.last_error_m:.1f}m ({self.intent.reason})"
        )
