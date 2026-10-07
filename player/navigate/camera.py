"""Aiming the camera.

The camera is an actuator, not a view setting. In a 3D game you only perceive what is in
the frustum, so a tower behind you is *unobserved*, not absent, and the correct response
is to turn and look. That makes perception steerable — and steering it costs something,
because while you are looking behind you, you are not looking at the boss.

The second thing the camera is: **the movement reference frame.** `W` means "away from the
camera". Rotating the camera therefore rewrites what every movement key does, mid-walk.
Two consequences are designed for here:

* The movement controller re-reads yaw every tick, so rotation is compensated rather than
  merely survived.
* Residual error is proportional to *camera angular velocity times control latency*, so
  the slew rate is capped while a movement target is active. Turning slower while walking
  is a real cost, and it is much cheaper than arriving in the wrong place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ..clock import now
from ..world.arena import ArenaPoint, CameraPose, bearing_deg, normalise_deg, shortest_turn_deg


class CameraMode(Enum):
    """How camera rotation reaches the game."""

    KEYBIND = "keybind"  # bound turn-left / turn-right keys. Predictable rate, preferred.
    MOUSE_DRAG = "mouse_drag"  # hold right mouse and move. Universal, needs a px/deg fit.


@dataclass(slots=True)
class CameraIntent:
    """A bearing to face, and how urgently."""

    yaw_deg: float
    tolerance_deg: float = 8.0
    deadline_at: float | None = None
    reason: str = ""
    priority: int = 50

    def expired(self, at: float) -> bool:
        return self.deadline_at is not None and at > self.deadline_at


@dataclass(slots=True)
class CameraCommand:
    """What the dispatcher should do this tick to turn the camera."""

    held_keys: set[str] = field(default_factory=set)
    mouse_dx: int = 0
    mouse_dy: int = 0
    drag_button: str | None = None  # held for the duration of a mouse-drag turn

    @property
    def idle(self) -> bool:
        return not self.held_keys and self.mouse_dx == 0 and self.mouse_dy == 0


@dataclass(slots=True)
class CameraController:
    """Slews the camera toward a target bearing.

    Deliberately not a snap-to-angle. A camera that teleports between bearings makes every
    frame captured mid-turn useless for detection — motion during exposure is exactly what
    a detector handles worst — and it makes the movement controller's yaw reading stale in
    the one moment it matters most.
    """

    mode: CameraMode = CameraMode.KEYBIND
    keys: dict[str, str] = field(
        default_factory=lambda: {"turn_left": "left", "turn_right": "right"}
    )
    # Degrees per second the game turns while a turn key is held. Measured once by the
    # calibration tool rather than assumed; the default is a plausible starting point.
    keybind_rate_deg_s: float = 90.0
    # Mouse pixels per degree in drag mode. Depends on the user's camera sensitivity, so
    # this is also measured rather than assumed.
    mouse_px_per_deg: float = 4.0
    # Cap while walking. This is the camera/movement coupling made explicit: the faster
    # the camera turns, the more stale the yaw the movement controller is using.
    slew_cap_deg_s_moving: float = 60.0
    slew_cap_deg_s_idle: float = 180.0
    max_mouse_step_px: int = 40

    intent: CameraIntent | None = None
    settled: bool = False
    last_error_deg: float = 0.0
    _last_tick_at: float = 0.0

    # -- intent ------------------------------------------------------------------

    def look_at_bearing(self, intent: CameraIntent) -> None:
        self.intent = intent
        self.settled = False

    def look_at_point(
        self,
        player: ArenaPoint,
        target: ArenaPoint,
        *,
        tolerance_deg: float = 8.0,
        reason: str = "",
        deadline_at: float | None = None,
        priority: int = 50,
    ) -> None:
        self.look_at_bearing(
            CameraIntent(
                yaw_deg=bearing_deg(player, target),
                tolerance_deg=tolerance_deg,
                reason=reason,
                deadline_at=deadline_at,
                priority=priority,
            )
        )

    def release(self, reason: str = "") -> None:
        self.intent = None
        self.settled = True

    @property
    def active(self) -> bool:
        return self.intent is not None and not self.settled

    # -- control -----------------------------------------------------------------

    def command(
        self,
        pose: CameraPose,
        *,
        moving: bool = False,
        at: float | None = None,
    ) -> CameraCommand:
        at = now() if at is None else at
        dt = min(0.1, max(0.0, at - self._last_tick_at)) if self._last_tick_at else 0.016
        self._last_tick_at = at

        intent = self.intent
        if intent is None or not pose.known:
            return CameraCommand()

        if intent.expired(at):
            self.release("deadline passed")
            return CameraCommand()

        error = shortest_turn_deg(pose.yaw_deg, intent.yaw_deg)
        self.last_error_deg = error

        if abs(error) <= intent.tolerance_deg:
            self.settled = True
            return CameraCommand()

        cap = self.slew_cap_deg_s_moving if moving else self.slew_cap_deg_s_idle

        if self.mode is CameraMode.KEYBIND:
            # Keybind turning has one speed; the only control is whether the key is down.
            # Under the moving cap, hold it only a duty-cycle fraction of the time.
            if moving and self.keybind_rate_deg_s > cap:
                duty = cap / self.keybind_rate_deg_s
                if (at * self.keybind_rate_deg_s / 90.0) % 1.0 > duty:
                    return CameraCommand()
            key = self.keys["turn_right"] if error > 0 else self.keys["turn_left"]
            return CameraCommand(held_keys={key})

        # Mouse drag: a bounded step toward the target, never the whole error at once.
        step_deg = max(-cap * dt, min(cap * dt, error))
        # Do not overshoot into the tolerance band and then oscillate around it.
        step_deg = max(-abs(error), min(abs(error), step_deg))
        dx = int(round(step_deg * self.mouse_px_per_deg))
        dx = max(-self.max_mouse_step_px, min(self.max_mouse_step_px, dx))
        if dx == 0:
            dx = 1 if error > 0 else -1
        return CameraCommand(mouse_dx=dx, drag_button="right")

    # -- calibration -------------------------------------------------------------

    def observe_turn(self, before: CameraPose, after: CameraPose, dt: float, input_amount: float) -> None:
        """Learn the turn rate from what actually happened.

        Camera sensitivity is a user setting, so assuming a constant is assuming something
        about someone else's config. Watching how far the camera moved for a given input is
        both self-correcting and free — the observations are already being made.
        """
        if dt <= 0 or input_amount == 0 or not (before.known and after.known):
            return
        turned = abs(shortest_turn_deg(before.yaw_deg, after.yaw_deg))
        if turned < 0.5:
            return
        if self.mode is CameraMode.KEYBIND:
            observed = turned / dt
            self.keybind_rate_deg_s = _blend(self.keybind_rate_deg_s, observed)
        else:
            observed = abs(input_amount) / turned
            self.mouse_px_per_deg = _blend(self.mouse_px_per_deg, observed)

    def status(self) -> str:
        if self.intent is None:
            return "cam: free"
        state = "settled" if self.settled else f"{self.last_error_deg:+.0f}deg"
        return f"cam: -> {self.intent.yaw_deg:.0f}deg {state} ({self.intent.reason})"


def _blend(current: float, observed: float, weight: float = 0.2) -> float:
    """Exponential moving average, clamped to something physically plausible."""
    if not math.isfinite(observed) or observed <= 0:
        return current
    blended = current * (1 - weight) + observed * weight
    return max(current * 0.5, min(current * 2.0, blended))
