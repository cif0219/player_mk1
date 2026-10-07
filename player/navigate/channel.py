"""The second dispatch channel: continuous movement and camera aim.

The discrete timeline in `act/` handles ability presses — things that happen at a moment.
This handles the things that have a *state*: which movement keys are down, where the
camera is pointing. Trying to express those as scheduled presses produces either a
stuttering character or a schedule that is rewritten every frame, and neither is the
timeline's job.

One object owns both because **they are coupled**. `W` means "away from the camera", so
camera rotation rewrites what every movement key does. Two mechanisms handle it:

* Movement re-reads camera yaw every tick, so rotation is compensated rather than endured.
* Camera slew is capped while movement is active, because the residual error is
  proportional to angular velocity times control latency.

Everything still passes through `SafetyGate`. A tripped guard releases the movement and
camera keys exactly like it releases everything else — a character left running into a
wall after the kill switch is pressed is the worst possible failure of this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..act.backend import InputBackend
from ..clock import now
from ..geometry import ScreenPoint
from ..safety.guards import SafetyGate
from ..world.arena import ArenaFrame, ArenaPoint, CameraPose, Observability
from .camera import CameraController, CameraIntent, CameraMode
from .gaze import GazePolicy
from .movement import MovementController


@dataclass(slots=True)
class NavigationState:
    """What the channel wants the input state to be this tick."""

    held_keys: set[str] = field(default_factory=set)
    mouse_dx: int = 0
    mouse_dy: int = 0
    drag_button: str | None = None
    blocked_reason: str = ""

    @property
    def idle(self) -> bool:
        return not self.held_keys and self.mouse_dx == 0 and self.mouse_dy == 0


class NavigationChannel:
    """Owns the movement keys and the camera, and reconciles them against the backend."""

    def __init__(
        self,
        movement: MovementController | None = None,
        camera: CameraController | None = None,
        gaze: GazePolicy | None = None,
        *,
        reserved_by_timeline: set[str] | None = None,
    ) -> None:
        self.movement = movement or MovementController()
        self.camera = camera or CameraController()
        self.gaze = gaze or GazePolicy()

        self._held: set[str] = set()
        self._dragging = False
        self._prev_pose: CameraPose | None = None
        self._prev_at: float = 0.0
        self._last_input_amount: float = 0.0

        self.dispatched = 0
        self.blocked = 0
        self._was_blocked = False

        conflicts = self.owned_keys() & (reserved_by_timeline or set())
        if conflicts:
            # A key claimed by both channels gets pressed by one and released by the
            # other. Catching it here is cheap; catching it in a raid is not.
            raise ValueError(
                f"navigation and timeline both claim {sorted(conflicts)} — rebind one of them"
            )

    def owned_keys(self) -> set[str]:
        return set(self.movement.keys.values()) | set(self.camera.keys.values())

    # -- decide ------------------------------------------------------------------

    def update(self, frame: ArenaFrame | None, at: float | None = None) -> NavigationState:
        """Compute the desired input state. Pure: dispatches nothing."""
        at = now() if at is None else at

        if frame is None or not frame.pose.known:
            # No trusted position or yaw. Standing still is the correct answer — moving on
            # a guessed heading is how you walk off the arena.
            self.movement.abandoned_reason = "no arena frame"
            return NavigationState(blocked_reason="arena frame unavailable")

        self._learn_camera_rate(frame.pose, at)

        # Gaze decides where the camera should point; the camera controller decides how to
        # get there. Keeping those separate is what lets the sweep policy be tested with
        # no camera model at all.
        decision = self.gaze.decide(frame.player, frame.observability, at)
        if decision is not None:
            bearing = decision.target.resolve_bearing(frame.player)
            if bearing is not None:
                self.camera.look_at_bearing(
                    CameraIntent(
                        yaw_deg=bearing,
                        tolerance_deg=decision.tolerance_deg,
                        deadline_at=decision.expires_at,
                        reason=decision.reason,
                        priority=int(decision.priority),
                    )
                )

        moving = self.movement.active
        move_keys = self.movement.desired_keys(frame.player, frame.pose, at)
        cam = self.camera.command(frame.pose, moving=moving, at=at)

        self._last_input_amount = float(cam.mouse_dx) if cam.mouse_dx else (
            1.0 if cam.held_keys else 0.0
        )

        return NavigationState(
            held_keys=move_keys | cam.held_keys,
            mouse_dx=cam.mouse_dx,
            mouse_dy=cam.mouse_dy,
            drag_button=cam.drag_button,
        )

    def _learn_camera_rate(self, pose: CameraPose, at: float) -> None:
        if self._prev_pose is not None and self._last_input_amount:
            self.camera.observe_turn(
                self._prev_pose, pose, at - self._prev_at, self._last_input_amount
            )
        self._prev_pose = pose
        self._prev_at = at

    # -- dispatch ----------------------------------------------------------------

    def pump(
        self,
        backend: InputBackend,
        gate: SafetyGate,
        frame: ArenaFrame | None,
        world_state=None,
        at: float | None = None,
    ) -> NavigationState:
        """Reconcile the backend's input state to what `update` wants."""
        at = now() if at is None else at

        if not gate.allowed(world_state, at):
            self.blocked += 1
            if not self._was_blocked:
                self.release_all(backend)
                self._was_blocked = True
            return NavigationState(blocked_reason=gate.status())
        self._was_blocked = False

        state = self.update(frame, at)
        if state.blocked_reason:
            self.release_all(backend)
            return state

        self._reconcile(backend, state, at)
        return state

    def _reconcile(self, backend: InputBackend, state: NavigationState, at: float) -> None:
        """Press what should be down, release what should not.

        A diff rather than a re-press. Re-pressing a held key produces key-repeat, which
        some games treat as a fresh input; failing to release one strands the character
        running in a straight line.
        """
        # Release before pressing: a diagonal that swaps sides (WA -> WD) must not have
        # both A and D down even for one dispatch, or the character stalls.
        for key in sorted(self._held - state.held_keys):
            backend.key_up(key)
            self.dispatched += 1
        for key in sorted(state.held_keys - self._held):
            backend.key_down(key)
            self.dispatched += 1
        self._held = set(state.held_keys)

        if state.drag_button and (state.mouse_dx or state.mouse_dy):
            if not self._dragging:
                _press_mouse(backend, state.drag_button, down=True)
                self._dragging = True
            _move_mouse_relative(backend, state.mouse_dx, state.mouse_dy)
            self.dispatched += 1
        elif self._dragging:
            _press_mouse(backend, "right", down=False)
            self._dragging = False

    def release_all(self, backend: InputBackend) -> None:
        """Let go of everything. Called on guard trip and on shutdown."""
        for key in sorted(self._held):
            try:
                backend.key_up(key)
            except Exception:
                pass
        self._held.clear()
        if self._dragging:
            try:
                _press_mouse(backend, "right", down=False)
            except Exception:
                pass
            self._dragging = False

    # -- reporting ---------------------------------------------------------------

    @property
    def held(self) -> set[str]:
        return set(self._held)

    def status(self, at: float | None = None) -> str:
        return " | ".join(
            [self.movement.status(), self.camera.status(), self.gaze.status(at)]
        )

    def stats(self) -> dict[str, object]:
        return {
            "dispatched": self.dispatched,
            "blocked": self.blocked,
            "held": sorted(self._held),
            "sweeps": self.gaze.sweeps_issued,
            "turn_rate_deg_s": round(self.camera.keybind_rate_deg_s, 1),
            "px_per_deg": round(self.camera.mouse_px_per_deg, 2),
        }


def _press_mouse(backend: InputBackend, button: str, down: bool) -> None:
    """Hold or release a mouse button for a camera drag.

    `InputBackend.click` is a full press-and-release, which is not what a drag needs, so
    this reaches for the backend's held-button support when it has it and degrades to a
    click when it does not.
    """
    hold = getattr(backend, "mouse_button", None)
    if callable(hold):
        hold(button, down)
    elif down:
        backend.click(button, None)


def _move_mouse_relative(backend: InputBackend, dx: int, dy: int) -> None:
    """Relative mouse motion, which is what a camera drag is.

    Absolute positioning is wrong here: the camera responds to *motion*, and warping the
    cursor to an absolute point produces one large jump followed by nothing. Backends that
    cannot do relative motion fall back to absolute, which is visibly worse but better
    than silently not turning at all.
    """
    relative = getattr(backend, "mouse_move_relative", None)
    if callable(relative):
        relative(dx, dy)
    else:
        backend.mouse_move(ScreenPoint(dx, dy))
