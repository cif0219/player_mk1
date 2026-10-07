"""Movement, camera, gaze, and the coupling between them."""

from __future__ import annotations

import pytest

from player.act.backend import NullBackend
from player.navigate.camera import CameraController, CameraIntent, CameraMode
from player.navigate.channel import NavigationChannel
from player.navigate.gaze import GazePolicy, GazePriority, GazeRequest, GazeTarget, SectorMemory
from player.navigate.movement import MovementController, MovementIntent
from player.safety.guards import KillSwitchGuard, SafetyGate
from player.world.arena import (
    ArenaFrame,
    ArenaPoint,
    CameraPose,
    Observability,
    default_ring_layout,
)

LAYOUT = default_ring_layout()


def make_frame(player=ArenaPoint(0, 0), yaw=0.0, confidence=1.0) -> ArenaFrame:
    pose = CameraPose(yaw_deg=yaw, confidence=confidence)
    return ArenaFrame(
        player=player,
        pose=pose,
        layout=LAYOUT,
        observability=Observability(pose=pose, player=player),
        confidence=confidence,
    )


# -- movement ---------------------------------------------------------------------


def test_moves_forward_toward_a_target_dead_ahead():
    controller = MovementController()
    controller.move_to(MovementIntent(target=ArenaPoint(0, 10)))
    keys = controller.desired_keys(ArenaPoint(0, 0), CameraPose(yaw_deg=0, confidence=1.0), at=1.0)
    assert keys == {"w"}


def test_movement_is_camera_relative_not_north_relative():
    """The whole reason camera yaw is load-bearing."""
    controller = MovementController()
    controller.move_to(MovementIntent(target=ArenaPoint(0, 10)))  # target due north

    # Camera facing east: north is now to the character's LEFT.
    keys = controller.desired_keys(ArenaPoint(0, 0), CameraPose(yaw_deg=90, confidence=1.0), at=1.0)
    assert keys == {"a"}

    # Camera facing south: north is now BEHIND.
    keys = controller.desired_keys(ArenaPoint(0, 0), CameraPose(yaw_deg=180, confidence=1.0), at=1.0)
    assert keys == {"s"}


def test_diagonal_holds_two_keys():
    controller = MovementController()
    controller.move_to(MovementIntent(target=ArenaPoint(10, 10)))
    keys = controller.desired_keys(ArenaPoint(0, 0), CameraPose(yaw_deg=0, confidence=1.0), at=1.0)
    assert keys == {"w", "d"}


def test_refuses_to_move_without_a_trusted_position():
    """Walking on a guessed position is worse than not walking."""
    controller = MovementController()
    controller.move_to(MovementIntent(target=ArenaPoint(0, 10)))
    assert controller.desired_keys(None, CameraPose(yaw_deg=0, confidence=1.0), at=1.0) == set()


def test_refuses_to_move_without_a_known_camera_yaw():
    controller = MovementController()
    controller.move_to(MovementIntent(target=ArenaPoint(0, 10)))
    assert controller.desired_keys(ArenaPoint(0, 0), CameraPose(confidence=0.0), at=1.0) == set()


def test_stops_on_arrival():
    controller = MovementController()
    controller.move_to(MovementIntent(target=ArenaPoint(0, 10), tolerance_m=1.0))
    keys = controller.desired_keys(ArenaPoint(0, 9.5), CameraPose(yaw_deg=0, confidence=1.0), at=1.0)
    assert keys == set()
    assert controller.arrived


def test_abandons_on_deadline():
    controller = MovementController()
    controller.move_to(MovementIntent(target=ArenaPoint(0, 10), deadline_at=5.0))
    keys = controller.desired_keys(ArenaPoint(0, 0), CameraPose(yaw_deg=0, confidence=1.0), at=6.0)
    assert keys == set()
    assert controller.intent is None


def test_detects_an_unreachable_deadline_early():
    """A mechanic that cannot be reached in time is better abandoned than half-run."""
    controller = MovementController()
    controller.move_to(MovementIntent(target=ArenaPoint(0, 40), deadline_at=1.5))
    controller.desired_keys(ArenaPoint(0, 0), CameraPose(yaw_deg=0, confidence=1.0), at=1.0)
    assert controller.will_miss_deadline(at=1.0, speed_m_s=6.0)


# -- camera -----------------------------------------------------------------------


def test_camera_turns_the_short_way():
    camera = CameraController(mode=CameraMode.KEYBIND)
    camera.look_at_bearing(CameraIntent(yaw_deg=10.0))
    command = camera.command(CameraPose(yaw_deg=350.0, confidence=1.0), at=1.0)
    assert command.held_keys == {"right"}  # +20 degrees, not -340


def test_camera_settles_inside_tolerance():
    camera = CameraController()
    camera.look_at_bearing(CameraIntent(yaw_deg=100.0, tolerance_deg=8.0))
    command = camera.command(CameraPose(yaw_deg=95.0, confidence=1.0), at=1.0)
    assert command.idle
    assert camera.settled


def test_camera_does_nothing_without_a_known_pose():
    camera = CameraController()
    camera.look_at_bearing(CameraIntent(yaw_deg=90.0))
    assert camera.command(CameraPose(confidence=0.0), at=1.0).idle


def test_mouse_drag_step_is_bounded_by_the_slew_cap():
    """Snapping the camera would make every mid-turn frame useless for detection."""
    camera = CameraController(mode=CameraMode.MOUSE_DRAG, slew_cap_deg_s_idle=90.0)
    camera.look_at_bearing(CameraIntent(yaw_deg=180.0))
    camera._last_tick_at = 1.0
    command = camera.command(CameraPose(yaw_deg=0.0, confidence=1.0), at=1.05)
    # 90 deg/s for 50ms is 4.5 degrees, times px_per_deg.
    assert 0 < abs(command.mouse_dx) <= 4.5 * camera.mouse_px_per_deg + 1


def test_slew_cap_is_tighter_while_moving():
    """Camera/movement coupling made explicit: turning fast makes the yaw stale."""
    camera = CameraController(mode=CameraMode.MOUSE_DRAG)
    camera.look_at_bearing(CameraIntent(yaw_deg=180.0))

    camera._last_tick_at = 1.0
    idle_step = abs(camera.command(CameraPose(yaw_deg=0.0, confidence=1.0), moving=False, at=1.05).mouse_dx)
    camera.settled = False
    camera._last_tick_at = 1.0
    moving_step = abs(camera.command(CameraPose(yaw_deg=0.0, confidence=1.0), moving=True, at=1.05).mouse_dx)

    assert moving_step < idle_step


# -- gaze -------------------------------------------------------------------------


def test_sector_memory_marks_what_is_in_frame():
    memory = SectorMemory(sectors=12)
    obs = Observability(
        pose=CameraPose(yaw_deg=0, confidence=1.0), player=ArenaPoint(0, 0), usable_fov_deg=70
    )
    memory.observe(obs, at=100.0)
    assert memory.last_seen[memory.sector_of(0)] == 100.0
    assert memory.last_seen[memory.sector_of(180)] == 0.0  # behind us, unseen


def test_sweep_is_issued_when_a_sector_goes_stale():
    """Converts 'no tower detected' into 'I have not looked there in a while'."""
    policy = GazePolicy(sweep_after_s=5.0)
    obs = Observability(pose=CameraPose(yaw_deg=0, confidence=1.0), player=ArenaPoint(0, 0))
    decision = policy.decide(ArenaPoint(0, 0), obs, at=100.0)
    assert decision is not None and decision.id == "sweep"


def test_mechanic_gaze_outranks_a_sweep():
    """You cannot dodge what is off screen."""
    policy = GazePolicy(sweep_after_s=1.0)
    policy.look_at_mechanic(ArenaPoint(0, 18), deadline_at=200.0)
    obs = Observability(pose=CameraPose(yaw_deg=0, confidence=1.0), player=ArenaPoint(0, 0))
    decision = policy.decide(ArenaPoint(0, 0), obs, at=100.0)
    assert decision.id == "mechanic"


def test_expired_gaze_requests_are_dropped():
    policy = GazePolicy(sweep_after_s=1000.0)
    policy.look_at_mechanic(ArenaPoint(0, 18), deadline_at=50.0)
    obs = Observability(pose=CameraPose(yaw_deg=0, confidence=1.0), player=ArenaPoint(0, 0))
    assert policy.decide(ArenaPoint(0, 0), obs, at=100.0) is None


def test_coverage_reports_how_much_of_the_circle_is_fresh():
    policy = GazePolicy()
    obs = Observability(pose=CameraPose(yaw_deg=0, confidence=1.0), player=ArenaPoint(0, 0))
    policy.decide(ArenaPoint(0, 0), obs, at=100.0)
    coverage = policy.coverage(at=100.0, within_s=5.0)
    assert 0.0 < coverage < 1.0  # a 70-degree cone is not the whole arena


# -- the channel ------------------------------------------------------------------


def test_channel_rejects_keys_the_timeline_already_owns():
    """A key pressed by one channel and released by the other strands the character."""
    with pytest.raises(ValueError, match="both claim"):
        NavigationChannel(reserved_by_timeline={"w"})


def test_channel_holds_and_releases_movement_keys():
    channel = NavigationChannel()
    backend = NullBackend()
    gate = SafetyGate()
    channel.movement.move_to(MovementIntent(target=ArenaPoint(0, 10)))

    channel.pump(backend, gate, make_frame(), at=1.0)
    assert channel.held == {"w"}

    channel.movement.stop()
    channel.pump(backend, gate, make_frame(), at=1.1)
    assert channel.held == set()
    assert any(e.kind == "key_up" and e.detail == "w" for e in backend.events)


def test_channel_releases_everything_when_a_guard_trips():
    """A character left running into a wall after the kill switch is the worst failure."""

    class Tripped:
        tripped = True
        paused = False

    channel = NavigationChannel()
    backend = NullBackend()
    channel.movement.move_to(MovementIntent(target=ArenaPoint(0, 10)))

    channel.pump(backend, SafetyGate(), make_frame(), at=1.0)
    assert channel.held

    channel.pump(backend, SafetyGate([KillSwitchGuard(Tripped())]), make_frame(), at=1.1)
    assert channel.held == set()


def test_channel_stands_still_without_an_arena_frame():
    channel = NavigationChannel()
    backend = NullBackend()
    channel.movement.move_to(MovementIntent(target=ArenaPoint(0, 10)))
    state = channel.pump(backend, SafetyGate(), None, at=1.0)
    assert state.held_keys == set()
    assert "unavailable" in state.blocked_reason


def test_channel_swaps_direction_without_holding_both_sides():
    """WA -> WD must never have A and D down together, even for one dispatch."""
    channel = NavigationChannel()
    backend = NullBackend()
    gate = SafetyGate()

    channel.movement.move_to(MovementIntent(target=ArenaPoint(-10, 10)))
    channel.pump(backend, gate, make_frame(), at=1.0)
    assert channel.held == {"w", "a"}

    channel.movement.move_to(MovementIntent(target=ArenaPoint(10, 10)))
    channel.pump(backend, gate, make_frame(), at=1.1)
    assert channel.held == {"w", "d"}

    # The release must be dispatched before the opposing press.
    kinds = [(e.kind, e.detail) for e in backend.events]
    assert kinds.index(("key_up", "a")) < kinds.index(("key_down", "d"))
