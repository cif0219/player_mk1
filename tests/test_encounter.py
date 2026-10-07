"""The mechanic loop, and its coordination with the rotation loop.

The tests that matter most here are the interaction ones at the bottom. Two loops sharing
one body is where this design either works or does not, and preemption-versus-anticipation
is the specific thing being asserted.
"""

from __future__ import annotations

import pytest

from conftest import make_state
from player.encounter.runner import MechanicRunner, Phase
from player.encounter.script import (
    CastTrigger,
    DebuffTrigger,
    EncounterScript,
    LookAt,
    LookPlan,
    Mechanic,
    Resolution,
    ResolutionKind,
    TimelineTrigger,
)
from player.policy.commitment import Commitment, CommitmentBoard, CommitmentKind
from player.policy.condition import Condition
from player.policy.rotation import Ability, AbilityKind, RotationPlanner, RotationProfile
from player.world.arena import (
    ArenaFrame,
    ArenaPoint,
    CameraPose,
    Observability,
    default_ring_layout,
)

LAYOUT = default_ring_layout(radius_m=18.0)


def frame(player=ArenaPoint(0, 0), yaw=0.0) -> ArenaFrame:
    pose = CameraPose(yaw_deg=yaw, confidence=1.0)
    return ArenaFrame(
        player=player,
        pose=pose,
        layout=LAYOUT,
        observability=Observability(pose=pose, player=player, usable_fov_deg=70),
        confidence=1.0,
    )


# -- triggers ---------------------------------------------------------------------


def test_cast_trigger_matches_case_insensitively():
    trigger = CastTrigger("Radiant Plume")
    assert trigger.fires(make_state({"boss.cast_name": "radiant plume"}), 0.0, 0.0)
    assert not trigger.fires(make_state({"boss.cast_name": "Eruption"}), 0.0, 0.0)


def test_cast_trigger_does_not_fire_on_an_empty_read():
    """OCR that returns nothing must not match a mechanic."""
    assert not CastTrigger("X").fires(make_state({"boss.cast_name": ""}), 0.0, 0.0)


def test_timeline_trigger_fires_inside_its_window():
    trigger = TimelineTrigger(at_s=45.0, window_s=1.5)
    assert trigger.fires(make_state(), 45.5, 0.0)
    assert not trigger.fires(make_state(), 43.0, 0.0)
    assert not trigger.fires(make_state(), 48.0, 0.0)


def test_debuff_trigger_reads_the_debuff_field():
    assert DebuffTrigger("light").fires(make_state({"debuff.light.active": True}), 0.0, 0.0)


# -- resolutions ------------------------------------------------------------------


def test_waymark_resolution_returns_the_mark():
    resolution = Resolution(kind=ResolutionKind.WAYMARK, waymark="A")
    assert resolution.destination(frame(), make_state()) == LAYOUT.get("A")


def test_avoid_telegraphs_picks_the_clear_candidate():
    resolution = Resolution(
        kind=ResolutionKind.AVOID_TELEGRAPHS, candidates=("A", "B", "C", "D"), clearance_m=5.0
    )
    # Telegraphs sitting on A, B and C; only D is clear.
    telegraphs = [LAYOUT.get("A"), LAYOUT.get("B"), LAYOUT.get("C")]
    assert resolution.destination(frame(), make_state(), None, telegraphs) == LAYOUT.get("D")


def test_avoid_telegraphs_reports_no_safe_spot_rather_than_the_least_bad():
    """Walking confidently into slightly less damage is not a resolution."""
    resolution = Resolution(
        kind=ResolutionKind.AVOID_TELEGRAPHS, candidates=("A", "B"), clearance_m=5.0
    )
    covered = [LAYOUT.get("A"), LAYOUT.get("B")]
    assert resolution.destination(frame(), make_state(), None, covered) is None


def test_relative_to_boss_places_behind_it():
    resolution = Resolution(kind=ResolutionKind.RELATIVE_TO_BOSS, angle_deg=180, distance_m=5.0)
    boss = ArenaPoint(0, 0)
    # Boss facing north; "behind" is 5m south.
    destination = resolution.destination(frame(), make_state({"boss.facing_deg": 0.0}), boss)
    assert destination.y == pytest.approx(-5.0, abs=0.01)


# -- script validation ------------------------------------------------------------


def _script(mechanics) -> EncounterScript:
    return EncounterScript(id="test", layout=LAYOUT, mechanics=mechanics)


def test_script_rejects_a_waymark_not_in_the_layout():
    script = _script(
        [
            Mechanic(
                id="m1",
                trigger=TimelineTrigger(1.0),
                resolve=Resolution(kind=ResolutionKind.WAYMARK, waymark="Z"),
            )
        ]
    )
    assert any("waymark 'Z'" in p for p in script.validate())


def test_script_rejects_duplicate_ids():
    m = Mechanic(id="dup", trigger=TimelineTrigger(1.0))
    assert any("duplicate" in p for p in _script([m, m]).validate())


# -- the five-phase loop ----------------------------------------------------------


def _dodge_mechanic(**kwargs) -> Mechanic:
    defaults = dict(
        id="plume",
        trigger=CastTrigger("Radiant Plume"),
        deadline_ms=5000,
        look=LookPlan(at=LookAt.WAYMARK, waymark="A", settle_s=0.0),
        resolve=Resolution(kind=ResolutionKind.WAYMARK, waymark="C"),
    )
    defaults.update(kwargs)
    return Mechanic(**defaults)


def test_runner_walks_the_five_phases():
    runner = MechanicRunner(_script([_dodge_mechanic()]))
    runner.note_combat_start(0.0)
    triggered = make_state({"boss.cast_name": "Radiant Plume"})
    quiet = make_state()

    # Trigger -> OBSERVE (the camera is not yet framing A).
    assert runner.tick(triggered, frame(yaw=180.0), at=1.0).phase is Phase.OBSERVE

    # Once framed, RESOLVE and POSITION chain in the same tick — a mechanic with a
    # one-second window cannot spend frames changing its mind about which phase it is in.
    looking_at_a = frame(yaw=0.0)  # A is due north; camera faces north
    intent = runner.tick(quiet, looking_at_a, at=1.1)
    assert intent.phase is Phase.POSITION
    assert intent.movement is not None
    assert intent.movement.target == LAYOUT.get("C")

    # Arrive -> RECOVER -> IDLE
    assert runner.tick(quiet, frame(player=LAYOUT.get("C")), at=2.0, arrived=True).phase in (
        Phase.RECOVER,
        Phase.IDLE,
    )
    assert "plume" in runner.completed or runner.active is not None


def test_phase_transitions_do_not_cost_a_frame_each():
    """Three transitions at 60Hz would be 50ms of a mechanic's budget spent on bookkeeping."""
    runner = MechanicRunner(_script([_dodge_mechanic(look=LookPlan(at=LookAt.KEEP))]))
    runner.note_combat_start(0.0)
    runner.tick(make_state({"boss.cast_name": "Radiant Plume"}), frame(), at=1.0)

    # KEEP means no camera work, so OBSERVE, RESOLVE and POSITION all resolve at once.
    intent = runner.tick(make_state(), frame(), at=1.016)
    assert intent.phase is Phase.POSITION
    assert intent.movement is not None


def test_observe_holds_the_camera_until_the_target_is_framed():
    """You cannot resolve what you have not looked at."""
    runner = MechanicRunner(_script([_dodge_mechanic(look=LookPlan(at=LookAt.WAYMARK, waymark="A"))]))
    runner.note_combat_start(0.0)
    runner.tick(make_state({"boss.cast_name": "Radiant Plume"}), frame(), at=1.0)

    # Camera pointing away from A (which is north).
    intent = runner.tick(make_state(), frame(yaw=180.0), at=1.05)
    assert intent.phase is Phase.OBSERVE
    assert intent.gaze is not None
    assert intent.gaze.target.point == LAYOUT.get("A")


def test_observe_gives_up_when_its_budget_runs_out():
    """A mechanic that never frames its target must still try, not hang."""
    runner = MechanicRunner(_script([_dodge_mechanic()]), observe_budget_frac=0.1)
    runner.note_combat_start(0.0)
    runner.tick(make_state({"boss.cast_name": "Radiant Plume"}), frame(yaw=180.0), at=1.0)

    intent = runner.tick(make_state(), frame(yaw=180.0), at=2.0)  # 1s > 10% of 5s

    assert intent.phase is Phase.POSITION  # gave up observing and resolved anyway
    assert "budget spent" in intent.note


def test_mechanic_does_not_restart_while_its_cast_is_still_up():
    """A cast trigger stays true for seconds; without debounce it never progresses."""
    runner = MechanicRunner(_script([_dodge_mechanic()]))
    runner.note_combat_start(0.0)
    triggered = make_state({"boss.cast_name": "Radiant Plume"})

    runner.tick(triggered, frame(), at=1.0)
    runner.tick(triggered, frame(yaw=0.0), at=1.1)
    runner.tick(triggered, frame(yaw=0.0), at=1.2)
    runner.tick(triggered, frame(player=LAYOUT.get("C")), at=1.3, arrived=True)

    assert len(runner.completed) <= 1


def test_missing_the_deadline_is_recorded_as_a_failure():
    runner = MechanicRunner(_script([_dodge_mechanic(deadline_ms=1000)]))
    runner.note_combat_start(0.0)
    runner.tick(make_state({"boss.cast_name": "Radiant Plume"}), frame(), at=1.0)
    intent = runner.tick(make_state(), frame(yaw=180.0), at=5.0)
    assert intent.phase is Phase.FAILED
    assert runner.failures


def test_losing_localisation_abandons_rather_than_guessing():
    """A mechanic resolved against wrong coordinates walks into the thing it dodged."""
    runner = MechanicRunner(_script([_dodge_mechanic()]))
    runner.note_combat_start(0.0)
    runner.tick(make_state({"boss.cast_name": "Radiant Plume"}), frame(), at=1.0)
    intent = runner.tick(make_state(), None, at=1.1)
    assert intent.phase is Phase.IDLE
    assert runner.failures


def test_resync_reanchors_the_encounter_clock():
    """Without this a timeline script drifts and every later mechanic fires late."""
    runner = MechanicRunner(_script([]))
    runner.note_combat_start(0.0)
    runner.resync(at=100.0, to_elapsed_s=45.0)
    assert runner.elapsed_s(100.0) == pytest.approx(45.0)


# -- the commitment board ---------------------------------------------------------


def test_board_reports_time_until_movement():
    board = CommitmentBoard()
    board.publish(Commitment(CommitmentKind.MOVING, start_at=105.0, end_at=108.0))
    assert board.cast_window_s(at=100.0) == pytest.approx(5.0)


def test_board_reports_zero_window_while_moving():
    board = CommitmentBoard()
    board.publish(Commitment(CommitmentKind.MOVING, start_at=99.0, end_at=105.0))
    assert board.cast_window_s(at=100.0) == 0.0


def test_empty_board_imposes_no_constraint():
    """A striking dummy session must behave exactly as it did before any of this."""
    assert CommitmentBoard().cast_window_s(at=100.0) == float("inf")


def test_replace_swaps_out_one_source():
    board = CommitmentBoard()
    board.publish(Commitment(CommitmentKind.MOVING, 100.0, 110.0, source="mechanic"))
    board.publish(Commitment(CommitmentKind.NO_TARGET, 100.0, 110.0, source="phase"))
    board.replace("mechanic", [])
    kinds = {c.kind for c in board.active(at=105.0)}
    assert kinds == {CommitmentKind.NO_TARGET}


# -- where the two loops meet -----------------------------------------------------


def _profile() -> RotationProfile:
    return RotationProfile(
        id="caster",
        abilities=[
            Ability(id="hardcast", key="1", kind=AbilityKind.GCD, cast_time_s=2.8),
            Ability(id="instant", key="2", kind=AbilityKind.GCD, cast_time_s=0.0),
            Ability(id="surecast", key="9", kind=AbilityKind.GCD, cast_time_s=0.0),
        ],
    )


def test_rotation_prefers_a_hard_cast_when_there_is_time():
    planner = RotationPlanner(_profile())
    board = CommitmentBoard()
    decision = planner.decide(make_state({"player.gcd_remaining_s": 0.0}), at=100.0, board=board)
    assert decision.gcd.id == "hardcast"


def test_rotation_picks_an_instant_when_movement_is_imminent():
    """Anticipation, not preemption: a cast that would be cancelled is never started."""
    planner = RotationPlanner(_profile())
    board = CommitmentBoard()
    board.publish(Commitment(CommitmentKind.MOVING, start_at=101.5, end_at=104.0, reason="dodge"))

    decision = planner.decide(make_state({"player.gcd_remaining_s": 0.0}), at=100.0, board=board)

    assert decision.gcd.id == "instant"
    assert planner.clips_avoided >= 1


def test_rotation_yields_a_reserved_gcd_to_the_mechanic():
    planner = RotationPlanner(_profile())
    board = CommitmentBoard()
    board.publish(
        Commitment(
            CommitmentKind.GCD_RESERVED,
            start_at=99.0,
            end_at=102.0,
            reason="knockback",
            ability_id="surecast",
        )
    )

    decision = planner.decide(make_state({"player.gcd_remaining_s": 0.0}), at=100.0, board=board)

    assert decision.gcd.id == "surecast"
    assert decision.yielded_to == "mechanic reserved"


def test_rotation_stops_entirely_when_there_is_no_target():
    planner = RotationPlanner(_profile())
    board = CommitmentBoard()
    board.publish(Commitment(CommitmentKind.NO_TARGET, 99.0, 105.0, reason="phase change"))
    assert planner.decide(make_state({"player.gcd_remaining_s": 0.0}), at=100.0, board=board) is None


def test_rotation_skips_weaves_when_an_ogcd_is_reserved():
    profile = _profile()
    profile.abilities.append(Ability(id="cd", key="8", kind=AbilityKind.OGCD))
    planner = RotationPlanner(profile)
    board = CommitmentBoard()
    board.publish(Commitment(CommitmentKind.OGCD_RESERVED, 99.0, 102.0, reason="mit"))

    decision = planner.decide(
        make_state({"player.gcd_remaining_s": 0.0}), at=100.0, board=board
    )
    assert decision.weaves == []


def test_runner_publishes_a_movement_claim_before_it_starts_moving():
    """The whole point: the rotation is warned seconds ahead, not at the moment of impact."""
    runner = MechanicRunner(_script([_dodge_mechanic()]))
    runner.note_combat_start(0.0)
    intent = runner.tick(make_state({"boss.cast_name": "Radiant Plume"}), frame(), at=1.0)

    moving = [c for c in intent.commitments if c.kind is CommitmentKind.MOVING]
    assert moving, "OBSERVE should already claim the coming movement window"
    assert moving[0].start_at > 1.0


def test_runner_reserves_the_gcd_its_ability_needs():
    runner = MechanicRunner(
        _script([_dodge_mechanic(ability_id="surecast", ability_lead_ms=800)])
    )
    runner.note_combat_start(0.0)
    intent = runner.tick(make_state({"boss.cast_name": "Radiant Plume"}), frame(), at=1.0)

    reserved = [c for c in intent.commitments if c.kind is CommitmentKind.GCD_RESERVED]
    assert reserved and reserved[0].ability_id == "surecast"


def test_two_loops_together_produce_an_instant_then_the_mechanic_ability():
    """End to end: the mechanic books time, the rotation adapts, the ability lands."""
    runner = MechanicRunner(_script([_dodge_mechanic(ability_id="surecast", deadline_ms=3000)]))
    runner.note_combat_start(0.0)
    planner = RotationPlanner(_profile())
    board = CommitmentBoard()

    intent = runner.tick(make_state({"boss.cast_name": "Radiant Plume"}), frame(), at=100.0)
    board.replace("mechanic", intent.commitments)

    # Movement is coming, so the rotation declines the 2.8s cast.
    early = planner.decide(make_state({"player.gcd_remaining_s": 0.0}), at=100.0, board=board)
    assert early.gcd.id == "instant"

    # At the reserved window the rotation hands the GCD over.
    planner._committed_until = 0.0
    late = planner.decide(
        make_state({"player.gcd_remaining_s": 0.0}), at=102.5, board=board
    )
    assert late.gcd.id == "surecast"
