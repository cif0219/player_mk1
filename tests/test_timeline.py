"""The input timeline.

The invariants under test are the ones the previous design got wrong: reflexes must not
queue behind rotation sleeps, late actions must be dropped rather than executed, and a
key must never be left held.
"""

from __future__ import annotations

import pytest

from player.act.timeline import (
    InputTimeline,
    KeyDown,
    KeyUp,
    Plan,
    Press,
    Priority,
    Step,
    weave,
)


@pytest.fixture
def timeline() -> InputTimeline:
    return InputTimeline()


def test_press_expands_to_down_and_up(timeline):
    timeline.submit(Plan.single("2", hold_ms=40), at=100.0)

    down = timeline.due(at=100.0)
    assert len(down) == 1
    assert down[0].event == KeyDown("2")

    assert timeline.due(at=100.02) == []  # still held

    up = timeline.due(at=100.04)
    assert len(up) == 1
    assert up[0].event == KeyUp("2")


def test_events_come_out_in_deadline_order(timeline):
    plan = Plan(
        name="seq",
        steps=(Step(200, Press("b", 10)), Step(0, Press("a", 10))),
    )
    timeline.submit(plan, at=0.0)
    events = timeline.due(at=1.0)
    keys = [e.event.key for e in events]
    assert keys == ["a", "a", "b", "b"]  # a down/up, then b down/up


def test_higher_priority_preempts_and_returns_pending_releases(timeline):
    timeline.submit(Plan.single("1", hold_ms=400), at=0.0)
    timeline.due(at=0.0)  # dispatch the key-down; the key-up is still scheduled

    releases = timeline.cancel_below(Priority.REFLEX, at=0.1)

    assert [r.event for r in releases] == [KeyUp("1")]
    assert timeline.pending() == 0


def test_preempting_plan_cancels_lower_priority_work(timeline):
    timeline.submit(
        Plan(name="rot", steps=(Step(500, Press("2")),), priority=Priority.ROTATION),
        at=0.0,
    )
    assert timeline.pending() == 2

    timeline.submit(
        Plan(name="dodge", steps=(Step(0, Press("s", 300)),), priority=Priority.REFLEX, preempt=True),
        at=0.1,
    )

    remaining = {e.plan_name for e in timeline.peek(10)}
    assert remaining == {"dodge"}


def test_equal_priority_does_not_preempt(timeline):
    """A rotation step must not cancel another rotation step."""
    timeline.submit(
        Plan(name="a", steps=(Step(500, Press("1")),), priority=Priority.ROTATION), at=0.0
    )
    accepted = timeline.submit(
        Plan(name="b", steps=(Step(500, Press("2")),), priority=Priority.ROTATION), at=0.0
    )
    assert accepted
    assert {e.plan_name for e in timeline.peek(10)} == {"a", "b"}


def test_lower_priority_is_rejected_while_higher_is_pending(timeline):
    timeline.submit(
        Plan(name="reflex", steps=(Step(300, Press("s")),), priority=Priority.REFLEX), at=0.0
    )
    accepted = timeline.submit(
        Plan(name="rotation", steps=(Step(0, Press("1")),), priority=Priority.ROTATION), at=0.0
    )
    assert not accepted
    assert timeline.rejected_plans == 1


def test_expired_press_is_dropped_rather_than_dispatched_late(timeline):
    """A late action is a wrong action, not a slow success."""
    timeline.submit(Plan.single("1", expires_in_ms=100), at=0.0)
    events = timeline.due(at=5.0)
    assert events == []
    assert timeline.expired_events >= 1


def test_release_is_never_expired(timeline):
    """The invariant that stops a key being left held."""
    timeline.submit(Plan.single("1", hold_ms=40, expires_in_ms=10), at=0.0)
    timeline.due(at=0.0)  # key-down goes out before the expiry bites

    late = timeline.due(at=100.0)
    assert [e.event for e in late] == [KeyUp("1")]


def test_orphaned_release_is_dropped(timeline):
    """A key-up is owed only if the key-down actually went out.

    When a press expires before dispatch, both halves must go. Sending the release alone
    is a spurious input event, and in a game that can cancel a cast or drop held
    movement.
    """
    timeline.submit(Plan.single("1", hold_ms=40, expires_in_ms=100), at=0.0)

    # Nothing dispatched before expiry: both down and up are dropped together.
    assert timeline.due(at=5.0) == []
    assert timeline.pending() == 0


def test_cancelling_an_undispatched_press_owes_no_release(timeline):
    """Both halves still pending means the key was never pressed."""
    timeline.submit(
        Plan(name="rot", steps=(Step(500, Press("2")),), priority=Priority.ROTATION), at=0.0
    )
    releases = timeline.cancel_below(Priority.REFLEX, at=0.1)
    assert releases == []


def test_flushing_an_undispatched_press_owes_no_release(timeline):
    timeline.submit(Plan(name="later", steps=(Step(500, Press("2")),)), at=0.0)
    assert timeline.flush() == []


def test_flushing_a_held_key_still_owes_its_release(timeline):
    timeline.submit(Plan.single("1", hold_ms=500), at=0.0)
    timeline.due(at=0.0)  # the key is now down
    assert [r.event for r in timeline.flush()] == [KeyUp("1")]


def test_flush_returns_releases_so_caller_can_lift_keys(timeline):
    timeline.submit(Plan.single("1", hold_ms=400), at=0.0)
    timeline.due(at=0.0)

    releases = timeline.flush()

    assert [r.event for r in releases] == [KeyUp("1")]
    assert timeline.pending() == 0


def test_next_deadline_reports_the_earliest(timeline):
    assert timeline.next_deadline() is None
    timeline.submit(Plan(name="late", steps=(Step(500, Press("b")),)), at=10.0)
    timeline.submit(Plan(name="soon", steps=(Step(100, Press("a")),)), at=10.0)
    assert timeline.next_deadline() == pytest.approx(10.1)


# -- weave scheduling -------------------------------------------------------------


def test_weave_places_ogcds_after_the_animation_lock():
    plan = weave("w", "1", ["8", "9"], animation_lock_ms=600, weave_gap_ms=20)
    offsets = [s.at_ms for s in plan.steps]
    assert offsets == [0, 620, 1240]


def test_weave_fits_inside_the_gcd_window():
    """Both weaves must land before the next GCD, or the second one clips it."""
    plan = weave("w", "1", ["8", "9"], animation_lock_ms=600, weave_gap_ms=20)
    last = max(s.at_ms for s in plan.steps)
    assert last + 600 < 2500


def test_weave_with_no_ogcds_is_a_single_press():
    plan = weave("w", "1", [])
    assert len(plan.steps) == 1


def test_plan_duration_includes_hold_time():
    plan = Plan(name="p", steps=(Step(100, Press("a", hold_ms=50)),))
    assert plan.duration_ms == 150


def test_preemption_pulls_owed_releases_forward():
    """A key already pressed by cancelled work is released immediately, never left down."""
    timeline = InputTimeline()
    timeline.submit(Plan(name="walk", steps=(Step(0, Press("w", 700)),), priority=Priority.ROTATION), at=10.0)
    pressed = timeline.due(10.0)
    assert [type(e.event).__name__ for e in pressed] == ["KeyDown"]
    # 100 ms in, a reflex preempts: the pending W release must come out now, before the dodge
    timeline.submit(Plan(name="dodge", steps=(Step(0, Press("d", 300)),), priority=Priority.REFLEX, preempt=True), at=10.1)
    now_due = timeline.due(10.1)
    kinds = [(type(e.event).__name__, e.event.key) for e in now_due]
    assert ("KeyUp", "w") in kinds and ("KeyDown", "d") in kinds
    assert kinds.index(("KeyUp", "w")) < kinds.index(("KeyDown", "d"))
    # and nothing later still owes a W release
    later = timeline.due(11.0)
    assert all(not (isinstance(e.event, KeyUp) and e.event.key == "w") for e in later)
