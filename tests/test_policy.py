"""Policy: conditions, reflexes, rotation.

Every test here constructs a `WorldState` directly and asserts on a decision. No image,
no clock, no game — which is the property the whole `WorldState` contract exists to buy.
"""

from __future__ import annotations

import pytest

from conftest import make_state
from player.act.timeline import Plan, Press, Priority, Step
from player.policy.condition import Condition, parse_condition
from player.policy.reflex import Reflex, ReflexLayer
from player.policy.rotation import Ability, AbilityKind, RotationPlanner, RotationProfile
from player.state import WorldState

# -- conditions -------------------------------------------------------------------


def test_leaf_comparison():
    state = make_state({"player.hp_frac": 0.25})
    assert Condition.field("player.hp_frac", "<", 0.3).evaluate(state)
    assert not Condition.field("player.hp_frac", ">", 0.3).evaluate(state)


def test_missing_field_is_false_by_default():
    """An unreadable HUD must make the player do less, never more."""
    assert not Condition.field("nope", "<", 1.0).evaluate(make_state())


def test_negation_of_a_missing_field_is_also_false():
    """`!x` on an unread field must not read as 'x is absent'.

    This is the rule that keeps an uncalibrated player standing still: a DoT-refresh
    condition of "thunder is not up" does not fire just because nothing can see whether
    thunder is up.
    """
    assert not Condition.falsy("nope").evaluate(make_state())
    assert not Condition.no_buff("thunder_dot").evaluate(make_state())


def test_low_confidence_field_is_treated_as_missing():
    state = WorldState(captured_at=0.0)
    state.set("player.hp_frac", 0.1, confidence=0.2)
    assert not Condition.hp_below(0.5).evaluate(state)


def test_boolean_composition():
    state = make_state({"player.in_combat": True, "player.hp_frac": 0.2})
    assert Condition.all_(
        Condition.truthy("player.in_combat"), Condition.hp_below(0.3)
    ).evaluate(state)
    assert not Condition.all_(
        Condition.truthy("player.in_combat"), Condition.hp_below(0.1)
    ).evaluate(state)
    assert Condition.any_(
        Condition.truthy("player.casting"), Condition.hp_below(0.3)
    ).evaluate(state)
    assert Condition.not_(Condition.truthy("player.casting")).evaluate(state)


def test_buff_and_ready_sugar():
    state = make_state({"buff.astral_fire.active": True, "action.fire4.ready": True})
    assert Condition.buff("astral_fire").evaluate(state)
    assert Condition.ready("fire4").evaluate(state)
    assert not Condition.no_buff("astral_fire").evaluate(state)


def test_parse_condition_forms():
    state = make_state(
        {"player.in_combat": True, "player.mp_frac": 0.5, "player.casting": False}
    )
    assert parse_condition("player.in_combat").evaluate(state)
    assert parse_condition("!player.casting").evaluate(state)
    assert parse_condition({"field": "player.mp_frac", "op": "<", "value": 0.6}).evaluate(state)
    assert parse_condition({"all": ["player.in_combat"]}).evaluate(state)
    assert parse_condition(True).evaluate(state)
    assert not parse_condition(False).evaluate(state)


def test_condition_reports_the_fields_it_reads():
    """Used at startup to fail fast when a policy reads a field nothing produces."""
    condition = Condition.all_(Condition.hp_below(0.3), Condition.buff("x"))
    assert condition.fields() == {"player.hp_frac", "buff.x.active"}


def test_condition_describes_itself():
    assert Condition.hp_below(0.3).describe() == "player.hp_frac < 0.3"


def test_type_mismatch_does_not_raise():
    """A config typo comparing a string field to a number must fail closed, not crash."""
    state = make_state({"target.castbar_name": "Ifrit"})
    assert not Condition.field("target.castbar_name", "<", 5).evaluate(state)


# -- reflexes ---------------------------------------------------------------------


def _dodge() -> Plan:
    return Plan(name="d", steps=(Step(0, Press("s", 300)),), priority=Priority.REFLEX)


def test_reflex_fires_when_condition_holds():
    layer = ReflexLayer([Reflex(id="r", condition=Condition.hp_below(0.3), plan=_dodge())])
    fired = layer.decide(make_state({"player.hp_frac": 0.2}), at=100.0)
    assert fired is not None
    assert fired[0].id == "r"


def test_reflex_respects_its_cooldown():
    """A telegraph stays on screen after you have dodged it; re-firing pins you moving."""
    layer = ReflexLayer(
        [Reflex(id="r", condition=Condition.hp_below(0.3), plan=_dodge(), cooldown_ms=1000)]
    )
    state = make_state({"player.hp_frac": 0.2})
    assert layer.decide(state, at=100.0) is not None
    assert layer.decide(state, at=100.5) is None
    assert layer.decide(state, at=101.5) is not None


def test_only_the_highest_priority_reflex_fires():
    """Two preempting plans would flush each other."""
    layer = ReflexLayer(
        [
            Reflex(id="low", condition=Condition.always(), plan=_dodge(), priority=Priority.RECOVERY),
            Reflex(id="high", condition=Condition.always(), plan=_dodge(), priority=Priority.REFLEX),
        ]
    )
    fired = layer.decide(make_state(), at=0.0)
    assert fired[0].id == "high"


def test_disabled_group_does_not_fire():
    layer = ReflexLayer(
        [Reflex(id="r", group="aoe", condition=Condition.always(), plan=_dodge())]
    )
    layer.set_group_enabled("aoe", False)
    assert layer.decide(make_state(), at=0.0) is None


def test_reflex_that_raises_does_not_break_the_loop():
    def explode(_state):
        raise RuntimeError("boom")

    layer = ReflexLayer([Reflex(id="bad", condition=Condition.always(), plan=explode)])
    assert layer.decide(make_state(), at=0.0) is None


def test_reflex_priority_overrides_the_plan_factory():
    layer = ReflexLayer(
        [
            Reflex(
                id="r",
                condition=Condition.always(),
                plan=Plan(name="p", steps=(Step(0, Press("s")),), priority=Priority.IDLE),
                priority=Priority.REFLEX,
            )
        ]
    )
    _, plan = layer.decide(make_state(), at=0.0)
    assert plan.priority is Priority.REFLEX
    assert plan.preempt


# -- rotation ---------------------------------------------------------------------


def _profile() -> RotationProfile:
    return RotationProfile(
        id="test",
        abilities=[
            Ability(
                id="dot",
                key="5",
                kind=AbilityKind.GCD,
                condition=Condition.no_buff("dot_up"),
            ),
            Ability(id="filler", key="1", kind=AbilityKind.GCD),
            Ability(
                id="cd1",
                key="8",
                kind=AbilityKind.OGCD,
                condition=Condition.truthy("player.in_combat"),
            ),
            Ability(
                id="cd2",
                key="9",
                kind=AbilityKind.OGCD,
                condition=Condition.truthy("player.in_combat"),
            ),
            Ability(
                id="cd3",
                key="0",
                kind=AbilityKind.OGCD,
                condition=Condition.truthy("player.in_combat"),
            ),
        ],
        gcd_recast_s=2.5,
        animation_lock_ms=600,
        weave_gap_ms=20,
    )


def test_priority_order_decides_which_gcd_fires():
    planner = RotationPlanner(_profile())
    state = make_state({"player.gcd_remaining_s": 0.0, "buff.dot_up.active": False})
    decision = planner.decide(state, at=0.0)
    assert decision.gcd.id == "dot"  # earlier in the list, and its condition holds


def test_gauge_dependent_ability_is_skipped_when_its_buff_is_unreadable():
    """An uncalibrated status probe must make the player fall through, not guess."""
    planner = RotationPlanner(_profile())
    decision = planner.decide(make_state({"player.gcd_remaining_s": 0.0}), at=0.0)
    assert decision.gcd.id == "filler"


def test_falls_through_to_filler_when_higher_priority_is_gated():
    planner = RotationPlanner(_profile())
    state = make_state({"player.gcd_remaining_s": 0.0, "buff.dot_up.active": True})
    assert planner.decide(state, at=0.0).gcd.id == "filler"


def test_no_decision_while_the_gcd_is_rolling():
    planner = RotationPlanner(_profile())
    assert planner.decide(make_state({"player.gcd_remaining_s": 1.2}), at=0.0) is None


def test_does_not_double_fire_on_consecutive_frames():
    """The cooldown probe reads zero for several frames; a naive planner fires 3x in 50ms."""
    planner = RotationPlanner(_profile())
    state = make_state({"player.gcd_remaining_s": 0.0})
    assert planner.decide(state, at=0.0) is not None
    assert planner.decide(state, at=0.016) is None
    assert planner.decide(state, at=0.033) is None


def test_becomes_ready_again_after_the_gcd():
    planner = RotationPlanner(_profile())
    state = make_state({"player.gcd_remaining_s": 0.0})
    planner.decide(state, at=0.0)
    assert planner.decide(state, at=2.5) is not None


def test_weaves_are_capped_by_the_window_not_by_availability():
    """Three off-GCDs are usable, but only two fit before the next GCD."""
    planner = RotationPlanner(_profile())
    state = make_state(
        {"player.gcd_remaining_s": 0.0, "player.in_combat": True, "buff.dot_up.active": True}
    )
    decision = planner.decide(state, at=0.0)
    assert len(decision.weaves) == 2


def test_weave_offsets_land_in_the_animation_lock_gap():
    planner = RotationPlanner(_profile())
    state = make_state(
        {"player.gcd_remaining_s": 0.0, "player.in_combat": True, "buff.dot_up.active": True}
    )
    decision = planner.decide(state, at=0.0)
    offsets = [s.at_ms for s in decision.plan.steps]
    assert offsets == [0, 620, 1240]
    assert max(offsets) + 600 < 2500  # last weave clears before the next GCD


def test_weave_capacity_derives_from_timing_constants():
    profile = _profile()
    assert profile.weave_capacity() == 2

    profile.gcd_recast_s = 1.5  # heavily hasted: only one weave fits
    assert profile.weave_capacity() == 1


def test_unavailable_ability_is_skipped():
    planner = RotationPlanner(_profile())
    state = make_state(
        {
            "player.gcd_remaining_s": 0.0,
            "action.dot.ready": False,
            "action.filler.ready": True,
        }
    )
    assert planner.decide(state, at=0.0).gcd.id == "filler"


def test_switching_profile_clears_commitment():
    planner = RotationPlanner(_profile())
    state = make_state({"player.gcd_remaining_s": 0.0})
    planner.decide(state, at=0.0)
    assert planner.decide(state, at=0.1) is None

    planner.set_profile(_profile())
    assert planner.decide(state, at=0.1) is not None


def test_min_gap_prevents_rapid_reuse():
    profile = RotationProfile(
        id="gapped",
        abilities=[Ability(id="rare", key="1", kind=AbilityKind.GCD, min_gap_s=10.0)],
    )
    planner = RotationPlanner(profile)
    state = make_state({"player.gcd_remaining_s": 0.0})
    assert planner.decide(state, at=0.0) is not None
    assert planner.decide(state, at=5.0) is None
    assert planner.decide(state, at=11.0) is not None


def test_rotation_plan_expires_before_the_next_gcd():
    planner = RotationPlanner(_profile())
    decision = planner.decide(make_state({"player.gcd_remaining_s": 0.0}), at=0.0)
    assert decision.plan.expires_in_ms < 2500
