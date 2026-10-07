"""Target selection: planned, reactive, and fast enough to matter.

The latency tests are the point. A healer whose tank drops to 20% cannot spend a GCD
acquiring the tank and another casting on them, so a swap has to ride inside the plan that
needs it.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_state
from games.ffxiv.castbar import CastLibrary, signature_of
from games.ffxiv.party import PartyListLayout, build_party_derived
from games.ffxiv.jobs import whm
from player.act.timeline import KeyDown, Plan, Press, Priority, Step
from player.policy.commitment import Commitment, CommitmentBoard, CommitmentKind
from player.policy.condition import Condition
from player.policy.rotation import Ability, AbilityKind, RotationPlanner, RotationProfile
from player.policy.targeting import TargetBindings, TargetKind, TargetResolver, TargetSpec
from player.state import WorldState


def party_state(hps: dict[int, float], **extra) -> WorldState:
    """A world state with a populated party list."""
    state = make_state(extra)
    for slot in range(8):
        present = slot in hps
        state.set(f"party.{slot}.present", present, confidence=1.0)
        if present:
            state.set(f"party.{slot}.hp_frac", hps[slot], confidence=1.0)
    build_party_derived(8).observe(None, None, state)
    return state


# -- resolution -------------------------------------------------------------------


def test_current_target_costs_nothing():
    """The common case for a DPS: no keys, no delay, no risk."""
    resolution = TargetResolver().resolve(TargetSpec.current(), make_state())
    assert resolution.is_noop
    assert resolution.keys == ()


def test_party_slot_maps_to_its_keybind():
    resolver = TargetResolver(TargetBindings(party_slots=("f1", "f2", "f3")))
    resolution = resolver.resolve(TargetSpec.party_slot(2), party_state({0: 1.0, 2: 0.5}))
    assert resolution.keys == ("f3",)


def test_lowest_ally_picks_the_hurt_one():
    resolver = TargetResolver()
    state = party_state({0: 0.95, 1: 0.40, 2: 0.80})
    resolution = resolver.resolve(TargetSpec.lowest_ally(), state)
    assert resolution.slot == 1
    assert resolution.keys == ("f2",)


def test_lowest_ally_fails_when_nobody_is_hurt():
    """This is how 'do not heal a healthy party' happens with no code."""
    resolver = TargetResolver()
    state = party_state({0: 1.0, 1: 0.98, 2: 0.99})
    assert resolver.resolve(TargetSpec.lowest_ally(hp_ceiling=0.7), state).failed


def test_empty_party_slot_is_not_targeted():
    """Targeting an empty slot does nothing, so the heal would land on the boss."""
    resolver = TargetResolver()
    state = party_state({0: 1.0})
    assert resolver.resolve(TargetSpec.party_slot(5), state).failed


def test_untrusted_party_frame_is_skipped_not_treated_as_zero():
    """An unread frame is not a dying ally."""
    state = party_state({0: 0.9, 1: 0.9})
    state.set("party.2.present", True, confidence=1.0)
    state.set("party.2.hp_frac", 0.05, confidence=0.1)  # read, but not trustworthy
    assert TargetResolver().resolve(TargetSpec.lowest_ally(hp_ceiling=0.5), state).failed


def test_lowest_ally_can_exclude_self():
    resolver = TargetResolver()
    state = party_state({0: 0.30, 1: 0.60}, **{"party.self_slot": 0})
    resolution = resolver.resolve(
        TargetSpec.lowest_ally(hp_ceiling=0.9, include_self=False), state
    )
    assert resolution.slot == 1


def test_nearest_enemy_reports_reduced_confidence():
    """Tab-cycle ordering is genuinely unreliable; saying so beats pretending."""
    resolution = TargetResolver().resolve(TargetSpec.nearest_enemy(), make_state())
    assert resolution.confidence < 1.0


def test_missing_keybind_fails_loudly_rather_than_guessing():
    resolver = TargetResolver(TargetBindings(party_slots=()))
    assert resolver.resolve(TargetSpec.party_slot(0), party_state({0: 0.5})).failed


# -- swap latency -----------------------------------------------------------------


def test_target_keys_ride_inside_the_same_plan():
    """The whole requirement: a swap costs tens of milliseconds, not a GCD."""
    plan = Plan.single("3", hold_ms=40).targeted(("f2",), settle_ms=60)
    offsets = [s.at_ms for s in plan.steps]
    assert offsets == [0, 60]
    assert plan.steps[0].action.key == "f2"
    assert plan.steps[1].action.key == "3"


def test_swap_plus_cast_fits_well_inside_one_gcd():
    plan = Plan.single("3").targeted(("f2",), settle_ms=60)
    assert plan.duration_ms < 250  # a GCD is 2500ms


def test_targeting_shortens_the_expiry_window():
    """A plan that was already marginal must not become late because it targets first."""
    plan = Plan.single("3", expires_in_ms=1000).targeted(("f2",), settle_ms=60)
    assert plan.expires_in_ms < 1000


def test_untargeted_plan_is_unchanged():
    plan = Plan.single("3")
    assert plan.targeted(()) is plan


def test_target_and_ability_are_preempted_together():
    """A reflex between the target key and the heal would land the heal on the boss.

    Preemption spares events that are already due — those are about to be dispatched
    anyway — so the check is that nothing *future* survives. The ability, which is the
    part that would misfire, is always in the future relative to its own target key.
    """
    from player.act.timeline import InputTimeline

    timeline = InputTimeline()
    heal = Plan.single("3", name="heal", expires_in_ms=1000).targeted(("f2",))
    timeline.submit(heal, at=0.0)

    timeline.submit(
        Plan(name="dodge", steps=(Step(0, Press("s", 300)),), priority=Priority.REFLEX, preempt=True),
        at=0.01,
    )

    survivors = [e for e in timeline.peek(10) if e.due_at > 0.01]
    assert {e.plan_name for e in survivors} == {"dodge"}
    assert not any(
        isinstance(e.event, KeyDown) and e.event.key == "3" for e in timeline.peek(10)
    )


# -- the healer profile -----------------------------------------------------------


def _healer_planner(with_mouseover: bool = True) -> RotationPlanner:
    keys = {name: str(i) for i, name in enumerate(whm.DEFAULT_SLOTS.values())}
    resolver = TargetResolver()
    if with_mouseover:
        # Party rows 100px apart down the left edge, as the real layout puts them.
        resolver.set_slot_locator(lambda slot: (60, 400 + slot * 40))
    return RotationPlanner(whm.build(keys), targets=resolver)


def test_healer_deals_damage_when_the_party_is_healthy():
    """Triage falls out of list order: every heal is gated, so damage is the default."""
    planner = _healer_planner()
    state = party_state({0: 1.0, 1: 1.0, 2: 1.0}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })
    decision = planner.decide(state, at=100.0)
    assert decision.gcd.id == "glare"


def test_healer_switches_to_a_heal_when_someone_drops():
    planner = _healer_planner()
    state = party_state({0: 1.0, 1: 0.45, 2: 1.0}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })
    decision = planner.decide(state, at=100.0)
    assert decision.gcd.id in ("cure2", "afflatus_solace")
    assert "party[1]" in decision.target


def test_healer_prefers_the_instant_heal_when_a_lily_is_up():
    """Instants do not clip, which matters more for a healer than raw potency."""
    planner = _healer_planner()
    state = party_state({0: 1.0, 1: 0.45}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.lily.active": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })
    assert planner.decide(state, at=100.0).gcd.id == "afflatus_solace"


def test_healer_hovers_the_lowest_ally_rather_than_targeting_them():
    """Mouseover keeps the boss as the hard target, so the next Glare still lands."""
    from player.act.timeline import MouseMove

    planner = _healer_planner()
    state = party_state({0: 0.9, 1: 0.9, 3: 0.35}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })

    decision = planner.decide(state, at=100.0)

    first = decision.plan.steps[0].action
    assert isinstance(first, MouseMove)
    assert first.point.y == 400 + 3 * 40  # party slot 3's row
    assert "mouseover" in decision.target


def test_mouseover_heal_does_not_claim_the_hard_target():
    """The belief must stay clear, or later damage would think it is aimed at an ally."""
    planner = _healer_planner()
    state = party_state({0: 1.0, 1: 0.45}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })
    planner.decide(state, at=100.0)
    assert planner._target_slot is None


def test_healer_falls_back_to_slot_keys_without_a_locator():
    """Worse — it costs the boss as a target — but better than not healing."""
    planner = _healer_planner(with_mouseover=False)
    state = party_state({0: 1.0, 1: 0.45}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })

    decision = planner.decide(state, at=100.0)

    assert decision.plan.steps[0].action.key == "f2"


def test_healer_prefers_the_instant_heal_when_movement_is_coming():
    """Cure II is a 2s cast; starting it 1.8s before a dodge throws the heal away."""
    planner = _healer_planner()
    board = CommitmentBoard()
    board.publish(Commitment(CommitmentKind.MOVING, start_at=101.8, end_at=104.0, reason="dodge"))
    state = party_state({0: 1.0, 1: 0.45}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.lily.active": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })

    decision = planner.decide(state, at=100.0, board=board)

    assert decision.gcd.id == "afflatus_solace"  # instant, not Cure II
    assert planner.clips_avoided >= 1


def test_healer_casts_nothing_rather_than_clipping():
    """With a 1s window and no instant available, saving the GCD beats wasting it."""
    planner = _healer_planner()
    board = CommitmentBoard()
    board.publish(Commitment(CommitmentKind.MOVING, start_at=101.0, end_at=104.0, reason="dodge"))
    state = party_state({0: 1.0, 1: 0.45}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })

    assert planner.decide(state, at=100.0, board=board) is None


def test_repeated_hard_targets_on_the_same_ally_skip_the_key():
    """A sustained slot-targeted heal costs nothing extra after the first acquisition."""
    planner = _healer_planner(with_mouseover=False)
    state = party_state({0: 1.0, 1: 0.45}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })

    first = planner.decide(state, at=100.0)
    planner._committed_until = 0.0
    second = planner.decide(state, at=101.0)

    assert first.plan.steps[0].action.key == "f2"
    assert second.plan.steps[0].action.key != "f2"  # already targeted


def test_target_belief_expires_and_the_key_is_pressed_again():
    """The game can change our target under us; the belief must not persist forever."""
    planner = _healer_planner(with_mouseover=False)
    state = party_state({0: 1.0, 1: 0.45}, **{
        "player.gcd_remaining_s": 0.0,
        "target.exists": True,
        "buff.dia_dot.active": True,
        "buff.dia_dot.remaining_s": 20.0,
    })

    planner.decide(state, at=100.0)
    planner._committed_until = 0.0
    later = planner.decide(state, at=100.0 + planner.target_belief_s + 1.0)

    assert later.plan.steps[0].action.key == "f2"


# -- planned add phases -----------------------------------------------------------


def _dps_profile() -> RotationProfile:
    return RotationProfile(
        id="dps",
        abilities=[
            Ability(id="nuke", key="1", kind=AbilityKind.GCD, cast_time_s=0.0),
        ],
    )


def test_add_phase_redirects_damage_without_taking_the_gcd():
    """The rotation keeps choosing what to press; it just aims somewhere else."""
    planner = RotationPlanner(
        _dps_profile(), targets=TargetResolver(TargetBindings(enemy_slots=("num1", "num2")))
    )
    board = CommitmentBoard()
    board.publish(
        Commitment(
            CommitmentKind.TARGET_OVERRIDE,
            start_at=99.0,
            end_at=110.0,
            reason="add spawn",
            target=TargetSpec.enemy_slot(1),
        )
    )

    decision = planner.decide(make_state({"player.gcd_remaining_s": 0.0}), at=100.0, board=board)

    assert decision.gcd.id == "nuke"  # the rotation still chose the ability
    assert decision.plan.steps[0].action.key == "num2"  # aimed at the add
    assert decision.yielded_to == "add phase target"


def test_add_phase_override_does_not_redirect_a_heal():
    """Pointing a regen at the add is worse than useless."""
    profile = RotationProfile(
        id="mixed",
        abilities=[
            Ability(
                id="heal",
                key="2",
                kind=AbilityKind.GCD,
                target=TargetSpec.lowest_ally(hp_ceiling=0.7),
                condition=Condition.field("party.lowest_hp_frac", "<", 0.7),
            ),
        ],
    )
    planner = RotationPlanner(profile, targets=TargetResolver())
    board = CommitmentBoard()
    board.publish(
        Commitment(
            CommitmentKind.TARGET_OVERRIDE,
            start_at=99.0,
            end_at=110.0,
            target=TargetSpec.nearest_enemy(),
        )
    )
    state = party_state({0: 1.0, 1: 0.4}, **{"player.gcd_remaining_s": 0.0})

    decision = planner.decide(state, at=100.0, board=board)

    assert decision.plan.steps[0].action.key == "f2"  # still the hurt ally


def test_override_expires_with_its_window():
    board = CommitmentBoard()
    board.publish(
        Commitment(
            CommitmentKind.TARGET_OVERRIDE, 99.0, 105.0, target=TargetSpec.enemy_slot(0)
        )
    )
    assert board.target_override(at=100.0) is not None
    assert board.target_override(at=106.0) is None


# -- cast bar ---------------------------------------------------------------------


def _text_image(pattern: list[int], height: int = 16, width: int = 160) -> np.ndarray:
    """A crop with bright 'glyph' columns where `pattern` says so."""
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    band = width // len(pattern)
    for index, on in enumerate(pattern):
        if on:
            image[3 : height - 3, index * band : (index + 1) * band] = 230
    return image


def test_signature_is_none_for_an_empty_bar():
    """No text means no cast, not an unrecognised one."""
    assert signature_of(np.full((16, 160, 3), 40, dtype=np.uint8)) is None


def test_library_matches_a_learned_signature():
    library = CastLibrary()
    plume = _text_image([1, 0, 1, 1, 0, 0, 1, 0])
    library.add("Radiant Plume", signature_of(plume))

    match = library.match(signature_of(plume))

    assert match is not None
    assert match[0] == "Radiant Plume"


def test_library_rejects_an_unknown_cast():
    """An unrecognised cast must not be forced onto the nearest known one."""
    library = CastLibrary()
    library.add("Radiant Plume", signature_of(_text_image([1, 0, 1, 1, 0, 0, 1, 0])))

    assert library.match(signature_of(_text_image([0, 1, 0, 0, 1, 1, 0, 1]))) is None


def test_library_declines_when_two_candidates_are_too_close():
    """Guessing between similar names fires the wrong mechanic."""
    library = CastLibrary(min_similarity=0.5, min_margin=0.5)
    library.add("A", signature_of(_text_image([1, 0, 1, 0, 1, 0, 1, 0])))
    library.add("B", signature_of(_text_image([1, 0, 1, 0, 1, 0, 0, 1])))

    assert library.match(signature_of(_text_image([1, 0, 1, 0, 1, 0, 1, 0]))) is None


def test_signature_survives_a_resolution_change():
    """One learned signature has to work across UI scales."""
    library = CastLibrary()
    library.add("Plume", signature_of(_text_image([1, 0, 1, 1, 0, 0, 1, 0], width=160)))

    wider = signature_of(_text_image([1, 0, 1, 1, 0, 0, 1, 0], width=320, height=24))

    assert library.match(wider)[0] == "Plume"


def test_empty_library_matches_nothing():
    """With no calibration the reader is honest and CastTrigger falls back to timeline."""
    assert CastLibrary().match((1.0,) * 48) is None
