"""End-to-end wiring, and the FFXIV profile's own consistency.

These are the tests that catch a contract drifting: a policy reading a field no sensor
produces, a keybind that does not resolve, a directive vocabulary the runtime cannot
apply.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_state
from games.ffxiv import DEFAULT_KEYS, FFXIVConfig, Layout
from games.ffxiv import build as build_ffxiv
from games.ffxiv.reflexes import telegraph_overlaps_player
from player.act.backend import NullBackend
from player.act.keymap import parse_combo, resolve_key, validate_keymap
from player.capture.source import StaticSource
from player.geometry import Rect
from player.runtime import Runtime, RuntimeConfig
from player.safety.guards import SafetyGate
from player.state import Entity, WorldState
from player.strategy.directives import (
    DirectiveBatch,
    EnableReflexGroup,
    Pause,
    SelectPlanProfile,
    SetObjective,
    SetParameter,
    parse_batch,
    response_schema,
)

# -- keymap -----------------------------------------------------------------------


def test_scancodes_resolve_for_the_default_bindings():
    for combo in DEFAULT_KEYS.values():
        parse_combo(combo)


def test_unknown_key_raises_at_config_time():
    """A silently ignored keybind is a rotation missing one ability. Fail loudly instead."""
    with pytest.raises(KeyError):
        resolve_key("nosuchkey")


def test_validate_keymap_reports_every_problem():
    problems = validate_keymap({"good": "1", "bad": "wat", "worse": ""})
    assert len(problems) == 2


def test_modifier_combos_parse():
    mods, base = parse_combo("ctrl+shift+3")
    assert mods == ["ctrl", "shift"]
    assert base == "3"


def test_extended_keys_are_flagged():
    assert resolve_key("up").extended
    assert not resolve_key("a").extended


# -- profile ----------------------------------------------------------------------


@pytest.fixture
def ffxiv_profile():
    return build_ffxiv(FFXIVConfig(rotation="blm.dummy_safe"))


def test_profile_validates_clean(ffxiv_profile):
    """Every field any policy reads must be produced by some sensor."""
    assert ffxiv_profile.validate() == []


def test_profile_exposes_both_rotations(ffxiv_profile):
    assert set(ffxiv_profile.rotations) == {"blm.single_target", "blm.dummy_safe"}


def test_profile_rejects_a_bad_keybind():
    with pytest.raises(ValueError, match="invalid keybinds"):
        build_ffxiv(FFXIVConfig(keys={**DEFAULT_KEYS, "fire4": "nonsense"}))


def test_profile_rejects_an_unknown_default_rotation():
    profile = build_ffxiv(FFXIVConfig(rotation="does.not.exist"))
    assert any("default_rotation" in p for p in profile.validate())


def test_ground_aoe_group_is_off_without_a_detector(ffxiv_profile):
    """An always-false reflex is noise on the status line; disable it and say why."""
    if ffxiv_profile.detector_backend == "none":
        assert "ground_aoe" not in ffxiv_profile.reflexes.enabled_groups()


def test_layout_round_trips_through_json(tmp_path):
    path = tmp_path / "layout.json"
    Layout().save(path)
    loaded = Layout.load(path)
    assert loaded.player_hp == Layout().player_hp
    assert loaded.hotbar.slots == Layout().hotbar.slots


def test_hotbar_slots_do_not_overlap():
    hotbar = Layout().hotbar
    regions = hotbar.regions()
    for left, right in zip(regions, regions[1:]):
        assert left.x + left.w <= right.x + 1e-9


# -- telegraph overlap ------------------------------------------------------------


def test_overlap_true_when_player_stands_in_a_telegraph():
    layout = Layout()
    state = WorldState(captured_at=0.0)
    # Player anchor is near frame centre; a box covering the centre must overlap.
    state.entities = [Entity(kind="telegraph", bbox=Rect(400, 400, 400, 400), confidence=0.8)]
    assert telegraph_overlaps_player(layout, state, frame_size=(1000, 1000))


def test_overlap_false_when_the_telegraph_is_elsewhere():
    layout = Layout()
    state = WorldState(captured_at=0.0)
    state.entities = [Entity(kind="telegraph", bbox=Rect(0, 0, 80, 80), confidence=0.8)]
    assert not telegraph_overlaps_player(layout, state, frame_size=(1000, 1000))


def test_overlap_false_with_no_entities():
    assert not telegraph_overlaps_player(Layout(), WorldState(captured_at=0.0), (1000, 1000))


# -- directives -------------------------------------------------------------------


def test_directive_schema_is_structured_output_compatible():
    """extra=forbid renders as additionalProperties:false, which the validator requires."""
    schema = response_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) >= {"analysis", "directives", "next_review_s"}


def test_directive_batch_parses_from_json():
    batch = parse_batch(
        """
        {"analysis": "stable",
         "directives": [{"kind": "set_objective", "reason": "start", "objective": "dps"}],
         "next_review_s": 10.0}
        """
    )
    assert isinstance(batch.directives[0], SetObjective)


def test_unknown_directive_kind_is_rejected():
    """A hallucinated directive must fail validation, not be half-applied."""
    with pytest.raises(Exception):
        parse_batch('{"analysis": "", "directives": [{"kind": "press_key", "reason": "x"}], "next_review_s": 5}')


def test_empty_directive_list_is_valid():
    """'Nothing should change' has to be expressible, or the model will invent a change."""
    batch = parse_batch('{"analysis": "no change", "directives": [], "next_review_s": 30.0}')
    assert batch.directives == []


# -- runtime ----------------------------------------------------------------------


@pytest.fixture
def runtime(ffxiv_profile):
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    return Runtime(
        profile=ffxiv_profile,
        source=StaticSource(image),
        backend=NullBackend(),
        gate=SafetyGate(),
        config=RuntimeConfig(dry_run=True),
    )


def test_preflight_passes_in_dry_run(runtime):
    assert runtime.preflight() == []


def test_preflight_refuses_live_without_a_killswitch(ffxiv_profile):
    """Running the dispatcher with no way to stop it is not a degraded mode worth having."""
    rt = Runtime(
        profile=ffxiv_profile,
        source=StaticSource(np.zeros((16, 16, 3), np.uint8)),
        config=RuntimeConfig(dry_run=False),
    )
    assert any("kill switch" in p for p in rt.preflight())


def test_directive_selects_a_known_profile(runtime):
    assert runtime.apply_directive(
        SelectPlanProfile(reason="better", profile_id="blm.single_target")
    )
    assert runtime.planner.profile.id == "blm.single_target"


def test_directive_rejects_an_unknown_profile(runtime):
    assert not runtime.apply_directive(SelectPlanProfile(reason="x", profile_id="nope"))
    assert runtime.planner.profile.id == "blm.dummy_safe"


def test_directive_rejects_an_undeclared_parameter(runtime):
    """The declared parameter list is what stops a directive reaching arbitrary internals."""
    assert not runtime.apply_directive(SetParameter(reason="x", key="secret", value="1"))


def test_directive_coerces_to_the_declared_type(runtime):
    runtime.apply_directive(SetParameter(reason="x", key="rotation.enabled", value="false"))
    assert runtime.parameters["rotation.enabled"] is False

    runtime.apply_directive(SetParameter(reason="x", key="potion.hp_threshold", value="0.5"))
    assert runtime.parameters["potion.hp_threshold"] == pytest.approx(0.5)


def test_directive_toggles_a_known_reflex_group(runtime):
    assert runtime.apply_directive(
        EnableReflexGroup(reason="dungeon", group="ground_aoe", enabled=True)
    )
    assert "ground_aoe" in runtime.reflexes.enabled_groups()


def test_directive_rejects_an_unknown_reflex_group(runtime):
    assert not runtime.apply_directive(
        EnableReflexGroup(reason="x", group="nonexistent", enabled=True)
    )


def test_set_objective_is_recorded(runtime):
    runtime.apply_directive(SetObjective(reason="asked", objective="clear the dungeon"))
    assert runtime.objective == "clear the dungeon"


def test_director_context_lists_only_real_options(runtime):
    """The prompt tells the model what is legal; this is what makes that list true."""
    context = runtime.director_context()
    assert set(context.profile_ids) == set(runtime.profile.rotations)
    assert set(context.reflex_groups) == runtime.reflexes.groups()
    assert set(context.parameters) == set(runtime.profile.parameters)


def test_status_line_renders_before_any_frame(runtime):
    assert "ticks=0" in runtime.status_line()
