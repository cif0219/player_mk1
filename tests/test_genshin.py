"""Genshin profile tests. Headless, like everything else in the suite."""

from __future__ import annotations

import pytest

from games.genshin import GenshinConfig, build, config_from_dict
from games.genshin.combat import build_attack_only, build_solo
from games.genshin.profile import default_keys
from games.genshin.layout import Layout, PartyStripLayout
from games.genshin.reflexes import build_reflexes
from player.act.backend import NullBackend
from player.act.keymap import mouse_button, parse_combo, validate_keymap
from player.policy.rotation import RotationPlanner

from conftest import make_state


# -- mouse pseudo-keys (the substrate change Genshin forced) -----------------------


class TestMouseKeys:
    def test_mouse1_is_a_valid_combo(self):
        mods, base = parse_combo("mouse1")
        assert mods == []
        assert base == "mouse1"

    def test_aliases_normalise(self):
        assert parse_combo("lmb") == ([], "mouse1")
        assert parse_combo("RMB") == ([], "mouse2")

    def test_modified_mouse_press_is_valid(self):
        assert parse_combo("shift+mouse1") == (["shift"], "mouse1")

    def test_mouse_button_cannot_be_a_modifier(self):
        with pytest.raises(ValueError):
            parse_combo("mouse1+e")

    def test_keymap_with_mouse_binding_validates(self):
        assert validate_keymap({"normal_attack": "mouse1", "skill": "e"}) == []

    def test_unknown_mouse_name_is_rejected(self):
        problems = validate_keymap({"attack": "mouse9"})
        assert len(problems) == 1

    def test_mouse_button_lookup(self):
        assert mouse_button("mouse1") == "left"
        assert mouse_button("rmb") == "right"
        assert mouse_button("e") is None

    def test_null_backend_accepts_mouse_pseudo_key(self):
        backend = NullBackend()
        backend.key_down("mouse1")
        backend.key_up("mouse1")
        assert backend.keys_pressed() == ["mouse1"]


# -- profile assembly --------------------------------------------------------------


class TestProfile:
    def test_builds_and_validates(self):
        profile = build()
        assert profile.validate() == []
        assert profile.name == "genshin"
        assert profile.window_title == "Genshin Impact"

    def test_all_rotations_present(self):
        profile = build()
        assert set(profile.rotations) == {
            "genshin.attack_only",
            "genshin.solo",
            "genshin.charged",
        }

    def test_key_override_is_merged_not_replaced(self):
        profile = build(GenshinConfig(keys={"skill": "r"}))
        solo = profile.rotations["genshin.solo"]
        assert solo.by_id("skill").key == "r"
        assert solo.by_id("normal_attack").key == "mouse1"  # default survived

    def test_typo_in_keybind_fails_at_build(self):
        with pytest.raises(ValueError, match="invalid keybinds"):
            build(GenshinConfig(keys={"skill": "not_a_key"}))

    def test_config_from_dict_roundtrip(self):
        config = config_from_dict(
            {
                "rotation": "genshin.solo",
                "attack_interval_s": 0.8,
                "skill_cooldown_s": 6.0,
                "keys": {"burst": "x"},
            }
        )
        assert config.rotation == "genshin.solo"
        assert config.attack_interval_s == 0.8
        assert config.skill_cooldown_s == 6.0
        profile = build(config)
        assert profile.default_rotation == "genshin.solo"
        assert profile.rotations["genshin.solo"].by_id("burst").key == "x"

    def test_layout_roundtrips_through_json(self, tmp_path):
        path = tmp_path / "layout.json"
        original = Layout(party=PartyStripLayout(origin_y=0.3))
        original.save(path)
        loaded = Layout.load(path)
        assert loaded.party.origin_y == 0.3
        assert loaded.player_hp == original.player_hp


# -- rotation decisions ------------------------------------------------------------


def keys():
    return default_keys()


class TestSoloRotation:
    def make_planner(self):
        return RotationPlanner(build_solo(keys()))

    def test_burst_wins_when_ready(self):
        planner = self.make_planner()
        state = make_state({"action.burst.ready": True, "action.skill.ready": True})
        decision = planner.decide(state, at=1000.0)
        assert decision is not None
        assert decision.gcd.id == "burst"

    def test_skill_when_burst_not_ready(self):
        planner = self.make_planner()
        state = make_state({"action.burst.ready": False, "action.skill.ready": True})
        decision = planner.decide(state, at=1000.0)
        assert decision.gcd.id == "skill"

    def test_normal_attack_is_the_fallback(self):
        planner = self.make_planner()
        state = make_state({"action.burst.ready": False, "action.skill.ready": False})
        decision = planner.decide(state, at=1000.0)
        assert decision.gcd.id == "normal_attack"
        # The press is the mouse pseudo-key, straight into the ordinary key path.
        presses = [s.action for s in decision.plan.steps]
        assert presses[0].key == "mouse1"

    def test_unreadable_icons_degrade_to_normal_attacks(self):
        # No skill fields at all — as when the icon regions are miscalibrated. The
        # conditions read missing fields as False, so the rotation falls through
        # rather than guessing.
        planner = self.make_planner()
        decision = planner.decide(make_state({}), at=1000.0)
        assert decision.gcd.id == "normal_attack"

    def test_pacing_commitment_holds_between_decisions(self):
        planner = self.make_planner()
        state = make_state({"action.burst.ready": False, "action.skill.ready": False})
        assert planner.decide(state, at=1000.0) is not None
        # Immediately after a decision the planner is committed; well past the
        # attack interval it decides again.
        assert planner.decide(state, at=1000.1) is None
        assert planner.decide(state, at=1001.0) is not None

    def test_min_gap_stops_burst_double_fire(self):
        planner = self.make_planner()
        ready = make_state({"action.burst.ready": True, "action.skill.ready": False})
        first = planner.decide(ready, at=1000.0)
        assert first.gcd.id == "burst"
        # The probe still reads ready during the burst cinematic; the gap holds the
        # planner to the filler instead of pressing Q again.
        second = planner.decide(ready, at=1001.0)
        assert second.gcd.id != "burst"


class TestAttackOnly:
    def test_decides_with_an_empty_state(self):
        planner = RotationPlanner(build_attack_only(keys()))
        decision = planner.decide(make_state({}), at=1000.0)
        assert decision is not None
        assert decision.gcd.id == "normal_attack"


# -- reflexes ----------------------------------------------------------------------


class TestReflexes:
    def make_layer(self):
        return build_reflexes(keys(), dash_hp_threshold=0.4, retreat_hp_threshold=0.2)

    def test_healthy_hp_fires_nothing(self):
        layer = self.make_layer()
        assert layer.decide(make_state({"player.hp_frac": 0.9}), at=1000.0) is None

    def test_low_hp_dashes(self):
        layer = self.make_layer()
        fired = layer.decide(make_state({"player.hp_frac": 0.3}), at=1000.0)
        assert fired is not None
        reflex, plan = fired
        assert reflex.id == "dash_low_hp"
        assert plan.preempt

    def test_critical_hp_retreats_instead(self):
        layer = self.make_layer()
        fired = layer.decide(make_state({"player.hp_frac": 0.1}), at=1000.0)
        assert fired is not None
        assert fired[0].id == "retreat_critical_hp"

    def test_cooldown_stops_dash_spam(self):
        layer = self.make_layer()
        low = make_state({"player.hp_frac": 0.3})
        assert layer.decide(low, at=1000.0) is not None
        assert layer.decide(low, at=1001.0) is None  # inside the 3s cooldown
        assert layer.decide(low, at=1004.0) is not None

    def test_missing_hp_field_fires_nothing(self):
        # An unreadable HP bar must make the player do less, not dodge randomly.
        layer = self.make_layer()
        assert layer.decide(make_state({}), at=1000.0) is None
