"""FantCraft profile tests. Headless, like everything else in the suite.

The manifest fixtures mirror what the game's playtest bridge actually posts
(fancraft client/src/dev/playtestBridge.ts) — if that shape changes over there,
these tests are the tripwire on this side.
"""

from __future__ import annotations

import json

import pytest

from games.fancraft import FancraftConfig, build, config_from_dict
from games.fancraft.combat import GCD_RECAST_S, keys_from_layout
from games.fancraft.layout import HP_GREEN, MP_BLUE, Layout, parse_hex
from games.fancraft.reflexes import build_reflexes
from games.fancraft.sensors import build_bundle
from player.geometry import Geometry, Rect
from player.capture.source import Frame
from player.policy.rotation import RotationPlanner
from player.state import WorldState

from conftest import make_state

import numpy as np


MANIFEST = {
    "generatedAt": 1756100000000,
    "viewport": {"w": 1920, "h": 1080, "dpr": 1},
    "colors": {"hpHigh": "#2ecc71", "hpLow": "#c0392b", "mp": "#3498db", "mpDark": "#2980b9"},
    "regions": {
        "playerHpBar": {"x": 0.010, "y": 0.050, "w": 0.115, "h": 0.015},
        "playerMpBar": {"x": 0.010, "y": 0.070, "w": 0.115, "h": 0.015},
        "targetFrame": {"x": 0.008, "y": 0.100, "w": 0.104, "h": 0.055},
        "targetHpBar": {"x": 0.010, "y": 0.120, "w": 0.099, "h": 0.015},
        "hotbar": {"x": 0.330, "y": 0.930, "w": 0.340, "h": 0.055},
    },
    "hotbarSlots": [
        {"key": "1", "abilityId": "iron_cleave", "abilityName": "Iron Cleave",
         "region": {"x": 0.330, "y": 0.930, "w": 0.030, "h": 0.055}},
        {"key": "2", "abilityId": "bulwark_slash", "abilityName": "Bulwark Slash",
         "region": {"x": 0.365, "y": 0.930, "w": 0.030, "h": 0.055}},
        {"key": "6", "abilityId": "shield_bash", "abilityName": "Shield Bash",
         "region": {"x": 0.505, "y": 0.930, "w": 0.030, "h": 0.055}},
        {"key": "7", "abilityId": None, "abilityName": None,
         "region": {"x": 0.540, "y": 0.930, "w": 0.030, "h": 0.055}},
    ],
}


@pytest.fixture()
def manifest_path(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(MANIFEST), encoding="utf-8")
    return str(p)


# -- layout / manifest -------------------------------------------------------------


class TestLayout:
    def test_parse_hex(self):
        assert parse_hex("#2ecc71", (0, 0, 0)) == (46, 204, 113)
        assert parse_hex("2ecc71", (0, 0, 0)) == (46, 204, 113)
        assert parse_hex("#fff", (0, 0, 0)) == (255, 255, 255)
        assert parse_hex("nonsense", (1, 2, 3)) == (1, 2, 3)

    def test_defaults_without_manifest(self):
        layout = Layout.load_or_default(None)
        assert layout.source == "defaults"
        assert layout.hp_color == HP_GREEN
        assert layout.mp_color == MP_BLUE
        assert layout.hotbar_slots == []

    def test_manifest_round_trip(self, manifest_path):
        layout = Layout.from_manifest(manifest_path)
        assert layout.source == manifest_path
        assert layout.player_hp.x == pytest.approx(0.010)
        assert layout.hp_color == (46, 204, 113)
        # Empty slots survive as geometry but carry no ability
        assert len(layout.hotbar_slots) == 4
        assert layout.slot_for_ability("iron_cleave").key == "1"
        assert layout.slot_for_ability("nothing") is None

    def test_missing_manifest_falls_back(self, tmp_path):
        layout = Layout.load_or_default(str(tmp_path / "absent.json"))
        assert layout.source == "defaults"


# -- keybinds from the manifest ----------------------------------------------------


class TestKeys:
    def test_manifest_keys_win(self, manifest_path):
        layout = Layout.from_manifest(manifest_path)
        keys = keys_from_layout(layout, "guardian")
        assert keys["iron_cleave"] == "1"
        assert keys["shield_bash"] == "6"

    def test_fallback_order_without_manifest(self):
        keys = keys_from_layout(Layout(), "guardian")
        assert keys["iron_cleave"] == "1"
        assert keys["shield_bash"] == "6"

    def test_unknown_job_yields_empty_fallback(self):
        assert keys_from_layout(Layout(), "necromancer") == {}


# -- profile assembly --------------------------------------------------------------


class TestProfile:
    def test_builds_and_validates(self):
        profile = build()
        assert profile.validate() == []
        assert profile.name == "fancraft"
        assert profile.window_title == "FantCraft"

    def test_all_rotations_present(self):
        profile = build()
        assert set(profile.rotations) == {
            "fancraft.pipeline",
            "fancraft.guardian",
            "fancraft.luminary",
        }

    def test_builds_from_manifest(self, manifest_path):
        profile = build(FancraftConfig(manifest_path=manifest_path))
        assert profile.validate() == []
        assert profile.rotations["fancraft.pipeline"].by_id("iron_cleave").key == "1"

    def test_key_override_is_merged_not_replaced(self):
        profile = build(FancraftConfig(keys={"iron_cleave": "q"}))
        assert profile.rotations["fancraft.pipeline"].by_id("iron_cleave").key == "q"

    def test_config_from_dict(self):
        config = config_from_dict(
            {"job": "luminary", "rotation": "fancraft.luminary", "retreat_hp_threshold": 0.3}
        )
        assert config.job == "luminary"
        assert config.rotation == "fancraft.luminary"
        assert config.retreat_hp_threshold == pytest.approx(0.3)


# -- sensors over synthetic pixels -------------------------------------------------


def render_hud(layout: Layout, hp: float, mp: float, target_hp: float | None) -> np.ndarray:
    """A crude FantCraft-shaped HUD: bars with number text punched over them."""
    img = np.full((1080, 1920, 3), 30, dtype=np.uint8)

    def draw_bar(rel, color, frac):
        r = rel.to_client(1920, 1080)
        img[r.y : r.bottom, r.x : r.right] = (20, 20, 24)
        filled = int(round(r.w * frac))
        if filled > 0:
            img[r.y : r.bottom, r.x : r.x + filled] = color
        # Fake "100 / 100" glyphs: white columns punched through the middle
        mid = r.y + r.h // 2
        for gx in range(r.x + r.w // 3, r.x + 2 * r.w // 3, 4):
            img[max(0, mid - 1) : mid + 1, gx : gx + 1] = (255, 255, 255)

    draw_bar(layout.player_hp, layout.hp_color, hp)
    draw_bar(layout.player_mp, layout.mp_color, mp)
    if target_hp is not None:
        draw_bar(layout.target_hp, layout.hp_color, target_hp)
    return img


class TestSensors:
    def observe(self, layout: Layout, img: np.ndarray) -> WorldState:
        bundle = build_bundle(layout, {"iron_cleave": GCD_RECAST_S})
        geo = Geometry(Rect(0, 0, 1920, 1080), (1920, 1080))
        state = WorldState(tick=1, captured_at=1.0, perceived_at=1.0)
        frame = Frame(index=1, image=img, captured_at=1.0, client_rect=Rect(0, 0, 1920, 1080))
        bundle.observe(1, frame, geo, state)
        return state

    def test_reads_vitals_through_text_overlay(self, manifest_path):
        layout = Layout.from_manifest(manifest_path)
        state = self.observe(layout, render_hud(layout, hp=0.75, mp=0.40, target_hp=0.60))
        assert state.num("player.hp_frac") == pytest.approx(0.75, abs=0.06)
        assert state.num("player.mp_frac") == pytest.approx(0.40, abs=0.06)
        assert state.num("target.hp_frac") == pytest.approx(0.60, abs=0.06)
        assert state.flag("target.exists") is True

    def test_hidden_target_frame_reads_as_no_target(self, manifest_path):
        layout = Layout.from_manifest(manifest_path)
        state = self.observe(layout, render_hud(layout, hp=1.0, mp=1.0, target_hp=None))
        assert state.flag("target.exists") is False
        # And the zero reading is below the trust gate, per the safety policy
        f = state.field("target.hp_frac")
        assert f is not None and f.confidence <= 0.45


class TestAdaptiveCooldownProbe:
    def make_slot(self, covered_frac: float, base: int = 80) -> "np.ndarray":
        """A 48x48 slot: bright ready art, top `covered_frac` swept dark."""
        img = np.full((48, 48, 3), base, dtype=np.uint8)
        covered = int(round(48 * covered_frac))
        if covered > 0:
            img[:covered, :] = int(base * 0.28)  # the 72%-black sweep
        return img

    def probe(self):
        from games.fancraft.sensors import AdaptiveCooldownProbe
        from player.geometry import RelRect

        return AdaptiveCooldownProbe("action.x.progress", RelRect(0, 0, 0.1, 0.1))

    def test_ready_slot_reads_zero_after_calibration(self):
        p = self.probe()
        r = p.read(self.make_slot(0.0))
        assert r.value == pytest.approx(0.0, abs=0.05)

    def test_sweep_fraction_tracks_coverage(self):
        p = self.probe()
        p.read(self.make_slot(0.0))  # calibrate on the ready state
        r = p.read(self.make_slot(0.6))
        assert r.value == pytest.approx(0.6, abs=0.08)

    def test_never_bright_reads_unreadable_not_on_cooldown(self):
        p = self.probe()
        r = p.read(self.make_slot(0.0, base=15))  # occluded/dark slot
        assert r.value is None
        assert r.confidence == 0.0


# -- rotations ---------------------------------------------------------------------


class TestRotations:
    def test_pipeline_holds_without_target(self):
        profile = build()
        planner = RotationPlanner(profile.rotations["fancraft.pipeline"])
        state = make_state({"player.hp_frac": 1.0, "target.exists": False})
        assert planner.decide(state, at=10.0) is None

    def test_pipeline_presses_with_target(self):
        profile = build()
        planner = RotationPlanner(profile.rotations["fancraft.pipeline"])
        state = make_state({"player.hp_frac": 1.0, "target.exists": True})
        decision = planner.decide(state, at=10.0)
        assert decision is not None
        assert decision.gcd.id == "iron_cleave"

    def test_luminary_heals_before_damaging(self):
        profile = build(FancraftConfig(job="luminary"))
        planner = RotationPlanner(profile.rotations["fancraft.luminary"])
        state = make_state(
            {"player.hp_frac": 0.4, "player.mp_frac": 0.8, "target.exists": True}
        )
        decision = planner.decide(state, at=10.0)
        assert decision is not None
        assert decision.gcd.id == "mending_light"

    def test_luminary_damages_when_healthy(self):
        profile = build(FancraftConfig(job="luminary"))
        planner = RotationPlanner(profile.rotations["fancraft.luminary"])
        state = make_state(
            {"player.hp_frac": 0.95, "player.mp_frac": 0.8, "target.exists": True}
        )
        decision = planner.decide(state, at=10.0)
        assert decision is not None
        assert decision.gcd.id == "starfire"

    def test_luminary_conserves_last_mana(self):
        profile = build(FancraftConfig(job="luminary"))
        planner = RotationPlanner(profile.rotations["fancraft.luminary"])
        state = make_state(
            {"player.hp_frac": 0.95, "player.mp_frac": 0.05, "target.exists": True}
        )
        assert planner.decide(state, at=10.0) is None


# -- reflexes ----------------------------------------------------------------------


class TestReflexes:
    def keys(self):
        return {"target_cycle": "tab", "back": "s", "sprint": "shift"}

    def test_acquires_target_when_none(self):
        layer = build_reflexes(self.keys())
        state = make_state({"player.hp_frac": 1.0, "target.exists": False})
        fired = layer.decide(state, at=10.0)
        assert fired is not None
        assert fired[0].id == "acquire_target"

    def test_no_tab_while_target_held(self):
        layer = build_reflexes(self.keys())
        state = make_state({"player.hp_frac": 1.0, "target.exists": True})
        assert layer.decide(state, at=10.0) is None

    def test_retreat_beats_targeting(self):
        layer = build_reflexes(self.keys())
        state = make_state({"player.hp_frac": 0.1, "target.exists": False})
        fired = layer.decide(state, at=10.0)
        assert fired is not None
        assert fired[0].id == "retreat_low_hp"
