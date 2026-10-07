"""API control, headless: a fake gateway, the FantCraft field extractor, the
intent backend, and the Warden tactics over synthetic snapshots.

No sockets. `FakeGateway` has the same surface `ApiBackend` and the key
actions use (`fire`, `latest`, `listeners`), and records every op sent.
"""

from __future__ import annotations

import math

import pytest

from games.fancraft import FancraftConfig, build
from games.fancraft.api import (
    FightLedger,
    build_key_actions,
    cover_cells,
    extract_fields,
)
from games.fancraft.tasks import HELPER_TASKS, task as helper_task
from games.fancraft.warden import PARRY_HOLD_MS, PARRY_LEAD_MS, WardenMemory, WardenTuning, build_warden_reflexes
from player.act.timeline import KeyDown, KeyUp, Press
from player.api import ApiBackend, ApiFrame, SnapshotSensor
from player.api.gateway import Event, Snapshot
from player.geometry import Rect
from player.perceive.pipeline import PerceptionPipeline
from player.state import WorldState

import numpy as np


# -- fixtures ---------------------------------------------------------------------


class FakeGateway:
    url = "ws://fake"

    def __init__(self, snapshot: dict | None = None) -> None:
        self.sent: list[tuple[str, dict]] = []
        self.snapshot = snapshot
        self.listeners: list = []
        self.connected = True

    def fire(self, op: str, **args) -> None:
        self.sent.append((op, args))

    def call(self, op: str, timeout_s: float = 30.0, **args) -> dict:
        self.sent.append((op, args))
        return {"ev": "ack", "op": op}

    def latest(self) -> Snapshot | None:
        return None if self.snapshot is None else Snapshot(data=self.snapshot, received_at=1000.0, seq=1)

    def ops(self, op: str) -> list[dict]:
        return [a for o, a in self.sent if o == op]

    def stats(self) -> dict:
        return {}


def yaw_toward(fx: float, fz: float, tx: float, tz: float) -> float:
    return math.atan2(-(tx - fx), -(tz - fz))


def make_snapshot(*, me=(64.0, 67.0, 70.0), boss=(64.0, 67.0, 60.0), hp=150, telegraph: dict | None = None,
                  phase="p1", horse=None, markers=(), skills=None, unlocked=("provoke",), detail="perfect:-",
                  attempt=0, remaining_ms=0, stock=64, downed=False, opening=None, recent: dict | None = None,
                  stuck_ms=0, inventory=(), players=()) -> dict:
    self_yaw = yaw_toward(me[0], me[2], boss[0], boss[2])
    entities = [
        {"entityId": 1, "type": "mob", "name": "Ironclad Warden", "mobId": "ironclad_warden", "x": boss[0], "y": boss[1], "z": boss[2],
         "rx": yaw_toward(boss[0], boss[2], me[0], me[2]), "hp": 2800, "maxHp": 2800, "alive": True,
         "distance": math.hypot(boss[0] - me[0], boss[2] - me[2]), "opening": opening},
    ]
    if horse is not None:
        entities.append({"entityId": 3, "type": "mob", "name": "Mechanical Steed", "mobId": "mechanical_horse", "x": horse[0], "y": 67.0, "z": horse[1],
                         "rx": 0.0, "hp": 999_999, "maxHp": 999_999, "alive": True, "distance": math.hypot(horse[0] - me[0], horse[1] - me[2]), "opening": None})
    for i, (px, pz) in enumerate(players):
        entities.append({"entityId": 10 + i, "type": "player", "name": f"Ally{i}", "mobId": None, "x": px, "y": 67.0, "z": pz, "rx": 0.0,
                         "hp": 150, "maxHp": 150, "alive": True, "distance": math.hypot(px - me[0], pz - me[2]), "opening": None})
    telegraphs = []
    if telegraph is not None:
        t = {"entityId": 1, "attackId": "warden_draw", "name": "Iai Draw", "windupMs": 1700, "startedAt": 1_000_000, "landsAt": 1_001_700,
             "landInMs": 1200, "yaw": yaw_toward(boss[0], boss[2], me[0], me[2]), "weight": 0.45, "medium": "physical",
             "shape": {"kind": "melee", "reach": 5.1, "arcDeg": 120}, "targetId": 2, "landed": False, "hits": []}
        t.update(telegraph)
        telegraphs.append(t)
    if recent is not None:
        r = {"entityId": 1, "attackId": "warden_overhead2", "name": "Great Serpent", "windupMs": 1100, "startedAt": 999_000, "landsAt": 1_000_100,
             "landInMs": -400, "yaw": 0.0, "weight": 0.9, "medium": "physical", "shape": {"kind": "melee", "reach": 4.9, "arcDeg": 60},
             "targetId": 2, "landed": True, "landedAgoMs": 400, "hits": []}
        r.update(recent)
        telegraphs.append(r)
    return {
        "ev": "snapshot", "t": 1_000_500, "zone": "trial_warden", "seq": 10,
        "self": {"entityId": 2, "x": me[0], "y": me[1], "z": me[2], "yaw": self_yaw, "hp": hp, "maxHp": 150, "mana": 1000, "maxMana": 1000,
                 "guarding": False, "downed": downed, "downedRemainingMs": 0, "jobId": "guardian", "coins": 0, "moveMul": 1,
                 "buildStock": stock, "facing": 1, "held": {"forward": 0, "strafe": 0, "sprint": False, "jump": False},
                 "speed": 0.0, "stuckMs": stuck_ms},
        "entities": entities, "telegraphs": telegraphs, "markers": list(markers),
        "boss": {"entityId": 1, "phase": phase, "detail": detail, "mastered": False, "killLocked": True, "attempt": attempt, "remainingMs": remaining_ms},
        "skills": skills if skills is not None else {"provoke": {"cooldownMs": 30000, "readyInMs": 0}},
        "unlocked": list(unlocked),
        "inventory": [{"slot": i, "itemId": item, "quantity": qty} for i, (item, qty) in enumerate(inventory)],
    }


def state_of(snapshot: dict, at: float = 1000.0) -> WorldState:
    state = WorldState(tick=1, captured_at=at, perceived_at=at)
    for name, (value, conf) in extract_fields(snapshot).items():
        state.set(name, value, confidence=conf, source="test")
    return state


def presses(plan) -> list[tuple[int, str, int]]:
    out = []
    for step in plan.steps:
        a = step.action
        if isinstance(a, Press):
            out.append((step.at_ms, a.key, a.hold_ms))
        elif isinstance(a, KeyDown):
            out.append((step.at_ms, f"down:{a.key}", 0))
        elif isinstance(a, KeyUp):
            out.append((step.at_ms, f"up:{a.key}", 0))
    return out


# -- fields ---------------------------------------------------------------------------


class TestExtractFields:
    def test_vitals_and_geometry(self):
        state = state_of(make_snapshot(hp=90))
        assert state.num("player.hp_frac") == pytest.approx(0.6)
        assert state.flag("boss.alive") and state.num("boss.distance_m") == pytest.approx(10.0)
        assert state.text("boss.phase") == "p1"
        assert state.text("target.role") == "boss"
        assert not state.flag("attack.active")
        assert state.flag("skill.provoke.ready")

    def test_telegraph_in_arc_is_a_threat_with_a_countdown(self):
        state = state_of(make_snapshot(me=(64.0, 67.0, 63.0), telegraph={"landInMs": 900}))
        assert state.flag("attack.active") and state.flag("attack.threatens") and state.flag("attack.in_arc")
        assert state.flag("attack.guardable")
        assert state.num("attack.land_in_ms") == 900
        assert state.text("attack.role") == "boss"

    def test_telegraph_out_of_reach_does_not_threaten(self):
        state = state_of(make_snapshot(me=(64.0, 67.0, 72.0), telegraph={}))  # 12 m out, reach 5.1
        assert state.flag("attack.active")
        assert not state.flag("attack.threatens")

    def test_shield_only_palm_is_not_guardable(self):
        state = state_of(make_snapshot(me=(64.0, 67.0, 62.0), telegraph={"attackId": "warden_palm", "shape": {"kind": "melee", "reach": 3.1, "arcDeg": 50}}))
        assert state.flag("attack.threatens")
        assert not state.flag("attack.guardable")

    def test_dodge_key_moves_away_from_the_swing_line(self):
        # The boss faces +z (toward us); his right hand is -x. Standing at +x we are on his
        # left, so the way out is further +x — which, facing -z ourselves, is our right: 'd'.
        snap = make_snapshot(me=(65.0, 67.0, 63.0), telegraph={"yaw": yaw_toward(64.0, 60.0, 64.0, 70.0)})
        assert state_of(snap).text("attack.dodge_key") == "d"
        snap = make_snapshot(me=(63.0, 67.0, 63.0), telegraph={"yaw": yaw_toward(64.0, 60.0, 64.0, 70.0)})
        assert state_of(snap).text("attack.dodge_key") == "a"

    def test_marker_fields(self):
        markers = [{"markerId": "orochi_1", "markerType": "circle_aoe", "position": {"x": 64.0, "y": 67.0, "z": 69.0}, "radius": 1.5, "remainingMs": 300, "color": "#fff"}]
        state = state_of(make_snapshot(markers=markers))
        assert state.num("marker.nearest_m") == pytest.approx(1.0 - 1.5)
        assert state.num("marker.radius_m") == 1.5
        # marker is 1 m toward the boss (we face him): away is straight back
        assert state.num("marker.away_forward") < -0.9

    def test_snapshot_sensor_through_the_pipeline(self):
        bundle_state = PerceptionPipeline(build(FancraftConfig(control="api")).sensors).process(
            ApiFrame(image=np.zeros((2, 2, 3), dtype=np.uint8), captured_at=5.0, index=1, client_rect=Rect(0, 0, 2, 2), snapshot=make_snapshot())
        )
        assert bundle_state.num("player.hp_frac") == 1.0
        assert bundle_state.field("player.x").confidence == pytest.approx(0.9)


# -- backend ---------------------------------------------------------------------------


class TestApiBackend:
    def test_movement_keys_fold_into_one_vector(self):
        gw = FakeGateway()
        backend = ApiBackend(gw)
        backend.key_down("w")
        backend.key_down("d")
        backend.key_down("shift")
        backend.key_up("w")
        backend.key_up("d")
        backend.key_up("shift")
        vectors = [(m["forward"], m["strafe"], m["sprint"]) for m in gw.ops("move")]
        assert vectors == [(1.0, 0.0, False), (1.0, 1.0, False), (1.0, 1.0, True), (0.0, 1.0, True), (0.0, 0.0, True), (0.0, 0.0, False)]

    def test_guard_is_lowered_then_raised_on_press(self):
        gw = FakeGateway()
        backend = ApiBackend(gw)
        backend.key_down("mouse2")
        backend.key_up("mouse2")
        assert [g["active"] for g in gw.ops("guard")] == [False, True, False]

    def test_overlapping_guard_holds_release_once(self):
        gw = FakeGateway()
        backend = ApiBackend(gw)
        backend.key_down("mouse2")   # parry A
        backend.key_down("mouse2")   # parry B, re-raised for its own window
        backend.key_up("mouse2")     # A releases: B still holds
        assert [g["active"] for g in gw.ops("guard")] == [False, True, False, True]
        backend.key_up("mouse2")
        assert [g["active"] for g in gw.ops("guard")][-1] is False

    def test_overlapping_movement_holds_release_with_the_last(self):
        gw = FakeGateway()
        backend = ApiBackend(gw)
        backend.key_down("d")   # dodge A
        backend.key_down("d")   # dodge B re-fired while A still holds
        backend.key_up("d")     # A releases: B must keep us moving
        assert gw.ops("move")[-1]["strafe"] == 1.0 and "d" in backend.held
        backend.key_up("d")
        assert gw.ops("move")[-1]["strafe"] == 0.0 and "d" not in backend.held

    def test_flee_holds_a_world_frame_vector(self):
        gw = FakeGateway()
        backend = ApiBackend(gw)
        backend.key_down("flee:7")
        backend.key_down("shift")
        backend.key_up("flee:7")
        moves = gw.ops("move")
        assert moves[0] == {"forward": 0.0, "strafe": 0.0, "sprint": False, "awayFrom": 7}
        assert moves[1] == {"forward": 0.0, "strafe": 0.0, "sprint": True, "awayFrom": 7}
        assert moves[2] == {"forward": 0.0, "strafe": 0.0, "sprint": True}

    def test_strike_and_unknown_keys(self):
        gw = FakeGateway()
        backend = ApiBackend(gw)
        backend.key_down("mouse1")
        backend.key_down("ctrl+9")
        assert len(gw.ops("attack")) == 1
        assert backend.unknown_keys == {"ctrl+9": 1}

    def test_release_all_clears_holds(self):
        gw = FakeGateway()
        backend = ApiBackend(gw)
        backend.key_down("w")
        backend.release_all()
        assert gw.ops("move")[-1] == {"forward": 0, "strafe": 0, "sprint": False}
        assert backend.held == set()

    def test_fancraft_intents_resolve_roles(self):
        gw = FakeGateway(make_snapshot(horse=(80.0, 60.0)))
        backend = ApiBackend(gw, build_key_actions(gw))
        backend.key_down("face:boss")
        backend.key_down("skill:provoke@horse")
        backend.key_down("tab")
        assert gw.ops("face")[0] == {"entityId": 1}
        assert gw.ops("skill")[0] == {"skillId": "provoke", "targetId": 3}
        assert gw.ops("face")[1] == {"entityId": 1}  # nearest living mob is still the boss

    def test_cover_builds_a_wall_toward_the_boss(self):
        gw = FakeGateway(make_snapshot(me=(64.2, 67.0, 70.4), boss=(64.0, 67.0, 48.0)))
        backend = ApiBackend(gw, build_key_actions(gw))
        backend.key_down("cover")
        placed = gw.ops("place")
        assert len(placed) == 9
        zs = {p["z"] for p in placed}
        xs = sorted({p["x"] for p in placed})
        ys = sorted({p["y"] for p in placed})
        assert zs == {68}  # between us (z 70.4) and him (z 48): one block out
        assert xs == [63, 64, 65] and ys == [67, 68, 69]

    def test_cover_cells_stand_on_our_floor(self):
        cells = cover_cells({"x": 10.5, "y": 67.0, "z": 10.5}, {"x": 30.0, "z": 10.5})
        assert all(y in (67, 68, 69) for _, y, _ in cells)
        assert all(x == 12 for x, _, _ in cells)


# -- the Warden -------------------------------------------------------------------------


class TestWardenTactics:
    def decide(self, layer, snapshot, at=1000.0):
        fired = layer.decide(state_of(snapshot, at), at)
        return fired if fired is None else (fired[0].id, fired[1])

    def test_a_base_cut_is_parried_just_before_it_lands(self):
        layer = build_warden_reflexes()
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 63.0), telegraph={"landInMs": 1200}))
        assert rid == "parry"
        p = presses(plan)
        assert (1200 - PARRY_LEAD_MS, "down:mouse2", 0) in p
        assert (1200 - PARRY_LEAD_MS + PARRY_HOLD_MS, "up:mouse2", 0) in p
        assert plan.preempt and plan.expires_in_ms > 1200

    def test_each_telegraph_is_answered_once(self):
        layer = build_warden_reflexes()
        snap = make_snapshot(me=(64.0, 67.0, 63.0), telegraph={"landInMs": 1200})
        assert self.decide(layer, snap) is not None
        # same telegraph (same startedAt) a tick later: no second plan, nothing else fires either
        snap["telegraphs"][0]["landInMs"] = 1100
        assert self.decide(layer, snap, at=1000.1) is None
        # a new swing is a new answer
        snap["telegraphs"][0]["startedAt"] = 1_000_900
        assert self.decide(layer, snap, at=1001.0)[0] == "parry"

    def test_the_palm_is_dodged_not_guarded(self):
        layer = build_warden_reflexes()
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 62.0), telegraph={
            "attackId": "warden_palm", "landInMs": 450, "shape": {"kind": "melee", "reach": 3.1, "arcDeg": 50}, "weight": 0.95}))
        assert rid == "dodge_unguardable"
        keys = {k for _, k, _ in presses(plan)}
        assert "shift" in keys and "s" in keys and ("a" in keys or "d" in keys)
        assert "down:mouse2" not in keys

    def test_cloud_seize_keeps_strafing_through_the_cloud(self):
        layer = build_warden_reflexes()
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 62.0), telegraph={
            "attackId": "warden_palm2", "landInMs": 400, "shape": {"kind": "melee", "reach": 3.1, "arcDeg": 50}, "weight": 0.95}))
        assert rid == "dodge_unguardable"
        strafes = [(at, k, hold) for at, k, hold in presses(plan) if k in ("a", "d")]
        assert strafes and strafes[0][2] == 400 + 1500

    def test_follow_up_blow_keeps_the_strafe_direction(self):
        layer = build_warden_reflexes()
        seize = make_snapshot(me=(63.0, 67.0, 62.0), telegraph={
            "attackId": "warden_palm2", "landInMs": 400, "shape": {"kind": "melee", "reach": 3.1, "arcDeg": 50}, "weight": 0.95})
        _, plan = self.decide(layer, seize)
        first = {k for _, k, _ in presses(plan) if k in ("a", "d")}
        # the cloud is re-aimed at us: on its line, the geometric pick could flip — it must not
        cloud = make_snapshot(me=(64.0, 67.0, 62.0), telegraph={
            "attackId": "warden_cloud", "landInMs": 600, "startedAt": 1_000_900, "shape": {"kind": "melee", "reach": 8.0, "arcDeg": 70}, "weight": 0.9,
            "yaw": yaw_toward(64.0, 60.0, 64.0, 62.0)})
        _, plan = self.decide(layer, cloud, at=1000.9)
        second = {k for _, k, _ in presses(plan) if k in ("a", "d")}
        assert first == second
        assert "flee:1" in {k for _, k, _ in presses(plan)}  # and out of its eight metres

    def test_wind_cutter_is_evaded_even_from_far_away(self):
        layer = build_warden_reflexes()
        # 12 m out: the draw itself cannot reach, the wave that follows can — orbit off its line
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 72.0), telegraph={"attackId": "warden_draw2", "landInMs": 1500, "weight": 0.6}))
        assert rid == "evade_far_reaching"
        p = presses(plan)
        assert any(k in ("a", "d") and at == 0 for at, k, _ in p)
        assert not any(k == "down:mouse2" for _, k, _ in p)

    def test_a_horse_bite_is_parried_facing_the_horse(self):
        layer = build_warden_reflexes()
        snap = make_snapshot(me=(64.0, 67.0, 66.0), horse=(66.5, 66.0), telegraph={
            "entityId": 3, "attackId": "horse_bite", "landInMs": 540, "weight": 0.55,
            "yaw": yaw_toward(66.5, 66.0, 64.0, 66.0), "shape": {"kind": "melee", "reach": 3.0, "arcDeg": 70}})
        rid, plan = self.decide(layer, snap)
        assert rid == "parry"
        p = presses(plan)
        assert p[0] == (0, "face:3", 20)
        assert (540 - PARRY_LEAD_MS, "down:mouse2", 0) in p
        assert p[-1][1] == "face:boss" and p[-1][0] > 540 - PARRY_LEAD_MS + PARRY_HOLD_MS

    def test_great_serpent_is_parried_and_wind_cutter_is_left(self):
        layer = build_warden_reflexes()
        # From melee a 4.9 m blade cannot be outrun in its windup: the cut is parried, the lightning fled after
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 63.0), telegraph={"attackId": "warden_overhead2", "landInMs": 1000, "weight": 0.9}))
        assert rid == "parry"
        assert any(k == "down:mouse2" for _, k, _ in presses(plan))
        # Wind Cutter's wave is left before it lands
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 63.0), telegraph={"attackId": "warden_draw2", "landInMs": 1500, "weight": 0.6}), at=5.0)
        assert rid == "evade_far_reaching"
        p = presses(plan)
        assert (0, "flee:1", 650) in p and (0, "shift", 650) in p
        assert not any(k == "down:mouse2" for _, k, _ in p)
        # re-decided every half second while the telegraph is up: far out it orbits instead of backing
        snap = make_snapshot(me=(64.0, 67.0, 74.0), telegraph={"attackId": "warden_draw2", "landInMs": 400, "weight": 0.6})
        rid, plan = self.decide(layer, snap, at=1005.6)
        assert rid == "evade_far_reaching"
        keys = {k for _, k, _ in presses(plan)}
        assert "flee:1" not in keys and "s" not in keys and ("a" in keys or "d" in keys)

    def test_lightning_keeps_the_evade_going_after_the_cut_lands(self):
        layer = build_warden_reflexes()
        # the cut landed 400 ms ago and its telegraph is gone: no attack, but the lightning walks
        snap = make_snapshot(me=(64.0, 67.0, 68.0), recent={"landedAgoMs": 400})
        rid, plan = self.decide(layer, snap)
        assert rid == "evade_far_reaching"
        keys = {k for _, k, _ in presses(plan)}
        assert "flee:1" in keys
        # and we never walk back in while it does (the evade keeps re-firing instead)
        fired = self.decide(layer, make_snapshot(me=(64.0, 67.0, 72.0), recent={"landedAgoMs": 2000}), at=1001.0)
        assert fired is None or fired[0] == "evade_far_reaching"
        # four seconds on it is over: engagement resumes
        rid, _ = self.decide(layer, make_snapshot(me=(64.0, 67.0, 72.0), recent={"landedAgoMs": 4200}), at=1002.0)
        assert rid == "approach"

    def test_wedged_evade_turns_sideways(self):
        layer = build_warden_reflexes()
        snap = make_snapshot(me=(64.0, 67.0, 63.0), telegraph={"attackId": "warden_draw2", "landInMs": 900, "weight": 0.6}, stuck_ms=450)
        rid, plan = self.decide(layer, snap)
        assert rid == "evade_far_reaching"
        keys = {k for _, k, _ in presses(plan)}
        assert "flee:1" not in keys and len(keys & {"a", "d"}) == 1

    def test_evade_never_backs_into_the_wall(self):
        layer = build_warden_reflexes()
        # 3 m from the boss but 2 m from the dojo wall (radius 20 around 64,64): strafe only
        snap = make_snapshot(me=(64.0, 67.0, 82.0), boss=(64.0, 67.0, 79.0), telegraph={"attackId": "warden_draw2", "landInMs": 1500, "weight": 0.6})
        rid, plan = self.decide(layer, snap)
        assert rid == "evade_far_reaching"
        keys = {k for _, k, _ in presses(plan)}
        assert "s" not in keys and "flee:1" not in keys

    def test_marker_dodge_slides_along_the_wall(self):
        layer = build_warden_reflexes()
        markers = [{"markerId": "orochi_3", "markerType": "circle_aoe", "position": {"x": 64.0, "y": 67.0, "z": 81.0}, "radius": 1.5, "remainingMs": 400, "color": "#fff"}]
        # pinned at z=83 (wall at 84) facing the boss to the south: the marker is 2 m in front of us
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 83.0), boss=(64.0, 67.0, 60.0), markers=markers))
        assert rid == "leave_marker"
        keys = {k for _, k, _ in presses(plan)}
        assert "s" not in keys and "w" not in keys and ("a" in keys or "d" in keys)

    def test_dodge_tactic_sprints_out_instead(self):
        layer = build_warden_reflexes(tactic="dodge")
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 63.0), telegraph={"landInMs": 1200}))
        assert rid == "parry"
        keys = {k for _, k, _ in presses(plan)}
        assert "down:mouse2" not in keys and "s" in keys and "shift" in keys

    def test_out_of_reach_swing_is_ignored_and_we_wait(self):
        layer = build_warden_reflexes()
        # 12 m out with a telegraph up: not threatened, but never walk into a swing
        assert self.decide(layer, make_snapshot(me=(64.0, 67.0, 72.0), telegraph={})) is None

    def test_engagement_when_quiet(self):
        layer = build_warden_reflexes()
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 72.0)))
        assert rid == "approach" and ("shift" in {k for _, k, _ in presses(plan)})
        # never walk in while lightning is stepping nearby: leave instead
        markers = [{"markerId": "orochi_4", "markerType": "circle_aoe", "position": {"x": 64.0, "y": 67.0, "z": 68.0}, "radius": 1.5, "remainingMs": 400, "color": "#fff"}]
        fired = self.decide(layer, make_snapshot(me=(64.0, 67.0, 72.0), markers=markers), at=1005.0)
        assert fired is None or fired[0] == "leave_marker"
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 62.5), opening={"kind": "recovery", "remainingMs": 600}), at=1010.0)
        assert rid == "strike"
        rid, _ = self.decide(layer, make_snapshot(me=(64.0, 67.0, 61.2)), at=1020.0)
        assert rid == "back_off"

    def test_horse_is_provoked_when_ready(self):
        layer = build_warden_reflexes()
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 63.0), phase="p2", horse=(80.0, 60.0)))
        assert rid == "hold_horse"
        assert presses(plan)[0][1] == "skill:provoke@horse"
        # on cooldown: engagement instead
        snap = make_snapshot(me=(64.0, 67.0, 63.0), phase="p2", horse=(80.0, 60.0), skills={"provoke": {"cooldownMs": 30000, "readyInMs": 12000}})
        fired = self.decide(layer, snap, at=1010.0)
        assert fired is None or fired[0] != "hold_horse"

    def test_no_provoke_inside_a_landing_window(self):
        layer = build_warden_reflexes(memory=WardenMemory(answered={"1:warden_draw:1000000": 0.0}))
        snap = make_snapshot(me=(64.0, 67.0, 63.0), phase="p2", horse=(80.0, 60.0), telegraph={"landInMs": 500})
        assert self.decide(layer, snap) is None

    def test_lightning_marker_is_left_away_from_the_boss(self):
        layer = build_warden_reflexes()
        # the step is 4 m off and closing: leave now, straight away from the boss
        markers = [{"markerId": "orochi_2", "markerType": "circle_aoe", "position": {"x": 64.0, "y": 67.0, "z": 58.0}, "radius": 1.5, "remainingMs": 400, "color": "#fff"}]
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 63.5), markers=markers))
        assert rid == "leave_marker"
        keys = {k for _, k, _ in presses(plan)}
        assert "flee:1" in keys and "shift" in keys and "w" not in keys and "s" not in keys
        # six metres out is still quiet
        far = [{"markerId": "orochi_2", "markerType": "circle_aoe", "position": {"x": 64.0, "y": 67.0, "z": 56.0}, "radius": 1.5, "remainingMs": 400, "color": "#fff"}]
        fired = self.decide(layer, make_snapshot(me=(64.0, 67.0, 63.5), markers=far), at=1001.0)
        assert fired is None or fired[0] != "leave_marker"

    def test_pulse_marker_does_not_trigger_marker_dodging(self):
        layer = build_warden_reflexes()
        markers = [{"markerId": "apex_pulse_1", "markerType": "circle_aoe", "position": {"x": 64.0, "y": 67.0, "z": 60.0}, "radius": 128, "remainingMs": 11000, "color": "#f00"}]
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 63.0), phase="casting", attempt=1, remaining_ms=11000, markers=markers))
        assert rid == "pulse_retreat"
        rid, plan = self.decide(layer, make_snapshot(me=(64.0, 67.0, 70.0), phase="casting", attempt=1, remaining_ms=9000, markers=markers), at=1002.0)
        assert rid == "pulse_cover" and presses(plan)[0][1] == "cover"
        # covered once per attempt, then hold still
        assert self.decide(layer, make_snapshot(me=(64.0, 67.0, 70.0), phase="casting", attempt=1, remaining_ms=8000, markers=markers), at=1004.0) is None

    def test_nothing_fires_while_downed(self):
        layer = build_warden_reflexes()
        assert self.decide(layer, make_snapshot(me=(64.0, 67.0, 63.0), telegraph={"landInMs": 800}, downed=True)) is None

    def test_profile_validates(self):
        profile = build(FancraftConfig(control="api"))
        assert profile.validate() == []
        assert profile.transport is not None


# -- ledger ---------------------------------------------------------------------------------


class TestFightLedger:
    def test_counts_outcomes_and_phases(self):
        ledger = FightLedger(started_at=0.0, self_id=2)
        ev = lambda t, typ, data: Event(seq=1, type=typ, t_wall_ms=0.0, received_at=t, data=data)  # noqa: E731
        ledger.on_event(ev(1.0, "boss_challenge", {"phase": "p1", "detail": "perfect:-"}))
        ledger.on_event(ev(2.0, "mob_attack_start", {"attackId": "warden_draw"}))
        ledger.on_event(ev(3.7, "combat_action", {"sourceId": 1, "targetId": 2, "skillId": "warden_draw", "hitKind": "parried"}))
        ledger.on_event(ev(4.0, "mob_attack_start", {"attackId": "warden_mid"}))
        ledger.on_event(ev(4.8, "combat_action", {"sourceId": 1, "targetId": 0, "skillId": "warden_mid", "hitKind": "miss"}))
        ledger.on_event(ev(9.0, "boss_challenge", {"phase": "summon", "detail": ""}))
        ledger.on_event(ev(12.0, "player_downed", {"reason": "combat"}))
        assert ledger.outcomes == {"warden_draw": {"parried": 1}, "warden_mid": {"miss": 1}}
        assert [p for _, p, _ in ledger.phases] == ["p1", "summon"]
        assert ledger.deaths == 1
        text = "\n".join(ledger.report())
        assert "summon@9s" in text and "parried=1" in text


# -- tasks, the coin, the bottle -------------------------------------------------------


class TestTasksAndTuning:
    def test_dodge_chance_declines_a_share_of_far_reaching_cuts(self):
        rolls = iter([0.9, 0.1])  # first cut: declined (0.9 >= 0.5); second: taken
        tuning = WardenTuning(dodge_chance=0.5, random=lambda: next(rolls))
        memory = WardenMemory()
        layer = build_warden_reflexes(memory=memory, tuning=tuning)
        first = make_snapshot(me=(64.0, 67.0, 72.0), telegraph={"attackId": "warden_draw2", "landInMs": 1500, "weight": 0.6, "startedAt": 1_000_000})
        fired = layer.decide(state_of(first), 1000.0)
        # declined: no evade; from 12 m nothing else fires either (the draw cannot reach, we do not walk in)
        assert fired is None or fired[0].id != "evade_far_reaching"
        assert memory.declined == 1
        second = make_snapshot(me=(64.0, 67.0, 72.0), telegraph={"attackId": "warden_draw2", "landInMs": 1500, "weight": 0.6, "startedAt": 1_009_000})
        fired = layer.decide(state_of(second, 1009.0), 1009.0)
        assert fired is not None and fired[0].id == "evade_far_reaching"
        assert memory.stats()["dodge_rolls"] == 2

    def test_a_declined_serpent_takes_its_lightning(self):
        tuning = WardenTuning(dodge_chance=0.0)
        layer = build_warden_reflexes(memory=WardenMemory(), tuning=tuning)
        markers = [{"markerId": "orochi_9", "markerType": "circle_aoe", "position": {"x": 64.0, "y": 67.0, "z": 60.0}, "radius": 1.5, "remainingMs": 400, "color": "#fff"}]
        snap = make_snapshot(me=(64.0, 67.0, 63.5), markers=markers, recent={"landedAgoMs": 800})
        fired = layer.decide(state_of(snap), 1000.0)
        assert fired is None or fired[0].id not in ("leave_marker", "evade_far_reaching")

    def test_holy_water_is_thrown_at_the_kneel(self):
        layer = build_warden_reflexes()
        snap = make_snapshot(me=(64.0, 67.0, 70.0), phase="kneel", remaining_ms=5000, detail="holy:0/2000", inventory=(("holy_water_supreme", 3),))
        fired = layer.decide(state_of(snap), 1000.0)
        assert fired is not None and fired[0].id == "purify"
        assert presses(fired[1])[0][1] == "use:holy_water_supreme@boss"
        # without a bottle, nothing to throw
        dry = make_snapshot(me=(64.0, 67.0, 70.0), phase="kneel", remaining_ms=5000, detail="holy:0/2000")
        fired = layer.decide(state_of(dry, 1002.0), 1002.0)
        assert fired is None or fired[0].id != "purify"

    def test_use_key_throws_at_the_boss(self):
        gw = FakeGateway(make_snapshot())
        backend = ApiBackend(gw, build_key_actions(gw))
        backend.key_down("use:holy_water_supreme@boss")
        assert gw.ops("use") == [{"itemId": "holy_water_supreme", "targetId": 1}]

    def test_tasks_switch_reflex_groups(self):
        profile = build(FancraftConfig(control="api"))
        assert profile.transport.apply_task is not None
        assert "horse" in profile.reflexes.enabled_groups() and "engage" in profile.reflexes.enabled_groups()
        profile.transport.apply_task("hold_horse")
        enabled = profile.reflexes.enabled_groups()
        assert "horse" in enabled and "engage" not in enabled and "parry" in enabled
        profile.transport.apply_task("evade_only")
        assert "parry" not in profile.reflexes.enabled_groups()
        assert helper_task("nonsense").id == "full"
        assert set(HELPER_TASKS) == {"full", "standby", "tank", "hold_horse", "samurai", "evade_only", "purify", "cover", "companion"}

    def test_hold_horse_kites_the_horse_not_the_boss(self):
        layer = build_warden_reflexes()
        layer.set_group_enabled("engage", False)
        snap = make_snapshot(me=(64.0, 67.0, 70.0), phase="p2", horse=(72.0, 70.0), skills={"provoke": {"cooldownMs": 30000, "readyInMs": 20000}})
        fired = layer.decide(state_of(snap), 1000.0)
        assert fired is not None and fired[0].id == "face_horse"
        snap["self"]["facing"] = 3
        fired = layer.decide(state_of(snap, 1001.0), 1001.0)
        assert fired is not None and fired[0].id == "approach_horse"

    def test_standby_follows_the_requester(self):
        layer = build_warden_reflexes()
        for group in ("engage", "horse", "cover", "purify"):
            layer.set_group_enabled(group, False)
        layer.set_group_enabled("follow", True)
        snap = make_snapshot(me=(64.0, 67.0, 80.0), players=((64.0, 70.0),))
        fired = layer.decide(state_of(snap), 1000.0)
        assert fired is not None and fired[0].id == "follow_ally"
        assert presses(fired[1])[0][1] == "face:ally"

    def test_helper_events_drive_the_source(self):
        profile = build(FancraftConfig(control="api", helper={"name": "helper_1", "requester": "abc", "task": "standby"}))
        assert profile.parameters["helper.task"] == "standby"
        source = profile.transport.source
        gw = profile.transport.gateway
        for listener in list(gw.listeners):
            listener(Event(seq=1, type="helper_task", t_wall_ms=0.0, received_at=5.0, data={"task": "tank"}))
            listener(Event(seq=2, type="helper_follow", t_wall_ms=0.0, received_at=6.0, data={"roomId": "r9", "zone": "trial_warden"}))
        assert "engage" in profile.reflexes.enabled_groups() and "follow" not in profile.reflexes.enabled_groups()
        assert source._pending_room == ("trial_warden", "r9")
        for listener in list(gw.listeners):
            listener(Event(seq=3, type="helper_dismiss", t_wall_ms=0.0, received_at=7.0, data={}))
        assert source.exhausted


class TestParty:
    def test_party_members_are_profiles_with_their_own_tasks(self):
        from player.cli import build_game_profile
        from player.config import AppConfig

        data = {
            "game": "fancraft",
            "fancraft": {"control": "api", "username": "p", "zone": "trial_warden",
                         "party": [{"name": "samurai", "task": "samurai"}, {"name": "horse", "task": "hold_horse", "shopping": {}}]},
        }
        config = AppConfig.from_dict(data)
        members = config.game_config["party"]
        base = {k: v for k, v in config.game_config.items() if k != "party"}
        profiles = []
        for m in members:
            merged = {**base, **{k: v for k, v in m.items() if k != "name"}}
            merged["username"] = f"p_{m['name']}"
            profiles.append(build_game_profile(AppConfig.from_dict({"game": "fancraft", "fancraft": merged})))
        assert [p.parameters["helper.task"] for p in profiles] == ["samurai", "hold_horse"]
        assert "horse" not in profiles[0].reflexes.enabled_groups() and "purify" in profiles[0].reflexes.enabled_groups()
        assert "horse" in profiles[1].reflexes.enabled_groups() and "engage" not in profiles[1].reflexes.enabled_groups()
        assert profiles[1].transport.source.session.shopping == ()
        assert profiles[0].transport.source.session.shopping == (("holy_water_supreme", 3),)

    def test_party_members_get_their_own_accounts(self):
        from player.cli import party_member_usernames

        base = {"username": "warden_party", "control": "api"}
        members = [{"name": "samurai"}, {"name": "horse", "username": "rider"}]
        assert party_member_usernames(base, members) == ["warden_party_samurai", "rider"]


def test_samurai_guard_lead_stays_inside_the_parry_window():
    """Heavy strokes need a late press (fancraft PARRY_WINDOW_*); light ones never wait past 320 ms."""
    from games.fancraft.samurai import guard_lead_ms, PARRY_HEAVY_MS, PARRY_LIGHT_MS
    for weight in (0, .3, .6, .75, .9, 1):
        window = PARRY_LIGHT_MS + (PARRY_HEAVY_MS - PARRY_LIGHT_MS) * weight
        lead = guard_lead_ms(weight)
        assert 0 < lead <= 320 and lead < window
    assert guard_lead_ms(.9) < 150 and guard_lead_ms(1) < guard_lead_ms(.9)


def test_samurai_predicts_the_stalking_serpent_step():
    """The next strike lands one stride from the visible one, towards us; none far away is predicted wrongly."""
    from games.fancraft.samurai import SamuraiTactics
    marks = [{"markerId": "orochi_2", "skillId": "warden_orochi", "position": {"x": 60, "z": 60}, "expiresAt": 1450, "radius": 1.5}]
    near = SamuraiTactics().orochi_ahead(marks, 1000, (61, 60))
    assert [(round(h.x, 2), round(h.z, 2), h.at) for h in near] == [(61.0, 60.0, 1900)]
    far = SamuraiTactics().orochi_ahead(marks, 1000, (70, 60))
    assert [(round(h.x, 2), round(h.z, 2)) for h in far] == [(62.25, 60.0)]  # one 2.25 m stride
    assert SamuraiTactics().orochi_ahead([], 1000, (61, 60)) == []


def test_samurai_memory_route_keeps_each_set_whole():
    """Circles of one set announced a millisecond apart are still one layer: the route avoids all of them."""
    from games.fancraft.samurai import SamuraiTactics, Hazard
    hazards = [Hazard(f"a{i}", 4000 + i, 64 + dx, 64, 1.5, group=1) for i, dx in enumerate((-3, 0, 3))]
    tactics = SamuraiTactics()
    goal = tactics.memory_route((64., 60.), hazards, 0.)
    assert goal is not None and all(h.margin(goal) >= .12 for h in hazards)
    assert len(tactics.route) == 1


def test_samurai_steering_never_dodges_into_a_boss_body():
    """With a strike landing on us and the Samurai right ahead, the escape goes around, not through."""
    from games.fancraft.samurai import SamuraiTactics, Hazard
    tactics = SamuraiTactics()
    hazard = Hazard("orochi", 400, 64, 60, 1.5)
    direction = tactics.steer((64., 60.), (64., 58.), [hazard], 0., bodies=[((64., 58.), 1.7)])
    assert direction != (0., 0.)
    assert direction[1] > -.5  # not straight into the body at z 58


def test_samurai_outwalks_the_serpent_away_from_walls_and_bodies():
    """A strike just behind sends us away from it; never into the wall or into the Samurai."""
    from games.fancraft.samurai import SamuraiTactics, distance
    marker = lambda x, z: [{"markerId": "o", "skillId": "warden_orochi", "position": {"x": x, "z": z}, "remainingMs": 300}]
    away = SamuraiTactics().outwalk_serpent((64., 60.), marker(64, 61))
    assert away[1] < -.9  # straight away along -z
    assert SamuraiTactics().outwalk_serpent((64., 60.), marker(64, 70)) is None
    wall = SamuraiTactics().outwalk_serpent((64., 46.), marker(64, 47))  # 18 m out: outwards is the wall
    assert distance((64 + wall[0] * 2.25, 46 + wall[1] * 2.25), (64., 64.)) <= 19
    # Strike at our feet with the Samurai 1.4 m away along +x: never step into him.
    body = SamuraiTactics().outwalk_serpent((60., 60.), marker(60, 60), bodies=[((61.4, 60.), 1.7)])
    assert body[0] < 0


def test_samurai_never_swings_while_the_serpent_walks():
    """A parried Great Serpent opens no punish window: the walk follows and a swing would root us."""
    from games.fancraft.samurai import SamuraiTactics
    tactics = SamuraiTactics()
    tactics.last_landed[1] = 1000
    snap = {"self": {"x": 64, "z": 60}, "telegraphs": [], "markers": []}
    assert tactics.punish_window(snap, {"entityId": 1}, 1200)
    snap["markers"] = [{"skillId": "warden_orochi", "position": {"x": 70, "z": 70}, "remainingMs": 400}]
    assert not tactics.punish_window(snap, {"entityId": 1}, 1200)


def test_samurai_leaves_an_unguardable_sweep_outwards_not_into_the_samurai():
    """Standing 2.5 m in front of a 150°/5.1 m Wind Cutter, the escape heads out of the arc, never through him."""
    from games.fancraft.samurai import SamuraiTactics, Hazard
    cut = Hazard("warden_draw2", 1700, 64, 64, 5.1, yaw=math.pi, arc=150, kind="cone")  # facing +z
    me = (64., 66.5)
    d = SamuraiTactics().leave_sweep(me, [cut], 0., bodies=[((64., 64.), 1.7)])
    assert d is not None
    end = (me[0] + d[0] * 5 * 1.6, me[1] + d[1] * 5 * 1.6)
    assert cut.margin(end) > 0
    assert SamuraiTactics().leave_sweep((64., 75.), [cut], 0.) is None  # already outside


def test_samurai_circles_before_a_great_serpent_cut():
    """Inside the last 0.9 s of a Great Serpent windup we move sideways around him, not away or in."""
    from games.fancraft.samurai import SamuraiTactics
    tells = [{"attackId": "warden_overhead2", "entityId": 1, "landInMs": 600}]
    ents = [{"entityId": 1, "mobId": "ironclad_warden", "x": 64, "z": 64}]
    d = SamuraiTactics().circle_before_serpent((64., 66.5), tells, ents)
    assert d is not None and abs(d[1]) < .01  # tangent to the radius (+z)
    assert SamuraiTactics().circle_before_serpent((64., 66.5), [dict(tells[0], landInMs=1500)], ents) is None


def _tensei_snapshot(remaining, stage="timing", fp="dark", hand_ready=True):
    """The main target 3.4 m in front of the Warden during 天晴, as the gateway shows it."""
    return {
        "t": 10_000, "zone": "trial_warden", "roomId": "r",
        "self": {"entityId": 7, "x": 64.0, "y": 67.0, "z": 60.6},
        "entities": [{"entityId": 1, "mobId": "ironclad_warden", "type": "mob", "x": 64.0, "y": 67.0, "z": 64.0, "alive": True},
                     {"entityId": 7, "type": "player", "x": 64.0, "y": 67.0, "z": 60.6, "alive": True}],
        "boss": {"entityId": 1, "phase": "casting", "attempt": 1, "receivedAt": 10_000,
                 "samurai": {"finale": {"phase": fp, "castId": 1, "targetId": 7, "center": {"x": 64, "z": 64}, "radius": 18},
                             "tensei": {"stage": stage, "targetId": 7, "remainingMs": remaining}}},
        "telegraphs": [{"attackId": "warden_tensei", "entityId": 1, "landInMs": remaining, "landed": False, "weight": 1, "startedAt": 5000}],
        "markers": [], "unlocked": ["iron_cleave"], "skills": {"iron_cleave": {"readyInMs": 0}},
        "hands": {"left": {"hand": "left", "weight": .4, "windupMs": 180, "readyInMs": 0 if hand_ready else 400}},
    }


def test_samurai_tensei_opening_swing_aims_up_and_away_without_a_target_lock():
    """The opener releases inside the last 180 ms, along the learned up-and-away line, with no targetId
    (a locked target would aim the server's ray at his chest instead of the blade)."""
    from games.fancraft.samurai import SamuraiTactics, TENSEI_PRESS_LEAD, TENSEI_AIM_UP
    early = SamuraiTactics().decide(_tensei_snapshot(180 + TENSEI_PRESS_LEAD[1] + 60))
    assert not [c for c in early if c["op"] == "attack"]
    cmds = SamuraiTactics().decide(_tensei_snapshot(180 + 130))
    attacks = [c for c in cmds if c["op"] == "attack"]
    assert len(attacks) == 1 and "targetId" not in attacks[0]
    p = attacks[0]["aim"]["point"]
    assert p["z"] < 60.6 and abs(p["x"] - 64) < 1e-9          # away from him (he is at +z), on his facing line
    assert abs(p["y"] - (67 + TENSEI_AIM_UP)) < 1e-9
    assert attacks[0]["aim"]["pitch"] < -1                      # steeply up (gateway pitch: negative looks up)
    faces = [c for c in cmds if c["op"] == "face"]
    assert faces[-1].get("entityId") is None                    # the last face clears any entity lock


def test_samurai_tensei_clash_keeps_swinging_every_ready_hand_and_skill():
    from games.fancraft.samurai import SamuraiTactics
    tactics = SamuraiTactics()
    cmds = tactics.decide(_tensei_snapshot(500, stage="clash", fp="cut"))
    assert [c["op"] for c in cmds if c["op"] in ("attack", "skill")] == ["attack", "skill"]
    assert all("targetId" not in c for c in cmds if c["op"] in ("attack", "skill"))
    # A hand still recovering is not pressed; the cleave is not resent within 120 ms...
    again = tactics.decide(dict(_tensei_snapshot(450, stage="clash", fp="cut", hand_ready=False), t=10_050))
    assert not [c for c in again if c["op"] in ("attack", "skill")]
    # ...but is offered again while the server still refuses it (the swing's recovery locks the hands)
    retry = tactics.decide(dict(_tensei_snapshot(350, stage="clash", fp="cut", hand_ready=False), t=10_150))
    assert [c["skillId"] for c in retry if c["op"] == "skill"] == ["iron_cleave"]
    # A cleave-only batch still turns off the boss first: the gateway would otherwise lock him as its target
    before = retry[:[c["op"] for c in retry].index("skill")]
    assert [c for c in before if c["op"] == "face"][-1].get("entityId") is None
    spent = _tensei_snapshot(300, stage="clash", fp="cut", hand_ready=False)
    spent["skills"]["iron_cleave"]["readyInMs"] = 5000
    assert not [c for c in tactics.decide(dict(spent, t=10_300)) if c["op"] == "skill"]


def test_samurai_holds_still_in_every_bound_finale_phase():
    from games.fancraft.samurai import SamuraiTactics, ROOTED_PHASES
    for fp in ROOTED_PHASES:
        cmds = SamuraiTactics().decide(_tensei_snapshot(3000, stage="pressure", fp=fp))
        moves = [c for c in cmds if c["op"] == "move"]
        assert moves and all(m["forward"] == 0 and m["strafe"] == 0 for m in moves), fp
