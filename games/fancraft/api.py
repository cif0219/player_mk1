"""FantCraft over the agent gateway: snapshot → fields, keys → intents.

The gateway (fancraft `scripts/agent-gateway.mts`) is an ordinary game
client that publishes what it sees as JSON. This module is the FantCraft
half of API control: it names the fields a snapshot yields (the table below
is the contract the Warden tactics in `warden.py` are written against) and
defines the intent keys a plan may press.

| Field                       | Type      | Meaning                                              |
| --------------------------- | --------- | ---------------------------------------------------- |
| `player.hp_frac` / `mp_frac`| float     | own vitals (authoritative `entity_health`)           |
| `player.downed`             | bool      | lying where we fell; nothing dispatches              |
| `player.guarding`           | bool      | a guard is up (`guard_state`)                        |
| `player.x/y/z/yaw`          | float     | dead-reckoned between corrections                    |
| `target.exists/distance_m`  |           | the entity the gateway auto-faces                    |
| `boss.*`                    |           | the arena boss: phase machine, distance, opening     |
| `horse.*`                   |           | the Warden's mount, once summoned                    |
| `attack.*`                  |           | the soonest telegraph that can reach us (see below)  |
| `marker.*`                  |           | nearest ground marker (lightning, pulse)             |
| `skill.<id>.ready`          | bool      | recast tracked by the gateway                        |
| `build.stock`               | int       | trial cover blocks left                              |

`attack.*` is the heart of it: `land_in_ms` counts down to the engine's
landing tick, `in_arc` says whether the swing's arc covers us right now,
`guardable` whether a raised guard means anything against it (shield-only
palms and ethereal blows do not care), and `dodge_key` which strafe key
carries us out of the arc fastest.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from player.api import GatewayClient, KeyActions
from player.api.gateway import Event
from player.api.sensor import Reading, SnapshotSensor
from player.clock import now
from player.perceive.sensor import SensorBundle

BOSS_ID = "ironclad_warden"
HORSE_ID = "mechanical_horse"

# Blows a raised guard cannot answer without the right gear (docs/WORLD_BOSSES.md):
# the palm seizes are shield-only, and we fight bare-handed.
SHIELD_ONLY = frozenset({"warden_palm", "warden_palm2"})
# Blows the tactics dodge rather than parry: the projected cloud and the mounted
# charge are wide, fast and either unguardable or too heavy to gamble on.
DODGE_ONLY = frozenset({"warden_cloud", "warden_charge", "warden_wave"})
# Blows that threaten from any distance: Wind Cutter's wave rides its locked line
# for twenty metres, Great Serpent's lightning walks to wherever we stand.
ALWAYS_THREATENS = frozenset({"warden_draw2", "warden_overhead2"})

# Skills whose readiness is exposed as fields.
TRACKED_SKILLS = ("provoke", "shield_bash", "iron_cleave", "earthen_roar")
# Bag items whose counts are exposed as `bag.<id>`.
TRACKED_ITEMS = ("holy_water_supreme",)

ARC_MARGIN_DEG = 10.0
REACH_MARGIN_M = 0.9
PLAYER_RADIUS = 0.3

# Arena discs (server/src/world/zones.ts `trial()`, arenaBuilder ARENA_RADIUS): the
# wall is the whole world in a trial, and running into it is how a dodge fails.
ARENAS: dict[str, tuple[float, float, float]] = {
    "trial_warden": (64.0, 64.0, 20.0), "samurai_phase": (64.0, 64.0, 20.0),
    "trial_chieftain": (64.0, 64.0, 24.0), "trial_prowler": (64.0, 64.0, 22.0), "trial_nightfang": (64.0, 64.0, 24.0),
    "trial_wyrm": (64.0, 64.0, 32.0), "trial_maw": (64.0, 64.0, 36.0), "trial_coil": (64.0, 64.0, 30.0),
}


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def forward_of(yaw: float) -> tuple[float, float]:
    """Shared physics convention: forward = (-sin yaw, -cos yaw)."""
    return -math.sin(yaw), -math.cos(yaw)


def right_of(yaw: float) -> tuple[float, float]:
    return math.cos(yaw), -math.sin(yaw)


# -- snapshot → fields ------------------------------------------------------------


def _entity_by_mob(snapshot: Mapping[str, Any], mob_id: str) -> dict[str, Any] | None:
    best = None
    for e in snapshot.get("entities", []):
        if e.get("mobId") == mob_id and (best is None or (e.get("alive") and not best.get("alive"))):
            best = e
    return best


def _telegraph_fields(snapshot: Mapping[str, Any], me: Mapping[str, Any], boss: Mapping[str, Any] | None, horse: Mapping[str, Any] | None) -> dict[str, Reading]:
    out: dict[str, Reading] = {"attack.active": (False, 1.0), "attack.threatens": (False, 1.0)}
    px, pz, pyaw = float(me.get("x", 0.0)), float(me.get("z", 0.0)), float(me.get("yaw", 0.0))
    entities = {e["entityId"]: e for e in snapshot.get("entities", [])}
    candidates = []
    for t in snapshot.get("telegraphs", []):
        if t.get("landed"):
            continue
        src = entities.get(t.get("entityId"))
        if src is None:
            continue
        shape = t.get("shape") or {}
        kind = shape.get("kind", "melee")
        ayaw = float(t.get("yaw", 0.0))
        dx, dz = px - float(src["x"]), pz - float(src["z"])
        dist = math.hypot(dx, dz)
        fx, fz = forward_of(ayaw)
        rx, rz = right_of(ayaw)
        along = dx * fx + dz * fz
        lateral = dx * rx + dz * rz
        angle = math.degrees(math.acos(max(-1.0, min(1.0, along / dist)))) if dist > 1e-6 else 0.0
        reach = float(shape.get("reach", shape.get("radius", shape.get("range", 0.0))) or 0.0)
        arc = float(shape.get("arcDeg", 360.0))
        if kind == "melee":
            in_arc = angle <= arc / 2 and dist - PLAYER_RADIUS <= reach
            threatens = angle <= arc / 2 + ARC_MARGIN_DEG and dist - PLAYER_RADIUS <= reach + REACH_MARGIN_M
        elif kind in ("self_aoe", "target_aoe"):
            in_arc = dist <= reach
            threatens = dist <= reach + REACH_MARGIN_M
        elif kind == "projectile":
            in_arc = t.get("targetId") == me.get("entityId")
            threatens = in_arc
        else:
            in_arc = threatens = False
        # The strafe that grows our lateral offset from the swing line, in our own camera frame.
        away_x, away_z = (rx, rz) if lateral >= 0 else (-rx, -rz)
        prx, prz = right_of(pyaw)
        dodge_key = "d" if (away_x * prx + away_z * prz) >= 0 else "a"
        role = "boss" if boss and src["entityId"] == boss["entityId"] else "horse" if horse and src["entityId"] == horse["entityId"] else "mob"
        attack_id = str(t.get("attackId", ""))
        if attack_id in ALWAYS_THREATENS:
            threatens = True
        guardable = str(t.get("medium", "physical")) == "physical" and attack_id not in SHIELD_ONLY and attack_id not in DODGE_ONLY
        candidates.append((threatens, float(t.get("landInMs", 0.0)), {
            "key": f"{t.get('entityId')}:{attack_id}:{int(t.get('startedAt', 0))}",
            "entity_id": int(t.get("entityId", 0)), "id": attack_id, "name": str(t.get("name", "")), "role": role,
            "land_in_ms": float(t.get("landInMs", 0.0)), "windup_ms": float(t.get("windupMs", 0.0)),
            "weight": float(t.get("weight", 0.0)), "medium": str(t.get("medium", "physical")), "shape": kind,
            "reach_m": reach, "arc_deg": arc, "in_arc": bool(in_arc), "threatens": bool(threatens), "guardable": guardable,
            "dodge_key": dodge_key, "distance_m": dist, "lateral_m": lateral, "angle_deg": angle,
        }))
    # The boss's most recently landed blow: its follow-ups (lightning walking for 3.6 s,
    # the wave 150 ms later) outlive the telegraph that announced them.
    recent = None
    for t in snapshot.get("telegraphs", []):
        if t.get("landed") and boss and t.get("entityId") == boss["entityId"]:
            if recent is None or float(t.get("landedAgoMs", 0.0)) < float(recent.get("landedAgoMs", 0.0)):
                recent = t
    out["recent.id"] = (str(recent.get("attackId", "")) if recent else "", 1.0)
    out["recent.since_ms"] = (float(recent.get("landedAgoMs", 0.0)) if recent else None, 1.0 if recent else 0.0)
    out["recent.key"] = (f"{recent.get('entityId')}:{recent.get('attackId')}:{int(recent.get('startedAt', 0))}" if recent else "", 1.0)
    if not candidates:
        return out
    # Soonest threatening blow first; a harmless one only if nothing threatens.
    candidates.sort(key=lambda c: (not c[0], c[1]))
    chosen = candidates[0][2]
    out["attack.active"] = (True, 1.0)
    for name, value in chosen.items():
        out[f"attack.{name}"] = (value, 1.0)
    return out


def _marker_fields(snapshot: Mapping[str, Any], me: Mapping[str, Any]) -> dict[str, Reading]:
    px, pz, pyaw = float(me.get("x", 0.0)), float(me.get("z", 0.0)), float(me.get("yaw", 0.0))
    markers = [m for m in snapshot.get("markers", []) if m.get("position")]
    out: dict[str, Reading] = {"marker.count": (len(markers), 1.0)}
    best = None
    for m in markers:
        pos = m["position"]
        dx, dz = px - float(pos["x"]), pz - float(pos["z"])
        dist = math.hypot(dx, dz)
        if m.get("markerType") == "line_aoe":
            # distance from the line's axis, over its length
            fx, fz = forward_of(float(m.get("yaw", 0.0)))
            along = -(dx * fx + dz * fz)  # marker forward points away from its origin
            lateral = abs(dx * fz - dz * fx)
            inside_len = 0.0 <= along <= float(m.get("length", 0.0))
            margin = (lateral - float(m.get("width", 0.0)) / 2) if inside_len else max(lateral, -along, along - float(m.get("length", 0.0)))
            radius = float(m.get("width", 0.0)) / 2
        else:
            radius = float(m.get("radius", 0.0))
            margin = dist - radius
        if best is None or margin < best[0]:
            away = (dx / dist, dz / dist) if dist > 1e-6 else (0.0, 1.0)
            best = (margin, m, away, radius)
    if best is None:
        out["marker.nearest_m"] = (None, 0.0)
        return out
    margin, m, away, radius = best
    fx, fz = forward_of(pyaw)
    rx, rz = right_of(pyaw)
    out["marker.nearest_m"] = (margin, 1.0)
    out["marker.kind"] = (str(m.get("markerType", "")), 1.0)
    out["marker.id"] = (str(m.get("markerId", "")), 1.0)
    out["marker.radius_m"] = (radius, 1.0)
    out["marker.remaining_ms"] = (float(m.get("remainingMs", 0.0)), 1.0)
    out["marker.away_forward"] = (away[0] * fx + away[1] * fz, 1.0)
    out["marker.away_right"] = (away[0] * rx + away[1] * rz, 1.0)
    return out


def extract_fields(snapshot: Mapping[str, Any]) -> dict[str, Reading]:
    me = snapshot.get("self") or {}
    out: dict[str, Reading] = {}
    max_hp = float(me.get("maxHp", 0.0)) or 0.0
    max_mp = float(me.get("maxMana", 0.0)) or 0.0
    vitals_conf = 1.0 if max_hp > 0 else 0.0
    out["player.hp"] = (float(me.get("hp", 0.0)), vitals_conf)
    out["player.max_hp"] = (max_hp, vitals_conf)
    out["player.hp_frac"] = ((float(me.get("hp", 0.0)) / max_hp) if max_hp > 0 else None, vitals_conf)
    out["player.mp_frac"] = ((float(me.get("mana", 0.0)) / max_mp) if max_mp > 0 else None, 1.0 if max_mp > 0 else 0.0)
    out["player.downed"] = (bool(me.get("downed", False)), 1.0)
    out["player.guarding"] = (bool(me.get("guarding", False)), 1.0)
    # Dead-reckoned between authoritative corrections (every 150 ms): trusted, not certain.
    for axis in ("x", "y", "z", "yaw"):
        out[f"player.{axis}"] = (float(me.get(axis, 0.0)), 0.9)
    out["player.move_mul"] = (float(me.get("moveMul", 1.0)), 1.0)
    out["player.speed_mps"] = (float(me.get("speed", 0.0)), 0.9)
    out["player.stuck_ms"] = (float(me.get("stuckMs", 0.0)), 0.9)
    out["build.stock"] = (int(me.get("buildStock", 0)), 1.0)
    out["zone.id"] = (str(snapshot.get("zone", "")), 1.0)
    arena = ARENAS.get(str(snapshot.get("zone", "")))
    if arena:
        ax, az, ar = arena
        from_centre = math.hypot(float(me.get("x", 0.0)) - ax, float(me.get("z", 0.0)) - az)
        out["arena.edge_m"] = (ar - from_centre, 0.9)
        out["arena.radius_m"] = (ar, 1.0)
    else:
        out["arena.edge_m"] = (None, 0.0)
        out["arena.radius_m"] = (None, 0.0)

    entities = {e["entityId"]: e for e in snapshot.get("entities", [])}
    facing = me.get("facing")
    target = entities.get(facing) if facing is not None else None
    out["target.exists"] = (bool(target and target.get("alive")), 1.0)
    out["target.entity_id"] = (int(target["entityId"]) if target else None, 1.0)
    out["target.distance_m"] = (float(target["distance"]) if target else None, 1.0 if target else 0.0)
    thp = float(target.get("maxHp", 0.0)) if target else 0.0
    out["target.hp_frac"] = ((float(target.get("hp", 0.0)) / thp) if target and thp > 0 else None, 1.0 if target and thp > 0 else 0.0)

    challenge = snapshot.get("boss") or {}
    boss = entities.get(challenge.get("entityId")) if challenge else None
    if boss is None:
        boss = _entity_by_mob(snapshot, BOSS_ID)
    horse = _entity_by_mob(snapshot, HORSE_ID)
    for role, ent in (("boss", boss), ("horse", horse)):
        out[f"{role}.exists"] = (ent is not None, 1.0)
        out[f"{role}.alive"] = (bool(ent and ent.get("alive")), 1.0)
        out[f"{role}.entity_id"] = (int(ent["entityId"]) if ent else None, 1.0)
        out[f"{role}.distance_m"] = (float(ent["distance"]) if ent else None, 1.0 if ent else 0.0)
        out[f"{role}.x"] = (float(ent["x"]) if ent else None, 1.0 if ent else 0.0)
        out[f"{role}.z"] = (float(ent["z"]) if ent else None, 1.0 if ent else 0.0)
        mhp = float(ent.get("maxHp", 0.0)) if ent else 0.0
        out[f"{role}.hp_frac"] = ((float(ent.get("hp", 0.0)) / mhp) if ent and mhp > 0 else None, 1.0 if ent and mhp > 0 else 0.0)
        opening = ent.get("opening") if ent else None
        out[f"{role}.opening"] = (bool(opening), 1.0)
        out[f"{role}.opening_kind"] = (str(opening.get("kind", "")) if opening else "", 1.0)
    out["target.role"] = ("boss" if target and boss and target["entityId"] == boss["entityId"] else "horse" if target and horse and target["entityId"] == horse["entityId"] else "mob" if target else "", 1.0)
    if boss and horse:
        out["horse.distance_to_boss_m"] = (math.hypot(float(boss["x"]) - float(horse["x"]), float(boss["z"]) - float(horse["z"])), 1.0)
    else:
        out["horse.distance_to_boss_m"] = (None, 0.0)
    out["boss.phase"] = (str(challenge.get("phase", "")) if challenge else "", 1.0 if challenge else 0.0)
    out["boss.detail"] = (str(challenge.get("detail", "")) if challenge else "", 1.0)
    out["boss.mastered"] = (bool(challenge.get("mastered", False)), 1.0)
    out["boss.kill_locked"] = (bool(challenge.get("killLocked", True)), 1.0)
    out["boss.attempt"] = (int(challenge.get("attempt", 0)), 1.0)
    out["boss.remaining_ms"] = (float(challenge.get("remainingMs", 0.0)), 1.0)

    out.update(_telegraph_fields(snapshot, me, boss, horse))
    out.update(_marker_fields(snapshot, me))

    counts: dict[str, int] = {}
    for s in snapshot.get("inventory") or []:
        counts[str(s.get("itemId"))] = counts.get(str(s.get("itemId")), 0) + int(s.get("quantity", 0))
    for item_id in TRACKED_ITEMS:
        out[f"bag.{item_id}"] = (counts.get(item_id, 0), 1.0)
    # Other players in the room (the requester, when we are a summoned helper): the nearest one.
    others = [e for e in snapshot.get("entities", []) if e.get("type") == "player" and e.get("entityId") != me.get("entityId")]
    nearest = min(others, key=lambda e: e.get("distance", 1e9)) if others else None
    out["ally.exists"] = (nearest is not None, 1.0)
    out["ally.entity_id"] = (int(nearest["entityId"]) if nearest else None, 1.0)
    out["ally.distance_m"] = (float(nearest["distance"]) if nearest else None, 1.0 if nearest else 0.0)

    skills = snapshot.get("skills") or {}
    unlocked = set(snapshot.get("unlocked") or [])
    for skill_id in TRACKED_SKILLS:
        entry = skills.get(skill_id)
        known = entry is not None and skill_id in unlocked
        ready_in = float(entry.get("readyInMs", 0.0)) if entry else 0.0
        out[f"skill.{skill_id}.ready"] = (known and ready_in <= 0.0, 1.0 if known else 0.6)
        out[f"skill.{skill_id}.ready_in_ms"] = (ready_in, 1.0 if known else 0.0)
    return out


PROVIDES: tuple[str, ...] = (
    "player.hp", "player.max_hp", "player.hp_frac", "player.mp_frac", "player.downed", "player.guarding",
    "player.x", "player.y", "player.z", "player.yaw", "player.move_mul", "player.speed_mps", "player.stuck_ms",
    "build.stock", "zone.id", "arena.edge_m", "arena.radius_m", "recent.id", "recent.since_ms", "recent.key",
    "bag.*", "ally.exists", "ally.entity_id", "ally.distance_m",
    "target.exists", "target.entity_id", "target.distance_m", "target.hp_frac", "target.role",
    "boss.*", "horse.*", "attack.*", "marker.*", "skill.*",
)


def build_api_bundle() -> SensorBundle:
    return SensorBundle([SnapshotSensor("fancraft", PROVIDES, extract_fields, cadence=1)])


# -- keys → intents ----------------------------------------------------------------


def _find(snapshot: Mapping[str, Any], role: str) -> dict[str, Any] | None:
    if role == "boss":
        challenge = snapshot.get("boss") or {}
        for e in snapshot.get("entities", []):
            if challenge and e.get("entityId") == challenge.get("entityId"):
                return e
        return _entity_by_mob(snapshot, BOSS_ID)
    if role == "horse":
        return _entity_by_mob(snapshot, HORSE_ID)
    if role == "nearest":
        mobs = [e for e in snapshot.get("entities", []) if e.get("type") == "mob" and e.get("alive")]
        return min(mobs, key=lambda e: e.get("distance", 1e9)) if mobs else None
    if role == "ally":
        me = (snapshot.get("self") or {}).get("entityId")
        others = [e for e in snapshot.get("entities", []) if e.get("type") == "player" and e.get("entityId") != me]
        return min(others, key=lambda e: e.get("distance", 1e9)) if others else None
    if role.isdigit():
        for e in snapshot.get("entities", []):
            if e.get("entityId") == int(role):
                return e
    return None


def cover_cells(me: Mapping[str, Any], origin: Mapping[str, Any], gap_m: float = 1.6, width: int = 3, height: int = 3) -> list[tuple[int, int, int]]:
    """Voxels for a wall between us and `origin` (the pulse's centre).

    The exam (`ChallengeCover.hasFullCover`) casts rays from the boss's chest
    to our head, torso, shins and both sides; a `width`×`height` plane one
    block out toward him, standing on the floor we stand on, blocks all of
    them at any range inside the arena.
    """
    px, py, pz = float(me.get("x", 0.0)), float(me.get("y", 0.0)), float(me.get("z", 0.0))
    dx, dz = float(origin.get("x", 0.0)) - px, float(origin.get("z", 0.0)) - pz
    dist = math.hypot(dx, dz)
    if dist < 1e-6:
        dx, dz, dist = 0.0, 1.0, 1.0
    dx, dz = dx / dist, dz / dist
    perp_x, perp_z = -dz, dx
    cx, cz = px + dx * gap_m, pz + dz * gap_m
    y0 = math.floor(py + 0.01)
    cells: list[tuple[int, int, int]] = []
    half = width // 2
    for i in range(-half, half + 1):
        bx = math.floor(cx + perp_x * i)
        bz = math.floor(cz + perp_z * i)
        for k in range(height):
            cells.append((bx, y0 + k, bz))
    return cells


@dataclass(slots=True)
class FightLedger:
    """What happened, from the gateway's events — printed after the run.

    This is the API-control counterpart of the oracle join: it says which
    telegraphs the tester met and how each resolved against us, when the
    boss changed phase, and how often we fell.
    """

    started_at: float = field(default_factory=now)
    self_id: int = 0
    telegraphs: dict[str, int] = field(default_factory=dict)
    outcomes: dict[str, dict[str, int]] = field(default_factory=dict)
    phases: list[tuple[float, str, str]] = field(default_factory=list)
    deaths: int = 0
    strikes: dict[str, int] = field(default_factory=dict)
    skill_fails: dict[str, int] = field(default_factory=dict)
    notices: list[tuple[float, str]] = field(default_factory=list)
    completed: bool = False
    hits_on_us: dict[str, dict[str, int]] = field(default_factory=dict)
    last_hit: tuple[float, str, int, str] | None = None
    _last_phase: str = ""

    def on_event(self, event: Event) -> None:
        t = event.received_at - self.started_at
        d = event.data
        if event.type == "mob_attack_start":
            self.telegraphs[str(d.get("attackId"))] = self.telegraphs.get(str(d.get("attackId")), 0) + 1
        elif event.type == "combat_action":
            src, tgt, skill = d.get("sourceId"), d.get("targetId"), str(d.get("skillId"))
            kind = str(d.get("hitKind"))
            damage = int(d.get("damage") or 0)
            if src == self.self_id:
                self.strikes[kind] = self.strikes.get(kind, 0) + 1
            elif tgt == self.self_id:
                per = (self.outcomes if skill in self.telegraphs else self.hits_on_us).setdefault(skill, {})
                per[kind] = per.get(kind, 0) + 1
                if damage > 0:
                    self.last_hit = (t, skill, damage, kind)
            elif tgt == 0 and skill in self.telegraphs:
                per = self.outcomes.setdefault(skill, {})
                per["miss"] = per.get("miss", 0) + 1
        elif event.type == "player_downed":
            self.deaths += 1
            last = f"; last hit {self.last_hit[1]} {self.last_hit[3]} {self.last_hit[2]} at t={self.last_hit[0]:.1f}s" if self.last_hit else ""
            self.notices.append((t, f"downed ({d.get('reason')}) during {self._last_phase or '?'}{last}"))
        elif event.type == "boss_challenge":
            phase = str(d.get("phase", ""))
            detail = str(d.get("detail", ""))
            if phase != self._last_phase:
                self._last_phase = phase
                self.phases.append((t, phase, detail))
        elif event.type == "skill_fail":
            key = f"{d.get('skillId')}:{d.get('reason')}"
            self.skill_fails[key] = self.skill_fails.get(key, 0) + 1
        elif event.type == "system_message":
            key = str((d.get("loc") or {}).get("key", ""))
            if key.startswith(("warden.", "challenge.", "sys_world", "sys_erosion")):
                self.notices.append((t, key))
        elif event.type == "dungeon_complete":
            self.completed = True
            self.notices.append((t, "dungeon_complete"))

    def report(self) -> list[str]:
        lines = [f"fight: phases {' → '.join(f'{p}@{t:.0f}s' for t, p, _ in self.phases) or '-'}; deaths={self.deaths}; completed={self.completed}"]
        for attack, n in sorted(self.telegraphs.items()):
            outcome = self.outcomes.get(attack, {})
            lines.append(f"  {attack:18s} x{n:<3d} " + (", ".join(f"{k}={v}" for k, v in sorted(outcome.items())) or "(never resolved against us)"))
        for skill, per in sorted(self.hits_on_us.items()):
            lines.append(f"  {skill:18s} (untelegraphed) " + ", ".join(f"{k}={v}" for k, v in sorted(per.items())))
        if self.strikes:
            lines.append("  our strikes: " + ", ".join(f"{k}={v}" for k, v in sorted(self.strikes.items())))
        if self.skill_fails:
            lines.append("  skill refusals: " + ", ".join(f"{k}={v}" for k, v in sorted(self.skill_fails.items())))
        for t, note in self.notices[-12:]:
            lines.append(f"  t={t:6.1f}s {note}")
        return lines


def build_key_actions(gateway: GatewayClient) -> KeyActions:
    """The FantCraft intent vocabulary a plan may press."""

    def snapshot() -> dict[str, Any]:
        latest = gateway.latest()
        return latest.data if latest else {}

    def face(role: str) -> None:
        ent = _find(snapshot(), role)
        if ent is None:
            gateway.fire("face")
        else:
            gateway.fire("face", entityId=int(ent["entityId"]))

    def skill(skill_id: str, role: str | None) -> None:
        args: dict[str, Any] = {"skillId": skill_id}
        if role:
            ent = _find(snapshot(), role)
            if ent is not None:
                args["targetId"] = int(ent["entityId"])
        gateway.fire("skill", **args)

    def use(item_id: str, role: str | None) -> None:
        args: dict[str, Any] = {"itemId": item_id}
        if role:
            ent = _find(snapshot(), role)
            if ent is not None:
                args["targetId"] = int(ent["entityId"])
        gateway.fire("use", **args)

    def cover() -> None:
        snap = snapshot()
        me = snap.get("self") or {}
        origin = _find(snap, "boss")
        if not me or origin is None:
            return
        for x, y, z in cover_cells(me, origin):
            gateway.fire("place", x=x, y=y, z=z)

    fixed: dict[str, Any] = {
        "tab": lambda down: face("nearest") if down else None,
        "escape": lambda down: gateway.fire("face") if down else None,
        "cover": lambda down: cover() if down else None,
    }

    def resolve(key: str):
        if key.startswith("samurai:"):
            # Hex preserves camelCase protocol fields through ApiBackend's
            # lowercase key normalization. This is an ordinary API intent list.
            commands = json.loads(bytes.fromhex(key[8:]).decode())
            def samurai_action(down):
                if down:
                    for command in commands:
                        args = dict(command)
                        op = args.pop("op")
                        if op in {"move", "face", "attack", "guard", "skill", "use"}:
                            gateway.fire(op, **args)
            return samurai_action
        if key.startswith("face:"):
            role = key[5:]
            return lambda down: face(role) if down else None
        if key.startswith("skill:"):
            spec = key[6:]
            skill_id, _, role = spec.partition("@")
            return lambda down: skill(skill_id, role or None) if down else None
        if key.startswith("chat:"):
            text = key[5:]
            return lambda down: gateway.fire("chat", message=text) if down else None
        if key.startswith("use:"):
            spec = key[4:]
            item_id, _, role = spec.partition("@")
            return lambda down: use(item_id, role or None) if down else None
        return None

    return KeyActions(fixed=fixed, resolve=resolve)


__all__ = [
    "ALWAYS_THREATENS", "ARENAS", "BOSS_ID", "DODGE_ONLY", "TRACKED_ITEMS", "FightLedger", "HORSE_ID", "PROVIDES", "SHIELD_ONLY", "TRACKED_SKILLS",
    "build_api_bundle", "build_key_actions", "cover_cells", "extract_fields",
]
