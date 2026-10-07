"""October Samurai tactics, using only the ordinary agent gateway's visible state.

The planner never imports FantCraft server code, reads truth/replay files, or asks
for unrevealed circles. Memory is built from circles while their cue is visible.
Every movement vector is normalized to the ordinary 5 m/s walk speed (no sprint
or faster diagonal). Plans are reconsidered on each 50 ms gateway snapshot.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from player.act.timeline import Plan, Press, Priority, Step
from player.policy.condition import Condition
from player.policy.reflex import Reflex

SPEED = 5.0
BODY = .5
LATENCY_MS = 100.0
# Never start a swing when a stroke aimed at us lands sooner than a basic
# attack's windup plus recovery: a busy hand cannot be turned into a guard.
HOLD_STRIKE_MS = 1000.0
# The Palm Seize steps in during its swing (measured live): extra cone reach.
PALM_STEP_IN = 1.2
# 大蛇: the lightning walk's stride and cadence (fancraft SAMURAI_FORMS).
OROCHI_STRIDE, OROCHI_EVERY_MS = 2.25, 450.0
# Within this of the newest strike, the walk is ours: keep walking away.
OROCHI_STALK_RANGE = 3.2
# Start circling the Samurai this long before a Great Serpent cut lands.
SERPENT_PREMOVE_MS = 900.0
# Goal weight while heading for a remembered cloud-memory pocket.
MEMORY_PULL = 4.0
# Turn to a guardable stroke's source this long before it lands.
FACE_ATTACKER_MS = 700.0
# Basic attacks only within this long after the victim's own stroke landed.
PUNISH_MS = 900.0
# Footprints (+ our body) that movement cannot pass through.
BODY_RADIUS = {"ironclad_warden": 1.7, "mechanical_horse": 1.8, "warden_exoskeleton": 2.2}
# The public combat rule (fancraft shared/src/combat.ts): a guard raised within
# the parry window parries; earlier, a stroke of weight >= .7 breaks it.
PARRY_LIGHT_MS, PARRY_HEAVY_MS = 700.0, 170.0


def guard_lead_ms(weight):
    """When to press guard before the announced landing, so it lands in the parry window.

    Measured live (10-05): the blow arrives 40 ms early to 140 ms late against
    `landInMs`, and our press waits up to one 50 ms snapshot plus transport. 60% of
    the window centres that spread for heavy strokes; light ones keep a 320 ms cap.
    """
    window = PARRY_LIGHT_MS + (PARRY_HEAVY_MS - PARRY_LIGHT_MS) * max(0., min(1., weight))
    return min(320., window * .6)
UNGUARDABLE = {"warden_draw2", "warden_wave", "warden_tachikaze", "warden_tachikaze_wave", "warden_palm", "warden_palm2", "warden_cloud", "warden_charge"}


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point(entity):
    return float(entity.get("x", 0)), float(entity.get("z", 0))


def angle_to(me, target):
    return math.atan2(-(target[0] - me[0]), -(target[1] - me[1]))


@dataclass(frozen=True)
class Hazard:
    """A public marker/telegraph and its announced landing time, in gateway ms."""
    key: str
    at: float
    x: float
    z: float
    radius: float
    group: int = 0
    yaw: float = 0
    arc: float = 360
    width: float = 0
    kind: str = "circle"

    def margin(self, p):
        dx, dz = p[0] - self.x, p[1] - self.z
        dist = math.hypot(dx, dz)
        if self.kind == "line":
            along = dx * -math.sin(self.yaw) + dz * -math.cos(self.yaw)
            lateral = abs(dx * math.cos(self.yaw) - dz * math.sin(self.yaw))
            return max(lateral - self.width / 2 - BODY, -along - BODY, along - self.radius - BODY)
        if self.kind == "cone" and dist > .001:
            a = abs(math.atan2(math.sin(angle_to((self.x, self.z), p) - self.yaw), math.cos(angle_to((self.x, self.z), p) - self.yaw)))
            return max(dist - self.radius - BODY, dist * math.sin(min(math.pi / 2, a - math.radians(self.arc / 2))) - BODY)
        return dist - self.radius - BODY


@dataclass
class SamuraiTactics:
    role: str = "samurai"
    remembered: dict[str, Hazard] = field(default_factory=dict)
    last_actions: dict[str, float] = field(default_factory=dict)
    guard_until: float = 0
    last_room: str = ""
    route: list[tuple[float, tuple[float, float]]] = field(default_factory=list)
    route_key: tuple = ()
    last_direction: tuple[float, float] = (0, 0)
    last_snapshot: float = -1
    last_blade: tuple[float, dict] | None = None
    previous_blade: tuple[float, dict] | None = None
    last_steer_at: float = -math.inf
    threat_key: tuple = ()
    blocked: tuple[float, float] = (0., 0.)
    last_landed: dict = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)

    def claim(self, key, at, interval):
        if at - self.last_actions.get(key, -math.inf) < interval:
            return False
        self.last_actions[key] = at
        self.stats[key.split(":")[0]] = self.stats.get(key.split(":")[0], 0) + 1
        return True

    def observe(self, snapshot):
        now = float(snapshot.get("t", 0))
        room = str(snapshot.get("roomId", ""))
        if room != self.last_room or now < self.last_snapshot:
            self.remembered.clear()
            self.route_key = ()
            self.last_actions.clear()
            self.last_room = room
        self.last_snapshot = now
        # A faded cue is remembered only if we actually saw it earlier. No late
        # joiner can discover the hidden groups by reading the gateway's cache.
        for marker in snapshot.get("markers", []):
            key = str(marker.get("markerId", ""))
            pos = marker.get("position")
            if not pos or not key:
                continue
            cloud = marker.get("skillId") == "warden_nyuudougumo"
            started = float(marker.get("startedAt", now))
            if cloud and now - started > float(marker.get("cueMs", 750)) and key not in self.remembered:
                continue
            at = float(marker.get("expiresAt", now + float(marker.get("remainingMs", 0))))
            kind = "line" if marker.get("markerType") == "line_aoe" else "cone" if marker.get("markerType") == "cone_aoe" else "circle"
            self.remembered[key] = Hazard(key, at, float(pos["x"]), float(pos["z"]),
                float(marker.get("length", marker.get("radius", 0))), int(marker.get("sequenceIndex", 0)) if cloud else 0,
                float(marker.get("yaw", 0)), float(marker.get("arcDeg", 360)), float(marker.get("width", 0)), kind)
        self.remembered = {key: h for key, h in self.remembered.items() if h.at >= now - 130}
        me = snapshot.get("self") or {}
        hazards = list(self.remembered.values()) + self.orochi_ahead(snapshot.get("markers", []), now,
            (float(me["x"]), float(me["z"])) if "x" in me else None)
        entities = {e["entityId"]: e for e in snapshot.get("entities", [])}
        for tell in snapshot.get("telegraphs", []):
            if tell.get("landed") or tell.get("attackId") == "warden_tensei":
                continue
            source = entities.get(tell.get("entityId"))
            if not source:
                continue
            # Guardable strokes are answered separately; only the aimed player
            # needs the escape planner for those. Palms/cones must be left.
            if tell.get("attackId") not in UNGUARDABLE and tell.get("medium", "physical") == "physical":
                continue
            shape = tell.get("shape") or {}
            kind = "cone" if shape.get("kind") == "melee" else "circle"
            at = float(tell.get("landsAt", now + float(tell.get("landInMs", 0))))
            reach = float(shape.get("reach", shape.get("radius", shape.get("maxRange", 0))))
            # The seize steps in while it swings: backing straight out along the
            # palm's axis was caught at 3.35 m live. Count the step, so the
            # planner prefers leaving the narrow cone sideways.
            if str(tell.get("attackId", "")).startswith("warden_palm"):
                reach += PALM_STEP_IN
            hazards.append(Hazard(str(tell.get("attackId")), at, float(source["x"]), float(source["z"]), reach,
                yaw=float(tell.get("yaw", 0)), arc=float(shape.get("arcDeg", 360)), kind=kind))
            # The released wave has the already announced locked direction;
            # remember that visible draw line rather than future server state.
            if tell.get("attackId") in ("warden_draw2", "warden_tachikaze"):
                hazards.append(Hazard("draw-wave", at + 150, float(source["x"]), float(source["z"]), 20,
                    yaw=float(tell.get("yaw", 0)), arc=30, kind="cone"))
        return now, hazards

    def outwalk_serpent(self, me, markers, center=(64., 64.), radius=20., bodies=()):
        """Keep walking straight away from the serpent's newest strike while it stalks us.

        The walk strides exactly our walking speed: zig-zagging loses ground and
        a straight, unbroken walk stays one stride (2.25 m > 1.5 m) ahead. Of
        sixteen headings, take the one leading furthest from the strike that
        neither runs into a big body (live: wedged against the Samurai for 170 ms
        under the first strike) nor into the wall.
        """
        newest = None
        for marker in markers:
            if marker.get("skillId") == "warden_orochi" and marker.get("position"):
                at = float(marker.get("expiresAt", marker.get("remainingMs", 0)))
                if newest is None or at > newest[2]:
                    newest = (float(marker["position"]["x"]), float(marker["position"]["z"]), at)
        if newest is None or distance(me, newest[:2]) > OROCHI_STALK_RANGE:
            return None
        best, best_score = None, -math.inf
        for i in range(16):
            d = (math.sin(i * math.tau / 16), math.cos(i * math.tau / 16))
            ahead = (me[0] + d[0] * OROCHI_STRIDE, me[1] + d[1] * OROCHI_STRIDE)
            score = distance(ahead, newest[:2])
            if distance(ahead, center) > radius - 1.:
                score -= 10
            for b, r in bodies:
                # any point of the next stride inside a body is a wall we would grind against
                if min(distance((me[0] + d[0] * k / 4 * OROCHI_STRIDE, me[1] + d[1] * k / 4 * OROCHI_STRIDE), b) for k in range(1, 5)) < r + .3:
                    score -= 10
            if self.last_direction != (0., 0.):
                score += .6 * (d[0] * self.last_direction[0] + d[1] * self.last_direction[1])
            if score > best_score:
                best, best_score = d, score
        return best

    def circle_before_serpent(self, me, telegraphs, entities, bodies=(), center=(64., 64.), radius=20.):
        """Already be walking when the Great Serpent's first strike lands on us.

        The walk's first strike is placed where the cut points, i.e. on the one
        who parried it, 450 ms before it lands; from a standstill the server's
        input buffer (~150-280 ms measured) eats most of that. So from 0.9 s before
        the cut we circle the Samurai at walking speed, still facing (and
        parrying) him; the walk then only has to be kept going.
        """
        for tell in telegraphs:
            if tell.get("landed") or tell.get("attackId") != "warden_overhead2":
                continue
            if float(tell.get("landInMs", 9999)) > SERPENT_PREMOVE_MS:
                continue
            source = next((e for e in entities if e.get("entityId") == tell.get("entityId")), None)
            if not source:
                continue
            b = point(source)
            rx, rz = me[0] - b[0], me[1] - b[1]
            r = math.hypot(rx, rz) or 1.
            options = [(-rz / r, rx / r), (rz / r, -rx / r)]
            def cost(d):
                ahead = (me[0] + d[0] * 1.5, me[1] + d[1] * 1.5)
                c = 10. if distance(ahead, center) > radius - 1.5 else 0.
                c += sum(5. for o, orad in bodies if o != b and distance(ahead, o) < orad + .4)
                return c - .5 * (d[0] * self.last_direction[0] + d[1] * self.last_direction[1])
            return min(options, key=cost)
        return None

    def leave_sweep(self, me, hazards, now, bodies=(), center=(64., 64.), radius=20.):
        """Walk out of an unguardable cut we are standing in, as soon as it is shown.

        The Wind Cutter sweeps 150° out to 5.1 m on a 1.7 s windup: trivially
        escaped by walking out at once, but the general planner only looks 1.4 s
        ahead and drifted into the Samurai (seen live). Of sixteen headings take
        the one whose reachable point at landing is furthest outside every cut
        landing in the next 2 s, never through a body or into the wall.
        """
        cuts = [h for h in hazards if h.kind == "cone" and h.key in UNGUARDABLE and 0 <= h.at - now <= 2000]
        if not cuts or min(h.margin(me) for h in cuts) >= .3:
            return None
        best, best_score = None, -math.inf
        for i in range(16):
            d = (math.sin(i * math.tau / 16), math.cos(i * math.tau / 16))
            score = math.inf
            for h in cuts:
                reach = SPEED * max(0., (h.at - now - LATENCY_MS) / 1000)
                end = (me[0] + d[0] * reach, me[1] + d[1] * reach)
                score = min(score, min(h.margin(end), 1.5))
            far = (me[0] + d[0] * 1.5, me[1] + d[1] * 1.5)
            if distance(far, center) > radius - 1.:
                score -= 5
            for b, r in bodies:
                if min(distance((me[0] + d[0] * k / 3, me[1] + d[1] * k / 3), b) for k in range(1, 5)) < r + .3:
                    score -= 5
            if score > best_score:
                best, best_score = d, score
        return best

    def punish_window(self, snapshot, victim, now):
        """Swing only right after the victim's own stroke landed, with nothing else coming.

        A started swing holds the hand through its recovery; one begun just before the
        Samurai's next windup cannot become a guard in time (seen live: a 180° middle
        cut killed both characters while their swings were still recovering).
        Phase two's progress never depends on our damage.
        """
        if any(not t.get("landed") for t in snapshot.get("telegraphs", [])):
            return False
        # A swing roots us for its windup and recovery (~340 ms): never while the
        # serpent walks (it follows the Great Serpent cut we just parried — seen
        # live) or while any ground strike near us is about to land.
        me = snapshot.get("self") or {}
        here = (float(me.get("x", 0)), float(me.get("z", 0)))
        for marker in snapshot.get("markers", []):
            pos = marker.get("position") or {}
            if marker.get("skillId") == "warden_orochi":
                return False
            land = float(marker.get("remainingMs", marker.get("expiresAt", now) - now))
            if pos and land <= 1200 and distance(here, (float(pos.get("x", 0)), float(pos.get("z", 0)))) <= float(marker.get("radius", 2)) + 2.5:
                return False
        return now - self.last_landed.get(victim.get("entityId"), -math.inf) <= PUNISH_MS

    def orochi_ahead(self, markers, now, me=None):
        """Where the serpent strikes next, from the step it is showing.

        The walk stalks its target (public behaviour: each 450 ms strike lands one
        2.5 m stride from the last, towards where the target stands when it lands).
        Standing within a stride of the visible strike means the next one lands on
        you; the planner sees that as a circle around the predicted spot.
        """
        live = None
        for marker in markers:
            if marker.get("skillId") == "warden_orochi" and marker.get("position"):
                at = float(marker.get("expiresAt", now + float(marker.get("remainingMs", 0))))
                if live is None or at > live[2]:
                    live = (float(marker["position"]["x"]), float(marker["position"]["z"]), at, float(marker.get("radius", 1.5)))
        if live is None or me is None:
            return []
        x, z, at, radius = live
        dx, dz = me[0] - x, me[1] - z
        gap = math.hypot(dx, dz)
        stride = min(gap, OROCHI_STRIDE)
        nx, nz = (x + dx / gap * stride, z + dz / gap * stride) if gap > .01 else (x, z)
        return [Hazard("orochi-next", at + OROCHI_EVERY_MS, nx, nz, radius)]

    def memory_route(self, me, hazards, now, center=(64., 64.), radius=20.):
        # One layer per remembered set. Its circles' timers differ by a millisecond or
        # two (sent one by one); keying on the time split a set into partial layers
        # whose "safe" points sat under the set's other palms (two deaths, 10-05).
        landing: dict[int, float] = {}
        for h in hazards:
            if h.group and h.at >= now - 50:
                landing[h.group] = min(landing.get(h.group, h.at), h.at)
        groups = sorted(landing.items(), key=lambda x: x[1])
        key = tuple((g, round(t), sum(h.group == g for h in hazards)) for g, t in groups)
        if key == self.route_key:
            return next((p for at, p in self.route if at >= now - 50), None)
        self.route_key = key
        self.route = []
        if not groups:
            return None
        layers = []
        # Broad safe pockets, not the generator's private witness route. A .75m
        # public-space sampling grid plus .5m body clearance finds usable areas.
        for group, at in groups:
            circles = [h for h in hazards if h.group == group]
            candidates = []
            count = math.ceil(2 * radius / .75)
            for i in range(count + 1):
                for j in range(count + 1):
                    p = (center[0] - radius + .75 * i, center[1] - radius + .75 * j)
                    if distance(p, center) > radius - .8:
                        continue
                    margin = min(h.margin(p) for h in circles)
                    if margin >= .12:
                        candidates.append((p, margin))
            # Preserve all candidates: pruning to points near the player can
            # discard the only route to group three on the opposite side.
            layers.append((at, candidates))
        states = [(me, 0., [])]
        previous = now
        for at, candidates in layers:
            reach = max(0, (at - previous - (LATENCY_MS if previous == now else 50)) / 1000) * SPEED
            next_states = []
            for p, margin in candidates:
                available = [(cost + distance(q, p) + .08 / (margin + .2), path) for q, cost, path in states if distance(q, p) <= reach]
                if available:
                    cost, path = min(available, key=lambda item: item[0])
                    next_states.append((p, cost, path + [(at, p)]))
            if not next_states:
                # Stay honest about a missed/unreachable route; choose the next
                # visible opening and let the encounter report the failure.
                self.stats["unreachable_memory"] = self.stats.get("unreachable_memory", 0) + 1
                break
            states = next_states
            previous = at
        if states and states[0][2]:
            self.route = min(states, key=lambda s: s[1])[2]
        return self.route[0][1] if self.route else None

    def steer(self, me, goal, hazards, now, allies=(), center=(64., 64.), radius=20., speed=SPEED, pull=.5, bodies=()):
        """Receding-horizon search over already visible landing circles/cones.

        Four 350ms segments, sixteen normalized headings, beam width 18. Hazards
        are evaluated at their landing instant; smoke is not treated as damage
        before its visible timer expires. Longer ash timers add escape urgency.
        """
        relevant = [h for h in hazards if -100 <= h.at - now <= 4500]
        headings = [(math.sin(i * math.tau / 16), math.cos(i * math.tau / 16)) for i in range(16)] + [(0., 0.)]
        beam = [(0., me, (0., 0.))]
        dt = .35
        for depth in range(4):
            states = []
            start, end = now + depth * dt * 1000, now + (depth + 1) * dt * 1000
            for cost, p, first in beam:
                for direction in headings:
                    q = (p[0] + direction[0] * speed * dt, p[1] + direction[1] * speed * dt)
                    if distance(q, center) > radius - .8:
                        continue
                    # Big bodies are solid: a dodge into the Samurai's chest goes nowhere
                    # (seen live: stuck 165 ms under the serpent's first strike).
                    if any(distance(q, b) < r and distance(q, b) < distance(p, b) for b, r in bodies):
                        continue
                    penalty = 0.
                    for h in relevant:
                        if start - 70 <= h.at <= end + 70:
                            u = min(1., max(0., (h.at - start) / (dt * 1000)))
                            hitp = (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u)
                            margin = h.margin(hitp)
                            penalty += max(0., .2 - margin) * 20000
                        elif h.at > end:
                            # Do not enter a pocket whose visible future seal
                            # is already too close to escape at walking speed.
                            deficit = -h.margin(q) - speed * max(0., (h.at - end - LATENCY_MS) / 1000)
                            penalty += max(0., deficit) * 5000
                    separation = sum(max(0, 2.6 - distance(q, ally)) ** 2 for ally in allies)
                    # The server reported us blocked: do not keep pushing into it.
                    if depth == 0 and self.blocked != (0., 0.):
                        penalty += max(0., direction[0] * self.blocked[0] + direction[1] * self.blocked[1]) * 40
                    smooth = (direction[0] - self.last_direction[0]) ** 2 + (direction[1] - self.last_direction[1]) ** 2
                    score = cost + penalty + distance(q, goal) * pull + separation * 2 + smooth * .06
                    states.append((score, q, direction if depth == 0 else first))
            states.sort(key=lambda state: state[0])
            # Spatial buckets keep a low-cost heading from filling the beam
            # with near-identical nodes and hiding another escape corridor.
            seen = set()
            beam = []
            for state in states:
                bucket = (round(state[1][0] * 1.5), round(state[1][1] * 1.5))
                if bucket not in seen:
                    seen.add(bucket)
                    beam.append(state)
                    if len(beam) == 18:
                        break
            if not beam:
                return (0., 0.)
        direction = beam[0][2]
        self.last_direction = direction
        return direction

    def decide(self, snapshot: Mapping[str, Any], role=None):
        me = snapshot.get("self") or {}
        if snapshot.get("zone") not in ("samurai_phase", "trial_warden"):
            return None
        if role is not None:
            self.role = role
        now, hazards = self.observe(snapshot)
        cmds = []
        if me.get("downed"):
            return [{"op": "move", "forward": 0, "strafe": 0, "sprint": False}]
        entities = list(snapshot.get("entities", []))
        challenge = snapshot.get("boss") or {}
        boss = next((e for e in entities if e.get("entityId") == challenge.get("entityId")), None)
        boss = boss or next((e for e in entities if e.get("mobId") == "ironclad_warden"), None)
        if not boss:
            return [{"op": "move", "forward": 0, "strafe": 0, "sprint": False}]
        engine = next((e for e in entities if e.get("mobId") in ("warden_exoskeleton", "mechanical_horse") and e.get("alive")), None)
        holder = self.role == "hold_horse" and engine is not None
        victim = engine if holder else boss
        pos, dest = point(me), point(victim)
        phase = str(challenge.get("phase", ""))
        samurai = challenge.get("samurai") or {}
        finale, tensei = samurai.get("finale") or {}, samurai.get("tensei") or {}
        fp = str(finale.get("phase", ""))
        cast = str(finale.get("castId", challenge.get("attempt", 0)))
        target_id = tensei.get("targetId", finale.get("targetId"))
        main = target_id == me.get("entityId")
        age = max(0., now - float(challenge.get("receivedAt", now)))
        remaining = float(tensei.get("remainingMs", 99999)) - age
        stage = tensei.get("stage", "")
        rooted = fp in ("bind", "dash", "prayer", "dark", "cut", "clash") and distance(pos, point(finale.get("center") or boss)) <= float(finale.get("radius", 18))
        attack = next((t for t in snapshot.get("telegraphs", []) if t.get("attackId") == "warden_tensei" and not t.get("landed")), None)
        visible_blade = (attack or {}).get("blade") or boss.get("blade")
        if visible_blade and (self.last_blade is None or now > self.last_blade[0]):
            self.previous_blade, self.last_blade = self.last_blade, (now, visible_blade)
        if attack and stage != "clash":
            remaining = float(attack.get("landInMs", remaining))
        # The assigned engine holder remains in melee, outside the main target's
        # bearing when possible, keeping its aimed ranged attacks away from them.
        goal = dest
        if distance(pos, dest) > .01:
            keep = 2.15 if holder else 2.35
            goal = (dest[0] + (pos[0] - dest[0]) / distance(pos, dest) * keep,
                    dest[1] + (pos[1] - dest[1]) / distance(pos, dest) * keep)
        if holder and target_id is not None:
            target = next((e for e in entities if e.get("entityId") == target_id), None)
            if target and distance(dest, point(target)) > .1:
                dx, dz = dest[0] - target["x"], dest[1] - target["z"]
                length = math.hypot(dx, dz)
                goal = (dest[0] + dx / length * 2.2, dest[1] + dz / length * 2.2)
        route_goal = self.memory_route(pos, hazards, now)
        if route_goal:
            goal = route_goal
        allies = [point(e) for e in entities if e.get("type") == "player" and e.get("alive") and e.get("entityId") != me.get("entityId")]
        # A remembered safe pocket is the destination, not a suggestion: pull hard and
        # share it with allies instead of shouldering each other out of it.
        pull = MEMORY_PULL if route_goal else .5
        if route_goal:
            allies = []
        # A telegraph or marker we have not planned around yet replans at once:
        # a 500 ms palm leaves no room for the usual 150 ms steering cadence.
        threat_key = tuple(sorted({(t.get("entityId"), t.get("startedAt")) for t in snapshot.get("telegraphs", []) if not t.get("landed")}
            | {(m.get("markerId"), 0) for m in snapshot.get("markers", [])}, key=str))
        fresh = threat_key != self.threat_key
        self.threat_key = threat_key
        self.blocked = self.last_direction if float(me.get("stuckMs", 0)) > 0 else (0., 0.)
        if self.blocked != (0., 0.):
            fresh = True
        big_bodies = [(point(e), BODY_RADIUS.get(e.get("mobId"), 1.)) for e in entities
                      if e.get("mobId") in BODY_RADIUS and e.get("alive", True)]
        flee = self.outwalk_serpent(pos, snapshot.get("markers", []), bodies=big_bodies)
        if flee is None:
            flee = self.leave_sweep(pos, hazards, now, big_bodies)
        if flee is None:
            flee = self.circle_before_serpent(pos, snapshot.get("telegraphs", []), entities, big_bodies)
        if rooted:
            direction = (0., 0.)
        elif flee is not None:
            direction = flee
            self.last_direction = flee
        elif fresh or now - self.last_steer_at >= 150:
            direction = self.steer(pos, goal, hazards, now, allies,
                speed=SPEED * max(0., float(me.get("moveMul", 1))), pull=pull,
                bodies=[(point(e), BODY_RADIUS.get(e.get("mobId"), 1.)) for e in entities
                        if e.get("mobId") in BODY_RADIUS and e.get("alive", True)])
            self.last_steer_at = now
        else:
            direction = self.last_direction
        # Face the actual threat, with exactly the same yaw used to project the
        # world movement vector; no diagonal/sprint speed advantage. A guard only
        # covers 130° in front: while a guardable stroke is on its way, turn to
        # whoever swings it and stay turned until it lands (the holder used to
        # parry the Samurai with its back while facing the horse).
        facing = victim
        for tell in snapshot.get("telegraphs", []):
            if tell.get("landed") or tell.get("attackId") in UNGUARDABLE or tell.get("attackId") == "warden_tensei":
                continue
            source = next((e for e in entities if e.get("entityId") == tell.get("entityId")), None)
            reach = float((tell.get("shape") or {}).get("reach", 4))
            if source and float(tell.get("landInMs", 9999)) <= FACE_ATTACKER_MS and distance(pos, point(source)) <= reach + 2.5:
                facing = source
                break
        for tell in snapshot.get("telegraphs", []):
            if tell.get("landed"):
                self.last_landed[tell.get("entityId")] = now
        yaw = angle_to(pos, point(facing))
        cmds.append({"op": "face", "entityId": int(facing["entityId"])})
        cmds.append({"op": "move", "forward": direction[0] * -math.sin(yaw) + direction[1] * -math.cos(yaw),
            "strafe": direction[0] * math.cos(yaw) - direction[1] * math.sin(yaw), "sprint": False})
        skills = snapshot.get("skills") or {}
        def ready(skill):
            return skill in snapshot.get("unlocked", []) and skill in skills and float(skills[skill].get("readyInMs", 0)) <= 0
        inventory = snapshot.get("inventory") or []
        keepsake = any(item.get("itemId") == "samurai_keepsake" and item.get("quantity", 0) > 0 for item in inventory)
        imminent = [t for t in snapshot.get("telegraphs", []) if not t.get("landed") and t.get("attackId") not in UNGUARDABLE and t.get("attackId") != "warden_tensei"
            and 40 < float(t.get("landInMs", 0)) <= guard_lead_ms(float(t.get("weight", .5)))]
        # A swing occupies the hand: one started now cannot be turned into a guard
        # before a stroke already on its way lands (seen live: a 210 middle cut).
        incoming = any(not t.get("landed") and 0 < float(t.get("landInMs", 0)) <= HOLD_STRIKE_MS
            and t.get("targetId") in (None, me.get("entityId")) and t.get("attackId") != "warden_tensei"
            for t in snapshot.get("telegraphs", []))
        if imminent:
            tell = min(imminent, key=lambda t: t["landInMs"])
            key = f"parry:{tell.get('entityId')}:{tell.get('startedAt')}"
            if self.claim(key, now, 999999):
                cmds += [{"op": "face", "entityId": int(tell["entityId"])}, {"op": "guard", "active": False}, {"op": "guard", "active": True}]
                self.guard_until = now + float(tell["landInMs"]) + 120
        elif now >= self.guard_until and me.get("guarding"):
            cmds.append({"op": "guard", "active": False})
        if phase == "kneel" and any(item.get("itemId") == "holy_water_supreme" and item.get("quantity", 0) for item in inventory) and self.claim("purify", now, 1400):
            cmds.append({"op": "use", "itemId": "holy_water_supreme", "targetId": int(boss["entityId"])})
        if not imminent and ready("provoke") and (holder or keepsake) and self.claim("provoke", now, 1600):
            cmds.append({"op": "skill", "skillId": "provoke", "targetId": int(victim["entityId"])})
        # Reserve the blade and skills through the final 1.2s. The visible hand
        # action supplies its windup; start early enough to land inside the
        # final 180ms with a 50ms transport/tick allowance. Allies wait for clash.
        hands = list((snapshot.get("hands") or {}).values())
        hand = next((h for h in hands if float(h.get("weight", 0)) >= .4 and float(h.get("readyInMs", 0)) <= 0), None)
        windup = float((hand or {}).get("windupMs", 180))
        finale_attack = (main and stage in ("pressure", "timing") and windup + 115 <= remaining <= windup + 180) or stage == "clash"
        if finale_attack and (not holder or not engine):
            blade = visible_blade
            if blade and hand and self.claim("clash:" + cast + ":" + str(stage), now, 900):
                a, b = dict(blade["a"]), dict(blade["b"])
                # Visual tracking only: estimate motion from two samples that
                # have already arrived. No rig sampling at a future timestamp.
                if self.previous_blade and 0 < now - self.previous_blade[0] <= 150 and stage != "clash":
                    previous_at, previous = self.previous_blade
                    for end_name, end in (("a", a), ("b", b)):
                        for axis in ("x", "y", "z"):
                            delta = (end[axis] - previous[end_name][axis]) * (windup + 50) / (now - previous_at)
                            end[axis] += max(-1.2, min(1.2, delta))
                # Aim at the nearest visible blade segment, not at a claimed
                # part id: the ordinary server raycast still decides contact.
                eye = {"x": pos[0], "y": float(me.get("y", 0)) + 1.6, "z": pos[1]}
                vec = {k: b[k] - a[k] for k in ("x", "y", "z")}
                u = max(.15, min(.85, sum((eye[k] - a[k]) * vec[k] for k in vec) / max(.001, sum(v*v for v in vec.values()))))
                aim_point = {k: a[k] + vec[k] * u for k in vec}
                dx, dy, dz = aim_point["x"] - eye["x"], aim_point["y"] - eye["y"], aim_point["z"] - eye["z"]
                aim = {"yaw": math.atan2(-dx, -dz), "pitch": math.atan2(-dy, math.hypot(dx, dz)), "point": aim_point}
                cmds += [{"op": "guard", "active": False}, {"op": "face", "yaw": aim["yaw"], "pitch": aim["pitch"]},
                         {"op": "attack", "hand": hand.get("hand", "left"), "targetId": int(boss["entityId"]), "aim": aim}]
                if stage == "clash" and ready("iron_cleave"):
                    cmds.append({"op": "skill", "skillId": "iron_cleave", "targetId": int(boss["entityId"]), "aim": aim})
            elif not blade or not hand:
                self.stats["clash_missing_visible_blade_or_weight"] = self.stats.get("clash_missing_visible_blade_or_weight", 0) + 1
        elif (not imminent and not incoming and now >= self.guard_until and not (fp and fp != "gather" and remaining < 1200)
              and distance(pos, dest) <= 2.9 and self.punish_window(snapshot, victim, now)):
            # P1/P2 hits maintain normal threat even outside opening; only the
            # keepsake holder provokes the Samurai. Support never steals it.
            if self.claim("strike", now, 950):
                cmds.append({"op": "attack", "targetId": int(victim["entityId"])})
            if samurai.get("awakened") and ready("iron_cleave") and self.claim("cleave", now, 1200):
                cmds.append({"op": "skill", "skillId": "iron_cleave", "targetId": int(victim["entityId"])})
        return cmds


def install_samurai_reflex(layer, gateway, current_task):
    tactics = SamuraiTactics()
    def plan(state):
        latest = gateway.latest()
        if latest is None:
            return None
        commands = tactics.decide(latest.data, current_task())
        if commands is None:
            return None
        return Plan("samurai_visible", (Step(0, Press("samurai:" + json.dumps(commands, separators=(",", ":")).encode().hex(), 1)),), expires_in_ms=150)
    # This dedicated API policy owns Samurai inputs; legacy cover-pulse tactics
    # remain available in other encounters. No desktop backend is touched.
    in_samurai = Condition.any_(Condition.field("zone.id", "==", "samurai_phase"), Condition.field("zone.id", "==", "trial_warden"))
    # The layer fires one reflex per tick; while this one cools down the legacy
    # dodge/parry/approach reflexes would otherwise answer the same telegraph with
    # contradicting moves (seen live: four headings inside 1 ms, then a palm hit).
    for reflex in list(layer):
        if reflex.group != "samurai":
            reflex.condition = Condition.all_(reflex.condition, Condition.not_(in_samurai))
    layer.add(Reflex("samurai_visible", in_samurai, plan,
        group="samurai", priority=Priority.REFLEX + 1, cooldown_ms=40, preempt=True))
    return tactics
