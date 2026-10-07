"""FantCraft journey tester: plays a story chapter end to end through the agent gateway.

Where the warden tactics test one fight, a journey tests the *game around* the
fights: can a character actually walk the authored roads with the server's
real movement, find and talk to every NPC, press the landmarks, take the
crystals, win the story fights, and how long does all of that take?

Everything goes through the gateway (fancraft `scripts/agent-gateway.mts`):
`path` plans over the chunks the server streamed us (A*, same step rules as
the game), `talk`/`choose` drive NPC dialogue, `landmark` presses F, `enter`
opens an instance. Nothing reads server internals; a journey sees what a
player's client sees.

The output is a report: every step's result and duration, every leg's
distance, walking time, replans, stuck points and falls, every death, and the
quest journal at the end. Findings are the stuck points and failures — they
are places a real player would struggle too.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from player.api.gateway import GatewayClient, GatewayError

# --- geometry ------------------------------------------------------------------------------


def yaw_toward(fx: float, fz: float, tx: float, tz: float) -> float:
    """The game's facing convention (shared/physics.ts): forward is (-sin yaw, -cos yaw)."""
    return math.atan2(-(tx - fx), -(tz - fz))


def horizontal(a: dict[str, Any], x: float, z: float) -> float:
    return math.hypot(a["x"] - x, a["z"] - z)


def within_goal(me: dict[str, Any], x: float, z: float, y: float | None, radius: float) -> bool:
    """Arrival is a server position, never an extrapolated step through a wall."""
    actual = me.get("server") or me
    return horizontal(actual, x, z) <= radius and (y is None or abs(actual["y"] - y) <= 3)


def waypoint_tolerance(distance_to_goal: float, radius: float, final: bool, complete: bool) -> float:
    """A final waypoint must bring the body inside the interaction radius.

    Accepting a waypoint 0.7 m early used to cause a tight no-move replan loop
    when a worktable required a closer radius than ordinary NPC dialogue.
    """
    return min(0.7, max(0.15, radius - distance_to_goal - 0.1)) if final and complete else 0.7


def polyline_length(points: Iterable[tuple[float, float]]) -> float:
    pts = list(points)
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


# --- report --------------------------------------------------------------------------------


@dataclass
class LegReport:
    label: str
    target: tuple[float, float]
    ok: bool = False
    seconds: float = 0.0
    straight_m: float = 0.0
    walked_m: float = 0.0
    replans: int = 0
    no_path: int = 0
    path_reasons: dict[str, int] = field(default_factory=dict)
    stuck: list[dict[str, float]] = field(default_factory=list)
    falls: list[dict[str, float]] = field(default_factory=list)
    reason: str = ""


@dataclass
class StepReport:
    kind: str
    label: str
    ok: bool = False
    seconds: float = 0.0
    detail: str = ""


@dataclass
class JourneyReport:
    name: str
    started_unix: float
    steps: list[StepReport] = field(default_factory=list)
    legs: list[LegReport] = field(default_factory=list)
    deaths: list[dict[str, Any]] = field(default_factory=list)
    kills: dict[str, int] = field(default_factory=dict)
    quests: list[dict[str, Any]] = field(default_factory=list)
    # Things the tester had to work around (game content it could not play as a player would)
    findings: list[str] = field(default_factory=list)
    character: str = ""
    finished: bool = False
    seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        walked = sum(leg.walked_m for leg in self.legs)
        walk_s = sum(leg.seconds for leg in self.legs)
        stuck = [dict(s, leg=leg.label) for leg in self.legs for s in leg.stuck]
        return {
            "name": self.name, "finished": self.finished, "seconds": round(self.seconds, 1),
            "steps_ok": sum(s.ok for s in self.steps), "steps": len(self.steps),
            "first_failure": next((asdict(s) for s in self.steps if not s.ok), None),
            "walked_m": round(walked), "walking_s": round(walk_s), "mean_speed_mps": round(walked / walk_s, 2) if walk_s else 0,
            "legs": len(self.legs), "legs_failed": [leg.label for leg in self.legs if not leg.ok],
            "stuck_points": len(stuck), "worst_stuck": sorted(stuck, key=lambda s: -s["seconds"])[:8],
            "falls": sum(len(leg.falls) for leg in self.legs), "deaths": len(self.deaths),
            "findings": list(self.findings),
            "slowest_legs": sorted(({"leg": leg.label, "s": round(leg.seconds), "m": round(leg.walked_m),
                                     "detour": round(leg.walked_m / leg.straight_m, 2) if leg.straight_m > 1 else None}
                                    for leg in self.legs), key=lambda r: -r["s"])[:6],
        }


# --- the journey ---------------------------------------------------------------------------


class JourneyFailed(RuntimeError):
    pass


class Journey:
    """Primitive actions over one gateway connection. Steps call these; failures raise."""

    TICK_S = 0.05

    def __init__(self, gw: GatewayClient, name: str, log: Callable[[str], None] = print) -> None:
        self.gw = gw
        self.log = log
        self.report = JourneyReport(name=name, started_unix=time.time())
        self._event_cursor = gw.event_cursor
        self._last_downed = False
        #: NPC companions to call into each instance through the party panel (docs/COMBAT.md §7.3); 0 = solo
        self.companions = 0

    # -- state --------------------------------------------------------------------------------

    def snap(self) -> dict[str, Any]:
        s = self.gw.latest()
        if s is None:
            raise JourneyFailed("no snapshot from the gateway")
        self._watch_death(s.data)
        return s.data

    def me(self) -> dict[str, Any]:
        return self.snap()["self"]

    def quest_state(self, quest_id: str) -> str | None:
        for q in self.snap().get("quests") or []:
            if q.get("id") == quest_id:
                return q.get("state")
        return None

    def quest_entry(self, quest_id: str) -> dict[str, Any] | None:
        return next((q for q in self.snap().get("quests") or [] if q.get("id") == quest_id), None)

    def count(self, item_id: str) -> int:
        """How many of an item are in the bag."""
        return sum(int(s.get("quantity", 0)) for s in self.snap().get("inventory") or [] if s.get("itemId") == item_id)

    def zone(self) -> str:
        return str(self.snap().get("zone") or "")

    def entity(self, pred: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
        best = None
        for e in self.snap().get("entities") or []:
            if pred(e) and (best is None or e.get("distance", 1e9) < best.get("distance", 1e9)):
                best = e
        return best

    def npc(self, npc_id: str) -> dict[str, Any] | None:
        return self.entity(lambda e: e.get("type") == "npc" and e.get("npcId") == npc_id)

    def _watch_death(self, data: dict[str, Any]) -> None:
        me = data.get("self") or {}
        if me.get("downed") and not self._last_downed:
            self.report.deaths.append({"t": round(time.time() - self.report.started_unix, 1), "zone": data.get("zone"),
                                       "x": round(me.get("x", 0), 1), "y": round(me.get("y", 0), 1), "z": round(me.get("z", 0), 1)})
            self.log(f"  ! downed at {me.get('x', 0):.0f},{me.get('z', 0):.0f}")
        self._last_downed = bool(me.get("downed"))

    def wait_event(self, type_: str, pred: Callable[[dict[str, Any]], bool] = lambda d: True, timeout_s: float = 8.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for ev in self.gw.events_since(self._event_cursor):
                self._event_cursor = max(self._event_cursor, ev.seq)
                if ev.type == type_ and pred(ev.data):
                    return ev.data
            time.sleep(0.02)
        raise JourneyFailed(f"no {type_} within {timeout_s:.0f}s")

    def drain_events(self) -> None:
        self._event_cursor = self.gw.event_cursor

    def stop(self) -> None:
        self.gw.fire("move", forward=0, strafe=0)

    # -- movement -----------------------------------------------------------------------------

    def walk_to(self, x: float, z: float, y: float | None = None, radius: float = 1.5, label: str = "",
                timeout_s: float | None = None) -> LegReport:
        """Walk to (x, z) on the ground: plan, follow, replan; jump when wedged, replan when stuck."""
        start = self.me()
        leg = LegReport(label=label or f"to {x:.0f},{z:.0f}", target=(x, z), straight_m=horizontal(start, x, z))
        timeout = timeout_s if timeout_s is not None else 60 + leg.straight_m * 0.6
        t0 = time.monotonic()
        s0 = start.get("server") or start
        last = (s0["x"], s0["y"], s0["z"])
        stuck_since: float | None = None
        stuck_logged = False
        last_plan: dict[str, Any] | None = None
        drop_wait_since = math.inf  # when we started waiting over a drop waypoint for the fall
        try:
            while time.monotonic() - t0 < timeout:
                me = self.me()
                if within_goal(me, x, z, y, radius):
                    leg.ok = True
                    return leg
                if me.get("downed"):
                    self.stop()
                    time.sleep(0.5)
                    continue
                try:
                    plan = self.gw.call("path", timeout_s=10, x=x, z=z, radius=radius, **({"y": y} if y is not None else {}))
                except GatewayError as exc:
                    raise JourneyFailed(f"path failed: {exc}") from exc
                waypoints = plan.get("waypoints") or []
                last_plan = plan
                leg.replans += 1
                if not plan.get("ok"):
                    reason = str(plan.get("reason") or "?")
                    leg.path_reasons[reason] = leg.path_reasons.get(reason, 0) + 1
                if not waypoints:
                    # Nothing known that gets closer: steer straight for a moment (new chunks may arrive)
                    leg.no_path += 1
                    waypoints = [{"x": x, "y": me["y"], "z": z}]
                follow_until = time.monotonic() + 20
                for wp in waypoints:
                    reached = False
                    while time.monotonic() < follow_until and time.monotonic() - t0 < timeout:
                        me = self.me()
                        if me.get("downed"):
                            break
                        # distance and falls from the server's corrected position, not the gateway's prediction
                        srv = me.get("server") or me
                        pos = (srv["x"], srv["y"], srv["z"])
                        leg.walked_m += math.hypot(pos[0] - last[0], pos[2] - last[2])
                        if last[1] - pos[1] > 4:
                            leg.falls.append({"x": round(pos[0], 1), "z": round(pos[2], 1), "drop": round(last[1] - pos[1], 1)})
                        last = pos
                        if horizontal(srv if wp is waypoints[-1] else me, wp["x"], wp["z"]) < waypoint_tolerance(
                            math.hypot(wp["x"] - x, wp["z"] - z), radius, wp is waypoints[-1], bool(plan.get("ok"))):
                            if "y" in wp and srv["y"] - wp["y"] > 1.5 and time.monotonic() - drop_wait_since < 1.5:
                                # the waypoint is a drop (a stair pit, a ledge): stop over it and let the body fall
                                # before heading on, or a sprint carries us across the gap onto its far rim
                                if drop_wait_since == math.inf:
                                    drop_wait_since = time.monotonic()
                                self.gw.fire("move", forward=0, strafe=0)
                                time.sleep(self.TICK_S)
                                continue
                            drop_wait_since = math.inf
                            reached = True
                            break
                        drop_wait_since = math.inf
                        yaw = yaw_toward(me["x"], me["z"], wp["x"], wp["z"])
                        self.gw.fire("face", yaw=yaw)
                        stuck_ms = me.get("stuckMs", 0)
                        jump = 700 < stuck_ms < 2600 and int(stuck_ms / 350) % 2 == 0
                        self.gw.fire("move", forward=1, strafe=0, sprint=horizontal(me, x, z) > 4, jump=jump)
                        if stuck_ms > 700:
                            stuck_since = stuck_since or time.monotonic()
                            if not stuck_logged and time.monotonic() - stuck_since > 1.5:
                                stuck_logged = True
                            if stuck_ms > 2600:
                                stuck_since = min(stuck_since, time.monotonic() - 2.7)  # wedged already: back off now
                                break  # replan from here
                        elif stuck_since is not None:
                            self._log_stuck(leg, me, time.monotonic() - stuck_since, stuck_logged)
                            stuck_since, stuck_logged = None, False
                        time.sleep(self.TICK_S)
                    if not reached:
                        break
                if stuck_since is not None and time.monotonic() - stuck_since > 2.6:
                    self._log_stuck(leg, self.me(), time.monotonic() - stuck_since, True)
                    stuck_since, stuck_logged = None, False
                    # back off a step and sidestep before replanning
                    self.gw.fire("move", forward=-1, strafe=1 if leg.replans % 2 else -1, sprint=False, jump=True)
                    time.sleep(0.6)
            leg.reason = "timeout"
            # what a human would want to see about a place the walker could not pass
            me = self.me()
            wps = [(round(w["x"], 1), round(w["y"], 1), round(w["z"], 1)) for w in (last_plan or {}).get("waypoints", [])[:5]]
            foes = [(e.get("mobId"), round(e.get("distance", 0), 1)) for e in self._hostiles(12.0)]
            self.log(f"  ? stuck at {me['x']:.1f},{me['y']:.1f},{me['z']:.1f} hp {me.get('hp')}/{me.get('maxHp')}; "
                     f"last plan ok={(last_plan or {}).get('ok')} len={(last_plan or {}).get('length')} first waypoints {wps}; mobs near {foes}")
            return leg
        finally:
            self.stop()
            leg.seconds = time.monotonic() - t0
            self.report.legs.append(leg)
            state = "ok" if leg.ok else f"FAILED ({leg.reason})"
            self.log(f"  leg {leg.label}: {state} {leg.seconds:.0f}s {leg.walked_m:.0f}m, {leg.replans} plans, {len(leg.stuck)} stuck")

    def _log_stuck(self, leg: LegReport, me: dict[str, Any], seconds: float, significant: bool) -> None:
        if significant or seconds > 1.0:
            leg.stuck.append({"x": round(me["x"], 1), "y": round(me["y"], 1), "z": round(me["z"], 1), "seconds": round(seconds, 1)})

    def walk_route(self, points: list[tuple[float, float]], label: str, radius: float = 2.5) -> bool:
        """Follow a coarse route (road polyline or hand-picked via points) point by point."""
        ok = True
        for i, (px, pz) in enumerate(points):
            last = i == len(points) - 1
            leg = self.walk_to(px, pz, radius=1.5 if last else radius, label=f"{label} {i + 1}/{len(points)}")
            ok = ok and leg.ok
            if not leg.ok:
                return False
        return ok

    # -- interaction --------------------------------------------------------------------------

    def approach_npc(self, npc_id: str, via: list[tuple[float, float]] | None = None) -> dict[str, Any]:
        if via:
            if not self.walk_route(via, f"via to {npc_id}"):
                raise JourneyFailed(f"could not walk the route to {npc_id}")
        npc = self.npc(npc_id)
        if npc is None and npc_id in NPC_POSTS:
            # Not streamed to us yet: walk toward where the map says they stand, then look again
            px, py, pz = NPC_POSTS[npc_id]
            self.walk_to(px, pz, y=py, radius=6, label=f"toward {npc_id}'s post")
            npc = self.npc(npc_id)
        if npc is None:
            raise JourneyFailed(f"NPC {npc_id} is not in view")
        leg = self.walk_to(npc["x"], npc["z"], y=npc["y"], radius=2.2, label=f"to {npc_id}")
        if not leg.ok:
            raise JourneyFailed(f"could not reach {npc_id}")
        return npc

    def quest(self, npc_id: str, quest_id: str, action: str, via: list[tuple[float, float]] | None = None,
              already_done_ok: bool = False) -> str | None:
        """Accept or complete a quest at an NPC through the real dialogue.

        `already_done_ok`: a turn-in that a recorded workaround already completed is skipped, not failed.
        """
        if already_done_ok and self.quest_state(quest_id) == "completed":
            return "skipped: already completed (see findings)"
        npc = self.approach_npc(npc_id, via)
        self.settle()
        self.drain_events()
        self.gw.fire("face", entityId=npc["entityId"])
        self.gw.call("talk", entityId=npc["entityId"])
        menu = self.wait_event("npc_dialog", lambda d: d.get("entityId") == npc["entityId"] and bool(d.get("options")))
        ids = [o.get("id") for o in menu.get("options") or []]
        if quest_id not in ids:
            raise JourneyFailed(f"{npc_id} does not offer {quest_id} (options {ids})")
        self.gw.call("choose", entityId=npc["entityId"], nodeId=menu["nodeId"], optionId=quest_id)
        node = self.wait_event("npc_dialog", lambda d: str(d.get("nodeId", "")).startswith(f"quest:{quest_id}") and bool(d.get("options")))
        opts = [o.get("id") for o in node.get("options") or []]
        if action not in opts:
            raise JourneyFailed(f"{quest_id} at {npc_id}: no '{action}' (options {opts})")
        want = "completed" if action == "complete" or action.startswith("complete:") else None
        self.gw.call("choose", entityId=npc["entityId"], nodeId=node["nodeId"], optionId=action)
        self.wait_event("quest_update", lambda d: any(q.get("id") == quest_id and (want is None or q.get("state") == want)
                                                       for q in d.get("journal") or []), timeout_s=8)
        self.gw.fire("face")
        self.gw.fire("close_dialog")

    def landmark(self, landmark_id: str, x: float, z: float, via: list[tuple[float, float]] | None = None) -> str:
        if via and not self.walk_route(via, f"via to {landmark_id}"):
            raise JourneyFailed(f"could not walk the route to {landmark_id}")
        for attempt, radius in enumerate((2.2, 1.2, 0.8)):
            leg = self.walk_to(x, z, radius=radius, label=f"to {landmark_id}" + (f" (retry {attempt})" if attempt else ""))
            if not leg.ok:
                raise JourneyFailed(f"could not reach {landmark_id}")
            self.settle()
            self.drain_events()
            self.gw.call("landmark", landmarkId=landmark_id)
            try:
                msg = self.wait_event("system_message", timeout_s=3)
                return str((msg.get("loc") or {}).get("key") or msg.get("message", ""))
            except JourneyFailed:
                me = self.me()
                self.log(f"  no answer from {landmark_id} at {me['x']:.1f},{me['y']:.1f},{me['z']:.1f}; closing in")
        raise JourneyFailed(f"{landmark_id} never answered F")

    def settle(self, timeout_s: float = 2.0) -> None:
        """Stand still until the server's corrections say we have stopped (our position is a prediction)."""
        self.stop()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.me().get("speed", 0) > 0.3:
            time.sleep(0.05)
        time.sleep(0.2)

    def travel(self, crystal: tuple[float, float], dest: str) -> None:
        """Walk to a crystal and /travel; read the server's refusal and act on it (closer, out of combat)."""
        leg = self.walk_to(crystal[0], crystal[1], radius=3.0, label=f"to crystal for {dest}")
        if not leg.ok:
            raise JourneyFailed("could not reach the crystal")
        before = self.me()
        time.sleep(1.2)  # attunement ticks once a second
        replies: list[str] = []
        for attempt in range(6):
            self.settle()
            self.drain_events()
            self.gw.call("chat", message=f"/travel {dest}")
            deadline = time.monotonic() + 6
            reply = ""
            while time.monotonic() < deadline:
                if horizontal(self.me(), before["x"], before["z"]) > 30:
                    return
                for ev in self.gw.events_since(self._event_cursor):
                    self._event_cursor = max(self._event_cursor, ev.seq)
                    if ev.type == "system_message":
                        key = str((ev.data.get("loc") or {}).get("key") or ev.data.get("message") or "")
                        if key.startswith("sys_travel") and key != "sys_travel_success":
                            reply = key
                if reply:
                    break
                time.sleep(0.1)
            replies.append(reply or "silence")
            if reply == "sys_travel_not_at_crystal":
                self.walk_to(crystal[0], crystal[1], radius=1.5, label=f"closer to crystal for {dest}")
            elif reply == "sys_travel_in_combat":
                # The gateway has no inCombat flag; the server's reply is the observable state.
                # Deal with nearby pursuers, then retry after allowing threat to decay.
                here = self.me()
                pursuers = [e for e in self._hostiles(14) if abs(e["y"] - here["y"]) < 4]
                self.log("  crystal refused combat; nearby: " + str([(e.get("mobId"), round(e["distance"], 1)) for e in pursuers]))
                if pursuers:
                    self.engage(pursuers[0], timeout_s=60)
                else:
                    self.stop(); time.sleep(15)
                self.walk_to(crystal[0], crystal[1], radius=1.5, label=f"return to crystal for {dest}")
            else:
                time.sleep(1.0)
        raise JourneyFailed(f"/travel {dest} did not move us (replies: {', '.join(replies)})")

    def gear_up(self, counter_npc: str, items: list[tuple[str, str]]) -> str:
        """Buy (items, hand) at a counter and equip them: left/right as the guild would."""
        self.approach_npc(counter_npc)
        for item_id, _ in items:
            self.gw.call("shop_buy", itemId=item_id, quantity=1)
            self.wait_event("shop_result", lambda d, i=item_id: d.get("itemId") == i or d.get("success") is not None, timeout_s=5)
        for item_id, hand in items:
            self.gw.call("equip", itemId=item_id, hand=hand)
            time.sleep(0.3)
        eq = (self.snap().get("equipment") or {}).get("equipment") or {}
        held = {k: v.get("itemId") for k, v in eq.items() if k.endswith("_hand")}
        if not all(held.get(f"{hand}_hand") == item_id for item_id, hand in items):
            raise JourneyFailed(f"equipment did not take: {held}")
        return f"holding {held}"

    # -- combat -------------------------------------------------------------------------------

    def fight(self, target: dict[str, Any], timeout_s: float = 90) -> bool:
        """Melee a target until it dies: close in, strike alternately, guard a strike aimed at us, dodge ground markers."""
        eid = target["entityId"]
        t0 = time.monotonic()
        hand = "left"
        next_strike = 0.0
        downed = False
        while time.monotonic() - t0 < timeout_s:
            s = self.snap()
            me = s["self"]
            if me.get("downed"):
                downed = True
                self.stop()
                time.sleep(0.5)
                continue
            if downed:
                # Back on our feet somewhere else (home, or the instance door): the caller re-plans the approach
                self.gw.fire("face")
                return False
            foe = next((e for e in s.get("entities") or [] if e["entityId"] == eid), None)
            if foe is None or not foe.get("alive"):
                self.stop()
                self.gw.fire("face")
                return True
            dodge = self._dodge_vector(s)
            self.gw.fire("face", entityId=eid)
            if dodge is not None:
                self.gw.fire("move", forward=dodge[0], strafe=dodge[1], sprint=True)
            elif foe["distance"] > 2.4:
                stuck = me.get("stuckMs", 0) > 700
                self.gw.fire("move", forward=1, strafe=0, sprint=True, toward=eid, jump=stuck)
            else:
                self.gw.fire("move", forward=0, strafe=0)
            incoming = any(t.get("targetId") == me.get("entityId") and 0 < t.get("landInMs", 0) < 450 and not t.get("landed")
                           for t in s.get("telegraphs") or [])
            now = time.monotonic()
            if incoming:
                self.gw.fire("guard", active=True)
            elif foe["distance"] <= 3.0 and now >= next_strike:
                self.gw.fire("guard", active=False)
                skill = self._ready_skill(s)
                if skill is not None:
                    # the job's hotbar (1-2-3 …) hits far harder than a bare swing; the cast lowers both hands
                    self.gw.fire("skill", skillId=skill, targetId=eid)
                    next_strike = now + 0.9
                else:
                    self.gw.fire("attack", hand=hand)
                    hand = "right" if hand == "left" else "left"
                    next_strike = now + 0.5
            time.sleep(self.TICK_S)
        self.stop()
        return False

    def _ready_skill(self, s: dict[str, Any]) -> str | None:
        """The best unlocked melee skill whose recast is up (the guardian's 1-2-3 plus the stun), if any."""
        unlocked = set(s.get("unlocked") or [])
        skills = s.get("skills") or {}
        for skill_id in MELEE_SKILLS:
            if skill_id in unlocked and (skills.get(skill_id) or {}).get("readyInMs", 1) <= 0:
                return skill_id
        return None

    def rest(self, min_frac: float = 0.9, timeout_s: float = 30.0) -> None:
        """Out of combat HP comes back at 5%/s: stand still before the next pull instead of starting it hurt."""
        t0 = time.monotonic()
        me = self.me()
        if not me.get("maxHp") or me.get("hp", 0) >= me["maxHp"] * min_frac:
            return
        self.stop()
        last_hp, last_gain = me.get("hp", 0), time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            me = self.me()
            if me.get("downed") or me.get("hp", 0) >= me.get("maxHp", 0) * min_frac:
                break
            if me.get("hp", 0) > last_hp:
                last_hp, last_gain = me["hp"], time.monotonic()
            elif time.monotonic() - last_gain > 4:
                break  # not regenerating: still in combat with something
            time.sleep(0.25)
        self.log(f"  rested {time.monotonic() - t0:.0f}s → {me.get('hp', 0):.0f}/{me.get('maxHp', 0):.0f} HP")

    def _dodge_vector(self, s: dict[str, Any]) -> tuple[float, float] | None:
        """(forward, strafe) to leave a marker we stand in, relative to our facing; None when safe."""
        me = s["self"]
        for m in s.get("markers") or []:
            pos = m.get("position")
            if not pos or m.get("remainingMs", 0) > 4200:
                continue
            if m.get("markerType") == "line_aoe":
                yaw = m.get("yaw", 0.0)
                fx, fz = -math.sin(yaw), -math.cos(yaw)
                dx, dz = me["x"] - pos["x"], me["z"] - pos["z"]
                along, across = dx * fx + dz * fz, dx * fz - dz * fx
                if -0.4 <= along <= m.get("length", 0.0) + 0.4 and abs(across) <= m.get("width", 0.0) / 2 + 0.8:
                    # Exit through the closest SIDE, not along a twenty-metre strip.
                    sign = 1 if across >= 0 else -1
                    wx, wz = fz * sign, -fx * sign
                    facing = me.get("yaw", 0.0)
                    return (wx * -math.sin(facing) + wz * -math.cos(facing),
                            wx * math.cos(facing) + wz * -math.sin(facing))
                continue
            d = math.hypot(me["x"] - pos["x"], me["z"] - pos["z"])
            inner = m.get("innerRadius") or 0.0
            outer = (m.get("radius") or 0.0) + 0.8
            if inner and inner - 0.4 < d < outer:
                return (1.0, 0.0) if d < (inner + outer) / 2 and d > 0.5 else (-1.0, 0.0)  # into the safe middle, or back out
            if not inner and d < outer:
                return (-1.0, 0.6)  # back away from its centre while facing the foe
        return None

    def kill(self, def_id: str, count: int, near: tuple[float, float] | None = None, timeout_s: float = 300) -> int:
        killed = 0
        t0 = time.monotonic()
        while killed < count and time.monotonic() - t0 < timeout_s:
            target = self.entity(lambda e: e.get("type") == "mob" and e.get("mobId") == def_id and e.get("alive"))
            if target is None:
                if near:
                    self.walk_to(near[0], near[1], radius=4, label=f"search {def_id}")
                time.sleep(1.0)
                continue
            if target["distance"] > 12:
                self.walk_to(target["x"], target["z"], radius=6, label=f"approach {def_id}")
            if self.fight(target):
                killed += 1
                self.report.kills[def_id] = self.report.kills.get(def_id, 0) + 1
                self.log(f"  killed {def_id} ({killed}/{count})")
        return killed

    def instance(self, zone: str, boss_id: str, timeout_s: float = 240) -> bool:
        """Enter a story instance, defeat its boss, return to the overworld."""
        self.drain_events()
        self.gw.call("enter", timeout_s=40, zone=zone)
        time.sleep(1.5)
        ok = False
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            boss = self.entity(lambda e: e.get("type") == "mob" and e.get("mobId") == boss_id)
            if boss is None:
                time.sleep(0.5)
                continue
            if not boss.get("alive"):
                ok = True
                break
            ok = self.fight(boss, timeout_s=timeout_s - (time.monotonic() - t0))
            break
        if ok:
            try:
                self.wait_event("dungeon_complete", timeout_s=10)
            except JourneyFailed:
                ok = False
        self.gw.call("enter", timeout_s=40, zone="overworld")
        time.sleep(2.0)
        return ok

    # -- campaign helpers (prologue → Act II) -------------------------------------------------

    def _hostiles(self, near_me_m: float) -> list[dict[str, Any]]:
        """Live mobs within `near_me_m` of us that can fight back (not dummies), nearest first."""
        return sorted((e for e in self.snap().get("entities") or []
                       if e.get("type") == "mob" and e.get("alive") and e.get("distance", 1e9) <= near_me_m
                       and e.get("mobId") not in PASSIVE_MOBS), key=lambda e: e.get("distance", 1e9))

    def engage(self, target: dict[str, Any], y: float | None = None, timeout_s: float = 90) -> bool:
        """Walk (planned, not straight) to within striking reach of a target, then fight it."""
        if target.get("distance", 0) > 8:
            self.walk_to(target["x"], target["z"], y=y, radius=5, label=f"approach {target.get('mobId')}",
                         timeout_s=40 + target.get("distance", 0) * 0.6)
        fresh = next((e for e in self.snap().get("entities") or [] if e["entityId"] == target["entityId"]), None)
        if fresh is None or not fresh.get("alive"):
            return True
        return self.fight(fresh, timeout_s=timeout_s)

    def _next_target(self, def_id: str, near: tuple[float, float], search_m: float,
                     clear_first: tuple[str, ...] = ()) -> dict[str, Any] | None:
        """Anything already on us first (a pack does not wait its turn), then the guards named in
        `clear_first` (so they are not pulled into the main fight), else the nearest wanted mob near `near`."""
        close = self._hostiles(6.0)
        if close:
            return close[0]

        def around(ids: tuple[str, ...]) -> list[dict[str, Any]]:
            return [e for e in self.snap().get("entities") or []
                    if e.get("type") == "mob" and e.get("mobId") in ids and e.get("alive")
                    and math.hypot(e["x"] - near[0], e["z"] - near[1]) <= search_m]
        for ids in (clear_first, (def_id,)):
            found = around(ids) if ids else []
            if found:
                return min(found, key=lambda e: e.get("distance", 1e9))
        return None

    def hunt(self, done: Callable[[], bool], def_id: str, near: tuple[float, float], y: float | None = None,
             back: Callable[[], Any] | None = None, search_m: float = 40, timeout_s: float = 360,
             clear_first: tuple[str, ...] = ()) -> str:
        """Fight `def_id` around `near` (and whatever jumps us on the way) until `done()`."""
        t0 = time.monotonic()
        fights = 0
        deaths0 = len(self.report.deaths)
        while time.monotonic() - t0 < timeout_s:
            if done():
                return f"{fights} fights, {len(self.report.deaths) - deaths0} deaths"
            me = self.me()
            if me.get("downed"):
                time.sleep(0.5)
                continue
            if horizontal(me, near[0], near[1]) > search_m + 10:
                # respawned far away (home): walk back the way a player would
                if back is not None and horizontal(me, near[0], near[1]) > 60:
                    back()
                self.walk_to(near[0], near[1], y=y, radius=5, label="to the hunting ground")
                continue
            target = self._next_target(def_id, near, search_m, clear_first)
            if target is None:
                if horizontal(me, near[0], near[1]) > 5:
                    self.walk_to(near[0], near[1], y=y, radius=4, label=f"search {def_id}")
                time.sleep(1.0)  # respawn timers
                continue
            if not self._hostiles(10.0):
                self.rest()  # never start the next pull hurt
            fights += 1
            won = self.engage(target, y=y)
            if won and target.get("mobId"):
                self.report.kills[target["mobId"]] = self.report.kills.get(target["mobId"], 0) + 1
                self.log(f"  down: {target['mobId']}")
            time.sleep(0.3)  # loot and the quest tally land a tick after the death
        raise JourneyFailed(f"hunting {def_id} timed out after {fights} fights")

    def kill_for_quest(self, quest_id: str, def_id: str, near: tuple[float, float], **kw: Any) -> str:
        """Kill `def_id` until the quest's tally reads ready (the server's count, not ours)."""
        detail = self.hunt(lambda: self.quest_state(quest_id) in ("ready", "completed"), def_id, near, **kw)
        q = self.quest_entry(quest_id) or {}
        return f"{quest_id} {q.get('have')}/{q.get('need')} — {detail}"

    def collect_drops(self, item_id: str, quantity: int, def_id: str, near: tuple[float, float], **kw: Any) -> str:
        """Hunt a mob until its drops put `quantity` of an item in the bag."""
        if self.count(item_id) >= quantity:
            return f"already holding {self.count(item_id)} {item_id}"
        detail = self.hunt(lambda: self.count(item_id) >= quantity, def_id, near, **kw)
        return f"holding {self.count(item_id)} {item_id} — {detail}"

    def mine(self, item_id: str, quantity: int, block_id: int, near: tuple[float, float], y: float,
             radius: float = 16, timeout_s: float = 180) -> str:
        """Find ore cells among the streamed chunks, stand within reach, break them until the bag holds enough."""
        t0 = time.monotonic()
        tried: set[tuple[int, int, int]] = set()
        broken = 0
        while self.count(item_id) < quantity:
            if time.monotonic() - t0 > timeout_s:
                raise JourneyFailed(f"mined {broken}, holding {self.count(item_id)}/{quantity} {item_id} when time ran out")
            found = self.gw.call("find_blocks", blockId=block_id, x=near[0], z=near[1], y=y, radius=radius, height=6).get("blocks") or []
            me = self.me()
            # only seams a player can see: ore buried in the rock is not a target
            seams = sorted((b for b in found if b.get("exposed", True) and (b["x"], b["y"], b["z"]) not in tried),
                           key=lambda b: math.hypot(b["x"] + 0.5 - me["x"], b["z"] + 0.5 - me["z"]))
            if not seams:
                raise JourneyFailed(f"no {item_id} seam left within {radius:.0f} m of {near} ({len(tried)} tried, holding {self.count(item_id)})")
            b = seams[0]
            tried.add((b["x"], b["y"], b["z"]))
            leg = self.walk_to(b["x"] + 0.5, b["z"] + 0.5, y=b["y"], radius=2.6, label=f"to seam {b['x']},{b['y']},{b['z']}")
            if not leg.ok:
                continue
            self.settle()
            before = self.count(item_id)
            self.drain_events()
            self.gw.call("break", x=b["x"], y=b["y"], z=b["z"])
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and self.count(item_id) <= before:
                time.sleep(0.1)
            if self.count(item_id) > before:
                broken += 1
                self.log(f"  seam {b['x']},{b['y']},{b['z']} → {item_id} ({self.count(item_id)}/{quantity})")
            else:
                self.log(f"  seam {b['x']},{b['y']},{b['z']} gave nothing")
        return f"mined {broken} seams, holding {self.count(item_id)} {item_id}"

    def party(self) -> dict[str, Any] | None:
        """The party panel's state (docs/COMBAT.md §7.3), asked for fresh."""
        self.gw.call("party", timeout_s=5)
        time.sleep(0.4)
        return self.snap().get("party")

    def call_companions(self, wanted: int, timeout_s: float = 120) -> int:
        """Press "call an NPC companion" up to `wanted` times (capped by the instance's headcount) and wait for them."""
        party = self.party() or {}
        if not party.get("enabled") or not party.get("service"):
            self.report.findings.append(f"{self.zone()}: no companion service (enabled={party.get('enabled')}, "
                                        f"service={party.get('service')}) — fighting alone")
            return 0
        room = max(0, int(party.get("cap", 1)) - len(party.get("members") or []) - int(party.get("pending", 0)))
        n = min(wanted, room)
        for _ in range(n):
            self.gw.call("summon_companion", timeout_s=5)
            time.sleep(0.3)
        t0 = time.monotonic()
        here = 0
        while time.monotonic() - t0 < timeout_s:
            party = self.party() or {}
            here = sum(1 for m in party.get("members") or [] if m.get("companion") and m.get("mine"))
            if here >= n:
                break
            time.sleep(1.5)
        self.log(f"  companions: {here}/{n} arrived in {time.monotonic() - t0:.0f}s")
        return here

    def dismiss_companions(self) -> None:
        try:
            self.gw.call("dismiss_companion", timeout_s=5)
        except Exception:  # noqa: BLE001 - leaving is best effort
            pass

    def clear_instance(self, zone: str, boss_id: str, timeout_s: float = 420, attempts: int = 2) -> str:
        """Enter an instance and fight through it — trash as it comes, then the boss — until it is cleared.

        With `self.companions`, NPC companions are called through the party panel first. When every player
        character is down the run resets inside the instance (docs/COMBAT.md §7.3): each reset spends an
        attempt; being thrown out does too, and then the next attempt is a fresh instance.
        """
        boss_low = 1.0
        why = ""
        resets = 0
        for attempt in range(1, attempts + 1):
            deaths0 = len(self.report.deaths)
            self.drain_events()
            cursor = self.gw.event_cursor
            self.gw.call("enter", timeout_s=40, zone=zone)
            time.sleep(1.5)
            called = self.call_companions(self.companions) if self.companions > 0 else 0
            t0 = time.monotonic()
            cleared = False
            try:
                while time.monotonic() - t0 < timeout_s:
                    evs = self.gw.events_since(cursor)
                    if evs:
                        cursor = evs[-1].seq
                    if any(ev.type == "dungeon_complete" for ev in evs):
                        cleared = True
                        break
                    if any(ev.type == "instance_reset" for ev in evs):
                        resets += 1
                        self.log(f"  wipe → the run reset ({resets}/{attempts})")
                        if resets >= attempts:
                            why = f"wiped {resets} times"
                            break
                    s = self.snap()
                    if s.get("zone") != zone:
                        why = f"thrown out of {zone} (zone {s.get('zone')})"
                        break
                    me = s["self"]
                    if me.get("downed"):
                        time.sleep(0.5)
                        continue
                    boss = self.entity(lambda e: e.get("type") == "mob" and e.get("mobId") == boss_id)
                    if boss is not None and boss.get("maxHp"):
                        boss_low = min(boss_low, boss["hp"] / boss["maxHp"])
                    target = self._hostiles(1e9)[0] if self._hostiles(1e9) else None
                    if target is None:
                        time.sleep(0.5)  # everything down: completion is a broadcast away
                        continue
                    if target.get("distance", 0) > 12:
                        self.rest()  # between pulls: full health before the next pack
                    self.engage(target, timeout_s=max(10.0, timeout_s - (time.monotonic() - t0)))
                else:
                    why = f"not cleared in {timeout_s:.0f}s"
            finally:
                self.stop()
                if called:
                    self.dismiss_companions()
                deaths = len(self.report.deaths) - deaths0
                if self.zone() != "overworld" or not cleared:
                    self.gw.call("enter", timeout_s=40, zone="overworld")
                    time.sleep(2.0)
            if cleared:
                if deaths >= 3:
                    # dying does not reset the fight here, so a clear can be an attrition run: worth knowing
                    self.report.findings.append(f"{zone}: cleared solo only by attrition — {deaths} deaths "
                                                f"(respawning inside the instance does not reset the boss)")
                return f"cleared on attempt {attempt}, {deaths} deaths"
            why = why or "wiped"
            self.log(f"  attempt {attempt} at {zone} failed: {why}; {deaths} deaths; boss low {boss_low:.0%}")
            if resets >= attempts:
                break
        raise JourneyFailed(f"{zone} not cleared in {attempts} attempts ({why}; boss at best {boss_low:.0%} HP)")

    def clear_or_workaround(self, zone: str, boss_id: str, quest_id: str, reason: str, **kw: Any) -> str:
        """Try an encounter a solo player may not be able to win; if it fails, record why and skip it with the
        playtest `/quest done` so the rest of the story can still be tested."""
        try:
            return self.clear_instance(zone, boss_id, **kw)
        except JourneyFailed as exc:
            finding = f"{quest_id}: {reason} — solo attempt failed ({exc}); skipped with /quest done {quest_id}"
            self.report.findings.append(finding)
            self.log(f"  ! {finding}")
            self.drain_events()
            self.gw.call("chat", message=f"/quest done {quest_id}")
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline and self.quest_state(quest_id) != "completed":
                time.sleep(0.1)
            if self.quest_state(quest_id) != "completed":
                raise JourneyFailed(f"{exc}; the /quest done workaround did not take (server not in playtest mode?)") from exc
            return f"WORKAROUND: {exc}"

    def equip_items(self, items: list[tuple[str, str]]) -> str:
        """Ensure requested hands are equipped, including gear restored on login."""
        eq = (self.snap().get("equipment") or {}).get("equipment") or {}
        missing = [(item_id, hand) for item_id, hand in items
                   if (eq.get(f"{hand}_hand") or {}).get("itemId") != item_id]
        deadline = time.monotonic() + 4  # a reward reaches the bag a snapshot or two after the quest_update
        while time.monotonic() < deadline and any(self.count(i) < 1 for i, _ in missing):
            time.sleep(0.1)
        for item_id, hand in missing:
            if self.count(item_id) < 1:
                bag = sorted({str(s.get("itemId")) for s in self.snap().get("inventory") or []})
                raise JourneyFailed(f"no {item_id} in the bag (holding {bag})")
        for item_id, hand in missing:
            self.gw.call("equip", itemId=item_id, hand=hand)
            time.sleep(0.4)
        eq = (self.snap().get("equipment") or {}).get("equipment") or {}
        held = {k: v.get("itemId") for k, v in eq.items() if k.endswith("_hand") and v}
        if not all(held.get(f"{hand}_hand") == item_id for item_id, hand in items):
            raise JourneyFailed(f"equipment did not take: {held}")
        return f"holding {held}"

    # -- steps --------------------------------------------------------------------------------

    def step(self, kind: str, label: str, fn: Callable[[], Any]) -> bool:
        t0 = time.monotonic()
        rep = StepReport(kind=kind, label=label)
        self.log(f"- {kind}: {label}")
        try:
            result = fn()
            rep.ok = result is not False
            if isinstance(result, str):
                rep.detail = result
        except (JourneyFailed, GatewayError) as exc:
            rep.detail = str(exc)
            self.log(f"  x {exc}")
        rep.seconds = time.monotonic() - t0
        self.report.steps.append(rep)
        return rep.ok


# --- Act III ---------------------------------------------------------------------------------

LUMINARA_CRYSTAL = (128.5, 116.5)
WINDREACH_CRYSTAL = (8.5, -190.5)

# Coarse routes a player would read off the M map: city gates, then the authored roads.
ROUTES: dict[str, list[tuple[float, float]]] = {
    "city_to_ironvein": [(128, 116), (104, 116), (86, 116), (72, 116), (60, 116)],
    "ironvein_to_city": [(72, 116), (86, 116), (104, 116), (128, 116)],
}


def act3_steps(j: Journey, roads: dict[str, list[list[float]]], gear: bool = True) -> list[tuple[str, str, Callable[[], Any]]]:
    woods = [tuple(p) for p in roads["woods"]]
    north = [tuple(p) for p in roads["north"]]
    stone = [tuple(p) for p in roads["stone"]]
    highland = [tuple(p) for p in roads["highland"]]
    lower = [tuple(p) for p in roads["lower"]]
    upper = [tuple(p) for p in roads["upper"]]
    pinnacle = [tuple(p) for p in roads["pinnacle"]]
    rev = lambda pts: list(reversed(pts))  # noqa: E731
    city_to_thornwatch = ROUTES["city_to_ironvein"] + woods[1:]
    thornwatch_to_city = rev(woods)[:-1] + ROUTES["ironvein_to_city"]
    to_camp = north
    to_galepost = ROUTES["city_to_ironvein"] + woods[1:] + [(-56, 22)] + north + lower[1:]
    kit: list[tuple[str, str, Callable[[], Any]]] = [
        ("setup", "guardian kit from Oswin's counter", lambda: j.gear_up("warden_oswin", [("guild_sword", "left"), ("guild_shield", "right")])),
    ] if gear else []
    return kit + [
        ("quest", "Tethis offers the charter", lambda: j.quest("tethis", "act3_charter", "accept")),
        ("quest", "Vane takes the charter", lambda: j.quest("aldris_vane", "act3_charter", "complete", via=[(128, 100), (128, 80)])),
        ("quest", "Vane sends us down the woods road", lambda: j.quest("aldris_vane", "act3_woods_road", "accept")),
        ("quest", "walk the woods road to Hadwin", lambda: j.quest("hadwin_briarcoat", "act3_woods_road", "complete", via=[(128, 90), (128, 116)] + city_to_thornwatch)),
        ("quest", "Hadwin asks for a reading", lambda: j.quest("hadwin_briarcoat", "act3_stone_reading", "accept")),
        ("landmark", "read the Sundered Stone", lambda: j.landmark("sundered_stone", -78.5, 39.5, via=stone)),
        ("quest", "report the reading", lambda: j.quest("hadwin_briarcoat", "act3_stone_reading", "complete", via=rev(stone))),
        ("quest", "clear the north road", lambda: j.quest("hadwin_briarcoat", "act3_north_road", "accept")),
        ("kill", "three Ashen Skulkers", lambda: j.kill("ashen_skulker", 3, near=(-46, -4)) >= 3),
        ("quest", "Talia at the camp", lambda: j.quest("scout_talia", "act3_north_road", "complete", via=to_camp)),
        ("quest", "Talia's survey", lambda: j.quest("scout_talia", "act3_foothills", "accept")),
        ("quest", "home to Kael", lambda: j.quest("seraine_kael", "act3_foothills", "complete",
                                                   via=rev(north) + thornwatch_to_city + [(152, 116), (182, 116), (200, 116)])),
        ("quest", "Kael: flags in the north", lambda: j.quest("seraine_kael", "act3_signal", "accept")),
        ("quest", "the North Watch", lambda: j.quest("signal_orrin", "act3_signal", "complete", via=[(183, 113), (183, 58), (206, 56)] + highland)),
        ("quest", "Orrin: to Stormbreak", lambda: j.quest("signal_orrin", "act3_stormbreak", "accept")),
        ("quest", "climb to Galepost", lambda: j.quest("weather_elara", "act3_stormbreak", "complete",
                                                        via=rev(highland) + [(206, 56), (183, 58), (183, 113), (152, 116)] + to_galepost)),
        ("quest", "Elara: the signal", lambda: j.quest("weather_elara", "act3_galepost_signal", "accept")),
        ("landmark", "raise the mast signal", lambda: j.landmark("galepost_mast", 1.5, -142.5)),
        ("quest", "report the signal", lambda: j.quest("weather_elara", "act3_galepost_signal", "complete")),
        ("quest", "Elara: to Windreach", lambda: j.quest("weather_elara", "act3_windreach", "accept")),
        ("quest", "Selara in the Athenaeum", lambda: j.quest("archscholar_selara", "act3_windreach", "complete", via=upper + [(8, -196), (8, -203)])),
        ("quest", "Selara: both ends", lambda: j.quest("archscholar_selara", "act3_two_ends", "accept")),
        ("travel", "Windreach → Luminara", lambda: j.travel(WINDREACH_CRYSTAL, "luminara")),
        ("quest", "Tethis hears it", lambda: j.quest("tethis", "act3_two_ends", "complete")),
        ("quest", "Tethis: the method", lambda: j.quest("tethis", "act3_open_method", "accept")),
        ("travel", "Luminara → Windreach", lambda: j.travel(LUMINARA_CRYSTAL, "windreach")),
        ("quest", "Ysolde's terms", lambda: j.quest("windcaller_ysolde", "act3_open_method", "complete")),
        ("quest", "Ysolde: ask Neris", lambda: j.quest("windcaller_ysolde", "act3_missing_pages", "accept")),
        ("quest", "Neris and the empty shelf", lambda: j.quest("librarian_neris", "act3_missing_pages", "complete", via=[(8, -196), (8, -203), (8, -207)])),
        ("quest", "Neris: the Pinnacle", lambda: j.quest("librarian_neris", "act3_pinnacle_crystal", "accept")),
        ("landmark", "read the original crystal", lambda: j.landmark("pinnacle_crystal", 16.5, -225.5, via=[(8, -203), (8, -196), (16, -208)] + pinnacle)),
        ("quest", "back to Neris", lambda: j.quest("librarian_neris", "act3_pinnacle_crystal", "complete", via=rev(pinnacle) + [(16, -208), (8, -203), (8, -207)])),
        ("quest", "Ysolde: the same wind", lambda: j.quest("windcaller_ysolde", "act3_tempest", "accept", via=[(8, -203), (8, -196)])),
        ("instance", "Spire of Tempests", lambda: j.instance("story_tempest", "spire_rimeguard")),
        ("quest", "report the quiet rods", lambda: j.quest("windcaller_ysolde", "act3_tempest", "complete")),
        ("quest", "Ysolde: the ship", lambda: j.quest("windcaller_ysolde", "act3_ship_log", "accept")),
        ("travel", "Windreach → Luminara", lambda: j.travel(WINDREACH_CRYSTAL, "luminara")),
        ("quest", "Dain's sealed statement", lambda: j.quest("portmaster_dain", "act3_ship_log", "complete", via=[(128, 118), (128, 136), (128, 146)])),
        ("quest", "Dain: to Quill", lambda: j.quest("portmaster_dain", "act3_cross_check", "accept")),
        ("quest", "Quill's sheet", lambda: j.quest("archivist_quill", "act3_cross_check", "complete",
                                                    via=[(128, 146), (128, 136), (128, 118), (152, 116), (152, 98), (167, 98), (167, 94)])),
        ("quest", "Quill: to Selara", lambda: j.quest("archivist_quill", "act3_cut_pages", "accept")),
        ("travel", "Luminara → Windreach", lambda: j.travel(LUMINARA_CRYSTAL, "windreach")),
        ("quest", "the cut pages", lambda: j.quest("archscholar_selara", "act3_cut_pages", "complete", via=[(8, -196), (8, -203)])),
        ("quest", "Selara: letters home", lambda: j.quest("archscholar_selara", "act3_convene", "accept")),
        ("travel", "Windreach → Luminara", lambda: j.travel(WINDREACH_CRYSTAL, "luminara")),
        ("quest", "Kael: three cities", lambda: j.quest("seraine_kael", "act3_convene", "complete", via=[(152, 116), (182, 116), (200, 116)])),
    ]


# --- Prologue, Act I, Act II ---------------------------------------------------------------

# Where the M map / the quest reminders say people stand (x, y, z): used only when an NPC has not been
# streamed to us yet. Everything else comes from the entities the server shows us.
NPC_POSTS: dict[str, tuple[float, float, float]] = {
    "warden_petra": (131, 75, 131), "tethis": (124, 75, 111), "portmaster_dain": (126, 66, 151),
    "sister_amalthe": (167, 78, 74), "archivist_quill": (167, 74, 90), "seraine_kael": (201, 75, 112),
    "guild_mira": (194, 76, 86), "recruit_ren": (194, 75, 101), "aldris_vane": (128, 80, 72),
    "brondt_ashvein": (60, 28, 11), "opaline_vess": (77, 28, 31), "grenn_stonewall": (59, 28, 50),
}
# Mobs that never fight back: not worth turning to when they are near.
PASSIVE_MOBS = {"training_dummy", "corrupted_seedling"}
# Job skills the fighter presses when their recast is up, best first (guardian: stun, then the 3-2-1 combo).
MELEE_SKILLS = ("shield_bash", "sentinel_strike", "bulwark_slash", "iron_cleave")

DEEP_LIGHT_CRYSTAL = (60.5, 33.5)  # stand on the plaza south of the Crystal of the Deep Light (60, 30)
CAVERN_Y = 25  # Duskhollow's walking height (floor y 24)
CRYSTAL_ORE = 105  # shared/src/blocks.ts BlockId.CRYSTAL_ORE: the glowing seams in the east gallery

CITY_WEST = [(152.0, 116.0), (128.0, 116.0)]  # Assembly Row westward to Dawnsquare
KEEP = [(152.0, 116.0), (182.0, 116.0), (200.0, 116.0)]  # Dawnsquare east along Assembly Row to the Keep gate
TO_ASSEMBLY = [(128.0, 100.0), (128.0, 80.0)]  # Dawnsquare north up the avenue to the Assembly hall
FROM_ASSEMBLY = [(128.0, 90.0), (128.0, 116.0)]
TO_HARBOR = [(128.0, 118.0), (128.0, 136.0), (128.0, 146.0)]
FROM_HARBOR = [(128.0, 146.0), (128.0, 136.0), (128.0, 118.0)]
TO_TEMPLE_LANE = [(152.0, 116.0), (152.0, 98.0)]  # the lane west of the archive and temple
WEST_ROAD = (40.0, 98.0)  # beyond the causeway: the West Road packs


def guild_steps(j: Journey) -> list[tuple[str, str, Callable[[], Any]]]:
    """A new arrival's weapons, the in-game way: Mira's orientation, Ren at the targets, the guild set."""
    return [
        ("quest", "Mira's orientation", lambda: j.quest("guild_mira", "guild_orientation", "accept")),
        ("quest", "Ren at the targets", lambda: j.quest("recruit_ren", "guild_orientation", "complete")),
        ("quest", "Ren: claim your weapons", lambda: j.quest("recruit_ren", "guild_armament", "accept")),
        ("quest", "Mira's guild weapons", lambda: j.quest("guild_mira", "guild_armament", "complete")),
        ("setup", "equip the guild sword and shield", lambda: j.equip_items([("guild_sword", "left"), ("guild_shield", "right")])),
    ]


def prologue_steps(j: Journey) -> list[tuple[str, str, Callable[[], Any]]]:
    """Five visits across Luminara: Petra → Tethis → Dain → Amalthe → Quill → Kael. No combat."""
    return [
        ("quest", "Petra's welcome", lambda: j.quest("warden_petra", "petra_welcome", "accept")),
        ("quest", "Tethis at the crystal", lambda: j.quest("tethis", "petra_welcome", "complete")),
        ("quest", "Tethis: the manifest", lambda: j.quest("tethis", "story_manifest", "accept")),
        ("quest", "down to Dain at the harbor", lambda: j.quest("portmaster_dain", "story_manifest", "complete", via=TO_HARBOR)),
        ("quest", "Dain: the beacon", lambda: j.quest("portmaster_dain", "story_beacon", "accept")),
        ("quest", "Amalthe in the temple", lambda: j.quest("sister_amalthe", "story_beacon", "complete", via=FROM_HARBOR + TO_TEMPLE_LANE)),
        ("quest", "Amalthe: the archive", lambda: j.quest("sister_amalthe", "story_archive", "accept")),
        ("quest", "Quill in the archive", lambda: j.quest("archivist_quill", "story_archive", "complete",
                                                         via=[(152.0, 98.0), (167.0, 98.0), (167.0, 94.0)])),
        ("quest", "Quill: report to Kael", lambda: j.quest("archivist_quill", "story_report", "accept")),
        ("quest", "Kael at the Keep gate", lambda: j.quest("seraine_kael", "story_report", "complete",
                                                          via=[(167.0, 98.0), (152.0, 98.0)] + KEEP)),
    ]


def act1_steps(j: Journey, roads: dict[str, list[list[float]]]) -> list[tuple[str, str, Callable[[], Any]]]:
    """Training, the Verdant Hollow, the Assembly's proof, the West Road, the Heart of the Hollow."""
    to_west = CITY_WEST + ROUTES["city_to_ironvein"] + [(50.0, 104.0)]
    from_west = [(50.0, 104.0)] + ROUTES["ironvein_to_city"]

    def back_to_west() -> None:
        j.walk_route(to_west, "back to the West Road")

    def chitin() -> str:
        have = j.count("hollow_chitin")
        if have >= 2:
            return f"already holding {have} from the Hollow"
        j.walk_route([(182.0, 116.0)] + to_west, "to the West Road")
        return j.collect_drops("hollow_chitin", 2, "hollowdeep_crawler", WEST_ROAD, search_m=16, back=back_to_west)

    return [
        ("quest", "Kael: the training yard", lambda: j.quest("seraine_kael", "kael_training", "accept")),
        ("kill", "three training dummies", lambda: j.kill_for_quest("kael_training", "training_dummy", (200.0, 102.0),
                                                                    search_m=12, timeout_s=180)),
        ("quest", "report the training", lambda: j.quest("seraine_kael", "kael_training", "complete")),
        ("quest", "Kael: the Verdant Hollow", lambda: j.quest("seraine_kael", "kael_hollow", "accept")),
        ("instance", "the Verdant Hollow", lambda: j.clear_instance("verdant_hollow", "thornweaver_herald")),
        ("quest", "report the Hollow", lambda: j.quest("seraine_kael", "kael_hollow", "complete")),
        ("quest", "Kael: chitin for the Assembly", lambda: j.quest("seraine_kael", "act1_chitin", "accept")),
        ("collect", "two Hollow Chitin", chitin),
        ("quest", "Vane takes the chitin", lambda: j.quest("aldris_vane", "act1_chitin", "complete",
                                                          via=(from_west if j.me()["x"] < 90 else CITY_WEST) + TO_ASSEMBLY)),
        ("quest", "Vane: ask Tethis", lambda: j.quest("aldris_vane", "act1_crystal", "accept")),
        ("quest", "Tethis compares the readings", lambda: j.quest("tethis", "act1_crystal", "complete", via=FROM_ASSEMBLY)),
        ("quest", "Kael: cull the West Road", lambda: j.quest("seraine_kael", "act1_cull", "accept", via=KEEP)),
        ("walk", "out to the West Road", lambda: j.walk_route([(182.0, 116.0)] + to_west, "to the West Road")),
        ("kill", "two Ashen Skulkers", lambda: j.kill_for_quest("act1_cull", "ashen_skulker", WEST_ROAD, search_m=16,
                                                                back=back_to_west)),
        ("quest", "report the cull", lambda: j.quest("seraine_kael", "act1_cull", "complete", via=from_west + KEEP)),
        ("quest", "Kael: the Heart of the Hollow", lambda: j.quest("seraine_kael", "act1_heart", "accept")),
        ("instance", "the Heart root chamber", lambda: j.clear_instance("story_hollow_heart", "hollow_rootheart")),
        ("quest", "report the Heart", lambda: j.quest("seraine_kael", "act1_heart", "complete")),
    ]


def act2_steps(j: Journey, roads: dict[str, list[list[float]]]) -> list[tuple[str, str, Callable[[], Any]]]:
    """Vane's envoy down the Delver's Descent to Duskhollow: the deep signal, the Drowned Gallery, word home."""
    to_descent = FROM_ASSEMBLY + ROUTES["city_to_ironvein"] + [(60.0, 108.0), (58.0, 104.0)]

    def descend() -> str:
        """The stair shaft (58, 67, 103) → (58, 25, 54), then the plaza road south to the Deep Light."""
        # into the stair mouth first (the pit in the camp pad, feet y 64 at z 100): the stairs run on under the pad
        for x, z, y, label in ((58.5, 100.5, 64, "into the stair mouth"), (58.0, 56.0, CAVERN_Y, "down the Delver's Descent"),
                               (60.0, 36.0, CAVERN_Y, "to the Deep Light plaza")):
            leg = j.walk_to(x, z, y=y, radius=1.0 if y != CAVERN_Y else 2.0, label=label)
            if not leg.ok:
                me = j.me()
                raise JourneyFailed(f"{label} failed ({leg.reason}) at {me['x']:.0f},{me['y']:.0f},{me['z']:.0f}")
        if j.me()["y"] > 40:
            raise JourneyFailed(f"ended on the surface (y {j.me()['y']:.0f}), not in Duskhollow")
        return f"in Duskhollow at y {j.me()['y']:.0f}"

    def back_down() -> None:
        # respawned at home: the long way back, as a player would walk it
        j.walk_route(CITY_WEST + ROUTES["city_to_ironvein"] + [(60.0, 108.0), (58.0, 104.0)], "back to the Descent")
        descend()

    return [
        ("quest", "Vane: the envoy", lambda: j.quest("aldris_vane", "act2_descent", "accept", via=CITY_WEST + TO_ASSEMBLY)),
        ("walk", "the west causeway to Ironvein", lambda: j.walk_route(to_descent, "to the Descent")),
        ("walk", "down the Delver's Descent", descend),
        ("quest", "Ashvein hears the warning", lambda: j.quest("brondt_ashvein", "act2_descent", "complete")),
        ("quest", "Opaline: the deep signal", lambda: j.quest("opaline_vess", "act2_signal", "accept")),
        ("walk", "into the east gallery", lambda: j.walk_to(92.0, 30.0, y=CAVERN_Y, radius=2.0, label="east gallery").ok),
        ("collect", "mine three Raw Crystal", lambda: j.mine("crystal_raw", 3, CRYSTAL_ORE, (96.0, 30.0), y=CAVERN_Y, radius=14)),
        ("quest", "Opaline reads the lattice", lambda: j.quest("opaline_vess", "act2_signal", "complete", via=[(84.0, 30.0)])),
        ("quest", "Grenn: the Drowned Gallery", lambda: j.quest("grenn_stonewall", "act2_gallery", "accept")),
        ("kill", "the Rift Aberration", lambda: j.kill_for_quest(
            "act2_gallery", "rift_aberration", (24.0, 42.0), y=CAVERN_Y, search_m=16, back=back_down,
            clear_first=("hollowdeep_crawler",), timeout_s=420)),
        ("quest", "Grenn: the hum stopped", lambda: j.quest("grenn_stonewall", "act2_gallery", "complete",
                                                           via=[(38.0, 40.0), (46.0, 42.0)])),
        ("quest", "Ashvein: word through the crystal", lambda: j.quest("brondt_ashvein", "act2_words", "accept")),
        ("travel", "Deep Light → Luminara", lambda: j.travel(DEEP_LIGHT_CRYSTAL, "luminara")),
        ("quest", "Tethis hears Duskhollow", lambda: j.quest("tethis", "act2_words", "complete")),
    ]


# The seven echoes (chapter `accord`): Oswin's investigation with the low-difficulty STORY_BOSSES
# (shared/src/storyBosses.ts), one story instance each. Not the seven apex trials.
ACCORD_ECHOES = [
    ("accord_chieftain", "story_storehouse", "storehouse_gnawer"), ("accord_warden", "story_checkpoint", "rusted_watchman"),
    ("accord_prowler", "story_courier", "wayward_runner"), ("accord_nightfang", "story_watch", "echo_stag"),
    ("accord_wyrm", "story_condenser", "cinder_sprout"), ("accord_maw", "story_tidal", "archive_mireling"),
    ("accord_coil", "story_seal", "seal_sapling"),
]


def accord_steps(j: Journey) -> list[tuple[str, str, Callable[[], Any]]]:
    """Kael → Oswin, seven story instances, then Quill → Amalthe → Kael. Starts after kael_training."""
    steps: list[tuple[str, str, Callable[[], Any]]] = [
        ("quest", "Kael: seven echoes", lambda: j.quest("seraine_kael", "accord_invitation", "accept")),
        ("quest", "Oswin at the crystal", lambda: j.quest("warden_oswin", "accord_invitation", "complete",
                                                         via=[(182.0, 116.0), (152.0, 116.0), (134.0, 116.0)])),
    ]
    for quest_id, zone, boss in ACCORD_ECHOES:
        steps += [
            ("quest", f"Oswin: {quest_id}", lambda q=quest_id: j.quest("warden_oswin", q, "accept")),
            ("instance", zone, lambda z=zone, b=boss: j.clear_instance(z, b)),
            ("quest", f"report {quest_id}", lambda q=quest_id: j.quest("warden_oswin", q, "complete")),
        ]
    return steps + [
        ("quest", "Oswin: the assembly", lambda: j.quest("warden_oswin", "accord_assembly", "accept")),
        ("quest", "Quill compares the seven", lambda: j.quest("archivist_quill", "accord_assembly", "complete",
                                                             via=[(152.0, 116.0), (152.0, 98.0), (167.0, 98.0), (167.0, 94.0)])),
        ("quest", "Quill: the beacon", lambda: j.quest("archivist_quill", "accord_beacon", "accept")),
        ("quest", "Amalthe's reading", lambda: j.quest("sister_amalthe", "accord_beacon", "complete", via=[(167.0, 98.0), (152.0, 98.0)])),
        ("quest", "Amalthe: home to Kael", lambda: j.quest("sister_amalthe", "accord_home", "accept")),
        ("quest", "Kael: the shared watch", lambda: j.quest("seraine_kael", "accord_home", "complete", via=[(152.0, 98.0)] + KEEP)),
    ]


def campaign_steps(j: Journey, roads: dict[str, list[list[float]]]) -> list[tuple[str, str, Callable[[], Any]]]:
    """A brand-new character through everything playable: prologue, guild weapons, Acts I–III."""
    return prologue_steps(j) + guild_steps(j) + act1_steps(j, roads) + act2_steps(j, roads) + act3_steps(j, roads, gear=False)


StepList = list[tuple[str, str, Callable[[], Any]]]
def city_steps(j: Journey, roads, alternate=False):
    from .city_life import steps
    return steps(j, alternate)


def city_phase2_steps(j: Journey, alternate: bool = False):
    from .city_phase2 import steps
    return steps(j, alternate)


from .duskhollow_life import steps as duskhollow_life_steps
from .duskhollow_vent import steps as duskhollow_vent_steps
from .luminara_community import steps as community_steps
from .windreach_life import steps as windreach_life_steps
from .act4_scar import steps as act4_scar_steps
from .act4_threshold import steps as act4_threshold_steps
from .act5_resonance import steps as act5_resonance_steps
from .act5_swarm import steps as act5_swarm_steps
from .act5_maw import steps as act5_maw_steps
from .act5_accord import steps as act5_accord_steps
from .act5_primarch import steps as act5_primarch_steps
from .act5_vexar import steps as act5_vexar_steps

SCENARIOS: dict[str, Callable[[Journey, dict[str, list[list[float]]]], StepList]] = {
    "triad_short": lambda j, roads: act5_accord_steps(j, roads, "short_pulses"),
    "triad_wide": lambda j, roads: act5_accord_steps(j, roads, "wide_window"),
    "maw_short": lambda j, roads: act5_maw_steps(j, roads, "short_pulses"),
    "maw_wide": lambda j, roads: act5_maw_steps(j, roads, "wide_window"),
    "primarch_short": lambda j, roads: act5_primarch_steps(j, roads, "short_pulses"),
    "primarch_wide": lambda j, roads: act5_primarch_steps(j, roads, "wide_window"),
    "swarm_short": lambda j, roads: act5_swarm_steps(j, roads, "short_pulses"),
    "swarm_wide": lambda j, roads: act5_swarm_steps(j, roads, "wide_window"),
    "vexar_short": lambda j, roads: act5_vexar_steps(j, roads, "short_pulses"),
    "vexar_wide": lambda j, roads: act5_vexar_steps(j, roads, "wide_window"),
    "resonance_short": lambda j, roads: act5_resonance_steps(j, roads, "short_pulses"),
    "resonance_wide": lambda j, roads: act5_resonance_steps(j, roads, "wide_window"),
    "threshold_short": lambda j, roads: act4_threshold_steps(j, roads, "short_pulses"),
    "threshold_wide": lambda j, roads: act4_threshold_steps(j, roads, "wide_window"),
    "scar_short": lambda j, roads: act4_scar_steps(j, roads, "short_pulses"),
    "scar_wide": lambda j, roads: act4_scar_steps(j, roads, "wide_window"),
    "windreach_00": lambda j, roads: windreach_life_steps(j, roads, 0, 0, False),
    "windreach_01": lambda j, roads: windreach_life_steps(j, roads, 0, 1, True),
    "windreach_10": lambda j, roads: windreach_life_steps(j, roads, 1, 0, False),
    "windreach_11": lambda j, roads: windreach_life_steps(j, roads, 1, 1, True),
    "community_00": lambda j, roads: community_steps(j, 0, 0, False),
    "community_01": lambda j, roads: community_steps(j, 0, 1, True),
    "community_10": lambda j, roads: community_steps(j, 1, 0, False),
    "community_11": lambda j, roads: community_steps(j, 1, 1, True),
    "duskhollow_vent": lambda j, roads: duskhollow_vent_steps(j),
    "duskhollow_vent_alternate": lambda j, roads: duskhollow_vent_steps(j, True),
    "duskhollow_both_00": lambda j, roads: duskhollow_vent_steps(j, False, False, False),
    "duskhollow_both_01": lambda j, roads: duskhollow_vent_steps(j, True, False, True),
    "duskhollow_both_10": lambda j, roads: duskhollow_vent_steps(j, False, True, False),
    "duskhollow_both_11": lambda j, roads: duskhollow_vent_steps(j, True, True, True),
    "duskhollow": lambda j, roads: duskhollow_life_steps(j),
    "duskhollow_alternate": lambda j, roads: duskhollow_life_steps(j, True),
    "luminara_phase2": lambda j, roads: city_phase2_steps(j),
    "luminara_phase2_alternate": lambda j, roads: city_phase2_steps(j, True),
    "luminara": lambda j, roads: city_steps(j, roads),
    "luminara_alternate": lambda j, roads: city_steps(j, roads, True),
    "prologue": lambda j, roads: prologue_steps(j),
    # a chapter on its own starts from a /quest done skip with an unarmed character: the guild arms it first
    "act1": lambda j, roads: guild_steps(j) + act1_steps(j, roads),
    "act2": lambda j, roads: guild_steps(j) + act2_steps(j, roads),
    "act3": act3_steps,
    "campaign": campaign_steps,
    # the seven-echo investigation (optional branch after the training; starts from a kael_training skip)
    "accord": lambda j, roads: guild_steps(j) + accord_steps(j),
}
# The journal chapters each scenario is about (for the end-of-run quest table).
SCENARIO_CHAPTERS = {"duskhollow": {"side"}, "duskhollow_alternate": {"side"}, "luminara_phase2": {"side"}, "luminara_phase2_alternate": {"side"}, "luminara": {"side"}, "luminara_alternate": {"side"}, "prologue": {"prologue"}, "act1": {"act1"}, "act2": {"act2"}, "act3": {"act3"},
                     "campaign": {"prologue", "act1", "act2", "act3"}, "accord": {"accord"}}
SCENARIO_CHAPTERS.update({name: {"side"} for name in SCENARIOS if name.startswith(("duskhollow_", "community_", "windreach_"))})

SCENARIO_CHAPTERS.update({"swarm_short":{"act5"},"swarm_wide":{"act5"}})
SCENARIO_CHAPTERS.update({"vexar_short":{"act5"},"vexar_wide":{"act5"}})
SCENARIO_CHAPTERS.update({"resonance_short":{"act5"},"resonance_wide":{"act5"}})
SCENARIO_CHAPTERS.update({"scar_short":{"act4"},"scar_wide":{"act4"},"threshold_short":{"act4"},"threshold_wide":{"act4"}})

def run_journey(gw: GatewayClient, scenario: str, username: str, password: str, out_dir: Path,
                setup: list[str], log: Callable[[str], None] = print, stop_on_failure: bool = True,
                character: str | None = None, companions: int = 0) -> JourneyReport:
    """Log in, run a scenario's steps, write report.json and report.md."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r} (have {', '.join(SCENARIOS)})")
    login = gw.call("login", timeout_s=30, username=username, password=password, register=True,
                    **({"character": character} if character else {}))
    log(f"character: {login.get('character')} (account {username})")
    gw.call("enter", timeout_s=40, zone="overworld")
    j = Journey(gw, scenario, log)
    j.companions = companions
    j.report.character = str(login.get("character") or "")
    gw.wait_snapshot(lambda d: bool(d.get("self", {}).get("entityId")), timeout_s=20, label="first snapshot")
    time.sleep(3.0)  # let the chunk stream fill in around us
    for command in setup:
        gw.call("chat", message=command)
        time.sleep(0.3)
    info = gw.call("world_info")
    steps = SCENARIOS[scenario](j, info["roads"])
    t0 = time.monotonic()
    try:
        for kind, label, fn in steps:
            if not j.step(kind, label, fn) and stop_on_failure:
                break
        else:
            j.report.finished = True
    finally:
        j.stop()
        j.report.seconds = time.monotonic() - t0
        try:
            time.sleep(0.3)  # the last turn-in reaches the snapshot a tick after its quest_update
            chapters = SCENARIO_CHAPTERS.get(scenario, {"act3"})
            j.report.quests = [q for q in (j.snap().get("quests") or []) if q.get("chapter") in chapters]
        except JourneyFailed:
            pass
        write_report(j.report, out_dir)
    return j.report


def write_report(report: JourneyReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"summary": report.summary(), **asdict(report)}
    (out_dir / "journey.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report.summary()
    lines = [f"# Journey {report.name}", "", f"- character: {report.character}",
             f"- finished: {s['finished']} ({s['steps_ok']}/{s['steps']} steps) in {s['seconds']} s",
             f"- walked {s['walked_m']} m in {s['walking_s']} s (mean {s['mean_speed_mps']} m/s), {s['legs']} legs, "
             f"{s['stuck_points']} stuck points, {s['falls']} falls, {s['deaths']} deaths", ""]
    if s["first_failure"]:
        lines += [f"**First failure:** {s['first_failure']['kind']} “{s['first_failure']['label']}” — {s['first_failure']['detail']}", ""]
    if report.findings:
        lines += ["## Findings (worked around)", ""] + [f"- {f}" for f in report.findings] + [""]
    lines += ["## Steps", "", "| # | kind | step | ok | s | detail |", "| --- | --- | --- | --- | ---: | --- |"]
    for i, st in enumerate(report.steps, 1):
        lines.append(f"| {i} | {st.kind} | {st.label} | {'✓' if st.ok else '✗'} | {st.seconds:.0f} | {st.detail} |")
    lines += ["", "## Slowest legs", "", "| leg | s | m | detour |", "| --- | ---: | ---: | ---: |"]
    for r in s["slowest_legs"]:
        lines.append(f"| {r['leg']} | {r['s']} | {r['m']} | {r['detour']} |")
    if s["worst_stuck"]:
        lines += ["", "## Stuck points", "", "| leg | x | y | z | s |", "| --- | ---: | ---: | ---: | ---: |"]
        for p in s["worst_stuck"]:
            lines.append(f"| {p['leg']} | {p['x']} | {p['y']} | {p['z']} | {p['seconds']} |")
    (out_dir / "journey.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

SCENARIO_CHAPTERS.update({"primarch_short":{"act5"},"primarch_wide":{"act5"}})

SCENARIO_CHAPTERS.update({"maw_short":{"act5"},"maw_wide":{"act5"}})

SCENARIO_CHAPTERS.update({"triad_short":{"act5","accord"},"triad_wide":{"act5","accord"}})

# Fresh ordinary progression through the root chamber, without quest-skip fallback.
SCENARIOS["heart_fresh"] = lambda j, roads: prologue_steps(j) + guild_steps(j) + act1_steps(j, roads)
SCENARIO_CHAPTERS["heart_fresh"] = {"prologue", "act1"}

from .early_campaign import steps as early_campaign_steps
SCENARIOS["early_catchup"] = lambda j, roads: early_campaign_steps(j,roads)
SCENARIOS["early_route"] = lambda j, roads: early_campaign_steps(j,roads,finale=False)
SCENARIO_CHAPTERS["early_catchup"] = {"prologue","act1","act2","act3","act4","act5","accord"}
SCENARIO_CHAPTERS["early_route"] = {"prologue","act1","act2","act3","act4"}


from .southern_rest import steps as southern_rest_steps
SCENARIOS.update({
 "rest_duty":lambda j,roads:southern_rest_steps(j,roads,"duty_seat","warm_cup"),
 "rest_quiet":lambda j,roads:southern_rest_steps(j,roads,"quiet_corner","quiet_game"),
 "rest_duty_game":lambda j,roads:southern_rest_steps(j,roads,"duty_seat","quiet_game"),
 "rest_quiet_cup":lambda j,roads:southern_rest_steps(j,roads,"quiet_corner","warm_cup"),
})
SCENARIO_CHAPTERS.update({name:{"side"} for name in SCENARIOS if name.startswith("rest_")})

from .southern_wagon import steps as southern_wagon_steps
SCENARIOS.update({"wagon_broad":lambda j,roads:southern_wagon_steps(j,roads,"broad_shade"),"wagon_small":lambda j,roads:southern_wagon_steps(j,roads,"small_shade")})
SCENARIO_CHAPTERS.update({"wagon_broad":{"side"},"wagon_small":{"side"}})
