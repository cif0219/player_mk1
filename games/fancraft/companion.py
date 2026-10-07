"""An NPC companion: player_mk1 fighting beside a player (fancraft docs/COMBAT.md §7.3).

A player calls a companion through the game (the party panel, the Trial Warden,
`/helper summon`); the game server queues the request with an account name, the
agent gateway launches this loop under that name, and the room recognises the
account as an NPC companion — a body that fights but never counts toward a wipe.

The loop is deliberately simple and readable, built from the journey tester's
primitives (games/fancraft/journey.py): stay near the one who called you, fight
what threatens them (anything within reach of them or already on us), guard what
is aimed at us, and follow them from room to room until dismissed.
"""

from __future__ import annotations

import random
import string
import time
from typing import Any, Callable

from player.api.gateway import GatewayClient

from .journey import PASSIVE_MOBS, Journey, JourneyFailed

#: How close to the one we serve we try to stay, and how far out we still defend them.
STAY_M = 4.0
DEFEND_M = 12.0
#: How far from the one we serve we are willing to chase a target before coming back.
LEASH_M = 22.0


class Companion:
    def __init__(self, gw: GatewayClient, requester_id: str, requester_name: str, log: Callable[[str], None] = print) -> None:
        self.gw = gw
        self.j = Journey(gw, "companion", log)
        self.requester_id = requester_id
        self.requester_name = requester_name
        self.log = log
        self.dismissed = False
        self._cursor = gw.event_cursor

    # -- orders from the helper desk ------------------------------------------------------------

    def _orders(self) -> None:
        for ev in self.gw.events_since(self._cursor):
            self._cursor = max(self._cursor, ev.seq)
            if ev.type == "helper_dismiss":
                self.log("dismissed")
                self.dismissed = True
            elif ev.type == "helper_follow":
                room, zone = ev.data.get("roomId"), ev.data.get("zone")
                if room and room != self.j.snap().get("roomId"):
                    self.log(f"following into {zone} ({room})")
                    self.j.stop()
                    try:
                        self.gw.call("enter", timeout_s=40, zone=zone, roomId=room)
                    except Exception as exc:  # noqa: BLE001 - a refused room (full, closed) is reported and waited out
                        self.log(f"could not follow: {exc}")
                    time.sleep(1.0)

    # -- what to do ------------------------------------------------------------------------------

    def leader(self) -> dict[str, Any] | None:
        return self.j.entity(lambda e: e.get("type") == "player" and e.get("name") == self.requester_name)

    def threat(self, leader: dict[str, Any] | None) -> dict[str, Any] | None:
        """The mob to fight: whatever swings at us, else the nearest one within reach of the leader."""
        s = self.j.snap()
        me = s["self"]
        mine = me.get("entityId")
        aimed = {t.get("entityId") for t in s.get("telegraphs") or [] if t.get("targetId") == mine and not t.get("landed")}
        best, best_d = None, 1e9
        for e in s.get("entities") or []:
            if e.get("type") != "mob" or not e.get("alive") or e.get("mobId") in PASSIVE_MOBS:
                continue
            if e["entityId"] in aimed:
                return e
            ref = leader or me
            d = ((e["x"] - ref["x"]) ** 2 + (e["z"] - ref["z"]) ** 2) ** 0.5
            if d <= DEFEND_M and d < best_d and e.get("distance", 1e9) <= LEASH_M:
                best, best_d = e, d
        return best

    def run(self, seconds: float) -> None:
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and not self.dismissed:
            self._orders()
            try:
                me = self.j.me()
            except JourneyFailed:
                time.sleep(0.5)
                continue
            if me.get("downed"):
                self.j.stop()
                time.sleep(0.5)
                continue
            leader = self.leader()
            foe = self.threat(leader)
            if foe is not None:
                # fight() returns when it dies, when we go down, or after the timeout: then look again
                self.j.fight(foe, timeout_s=20)
                continue
            if leader is not None and leader.get("distance", 0) > STAY_M:
                self.j.walk_to(leader["x"], leader["z"], radius=STAY_M - 1, label=f"to {self.requester_name}", timeout_s=12)
                continue
            self.j.stop()
            self.gw.fire("face", **({"entityId": leader["entityId"]} if leader else {}))
            time.sleep(0.3)


def run_companion(gw: GatewayClient, username: str, password: str, room_id: str, zone: str,
                  requester_id: str, requester_name: str, seconds: float, log: Callable[[str], None] = print) -> None:
    character = "Ally" + "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    login = gw.call("login", timeout_s=30, username=username, password=password, register=True, character=character)
    log(f"companion {login.get('character')} (account {username}) serving {requester_name}")
    # Registering as a helper lets the desk's follow and dismiss orders reach this connection
    gw.call("helper", name=username, requester=requester_id)
    gw.call("enter", timeout_s=40, zone=zone, roomId=room_id)
    gw.wait_snapshot(lambda d: bool(d.get("self", {}).get("entityId")), timeout_s=20, label="first snapshot")
    time.sleep(1.5)  # the chunk stream fills in around us
    Companion(gw, requester_id, requester_name, log).run(seconds)
    gw.call("leave", timeout_s=10)
