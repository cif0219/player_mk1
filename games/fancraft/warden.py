"""The Ironclad Warden, as reflexes over API fields.

FantCraft's samurai world boss (fancraft `docs/WORLD_BOSSES.md`) is the
game's difficulty ceiling: three base cuts that must each be handled cleanly
once, a shield-only palm seize, then a mechanical horse that has to be held
by threat, three sets of the upgraded kit, a kneel that wants holy damage,
and the apex cover pulse. None of that is a rotation. It is a set of
*answers to telegraphs*, and the deadline timeline is the right tool: the
gateway reports when a cut lands, so the parry is scheduled at
`land − 100 ms` rather than reacted to.

Every answer is a reflex on `attack.*` (games/fancraft/api.py):

* **dodge** — a blow a raised guard cannot answer (the palm seizes, the
  thunderhead cloud, the mounted charge): strafe out of its arc, the side
  chosen by `attack.dodge_key`.
* **parry** — anything else physical that can reach us: a fresh guard raised
  just before it lands, facing the attacker (the guard arc is 130° around
  our facing, and the horse bites from wherever it stands, not from where
  the boss is).
* **evade** — the two phase-2 cuts that reach past their arcs are never
  parried in melee, and Great Serpent's evade keeps going for the 3.6 s its
  lightning walks after the cut has landed; wedged on a dojo prop, the evade
  turns sideways instead of pushing. Wind Cutter's wave rides the locked draw line for twenty
  metres, so the windup is spent leaving that line diagonally; Great
  Serpent's lightning starts four metres in front of the blade the moment
  the cut lands, with no telegraph of its own, and walks toward us — the
  windup is spent sprinting away and the marker reflex keeps us ahead.
* **lightning / charge markers** — a small ground marker within five metres:
  flee straight away from the boss in world terms (`flee:<id>`, re-aimed by
  the gateway every tick, so a horse-bite parry turning us does not bend the
  retreat — the one direction that always gains on lightning walking out from
  him); pinned against the wall, slide along it.
* **the pulse** — retreat, then build a wall between us and the boss.
* **the horse** — provoke it whenever the recast allows: threat above the
  chase floor is what keeps it off its master.
* **engage** — face the boss, close to melee, strike into his openings.

`tactic="dodge"` swaps the parry for a sprint out of reach, which also
counts as a clean answer (nobody died) and is what a non-defence job does.

Reflex groups (what a helper task switches on, games/fancraft/tasks.py):
`parry`, `dodge`, `evade`, `cover`, `horse`, `engage`, `purify`, `follow`.
`WardenTuning.dodge_chance` decides, once per far-reaching cut, whether the
tester leaves Wind Cutter's line or outruns Great Serpent at all — at 0.5 it
takes half of them, which is how the boss's lethality gets measured.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from player.act.timeline import KeyDown, KeyUp, Plan, Press, Priority, Step
from player.policy.condition import Condition
from player.policy.reflex import Reflex, ReflexLayer
from player.state import WorldState

ALIVE = Condition.falsy("player.downed")
ATTACK_THREATENS = Condition.all_(Condition.truthy("attack.active"), Condition.truthy("attack.threatens"))
NO_ATTACK = Condition.falsy("attack.active", on_missing=True)
NOT_CASTING = Condition.not_(Condition.field("boss.phase", "==", "casting"))

# Raise the guard this long before the blow lands. The parry window is lerp(700, 170,
# weight) ms before impact, so 180 ms is inside it for every cut in the kit — and on a
# real server the guard message needs a move-stream slot plus a tick (50-100 ms) to
# count, which a 100 ms lead did not leave (a live Iron Breaker landed on a guard the
# server had not seen yet).
PARRY_LEAD_MS = 180
PARRY_HOLD_MS = 260
MELEE_M = 3.0
APPROACH_FROM_M = 3.4
TOO_CLOSE_M = 1.6
PULSE_SAFE_M = 6.0
COVER_BLOCKS = 9
# Past this distance an evade stops backing away and orbits instead: the dojo wall is
# 14 m behind the boss on one side, and a dodge that ends pinned against it is a death.
EVADE_ORBIT_FROM_M = 9.0
WALL_MARGIN_M = 3.0
# A sprinting flee covers 4-5 m per hold: stop backing straight away from the boss while
# there is less room than that behind us, or one hold ends pinned on the wall with the
# lightning still walking. Along the wall it is; the strafe is faster than the lightning.
FLEE_ROOM_M = 7.0
# A lightning step closes 2.5 m every 450 ms: start leaving while there are still two
# steps of room, or the first one we react to is the one that lands on us.
MARKER_FLEE_M = 5.0
# Great Serpent's eight steps take 3.6 s after the cut lands: keep leaving that long.
LIGHTNING_MS = 3800
# Commanded to move but going nowhere for this long: wedged on a dojo prop — go sideways.
STUCK_MS = 300


@dataclass(slots=True)
class WardenTuning:
    """Knobs the reflexes read at decision time, so a helper's task can change them mid-fight.

    `tactic`: "parry" (a fresh guard before the blow) or "dodge" (leave its reach).
    `dodge_chance`: the share of far-reaching cuts (Wind Cutter's line, Great
    Serpent's lightning) the tester actually leaves — the rest it takes as a
    player who misread the tell would, which is what measures the boss.
    """

    tactic: str = "parry"
    dodge_chance: float = 1.0
    random: Callable[[], float] = random.random


@dataclass(slots=True)
class WardenMemory:
    """What the reflexes remember between ticks: which telegraph was answered."""

    answered: dict[str, float] = field(default_factory=dict)
    covered_attempt: int = -1
    # Per far-reaching cut: did the coin say leave? Rolled once per telegraph.
    dodge_rolls: dict[str, bool] = field(default_factory=dict)
    declined: int = 0
    purified_attempt: int = -1

    def will_dodge(self, key: str, chance: float, roll: Callable[[], float]) -> bool:
        if not key:
            return True
        if key not in self.dodge_rolls:
            self.dodge_rolls[key] = chance >= 1.0 or roll() < chance
            if not self.dodge_rolls[key]:
                self.declined += 1
            if len(self.dodge_rolls) > 256:
                for old in list(self.dodge_rolls)[:128]:
                    del self.dodge_rolls[old]
        return self.dodge_rolls[key]

    def stats(self) -> dict[str, int]:
        return {"answered": len(self.answered), "dodge_rolls": len(self.dodge_rolls), "declined": self.declined}
    # The strafe direction of the dodge in progress: a follow-up blow aimed at where we
    # are now (the cloud after the seize, the wave after the draw) sees us on its line
    # and would pick a side at random — reversing a strafe that was already working.
    last_dodge: tuple[str, float] | None = None

    def strafe_key(self, preferred: str, at: float, keep_for_s: float = 2.0) -> str:
        if self.last_dodge is not None and at - self.last_dodge[1] < keep_for_s:
            preferred = self.last_dodge[0]
        self.last_dodge = (preferred, at)
        return preferred

    def claim(self, key: str, at: float) -> bool:
        if key in self.answered:
            return False
        self.answered[key] = at
        if len(self.answered) > 256:
            for old in list(self.answered)[:128]:
                del self.answered[old]
        return True


def _flee(state: WorldState) -> str:
    """The world-frame retreat key: straight away from the boss, whatever we face."""
    boss = state.get("boss.entity_id")
    return f"flee:{int(boss)}" if isinstance(boss, (int, float)) else "s"


def _hold(key: str, at_ms: int, hold_ms: int) -> tuple[Step, Step]:
    return Step(at_ms, KeyDown(key)), Step(at_ms + hold_ms, KeyUp(key))


def _sprint(keys: list[str], at_ms: int, hold_ms: int) -> list[Step]:
    steps = [Step(at_ms, Press(k, hold_ms)) for k in keys]
    steps.append(Step(at_ms, Press("shift", hold_ms)))
    return steps


def build_warden_reflexes(tactic: str = "parry", memory: WardenMemory | None = None, tuning: WardenTuning | None = None) -> ReflexLayer:
    memory = memory or WardenMemory()
    tuning = tuning or WardenTuning(tactic=tactic)
    layer = ReflexLayer()

    # -- the blows -----------------------------------------------------------------

    def dodge_plan(state: WorldState) -> Plan | None:
        key = state.text("attack.key")
        if not memory.claim(key, state.captured_at):
            return None
        land = int(state.num("attack.land_in_ms"))
        attack = state.text("attack.id")
        wide = attack in ("warden_cloud", "warden_charge", "warden_wave")
        hold = max(400, min(1200 if wide else 900, land + 300))
        if attack == "warden_palm2":
            # Cloud Seize is followed 600 ms later by the thunderhead cloud (70°, 8 m) aimed
            # at whoever holds threat — us. Keep strafing through both.
            hold = land + 1500
        keys = [memory.strafe_key(state.text("attack.dodge_key", "d"), state.captured_at)]
        if attack == "warden_cloud":
            # Eight metres of reach: from far out a strafe alone turns too few degrees in
            # 600 ms to clear a 70° cone, but leaving its reach always works. Do both.
            keys.append(_flee(state))
        steps = _sprint(keys, 0, hold)
        if attack in ("warden_palm", "warden_palm2"):
            steps.append(Step(0, Press("s", min(hold, 450))))  # the seize is short: back out of it too
        return Plan(name=f"dodge:{attack}", steps=tuple(steps), expires_in_ms=400)

    layer.add(Reflex(
        id="dodge_unguardable", group="dodge",
        condition=Condition.all_(ALIVE, ATTACK_THREATENS, Condition.falsy("attack.guardable"), Condition.field("attack.land_in_ms", ">", 0)),
        plan=dodge_plan, priority=Priority.REFLEX, cooldown_ms=0, preempt=True,
    ))

    def parry_plan(state: WorldState) -> Plan | None:
        key = state.text("attack.key")
        if not memory.claim(key, state.captured_at):
            return None
        land = int(state.num("attack.land_in_ms"))
        attack = state.text("attack.id")
        dodge_key = state.text("attack.dodge_key", "d")
        if tuning.tactic == "dodge":
            hold = max(400, min(1600, land + 200))
            return Plan(name=f"evade:{attack}", steps=tuple(_sprint(["s", dodge_key], 0, hold)), expires_in_ms=400)
        raise_at = max(0, land - PARRY_LEAD_MS)
        steps: list[Step] = list(_hold("mouse2", raise_at, PARRY_HOLD_MS))
        if state.text("attack.role") != "boss":
            # Square up to whoever is swinging, then back to the boss once the hold ends.
            steps.insert(0, Step(0, Press(f"face:{int(state.num('attack.entity_id'))}", 20)))
            steps.append(Step(raise_at + PARRY_HOLD_MS + 40, Press("face:boss", 20)))
        return Plan(name=f"parry:{attack}", steps=tuple(steps), expires_in_ms=land + 700)

    NO_MARKER_NEAR = Condition.any_(
        Condition.field("marker.nearest_m", ">", 6.0, on_missing=True), Condition.field("marker.radius_m", ">=", 30.0, on_missing=True),
    )
    NO_MARKER_NEAR_H = NO_MARKER_NEAR

    # Left before it lands: Wind Cutter's wave (a line to the wall; a sidestep beats it).
    # Great Serpent's cut is parried like any other — from melee a 4.9 m blade cannot be
    # outrun in its windup, and the lightning it starts is fled afterwards (LIGHTNING_WALKING).
    FAR_REACHING = Condition.field("attack.id", "==", "warden_draw2")

    def unstick(state: WorldState) -> list[str] | None:
        # Wedged on a prop: drop the flee, go sideways — the other way from last time.
        if state.num("player.stuck_ms", 0.0) < STUCK_MS:
            return None
        last = memory.last_dodge[0] if memory.last_dodge else "d"
        key = "a" if last == "d" else "d"
        memory.last_dodge = (key, state.captured_at)
        return [key]

    def evade_keys(state: WorldState) -> list[str]:
        stuck = unstick(state)
        if stuck is not None:
            return stuck
        # Strafe always; back away too (in world terms) while there is room behind us.
        keys = [memory.strafe_key(state.text("attack.dodge_key", "d"), state.captured_at)]
        edge = state.num("arena.edge_m", 99.0)
        if state.num("boss.distance_m", 0.0) < EVADE_ORBIT_FROM_M and edge > FLEE_ROOM_M:
            keys.append(_flee(state))
        return keys

    def evade_plan(state: WorldState) -> Plan | None:
        # Wind Cutter's wave rides the locked draw line to the wall, 150 ms after the cut;
        # Great Serpent's lightning starts 4 m in front of the blade the instant it lands
        # and walks toward us. Neither is answered in melee: leave, and keep re-deciding
        # the direction every half second so the wall never ends the dodge.
        # The coin (tuning.dodge_chance) is tossed once per telegraph: a "no" stands in
        # for the player who misread the tell, and the parry reflex answers as it would.
        key = state.text("attack.key") if state.flag("attack.active") else state.text("recent.key")
        if not memory.will_dodge(key, tuning.dodge_chance, tuning.random):
            return None
        return Plan(name=f"evade:{state.text('attack.id') or state.text('recent.id')}", steps=tuple(_sprint(evade_keys(state), 0, 650)), expires_in_ms=300)

    LIGHTNING_WALKING = Condition.all_(Condition.field("recent.id", "==", "warden_overhead2"), Condition.field("recent.since_ms", "<", LIGHTNING_MS))

    layer.add(Reflex(
        id="evade_far_reaching", group="evade",
        condition=Condition.all_(ALIVE, Condition.any_(
            Condition.all_(Condition.truthy("attack.active"), FAR_REACHING, Condition.field("attack.land_in_ms", ">", -700)),
            LIGHTNING_WALKING,
        )),
        plan=evade_plan, priority=Priority.REFLEX, cooldown_ms=500, preempt=True,
    ))

    layer.add(Reflex(
        id="parry", group="parry",
        condition=Condition.all_(ALIVE, ATTACK_THREATENS, Condition.truthy("attack.guardable"), Condition.field("attack.land_in_ms", ">", 60)),
        plan=parry_plan, priority=Priority.REFLEX, cooldown_ms=0, preempt=True,
    ))

    # -- ground markers ------------------------------------------------------------

    def away_plan(state: WorldState) -> Plan | None:
        # A declined Great Serpent stays declined: its lightning is taken, not left.
        if state.text("recent.id") == "warden_overhead2" and not memory.will_dodge(state.text("recent.key"), tuning.dodge_chance, tuning.random):
            return None
        # Lightning walks from the boss toward us at 5.5 m/s and hits where its marker
        # appears, so the one direction that always gains ground is straight away from
        # the boss — chosen in world terms (face him, back off) so it stays the same
        # direction even while a horse-bite parry has us facing elsewhere. Against the
        # wall the only move left is to slide along it.
        right = state.num("marker.away_right")
        # Near the wall the flee would only pin us: slide along it while there is still
        # some room, and keep sliding once pinned.
        no_room = state.num("arena.edge_m", 99.0) < FLEE_ROOM_M
        stuck = unstick(state)
        if stuck is not None:
            keys = stuck
        elif no_room or not state.flag("boss.alive"):
            keys = ["d" if right >= 0 else "a"]
        else:
            keys = [_flee(state)]
        return Plan(name=f"away:{state.text('marker.kind')}", steps=tuple(_sprint(keys, 0, 650)), expires_in_ms=300)

    layer.add(Reflex(
        id="leave_marker", group="evade",
        condition=Condition.all_(
            ALIVE, Condition.field("marker.nearest_m", "<", MARKER_FLEE_M), Condition.field("marker.radius_m", "<", 30.0),
            Condition.field("marker.remaining_ms", ">", 0),
        ),
        plan=away_plan, priority=Priority.REFLEX, cooldown_ms=450, preempt=True,
    ))

    # -- the pulse -------------------------------------------------------------------

    layer.add(Reflex(
        id="pulse_retreat", group="cover",
        condition=Condition.all_(ALIVE, Condition.field("boss.phase", "==", "casting"), Condition.field("boss.distance_m", "<", PULSE_SAFE_M)),
        plan=Plan(name="pulse_retreat", steps=tuple(_sprint(["s"], 0, 700)), expires_in_ms=400),
        priority=Priority.RECOVERY, cooldown_ms=750, preempt=True,
    ))

    def cover_plan(state: WorldState) -> Plan | None:
        attempt = int(state.num("boss.attempt"))
        if memory.covered_attempt == attempt:
            return None
        memory.covered_attempt = attempt
        return Plan(name="cover", steps=(Step(0, Press("cover", 40)),), expires_in_ms=1500)

    layer.add(Reflex(
        id="pulse_cover", group="cover",
        condition=Condition.all_(
            ALIVE, Condition.field("boss.phase", "==", "casting"), Condition.field("boss.distance_m", ">=", PULSE_SAFE_M),
            Condition.field("build.stock", ">=", COVER_BLOCKS), Condition.field("boss.remaining_ms", ">", 1200),
        ),
        plan=cover_plan, priority=Priority.RECOVERY, cooldown_ms=1000, preempt=False,
    ))

    # -- the horse -----------------------------------------------------------------

    # -- the kneel: Supreme Holy Water on the revenant (WORLD_BOSSES.md phase 3) ----

    def purify_plan(state: WorldState) -> Plan | None:
        # One bottle per kneel is the threshold; a second only if the first fell short.
        holy_short = "holy:" in state.text("boss.detail") and state.text("boss.detail").split("holy:")[1].split("/")[0].strip() in ("0", "")
        attempt = int(state.num("boss.remaining_ms"))  # the kneel timer: a fresh kneel has a fresh, larger value
        if memory.purified_attempt >= 0 and not holy_short:
            return None
        memory.purified_attempt = attempt
        return Plan(name="purify", steps=(Step(0, Press("use:holy_water_supreme@boss", 40)),), expires_in_ms=800)

    layer.add(Reflex(
        id="purify", group="purify",
        condition=Condition.all_(
            ALIVE, Condition.field("boss.phase", "==", "kneel"), Condition.field("boss.distance_m", "<", 22.0),
            Condition.field("bag.holy_water_supreme", ">", 0), Condition.field("boss.remaining_ms", ">", 400),
        ),
        plan=purify_plan, priority=Priority.RECOVERY, cooldown_ms=1200, preempt=True,
    ))

    layer.add(Reflex(
        id="hold_horse", group="horse",
        condition=Condition.all_(
            ALIVE, Condition.truthy("horse.alive"), Condition.truthy("skill.provoke.ready"),
            Condition.field("horse.distance_m", "<", 24.0), NOT_CASTING,
            # a skill cast lowers both hands for a moment: never inside a landing window
            Condition.not_(Condition.all_(ATTACK_THREATENS, Condition.field("attack.land_in_ms", "<", 900))),
        ),
        plan=Plan(name="provoke_horse", steps=(Step(0, Press("skill:provoke@horse", 40)),), expires_in_ms=600),
        priority=Priority.ROTATION, cooldown_ms=2500, preempt=False,
    ))

    # -- the horse, kited: keep its attention with strikes between provokes -----------

    layer.add(Reflex(
        id="face_horse", group="horse",
        condition=Condition.all_(ALIVE, Condition.truthy("horse.alive"), Condition.not_(Condition.field("target.role", "==", "horse")), NO_ATTACK),
        plan=Plan(name="face_horse", steps=(Step(0, Press("face:horse", 40)),), expires_in_ms=500),
        priority=Priority.ROTATION, cooldown_ms=1500, preempt=False,
    ))
    layer.add(Reflex(
        id="approach_horse", group="horse",
        condition=Condition.all_(ALIVE, NO_ATTACK, NO_MARKER_NEAR_H, Condition.truthy("horse.alive"), Condition.field("target.role", "==", "horse"),
                                 Condition.field("horse.distance_m", ">", APPROACH_FROM_M), NOT_CASTING),
        plan=Plan(name="approach_horse", steps=(Step(0, Press("w", 350)),), expires_in_ms=300),
        priority=Priority.ROTATION, cooldown_ms=250, preempt=False,
    ))
    layer.add(Reflex(
        id="strike_horse", group="horse",
        condition=Condition.all_(ALIVE, NO_ATTACK, Condition.truthy("horse.alive"), Condition.field("target.role", "==", "horse"),
                                 Condition.field("horse.distance_m", "<=", MELEE_M), NOT_CASTING),
        plan=Plan(name="strike_horse", steps=(Step(0, Press("mouse1", 40)),), expires_in_ms=400),
        priority=Priority.ROTATION, cooldown_ms=950, preempt=False,
    ))

    # -- standing by: stay near whoever summoned us ------------------------------------

    layer.add(Reflex(
        id="follow_ally", group="follow",
        condition=Condition.all_(ALIVE, NO_ATTACK, Condition.truthy("ally.exists"), Condition.field("ally.distance_m", ">", 5.0)),
        plan=Plan(name="follow_ally", steps=(Step(0, Press("face:ally", 20)), Step(60, Press("w", 400)), Step(60, Press("shift", 400))), expires_in_ms=300),
        priority=Priority.ROTATION, cooldown_ms=300, preempt=False,
    ))

    # -- engagement ----------------------------------------------------------------

    layer.add(Reflex(
        id="face_boss", group="engage",
        condition=Condition.all_(ALIVE, Condition.truthy("boss.alive"), Condition.not_(Condition.field("target.role", "==", "boss"))),
        plan=Plan(name="face_boss", steps=(Step(0, Press("face:boss", 40)),), expires_in_ms=500),
        priority=Priority.ROTATION, cooldown_ms=1500, preempt=False,
    ))

    def approach_plan(state: WorldState) -> Plan:
        far = state.num("boss.distance_m") > 7.0
        steps = [Step(0, Press("w", 350))]
        if far:
            steps.append(Step(0, Press("shift", 350)))
        return Plan(name="approach", steps=tuple(steps), expires_in_ms=300)


    layer.add(Reflex(
        id="approach", group="engage",
        condition=Condition.all_(ALIVE, NO_ATTACK, NO_MARKER_NEAR, Condition.not_(LIGHTNING_WALKING), Condition.truthy("boss.alive"),
                                 Condition.field("boss.distance_m", ">", APPROACH_FROM_M), NOT_CASTING, Condition.field("target.role", "==", "boss")),
        plan=approach_plan, priority=Priority.ROTATION, cooldown_ms=250, preempt=False,
    ))

    layer.add(Reflex(
        id="back_off", group="engage",
        condition=Condition.all_(ALIVE, NO_ATTACK, Condition.truthy("boss.alive"), Condition.field("boss.distance_m", "<", TOO_CLOSE_M), NOT_CASTING),
        plan=Plan(name="back_off", steps=(Step(0, Press("s", 220)),), expires_in_ms=300),
        priority=Priority.ROTATION, cooldown_ms=300, preempt=False,
    ))

    layer.add(Reflex(
        id="strike", group="engage",
        condition=Condition.all_(
            ALIVE, NO_ATTACK, Condition.truthy("boss.alive"), Condition.field("boss.distance_m", "<=", MELEE_M), NOT_CASTING,
            Condition.any_(Condition.truthy("boss.opening"), Condition.field("boss.phase", "==", "mastered")),
        ),
        plan=Plan(name="strike", steps=(Step(0, Press("mouse1", 40)),), expires_in_ms=400),
        priority=Priority.ROTATION, cooldown_ms=950, preempt=False,
    ))

    return layer


__all__ = ["WardenMemory", "WardenTuning", "build_warden_reflexes", "PARRY_LEAD_MS", "PARRY_HOLD_MS"]
