"""FantCraft reflexes.

Three rungs, all driven by fields the sensors already read:

* **acquire_target** — Tab cycles the hard-lock to the nearest living mob and
  (since the auto-face change) turns the camera toward it. No target on
  screen means one keypress fixes both aim and facing.

* **approach_target** — the movement half of playing: when the primary
  ability's slot wears the out-of-range wash, hold W. Tab already pointed the
  camera at the target, so forward *is* toward it. Fires repeatedly until the
  wash clears, at which point the rotation takes over.

* **retreat_low_hp** — FantCraft has no dash i-frames, so low-HP survival is
  distance plus the out-of-combat regen that kicks in once the mob leashes.
  Backpedal with sprint and let the rotation resume when HP recovers.
"""

from __future__ import annotations

from player.act.timeline import Plan, Press, Priority, Step
from player.clock import now
from player.policy.condition import Condition
from player.policy.reflex import Reflex, ReflexLayer


def build_reflexes(
    keys: dict[str, str],
    *,
    retreat_hp_threshold: float = 0.25,
    approach_ability: str = "iron_cleave",
) -> ReflexLayer:
    layer = ReflexLayer()

    layer.add(
        Reflex(
            id="acquire_target",
            group="targeting",
            condition=Condition.falsy("target.exists"),
            plan=lambda _state: Plan(
                name="tab_target",
                steps=(Step(0, Press(keys["target_cycle"], 40)),),
                priority=Priority.ROTATION,
                preempt=False,
                expires_in_ms=400,
            ),
            priority=Priority.ROTATION,
            # Tab again only after the target frame has had time to appear and
            # the vitals sensor to read it — re-tabbing every tick would cycle
            # straight past the mob we just acquired.
            cooldown_ms=1500,
            preempt=False,
        )
    )

    # Streak accounting lives in a closure: consecutive approach fires with no
    # gap mean the range wash never cleared — i.e. walking isn't working, we
    # are pinned on geometry. A >3s gap between fires means the wash cleared
    # (or the target changed) and the episode starts over.
    approach_state = {"fires": 0, "last_at": float("-inf")}

    def approach_plan(_state) -> Plan:
        at = now()
        if at - approach_state["last_at"] > 3.0:
            approach_state["fires"] = 0
        approach_state["last_at"] = at
        approach_state["fires"] += 1

        # Walk + hop: the hop clears the 1-block fences voxel towns love
        # (live2's bot stood at one until the timer ran out).
        steps = [
            Step(0, Press(keys["forward"], 700)),
            Step(200, Press(keys["jump"], 80)),
        ]
        fires = approach_state["fires"]
        if fires % 4 == 3:
            # Re-face: auto-face fires on Tab, and any detour makes the old
            # heading stale — phaseA2's bot rounded the wall and then marched
            # 200 units past the target on the original bearing.
            steps.append(Step(60, Press(keys["target_cycle"], 40)))
        if 6 <= fires < 14:
            # ~5s of walking without the wash clearing: pinned on a wall
            # (phaseA reproduced it on a building corner). Slide along it —
            # forward stays held, the alternating strafe traces the obstacle.
            strafe = keys["left"] if fires % 4 < 2 else keys["right"]
            steps.append(Step(80, Press(strafe, 600)))
        if fires >= 14:
            # Hopeless episode: drop the lock. target.exists goes false, this
            # reflex stops firing, and acquire_target hunts fresh from wherever
            # the detours left us.
            approach_state["fires"] = 0
            return Plan(
                name="abandon_target",
                steps=(Step(0, Press(keys["target_clear"], 40)),),
                priority=Priority.ROTATION,
                preempt=False,
                expires_in_ms=500,
            )
        return Plan(
            name="approach",
            steps=tuple(steps),
            priority=Priority.ROTATION,
            preempt=False,
            expires_in_ms=500,
        )

    layer.add(
        Reflex(
            id="approach_target",
            group="movement",
            condition=Condition.all_(
                Condition.truthy("target.exists"),
                Condition.truthy(f"action.{approach_ability}.out_of_range"),
            ),
            plan=approach_plan,
            priority=Priority.ROTATION,
            cooldown_ms=800,
            preempt=False,
        )
    )

    layer.add(
        Reflex(
            id="retreat_low_hp",
            group="survival",
            condition=Condition.hp_below(retreat_hp_threshold),
            plan=lambda _state: Plan(
                name="retreat",
                steps=(
                    Step(0, Press(keys["back"], 1200)),
                    Step(60, Press(keys["sprint"], 1100)),
                ),
                priority=Priority.REFLEX,
                preempt=True,
                expires_in_ms=400,
            ),
            priority=Priority.REFLEX,
            cooldown_ms=5000,
            preempt=True,
        )
    )

    return layer
