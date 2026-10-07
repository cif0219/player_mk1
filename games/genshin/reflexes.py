"""Genshin reflexes.

Genshin telegraphs attacks with animation wind-ups rather than persistent ground decals,
so there is no cheap HSV segmentation to bootstrap a dodge detector from — the FFXIV
trick does not transfer. What survives without a detector is HP-triggered survival: the
dash has generous invulnerability frames, so dashing on a sudden HP drop is the honest
80% of defensive play, and it needs nothing but the HP bar the vitals sensor already
reads.
"""

from __future__ import annotations

from player.act.timeline import Plan, Press, Priority, Step
from player.policy.condition import Condition
from player.policy.reflex import Reflex, ReflexLayer


def dash_plan(sprint_key: str, hold_ms: int = 280) -> Plan:
    """A dash: the sprint key tapped-and-held briefly.

    A short hold produces a dash rather than sustained sprint, which is what carries the
    i-frames. No direction key — dashing backward from whatever is hitting you is the
    default the camera already provides.
    """
    return Plan(
        name="dash",
        steps=(Step(0, Press(sprint_key, hold_ms)),),
        priority=Priority.REFLEX,
        preempt=True,
        expires_in_ms=250,  # a late dash spends its i-frames after the hit landed
    )


def build_reflexes(
    keys: dict[str, str],
    *,
    dash_hp_threshold: float = 0.4,
    retreat_hp_threshold: float = 0.2,
) -> ReflexLayer:
    """Two survival rungs, both in the `survival` group so the director can toggle them.

    * dash at `dash_hp_threshold` — cheap, spammable, buys i-frames.
    * retreat at `retreat_hp_threshold` — dash plus a burst of backward movement,
      because at 20% HP the correct play is distance, not damage.
    """
    layer = ReflexLayer()

    layer.add(
        Reflex(
            id="dash_low_hp",
            group="survival",
            condition=Condition.all_(
                Condition.hp_below(dash_hp_threshold),
                Condition.field("player.hp_frac", ">=", retreat_hp_threshold),
            ),
            plan=lambda _state: dash_plan(keys["sprint"]),
            priority=Priority.REFLEX,
            # HP stays low after the dash, so without a cooldown this re-fires every
            # frame and pins the character dashing in place.
            cooldown_ms=3000,
            preempt=True,
        )
    )

    layer.add(
        Reflex(
            id="retreat_critical_hp",
            group="survival",
            condition=Condition.hp_below(retreat_hp_threshold),
            plan=lambda _state: Plan(
                name="retreat",
                steps=(
                    Step(0, Press(keys["back"], 900)),
                    Step(80, Press(keys["sprint"], 700)),
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
