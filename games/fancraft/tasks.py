"""Helper tasks: what a summoned player_mk1 does (fancraft docs/PLAYTEST.md §7).

Mirrors fancraft `shared/src/helperTasks.ts` — keep the two in lockstep. A
task is a set of reflex groups (games/fancraft/warden.py) plus the answer
style; every task keeps the survival groups on, what differs is what the
helper goes after.
"""

from __future__ import annotations

from dataclasses import dataclass

ALL_GROUPS = ("parry", "dodge", "evade", "cover", "horse", "engage", "purify", "follow")


@dataclass(frozen=True, slots=True)
class HelperTask:
    id: str
    groups: tuple[str, ...]
    tactic: str = "parry"


HELPER_TASKS: dict[str, HelperTask] = {t.id: t for t in (
    HelperTask("full", ("parry", "dodge", "evade", "cover", "horse", "engage", "purify")),
    HelperTask("standby", ("parry", "dodge", "evade", "follow")),
    HelperTask("tank", ("parry", "dodge", "evade", "engage")),
    HelperTask("hold_horse", ("parry", "dodge", "evade", "horse")),
    HelperTask("samurai", ("parry", "dodge", "evade", "cover", "engage", "purify")),
    HelperTask("evade_only", ("dodge", "evade", "engage"), tactic="dodge"),
    HelperTask("purify", ("parry", "dodge", "evade", "engage", "purify")),
    HelperTask("cover", ("parry", "dodge", "evade", "cover")),
    # Any instance (fancraft docs/COMBAT.md §7.3): the gateway runs games/fancraft/companion.py for this task
    HelperTask("companion", ("follow", "engage")),
)}


def task(task_id: str) -> HelperTask:
    return HELPER_TASKS.get(task_id, HELPER_TASKS["full"])


__all__ = ["ALL_GROUPS", "HELPER_TASKS", "HelperTask", "task"]
