"""Per-job rotation profiles.

A profile is a priority-ordered ability list plus timing constants. Order is the
priority, which mirrors how players write and review rotations — and the person
maintaining these lists is reasoning about the game, not about the code.
"""

from . import blm

BUILDERS = {
    "blm": blm.build,
}


def available_jobs() -> list[str]:
    return sorted(BUILDERS)


__all__ = ["BUILDERS", "available_jobs", "blm"]
