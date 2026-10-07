"""Executing scripted encounter mechanics.

The design argument is in `docs/ENCOUNTERS.md`; the short version is that a raid guide is
*compiled* offline into an `EncounterScript` and executed deterministically here, with no
model in the runtime loop.

This package implements the mechanic loop, which runs **concurrently with the rotation
loop** and coordinates with it through `policy/commitment.py` rather than by preemption.
"""

from .runner import MechanicIntent, MechanicRunner, Phase
from .script import (
    CastTrigger,
    DebuffTrigger,
    EncounterScript,
    LookPlan,
    Mechanic,
    Resolution,
    TimelineTrigger,
    Trigger,
)

__all__ = [
    "CastTrigger",
    "DebuffTrigger",
    "EncounterScript",
    "LookPlan",
    "Mechanic",
    "MechanicIntent",
    "MechanicRunner",
    "Phase",
    "Resolution",
    "TimelineTrigger",
    "Trigger",
]
