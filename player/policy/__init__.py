"""Deciding what to do, given a `WorldState`.

Policy never touches a pixel and never touches the OS. It reads the contract and emits
plans. That is what makes every decision in here testable in three lines with no image,
no clock, and no game.
"""

from .condition import Condition, parse_condition
from .reflex import Reflex, ReflexLayer
from .rotation import Ability, AbilityKind, RotationPlanner, RotationProfile

__all__ = [
    "Ability",
    "AbilityKind",
    "Condition",
    "Reflex",
    "ReflexLayer",
    "RotationPlanner",
    "RotationProfile",
    "parse_condition",
]
