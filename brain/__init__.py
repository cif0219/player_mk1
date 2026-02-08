"""
Player MK1 - Brain Module
Async three-layer decision architecture with commander interface
"""

from .reaction import (
    ReactionLayer, ReactionRule, 
    Trigger, PixelTrigger, RegionTrigger, ValueTrigger, CompositeTrigger,
    ReactionThread
)
from .tactical import (
    TacticalLayer, TacticalThread, TacticalBehavior, TacticalContext,
    Goal, TaskStatus,
    WaitBehavior, SequenceBehavior, RepeatBehavior
)
from .strategic import (
    StrategicLayer, StrategicThread,
    StrategicState, StrategicDecision,
    Commander, CommanderDirective, QueueCommander, WebSocketCommander
)

__all__ = [
    # Reaction
    "ReactionLayer", "ReactionRule", "ReactionThread",
    "Trigger", "PixelTrigger", "RegionTrigger", "ValueTrigger", "CompositeTrigger",
    # Tactical
    "TacticalLayer", "TacticalThread", "TacticalBehavior", "TacticalContext",
    "Goal", "TaskStatus",
    "WaitBehavior", "SequenceBehavior", "RepeatBehavior",
    # Strategic
    "StrategicLayer", "StrategicThread",
    "StrategicState", "StrategicDecision",
    # Commander interface
    "Commander", "CommanderDirective", "QueueCommander", "WebSocketCommander",
]
