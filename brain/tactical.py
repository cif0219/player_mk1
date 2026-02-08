"""
Tactical Layer (戰術層)
Async goal execution at medium frequency.
Manages reaction rules dynamically based on current situation.
Target: 10-30 Hz
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
from queue import Queue

from .reaction import ReactionLayer, ReactionRule


class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class Goal:
    """A tactical goal assigned by the strategic layer."""
    id: str
    type: str
    params: dict = field(default_factory=dict)
    priority: int = 50
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    # Reaction rules to install when this goal is active
    reaction_rules: list[ReactionRule] = field(default_factory=list)
    # Tag for all rules from this goal (for easy cleanup)
    rule_tag: str = field(default="")
    
    def __post_init__(self):
        if not self.rule_tag:
            self.rule_tag = f"goal:{self.id}"
        for rule in self.reaction_rules:
            rule.tags.add(self.rule_tag)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class TacticalContext:
    """Context passed to behaviors each tick."""
    reaction_layer: ReactionLayer
    frame: Any  # numpy array or None
    game_state: dict
    goal: "Goal"
    
    def install_rule(self, rule: ReactionRule):
        rule.tags.add(self.goal.rule_tag)
        self.reaction_layer.add_rule(rule)
    
    def remove_rule(self, rule_id: str):
        self.reaction_layer.remove_rule(rule_id)


class TacticalBehavior(ABC):
    """Base class for tactical behaviors."""
    
    @property
    @abstractmethod
    def goal_type(self) -> str:
        """The goal type this behavior handles."""
        pass
    
    @abstractmethod
    def tick(self, ctx: TacticalContext) -> Optional[dict]:
        """Execute one tick. Returns action dict or None."""
        pass
    
    def on_start(self, ctx: TacticalContext):
        """Called when goal starts."""
        pass
    
    def on_complete(self, ctx: TacticalContext):
        """Called when goal completes."""
        pass
    
    def on_cancel(self, ctx: TacticalContext):
        """Called when goal is cancelled."""
        pass


class TacticalLayer:
    """
    Thread-safe tactical goal manager.
    Goals can be pushed/cancelled from any thread (strategic layer).
    """
    
    def __init__(self, reaction_layer: ReactionLayer):
        self.reaction_layer = reaction_layer
        self._goals: list[Goal] = []
        self._active_goal: Optional[Goal] = None
        self._behaviors: dict[str, TacticalBehavior] = {}
        self._active_behavior: Optional[TacticalBehavior] = None
        self._game_state: dict = {}
        self._lock = threading.RLock()
    
    def register_behavior(self, behavior: TacticalBehavior):
        self._behaviors[behavior.goal_type] = behavior
    
    def push_goal(self, goal: Goal):
        with self._lock:
            # Don't add duplicate goals
            if any(g.id == goal.id for g in self._goals):
                return
            self._goals.append(goal)
            self._goals.sort(key=lambda g: -g.priority)
    
    def cancel_goal(self, goal_id: str):
        with self._lock:
            for goal in self._goals:
                if goal.id == goal_id:
                    goal.status = TaskStatus.CANCELLED
            
            if self._active_goal and self._active_goal.id == goal_id:
                self._cleanup_active()
    
    def cancel_by_type(self, goal_type: str):
        with self._lock:
            for goal in self._goals:
                if goal.type == goal_type:
                    goal.status = TaskStatus.CANCELLED
            
            if self._active_goal and self._active_goal.type == goal_type:
                self._cleanup_active()
    
    def clear_goals(self):
        with self._lock:
            for goal in self._goals:
                self.reaction_layer.remove_by_tag(goal.rule_tag)
            self._goals.clear()
            self._cleanup_active()
    
    def update_game_state(self, state: dict):
        with self._lock:
            self._game_state.update(state)
    
    def process(self, frame: Any) -> Optional[dict]:
        """Process one tactical tick. Returns action or None."""
        with self._lock:
            # Clean completed/failed/cancelled
            self._goals = [g for g in self._goals 
                          if g.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]
            
            # Select goal if needed
            if self._active_goal is None or self._active_goal.status != TaskStatus.RUNNING:
                self._select_next_goal(frame)
            
            # Execute
            if self._active_goal and self._active_behavior:
                ctx = TacticalContext(
                    reaction_layer=self.reaction_layer,
                    frame=frame,
                    game_state=self._game_state.copy(),
                    goal=self._active_goal
                )
                
                try:
                    action = self._active_behavior.tick(ctx)
                    
                    if self._active_goal.status == TaskStatus.COMPLETED:
                        self._active_behavior.on_complete(ctx)
                        self._cleanup_active()
                    
                    return action
                except Exception as e:
                    self._active_goal.status = TaskStatus.FAILED
                    self._cleanup_active()
            
            return None
    
    def _select_next_goal(self, frame: Any):
        for goal in self._goals:
            if goal.status != TaskStatus.PENDING:
                continue
            
            behavior = self._behaviors.get(goal.type)
            if behavior:
                self._active_goal = goal
                self._active_behavior = behavior
                goal.status = TaskStatus.RUNNING
                
                # Install goal's reaction rules
                for rule in goal.reaction_rules:
                    self.reaction_layer.add_rule(rule)
                
                ctx = TacticalContext(
                    reaction_layer=self.reaction_layer,
                    frame=frame,
                    game_state=self._game_state.copy(),
                    goal=goal
                )
                behavior.on_start(ctx)
                return
            else:
                goal.status = TaskStatus.FAILED
    
    def _cleanup_active(self):
        if self._active_goal:
            self.reaction_layer.remove_by_tag(self._active_goal.rule_tag)
            self._active_goal = None
            self._active_behavior = None
    
    @property
    def active_goal(self) -> Optional[Goal]:
        with self._lock:
            return self._active_goal
    
    @property
    def pending_goals(self) -> list[Goal]:
        with self._lock:
            return [g for g in self._goals if g.status == TaskStatus.PENDING]
    
    @property
    def goal_count(self) -> int:
        return len(self._goals)


class TacticalThread(threading.Thread):
    """
    Async tactical processor.
    Reads frames, outputs actions to queue.
    """
    
    def __init__(
        self,
        frame_buffer,  # SharedFrameBuffer
        action_queue: Queue,
        layer: TacticalLayer,
        target_fps: int = 20
    ):
        super().__init__(daemon=True)
        self.frame_buffer = frame_buffer
        self.action_queue = action_queue
        self.layer = layer
        self.target_fps = target_fps
        
        self._running = False
        self._last_frame_index = 0
        
        # Stats
        self.ticks_per_sec = 0.0
        self.actions_per_sec = 0.0
    
    def run(self):
        self._running = True
        interval = 1.0 / self.target_fps
        last_stats = time.perf_counter()
        ticks = 0
        actions = 0
        
        while self._running:
            start = time.perf_counter()
            
            frame_obj = self.frame_buffer.get_latest()
            frame = frame_obj.image if frame_obj else None
            
            action = self.layer.process(frame)
            ticks += 1
            
            if action:
                try:
                    self.action_queue.put_nowait(("tactical", action, time.perf_counter()))
                    actions += 1
                except:
                    pass
            
            # Stats
            now = time.perf_counter()
            if now - last_stats >= 1.0:
                self.ticks_per_sec = ticks / (now - last_stats)
                self.actions_per_sec = actions / (now - last_stats)
                ticks = 0
                actions = 0
                last_stats = now
            
            elapsed = time.perf_counter() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def stop(self):
        self._running = False


# ============================================================
# Built-in Behaviors
# ============================================================

class WaitBehavior(TacticalBehavior):
    goal_type = "wait"
    
    def tick(self, ctx: TacticalContext) -> Optional[dict]:
        duration = ctx.goal.params.get("duration_ms", 1000)
        elapsed = (time.time() - ctx.goal.created_at) * 1000
        ctx.goal.progress = min(1.0, elapsed / duration)
        
        if elapsed >= duration:
            ctx.goal.status = TaskStatus.COMPLETED
        
        return None  # No action, just wait


class SequenceBehavior(TacticalBehavior):
    goal_type = "sequence"
    
    def on_start(self, ctx: TacticalContext):
        ctx.goal.params.setdefault("_index", 0)
    
    def tick(self, ctx: TacticalContext) -> Optional[dict]:
        actions = ctx.goal.params.get("actions", [])
        index = ctx.goal.params.get("_index", 0)
        
        if index >= len(actions):
            ctx.goal.status = TaskStatus.COMPLETED
            ctx.goal.progress = 1.0
            return None
        
        action = actions[index]
        ctx.goal.params["_index"] = index + 1
        ctx.goal.progress = (index + 1) / len(actions)
        
        return action


class RepeatBehavior(TacticalBehavior):
    goal_type = "repeat"
    
    def on_start(self, ctx: TacticalContext):
        ctx.goal.params.setdefault("_count", 0)
    
    def tick(self, ctx: TacticalContext) -> Optional[dict]:
        action = ctx.goal.params.get("action", {"type": "wait", "ms": 100})
        times = ctx.goal.params.get("times", 1)
        count = ctx.goal.params["_count"]
        
        if count >= times:
            ctx.goal.status = TaskStatus.COMPLETED
            return None
        
        ctx.goal.params["_count"] = count + 1
        ctx.goal.progress = (count + 1) / times
        
        return action
