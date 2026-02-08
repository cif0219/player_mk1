"""
Reaction Layer (反應層)
Async ultra-fast responses using pure math and pattern matching.
Runs in dedicated thread at highest frequency.
Target latency: <1ms per check
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from abc import ABC, abstractmethod
from queue import Queue, Empty
import numpy as np


# ============================================================
# Triggers - conditions that fire reactions
# ============================================================

class Trigger(ABC):
    """Base class for reaction triggers."""
    
    @abstractmethod
    def check(self, frame: np.ndarray) -> bool:
        """Check if trigger condition is met. Must be FAST (<1ms)."""
        pass


@dataclass
class PixelTrigger(Trigger):
    """Trigger when pixels in a region match a color."""
    region: tuple[int, int, int, int]  # x, y, width, height
    color: tuple[int, int, int]        # RGB
    threshold: float = 0.8             # % of pixels that must match
    tolerance: int = 20                # color distance tolerance
    
    def check(self, frame: np.ndarray) -> bool:
        x, y, w, h = self.region
        roi = frame[y:y+h, x:x+w]
        diff = np.abs(roi.astype(np.int16) - np.array(self.color))
        matches = np.all(diff <= self.tolerance, axis=2)
        return np.mean(matches) >= self.threshold


@dataclass
class RegionTrigger(Trigger):
    """Trigger based on region brightness or change detection."""
    region: tuple[int, int, int, int]
    mode: str = "brightness"  # brightness, change
    threshold: float = 0.5
    _prev_roi: Optional[np.ndarray] = field(default=None, repr=False)
    
    def check(self, frame: np.ndarray) -> bool:
        x, y, w, h = self.region
        roi = frame[y:y+h, x:x+w]
        
        if self.mode == "brightness":
            return np.mean(roi) / 255.0 >= self.threshold
        elif self.mode == "change":
            if self._prev_roi is None:
                self._prev_roi = roi.copy()
                return False
            diff = np.mean(np.abs(roi.astype(np.int16) - self._prev_roi.astype(np.int16))) / 255.0
            self._prev_roi = roi.copy()
            return diff >= self.threshold
        return False


@dataclass
class ValueTrigger(Trigger):
    """Trigger based on external value (e.g., parsed health bar)."""
    getter: Callable[[], float]
    operator: str = "lt"  # lt, gt, le, ge, eq
    threshold: float = 0.3
    
    def check(self, frame: np.ndarray) -> bool:
        value = self.getter()
        ops = {"lt": lambda v, t: v < t, "gt": lambda v, t: v > t,
               "le": lambda v, t: v <= t, "ge": lambda v, t: v >= t,
               "eq": lambda v, t: abs(v - t) < 0.01}
        return ops.get(self.operator, lambda v, t: False)(value, self.threshold)


@dataclass
class CompositeTrigger(Trigger):
    """Combine multiple triggers with AND/OR logic."""
    triggers: list[Trigger]
    mode: str = "and"  # and, or
    
    def check(self, frame: np.ndarray) -> bool:
        if self.mode == "and":
            return all(t.check(frame) for t in self.triggers)
        else:
            return any(t.check(frame) for t in self.triggers)


# ============================================================
# Reaction Rules
# ============================================================

@dataclass
class ReactionRule:
    """A single reaction rule: trigger → action."""
    id: str
    trigger: Trigger
    action: dict                       # Action dict for operator
    priority: int = 50                 # Higher = checked first
    cooldown_ms: float = 0
    enabled: bool = True
    max_fires: Optional[int] = None
    tags: set[str] = field(default_factory=set)  # For batch enable/disable
    
    _last_fired: float = field(default=0, repr=False)
    _fire_count: int = field(default=0, repr=False)
    
    def can_fire(self) -> bool:
        if not self.enabled:
            return False
        if self.max_fires and self._fire_count >= self.max_fires:
            return False
        if self.cooldown_ms > 0:
            elapsed = (time.perf_counter() - self._last_fired) * 1000
            if elapsed < self.cooldown_ms:
                return False
        return True
    
    def mark_fired(self):
        self._last_fired = time.perf_counter()
        self._fire_count += 1
        if self.max_fires and self._fire_count >= self.max_fires:
            self.enabled = False
    
    def reset(self):
        self._fire_count = 0
        self._last_fired = 0
        self.enabled = True


# ============================================================
# Reaction Layer (Thread-Safe)
# ============================================================

class ReactionLayer:
    """
    Thread-safe reaction rule manager.
    Rules can be added/removed from any thread.
    """
    
    def __init__(self):
        self._rules: dict[str, ReactionRule] = {}
        self._sorted_rules: list[ReactionRule] = []
        self._lock = threading.RLock()
    
    def add_rule(self, rule: ReactionRule):
        with self._lock:
            self._rules[rule.id] = rule
            self._rebuild_sorted()
    
    def add_rules(self, rules: list[ReactionRule]):
        with self._lock:
            for rule in rules:
                self._rules[rule.id] = rule
            self._rebuild_sorted()
    
    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                self._rebuild_sorted()
                return True
            return False
    
    def remove_by_tag(self, tag: str) -> int:
        """Remove all rules with a specific tag."""
        with self._lock:
            to_remove = [rid for rid, r in self._rules.items() if tag in r.tags]
            for rid in to_remove:
                del self._rules[rid]
            if to_remove:
                self._rebuild_sorted()
            return len(to_remove)
    
    def enable_rule(self, rule_id: str, enabled: bool = True):
        with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id].enabled = enabled
    
    def enable_by_tag(self, tag: str, enabled: bool = True):
        with self._lock:
            for rule in self._rules.values():
                if tag in rule.tags:
                    rule.enabled = enabled
    
    def clear_rules(self):
        with self._lock:
            self._rules.clear()
            self._sorted_rules.clear()
    
    def _rebuild_sorted(self):
        self._sorted_rules = sorted(self._rules.values(), key=lambda r: -r.priority)
    
    def process(self, frame: np.ndarray) -> Optional[dict]:
        """Check all rules. Returns first matching action or None."""
        with self._lock:
            rules = list(self._sorted_rules)
        
        for rule in rules:
            if not rule.can_fire():
                continue
            try:
                if rule.trigger.check(frame):
                    rule.mark_fired()
                    return rule.action
            except Exception:
                pass
        return None
    
    def get_rules(self) -> list[ReactionRule]:
        with self._lock:
            return list(self._sorted_rules)
    
    @property
    def rule_count(self) -> int:
        return len(self._rules)


# ============================================================
# Reaction Thread
# ============================================================

class ReactionThread(threading.Thread):
    """
    Async reaction processor.
    Reads from shared frame buffer, outputs actions to queue.
    """
    
    def __init__(
        self,
        frame_buffer,  # SharedFrameBuffer
        action_queue: Queue,
        layer: Optional[ReactionLayer] = None,
        target_fps: int = 120
    ):
        super().__init__(daemon=True)
        self.frame_buffer = frame_buffer
        self.action_queue = action_queue
        self.layer = layer or ReactionLayer()
        self.target_fps = target_fps
        
        self._running = False
        self._last_frame_index = 0
        
        # Stats
        self.checks_per_sec = 0.0
        self.fires_per_sec = 0.0
        self._check_times: list[float] = []
        self._fire_count = 0
    
    def run(self):
        self._running = True
        interval = 1.0 / self.target_fps
        last_stats = time.perf_counter()
        checks = 0
        fires = 0
        
        while self._running:
            start = time.perf_counter()
            
            # Get latest frame
            frame = self.frame_buffer.get_latest()
            if frame is None or frame.index == self._last_frame_index:
                time.sleep(0.001)  # No new frame
                continue
            
            self._last_frame_index = frame.index
            
            # Process
            action = self.layer.process(frame.image)
            checks += 1
            
            if action:
                try:
                    self.action_queue.put_nowait(("reaction", action, time.perf_counter()))
                    fires += 1
                except:
                    pass
            
            # Stats every second
            now = time.perf_counter()
            if now - last_stats >= 1.0:
                self.checks_per_sec = checks / (now - last_stats)
                self.fires_per_sec = fires / (now - last_stats)
                checks = 0
                fires = 0
                last_stats = now
            
            # Maintain FPS
            elapsed = time.perf_counter() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def stop(self):
        self._running = False


def benchmark_reactions(layer: ReactionLayer, frame: np.ndarray, iterations: int = 1000) -> float:
    """Returns avg microseconds per process()."""
    start = time.perf_counter()
    for _ in range(iterations):
        layer.process(frame)
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1_000_000


if __name__ == "__main__":
    layer = ReactionLayer()
    rule = ReactionRule(
        id="test_red",
        trigger=PixelTrigger(region=(100, 100, 50, 50), color=(255, 0, 0), threshold=0.5),
        action={"type": "key", "key": "space"},
        priority=100
    )
    layer.add_rule(rule)
    
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[100:150, 100:150] = [255, 0, 0]
    
    avg_us = benchmark_reactions(layer, frame, 10000)
    print(f"Avg reaction time: {avg_us:.2f} µs ({1_000_000/avg_us:.0f} checks/sec possible)")
