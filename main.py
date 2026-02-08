"""
Player MK1 - General Purpose Game Player
Fully async three-layer architecture

Architecture:
┌─────────────────────────────────────────────────────────────┐
│  VisionThread (60+ FPS)                                     │
│    └──▶ SharedFrameBuffer                                   │
└─────────────────────────────────────────────────────────────┘
                    │
          ┌────────┴────────┬─────────────────┐
          ▼                 ▼                 ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ ReactionThread   │ │TacticalThread│ │ StrategicThread  │
│ (120+ FPS)       │ │ (20 Hz)      │ │ (every 5s)       │
│                  │ │              │ │                  │
│ Pattern matching │ │ Goal exec    │ │ LLM analysis     │
│ Instant response │ │ Rule mgmt    │ │ Set objectives   │
└────────┬─────────┘ └──────┬───────┘ └────────┬─────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  ActionQueue   │
                   └────────┬───────┘
                            ▼
                   ┌────────────────┐
                   │ OperatorThread │
                   │ (executes all) │
                   └────────────────┘
"""

import time
import yaml
import argparse
import threading
from queue import Queue, Empty
from typing import Optional

from vision import SharedFrameBuffer, VisionThread
from brain import ReactionLayer, TacticalLayer
from brain.reaction import ReactionThread, ReactionRule, PixelTrigger
from brain.tactical import TacticalThread, Goal, WaitBehavior, SequenceBehavior, RepeatBehavior
from brain.strategic import StrategicLayer, StrategicThread
from operator import Operator, SafetyConfig


class OperatorThread(threading.Thread):
    """
    Consumes actions from queue and executes them.
    Prioritizes reaction layer actions.
    """
    
    def __init__(self, action_queue: Queue, operator: Operator):
        super().__init__(daemon=True)
        self.action_queue = action_queue
        self.operator = operator
        self._running = False
        
        # Stats
        self.actions_executed = 0
        self.by_layer = {"reaction": 0, "tactical": 0, "strategic": 0}
    
    def run(self):
        self._running = True
        
        while self._running:
            try:
                layer, action, timestamp = self.action_queue.get(timeout=0.1)
                
                # Skip stale actions (older than 100ms)
                age_ms = (time.perf_counter() - timestamp) * 1000
                if age_ms > 100:
                    continue
                
                if not self.operator.is_stopped:
                    self.operator.execute(action)
                    self.actions_executed += 1
                    self.by_layer[layer] = self.by_layer.get(layer, 0) + 1
                    
            except Empty:
                continue
    
    def stop(self):
        self._running = False


class Player:
    """
    Main game player - orchestrates all async components.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        
        # Shared resources
        self.frame_buffer = SharedFrameBuffer(max_frames=10)
        self.action_queue = Queue(maxsize=100)
        
        # Vision
        vision_cfg = self.config["vision"]
        self.vision_thread = VisionThread(
            buffer=self.frame_buffer,
            region=vision_cfg.get("region"),
            resize=tuple(vision_cfg["resize"]) if vision_cfg.get("resize") else None,
            target_fps=vision_cfg.get("fps", 60)
        )
        
        # Brain layers
        self.reaction_layer = ReactionLayer()
        self.tactical_layer = TacticalLayer(self.reaction_layer)
        self.strategic_layer = StrategicLayer(
            self.tactical_layer,
            provider=self.config["brain"]["provider"],
            model=self.config["brain"]["model"],
            review_interval_ms=self.config["brain"].get("strategic_interval_ms", 5000)
        )
        
        # Register default behaviors
        self.tactical_layer.register_behavior(WaitBehavior())
        self.tactical_layer.register_behavior(SequenceBehavior())
        self.tactical_layer.register_behavior(RepeatBehavior())
        
        # Brain threads
        reaction_cfg = self.config["brain"].get("reaction", {})
        self.reaction_thread = ReactionThread(
            frame_buffer=self.frame_buffer,
            action_queue=self.action_queue,
            layer=self.reaction_layer,
            target_fps=reaction_cfg.get("fps", 120)
        )
        
        tactical_cfg = self.config["brain"].get("tactical", {})
        self.tactical_thread = TacticalThread(
            frame_buffer=self.frame_buffer,
            action_queue=self.action_queue,
            layer=self.tactical_layer,
            target_fps=tactical_cfg.get("fps", 20)
        )
        
        self.strategic_thread = StrategicThread(
            frame_buffer=self.frame_buffer,
            layer=self.strategic_layer,
            review_interval_ms=self.config["brain"].get("strategic_interval_ms", 5000)
        )
        
        # Operator
        safety_cfg = self.config["operator"].get("safety", {})
        self.operator = Operator(
            action_delay_ms=self.config["operator"].get("action_delay_ms", 30),
            safety=SafetyConfig(
                enabled=safety_cfg.get("enabled", True),
                kill_key=safety_cfg.get("kill_key", "f12"),
                max_actions_per_sec=safety_cfg.get("max_actions_per_sec", 50)
            )
        )
        
        self.operator_thread = OperatorThread(self.action_queue, self.operator)
        
        self._running = False
    
    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def setup_objectives(self, objectives: list[str], context: str = ""):
        """Set strategic objectives."""
        self.strategic_layer.set_objectives(objectives)
        if context:
            self.strategic_layer.set_context(context)
    
    def add_reaction_rule(self, rule: ReactionRule):
        """Add a permanent reaction rule."""
        self.reaction_layer.add_rule(rule)
    
    def push_goal(self, goal: Goal):
        """Push a tactical goal."""
        self.tactical_layer.push_goal(goal)
    
    def start(self):
        """Start all threads."""
        print("🎮 Player MK1 starting...")
        print(f"   Architecture: Async 3-layer (Reaction→Tactical→Strategic)")
        print(f"   Vision: {self.config['vision'].get('resize')} @ {self.config['vision'].get('fps', 60)} FPS")
        print(f"   Reaction: {self.config['brain'].get('reaction', {}).get('fps', 120)} Hz")
        print(f"   Tactical: {self.config['brain'].get('tactical', {}).get('fps', 20)} Hz")
        print(f"   Strategic: every {self.config['brain'].get('strategic_interval_ms', 5000)}ms")
        print(f"   Brain: {self.config['brain']['provider']}/{self.config['brain']['model']}")
        print(f"   Kill switch: {self.config['operator']['safety']['kill_key']}")
        print()
        
        self._running = True
        
        # Start all threads
        self.vision_thread.start()
        self.reaction_thread.start()
        self.tactical_thread.start()
        self.strategic_thread.start()
        self.operator_thread.start()
        
        print("✓ All threads running")
    
    def run(self, duration_sec: Optional[float] = None):
        """Run the player (blocking)."""
        self.start()
        
        start_time = time.time()
        last_status = time.time()
        
        try:
            while self._running:
                if duration_sec and (time.time() - start_time) >= duration_sec:
                    break
                
                # Print status every 5 seconds
                if time.time() - last_status >= 5:
                    self._print_status()
                    last_status = time.time()
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n👋 Stopping...")
        
        self.stop()
    
    def _print_status(self):
        active = self.tactical_layer.active_goal
        print(f"[Stats] "
              f"frames={self.frame_buffer.frame_count} "
              f"vision={self.vision_thread.actual_fps:.0f}fps "
              f"reaction={self.reaction_thread.checks_per_sec:.0f}hz "
              f"tactical={self.tactical_thread.ticks_per_sec:.0f}hz "
              f"strategic={self.strategic_thread.reviews} "
              f"actions={self.operator_thread.actions_executed} "
              f"goal={active.id if active else 'None'}")
    
    def stop(self):
        """Stop all threads."""
        self._running = False
        
        self.vision_thread.stop()
        self.reaction_thread.stop()
        self.tactical_thread.stop()
        self.strategic_thread.stop()
        self.operator_thread.stop()
        self.operator.stop()
        
        print("✓ All threads stopped")
        self._print_status()
    
    def force_strategic_review(self):
        """Trigger immediate strategic review."""
        self.strategic_thread.force_review()


def main():
    parser = argparse.ArgumentParser(description="Player MK1 - Async Game Player")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
    parser.add_argument("-t", "--time", type=float, help="Run duration (seconds)")
    parser.add_argument("--objectives", nargs="+", help="Strategic objectives")
    args = parser.parse_args()
    
    player = Player(config_path=args.config)
    
    if args.objectives:
        player.setup_objectives(args.objectives)
    
    player.run(duration_sec=args.time)


if __name__ == "__main__":
    main()
