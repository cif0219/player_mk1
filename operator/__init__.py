"""
Player MK1 - Operator Module
Action execution layer
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController, Listener


@dataclass 
class SafetyConfig:
    enabled: bool = True
    kill_key: str = "f12"
    max_actions_per_sec: int = 10


class Operator:
    """Executes actions on the system (mouse, keyboard)."""
    
    def __init__(
        self,
        action_delay_ms: int = 100,
        safety: Optional[SafetyConfig] = None
    ):
        self.action_delay_ms = action_delay_ms
        self.safety = safety or SafetyConfig()
        
        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        
        self._stopped = False
        self._action_count = 0
        self._action_window_start = time.time()
        
        # Kill switch listener
        if self.safety.enabled:
            self._start_kill_switch()
    
    def _start_kill_switch(self):
        """Start listening for kill switch key."""
        def on_press(key):
            try:
                key_name = key.char if hasattr(key, 'char') else key.name
                if key_name == self.safety.kill_key:
                    self._stopped = True
                    print(f"\n🛑 KILL SWITCH ACTIVATED ({self.safety.kill_key})")
            except:
                pass
        
        listener = Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
    
    def _check_safety(self) -> bool:
        """Check if action is allowed."""
        if self._stopped:
            return False
        
        if not self.safety.enabled:
            return True
        
        # Rate limiting
        now = time.time()
        if now - self._action_window_start >= 1.0:
            self._action_window_start = now
            self._action_count = 0
        
        if self._action_count >= self.safety.max_actions_per_sec:
            return False
        
        self._action_count += 1
        return True
    
    def execute(self, action: dict) -> bool:
        """
        Execute an action.
        
        Args:
            action: Action dict with 'type' and params
        
        Returns:
            True if executed, False if blocked
        """
        if not self._check_safety():
            return False
        
        action_type = action.get("type", "wait")
        
        if action_type == "click":
            self._click(
                x=action.get("x"),
                y=action.get("y"),
                button=action.get("button", "left"),
                clicks=action.get("clicks", 1)
            )
        elif action_type == "key":
            self._key(
                key=action.get("key"),
                hold=action.get("hold", False),
                duration_ms=action.get("duration_ms", 100)
            )
        elif action_type == "move":
            self._move(x=action.get("x"), y=action.get("y"))
        elif action_type == "wait":
            time.sleep(action.get("ms", 100) / 1000)
        elif action_type == "sequence":
            for sub_action in action.get("actions", []):
                if not self.execute(sub_action):
                    return False
        else:
            print(f"Unknown action type: {action_type}")
            return False
        
        # Post-action delay
        time.sleep(self.action_delay_ms / 1000)
        return True
    
    def _click(self, x: Optional[int], y: Optional[int], button: str = "left", clicks: int = 1):
        """Click at position."""
        if x is not None and y is not None:
            self._mouse.position = (x, y)
        
        btn = Button.left if button == "left" else Button.right
        self._mouse.click(btn, clicks)
    
    def _key(self, key: str, hold: bool = False, duration_ms: int = 100):
        """Press or hold a key."""
        # Handle special keys
        key_map = {
            "space": Key.space,
            "enter": Key.enter,
            "tab": Key.tab,
            "escape": Key.esc,
            "esc": Key.esc,
            "shift": Key.shift,
            "ctrl": Key.ctrl,
            "alt": Key.alt,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "backspace": Key.backspace,
            "delete": Key.delete,
        }
        
        k = key_map.get(key.lower(), key)
        
        if hold:
            self._keyboard.press(k)
            time.sleep(duration_ms / 1000)
            self._keyboard.release(k)
        else:
            self._keyboard.press(k)
            self._keyboard.release(k)
    
    def _move(self, x: int, y: int):
        """Move mouse to position."""
        self._mouse.position = (x, y)
    
    def stop(self):
        """Emergency stop."""
        self._stopped = True
    
    def resume(self):
        """Resume after stop."""
        self._stopped = False
    
    @property
    def is_stopped(self) -> bool:
        return self._stopped


if __name__ == "__main__":
    op = Operator(action_delay_ms=200)
    print("Operator ready. Press F12 to test kill switch.")
    print("Moving mouse in 2 seconds...")
    time.sleep(2)
    op.execute({"type": "move", "x": 500, "y": 500})
    print("Done!")
