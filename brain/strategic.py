"""
Strategic Layer (戰略層)
Async high-level decision making using LLMs.
Runs in background at low frequency, sets goals for tactical layer.
Target: every 2-10 seconds
"""

import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, Any
from queue import Queue, Empty

from .tactical import TacticalLayer, Goal


@dataclass
class StrategicState:
    """Snapshot of game state for strategic analysis."""
    timestamp: float = field(default_factory=time.time)
    screenshot_b64: Optional[str] = None
    game_state: dict = field(default_factory=dict)
    active_goals: list[str] = field(default_factory=list)
    pending_goals: list[str] = field(default_factory=list)
    recent_events: list[str] = field(default_factory=list)


@dataclass 
class StrategicDecision:
    """Output from strategic layer."""
    goals: list[Goal] = field(default_factory=list)
    cancel_goals: list[str] = field(default_factory=list)
    analysis: str = ""
    next_review_ms: int = 5000


class StrategicLayer:
    """
    Strategic decision maker using LLM.
    Thread-safe, runs analysis in background.
    """
    
    def __init__(
        self,
        tactical_layer: TacticalLayer,
        provider: str = "openai",
        model: str = "gpt-4o",
        review_interval_ms: int = 5000
    ):
        self.tactical = tactical_layer
        self.provider = provider
        self.model = model
        self.review_interval_ms = review_interval_ms
        
        self._client = None
        self._objectives: list[str] = []
        self._context: str = ""
        self._history: list[dict] = []
        self._lock = threading.RLock()
    
    def _ensure_client(self):
        if self._client is not None:
            return
        
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic()
    
    def set_objectives(self, objectives: list[str]):
        with self._lock:
            self._objectives = list(objectives)
    
    def add_objective(self, objective: str):
        with self._lock:
            if objective not in self._objectives:
                self._objectives.append(objective)
    
    def remove_objective(self, objective: str):
        with self._lock:
            if objective in self._objectives:
                self._objectives.remove(objective)
    
    def set_context(self, context: str):
        with self._lock:
            self._context = context
    
    @property
    def objectives(self) -> list[str]:
        with self._lock:
            return list(self._objectives)
    
    def analyze(self, state: StrategicState) -> StrategicDecision:
        """Analyze state and return decision (blocking)."""
        self._ensure_client()
        
        prompt = self._build_prompt(state)
        
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt, state.screenshot_b64)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt, state.screenshot_b64)
            else:
                return StrategicDecision()
            
            return self._parse_response(response)
        except Exception as e:
            return StrategicDecision(analysis=f"Error: {e}")
    
    def apply_decision(self, decision: StrategicDecision):
        """Apply decision to tactical layer."""
        for goal_id in decision.cancel_goals:
            self.tactical.cancel_goal(goal_id)
        
        for goal in decision.goals:
            self.tactical.push_goal(goal)
        
        with self._lock:
            self._history.append({
                "timestamp": time.time(),
                "goals_added": [g.id for g in decision.goals],
                "goals_cancelled": decision.cancel_goals,
                "analysis": decision.analysis
            })
            if len(self._history) > 50:
                self._history = self._history[-50:]
    
    def _build_prompt(self, state: StrategicState) -> str:
        with self._lock:
            objectives = self._objectives.copy()
            context = self._context
        
        obj_str = "\n".join(f"- {o}" for o in objectives) or "None"
        active_str = "\n".join(f"- {g}" for g in state.active_goals) or "None"
        pending_str = "\n".join(f"- {g}" for g in state.pending_goals) or "None"
        events_str = "\n".join(f"- {e}" for e in state.recent_events[-10:]) or "None"
        state_str = json.dumps(state.game_state, indent=2) if state.game_state else "{}"
        
        return f"""You are a strategic game AI. Analyze the situation and set goals.

## Objectives
{obj_str}

## Game Context
{context or "Not specified."}

## Current State
{state_str}

## Active Goal
{active_str}

## Pending Goals
{pending_str}

## Recent Events
{events_str}

## Instructions
1. Analyze the screenshot and state
2. Decide what goals to add or cancel
3. Prioritize based on objectives

Respond with ONLY valid JSON:
{{
    "analysis": "Brief analysis",
    "goals": [
        {{"id": "goal_id", "type": "wait|sequence|navigate|combat|collect|interact", "params": {{}}, "priority": 50}}
    ],
    "cancel_goals": ["goal_id_to_cancel"],
    "next_review_ms": 5000
}}

Goal types:
- wait: params.duration_ms
- sequence: params.actions (list of action dicts)
- navigate: params.target (description)
- combat: params.strategy (aggressive|defensive|kite)
- collect: params.item (description)
- interact: params.target (description)
"""
    
    def _call_openai(self, prompt: str, image_b64: Optional[str]) -> str:
        content = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
        
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.3,
            messages=[{"role": "user", "content": content}]
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, image_b64: Optional[str]) -> str:
        content = []
        if image_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}
            })
        content.append({"type": "text", "text": prompt})
        
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": content}]
        )
        return response.content[0].text
    
    def _parse_response(self, response: str) -> StrategicDecision:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        try:
            data = json.loads(text.strip())
            goals = [
                Goal(
                    id=g.get("id", f"goal_{time.time()}"),
                    type=g.get("type", "wait"),
                    params=g.get("params", {}),
                    priority=g.get("priority", 50)
                )
                for g in data.get("goals", [])
            ]
            return StrategicDecision(
                goals=goals,
                cancel_goals=data.get("cancel_goals", []),
                analysis=data.get("analysis", ""),
                next_review_ms=data.get("next_review_ms", self.review_interval_ms)
            )
        except json.JSONDecodeError:
            return StrategicDecision(analysis=f"Parse error: {response[:200]}")


class StrategicThread(threading.Thread):
    """
    Async strategic processor.
    Periodically analyzes game state and updates goals.
    """
    
    def __init__(
        self,
        frame_buffer,  # SharedFrameBuffer
        layer: StrategicLayer,
        review_interval_ms: int = 5000
    ):
        super().__init__(daemon=True)
        self.frame_buffer = frame_buffer
        self.layer = layer
        self.review_interval_ms = review_interval_ms
        
        self._running = False
        self._force_review = threading.Event()
        
        # Stats
        self.reviews = 0
        self.last_analysis = ""
    
    def run(self):
        self._running = True
        
        while self._running:
            # Wait for interval or forced review
            self._force_review.wait(timeout=self.review_interval_ms / 1000)
            self._force_review.clear()
            
            if not self._running:
                break
            
            # Build state
            frame = self.frame_buffer.get_latest()
            state = StrategicState(
                screenshot_b64=frame.to_base64() if frame else None,
                active_goals=[self.layer.tactical.active_goal.id] 
                    if self.layer.tactical.active_goal else [],
                pending_goals=[g.id for g in self.layer.tactical.pending_goals]
            )
            
            # Analyze
            decision = self.layer.analyze(state)
            self.layer.apply_decision(decision)
            
            self.reviews += 1
            self.last_analysis = decision.analysis
            
            # Adjust interval based on decision
            self.review_interval_ms = decision.next_review_ms
    
    def force_review(self):
        """Trigger immediate review."""
        self._force_review.set()
    
    def stop(self):
        self._running = False
        self._force_review.set()  # Unblock wait
