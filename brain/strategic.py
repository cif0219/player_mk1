"""
Strategic Layer (戰略層)
Async high-level decision making using LLMs.
Includes interface for external commanders (user, AI assistant, etc.)
"""

import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from queue import Queue, Empty
from abc import ABC, abstractmethod

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


@dataclass
class CommanderDirective:
    """
    A directive from an external commander (user, AI assistant, etc.)
    Higher priority than LLM-generated decisions.
    """
    id: str
    type: str  # objective, goal, cancel, context, pause, resume, override
    payload: dict = field(default_factory=dict)
    priority: int = 100  # Higher than normal strategic decisions
    timestamp: float = field(default_factory=time.time)
    source: str = "external"  # external, user, assistant, api


class Commander(ABC):
    """
    Abstract interface for external commanders.
    Implement this to control the strategic layer from outside.
    """
    
    @abstractmethod
    def get_directives(self) -> list[CommanderDirective]:
        """Return pending directives (called each strategic cycle)."""
        pass
    
    def on_state_update(self, state: StrategicState):
        """Called when strategic layer has new state (optional override)."""
        pass
    
    def on_decision(self, decision: StrategicDecision):
        """Called when strategic layer makes a decision (optional override)."""
        pass


class QueueCommander(Commander):
    """
    Commander that accepts directives via a thread-safe queue.
    Use this for programmatic control.
    """
    
    def __init__(self):
        self._queue: Queue[CommanderDirective] = Queue()
        self._state_callbacks: list[Callable[[StrategicState], None]] = []
        self._decision_callbacks: list[Callable[[StrategicDecision], None]] = []
    
    def send(self, directive: CommanderDirective):
        """Send a directive to the strategic layer."""
        self._queue.put(directive)
    
    def set_objective(self, objective: str, priority: int = 100):
        """Convenience: add an objective."""
        self.send(CommanderDirective(
            id=f"obj_{time.time()}",
            type="objective",
            payload={"objective": objective, "action": "add"},
            priority=priority
        ))
    
    def remove_objective(self, objective: str):
        """Convenience: remove an objective."""
        self.send(CommanderDirective(
            id=f"obj_{time.time()}",
            type="objective",
            payload={"objective": objective, "action": "remove"}
        ))
    
    def push_goal(self, goal: Goal):
        """Convenience: push a goal directly."""
        self.send(CommanderDirective(
            id=f"goal_{time.time()}",
            type="goal",
            payload={"goal": goal}
        ))
    
    def cancel_goal(self, goal_id: str):
        """Convenience: cancel a goal."""
        self.send(CommanderDirective(
            id=f"cancel_{time.time()}",
            type="cancel",
            payload={"goal_id": goal_id}
        ))
    
    def set_context(self, context: str):
        """Convenience: update strategic context."""
        self.send(CommanderDirective(
            id=f"ctx_{time.time()}",
            type="context",
            payload={"context": context}
        ))
    
    def pause(self):
        """Pause strategic decision making."""
        self.send(CommanderDirective(
            id=f"pause_{time.time()}",
            type="pause",
            payload={}
        ))
    
    def resume(self):
        """Resume strategic decision making."""
        self.send(CommanderDirective(
            id=f"resume_{time.time()}",
            type="resume",
            payload={}
        ))
    
    def override_decision(self, decision: StrategicDecision):
        """Override with a complete decision."""
        self.send(CommanderDirective(
            id=f"override_{time.time()}",
            type="override",
            payload={"decision": decision}
        ))
    
    def get_directives(self) -> list[CommanderDirective]:
        directives = []
        while True:
            try:
                directives.append(self._queue.get_nowait())
            except Empty:
                break
        return directives
    
    def on_state_update(self, state: StrategicState):
        for cb in self._state_callbacks:
            try:
                cb(state)
            except:
                pass
    
    def on_decision(self, decision: StrategicDecision):
        for cb in self._decision_callbacks:
            try:
                cb(decision)
            except:
                pass
    
    def add_state_callback(self, callback: Callable[[StrategicState], None]):
        self._state_callbacks.append(callback)
    
    def add_decision_callback(self, callback: Callable[[StrategicDecision], None]):
        self._decision_callbacks.append(callback)


class WebSocketCommander(Commander):
    """
    Commander that accepts directives via WebSocket.
    For remote control by user or AI assistant.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._queue: Queue[CommanderDirective] = Queue()
        self._server = None
        self._running = False
        self._state: Optional[StrategicState] = None
        self._clients: set = set()
    
    def start(self):
        """Start WebSocket server in background thread."""
        import asyncio
        import websockets
        import threading
        
        async def handler(websocket, path):
            self._clients.add(websocket)
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        directive = CommanderDirective(
                            id=data.get("id", f"ws_{time.time()}"),
                            type=data.get("type", "objective"),
                            payload=data.get("payload", {}),
                            priority=data.get("priority", 100),
                            source="websocket"
                        )
                        self._queue.put(directive)
                        await websocket.send(json.dumps({"status": "ok", "id": directive.id}))
                    except Exception as e:
                        await websocket.send(json.dumps({"status": "error", "error": str(e)}))
            finally:
                self._clients.discard(websocket)
        
        async def serve():
            async with websockets.serve(handler, self.host, self.port):
                while self._running:
                    await asyncio.sleep(0.1)
        
        def run_server():
            self._running = True
            asyncio.run(serve())
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
    
    def stop(self):
        self._running = False
    
    def get_directives(self) -> list[CommanderDirective]:
        directives = []
        while True:
            try:
                directives.append(self._queue.get_nowait())
            except Empty:
                break
        return directives
    
    def on_state_update(self, state: StrategicState):
        self._state = state
        # Broadcast to all connected clients
        if self._clients:
            import asyncio
            msg = json.dumps({
                "type": "state",
                "timestamp": state.timestamp,
                "active_goals": state.active_goals,
                "pending_goals": state.pending_goals,
                "game_state": state.game_state
            })
            # Note: proper async broadcast would need more work
    
    def on_decision(self, decision: StrategicDecision):
        if self._clients:
            msg = json.dumps({
                "type": "decision",
                "analysis": decision.analysis,
                "goals": [g.id for g in decision.goals],
                "cancel_goals": decision.cancel_goals
            })


class StrategicLayer:
    """
    Strategic decision maker with external commander interface.
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
        
        # Commander interface
        self._commanders: list[Commander] = []
        self._paused = False
    
    def add_commander(self, commander: Commander):
        """Register an external commander."""
        self._commanders.append(commander)
    
    def remove_commander(self, commander: Commander):
        """Unregister a commander."""
        if commander in self._commanders:
            self._commanders.remove(commander)
    
    def _process_directives(self):
        """Process all pending directives from commanders."""
        for commander in self._commanders:
            directives = commander.get_directives()
            for d in sorted(directives, key=lambda x: -x.priority):
                self._apply_directive(d)
    
    def _apply_directive(self, directive: CommanderDirective):
        """Apply a single directive."""
        t = directive.type
        p = directive.payload
        
        if t == "objective":
            if p.get("action") == "add":
                self.add_objective(p.get("objective", ""))
            elif p.get("action") == "remove":
                self.remove_objective(p.get("objective", ""))
            elif p.get("action") == "set":
                self.set_objectives(p.get("objectives", []))
        
        elif t == "goal":
            goal = p.get("goal")
            if isinstance(goal, Goal):
                self.tactical.push_goal(goal)
            elif isinstance(goal, dict):
                self.tactical.push_goal(Goal(
                    id=goal.get("id", f"cmd_{time.time()}"),
                    type=goal.get("type", "wait"),
                    params=goal.get("params", {}),
                    priority=goal.get("priority", 50)
                ))
        
        elif t == "cancel":
            self.tactical.cancel_goal(p.get("goal_id", ""))
        
        elif t == "context":
            self.set_context(p.get("context", ""))
        
        elif t == "pause":
            self._paused = True
        
        elif t == "resume":
            self._paused = False
        
        elif t == "override":
            decision = p.get("decision")
            if isinstance(decision, StrategicDecision):
                self.apply_decision(decision)
    
    def _notify_state(self, state: StrategicState):
        """Notify all commanders of state update."""
        for commander in self._commanders:
            try:
                commander.on_state_update(state)
            except:
                pass
    
    def _notify_decision(self, decision: StrategicDecision):
        """Notify all commanders of decision."""
        for commander in self._commanders:
            try:
                commander.on_decision(decision)
            except:
                pass
    
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
            if objective and objective not in self._objectives:
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
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    def analyze(self, state: StrategicState) -> StrategicDecision:
        """Analyze state and return decision (blocking)."""
        # Process directives first
        self._process_directives()
        
        # Notify commanders
        self._notify_state(state)
        
        # Skip LLM if paused
        if self._paused:
            return StrategicDecision(analysis="Paused by commander")
        
        self._ensure_client()
        prompt = self._build_prompt(state)
        
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt, state.screenshot_b64)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt, state.screenshot_b64)
            else:
                return StrategicDecision()
            
            decision = self._parse_response(response)
            self._notify_decision(decision)
            return decision
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

## Objectives (from commander)
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
3. Prioritize based on commander's objectives

Respond with ONLY valid JSON:
{{
    "analysis": "Brief analysis",
    "goals": [
        {{"id": "goal_id", "type": "wait|sequence|navigate|combat|collect|interact", "params": {{}}, "priority": 50}}
    ],
    "cancel_goals": ["goal_id_to_cancel"],
    "next_review_ms": 5000
}}
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
    Async strategic processor with commander support.
    """
    
    def __init__(
        self,
        frame_buffer,
        layer: StrategicLayer,
        review_interval_ms: int = 5000
    ):
        super().__init__(daemon=True)
        self.frame_buffer = frame_buffer
        self.layer = layer
        self.review_interval_ms = review_interval_ms
        
        self._running = False
        self._force_review = threading.Event()
        
        self.reviews = 0
        self.last_analysis = ""
    
    def run(self):
        self._running = True
        
        while self._running:
            self._force_review.wait(timeout=self.review_interval_ms / 1000)
            self._force_review.clear()
            
            if not self._running:
                break
            
            # Always process directives, even when paused
            self.layer._process_directives()
            
            if self.layer.is_paused:
                continue
            
            frame = self.frame_buffer.get_latest()
            state = StrategicState(
                screenshot_b64=frame.to_base64() if frame else None,
                active_goals=[self.layer.tactical.active_goal.id] 
                    if self.layer.tactical.active_goal else [],
                pending_goals=[g.id for g in self.layer.tactical.pending_goals]
            )
            
            decision = self.layer.analyze(state)
            self.layer.apply_decision(decision)
            
            self.reviews += 1
            self.last_analysis = decision.analysis
            self.review_interval_ms = decision.next_review_ms
    
    def force_review(self):
        self._force_review.set()
    
    def stop(self):
        self._running = False
        self._force_review.set()
