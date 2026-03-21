import time
from typing import Optional
from .tactical import TacticalBehavior, TacticalContext, TaskStatus
from .reaction import ReactionRule, PixelTrigger

class CombatBehavior(TacticalBehavior):
    goal_type = "combat"

    def on_start(self, ctx: TacticalContext):
        # When combat starts, install an auto-attack reaction rule
        # e.g. check a center region for enemy colors and press space
        rule = ReactionRule(
            id=f"auto_attack_{ctx.goal.id}",
            trigger=PixelTrigger(
                region=(600, 300, 100, 100),  # Example region
                color=(255, 0, 0),            # Example red enemy color
                threshold=0.5
            ),
            action={"type": "key", "key": "space", "duration_ms": 50},
            priority=80,
            cooldown_ms=500
        )
        ctx.install_rule(rule)
        ctx.goal.params.setdefault("start_time", time.time())

    def tick(self, ctx: TacticalContext) -> Optional[dict]:
        duration = ctx.goal.params.get("duration_ms", 5000)
        start_time = ctx.goal.params["start_time"]
        elapsed = (time.time() - start_time) * 1000
        ctx.goal.progress = min(1.0, elapsed / duration)

        if elapsed >= duration:
            ctx.goal.status = TaskStatus.COMPLETED

        return None  # Actions are handled by the fast reaction rule

    def on_complete(self, ctx: TacticalContext):
        # Clean up the auto-attack rule
        ctx.remove_rule(f"auto_attack_{ctx.goal.id}")

    def on_cancel(self, ctx: TacticalContext):
        ctx.remove_rule(f"auto_attack_{ctx.goal.id}")

class NavigateBehavior(TacticalBehavior):
    goal_type = "navigate"

    def on_start(self, ctx: TacticalContext):
        ctx.goal.params.setdefault("start_time", time.time())

    def tick(self, ctx: TacticalContext) -> Optional[dict]:
        duration = ctx.goal.params.get("duration_ms", 3000)
        start_time = ctx.goal.params["start_time"]
        elapsed = (time.time() - start_time) * 1000
        ctx.goal.progress = min(1.0, elapsed / duration)

        if elapsed >= duration:
            ctx.goal.status = TaskStatus.COMPLETED
            return None

        # Example action: hold W key to move forward
        return {"type": "key", "key": "w", "hold": False, "duration_ms": 100}
