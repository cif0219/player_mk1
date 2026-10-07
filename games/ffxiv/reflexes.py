"""FFXIV reflexes: the things that must happen faster than the rotation cares about.

Each reflex is a pure function of `WorldState`, which is what makes it testable in three
lines with no image and no game. That property is the entire reason perception produces a
typed state instead of letting rules read pixels.
"""

from __future__ import annotations

from player.act.timeline import Plan, Press, Priority, Step
from player.geometry import Geometry, Rect
from player.policy.condition import Condition
from player.policy.reflex import Reflex, ReflexLayer
from player.state import WorldState

from .layout import Layout


def _player_point(layout: Layout, state: WorldState) -> tuple[float, float]:
    """The player's approximate position in frame-normalised coordinates.

    Normalised rather than pixels so the overlap test works without a `Geometry` — which
    keeps the reflex a pure function of `WorldState` and therefore testable.
    """
    anchor = layout.player_anchor
    return anchor.x + anchor.w / 2, anchor.y + anchor.h / 2


def telegraph_overlaps_player(
    layout: Layout, state: WorldState, frame_size: tuple[int, int] | None = None
) -> bool:
    """Is the player standing in a detected ground telegraph?

    Entity boxes are in frame pixels and the anchor is normalised, so one of them has to
    be converted. The frame size comes from the largest entity's own bounds when not
    supplied, which is crude but sufficient — the test only needs relative position.
    """
    telegraphs = state.entities_of("telegraph")
    if not telegraphs:
        return False

    px, py = _player_point(layout, state)
    if frame_size is None:
        # Infer a plausible frame extent from the detections themselves.
        max_x = max(t.bbox.right for t in telegraphs)
        max_y = max(t.bbox.bottom for t in telegraphs)
        frame_size = (max(max_x, 1), max(max_y, 1))
    fw, fh = frame_size
    x, y = px * fw, py * fh

    return any(t.bbox.contains(int(x), int(y)) for t in telegraphs)


def dodge_plan(direction_key: str, hold_ms: int = 320) -> Plan:
    """Move out. A held movement key, not a dash.

    Deliberately simple: dashes have cooldowns and directional constraints, and getting
    them wrong puts you somewhere worse. Holding a movement key for ~300ms clears most
    ground AoEs and cannot make the situation worse.
    """
    return Plan(
        name="dodge",
        steps=(Step(0, Press(direction_key, hold_ms)),),
        priority=Priority.REFLEX,
        preempt=True,
        expires_in_ms=250,  # a late dodge moves you into the damage, not out of it
    )


def build_reflexes(
    layout: Layout,
    keys: dict[str, str],
    *,
    dodge_key: str = "s",
    potion_hp_threshold: float = 0.35,
) -> ReflexLayer:
    """The reflex set for the vertical slice.

    Two groups, so the director can toggle them independently:

    * `ground_aoe` — dodging telegraphs. Disabled when no detector backend is available,
      since the condition can never be true and an always-false reflex is just noise.
    * `survival` — emergency healing. Kept separate because it is useful on a dummy where
      dodging is not.
    """
    layer = ReflexLayer()

    layer.add(
        Reflex(
            id="dodge_ground_aoe",
            group="ground_aoe",
            condition=Condition.field("_telegraph_overlap", "==", True, on_missing=False),
            plan=lambda _state: dodge_plan(dodge_key),
            priority=Priority.REFLEX,
            # A telegraph stays on screen for seconds after you have already left it, so
            # without this the reflex re-fires the whole time and pins you moving.
            cooldown_ms=1200,
            preempt=True,
        )
    )

    if "potion" in keys:
        layer.add(
            Reflex(
                id="emergency_potion",
                group="survival",
                condition=Condition.all_(
                    Condition.truthy("player.in_combat"),
                    Condition.hp_below(potion_hp_threshold),
                ),
                plan=Plan.single(
                    keys["potion"],
                    name="potion",
                    priority=Priority.RECOVERY,
                    expires_in_ms=1500,
                ),
                priority=Priority.RECOVERY,
                cooldown_ms=30_000,
                # Recovery does not preempt: a potion is worth one GCD, not worth
                # throwing away a weave that is already in flight.
                preempt=False,
            )
        )

    return layer


class TelegraphOverlapSensor:
    """Writes `_telegraph_overlap` so the dodge reflex stays a pure condition.

    The alternative is a reflex whose condition does geometry against the entity list,
    which works but makes the reflex untestable without constructing entities and a
    frame size. Computing the predicate once, in perception, keeps the policy layer a
    pure function of named fields — which is the property the whole architecture is built
    around, so it is worth one extra sensor to preserve it.
    """

    provides = ("_telegraph_overlap",)
    cadence = 1

    def __init__(self, layout: Layout, name: str = "telegraph_overlap") -> None:
        self.layout = layout
        self.name = name

    def observe(self, frame, geo: Geometry, state: WorldState) -> None:
        overlap = telegraph_overlaps_player(self.layout, state, frame.size)
        confidence = 1.0 if state.entities_of("telegraph") else 0.9
        state.set(
            "_telegraph_overlap",
            overlap,
            confidence=confidence,
            source="derived:telegraph",
        )


__all__ = [
    "TelegraphOverlapSensor",
    "build_reflexes",
    "dodge_plan",
    "telegraph_overlaps_player",
]
