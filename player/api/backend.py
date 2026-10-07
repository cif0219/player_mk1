"""`ApiBackend` — an `InputBackend` whose keys are intents.

The timeline, the dispatcher and the guard chain all speak in `KeyDown` /
`KeyUp` of named keys, and that vocabulary is worth keeping: a parry is
"hold mouse2 from t−100 ms for 260 ms" whether the hold reaches the OS or a
gateway. So this backend keeps the shape and reinterprets the names:

* movement keys are *held* — the set of keys currently down is folded into
  one movement vector and streamed by the gateway every tick; holds are
  reference-counted, so two overlapping plans pressing the same key keep it
  down until the last of them lets go (a re-fired dodge must not be cut short
  by the release of the one it replaced);
* `mouse1` strikes on key-down, `mouse2` raises the guard on key-down and
  lowers it on key-up;
* `flee:<entityId>` / `chase:<entityId>` are held like movement keys but ask the
  gateway for a world-frame vector re-aimed at that entity every tick, so a
  retreat stays a retreat while our facing flips to parry something else;
* anything else is looked up in `KeyActions`, which a game profile fills
  with its own intents (`face:boss`, `skill:provoke@horse`, `cover`).

A game supplies `KeyActions`; this module knows nothing about any game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..clock import now
from ..geometry import ScreenPoint
from .gateway import GatewayClient, GatewayError

# Action for a named key: called with `down=True` on press and `down=False` on release.
KeyAction = Callable[[bool], None]


@dataclass(slots=True)
class KeyActions:
    """The game's intent vocabulary, keyed by the name a plan presses.

    `resolve` handles names that carry parameters (`skill:<id>@<role>`); a
    plain dict lookup runs first so the common case costs nothing.
    """

    fixed: dict[str, KeyAction] = field(default_factory=dict)
    resolve: Callable[[str], KeyAction | None] | None = None

    def lookup(self, key: str) -> KeyAction | None:
        action = self.fixed.get(key)
        if action is not None:
            return action
        if self.resolve is not None:
            return self.resolve(key)
        return None


@dataclass(slots=True)
class SentIntent:
    at: float
    kind: str
    detail: str


class ApiBackend:
    """Sends intents through a `GatewayClient`."""

    name = "api"

    MOVE_KEYS = {"w": ("forward", 1.0), "s": ("forward", -1.0), "d": ("strafe", 1.0), "a": ("strafe", -1.0)}

    def __init__(self, gateway: GatewayClient, actions: KeyActions | None = None, log_limit: int = 512) -> None:
        self.gateway = gateway
        self.actions = actions or KeyActions()
        self.held: set[str] = set()
        self._holds: dict[str, int] = {}
        self.events: list[SentIntent] = []
        self.log_limit = log_limit
        self.errors = 0
        self.unknown_keys: dict[str, int] = {}
        self._last_vector: tuple[float, float, bool] | None = None
        # Overlapping parries: two blows landing close together each schedule their own
        # hold. The guard comes down only when the last hold releases, and every press
        # re-raises it so the newest parry window is the one the engine dates from.
        self._guard_holds = 0
        self._relative: tuple[str, int] | None = None  # ("awayFrom"|"toward", entity id)

    # -- InputBackend --------------------------------------------------------------

    def key_down(self, key: str) -> None:
        key = key.lower()
        if key in self.MOVE_KEYS or key == "shift":
            self._holds[key] = self._holds.get(key, 0) + 1
            self.held.add(key)
            self._push_movement()
            return
        if key.startswith(("flee:", "chase:")):
            mode, _, ident = key.partition(":")
            if ident.isdigit():
                self._relative = ("awayFrom" if mode == "flee" else "toward", int(ident))
                self._push_movement(force=True)
            return
        if key == "space":
            self._fire("move", jump=True, **self._vector_args())
            self._record("jump", "")
            return
        if key in ("mouse1", "lmb"):
            self._fire("attack")
            self._record("attack", "")
            return
        if key in ("mouse2", "rmb"):
            # A fresh raise, always: the engine dates the parry window from the moment the
            # guard went up, and a guard that was already up (or an attack still pending,
            # which counts as one) would date it from the wrong moment.
            self._guard_holds += 1
            self._fire("guard", active=False)
            self._fire("guard", active=True)
            self._record("guard", "up")
            return
        action = self.actions.lookup(key)
        if action is None:
            self.unknown_keys[key] = self.unknown_keys.get(key, 0) + 1
            return
        try:
            action(True)
            self._record("intent", key)
        except GatewayError:
            self.errors += 1

    def key_up(self, key: str) -> None:
        key = key.lower()
        if key in self.MOVE_KEYS or key == "shift":
            left = max(0, self._holds.get(key, 0) - 1)
            self._holds[key] = left
            if left == 0:
                self.held.discard(key)
                self._push_movement()
            return
        if key.startswith(("flee:", "chase:")):
            self._relative = None
            self._push_movement(force=True)
            return
        if key in ("space", "mouse1", "lmb"):
            return
        if key in ("mouse2", "rmb"):
            self._guard_holds = max(0, self._guard_holds - 1)
            if self._guard_holds == 0:
                self._fire("guard", active=False)
                self._record("guard", "down")
            return
        action = self.actions.lookup(key)
        if action is not None:
            try:
                action(False)
            except GatewayError:
                self.errors += 1

    def mouse_move(self, point: ScreenPoint) -> None:
        return None  # aiming is `face`, not a cursor

    def click(self, button: str, point: ScreenPoint | None) -> None:
        self.key_down("mouse1" if button in ("left", "mouse1") else "mouse2")
        self.key_up("mouse1" if button in ("left", "mouse1") else "mouse2")

    def mouse_button(self, button: str, down: bool) -> None:
        key = "mouse1" if button in ("left", "mouse1") else "mouse2"
        self.key_down(key) if down else self.key_up(key)

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        return None

    def release_all(self) -> None:
        """Lift every hold. Called on guard trips and shutdown; must never raise."""
        self.held.clear()
        self._holds.clear()
        self._guard_holds = 0
        self._relative = None
        try:
            self._fire("move", forward=0, strafe=0, sprint=False)
            self._fire("guard", active=False)
        except GatewayError:
            pass
        self._last_vector = None

    def close(self) -> None:
        self.release_all()

    # -- movement ------------------------------------------------------------------

    def _vector_args(self) -> dict[str, Any]:
        forward = 0.0
        strafe = 0.0
        for key in self.held:
            spec = self.MOVE_KEYS.get(key)
            if spec is None:
                continue
            axis, sign = spec
            if axis == "forward":
                forward += sign
            else:
                strafe += sign
        forward = max(-1.0, min(1.0, forward))
        strafe = max(-1.0, min(1.0, strafe))
        args: dict[str, Any] = {"forward": forward, "strafe": strafe, "sprint": "shift" in self.held}
        if self._relative is not None:
            args[self._relative[0]] = self._relative[1]
        return args

    def _push_movement(self, force: bool = False) -> None:
        args = self._vector_args()
        vector = (args["forward"], args["strafe"], args["sprint"])
        if vector == self._last_vector and not force:
            return
        self._last_vector = vector
        self._fire("move", **args)
        rel = f" {self._relative[0]}={self._relative[1]}" if self._relative else ""
        self._record("move", f"{args['forward']:+.0f},{args['strafe']:+.0f}{'!' if args['sprint'] else ''}{rel}")

    # -- plumbing ------------------------------------------------------------------

    def _fire(self, op: str, **args: Any) -> None:
        try:
            self.gateway.fire(op, **args)
        except GatewayError:
            self.errors += 1

    def _record(self, kind: str, detail: str) -> None:
        self.events.append(SentIntent(at=now(), kind=kind, detail=detail))
        if len(self.events) > self.log_limit:
            self.events.pop(0)

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for ev in self.events:
            counts[ev.kind] = counts.get(ev.kind, 0) + 1
        return {"sent": counts, "errors": self.errors, "unknown_keys": dict(self.unknown_keys), "held": sorted(self.held)}


__all__ = ["ApiBackend", "KeyAction", "KeyActions", "SentIntent"]
