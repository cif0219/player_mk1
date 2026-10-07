"""`ApiSource` — a `CaptureSource` whose frames carry a gateway snapshot.

The runtime's capture thread calls `grab()`; the decide thread runs sensors
over whatever comes back. Here a "frame" is the newest gateway snapshot
wrapped in an `ApiFrame`, with a 2×2 placeholder image so every consumer
that expects pixels (the recorder's frame path, `Geometry`) keeps working
untouched. Nothing reads those pixels; `SnapshotSensor` reads `.snapshot`.

`open()` is the whole session bring-up — connect, log in, enter the zone,
run the setup commands — because that is where the screen path opens the
capture device, and the runtime treats a failing `open()` as "cannot start".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..capture.source import Frame
from ..clock import now
from ..geometry import Rect
from .gateway import GatewayClient, GatewayError, Snapshot

_PLACEHOLDER = np.zeros((2, 2, 3), dtype=np.uint8)
_RECT = Rect(0, 0, 2, 2)


@dataclass(frozen=True, slots=True)
class ApiFrame(Frame):
    """A `Frame` with the gateway snapshot attached.

    `captured_at` is when the snapshot arrived on this machine (monotonic), so
    staleness — the age of the *information* — is measured the same way as
    for a pixel frame.
    """

    snapshot: dict[str, Any] = None  # type: ignore[assignment]

    @property
    def wall_ms(self) -> float:
        return float(self.snapshot.get("t", 0.0)) if self.snapshot else 0.0


@dataclass(slots=True)
class SessionSpec:
    """What `ApiSource.open()` does before the first frame."""

    username: str = ""
    password: str = "agent-pass-123"
    character: str = ""
    register: bool = True
    zone: str = "overworld"
    room_id: str = ""
    # Chat lines sent once in-world (dev/playtest commands such as `/skill * 30`).
    setup_commands: tuple[str, ...] = ()
    # Wait this long for the first snapshot that satisfies `ready` before giving up.
    ready_timeout_s: float = 30.0
    # Items to buy (id, quantity) at a free counter before entering `zone`: the overworld is
    # entered first, `shop_teleport` sent (a playtest `/tp` beside the merchant), then bought.
    shopping: tuple[tuple[str, int], ...] = ()
    shop_teleport: str = ""
    # Summoned test helper: register with the gateway so orders reach this connection.
    helper_name: str = ""
    helper_requester: str = ""


class ApiSource:
    """Frames from a gateway instead of a screen."""

    def __init__(self, gateway: GatewayClient, session: SessionSpec | None = None, ready=None) -> None:
        self.gateway = gateway
        self.session = session or SessionSpec()
        # Predicate over a snapshot dict: the first frame is delivered only once it holds.
        self.ready = ready or (lambda snap: snap.get("self", {}).get("entityId", 0) > 0)
        self._last_seq = 0
        self._index = 0
        self.login: dict[str, Any] = {}
        self.entered: dict[str, Any] = {}
        self.exhausted = False
        self.bought: dict[str, int] = {}
        self.warnings: list[str] = []
        self._pending_room: tuple[str, str] | None = None  # (zone, roomId) a helper was told to follow into

    # -- CaptureSource ------------------------------------------------------------

    def open(self) -> None:
        self.gateway.connect()
        self.login = self.gateway.call(
            "login",
            username=self.session.username or None,
            password=self.session.password,
            register=self.session.register,
            character=self.session.character or None,
        )
        if self.session.helper_name:
            self.gateway.call("helper", name=self.session.helper_name, requester=self.session.helper_requester or None)
        if self.session.shopping:
            self._shop()
        enter_args: dict[str, Any] = {"zone": self.session.zone}
        if self.session.room_id:
            enter_args["roomId"] = self.session.room_id
        self.entered = self.gateway.call("enter", timeout_s=45.0, **enter_args)
        for line in self.session.setup_commands:
            self.gateway.fire("chat", message=line)
        self.gateway.wait_snapshot(self.ready, self.session.ready_timeout_s, "the first usable snapshot")

    def _shop(self) -> None:
        """Buy the list at a free counter in the overworld, then carry on to the zone.

        A counter refuses from further than eight metres, so the playtest teleport
        (`/tp x z`, FANCRAFT_PLAYTEST=1) lands us beside it first. A server without
        the teleport simply refuses the purchase; that is a warning, not a failure —
        the fight goes on without the bottle.
        """
        self.gateway.call("enter", timeout_s=45.0, zone="overworld")
        self.gateway.wait_snapshot(self.ready, self.session.ready_timeout_s, "the overworld")
        if self.session.shop_teleport:
            self.gateway.fire("chat", message=self.session.shop_teleport)
            time.sleep(0.6)  # the teleport lands on the next tick
        for item_id, qty in self.session.shopping:
            self.gateway.fire("shop_buy", itemId=item_id, quantity=qty)
            try:
                snap = self.gateway.wait_snapshot(
                    lambda s, item=item_id: any(row.get("itemId") == item for row in (s.get("inventory") or [])), 6.0, f"{item_id} in the bag")
                self.bought[item_id] = sum(int(r.get("quantity", 0)) for r in snap.get("inventory", []) if r.get("itemId") == item_id)
            except GatewayError:
                self.warnings.append(f"could not buy {item_id} (no free counter in reach — is the server running with FANCRAFT_PLAYTEST=1?)")

    # -- helper orders (delivered by the gateway as events) ----------------------

    def request_room(self, zone: str, room_id: str) -> None:
        """Follow the requester into another room; performed on the capture thread."""
        if zone or room_id:
            self._pending_room = (zone, room_id)

    def dismiss(self) -> None:
        self.exhausted = True

    def grab(self) -> ApiFrame | None:
        pending = self._pending_room
        if pending is not None:
            self._pending_room = None
            zone, room_id = pending
            try:
                args: dict[str, Any] = {"zone": zone or self.session.zone}
                if room_id:
                    args["roomId"] = room_id
                self.entered = self.gateway.call("enter", timeout_s=45.0, **args)
                for line in self.session.setup_commands:
                    self.gateway.fire("chat", message=line)
            except GatewayError as exc:
                self.warnings.append(f"could not follow into {zone} {room_id}: {exc}")
        if self.exhausted:
            return None
        snap: Snapshot | None = self.gateway.latest()
        if snap is None or snap.seq == self._last_seq:
            if not self.gateway.connected:
                self.exhausted = True
            return None
        self._last_seq = snap.seq
        self._index += 1
        return ApiFrame(
            image=_PLACEHOLDER,
            captured_at=snap.received_at,
            index=self._index,
            client_rect=_RECT,
            snapshot=snap.data,
        )

    def close(self) -> None:
        try:
            if self.gateway.connected:
                self.gateway.fire("move", forward=0, strafe=0, sprint=False)
                self.gateway.call("leave", timeout_s=5.0)
        except GatewayError:
            pass
        finally:
            self.gateway.close()

    # -- observability -----------------------------------------------------------

    def describe(self) -> str:
        who = self.login.get("character", "?")
        where = self.entered.get("zone", self.session.zone)
        bag = f", bought {self.bought}" if self.bought else ""
        warn = f", warnings: {'; '.join(self.warnings)}" if self.warnings else ""
        helper = f", helper {self.session.helper_name}" if self.session.helper_name else ""
        return f"api: {self.gateway.url} as {who} in {where} (frames={self._index}{bag}{helper}{warn})"


__all__ = ["ApiFrame", "ApiSource", "SessionSpec"]
