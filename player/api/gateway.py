"""A client for a game's agent gateway (FantCraft: `scripts/agent-gateway.mts`).

Newline-free JSON objects over one WebSocket. The gateway streams a
`snapshot` every server tick and `event` objects for discrete messages;
commands are `{op, ...}` with an optional `id` that is echoed on `ack`/`nack`.

Threading: one reader thread owns `recv`; `send` is called from the decide,
dispatch and CLI threads and is serialised by a lock. The newest snapshot is
kept in a single slot (like `FrameSlot`) — an old snapshot is a worse
snapshot, never a queue to catch up on. Events are appended to a bounded
list with a monotonic counter so readers can resume from where they left off.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..clock import now


class GatewayError(RuntimeError):
    pass


@dataclass(slots=True)
class Snapshot:
    """One gateway snapshot plus when this process received it."""

    data: dict[str, Any]
    received_at: float  # monotonic, `player.clock.now()`
    seq: int  # counts snapshots received; a frame is new when this grew


@dataclass(slots=True)
class Event:
    seq: int
    type: str
    t_wall_ms: float
    received_at: float
    data: dict[str, Any] = field(default_factory=dict)


class GatewayClient:
    """Connects to the gateway and keeps the latest state.

    `connect()` opens the socket and starts the reader; `call()` sends a
    command and waits for its reply; `fire()` sends without waiting (the
    movement stream, guards, strikes). `latest()` peeks the newest snapshot;
    `events_since()` returns discrete events after a cursor.
    """

    def __init__(self, url: str, connect_timeout_s: float = 10.0, max_events: int = 4096) -> None:
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self.max_events = max_events
        self._ws: Any = None
        self._reader: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = threading.Event()
        self._snapshot: Snapshot | None = None
        self._snapshots = 0
        self._events: list[Event] = []
        self._event_seq = 0
        self._pending: dict[int, tuple[threading.Event, list[dict[str, Any]]]] = {}
        self._next_id = 1
        self.hello: dict[str, Any] = {}
        self.listeners: list[Callable[[Event], None]] = []
        self.last_error: str = ""

    # -- lifecycle ---------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._closed.is_set()

    def connect(self) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise GatewayError("API control needs the `websockets` package: pip install websockets") from exc
        try:
            self._ws = connect(self.url, open_timeout=self.connect_timeout_s, max_size=8 * 1024 * 1024)
        except Exception as exc:
            raise GatewayError(f"cannot reach gateway at {self.url}: {exc}") from exc
        self._closed.clear()
        self._reader = threading.Thread(target=self._read_loop, name="gateway-reader", daemon=True)
        self._reader.start()

    def close(self) -> None:
        self._closed.set()
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=2.0)
        for done, _ in list(self._pending.values()):
            done.set()

    # -- sending -----------------------------------------------------------------

    def fire(self, op: str, **args: Any) -> None:
        """Send a command without waiting for a reply."""
        self._send({"op": op, **args})

    def call(self, op: str, timeout_s: float = 30.0, **args: Any) -> dict[str, Any]:
        """Send a command and wait for its ack. Raises `GatewayError` on nack or timeout."""
        with self._state_lock:
            msg_id = self._next_id
            self._next_id += 1
            done = threading.Event()
            box: list[dict[str, Any]] = []
            self._pending[msg_id] = (done, box)
        self._send({"op": op, "id": msg_id, **args})
        if not done.wait(timeout_s):
            self._pending.pop(msg_id, None)
            raise GatewayError(f"gateway did not answer {op} within {timeout_s:.0f}s")
        reply = box[0] if box else {}
        if reply.get("ev") != "ack":
            raise GatewayError(f"{op} refused: {reply.get('error', 'connection closed')}")
        return reply

    def _send(self, obj: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None or self._closed.is_set():
            raise GatewayError("gateway not connected")
        payload = json.dumps(obj, separators=(",", ":"))
        with self._send_lock:
            try:
                ws.send(payload)
            except Exception as exc:
                self.last_error = str(exc)
                self._closed.set()
                raise GatewayError(f"gateway send failed: {exc}") from exc

    # -- receiving ---------------------------------------------------------------

    def _read_loop(self) -> None:
        ws = self._ws
        while not self._closed.is_set() and ws is not None:
            try:
                raw = ws.recv()
            except Exception as exc:
                if not self._closed.is_set():
                    self.last_error = str(exc)
                break
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            self._dispatch(msg)
        self._closed.set()
        for done, _ in list(self._pending.values()):
            done.set()

    def _dispatch(self, msg: dict[str, Any]) -> None:
        ev = msg.get("ev")
        at = now()
        if ev == "snapshot":
            with self._state_lock:
                self._snapshots += 1
                self._snapshot = Snapshot(data=msg, received_at=at, seq=self._snapshots)
            return
        if ev == "event":
            with self._state_lock:
                self._event_seq += 1
                event = Event(
                    seq=self._event_seq,
                    type=str(msg.get("type", "")),
                    t_wall_ms=float(msg.get("t", 0.0)),
                    received_at=at,
                    data=msg.get("data") or {},
                )
                self._events.append(event)
                if len(self._events) > self.max_events:
                    del self._events[: len(self._events) - self.max_events]
            for listener in list(self.listeners):
                try:
                    listener(event)
                except Exception:
                    pass
            return
        if ev == "hello":
            self.hello = msg
            return
        if ev in ("ack", "nack"):
            msg_id = msg.get("id")
            if isinstance(msg_id, int):
                pending = self._pending.pop(msg_id, None)
                if pending is not None:
                    done, box = pending
                    box.append(msg)
                    done.set()

    # -- reads -------------------------------------------------------------------

    def latest(self) -> Snapshot | None:
        with self._state_lock:
            return self._snapshot

    def events_since(self, cursor: int) -> list[Event]:
        with self._state_lock:
            if not self._events or self._events[-1].seq <= cursor:
                return []
            # events are in seq order; find the first one past the cursor
            start = 0
            for i in range(len(self._events) - 1, -1, -1):
                if self._events[i].seq <= cursor:
                    start = i + 1
                    break
            return list(self._events[start:])

    @property
    def event_cursor(self) -> int:
        with self._state_lock:
            return self._event_seq

    def wait_snapshot(self, predicate: Callable[[dict[str, Any]], bool], timeout_s: float, label: str = "snapshot") -> dict[str, Any]:
        deadline = now() + timeout_s
        while now() < deadline and not self._closed.is_set():
            snap = self.latest()
            if snap is not None and predicate(snap.data):
                return snap.data
            self._closed.wait(0.02)
        raise GatewayError(f"timed out waiting for {label}")

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            return {"snapshots": self._snapshots, "events": self._event_seq, "connected": self.connected, "error": self.last_error}


__all__ = ["Event", "GatewayClient", "GatewayError", "Snapshot"]
