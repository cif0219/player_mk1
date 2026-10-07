"""The seam between the two concurrent loops.

There are two loops running at once and they contend for the same body:

* the **rotation loop**, pressing abilities to keep damage flowing, and
* the **mechanic loop**, moving, looking, and casting whatever the encounter demands.

The naive coupling is preemption: when a mechanic needs to act, it interrupts the
rotation. That works and it is wasteful, because it is purely reactive — the rotation
starts a three-second cast, the mechanic fires, the cast is cancelled, and the damage is
simply lost.

The better coupling is **anticipation**. The mechanic loop knows its own schedule several
seconds ahead — that is what a scripted encounter *is* — so instead of interrupting, it
publishes claims on future time. The rotation reads those claims and plans inside the gaps:
if movement starts in 1.8 seconds, it picks an instant now rather than a hard cast it would
have to throw away.

So the two loops are not master and slave. They share a board, and the rotation plans
around what the mechanic loop has already booked.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

from ..clock import now


class CommitmentKind(Enum):
    """What a claim on future time actually prevents."""

    # The character will be moving. Hard casts started now would be cancelled.
    MOVING = "moving"
    # A GCD is reserved for a mechanic ability; the rotation must not spend it.
    GCD_RESERVED = "gcd_reserved"
    # An off-GCD slot is reserved. Blocks weaving, not the GCD itself.
    OGCD_RESERVED = "ogcd_reserved"
    # The camera is pointed away from the boss. Targeting and cast-bar reads are
    # unreliable, so anything depending on them should wait.
    CAMERA_AWAY = "camera_away"
    # Deliberate downtime — a phase transition, an untargetable boss. Nothing to press.
    NO_TARGET = "no_target"
    # A scripted add phase: everything the rotation presses should go at this instead.
    # Deliberately not a GCD reservation — the rotation keeps choosing its own abilities,
    # it just points them somewhere else. Taking the GCD would throw away the rotation's
    # knowledge of what actually does damage.
    TARGET_OVERRIDE = "target_override"


@dataclass(frozen=True, slots=True)
class Commitment:
    """A claim on a window of future time."""

    kind: CommitmentKind
    start_at: float
    end_at: float
    reason: str = ""
    ability_id: str | None = None
    # Carried by TARGET_OVERRIDE. Typed loosely so `commitment` need not import
    # `targeting`, which would make the two modules mutually dependent.
    target: object | None = None
    source: str = "mechanic"

    def active_at(self, at: float) -> bool:
        return self.start_at <= at < self.end_at

    def starts_within(self, at: float, horizon_s: float) -> bool:
        return at <= self.start_at <= at + horizon_s

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_at - self.start_at)


class CommitmentBoard:
    """Shared, thread-safe view of what the mechanic loop has booked.

    Written by the mechanic loop, read by the rotation planner. Expired entries are swept
    on read rather than by a timer — the board is only interesting when someone is looking
    at it, and a sweep thread would be one more thing to shut down cleanly.
    """

    def __init__(self) -> None:
        self._items: list[Commitment] = []
        self._lock = threading.Lock()
        self.published = 0

    # -- writing -----------------------------------------------------------------

    def publish(self, commitment: Commitment) -> None:
        with self._lock:
            self._items.append(commitment)
            self.published += 1

    def replace(self, source: str, commitments: list[Commitment]) -> None:
        """Swap out everything from one source.

        The mechanic runner re-derives its whole schedule each tick, so replacing by
        source keeps the board consistent without anyone having to track individual
        entries — and a runner that dies mid-mechanic leaves nothing stale behind on its
        next tick.
        """
        with self._lock:
            self._items = [c for c in self._items if c.source != source]
            self._items.extend(commitments)
            self.published += len(commitments)

    def clear(self, source: str | None = None) -> None:
        with self._lock:
            self._items = [] if source is None else [c for c in self._items if c.source != source]

    # -- reading -----------------------------------------------------------------

    def active(self, at: float | None = None, kind: CommitmentKind | None = None) -> list[Commitment]:
        at = now() if at is None else at
        with self._lock:
            self._items = [c for c in self._items if c.end_at > at - 1.0]
            return [
                c for c in self._items if c.active_at(at) and (kind is None or c.kind is kind)
            ]

    def upcoming(
        self, at: float | None = None, horizon_s: float = 8.0, kind: CommitmentKind | None = None
    ) -> list[Commitment]:
        at = now() if at is None else at
        with self._lock:
            return sorted(
                (
                    c
                    for c in self._items
                    if c.start_at > at
                    and c.start_at <= at + horizon_s
                    and (kind is None or c.kind is kind)
                ),
                key=lambda c: c.start_at,
            )

    def is_active(self, kind: CommitmentKind, at: float | None = None) -> bool:
        return bool(self.active(at, kind))

    def reserved_ability(self, at: float | None = None) -> str | None:
        """The ability a mechanic has booked this GCD for, if any.

        When this returns something, the rotation must yield: the mechanic's ability is
        not optional and the GCD is not shared.
        """
        for commitment in self.active(at, CommitmentKind.GCD_RESERVED):
            if commitment.ability_id:
                return commitment.ability_id
        return None

    def target_override(self, at: float | None = None) -> object | None:
        """The target a scripted phase wants everything pointed at, if any.

        Add phases are the case: the mob has to die in eight seconds, so the rotation
        keeps making its own choices about what to press and simply aims them elsewhere.
        """
        for commitment in self.active(at, CommitmentKind.TARGET_OVERRIDE):
            if commitment.target is not None:
                return commitment.target
        return None

    def cast_window_s(self, at: float | None = None, horizon_s: float = 8.0) -> float:
        """How long a cast can safely be started right now.

        This is the number the rotation actually needs. If movement begins in 1.8 seconds,
        a 2.8-second cast is a cast that gets cancelled — so the planner asks how much
        uninterrupted time it has and only considers abilities that fit.

        Returns `inf` when nothing is booked, so an encounter-free session (a striking
        dummy) behaves exactly as it did before any of this existed.
        """
        at = now() if at is None else at
        blocking = (CommitmentKind.MOVING, CommitmentKind.NO_TARGET)

        for kind in blocking:
            if self.is_active(kind, at):
                return 0.0

        soonest = float("inf")
        for kind in blocking:
            for commitment in self.upcoming(at, horizon_s, kind):
                soonest = min(soonest, commitment.start_at - at)
        return max(0.0, soonest)

    def moving_until(self, at: float | None = None) -> float | None:
        at = now() if at is None else at
        ends = [c.end_at for c in self.active(at, CommitmentKind.MOVING)]
        return max(ends) if ends else None

    # -- reporting ---------------------------------------------------------------

    def snapshot(self, at: float | None = None) -> dict[str, object]:
        at = now() if at is None else at
        active = self.active(at)
        window = self.cast_window_s(at)
        return {
            "active": [f"{c.kind.value}({c.reason})" for c in active],
            "cast_window_s": None if window == float("inf") else round(window, 2),
            "reserved": self.reserved_ability(at),
            "pending": len(self.upcoming(at)),
        }

    def status(self, at: float | None = None) -> str:
        at = now() if at is None else at
        active = self.active(at)
        if not active:
            window = self.cast_window_s(at)
            return "commit: clear" if window == float("inf") else f"commit: {window:.1f}s to move"
        return "commit: " + ",".join(c.kind.value for c in active)
