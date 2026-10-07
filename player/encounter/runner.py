"""The mechanic loop.

    IDLE ──trigger──▶ OBSERVE ──camera settled──▶ RESOLVE ──target──▶ POSITION
                                                                          │
                        IDLE ◀──uptime──── RECOVER ◀──ability──── ACT ◀───┘

Five phases, because a mechanic is five distinct problems and conflating them hides which
one failed. "The player died to the tower" is useless; "the camera never framed the tower,
so RESOLVE timed out" is a bug report.

The phases map to real costs:

* **OBSERVE** — point the camera at the information. In a 3D game you cannot resolve what
  you have not looked at, and looking takes time you have to budget for.
* **RESOLVE** — bind the mechanic's random parameters from what is now visible.
* **POSITION** — walk there, closed-loop.
* **ACT** — press what the mechanic requires (anti-knockback, mitigation, a soak).
* **RECOVER** — return to an uptime position so the rotation stops being penalised.

Running alongside all of this is the rotation loop. The two coordinate through
`CommitmentBoard`, not through preemption: this loop publishes what it will be doing and
when, and the rotation plans inside the gaps. That is why the runner computes its whole
schedule up front — the schedule *is* the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..act.timeline import Plan, Priority
from ..clock import now
from ..navigate.gaze import GazePriority, GazeRequest, GazeTarget
from ..navigate.movement import MovementIntent
from ..policy.commitment import Commitment, CommitmentKind
from ..state import WorldState
from ..world.arena import ArenaFrame, ArenaPoint
from .script import EncounterScript, LookAt, Mechanic


class Phase(Enum):
    IDLE = "idle"
    OBSERVE = "observe"
    RESOLVE = "resolve"
    POSITION = "position"
    ACT = "act"
    RECOVER = "recover"
    FAILED = "failed"


@dataclass(slots=True)
class MechanicIntent:
    """What the mechanic loop wants this tick.

    Returned rather than applied. The runner stays a pure function of state, so the whole
    five-phase machine is testable with no camera, no character, and no clock.
    """

    phase: Phase = Phase.IDLE
    mechanic_id: str = ""
    gaze: GazeRequest | None = None
    movement: MovementIntent | None = None
    plan: Plan | None = None
    commitments: list[Commitment] = field(default_factory=list)
    note: str = ""

    @property
    def idle(self) -> bool:
        return self.phase is Phase.IDLE


@dataclass(slots=True)
class _Active:
    """Bookkeeping for the mechanic currently being resolved."""

    mechanic: Mechanic
    started_at: float
    deadline_at: float
    phase: Phase = Phase.OBSERVE
    phase_started_at: float = 0.0
    destination: ArenaPoint | None = None
    look_point: ArenaPoint | None = None
    ability_fired: bool = False
    settle_until: float = 0.0

    def time_left(self, at: float) -> float:
        return self.deadline_at - at


class MechanicRunner:
    """Drives one mechanic at a time through the five phases.

    One at a time deliberately. Overlapping mechanics exist in real fights, but resolving
    two simultaneously means two movement targets and one character — and the failure mode
    of guessing is walking between them and being hit by both. Serialising and reporting
    the overlap is the honest behaviour.
    """

    def __init__(
        self,
        script: EncounterScript,
        *,
        recover_to: str = "",
        observe_budget_frac: float = 0.25,
        act_budget_ms: int = 600,
    ) -> None:
        self.script = script
        # Where to stand between mechanics — usually melee range or a caster's uptime
        # spot. Empty means "stay put", which is right for a striking dummy.
        self.recover_to = recover_to
        self.observe_budget_frac = observe_budget_frac
        self.act_budget_ms = act_budget_ms

        self.combat_started_at: float | None = None
        self.active: _Active | None = None
        self.completed: list[str] = []
        self.failures: list[tuple[str, str]] = []
        self._recent: dict[str, float] = {}

    # -- clock -------------------------------------------------------------------

    def note_combat_start(self, at: float) -> None:
        self.combat_started_at = at
        self._recent.clear()

    def resync(self, at: float, to_elapsed_s: float) -> None:
        """Re-anchor the encounter clock on a named cast.

        Without this a timeline-based script drifts and every later mechanic fires at the
        wrong moment. Every named cast is a resync point, so drift never accumulates past
        one mechanic.
        """
        self.combat_started_at = at - to_elapsed_s

    def elapsed_s(self, at: float) -> float:
        return 0.0 if self.combat_started_at is None else at - self.combat_started_at

    # -- the loop ----------------------------------------------------------------

    def tick(
        self,
        state: WorldState,
        frame: ArenaFrame | None,
        at: float | None = None,
        *,
        boss: ArenaPoint | None = None,
        telegraphs: list[ArenaPoint] | None = None,
        arrived: bool = False,
    ) -> MechanicIntent:
        at = now() if at is None else at

        if frame is None or not frame.pose.known:
            # Unlocalised. Abandon rather than resolve on a guessed position — a mechanic
            # resolved against the wrong coordinates walks into the thing it was dodging.
            if self.active is not None:
                self._fail(self.active, "lost localisation", at)
            return MechanicIntent(note="unlocalised")

        if self.active is None:
            return self._maybe_start(state, at)

        active = self.active
        if at > active.deadline_at and active.phase not in (Phase.RECOVER, Phase.IDLE):
            self._fail(active, f"deadline missed in {active.phase.value}", at)
            return MechanicIntent(phase=Phase.FAILED, mechanic_id=active.mechanic.id)

        if active.phase is Phase.OBSERVE:
            return self._observe(active, state, frame, at, boss, telegraphs or [])
        if active.phase is Phase.RESOLVE:
            return self._resolve(active, state, frame, at, boss, telegraphs or [])
        if active.phase is Phase.POSITION:
            return self._position(active, frame, at, arrived)
        if active.phase is Phase.ACT:
            return self._act(active, at)
        if active.phase is Phase.RECOVER:
            return self._recover(active, frame, at, arrived)
        return MechanicIntent()

    # -- phases ------------------------------------------------------------------

    def _maybe_start(self, state: WorldState, at: float) -> MechanicIntent:
        mechanic = self.script.matching(state, self.elapsed_s(at), at)
        if mechanic is None:
            return MechanicIntent()

        # A cast trigger stays true for the whole cast, so without this the mechanic
        # restarts every frame and never gets past OBSERVE.
        last = self._recent.get(mechanic.id)
        if last is not None and (at - last) < (mechanic.deadline_ms / 1000.0) * 2:
            return MechanicIntent()
        self._recent[mechanic.id] = at

        self.active = _Active(
            mechanic=mechanic,
            started_at=at,
            deadline_at=at + mechanic.deadline_ms / 1000.0,
            phase=Phase.OBSERVE,
            phase_started_at=at,
        )
        return MechanicIntent(
            phase=Phase.OBSERVE,
            mechanic_id=mechanic.id,
            note=f"triggered by {mechanic.trigger.describe()}",
            commitments=self._schedule(self.active, at),
        )

    def _observe(
        self,
        active: _Active,
        state: WorldState,
        frame: ArenaFrame,
        at: float,
        boss: ArenaPoint | None,
        telegraphs: list[ArenaPoint],
    ) -> MechanicIntent:
        """Point the camera at whatever tells us how this mechanic resolves."""
        look = active.mechanic.look

        # Phase transitions chain within a single tick rather than costing a frame each.
        # Three transitions at 60Hz is 50ms, and a mechanic with a one-second window
        # cannot afford to spend it changing its own mind about which phase it is in.
        if look.at is LookAt.KEEP:
            self._advance(active, Phase.RESOLVE, at)
            return self._resolve(active, state, frame, at, boss, telegraphs)

        point = look.resolve(frame, boss)
        if point is None:
            # Nothing to aim at — the boss is not localised, or the waymark is missing.
            # Resolve anyway; some resolutions do not need the view.
            self._advance(active, Phase.RESOLVE, at)
            intent = self._resolve(active, state, frame, at, boss, telegraphs)
            intent.note = f"look target unavailable; {intent.note}"
            return intent

        active.look_point = point
        budget_s = (active.mechanic.deadline_ms / 1000.0) * self.observe_budget_frac
        elapsed = at - active.phase_started_at

        in_view = frame.observability.covers(point)
        if in_view and active.settle_until == 0.0:
            active.settle_until = at + look.settle_s

        # Hold still once framed. A detector fed frames captured mid-slew does badly, and
        # this is exactly the moment its answer is load-bearing.
        settled = active.settle_until > 0.0 and at >= active.settle_until
        if settled or elapsed > budget_s:
            self._advance(active, Phase.RESOLVE, at)
            intent = self._resolve(active, state, frame, at, boss, telegraphs)
            intent.note = ("framed; " if settled else "observe budget spent; ") + intent.note
            return intent

        return MechanicIntent(
            phase=Phase.OBSERVE,
            mechanic_id=active.mechanic.id,
            gaze=GazeRequest(
                id="mechanic",
                target=GazeTarget(point=point),
                priority=GazePriority.MECHANIC,
                tolerance_deg=look.tolerance_deg,
                expires_at=active.deadline_at,
                reason=f"see {active.mechanic.id}",
                dwell_s=look.settle_s,
            ),
            commitments=self._schedule(active, at),
        )

    def _resolve(
        self,
        active: _Active,
        state: WorldState,
        frame: ArenaFrame,
        at: float,
        boss: ArenaPoint | None,
        telegraphs: list[ArenaPoint],
    ) -> MechanicIntent:
        """Bind the mechanic's random parameters into a destination."""
        destination = active.mechanic.resolve.destination(frame, state, boss, telegraphs)
        active.destination = destination

        if destination is None:
            # No safe spot, or a resolution that needs nothing. Distinguishing them here
            # would be false precision; either way there is nowhere to go.
            if active.mechanic.ability_id:
                self._advance(active, Phase.ACT, at)
                return self._act(active, at)
            self._advance(active, Phase.RECOVER, at)
            return MechanicIntent(
                phase=Phase.RECOVER,
                mechanic_id=active.mechanic.id,
                note="no destination resolved",
                commitments=self._schedule(active, at),
            )

        self._advance(active, Phase.POSITION, at)
        intent = self._position(active, frame, at, arrived=False)
        intent.note = f"{active.mechanic.resolve.describe()}; {intent.note}"
        return intent

    def _position(
        self, active: _Active, frame: ArenaFrame, at: float, arrived: bool
    ) -> MechanicIntent:
        """Walk there, and publish that we are moving so the rotation stops hard-casting."""
        destination = active.destination
        if destination is None:
            self._advance(active, Phase.RECOVER, at)
            return MechanicIntent(phase=Phase.RECOVER, mechanic_id=active.mechanic.id)

        distance = frame.player.distance_to(destination)
        act_lead_s = active.mechanic.ability_lead_ms / 1000.0
        close_enough = arrived or distance <= 1.0

        if close_enough:
            self._advance(active, Phase.ACT if active.mechanic.ability_id else Phase.RECOVER, at)
            return MechanicIntent(
                phase=active.phase,
                mechanic_id=active.mechanic.id,
                note=f"in position ({distance:.1f}m)",
                commitments=self._schedule(active, at),
            )

        return MechanicIntent(
            phase=Phase.POSITION,
            mechanic_id=active.mechanic.id,
            movement=MovementIntent(
                target=destination,
                tolerance_m=1.0,
                deadline_at=active.deadline_at - act_lead_s,
                reason=active.mechanic.id,
            ),
            commitments=self._schedule(active, at),
            note=f"{distance:.1f}m to go",
        )

    def _act(self, active: _Active, at: float) -> MechanicIntent:
        """Press what the mechanic requires.

        The GCD was already reserved on the board when the mechanic started, so the
        rotation has been planning around this moment for seconds rather than being
        interrupted by it.
        """
        ability_id = active.mechanic.ability_id
        if not ability_id or active.ability_fired:
            self._advance(active, Phase.RECOVER, at)
            return MechanicIntent(phase=Phase.RECOVER, mechanic_id=active.mechanic.id)

        active.ability_fired = True
        self._advance(active, Phase.RECOVER, at)
        return MechanicIntent(
            phase=Phase.RECOVER,
            mechanic_id=active.mechanic.id,
            plan=Plan.single(
                ability_id,
                name=f"mechanic:{active.mechanic.id}:{ability_id}",
                priority=Priority.RECOVERY,
                expires_in_ms=max(200, int(active.time_left(at) * 1000)),
            ),
            note=f"cast {ability_id}",
            commitments=self._schedule(active, at),
        )

    def _recover(
        self, active: _Active, frame: ArenaFrame, at: float, arrived: bool
    ) -> MechanicIntent:
        """Return to an uptime position and hand the camera back to the boss."""
        target = frame.waymark(self.recover_to) if self.recover_to else None

        if target is None or arrived or frame.player.distance_to(target) <= 1.5:
            self.completed.append(active.mechanic.id)
            self.active = None
            return MechanicIntent(
                phase=Phase.IDLE,
                mechanic_id=active.mechanic.id,
                note="resolved",
            )

        return MechanicIntent(
            phase=Phase.RECOVER,
            mechanic_id=active.mechanic.id,
            movement=MovementIntent(
                target=target, tolerance_m=1.5, reason=f"{active.mechanic.id} recover"
            ),
            commitments=self._schedule(active, at),
        )

    # -- the interface to the rotation loop --------------------------------------

    def _schedule(self, active: _Active, at: float) -> list[Commitment]:
        """The whole remaining schedule for this mechanic, as claims on future time.

        This is the interface between the two loops, and it is why the runner projects
        forward instead of only describing now. A rotation that learns about movement when
        it starts can only react; one that learns two seconds early can pick an instant and
        lose nothing.
        """
        mechanic = active.mechanic
        act_lead_s = mechanic.ability_lead_ms / 1000.0
        out: list[Commitment] = []

        needs_movement = active.phase in (Phase.POSITION, Phase.RECOVER) or (
            active.phase in (Phase.OBSERVE, Phase.RESOLVE) and mechanic.resolve.kind.value != "stay"
        )
        if needs_movement:
            out.append(
                Commitment(
                    kind=CommitmentKind.MOVING,
                    # Movement may not have started yet — during OBSERVE the claim is on
                    # the near future, which is exactly the warning the rotation needs.
                    start_at=at if active.phase is Phase.POSITION else at + 0.2,
                    end_at=active.deadline_at - act_lead_s,
                    reason=f"{mechanic.id} positioning",
                )
            )

        if mechanic.ability_id and not active.ability_fired:
            out.append(
                Commitment(
                    kind=CommitmentKind.GCD_RESERVED,
                    start_at=active.deadline_at - act_lead_s,
                    end_at=active.deadline_at,
                    reason=f"{mechanic.id} requires {mechanic.ability_id}",
                    ability_id=mechanic.ability_id,
                )
            )

        if active.phase is Phase.OBSERVE and mechanic.look.at is not LookAt.BOSS:
            out.append(
                Commitment(
                    kind=CommitmentKind.CAMERA_AWAY,
                    start_at=at,
                    end_at=active.deadline_at,
                    reason=f"{mechanic.id} needs the camera elsewhere",
                )
            )

        return out

    # -- bookkeeping -------------------------------------------------------------

    def _advance(self, active: _Active, phase: Phase, at: float) -> None:
        active.phase = phase
        active.phase_started_at = at
        active.settle_until = 0.0

    def _fail(self, active: _Active, reason: str, at: float) -> None:
        self.failures.append((active.mechanic.id, reason))
        self.active = None

    def status(self) -> str:
        if self.active is None:
            return f"mech: idle done={len(self.completed)} failed={len(self.failures)}"
        return (
            f"mech: {self.active.mechanic.id} [{self.active.phase.value}] "
            f"done={len(self.completed)} failed={len(self.failures)}"
        )

    def stats(self) -> dict[str, object]:
        return {
            "script": self.script.id,
            "completed": len(self.completed),
            "failed": len(self.failures),
            "failures": self.failures[-4:],
            "active": self.active.mechanic.id if self.active else None,
            "phase": self.active.phase.value if self.active else "idle",
        }
