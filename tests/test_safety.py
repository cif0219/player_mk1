"""The guard chain, and the dispatcher's obligation to release held keys."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from conftest import make_state
from player.act.backend import NullBackend
from player.act.timeline import InputTimeline, Plan, Priority
from player.runtime import Dispatcher
from player.safety.guards import (
    ConfidenceGuard,
    ForegroundGuard,
    Guard,
    GuardStatus,
    KillSwitchGuard,
    RateGuard,
    SafetyGate,
    StalenessGuard,
    TakeoverGuard,
)
from player.state import WorldState


@dataclass
class FakeKillSwitch:
    tripped: bool = False
    paused: bool = False
    last_human_key_at: float = 0.0

    def expect(self, scancode: int) -> None:
        return None


@dataclass
class FakeTracker:
    available: bool = True
    foreground: bool = True

    def is_foreground(self) -> bool:
        return self.foreground


# -- individual guards ------------------------------------------------------------


def test_killswitch_guard_blocks_when_tripped():
    guard = KillSwitchGuard(FakeKillSwitch(tripped=True))
    assert guard.check(None, 0.0).blocking


def test_killswitch_guard_blocks_when_paused():
    assert KillSwitchGuard(FakeKillSwitch(paused=True)).check(None, 0.0).blocking


def test_foreground_guard_blocks_when_window_unfocused():
    guard = ForegroundGuard(FakeTracker(foreground=False))
    assert guard.check(None, 0.0).blocking


def test_foreground_guard_blocks_when_tracking_unavailable():
    """No way to tell where input would land means do not send input."""
    guard = ForegroundGuard(FakeTracker(available=False))
    assert guard.check(None, 0.0).blocking


def test_foreground_guard_can_be_switched_off_for_replay():
    guard = ForegroundGuard(FakeTracker(available=False), required=False)
    assert not guard.check(None, 0.0).blocking


def test_staleness_guard_blocks_on_old_state():
    guard = StalenessGuard(max_age_ms=250)
    state = make_state(at=1000.0)
    assert not guard.check(state, 1000.1).blocking
    assert guard.check(state, 1000.5).blocking


def test_staleness_guard_blocks_when_there_is_no_state():
    assert StalenessGuard().check(None, 0.0).blocking


def test_confidence_guard_blocks_on_untrusted_fields():
    guard = ConfidenceGuard(("player.hp_frac",), min_confidence=0.5)
    good = make_state({"player.hp_frac": 0.9})
    assert not guard.check(good, good.captured_at).blocking

    bad = WorldState(captured_at=0.0)
    bad.set("player.hp_frac", 0.9, confidence=0.1)
    assert guard.check(bad, 0.0).blocking


def test_confidence_guard_blocks_on_missing_field():
    guard = ConfidenceGuard(("player.hp_frac",))
    assert guard.check(make_state(), 0.0).blocking


def test_rate_guard_blocks_past_the_ceiling():
    guard = RateGuard(max_per_sec=3)
    for _ in range(3):
        assert guard.consume(at=100.0)
    assert not guard.consume(at=100.0)
    assert guard.check(None, 100.0).blocking
    # The window slides.
    assert guard.consume(at=101.5)


def test_takeover_guard_yields_after_human_input():
    ks = FakeKillSwitch(last_human_key_at=100.0)
    guard = TakeoverGuard(ks, cooldown_ms=1000)
    assert guard.check(None, 100.5).blocking
    assert not guard.check(None, 101.5).blocking


def test_takeover_guard_is_inert_before_any_human_input():
    """A guard that always blocks is a guard nobody keeps enabled."""
    assert not TakeoverGuard(FakeKillSwitch()).check(None, 100.0).blocking


# -- the gate ---------------------------------------------------------------------


def test_gate_allows_when_every_guard_is_clear():
    gate = SafetyGate([KillSwitchGuard(FakeKillSwitch()), StalenessGuard()])
    state = make_state(at=1000.0)
    assert gate.allowed(state, 1000.05)


def test_gate_reports_everything_blocking_not_just_the_first():
    """So 'why is it doing nothing' is answerable at a glance."""
    gate = SafetyGate(
        [
            KillSwitchGuard(FakeKillSwitch(tripped=True)),
            ForegroundGuard(FakeTracker(foreground=False)),
            StalenessGuard(),
        ]
    )
    assert not gate.allowed(None, 0.0)
    assert {s.name for s in gate.blocking()} == {"killswitch", "foreground", "staleness"}


def test_gate_fails_closed_when_a_guard_raises():
    class Exploding:
        name = "boom"

        def check(self, state, at):
            raise RuntimeError("nope")

    gate = SafetyGate([Exploding()])
    assert not gate.allowed(None, 0.0)
    assert "guard error" in gate.blocking()[0].reason


# -- dispatcher ------------------------------------------------------------------


def _dispatcher(gate: SafetyGate) -> tuple[Dispatcher, InputTimeline, NullBackend]:
    timeline = InputTimeline()
    backend = NullBackend()
    return Dispatcher(timeline, backend, gate), timeline, backend


def test_dispatcher_sends_when_the_gate_is_clear():
    dispatcher, timeline, backend = _dispatcher(SafetyGate())
    timeline.submit(Plan.single("1", hold_ms=10), at=0.0)
    dispatcher.pump(make_state(at=0.0), at=0.0)
    assert backend.keys_pressed() == ["1"]


def test_dispatcher_sends_nothing_when_blocked():
    """Not even a stray release: nothing was pressed, so nothing is owed."""
    gate = SafetyGate([KillSwitchGuard(FakeKillSwitch(tripped=True))])
    dispatcher, timeline, backend = _dispatcher(gate)
    timeline.submit(Plan.single("1"), at=0.0)
    dispatcher.pump(make_state(at=0.0), at=0.0)
    assert backend.events == []


def test_blocking_releases_keys_that_were_already_held():
    """The invariant that stops a key being left down when a guard trips mid-hold."""
    ks = FakeKillSwitch()
    gate = SafetyGate([KillSwitchGuard(ks)])
    dispatcher, timeline, backend = _dispatcher(gate)

    timeline.submit(Plan.single("1", hold_ms=500), at=0.0)
    dispatcher.pump(make_state(at=0.0), at=0.0)
    assert dispatcher.held == {"1"}

    ks.tripped = True
    dispatcher.pump(make_state(at=0.1), at=0.1)

    assert dispatcher.held == set()
    assert backend.events[-1].kind == "key_up"
    assert timeline.pending() == 0


def test_blocking_flushes_the_timeline():
    """Plans built for a world that has since expired must not run when the gate clears."""
    ks = FakeKillSwitch()
    gate = SafetyGate([KillSwitchGuard(ks)])
    dispatcher, timeline, backend = _dispatcher(gate)

    timeline.submit(Plan(name="later", steps=(), priority=Priority.ROTATION), at=0.0)
    timeline.submit(Plan.single("2"), at=0.0)
    ks.tripped = True
    dispatcher.pump(make_state(at=0.0), at=0.0)

    assert timeline.pending() == 0


def test_rate_guard_reopens_once_the_window_slides():
    """A full window must not latch: check() alone has to let old dispatches expire,
    or the dispatcher never consumes again and the gate stays shut for the rest of the run."""
    from player.safety.guards import RateGuard
    guard = RateGuard(max_per_sec=3)
    for t in (10.0, 10.1, 10.2):
        assert guard.consume(t)
    assert guard.check(None, 10.3).blocking
    assert not guard.check(None, 11.25).blocking   # the first two have left the window
    assert guard.consume(11.25)
