"""Full pipeline over a synthetic session.

Capture → perception → policy → timeline → guard chain → backend, with no game and no
screen. This is the test that catches a break in the *wiring* rather than in any one
component, which unit tests structurally cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest

from games.ffxiv import FFXIVConfig
from games.ffxiv import build as build_ffxiv
from games.ffxiv.layout import Layout
from player.act.backend import NullBackend
from player.capture.replay import ReplaySource
from player.capture.source import Frame
from player.clock import now
from player.geometry import Rect
from player.perceive.pipeline import PerceptionPipeline
from player.policy.rotation import RotationPlanner
from player.runtime import Dispatcher
from player.act.timeline import InputTimeline
from player.safety.guards import SafetyGate
from tools.synth import generate, render_frame

SIZE = (960, 540)


@pytest.fixture(scope="module")
def synthetic_session(tmp_path_factory):
    # Long enough to span more than one GCD sweep — a session shorter than 2.5s never
    # shows the planner a ready GCD, and the test would pass or fail on session length
    # rather than on anything about the pipeline.
    root = tmp_path_factory.mktemp("sessions")
    return generate(root, "synth", seconds=6.0, fps=20, size=SIZE)


@pytest.fixture
def profile():
    return build_ffxiv(FFXIVConfig(rotation="blm.dummy_safe"))


def _frame(image: np.ndarray, at: float, index: int) -> Frame:
    h, w = image.shape[:2]
    return Frame(image=image, captured_at=at, index=index, client_rect=Rect(0, 0, w, h))


# -- perception over synthetic frames ---------------------------------------------


def test_perception_reads_the_synthetic_hud(profile):
    """The bars drawn by the generator must come back out of the probes."""
    layout = Layout()
    image = render_frame(layout, SIZE, t=0.5)
    state = PerceptionPipeline(profile.sensors).process(_frame(image, now(), 1))

    hp = state.field("player.hp_frac")
    assert hp is not None and hp.value is not None
    assert 0.5 < hp.value <= 1.0
    assert state.flag("target.exists")


def test_gcd_field_tracks_the_sweeping_overlay(profile):
    """Freshly used reads as more remaining than nearly-recovered."""
    layout = Layout()
    pipeline = PerceptionPipeline(profile.sensors)

    fresh = pipeline.process(_frame(render_frame(layout, SIZE, t=0.05), now(), 1))
    nearly = pipeline.process(_frame(render_frame(layout, SIZE, t=2.45), now(), 2))

    assert fresh.num("player.gcd_remaining_s") > nearly.num("player.gcd_remaining_s")


def test_uncalibrated_status_fields_are_present_but_untrusted(profile):
    """Declared so validation passes; zero-confidence so nothing acts on them."""
    state = PerceptionPipeline(profile.sensors).process(
        _frame(render_frame(Layout(), SIZE, t=1.0), now(), 1)
    )
    field = state.field("buff.astral_fire.active")
    assert field is not None
    assert field.confidence == 0.0


# -- the whole loop ---------------------------------------------------------------


def test_pipeline_drives_dispatch_end_to_end(profile, synthetic_session):
    """Replay a session and assert real keypresses come out of the far end."""
    source = ReplaySource(synthetic_session)
    source.open()

    pipeline = PerceptionPipeline(profile.sensors)
    planner = RotationPlanner(profile.rotation("blm.dummy_safe"))
    timeline = InputTimeline()
    backend = NullBackend()
    dispatcher = Dispatcher(timeline, backend, SafetyGate())

    # A virtual clock, so the loop is not bounded by how fast frames decode. Policy
    # timing is real-time by design; driving it explicitly is what lets a 2-second
    # session be evaluated deterministically in milliseconds.
    clock = 1000.0
    while (frame := source.grab()) is not None:
        clock += 0.05
        state = pipeline.process(frame)
        state.captured_at = clock  # align the state with the virtual clock
        decision = planner.decide(state, at=clock)
        if decision is not None:
            timeline.submit(decision.plan, at=clock)
        dispatcher.pump(state, at=clock)

    source.close()

    assert planner.decisions >= 1
    assert dispatcher.dispatched >= 1
    assert backend.keys_pressed(), "no key ever reached the backend"


def test_every_press_is_released(profile, synthetic_session):
    """The invariant that matters most: no key is left held at the end of a run."""
    source = ReplaySource(synthetic_session)
    source.open()

    pipeline = PerceptionPipeline(profile.sensors)
    planner = RotationPlanner(profile.rotation("blm.dummy_safe"))
    timeline = InputTimeline()
    backend = NullBackend()
    dispatcher = Dispatcher(timeline, backend, SafetyGate())

    clock = 1000.0
    while (frame := source.grab()) is not None:
        clock += 0.05
        state = pipeline.process(frame)
        state.captured_at = clock
        decision = planner.decide(state, at=clock)
        if decision is not None:
            timeline.submit(decision.plan, at=clock)
        dispatcher.pump(state, at=clock)

    # Drain anything still scheduled, then confirm nothing is held.
    clock += 5.0
    dispatcher.pump(_stale_free_state(pipeline, clock), at=clock)
    source.close()

    downs = [e for e in backend.events if e.kind == "key_down"]
    ups = [e for e in backend.events if e.kind == "key_up"]
    assert len(downs) == len(ups)
    assert dispatcher.held == set()


def _stale_free_state(pipeline: PerceptionPipeline, at: float):
    state = pipeline._previous.copy()  # noqa: SLF001 - test reaches in deliberately
    state.captured_at = at
    return state


def test_replay_produces_identical_decisions_twice(profile, synthetic_session):
    """Determinism is what makes replay a regression harness rather than a demo."""

    def run() -> list[str]:
        source = ReplaySource(synthetic_session)
        source.open()
        pipeline = PerceptionPipeline(profile.sensors)
        planner = RotationPlanner(profile.rotation("blm.dummy_safe"))
        decisions: list[str] = []
        clock = 1000.0
        while (frame := source.grab()) is not None:
            clock += 0.05
            state = pipeline.process(frame)
            state.captured_at = clock
            decision = planner.decide(state, at=clock)
            if decision is not None:
                decisions.append(f"{clock:.2f}:{decision.describe()}")
        source.close()
        return decisions

    assert run() == run()
