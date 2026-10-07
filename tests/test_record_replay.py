"""Recording and replay.

Replay determinism is what makes every other test in this suite possible against real
data, so it gets tested directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from player.capture.replay import ReplaySource
from player.capture.source import Frame, FrameSlot
from player.geometry import Rect
from player.record.recorder import Recorder, RecorderConfig
from player.record.session import Session
from player.state import WorldState


@pytest.fixture
def recorded_session(tmp_path):
    """A three-frame session with states and dispatches, flushed to disk."""
    recorder = Recorder(
        RecorderConfig(enabled=True, root=tmp_path, name="s1", frame_every_n=1, jpeg_quality=95)
    )
    recorder.start({"profile": "test", "client_rect": [0, 0, 64, 48]})

    for index in range(1, 4):
        image = np.full((48, 64, 3), index * 40, dtype=np.uint8)
        recorder.frame(index, image, captured_at=recorder.t0 + index * 0.1, rect=Rect(0, 0, 64, 48))
        state = WorldState(tick=index, captured_at=recorder.t0 + index * 0.1)
        state.set("player.hp_frac", 1.0 - index * 0.1)
        recorder.state(state)
        recorder.dispatch("1", "key_down")

    recorder.stop()
    return tmp_path / "s1"


def test_session_round_trips(recorded_session):
    session = Session(recorded_session)
    assert session.meta["profile"] == "test"
    assert len(session.frame_paths()) == 3
    assert len(list(session.states())) == 3
    assert len(list(session.trace("dispatch"))) == 3


def test_trace_times_are_relative_to_session_start(recorded_session):
    times = [ev.t for ev in Session(recorded_session).trace("frame")]
    assert times == sorted(times)
    assert times[0] == pytest.approx(0.1, abs=0.05)


def test_recorded_states_replay_without_frames(recorded_session):
    """Policy regression tests run from these alone, which is why they run in ms."""
    states = list(Session(recorded_session).states())
    assert states[0].data["fields"]["player.hp_frac"] == pytest.approx(0.9)


def test_replay_source_emits_every_frame_in_order(recorded_session):
    source = ReplaySource(recorded_session)
    source.open()
    indices = []
    while (frame := source.grab()) is not None:
        indices.append(frame.index)
    source.close()
    assert indices == [1, 2, 3]


def test_replay_is_deterministic_across_runs(recorded_session):
    """Two passes must produce identical pixels, or no regression test means anything."""

    def collect():
        source = ReplaySource(recorded_session)
        source.open()
        out = []
        while (frame := source.grab()) is not None:
            out.append(frame.image.copy())
        source.close()
        return out

    first, second = collect(), collect()
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_replay_marks_itself_exhausted(recorded_session):
    source = ReplaySource(recorded_session)
    source.open()
    while source.grab() is not None:
        pass
    assert source.exhausted


def test_replay_can_loop(recorded_session):
    source = ReplaySource(recorded_session, loop=True)
    source.open()
    indices = [source.grab().index for _ in range(5)]
    assert indices == [1, 2, 3, 1, 2]


def test_replay_recovers_the_original_client_rect(recorded_session):
    source = ReplaySource(recorded_session)
    source.open()
    assert source.grab().client_rect == Rect(0, 0, 64, 48)


def test_disabled_recorder_is_a_noop(tmp_path):
    recorder = Recorder(RecorderConfig(enabled=False, root=tmp_path))
    recorder.start()
    recorder.frame(1, np.zeros((4, 4, 3), np.uint8), 0.0, Rect(0, 0, 4, 4))
    recorder.note("hello")
    recorder.stop()
    assert not (tmp_path / "trace.jsonl").exists()


# -- frame slot -------------------------------------------------------------------


def _frame(index: int) -> Frame:
    return Frame(np.zeros((4, 4, 3), np.uint8), captured_at=float(index), index=index,
                 client_rect=Rect(0, 0, 4, 4))


def test_slot_holds_only_the_newest_frame():
    """A decision on a two-frame-old image is a worse decision; there is no reason to keep it."""
    slot = FrameSlot()
    slot.publish(_frame(1))
    slot.publish(_frame(2))
    assert slot.take(timeout=0.01).index == 2
    assert slot.dropped == 1


def test_slot_take_consumes():
    slot = FrameSlot()
    slot.publish(_frame(1))
    assert slot.take(timeout=0.01) is not None
    assert slot.take(timeout=0.01) is None


def test_slot_latest_peeks_without_consuming():
    slot = FrameSlot()
    slot.publish(_frame(7))
    assert slot.latest().index == 7
    assert slot.take(timeout=0.01).index == 7
