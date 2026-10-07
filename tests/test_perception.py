"""Probes, the sensor bundle, and `WorldState` semantics."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_bar_image
from player.capture.source import Frame, StaticSource
from player.geometry import Geometry, Rect, RelRect
from player.perceive.detector import TelegraphSegmenter, rgb_to_hsv
from player.perceive.pipeline import PerceptionPipeline
from player.perceive.probes import (
    BarProbe,
    ColorProbe,
    CooldownRingProbe,
    PresenceProbe,
    TemplateProbe,
)
from player.perceive.sensor import DerivedSensor, ProbeSensor, SensorBundle
from player.state import Field, WorldState

GREEN = (126, 202, 108)
REGION = RelRect(0.0, 0.0, 1.0, 1.0)


# -- bar probe --------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_bar_probe_reads_fill_fraction(fraction):
    probe = BarProbe("hp", REGION, GREEN)
    reading = probe.read(make_bar_image(fraction, GREEN))
    assert reading.value == pytest.approx(fraction, abs=0.02)


def test_bar_probe_tolerates_colour_drift():
    """The game applies bloom; an exact-match probe would read zero constantly."""
    probe = BarProbe("hp", REGION, GREEN, tolerance=45)
    shifted = (GREEN[0] + 20, GREEN[1] - 15, GREEN[2] + 10)
    assert probe.read(make_bar_image(0.6, shifted)).value == pytest.approx(0.6, abs=0.03)


def test_bar_probe_reports_low_confidence_on_noise():
    """A region full of unrelated pixels must not read as a confident bar."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (8, 100, 3), dtype=np.uint8)
    assert BarProbe("hp", REGION, GREEN).read(noise).confidence < 0.9


def test_bar_probe_handles_empty_crop():
    assert BarProbe("hp", REGION, GREEN).read(np.zeros((0, 0, 3), np.uint8)).value is None


# -- other probes -----------------------------------------------------------------


def test_color_probe_thresholds_to_bool():
    probe = ColorProbe("lit", REGION, GREEN, threshold=0.5)
    assert probe.read(make_bar_image(0.8, GREEN)).value is True
    assert probe.read(make_bar_image(0.2, GREEN)).value is False


def test_color_probe_confidence_drops_near_the_threshold():
    probe = ColorProbe("lit", REGION, GREEN, threshold=0.5)
    near = probe.read(make_bar_image(0.5, GREEN)).confidence
    far = probe.read(make_bar_image(1.0, GREEN)).confidence
    assert near < far


def test_presence_probe_distinguishes_populated_from_blank():
    blank = np.full((30, 60, 3), 18, dtype=np.uint8)
    assert PresenceProbe("p", REGION).read(blank).value is False

    populated = blank.copy()
    populated[5:25, 5:55] = 200
    assert PresenceProbe("p", REGION).read(populated).value is True


def test_template_probe_matches_its_own_patch():
    rng = np.random.default_rng(1)
    scene = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    template = scene[10:20, 10:20].copy()
    probe = TemplateProbe("icon", REGION, template, threshold=None)
    assert probe.read(scene).value > 0.99


def test_template_probe_rejects_a_different_patch():
    rng = np.random.default_rng(2)
    scene = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (10, 10, 3), dtype=np.uint8)
    probe = TemplateProbe("icon", REGION, other, threshold=0.9)
    assert probe.read(scene).value is False


def test_cooldown_probe_progress_tracks_the_overlay():
    """Zero when ready, approaching one when freshly used."""
    ready = np.full((40, 40, 3), 200, dtype=np.uint8)
    probe = CooldownRingProbe("cd", REGION)
    assert probe.read(ready).value == pytest.approx(0.0, abs=0.02)

    half = ready.copy()
    half[:, :20] = 20  # half the icon overlaid
    assert probe.read(half).value == pytest.approx(0.5, abs=0.05)

    full = np.full((40, 40, 3), 20, dtype=np.uint8)
    assert probe.read(full).value == pytest.approx(1.0, abs=0.02)


def test_cooldown_probe_is_ready_below_epsilon():
    probe = CooldownRingProbe("cd", REGION)
    assert probe.is_ready(0.01)
    assert not probe.is_ready(0.5)


# -- world state ------------------------------------------------------------------


def test_field_staleness_and_trust():
    field = Field(value=0.5, confidence=0.9, updated_at=100.0)
    assert not field.is_stale(100.1, 0.25)
    assert field.is_stale(100.5, 0.25)
    assert field.trusted(0.5)
    assert not Field(value=0.5, confidence=0.2, updated_at=100.0).trusted(0.5)


def test_typed_accessors_coerce_safely():
    state = WorldState(captured_at=0.0)
    state.set("n", 0.5)
    state.set("b", True)
    state.set("s", "hello")
    assert state.num("n") == 0.5
    assert state.flag("b") is True
    assert state.text("s") == "hello"
    assert state.num("missing", 9.0) == 9.0
    assert state.num("s") == 0.0  # a string is not a number; do not guess


def test_bool_is_not_treated_as_a_number():
    state = WorldState(captured_at=0.0)
    state.set("b", True)
    assert state.num("b", -1.0) == -1.0


def test_missing_reports_absent_required_fields():
    state = WorldState(captured_at=0.0)
    state.set("a", 1)
    assert state.missing(["a", "b"]) == ["b"]


def test_summary_omits_metadata():
    state = WorldState(captured_at=0.0)
    state.set("a", 0.123456, confidence=0.3)
    summary = state.to_summary()
    assert summary == {"a": 0.123}


# -- sensors and pipeline ---------------------------------------------------------


def _frame(image: np.ndarray) -> Frame:
    h, w = image.shape[:2]
    return Frame(image=image, captured_at=100.0, index=1, client_rect=Rect(0, 0, w, h))


def test_probe_sensor_writes_fields(geo):
    image = make_bar_image(0.6, GREEN, width=200, height=20)
    sensor = ProbeSensor("v", [BarProbe("player.hp_frac", REGION, GREEN)])
    state = WorldState(captured_at=100.0)
    frame = _frame(image)
    sensor.observe(frame, Geometry(Rect(0, 0, 200, 20), (200, 20)), state)
    assert state.num("player.hp_frac") == pytest.approx(0.6, abs=0.02)


def test_probe_that_raises_yields_zero_confidence():
    """One bad probe must not blind the rest of the bundle."""

    class Exploding:
        name = "bad"
        region = REGION

        def read(self, crop):
            raise RuntimeError("boom")

    sensor = ProbeSensor("v", [Exploding()])
    state = WorldState(captured_at=0.0)
    sensor.observe(_frame(np.zeros((20, 20, 3), np.uint8)), Geometry(Rect(0, 0, 20, 20), (20, 20)), state)
    assert state.field("bad").confidence == 0.0


def test_derived_sensor_runs_after_pixel_sensors():
    def compute(state: WorldState) -> None:
        state.set("derived", state.num("raw") * 2)

    bundle = SensorBundle(
        [
            ProbeSensor("v", [BarProbe("raw", REGION, GREEN)]),
            DerivedSensor("d", ("derived",), compute),
        ]
    )
    state = WorldState(captured_at=0.0)
    image = make_bar_image(0.5, GREEN, width=100, height=10)
    bundle.observe(1, _frame(image), Geometry(Rect(0, 0, 100, 10), (100, 10)), state)
    assert state.num("derived") == pytest.approx(1.0, abs=0.05)


def test_bundle_reports_unsatisfied_requirements():
    bundle = SensorBundle([ProbeSensor("v", [BarProbe("a", REGION, GREEN)])])
    assert bundle.check_requirements({"a", "b"}) == ["b"]


def test_bundle_honours_wildcard_providers():
    class Wild:
        name = "w"
        provides = ("action.*",)
        cadence = 1

        def observe(self, frame, geo, state):
            return None

    bundle = SensorBundle([Wild()])
    assert bundle.check_requirements({"action.fire4.ready"}) == []


def test_cadence_skips_ticks():
    calls = []

    class Counting:
        name = "c"
        provides = ()
        cadence = 3

        def observe(self, frame, geo, state):
            calls.append(1)

    bundle = SensorBundle([Counting()])
    frame = _frame(np.zeros((10, 10, 3), np.uint8))
    geo = Geometry(Rect(0, 0, 10, 10), (10, 10))
    for tick in range(1, 7):
        bundle.observe(tick, frame, geo, WorldState())
    assert len(calls) == 2  # ticks 3 and 6


def test_pipeline_carries_fields_forward_between_low_cadence_runs():
    """Otherwise a 4-tick sensor's fields vanish for three of every four frames."""
    bundle = SensorBundle([ProbeSensor("v", [BarProbe("hp", REGION, GREEN)], cadence=3)])
    pipeline = PerceptionPipeline(bundle)
    source = StaticSource(make_bar_image(0.5, GREEN, width=100, height=10))

    first = pipeline.process(source.grab())
    assert first.field("hp") is not None

    second = pipeline.process(source.grab())
    assert second.field("hp") is not None  # carried, not re-read
    assert second.field("hp").updated_at == first.field("hp").updated_at


def test_pipeline_rebuilds_geometry_when_the_window_moves():
    pipeline = PerceptionPipeline(SensorBundle([]))
    image = np.zeros((100, 100, 3), np.uint8)

    pipeline.process(Frame(image, 0.0, 1, Rect(0, 0, 100, 100)))
    pipeline.process(Frame(image, 0.0, 2, Rect(0, 0, 100, 100)))
    assert pipeline.geometry_rebuilds == 1

    pipeline.process(Frame(image, 0.0, 3, Rect(500, 300, 100, 100)))
    assert pipeline.geometry_rebuilds == 2


# -- detector ---------------------------------------------------------------------


def test_rgb_to_hsv_matches_known_values():
    pixels = np.array([[[255, 0, 0], [0, 255, 0], [255, 255, 255]]], dtype=np.uint8)
    hsv = rgb_to_hsv(pixels)
    assert hsv[0, 0, 0] == pytest.approx(0.0, abs=1.0)
    assert hsv[0, 1, 0] == pytest.approx(120.0, abs=1.0)
    assert hsv[0, 2, 1] == pytest.approx(0.0, abs=0.01)  # white is unsaturated


def test_segmenter_finds_a_saturated_orange_blob():
    image = np.full((200, 200, 3), 40, dtype=np.uint8)
    image[60:140, 60:140] = (240, 110, 30)
    detections = TelegraphSegmenter(stride=2).detect(image)
    assert detections
    box = detections[0].bbox
    assert box.contains(100, 100)


def test_segmenter_ignores_desaturated_scenery():
    image = np.full((200, 200, 3), 40, dtype=np.uint8)
    image[60:140, 60:140] = (90, 80, 75)  # brownish, low saturation
    assert TelegraphSegmenter(stride=2).detect(image) == []
