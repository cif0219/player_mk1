"""Arena maths: bearings, transforms, and self-calibrating localisation."""

from __future__ import annotations

import math

import pytest

from player.world.arena import (
    ArenaPoint,
    CameraPose,
    Observability,
    bearing_deg,
    default_ring_layout,
    normalise_deg,
    shortest_turn_deg,
)
from player.world.localize import Localizer, Sighting
from player.world.transform import fit_homography, fit_similarity

# -- bearings ---------------------------------------------------------------------


def test_bearing_is_clockwise_from_north():
    origin = ArenaPoint(0, 0)
    assert bearing_deg(origin, ArenaPoint(0, 10)) == pytest.approx(0)
    assert bearing_deg(origin, ArenaPoint(10, 0)) == pytest.approx(90)
    assert bearing_deg(origin, ArenaPoint(0, -10)) == pytest.approx(180)
    assert bearing_deg(origin, ArenaPoint(-10, 0)) == pytest.approx(270)


def test_shortest_turn_takes_the_short_way_round():
    """Taking the 350-degree route instead of the 10-degree one loses the mechanic."""
    assert shortest_turn_deg(350, 10) == pytest.approx(20)
    assert shortest_turn_deg(10, 350) == pytest.approx(-20)
    assert shortest_turn_deg(0, 180) == pytest.approx(180)


def test_rotation_is_clockwise():
    north = ArenaPoint(0, 10)
    east = north.rotated(90)
    assert east.x == pytest.approx(10)
    assert east.y == pytest.approx(0, abs=1e-9)


def test_camera_relative_puts_forward_on_y():
    """+Y is the W direction. Getting this wrong walks the character into a wall."""
    pose = CameraPose(yaw_deg=90, confidence=1.0)  # facing east
    local = pose.to_camera_relative(ArenaPoint(10, 0))  # something due east
    assert local.y == pytest.approx(10)  # straight ahead
    assert local.x == pytest.approx(0, abs=1e-9)


def test_camera_relative_puts_right_on_x():
    pose = CameraPose(yaw_deg=0, confidence=1.0)  # facing north
    local = pose.to_camera_relative(ArenaPoint(10, 0))  # due east == to the right
    assert local.x == pytest.approx(10)
    assert local.y == pytest.approx(0, abs=1e-9)


# -- observability ----------------------------------------------------------------


def test_observability_covers_only_what_is_in_the_cone():
    obs = Observability(
        pose=CameraPose(yaw_deg=0, confidence=1.0),
        player=ArenaPoint(0, 0),
        usable_fov_deg=70,
    )
    assert obs.covers(ArenaPoint(0, 10))  # straight ahead
    assert not obs.covers(ArenaPoint(0, -10))  # directly behind


def test_unknown_pose_covers_nothing():
    """'I do not know where I am looking' must not read as 'I can see everything'."""
    obs = Observability(pose=CameraPose(confidence=0.0), player=ArenaPoint(0, 0))
    assert not obs.covers(ArenaPoint(0, 5))


def test_observability_respects_range():
    obs = Observability(
        pose=CameraPose(yaw_deg=0, confidence=1.0), player=ArenaPoint(0, 0), max_range_m=20
    )
    assert not obs.covers(ArenaPoint(0, 50))


# -- similarity -------------------------------------------------------------------


def _apply(points, rotation_deg, scale, tx, ty):
    rad = math.radians(rotation_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    return [
        ArenaPoint(scale * (p.x * cos - p.y * sin) + tx, scale * (p.x * sin + p.y * cos) + ty)
        for p in points
    ]


def test_similarity_recovers_rotation_scale_and_translation():
    source = [ArenaPoint(0, 10), ArenaPoint(10, 0), ArenaPoint(-10, 0)]
    observed = _apply(source, rotation_deg=37.0, scale=2.5, tx=11.0, ty=-4.0)

    fit = fit_similarity(source, observed)

    assert fit is not None
    assert fit.rotation_deg == pytest.approx(37.0, abs=1e-6)
    assert fit.scale == pytest.approx(2.5, abs=1e-9)
    assert fit.tx == pytest.approx(11.0, abs=1e-6)
    assert fit.residual_rms == pytest.approx(0.0, abs=1e-9)


def test_similarity_inverse_round_trips():
    source = [ArenaPoint(0, 10), ArenaPoint(10, 0)]
    observed = _apply(source, 45.0, 3.0, 5.0, 5.0)
    fit = fit_similarity(source, observed)
    recovered = fit.unapply(fit.apply(ArenaPoint(3, 7)))
    assert recovered.x == pytest.approx(3, abs=1e-6)
    assert recovered.y == pytest.approx(7, abs=1e-6)


def test_similarity_rejects_degenerate_input():
    assert fit_similarity([ArenaPoint(0, 0)], [ArenaPoint(1, 1)]) is None
    assert fit_similarity([ArenaPoint(1, 1), ArenaPoint(1, 1)], [ArenaPoint(0, 0), ArenaPoint(2, 2)]) is None


def test_similarity_residual_flags_a_bad_correspondence():
    """The residual is what lets the localiser reject its own answer."""
    source = [ArenaPoint(0, 10), ArenaPoint(10, 0), ArenaPoint(0, -10), ArenaPoint(-10, 0)]
    observed = _apply(source, 20.0, 2.0, 0.0, 0.0)
    observed[2] = ArenaPoint(observed[2].x + 40, observed[2].y - 30)  # one mark misidentified

    fit = fit_similarity(source, observed)
    assert fit.residual_rms > 5.0


# -- homography -------------------------------------------------------------------


def test_homography_round_trips_a_known_projection():
    arena = [ArenaPoint(-10, -10), ArenaPoint(10, -10), ArenaPoint(10, 10), ArenaPoint(-10, 10)]
    # A trapezoid: the ground plane seen at an angle.
    screen = [(200.0, 700.0), (1000.0, 700.0), (800.0, 400.0), (400.0, 400.0)]

    homography = fit_homography(arena, screen)

    assert homography is not None
    assert homography.residual_rms < 1e-6
    for a, s in zip(arena, screen):
        projected = homography.apply(a)
        assert projected[0] == pytest.approx(s[0], abs=1e-3)
        recovered = homography.unapply(s)
        assert recovered.x == pytest.approx(a.x, abs=1e-3)


def test_homography_needs_four_points():
    arena = [ArenaPoint(0, 0), ArenaPoint(1, 0), ArenaPoint(0, 1)]
    assert fit_homography(arena, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]) is None


# -- localiser --------------------------------------------------------------------


def _minimap_sightings(layout, player, yaw_deg, scale, marks=("A", "B", "C", "D")):
    """Render waymark positions onto a synthetic minimap, the way the game would."""
    pose = CameraPose(yaw_deg=yaw_deg, confidence=1.0)
    out = []
    for mark_id in marks:
        point = layout.get(mark_id)
        local = pose.to_camera_relative(point - player)
        out.append(
            Sighting(
                mark_id=mark_id,
                minimap_offset=(local.x * scale, -local.y * scale),
                candidates=(mark_id,),
            )
        )
    return out


def test_localiser_recovers_position_and_yaw():
    """The core result: two-plus marks give position, scale and yaw at once."""
    layout = default_ring_layout(radius_m=18.0)
    truth = ArenaPoint(4.0, -7.0)
    localizer = Localizer(layout)

    result = localizer.from_minimap(_minimap_sightings(layout, truth, 123.0, 3.2), at=1.0)

    assert result.usable
    assert result.player.distance_to(truth) < 0.01
    assert abs(shortest_turn_deg(result.pose.yaw_deg, 123.0)) < 0.01
    assert result.scale_px_per_m == pytest.approx(3.2, abs=1e-6)


@pytest.mark.parametrize("yaw", [0.0, 45.0, 90.0, 180.0, 270.0, 359.0])
def test_localiser_yaw_is_correct_at_every_bearing(yaw):
    """A sign error here shows up at some bearings and not others, so sweep them."""
    layout = default_ring_layout()
    truth = ArenaPoint(-3.0, 5.0)
    result = Localizer(layout).from_minimap(_minimap_sightings(layout, truth, yaw, 3.0), at=1.0)
    assert abs(shortest_turn_deg(result.pose.yaw_deg, yaw)) < 0.01


def test_localiser_needs_two_marks():
    layout = default_ring_layout()
    one = _minimap_sightings(layout, ArenaPoint(0, 0), 0.0, 3.0, marks=("A",))
    assert not Localizer(layout).from_minimap(one, at=1.0).usable


def test_localiser_rejects_a_misidentified_mark():
    """Reporting a confident wrong position is worse than reporting none."""
    layout = default_ring_layout()
    sightings = _minimap_sightings(layout, ArenaPoint(0, 0), 0.0, 3.0)
    bad = sightings[:-1] + [
        Sighting(mark_id="D", minimap_offset=(999.0, -999.0), candidates=("D",))
    ]
    result = Localizer(layout).from_minimap(bad, at=1.0)
    assert not result.usable
    assert "residual" in result.reason


def test_two_marks_are_capped_below_full_confidence():
    """An exact fit on two points means nothing: there was no freedom to be wrong."""
    layout = default_ring_layout()
    two = _minimap_sightings(layout, ArenaPoint(0, 0), 0.0, 3.0, marks=("A", "B"))
    four = _minimap_sightings(layout, ArenaPoint(0, 0), 0.0, 3.0)
    assert Localizer(layout).from_minimap(two, at=1.0).confidence < 1.0
    assert Localizer(layout).from_minimap(four, at=1.0).confidence == pytest.approx(1.0)


def _ambiguous_sightings(layout, player, yaw_deg, scale=3.0):
    """Sightings where each blob could be its lettered or numbered twin.

    Mirrors reality: A and 1 are both red in game, so a colour-keyed detector narrows a
    blob to a pair and no further.
    """
    pose = CameraPose(yaw_deg=yaw_deg, confidence=1.0)
    out = []
    for mark_id, pair in (("A", ("A", "1")), ("B", ("B", "2")), ("C", ("C", "3"))):
        local = pose.to_camera_relative(layout.get(mark_id) - player)
        out.append(
            Sighting(
                mark_id=mark_id,
                minimap_offset=(local.x * scale, -local.y * scale),
                candidates=pair,
            )
        )
    return out


def test_symmetric_layout_is_flagged_ambiguous_without_history():
    """A/B/C/D on the cardinals and 1/2/3/4 on the intercardinals is the *same shape*
    rotated 45 degrees, so both assignments fit exactly and geometry cannot choose.

    The honest behaviour is to say so with low confidence, not to pick and sound sure.
    """
    layout = default_ring_layout()
    localizer = Localizer(layout)

    result = localizer.from_minimap(
        _ambiguous_sightings(layout, ArenaPoint(2.0, 3.0), 200.0), at=1.0
    )

    assert result.usable
    assert result.ambiguous
    assert result.confidence < 0.4
    assert "symmetric" in result.reason


def test_history_breaks_a_symmetric_tie():
    """The character did not teleport and the camera did not spin 45 degrees in one frame."""
    layout = default_ring_layout()
    localizer = Localizer(layout)
    truth = ArenaPoint(2.0, 3.0)

    # Seed history from an unambiguous frame.
    localizer.from_minimap(_minimap_sightings(layout, truth, 200.0, 3.0), at=1.0)

    result = localizer.from_minimap(_ambiguous_sightings(layout, truth, 201.0), at=1.1)

    assert result.usable
    assert result.player.distance_to(truth) < 0.05
    assert abs(shortest_turn_deg(result.pose.yaw_deg, 201.0)) < 0.5
    assert result.confidence > 0.5  # trusted, because continuity settled it


def test_unambiguous_marks_need_no_tiebreak():
    layout = default_ring_layout()
    result = Localizer(layout).from_minimap(
        _minimap_sightings(layout, ArenaPoint(1.0, 1.0), 30.0, 3.0), at=1.0
    )
    assert not result.ambiguous
