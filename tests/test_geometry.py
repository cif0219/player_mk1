"""Coordinate spaces.

These are the tests that exist because mixing spaces is the most productive bug source in
screen automation, and a round-trip test catches every scale-factor mistake at once.
"""

from __future__ import annotations

import pytest

from player.geometry import ClientPoint, FramePoint, Geometry, Rect, RelRect, ScreenPoint


def test_client_to_screen_offsets_by_window_origin(geo):
    assert geo.client_to_screen(ClientPoint(0, 0)) == ScreenPoint(100, 50)
    assert geo.client_to_screen(ClientPoint(10, 20)) == ScreenPoint(110, 70)


def test_screen_client_roundtrip(geo):
    original = ScreenPoint(640, 480)
    assert geo.client_to_screen(geo.screen_to_client(original)) == original


def test_frame_client_roundtrip_at_full_scale(geo):
    original = FramePoint(300, 400)
    assert geo.client_to_frame(geo.frame_to_client(original)) == original


def test_downscaled_capture_halves_frame_coordinates(half_geo):
    assert half_geo.client_to_frame(ClientPoint(400, 200)) == FramePoint(200, 100)
    assert half_geo.frame_to_client(FramePoint(200, 100)) == ClientPoint(400, 200)


def test_frame_to_screen_composes_both_conversions(half_geo):
    # Frame (200,100) -> client (400,200) -> screen (500,250).
    assert half_geo.frame_to_screen(FramePoint(200, 100)) == ScreenPoint(500, 250)


def test_relrect_resolves_against_client_size():
    region = RelRect(0.5, 0.25, 0.1, 0.05)
    assert region.to_client(1920, 1080) == Rect(960, 270, 192, 54)


def test_relrect_survives_resolution_change():
    """The whole reason layout is authored in fractions."""
    region = RelRect(0.5, 0.25, 0.1, 0.05)
    at_1440 = region.to_client(2560, 1440)
    at_1080 = region.to_client(1920, 1080)
    assert at_1440.x / 2560 == pytest.approx(at_1080.x / 1920, abs=1e-3)
    assert at_1440.w / 2560 == pytest.approx(at_1080.w / 1920, abs=1e-3)


def test_region_to_frame_applies_capture_downscale(half_geo):
    region = RelRect(0.5, 0.5, 0.1, 0.1)
    frame_rect = half_geo.region_to_frame(region)
    assert frame_rect.x == 480  # 0.5 * 1920 * 0.5
    assert frame_rect.w == 96  # 0.1 * 1920 * 0.5


def test_region_to_frame_clamps_to_frame_bounds(geo):
    """A window resize mid-flight can push a region past the edge; it must clip, not throw."""
    region = RelRect(0.95, 0.95, 0.2, 0.2)
    frame_rect = geo.region_to_frame(region)
    assert frame_rect.right <= geo.frame_w
    assert frame_rect.bottom <= geo.frame_h


def test_rect_contains_is_half_open():
    rect = Rect(10, 10, 5, 5)
    assert rect.contains(10, 10)
    assert rect.contains(14, 14)
    assert not rect.contains(15, 15)  # exclusive upper bound


def test_geometry_rejects_degenerate_window():
    with pytest.raises(ValueError):
        Geometry(Rect(0, 0, 0, 100), (100, 100))
    with pytest.raises(ValueError):
        Geometry(Rect(0, 0, 100, 100), (0, 100))
