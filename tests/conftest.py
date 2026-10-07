"""Shared fixtures. Everything here is headless — no game, no screen, no network."""

from __future__ import annotations

import numpy as np
import pytest

from player.geometry import Geometry, Rect
from player.state import WorldState


@pytest.fixture
def geo() -> Geometry:
    """A 1920x1080 window at (100, 50), captured 1:1."""
    return Geometry(client_rect_screen=Rect(100, 50, 1920, 1080), frame_size=(1920, 1080))


@pytest.fixture
def half_geo() -> Geometry:
    """The same window captured at half resolution."""
    return Geometry(client_rect_screen=Rect(100, 50, 1920, 1080), frame_size=(960, 540))


def make_state(fields: dict | None = None, at: float = 1000.0, **kwargs) -> WorldState:
    """A `WorldState` with the given fields, all fully confident."""
    state = WorldState(tick=1, captured_at=at, perceived_at=at)
    for name, value in (fields or {}).items():
        state.set(name, value, confidence=1.0, source="test")
    for name, value in kwargs.items():
        state.set(name.replace("__", "."), value, confidence=1.0, source="test")
    return state


def make_bar_image(
    fill_fraction: float,
    color: tuple[int, int, int] = (126, 202, 108),
    width: int = 100,
    height: int = 8,
    empty: tuple[int, int, int] = (20, 20, 24),
) -> np.ndarray:
    """A synthetic horizontal bar filled to `fill_fraction`."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = empty
    filled = int(round(width * fill_fraction))
    if filled > 0:
        img[:, :filled] = color
    return img


@pytest.fixture
def bar_image():
    return make_bar_image
