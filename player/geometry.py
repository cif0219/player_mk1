"""Coordinate spaces, made unmixable.

There are three distinct pixel spaces in a screen-perception player and mixing them is
the single most productive bug source in this kind of project:

  Screen  physical desktop pixels        -- what SendInput and the OS cursor consume
  Client  relative to the window's client area -- what HUD layout is authored in
  Frame   the captured buffer, possibly downscaled -- what perception computes on

They are separate types here so that passing one where another is expected is a type
error at review time rather than an off-by-a-scale-factor click at runtime. Conversion
goes through `Geometry`, which is the only object that knows the window rect, the DPI
scale, and the capture downscale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScreenPoint:
    """Physical desktop pixels."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class ClientPoint:
    """Pixels relative to the game window's client area origin."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class FramePoint:
    """Pixels in the captured buffer."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Rect:
    """An integer rectangle. Space is implied by whoever holds it."""

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def clamp_to(self, w: int, h: int) -> "Rect":
        """Clip against a `w`x`h` bound. Returns a zero-area rect if fully outside."""
        x0 = max(0, min(self.x, w))
        y0 = max(0, min(self.y, h))
        x1 = max(0, min(self.right, w))
        y1 = max(0, min(self.bottom, h))
        return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


@dataclass(frozen=True, slots=True)
class RelRect:
    """A HUD region as fractions of client size.

    Layout is authored in this form so a calibration done at 2560x1440 still resolves
    correctly at 1920x1080, as long as the in-game HUD scale is unchanged. Storing
    absolute pixels would silently break on every resolution change.
    """

    x: float
    y: float
    w: float
    h: float

    def to_client(self, client_w: int, client_h: int) -> Rect:
        return Rect(
            x=round(self.x * client_w),
            y=round(self.y * client_h),
            w=max(1, round(self.w * client_w)),
            h=max(1, round(self.h * client_h)),
        )


class Geometry:
    """Converts between the three spaces for one window at one capture scale.

    Rebuilt whenever the tracked window moves or resizes; every probe region is
    reprojected from its `RelRect` at that point, so a window move is not a recalibration.
    """

    def __init__(
        self,
        client_rect_screen: Rect,
        frame_size: tuple[int, int],
        dpi_scale: float = 1.0,
    ) -> None:
        if client_rect_screen.w <= 0 or client_rect_screen.h <= 0:
            raise ValueError(f"client rect must have positive area, got {client_rect_screen}")
        frame_w, frame_h = frame_size
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(f"frame size must be positive, got {frame_size}")

        self.client_rect_screen = client_rect_screen
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.dpi_scale = dpi_scale
        # Frame may be downscaled from the client area independently on each axis.
        self.scale_x = frame_w / client_rect_screen.w
        self.scale_y = frame_h / client_rect_screen.h

    @property
    def client_size(self) -> tuple[int, int]:
        return self.client_rect_screen.w, self.client_rect_screen.h

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.frame_w, self.frame_h

    # -- client <-> screen -------------------------------------------------------

    def client_to_screen(self, p: ClientPoint) -> ScreenPoint:
        return ScreenPoint(
            x=self.client_rect_screen.x + p.x,
            y=self.client_rect_screen.y + p.y,
        )

    def screen_to_client(self, p: ScreenPoint) -> ClientPoint:
        return ClientPoint(
            x=p.x - self.client_rect_screen.x,
            y=p.y - self.client_rect_screen.y,
        )

    # -- client <-> frame --------------------------------------------------------

    def client_to_frame(self, p: ClientPoint) -> FramePoint:
        return FramePoint(x=round(p.x * self.scale_x), y=round(p.y * self.scale_y))

    def frame_to_client(self, p: FramePoint) -> ClientPoint:
        return ClientPoint(x=round(p.x / self.scale_x), y=round(p.y / self.scale_y))

    # -- frame <-> screen --------------------------------------------------------

    def frame_to_screen(self, p: FramePoint) -> ScreenPoint:
        return self.client_to_screen(self.frame_to_client(p))

    def screen_to_frame(self, p: ScreenPoint) -> FramePoint:
        return self.client_to_frame(self.screen_to_client(p))

    # -- regions -----------------------------------------------------------------

    def region_to_frame(self, region: RelRect) -> Rect:
        """Resolve a HUD region straight to the frame-space slice a probe reads.

        This is the hot path: it runs once per probe per calibration, not per frame,
        because the pipeline caches resolved regions until `Geometry` changes.
        """
        client = region.to_client(*self.client_size)
        return Rect(
            x=round(client.x * self.scale_x),
            y=round(client.y * self.scale_y),
            w=max(1, round(client.w * self.scale_x)),
            h=max(1, round(client.h * self.scale_y)),
        ).clamp_to(self.frame_w, self.frame_h)

    def region_to_screen(self, region: RelRect) -> Rect:
        client = region.to_client(*self.client_size)
        return Rect(
            x=self.client_rect_screen.x + client.x,
            y=self.client_rect_screen.y + client.y,
            w=client.w,
            h=client.h,
        )

    def __repr__(self) -> str:
        return (
            f"Geometry(client={self.client_rect_screen}, "
            f"frame={self.frame_w}x{self.frame_h}, "
            f"scale=({self.scale_x:.3f},{self.scale_y:.3f}))"
        )
