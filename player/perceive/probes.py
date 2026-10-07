"""Numpy probes over fixed HUD regions.

A probe answers one question about one rectangle. It returns a value *and* a confidence,
because "the bar is at 0%" and "I could not find the bar" are different situations and a
bare float cannot distinguish them — which is exactly the distinction a safety guard
needs to make.

Regions are authored as `RelRect` (fractions of client size) and resolved to frame-space
`Rect` once per `Geometry` change, not per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..geometry import RelRect
from ..state import FieldValue

RGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ProbeReading:
    value: FieldValue
    confidence: float


@runtime_checkable
class Probe(Protocol):
    """One question about one rectangle."""

    name: str
    region: RelRect

    def read(self, crop: np.ndarray) -> ProbeReading: ...


def _color_mask(crop: np.ndarray, color: RGB, tolerance: int) -> np.ndarray:
    """Per-pixel boolean mask of "close enough to `color`" under Chebyshev distance.

    Chebyshev rather than Euclidean because HUD colours are flat and the failure we care
    about is a channel being clearly wrong, not a small overall drift.
    """
    diff = np.abs(crop.astype(np.int16) - np.asarray(color, dtype=np.int16))
    return np.all(diff <= tolerance, axis=2)


@dataclass(slots=True)
class BarProbe:
    """Fill fraction of a horizontal bar.

    Finds the boundary between filled and empty rather than counting matching pixels:
    counting is fooled by any decoration that happens to share the fill colour, while the
    boundary is what the bar actually means.

    Confidence reports how sharp that boundary is. A clean bar transitions over one or
    two columns; a smeared transition means the region is misaligned or something is
    overlapping the bar, and the reading should be distrusted rather than used.
    """

    name: str
    region: RelRect
    fill_color: RGB
    tolerance: int = 40
    column_threshold: float = 0.5  # share of a column that must match to count as filled
    vertical: bool = False

    def read(self, crop: np.ndarray) -> ProbeReading:
        if crop.size == 0:
            return ProbeReading(None, 0.0)

        mask = _color_mask(crop, self.fill_color, self.tolerance)
        # Collapse to a 1-D profile along the fill axis.
        profile = mask.mean(axis=0) if not self.vertical else mask.mean(axis=1)
        if profile.size == 0:
            return ProbeReading(None, 0.0)

        filled = profile >= self.column_threshold
        if not filled.any():
            # Genuinely empty is a real reading, but only if the region looks like a bar
            # at all. A region full of unrelated pixels also produces no matches.
            return ProbeReading(0.0, 0.6)

        # Last filled index, walking from the fill origin. Bars deplete toward one end,
        # so trailing matches beyond a gap are decoration, not fill.
        indices = np.flatnonzero(filled)
        boundary = int(indices[-1]) + 1
        if self.vertical:
            boundary = profile.size - int(indices[0])
        fraction = boundary / profile.size

        confidence = _boundary_sharpness(profile, self.column_threshold)
        return ProbeReading(round(float(np.clip(fraction, 0.0, 1.0)), 4), confidence)


def _boundary_sharpness(profile: np.ndarray, threshold: float) -> float:
    """How cleanly the profile separates into filled and empty.

    Counts columns sitting in the ambiguous middle band. Many ambiguous columns means the
    region is not framing a bar cleanly.
    """
    if profile.size == 0:
        return 0.0
    lo, hi = threshold * 0.4, min(1.0, threshold * 1.6)
    ambiguous = np.count_nonzero((profile > lo) & (profile < hi))
    return float(np.clip(1.0 - (ambiguous / profile.size) * 4.0, 0.0, 1.0))


@dataclass(slots=True)
class ColorProbe:
    """Share of the region matching a colour, optionally thresholded to a bool.

    Used for indicator lights: in-combat marker, buff icon presence, cast bar visibility.
    """

    name: str
    region: RelRect
    color: RGB
    tolerance: int = 30
    threshold: float | None = None  # set to return a bool instead of a fraction

    def read(self, crop: np.ndarray) -> ProbeReading:
        if crop.size == 0:
            return ProbeReading(None, 0.0)
        ratio = float(_color_mask(crop, self.color, self.tolerance).mean())
        if self.threshold is None:
            return ProbeReading(round(ratio, 4), 1.0)
        # Confidence scales with distance from the decision boundary: a reading that
        # lands right on the threshold is exactly the one not to act on.
        margin = abs(ratio - self.threshold) / max(self.threshold, 1e-6)
        return ProbeReading(ratio >= self.threshold, float(np.clip(margin * 2.0, 0.0, 1.0)))


@dataclass(slots=True)
class PresenceProbe:
    """Whether a region contains anything, by contrast against its own background.

    Cheaper and far more robust than a template match for "is the target frame showing at
    all" — it does not care what the target is, only that the panel is populated.
    """

    name: str
    region: RelRect
    min_std: float = 12.0

    def read(self, crop: np.ndarray) -> ProbeReading:
        if crop.size == 0:
            return ProbeReading(None, 0.0)
        spread = float(crop.astype(np.float32).std())
        present = spread >= self.min_std
        margin = abs(spread - self.min_std) / self.min_std
        return ProbeReading(present, float(np.clip(margin, 0.0, 1.0)))


@dataclass(slots=True)
class TemplateProbe:
    """Normalised cross-correlation against a reference patch.

    Returns the best match score as a float, or a bool once thresholded. Used for status
    icons, where the icon art is the only reliable identifier.
    """

    name: str
    region: RelRect
    template: np.ndarray  # (h, w, 3) uint8
    threshold: float | None = 0.8

    def read(self, crop: np.ndarray) -> ProbeReading:
        if crop.size == 0 or self.template.size == 0:
            return ProbeReading(None, 0.0)
        th, tw = self.template.shape[:2]
        ch, cw = crop.shape[:2]
        if ch < th or cw < tw:
            return ProbeReading(None, 0.0)

        score = _best_ncc(crop, self.template)
        if self.threshold is None:
            return ProbeReading(round(score, 4), 1.0)
        return ProbeReading(score >= self.threshold, float(np.clip(abs(score - self.threshold) * 3, 0, 1)))


def _best_ncc(crop: np.ndarray, template: np.ndarray) -> float:
    """Best zero-mean normalised cross-correlation over all offsets.

    Written as an explicit sliding window rather than an FFT: HUD icons are ~40px and the
    search region is barely larger, so the direct form is both faster and easier to be
    confident about than a convolution with padding to reason through.
    """
    c = crop.astype(np.float32)
    t = template.astype(np.float32)
    th, tw = t.shape[:2]
    ch, cw = c.shape[:2]

    t_centered = t - t.mean()
    t_norm = float(np.sqrt((t_centered**2).sum()))
    if t_norm < 1e-6:
        return 0.0

    best = -1.0
    for y in range(ch - th + 1):
        for x in range(cw - tw + 1):
            window = c[y : y + th, x : x + tw]
            w_centered = window - window.mean()
            w_norm = float(np.sqrt((w_centered**2).sum()))
            if w_norm < 1e-6:
                continue
            score = float((w_centered * t_centered).sum() / (w_norm * t_norm))
            if score > best:
                best = score
                if best > 0.995:  # good enough; stop paying for the rest of the sweep
                    return best
    return max(best, 0.0)


@dataclass(slots=True)
class CooldownRingProbe:
    """How much of an action's recast remains, from the darkened cooldown overlay.

    FFXIV draws a recast as a darkening overlay that sweeps around the icon. The share of
    darkened pixels is therefore proportional to remaining recast — so the probe reports
    *progress*, a unitless 0–1, and the ability definition supplies the recast duration to
    turn it into seconds. Keeping the constant in the ability rather than the probe is
    what lets one probe implementation serve every slot on the hotbar.

    `progress` is 0.0 when ready and approaches 1.0 just after use.
    """

    name: str
    region: RelRect
    dark_threshold: int = 70  # luminance below this counts as overlaid
    ready_epsilon: float = 0.06

    def read(self, crop: np.ndarray) -> ProbeReading:
        if crop.size == 0:
            return ProbeReading(None, 0.0)
        # Rec. 601 luma; the overlay darkens uniformly so any sane weighting works, but
        # matching the game's own gamma assumptions keeps the threshold stable.
        luma = crop[..., 0] * 0.299 + crop[..., 1] * 0.587 + crop[..., 2] * 0.114
        dark_ratio = float((luma < self.dark_threshold).mean())

        # An icon that is simply dark art, not overlaid, reads as permanently on
        # cooldown. Bimodality separates the two: an overlay creates two clean
        # populations, dark art creates one.
        bright_ratio = float((luma > self.dark_threshold + 60).mean())
        confidence = float(np.clip(dark_ratio + bright_ratio, 0.0, 1.0))

        return ProbeReading(round(dark_ratio, 4), confidence)

    def is_ready(self, progress: float) -> bool:
        return progress <= self.ready_epsilon


@dataclass(slots=True)
class DigitProbe:
    """Reads a small integer by matching each glyph position against a digit template set.

    Used for buff timers. Deliberately not a general OCR: the glyph set is fixed, the
    positions are fixed, and a purpose-built matcher is both faster and more accurate
    here than anything general.
    """

    name: str
    region: RelRect
    glyphs: dict[str, np.ndarray]
    max_digits: int = 2
    min_score: float = 0.7

    def read(self, crop: np.ndarray) -> ProbeReading:
        if crop.size == 0 or not self.glyphs:
            return ProbeReading(None, 0.0)
        h, w = crop.shape[:2]
        cell_w = max(1, w // self.max_digits)

        digits: list[str] = []
        scores: list[float] = []
        for i in range(self.max_digits):
            cell = crop[:, i * cell_w : (i + 1) * cell_w]
            if cell.size == 0:
                continue
            best_char, best_score = "", 0.0
            for char, glyph in self.glyphs.items():
                if cell.shape[0] < glyph.shape[0] or cell.shape[1] < glyph.shape[1]:
                    continue
                score = _best_ncc(cell, glyph)
                if score > best_score:
                    best_char, best_score = char, score
            if best_score >= self.min_score:
                digits.append(best_char)
                scores.append(best_score)

        if not digits:
            return ProbeReading(None, 0.0)
        try:
            return ProbeReading(int("".join(digits)), float(np.mean(scores)))
        except ValueError:
            return ProbeReading(None, 0.0)
