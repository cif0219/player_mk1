"""Reading the boss cast bar.

The instinct is OCR, and OCR is the wrong tool. Stylised fonts, variable widths, a
translucent bar over a moving 3D scene, and a hard latency budget — general text
recognition is both the hardest option and the least reliable one here.

But the problem is not "what does this say". An `EncounterScript` **declares every cast
name it cares about**, so the real question is "which of these twelve known casts is this,
or none of them". That is closed-set classification, which is a far easier problem and one
that can be solved with a signature comparison instead of a recognition model.

The signature is a normalised column profile of the binarised text: how much ink stands in
each horizontal slice of the bar. It is cheap, robust to small rendering differences and
to the bar's own fill sweeping underneath, and discriminative enough to separate a dozen
names — because cast names differ in length and letter distribution far more than they
resemble each other.

Signatures are **learned from recordings**, not shipped. `tools/learn_casts.py` extracts
them from a session where the casts are known, which means calibration is a replay rather
than a data-entry exercise. With no signatures registered the reader honestly returns
nothing, and `CastTrigger` falls through to the timeline trigger.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from player.geometry import Geometry, RelRect
from player.state import WorldState

SIGNATURE_BINS = 48


@dataclass(frozen=True, slots=True)
class CastSignature:
    """A fingerprint of one cast name's rendered text."""

    name: str
    profile: tuple[float, ...]
    samples: int = 1

    def similarity(self, other: tuple[float, ...]) -> float:
        """Cosine similarity in [0, 1]. Scale-invariant, so bar brightness does not matter."""
        if len(other) != len(self.profile):
            return 0.0
        a = np.asarray(self.profile, dtype=np.float32)
        b = np.asarray(other, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        return float(max(0.0, np.dot(a, b) / denom))


def signature_of(crop: np.ndarray, bins: int = SIGNATURE_BINS) -> tuple[float, ...] | None:
    """Column-ink profile of the text in a cast-bar crop.

    Binarised against a threshold derived from the crop itself rather than a constant:
    the cast bar's own fill sweeps left to right underneath the text, so a fixed threshold
    would read a different amount of "ink" at the start and end of the same cast.
    """
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < bins:
        return None

    luma = crop[..., 0] * 0.299 + crop[..., 1] * 0.587 + crop[..., 2] * 0.114

    # Text is the bright minority against the bar. Otsu-ish split: threshold between the
    # median and the maximum, which separates glyphs from background without assuming
    # either one's absolute level.
    median = float(np.median(luma))
    peak = float(luma.max())
    if peak - median < 25.0:
        return None  # nothing bright enough to be text — the bar is empty
    threshold = median + (peak - median) * 0.55
    ink = (luma > threshold).astype(np.float32)

    columns = ink.mean(axis=0)
    # Resample to a fixed width so bars of different pixel widths compare directly. This
    # is what makes one learned signature work across resolutions and UI scales.
    indices = np.linspace(0, columns.size - 1, bins)
    resampled = np.interp(indices, np.arange(columns.size), columns)

    total = float(resampled.sum())
    if total < 0.5:
        return None
    return tuple(float(v) for v in resampled)


@dataclass(slots=True)
class CastLibrary:
    """The known casts for a loaded encounter."""

    signatures: dict[str, CastSignature] = field(default_factory=dict)
    min_similarity: float = 0.90
    # A match is only trusted if it beats the runner-up by this much. Two casts with
    # similar-length names produce similar profiles, and a confident wrong answer here
    # fires the wrong mechanic.
    min_margin: float = 0.04

    def add(self, name: str, profile: tuple[float, ...]) -> None:
        existing = self.signatures.get(name)
        if existing is None:
            self.signatures[name] = CastSignature(name=name, profile=profile)
            return
        # Average across samples so a signature learned from many frames is more stable
        # than one learned from a single lucky capture.
        n = existing.samples
        blended = tuple(
            (old * n + new) / (n + 1) for old, new in zip(existing.profile, profile)
        )
        self.signatures[name] = CastSignature(name=name, profile=blended, samples=n + 1)

    def match(self, profile: tuple[float, ...]) -> tuple[str, float] | None:
        if not self.signatures:
            return None
        scored = sorted(
            ((sig.similarity(profile), name) for name, sig in self.signatures.items()),
            reverse=True,
        )
        best_score, best_name = scored[0]
        if best_score < self.min_similarity:
            return None
        if len(scored) > 1 and (best_score - scored[1][0]) < self.min_margin:
            # Two candidates too close to call. Reporting nothing lets the timeline
            # trigger carry, which is late but correct; guessing fires a mechanic that
            # resolves to the wrong place.
            return None
        return best_name, best_score

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    name: {"profile": list(sig.profile), "samples": sig.samples}
                    for name, sig in self.signatures.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CastLibrary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        library = cls()
        for name, entry in data.items():
            library.signatures[name] = CastSignature(
                name=name,
                profile=tuple(entry["profile"]),
                samples=int(entry.get("samples", 1)),
            )
        return library

    @property
    def known(self) -> set[str]:
        return set(self.signatures)


class CastBarSensor:
    """Writes `boss.cast_name` and `boss.cast_progress`.

    The name field is what `CastTrigger` reads, and a named cast is also the encounter
    clock's resync point — so this sensor is load-bearing for both *which* mechanic fires
    and *when* every later one does.
    """

    provides = ("boss.cast_name", "boss.cast_progress", "boss.casting")
    cadence = 1  # a cast bar can appear and matter within a couple of frames

    def __init__(
        self,
        text_region: RelRect,
        bar_region: RelRect,
        library: CastLibrary | None = None,
        name: str = "castbar",
    ) -> None:
        self.name = name
        self.text_region = text_region
        self.bar_region = bar_region
        self.library = library or CastLibrary()
        self.reads = 0
        self.matches = 0
        self.unmatched = 0
        self.last_name = ""

    @property
    def calibrated(self) -> bool:
        return bool(self.library.signatures)

    def observe(self, frame, geo: Geometry, state: WorldState) -> None:
        text_rect = geo.region_to_frame(self.text_region)
        crop = frame.crop(text_rect)
        profile = signature_of(crop)

        if profile is None:
            state.set("boss.casting", False, confidence=0.9, source="castbar")
            state.set("boss.cast_name", "", confidence=0.9, source="castbar")
            state.set("boss.cast_progress", None, confidence=0.0, source="castbar")
            self.last_name = ""
            return

        self.reads += 1
        state.set("boss.casting", True, confidence=0.9, source="castbar")

        matched = self.library.match(profile)
        if matched is None:
            self.unmatched += 1
            # A cast is happening but it is not one we know. Reporting the fact without a
            # name is honest, and lets a policy decide to be careful rather than assuming
            # nothing is happening.
            state.set("boss.cast_name", "", confidence=0.3, source="castbar:unmatched")
        else:
            name, score = matched
            self.matches += 1
            self.last_name = name
            state.set("boss.cast_name", name, confidence=round(score, 3), source="castbar")

        bar_rect = geo.region_to_frame(self.bar_region)
        state.set(
            "boss.cast_progress",
            _fill_fraction(frame.crop(bar_rect)),
            confidence=0.8,
            source="castbar",
        )

    def stats(self) -> dict[str, object]:
        return {
            "known_casts": len(self.library.signatures),
            "reads": self.reads,
            "matched": self.matches,
            "unmatched": self.unmatched,
            "last": self.last_name,
        }


def _fill_fraction(crop: np.ndarray) -> float | None:
    """How far the cast bar has filled, from the brightness boundary along its length."""
    if crop.size == 0:
        return None
    luma = crop[..., 0] * 0.299 + crop[..., 1] * 0.587 + crop[..., 2] * 0.114
    columns = luma.mean(axis=0)
    if columns.size < 4:
        return None
    threshold = (float(columns.min()) + float(columns.max())) / 2.0
    filled = np.flatnonzero(columns > threshold)
    if filled.size == 0:
        return 0.0
    return round(float(int(filled[-1]) + 1) / columns.size, 3)


def learn_from_session(
    session_root: str | Path,
    text_region: RelRect,
    labels: dict[int, str],
    library: CastLibrary | None = None,
) -> CastLibrary:
    """Build signatures from a recording where the casts are known.

    `labels` maps frame index to cast name. Producing it is the calibration step, and it
    is deliberately a replay rather than a data-entry exercise: scrub a recorded pull, note
    which frames show which cast, and the signatures fall out.
    """
    from player.geometry import Rect
    from player.record.session import Session

    library = library or CastLibrary()
    session = Session(session_root)

    for index, _t, image in session.frames():
        name = labels.get(index)
        if not name:
            continue
        h, w = image.shape[:2]
        geo = Geometry(Rect(0, 0, w, h), (w, h))
        profile = signature_of(image[
            geo.region_to_frame(text_region).y : geo.region_to_frame(text_region).bottom,
            geo.region_to_frame(text_region).x : geo.region_to_frame(text_region).right,
        ])
        if profile is not None:
            library.add(name, profile)

    return library
