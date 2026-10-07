"""Finding things at unknown positions.

Probes handle everything at a known rectangle. This module handles what is left: ground
AoE telegraphs anywhere in the 3D field, nameplates, markers. That is the part where a
model earns its cost.

The interesting problem is training data, and `TelegraphSegmenter` is the answer. FFXIV
telegraphs are strongly saturated orange/red ground decals, so HSV segmentation extracts
them from any recorded session with no manual annotation at all. Those weak labels train
the first detector, which then generalises to the partially occluded, edge-of-screen, and
oddly-lit cases segmentation misses. Weak labels bootstrap; the model is the product.

Segmentation is also a usable fallback in its own right, so the reflex layer has
something to work with before any model exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..capture.source import Frame
from ..geometry import Geometry, Rect
from ..state import Entity, WorldState


@dataclass(frozen=True, slots=True)
class Detection:
    kind: str
    bbox: Rect
    confidence: float
    attrs: dict


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV. H in [0,360), S and V in [0,1].

    Hand-rolled rather than pulled from OpenCV: this is the only OpenCV call the project
    would need, and a 300MB dependency for one colour-space conversion is a bad trade.
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    delta = mx - mn

    hue = np.zeros_like(mx)
    nz = delta > 1e-6
    # Piecewise by which channel is the max; np.where keeps it branch-free per pixel.
    idx = (mx == r) & nz
    hue[idx] = (60 * ((g[idx] - b[idx]) / delta[idx])) % 360
    idx = (mx == g) & nz
    hue[idx] = 60 * ((b[idx] - r[idx]) / delta[idx]) + 120
    idx = (mx == b) & nz
    hue[idx] = 60 * ((r[idx] - g[idx]) / delta[idx]) + 240

    sat = np.zeros_like(mx)
    sat[mx > 1e-6] = delta[mx > 1e-6] / mx[mx > 1e-6]
    return np.stack([hue, sat, mx], axis=-1)


class TelegraphSegmenter:
    """HSV segmentation for saturated ground decals.

    Two jobs: a working fallback when no model is loaded, and the weak-label generator
    that produces the first model's training set.

    Defaults target FFXIV's orange/red telegraph palette. `stride` subsamples before
    segmenting — a telegraph is hundreds of pixels across, so full resolution buys
    nothing here and costs the whole frame's worth of HSV conversion.
    """

    def __init__(
        self,
        hue_ranges: tuple[tuple[float, float], ...] = ((0, 40), (340, 360)),
        min_saturation: float = 0.45,
        min_value: float = 0.30,
        min_area_frac: float = 0.0015,
        stride: int = 4,
    ) -> None:
        self.hue_ranges = hue_ranges
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.min_area_frac = min_area_frac
        self.stride = max(1, stride)

    def mask(self, image: np.ndarray) -> np.ndarray:
        small = image[:: self.stride, :: self.stride]
        hsv = rgb_to_hsv(small)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        hue_ok = np.zeros(h.shape, dtype=bool)
        for lo, hi in self.hue_ranges:
            hue_ok |= (h >= lo) & (h <= hi)
        return hue_ok & (s >= self.min_saturation) & (v >= self.min_value)

    def detect(self, image: np.ndarray) -> list[Detection]:
        mask = self.mask(image)
        if not mask.any():
            return []
        h, w = mask.shape
        out: list[Detection] = []
        for comp, area in _connected_components(mask):
            if area / (h * w) < self.min_area_frac:
                continue
            y0, x0, y1, x1 = comp
            bbox = Rect(
                x=x0 * self.stride,
                y=y0 * self.stride,
                w=(x1 - x0 + 1) * self.stride,
                h=(y1 - y0 + 1) * self.stride,
            )
            # Fill ratio separates a solid decal from scattered colour noise. It is a
            # weak signal, which is honest: these are weak labels.
            fill = area / max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
            out.append(
                Detection(
                    kind="telegraph",
                    bbox=bbox,
                    confidence=float(np.clip(fill, 0.0, 1.0)) * 0.7,
                    attrs={"source": "segmenter", "fill": round(fill, 3)},
                )
            )
        return out


def _connected_components(mask: np.ndarray, max_components: int = 32):
    """Bounding boxes of connected true-regions, via iterative flood fill.

    Iterative rather than recursive: a full-screen telegraph is tens of thousands of
    pixels and Python's recursion limit is 1000.
    """
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    results = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx] or len(results) >= max_components:
            continue
        stack = [(int(sy), int(sx))]
        visited[sy, sx] = True
        min_y = max_y = int(sy)
        min_x = max_x = int(sx)
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_x, max_x = min(min_x, x), max(max_x, x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        results.append(((min_y, min_x, max_y, max_x), area))
    return results


class OnnxDetector:
    """ONNX Runtime wrapper for a trained detector.

    Kept behind a soft import: the whole project runs, tests included, with no model and
    no onnxruntime. A missing detector disables the reflexes that need it and says so at
    startup, rather than failing at the first telegraph.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        class_names: tuple[str, ...] = ("telegraph",),
        score_threshold: float = 0.35,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.class_names = class_names
        self.score_threshold = score_threshold
        self._session = None

    @property
    def available(self) -> bool:
        return self._session is not None

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            import onnxruntime  # noqa: PLC0415 - intentionally lazy
        except ImportError:
            return False
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = onnxruntime.InferenceSession(
            str(self.model_path),
            providers=[p for p in providers if p in onnxruntime.get_available_providers()],
        )
        return True

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self._session is None:
            return []
        h, w = image.shape[:2]
        tensor, scale = _letterbox(image, self.input_size)
        name = self._session.get_inputs()[0].name
        raw = self._session.run(None, {name: tensor})[0]
        return _decode_boxes(
            raw,
            scale=scale,
            original=(w, h),
            class_names=self.class_names,
            threshold=self.score_threshold,
        )


def _letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize preserving aspect ratio, pad to `size`, return NCHW float32 and the scale."""
    tw, th = size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    # Nearest-neighbour via index arrays; adequate for detection input and avoids pulling
    # in a resize dependency.
    yi = (np.arange(nh) / scale).astype(np.int32).clip(0, h - 1)
    xi = (np.arange(nw) / scale).astype(np.int32).clip(0, w - 1)
    resized = image[yi][:, xi]
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, scale


def _decode_boxes(raw, scale, original, class_names, threshold) -> list[Detection]:
    """Decode `[N, 6]` rows of `(x1, y1, x2, y2, score, class)` back to frame space."""
    out: list[Detection] = []
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 6:
        return out
    ow, oh = original
    for row in arr:
        score = float(row[4])
        if score < threshold:
            continue
        x1, y1, x2, y2 = (float(v) / scale for v in row[:4])
        cls = int(row[5])
        kind = class_names[cls] if 0 <= cls < len(class_names) else "unknown"
        bbox = Rect(
            x=int(max(0, min(x1, ow))),
            y=int(max(0, min(y1, oh))),
            w=int(max(1, min(x2 - x1, ow))),
            h=int(max(1, min(y2 - y1, oh))),
        )
        out.append(Detection(kind=kind, bbox=bbox, confidence=score, attrs={"source": "onnx"}))
    return out


class DetectorSensor:
    """Writes detector output into `WorldState.entities`.

    Runs at `cadence` rather than every frame. Telegraphs persist for 1–3 seconds, so
    inference at 60Hz spends GPU to re-learn what it already knows. Between runs the
    previous entities are carried forward, aged by their own timestamp.
    """

    provides = ("_entities",)

    def __init__(
        self,
        detector: OnnxDetector | None = None,
        fallback: TelegraphSegmenter | None = None,
        cadence: int = 4,
        name: str = "detector",
    ) -> None:
        self.name = name
        self.detector = detector
        self.fallback = fallback
        self.cadence = max(1, cadence)
        self._last: list[Entity] = []

    @property
    def backend(self) -> str:
        if self.detector is not None and self.detector.available:
            return "onnx"
        return "segmenter" if self.fallback is not None else "none"

    def observe(self, frame: Frame, geo: Geometry, state: WorldState) -> None:
        detections: list[Detection] = []
        if self.detector is not None and self.detector.available:
            detections = self.detector.detect(frame.image)
        elif self.fallback is not None:
            detections = self.fallback.detect(frame.image)

        self._last = [
            Entity(kind=d.kind, bbox=d.bbox, confidence=d.confidence, attrs=d.attrs)
            for d in detections
        ]
        state.entities = list(self._last)
