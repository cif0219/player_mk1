"""Monotonic time and per-stage latency accounting.

Everything in the player agrees on one clock: `time.perf_counter()`, in seconds, as a
float. Wall-clock time appears exactly once, in the session recorder's metadata, and
never in a decision.

The `Stopwatch` exists because a latency budget you do not measure is a latency budget
you do not have. Each pipeline stage times itself into a shared registry and the CLI
prints the distribution on exit.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


def now() -> float:
    """Monotonic seconds. The only time source any decision may consult."""
    return time.perf_counter()


def ms_since(t: float) -> float:
    return (now() - t) * 1000.0


@dataclass
class Histogram:
    """Fixed-capacity reservoir of latency samples, in milliseconds.

    Keeps the most recent `capacity` samples rather than an exact distribution: we care
    about "is the loop healthy now", not about the tail from ten minutes ago.
    """

    name: str
    capacity: int = 2048
    _samples: list[float] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    count: int = 0
    worst_ms: float = 0.0

    def add(self, value_ms: float) -> None:
        with self._lock:
            self.count += 1
            self.worst_ms = max(self.worst_ms, value_ms)
            if len(self._samples) >= self.capacity:
                self._samples.pop(0)
            self._samples.append(value_ms)

    def percentile(self, p: float) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            ordered = sorted(self._samples)
        # Nearest-rank; exact interpolation buys nothing at this sample count.
        idx = min(len(ordered) - 1, max(0, math.ceil(p / 100.0 * len(ordered)) - 1))
        return ordered[idx]

    def summary(self) -> str:
        return (
            f"{self.name}: n={self.count} "
            f"p50={self.percentile(50):.1f}ms "
            f"p95={self.percentile(95):.1f}ms "
            f"p99={self.percentile(99):.1f}ms "
            f"max={self.worst_ms:.1f}ms"
        )


class LatencyBudget:
    """Registry of per-stage histograms.

    One instance is shared by the whole runtime. `measure` is the only entry point, so a
    stage that forgets to register still shows up.
    """

    def __init__(self) -> None:
        self._stages: dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def histogram(self, stage: str) -> Histogram:
        with self._lock:
            hist = self._stages.get(stage)
            if hist is None:
                hist = Histogram(stage)
                self._stages[stage] = hist
            return hist

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = now()
        try:
            yield
        finally:
            self.histogram(stage).add((now() - start) * 1000.0)

    def record(self, stage: str, value_ms: float) -> None:
        """Record a span timed elsewhere — e.g. capture-to-dispatch across threads."""
        self.histogram(stage).add(value_ms)

    def report(self) -> list[str]:
        with self._lock:
            stages = list(self._stages.values())
        return [h.summary() for h in sorted(stages, key=lambda h: h.name)]


class RateLimiter:
    """Sliding-window rate limiter over a one-second window.

    Used by the dispatch rate guard. Kept here rather than in `safety` because the
    timeline also needs it and `safety` importing `act` would invert the dependency.
    """

    def __init__(self, max_per_sec: int) -> None:
        self.max_per_sec = max_per_sec
        self._events: list[float] = []
        self._lock = threading.Lock()

    def allow(self, at: float | None = None) -> bool:
        at = now() if at is None else at
        with self._lock:
            cutoff = at - 1.0
            # Events are appended in time order, so trimming from the front is enough.
            while self._events and self._events[0] < cutoff:
                self._events.pop(0)
            if len(self._events) >= self.max_per_sec:
                return False
            self._events.append(at)
            return True

    def count(self, at: float | None = None) -> int:
        """Events still inside the window ending at `at` (trims the expired ones)."""
        at = now() if at is None else at
        with self._lock:
            cutoff = at - 1.0
            while self._events and self._events[0] < cutoff:
                self._events.pop(0)
            return len(self._events)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
