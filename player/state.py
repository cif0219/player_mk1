"""`WorldState` — the contract between perception and policy.

Perception's entire job is to produce one of these per frame. Policy's entire input is
one of these. Nothing downstream of perception ever sees a pixel.

The wrapper type `Field` is the part that earns its keep: a bare `float` cannot tell a
guard that the health bar has not been read for 400ms, or that the template match that
produced it scored 0.31. Both of those are reasons not to act.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .clock import now
from .geometry import Rect

FieldValue = float | int | bool | str | None


@dataclass(frozen=True, slots=True)
class Field:
    """A single sensor reading, with enough metadata to decide whether to trust it."""

    value: FieldValue
    confidence: float = 1.0
    source: str = ""
    updated_at: float = 0.0

    def is_stale(self, at: float, max_age_s: float) -> bool:
        return (at - self.updated_at) > max_age_s

    def age_ms(self, at: float | None = None) -> float:
        return ((now() if at is None else at) - self.updated_at) * 1000.0

    def trusted(self, min_confidence: float = 0.5) -> bool:
        return self.value is not None and self.confidence >= min_confidence


@dataclass(frozen=True, slots=True)
class Entity:
    """A detector hit. Position is frame space; convert via `Geometry` to act on it."""

    kind: str
    bbox: Rect
    confidence: float = 1.0
    attrs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        return self.bbox.center


@dataclass(slots=True)
class WorldState:
    """One frame's worth of understood game state.

    `captured_at` is when the pixels were grabbed; `perceived_at` is when assembly
    finished. Staleness is measured against `captured_at` — the age that matters is how
    old the *information* is, not how recently we finished thinking about it.
    """

    tick: int = 0
    captured_at: float = 0.0
    perceived_at: float = 0.0
    fields: dict[str, Field] = field(default_factory=dict)
    entities: list[Entity] = field(default_factory=list)

    # -- reads -------------------------------------------------------------------

    def field(self, name: str) -> Field | None:
        """The wrapper, for callers that care about confidence or age."""
        return self.fields.get(name)

    def get(self, name: str, default: FieldValue = None) -> FieldValue:
        """The value, for callers that do not. Missing and `None` collapse together."""
        f = self.fields.get(name)
        if f is None or f.value is None:
            return default
        return f.value

    def num(self, name: str, default: float = 0.0) -> float:
        value = self.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return default

    def flag(self, name: str, default: bool = False) -> bool:
        value = self.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def text(self, name: str, default: str = "") -> str:
        value = self.get(name)
        return value if isinstance(value, str) else default

    def entities_of(self, kind: str) -> list[Entity]:
        return [e for e in self.entities if e.kind == kind]

    # -- writes ------------------------------------------------------------------

    def set(
        self,
        name: str,
        value: FieldValue,
        *,
        confidence: float = 1.0,
        source: str = "",
        at: float | None = None,
    ) -> None:
        self.fields[name] = Field(
            value=value,
            confidence=confidence,
            source=source,
            updated_at=self.captured_at if at is None else at,
        )

    # -- health ------------------------------------------------------------------

    def age_ms(self, at: float | None = None) -> float:
        return ((now() if at is None else at) - self.captured_at) * 1000.0

    def missing(self, required: Iterable[str]) -> list[str]:
        """Required names that are absent or hold `None`. Checked once at startup."""
        return [n for n in required if self.fields.get(n) is None or self.fields[n].value is None]

    def untrusted(self, required: Iterable[str], min_confidence: float = 0.5) -> list[str]:
        out = []
        for name in required:
            f = self.fields.get(name)
            if f is None or not f.trusted(min_confidence):
                out.append(name)
        return out

    def to_summary(self, include: Iterable[str] | None = None) -> dict[str, Any]:
        """Compact dict for logging and for the LLM director's context.

        Deliberately drops confidence and timestamps: the director reasons about the
        situation, not about sensor health, and every token spent on metadata is a token
        not spent on the situation.
        """
        names = list(include) if include is not None else sorted(self.fields)
        out: dict[str, Any] = {}
        for name in names:
            f = self.fields.get(name)
            if f is not None and f.value is not None:
                out[name] = round(f.value, 3) if isinstance(f.value, float) else f.value
        if self.entities:
            counts: dict[str, int] = {}
            for e in self.entities:
                counts[e.kind] = counts.get(e.kind, 0) + 1
            out["_entities"] = counts
        return out

    def copy(self) -> "WorldState":
        return WorldState(
            tick=self.tick,
            captured_at=self.captured_at,
            perceived_at=self.perceived_at,
            fields=dict(self.fields),
            entities=list(self.entities),
        )
