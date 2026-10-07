"""Loading and validating configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SafetyConfig:
    kill_key: str = "f12"
    pause_key: str | None = "f11"
    max_actions_per_sec: int = 20
    max_state_age_ms: float = 250.0
    min_confidence: float = 0.5
    require_foreground: bool = True
    takeover_cooldown_ms: float = 1500.0


@dataclass(slots=True)
class CaptureConfig:
    downscale: float = 1.0


@dataclass(slots=True)
class AppConfig:
    game: str = "ffxiv"
    runtime: dict[str, Any] = field(default_factory=dict)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    record: dict[str, Any] = field(default_factory=dict)
    director: dict[str, Any] = field(default_factory=dict)
    game_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        game = data.get("game", "ffxiv")
        return cls(
            game=game,
            runtime=data.get("runtime", {}) or {},
            capture=CaptureConfig(**(data.get("capture", {}) or {})),
            safety=SafetyConfig(**(data.get("safety", {}) or {})),
            record=data.get("record", {}) or {},
            director=data.get("director", {}) or {},
            # Game-specific settings live under a key named for the game, so adding a
            # second game does not require touching this loader.
            game_config=data.get(game, {}) or {},
        )

    def validate(self) -> list[str]:
        problems = []
        if not 0.0 < self.capture.downscale <= 1.0:
            problems.append(f"capture.downscale must be in (0, 1], got {self.capture.downscale}")
        if self.safety.max_actions_per_sec <= 0:
            problems.append("safety.max_actions_per_sec must be positive")
        if self.safety.max_state_age_ms <= 0:
            problems.append("safety.max_state_age_ms must be positive")
        if not 0.0 <= self.safety.min_confidence <= 1.0:
            problems.append("safety.min_confidence must be in [0, 1]")
        return problems
