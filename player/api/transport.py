"""`ApiTransport` — the pieces a profile hands the runtime for API control.

A `GameProfile` built for API control carries one of these instead of a
window title to capture. The CLI reads it, uses its source and backend in
place of the screen source and SendInput, relaxes the foreground guard (there
is no window), and prints its report after the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .backend import ApiBackend
from .gateway import GatewayClient
from .source import ApiSource


@dataclass(slots=True)
class ApiTransport:
    gateway: GatewayClient
    source: ApiSource
    backend: ApiBackend
    # Extra report lines at the end of a run — a fight ledger, typically.
    report: Callable[[], list[str]] = field(default_factory=lambda: (lambda: []))
    # Switch what the tester is doing (a helper's task, games/fancraft/tasks.py); None when the game has no tasks.
    apply_task: Callable[[str], None] | None = None

    def describe(self) -> str:
        return self.source.describe()


__all__ = ["ApiTransport"]
