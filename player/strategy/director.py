"""The Claude director.

Runs on an **event trigger**, not a fixed interval. A 5-second timer spends tokens
describing a striking dummy that has not changed since the last five-second timer. The
triggers that matter are transitions: combat started or ended, an objective stopped
making progress, a state nothing knows how to handle.

Three properties keep it cheap and keep it safe:

*Schema-enforced.* Responses come back through `output_config.format` against the
`DirectiveBatch` JSON Schema, so a malformed response fails validation instead of being
half-parsed by a regex.

*Budgeted.* A hard per-session output-token ceiling stops the director rather than
quietly overspending. When it stops, everything below it keeps playing — the deterministic
layers never depended on it.

*Confined.* It can only emit directives from `directives.py`, and every directive is
re-validated against the live catalog before it reaches policy. Nothing it can say
dispatches input.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..clock import now
from ..state import WorldState
from .directives import (
    DirectiveBatch,
    EnableReflexGroup,
    Pause,
    SelectPlanProfile,
    SetObjective,
    SetParameter,
    parse_batch,
    response_schema,
    summarise,
)

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You are the strategic layer of a screen-perception game-playing program.

Everything below you is deterministic: perception reads the HUD, a priority list picks
abilities, and reflex rules handle telegraphed damage. You do not press keys and cannot
press keys. You choose which rotation profile is active, set objectives, toggle reflex
groups, and adjust declared parameters.

You are given a compact snapshot of understood game state and, sometimes, a downscaled
screenshot. Read them and decide whether anything should change.

Principles:
- Prefer no change. An empty directive list is the correct answer most of the time.
- Only use profile ids, parameter keys, and reflex group names from the lists provided.
  Anything else is rejected.
- If the situation is one you do not understand, pause rather than guessing.
- Set next_review_s longer when things are stable. Reviewing a static situation
  repeatedly costs money and tells you nothing new.
"""


@dataclass(slots=True)
class DirectorConfig:
    enabled: bool = False
    model: str = MODEL
    max_tokens: int = 2048
    effort: str = "low"  # this is triage, not deep reasoning; escalate per-call if needed
    min_interval_s: float = 5.0
    screenshot_every_n_calls: int = 3
    screenshot_max_width: int = 640
    # Hard session ceiling. Reaching it stops the director; play continues without it.
    max_output_tokens_per_session: int = 60_000
    request_timeout_s: float = 30.0
    # "anthropic" (default) or "openai" — the latter speaks the OpenAI
    # chat-completions dialect against `base_url`, which is how a locally
    # served SLM (llama.cpp, vllm, ollama, Hermes) becomes the director:
    # same directive vocabulary, same budget, zero API cost.
    provider: str = "anthropic"
    base_url: str = ""      # e.g. "http://host0:8000/v1"
    api_key: str = ""       # most local servers ignore it; sent if set
    # Reasoning-tuned SLMs (Qwen3.x et al) burn the whole budget thinking
    # unless told not to. Sends /no_think + chat_template_kwargs.
    disable_thinking: bool = True


@dataclass(slots=True)
class DirectorContext:
    """What the director is allowed to know and allowed to change.

    Passing the catalogs in explicitly, rather than letting the director introspect the
    runtime, is what makes "only ids from the list" enforceable rather than aspirational.
    """

    objective: str = ""
    profile_ids: tuple[str, ...] = ()
    active_profile: str = ""
    reflex_groups: tuple[str, ...] = ()
    enabled_reflex_groups: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    state_fields: tuple[str, ...] = ()
    recent_events: tuple[str, ...] = ()


class Director:
    """Event-triggered strategic advisor. Never blocks the play loop."""

    def __init__(
        self,
        config: DirectorConfig,
        apply: Callable[[Any], bool],
        client: Any | None = None,
    ) -> None:
        self.config = config
        # Applying is the runtime's job: it owns the catalogs and does the final
        # validation, so a directive that names something unknown dies there.
        self.apply = apply
        self._client = client
        self._lock = threading.Lock()
        self._trigger = threading.Event()
        self._pending_reason = ""

        self.calls = 0
        self.output_tokens = 0
        self.input_tokens = 0
        self.rejected = 0
        self.last_error: str | None = None
        self.last_analysis: str = ""
        self.last_call_at: float = 0.0
        self.next_review_s: float = config.min_interval_s
        self.stopped_reason: str | None = None

    # -- availability ------------------------------------------------------------

    @property
    def available(self) -> bool:
        if not self.config.enabled or self.stopped_reason:
            return False
        if self.config.provider == "openai":
            return bool(self.config.base_url)  # urllib only; nothing to install
        return self._client is not None or _anthropic_installed()

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: the whole player runs without the llm extra

            self._client = anthropic.Anthropic(timeout=self.config.request_timeout_s)
        return self._client

    @property
    def budget_remaining(self) -> int:
        return max(0, self.config.max_output_tokens_per_session - self.output_tokens)

    # -- triggering --------------------------------------------------------------

    def request_review(self, reason: str) -> None:
        """Ask for a review at the next opportunity. Cheap, idempotent, non-blocking."""
        with self._lock:
            self._pending_reason = reason
        self._trigger.set()

    def should_review(self, at: float | None = None) -> bool:
        at = now() if at is None else at
        if not self.available:
            return False
        if (at - self.last_call_at) < self.config.min_interval_s:
            return False
        if self._trigger.is_set():
            return True
        # Heartbeat: even a stable situation gets checked eventually, at whatever cadence
        # the model itself asked for.
        return (at - self.last_call_at) >= max(self.next_review_s, self.config.min_interval_s)

    # -- the call ----------------------------------------------------------------

    def review(
        self,
        state: WorldState,
        context: DirectorContext,
        frame: np.ndarray | None = None,
    ) -> DirectiveBatch | None:
        if not self.available:
            return None
        if self.budget_remaining <= self.config.max_tokens:
            self.stopped_reason = "token budget exhausted"
            return None

        with self._lock:
            reason = self._pending_reason or "scheduled review"
            self._pending_reason = ""
        self._trigger.clear()
        self.last_call_at = now()

        include_shot = (
            frame is not None
            and self.config.screenshot_every_n_calls > 0
            and self.calls % self.config.screenshot_every_n_calls == 0
        )

        try:
            batch = self._call(state, context, reason, frame if include_shot else None)
        except Exception as exc:
            # A failed director call is not a failed player. Keep the previous objective
            # and let the deterministic layers carry on.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

        if batch is None:
            return None

        self.last_analysis = batch.analysis
        self.next_review_s = max(self.config.min_interval_s, min(batch.next_review_s, 120.0))

        for directive in batch.directives:
            if not self.apply(directive):
                self.rejected += 1
        return batch

    def _call(
        self,
        state: WorldState,
        context: DirectorContext,
        reason: str,
        frame: np.ndarray | None,
    ) -> DirectiveBatch | None:
        if self.config.provider == "openai":
            return self._call_openai(state, context, reason)
        client = self._ensure_client()
        content: list[dict[str, Any]] = []

        if frame is not None:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _encode_jpeg(frame, self.config.screenshot_max_width),
                    },
                }
            )
        content.append({"type": "text", "text": _build_prompt(state, context, reason)})

        response = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            # The system prompt is byte-identical across every call in a session, so it
            # is the whole point of caching here — it is most of the input tokens.
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": response_schema()},
            },
            messages=[{"role": "user", "content": content}],
        )

        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            self.last_error = f"refusal: {getattr(details, 'category', 'unknown')}"
            return None

        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"), None
        )
        if not text:
            self.last_error = "no text block in response"
            return None

        # `output_config.format` guarantees valid JSON matching the schema, so a failure
        # here means the schema and the model disagree — worth surfacing, not swallowing.
        return parse_batch(text)

    def _call_openai(
        self,
        state: WorldState,
        context: DirectorContext,
        reason: str,
    ) -> DirectiveBatch | None:
        """One review via an OpenAI-compatible /chat/completions endpoint.

        Local SLM servers vary in JSON-mode support, so the contract is asked
        for in the prompt and enforced by extraction + `parse_batch` — a
        malformed reply costs one skipped review, never a crash. Screenshots
        are not sent: the local models this path serves are text-only, and the
        state summary is the part that matters.
        """
        import urllib.request

        schema = json.dumps(response_schema())
        system = (
            SYSTEM_PROMPT
            + "\n\nReply with ONLY a JSON object (no prose, no markdown fences) "
            + "matching this JSON Schema:\n"
            + schema
        )
        if self.config.disable_thinking:
            system = "/no_think " + system

        body = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": 0.2,
            # Ignored by servers that don't know it; turns off Qwen-style
            # reasoning on the ones that do.
            "chat_template_kwargs": {"enable_thinking": not self.config.disable_thinking}
            if self.config.disable_thinking
            else {},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _build_prompt(state, context, reason)},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.request_timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        self.calls += 1
        usage = payload.get("usage") or {}
        self.output_tokens += int(usage.get("completion_tokens") or 0)
        self.input_tokens += int(usage.get("prompt_tokens") or 0)

        text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        text = _extract_json_object(text)
        if not text:
            self.last_error = "no JSON object in response"
            return None
        return parse_batch(text)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "available": self.available,
            "calls": self.calls,
            "in_tokens": self.input_tokens,
            "out_tokens": self.output_tokens,
            "budget_left": self.budget_remaining,
            "rejected": self.rejected,
            "stopped": self.stopped_reason,
            "error": self.last_error,
        }


def _build_prompt(state: WorldState, context: DirectorContext, reason: str) -> str:
    """The user turn: current situation plus the exact vocabulary that is legal.

    Listing the catalogs inline is what makes "only ids from the list" a usable
    instruction rather than a hope. It costs tokens and is worth them.
    """
    snapshot = state.to_summary(context.state_fields or None)
    lines = [
        f"Review triggered by: {reason}",
        "",
        f"Current objective: {context.objective or '(none set)'}",
        f"Active rotation profile: {context.active_profile or '(none)'}",
        "",
        "Available rotation profiles:",
        *_bullets(context.profile_ids),
        "",
        "Reflex groups (* = currently enabled):",
        *_bullets(
            f"{g}{'*' if g in context.enabled_reflex_groups else ''}"
            for g in context.reflex_groups
        ),
        "",
        "Declared parameters:",
        *_bullets(f"{k} = {v!r}" for k, v in sorted(context.parameters.items())),
        "",
        "Game state:",
        json.dumps(snapshot, indent=2, sort_keys=True),
        "",
        "Recent events:",
        *_bullets(context.recent_events[-8:]),
    ]
    return "\n".join(lines)


def _bullets(items) -> list[str]:
    """Indented bullet list, with an explicit placeholder when empty.

    An empty section renders as nothing at all otherwise, which reads to the model as a
    missing section rather than an empty one — and "there are no reflex groups" is a
    materially different fact from "I was not told about reflex groups".
    """
    rendered = [f"  - {item}" for item in items]
    return rendered or ["  (none)"]


def _encode_jpeg(image: np.ndarray, max_width: int) -> str:
    """Downscale and JPEG-encode a frame for the vision input.

    Full-resolution frames are the most expensive thing the director could send and the
    least useful: it is judging the situation, not reading the cooldown timers, which the
    state snapshot already contains far more accurately than an image would.
    """
    from PIL import Image

    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = image[:: max(1, int(1 / scale)), :: max(1, int(1 / scale))]
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json_object(text: str) -> str:
    """The outermost {...} in a possibly chatty reply.

    Local models fenced-code or preamble their JSON often enough that strict
    parsing throws away good directives. Slicing from the first '{' to the
    last '}' is crude but `parse_batch` still validates every byte of it.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _anthropic_installed() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = [
    "Director",
    "DirectorConfig",
    "DirectorContext",
    "EnableReflexGroup",
    "Pause",
    "SelectPlanProfile",
    "SetObjective",
    "SetParameter",
    "summarise",
]
