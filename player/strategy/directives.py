"""What the director is allowed to say.

The vocabulary is deliberately narrow, and the narrowness is the safety property: there
is no directive that means "press this key". The model selects from a registered catalog,
flips declared parameters, and sets objectives. The worst outcome of a confused or
adversarially-prompted model is that the player keeps doing what it was already doing.

Every directive carries a `reason`. Not decoration — it is what makes a session log
readable when the player does something surprising at 3am, and it is the cheapest
possible interpretability mechanism.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    # extra="forbid" renders as additionalProperties:false, which the structured-output
    # schema validator requires — without it the schema is rejected outright.
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(description="One sentence on why, for the session log.")


class SetObjective(_Base):
    """Set the high-level goal. Advisory: it shapes future directives, not this tick."""

    kind: Literal["set_objective"] = "set_objective"
    objective: str = Field(description="What the player should be trying to achieve.")


class SelectPlanProfile(_Base):
    """Switch the active rotation profile.

    `profile_id` must already exist in the registered catalog. An unknown id is rejected
    and logged rather than coerced — a hallucinated profile name must not put the player
    into an undefined state.
    """

    kind: Literal["select_plan_profile"] = "select_plan_profile"
    profile_id: str = Field(description="Id from the catalog listed in the prompt.")


class SetParameter(_Base):
    """Change a declared runtime parameter.

    Only keys the runtime declares are accepted, so this cannot reach arbitrary internals.
    """

    kind: Literal["set_parameter"] = "set_parameter"
    key: str = Field(description="Parameter name from the declared list.")
    value: str = Field(description="New value, as a string. Coerced to the declared type.")


class EnableReflexGroup(_Base):
    """Turn a whole category of reflexes on or off."""

    kind: Literal["enable_reflex_group"] = "enable_reflex_group"
    group: str = Field(description="Reflex group name from the list in the prompt.")
    enabled: bool = Field(description="True to enable, false to disable.")


class Pause(_Base):
    """Stop acting. The escape hatch when the model does not understand the situation.

    Included deliberately: a model that can only choose between actions will choose an
    action. Giving it an explicit "I do not know what is happening" option is what makes
    the honest answer available.
    """

    kind: Literal["pause"] = "pause"
    paused: bool = Field(description="True to stop acting, false to resume.")


Directive = Annotated[
    Union[SetObjective, SelectPlanProfile, SetParameter, EnableReflexGroup, Pause],
    Field(discriminator="kind"),
]


class DirectiveBatch(BaseModel):
    """The whole response. One round trip returns a coherent set, not a single change."""

    model_config = ConfigDict(extra="forbid")

    analysis: str = Field(description="Two sentences at most on what is happening.")
    directives: list[Directive] = Field(
        description="Directives to apply now. Empty if nothing should change."
    )
    next_review_s: float = Field(
        description="Seconds until the next review. Larger when the situation is stable."
    )


def response_schema() -> dict[str, Any]:
    """JSON Schema for the structured-output request.

    Pydantic emits `$defs`/`$ref` for the union members, which structured outputs
    supports. What it does not support is numeric/string constraints, so the models above
    deliberately carry none — validation of ranges happens in Python after parsing.
    """
    return DirectiveBatch.model_json_schema()


def parse_batch(payload: str | dict[str, Any]) -> DirectiveBatch:
    """Validate a model response into a `DirectiveBatch`, raising on anything malformed."""
    if isinstance(payload, str):
        return DirectiveBatch.model_validate_json(payload)
    return DirectiveBatch.model_validate(payload)


def summarise(batch: DirectiveBatch) -> str:
    if not batch.directives:
        return f"(no change) {batch.analysis}"
    parts = []
    for d in batch.directives:
        if isinstance(d, SetObjective):
            parts.append(f"objective={d.objective!r}")
        elif isinstance(d, SelectPlanProfile):
            parts.append(f"profile={d.profile_id}")
        elif isinstance(d, SetParameter):
            parts.append(f"{d.key}={d.value}")
        elif isinstance(d, EnableReflexGroup):
            parts.append(f"reflex[{d.group}]={'on' if d.enabled else 'off'}")
        elif isinstance(d, Pause):
            parts.append("pause" if d.paused else "resume")
    return "; ".join(parts)
