"""A small declarative condition language over `WorldState`.

Rotations are mostly "use X when Y" and Y is nearly always a comparison against a field
or a conjunction of them. Expressing those as data rather than as Python closures buys
three things: a job profile becomes a YAML file a non-programmer can edit, conditions
serialise into the session trace so you can see *why* an ability fired, and the LLM
director can be handed a vocabulary it cannot escape from.

Deliberately not a general expression evaluator. `eval` on config is how config becomes
code execution, and the extra power buys nothing a rotation needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..state import WorldState

Op = Literal["<", "<=", ">", ">=", "==", "!=", "truthy", "falsy"]

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass(frozen=True, slots=True)
class Condition:
    """A predicate over `WorldState`, closed under and/or/not.

    Exactly one of `field_name` (leaf) or `children` (branch) is meaningful, decided by
    `kind`.
    """

    kind: Literal["leaf", "all", "any", "not", "const"] = "const"
    field_name: str = ""
    op: Op = "truthy"
    value: Any = None
    children: tuple["Condition", ...] = ()
    const: bool = True
    # A leaf whose field is missing or untrusted evaluates to this. Defaulting to False
    # means an unreadable HUD makes the player do less, never more.
    on_missing: bool = False
    min_confidence: float = 0.5

    # -- constructors ------------------------------------------------------------

    @staticmethod
    def field(name: str, op: Op, value: Any = None, **kw) -> "Condition":
        return Condition(kind="leaf", field_name=name, op=op, value=value, **kw)

    @staticmethod
    def truthy(name: str, **kw) -> "Condition":
        return Condition(kind="leaf", field_name=name, op="truthy", **kw)

    @staticmethod
    def falsy(name: str, **kw) -> "Condition":
        return Condition(kind="leaf", field_name=name, op="falsy", **kw)

    @staticmethod
    def all_(*children: "Condition") -> "Condition":
        return Condition(kind="all", children=tuple(children))

    @staticmethod
    def any_(*children: "Condition") -> "Condition":
        return Condition(kind="any", children=tuple(children))

    @staticmethod
    def not_(child: "Condition") -> "Condition":
        return Condition(kind="not", children=(child,))

    @staticmethod
    def always() -> "Condition":
        return Condition(kind="const", const=True)

    @staticmethod
    def never() -> "Condition":
        return Condition(kind="const", const=False)

    # -- sugar for the common FFXIV shapes ---------------------------------------

    @staticmethod
    def buff(buff_id: str) -> "Condition":
        return Condition.truthy(f"buff.{buff_id}.active")

    @staticmethod
    def no_buff(buff_id: str) -> "Condition":
        return Condition.falsy(f"buff.{buff_id}.active")

    @staticmethod
    def ready(action_id: str) -> "Condition":
        return Condition.truthy(f"action.{action_id}.ready")

    @staticmethod
    def hp_below(fraction: float) -> "Condition":
        return Condition.field("player.hp_frac", "<", fraction)

    @staticmethod
    def mp_below(fraction: float) -> "Condition":
        return Condition.field("player.mp_frac", "<", fraction)

    # -- evaluation --------------------------------------------------------------

    def __call__(self, state: WorldState) -> bool:
        return self.evaluate(state)

    def evaluate(self, state: WorldState) -> bool:
        if self.kind == "const":
            return self.const
        if self.kind == "all":
            return all(c.evaluate(state) for c in self.children)
        if self.kind == "any":
            return any(c.evaluate(state) for c in self.children)
        if self.kind == "not":
            return not self.children[0].evaluate(state)

        f = state.field(self.field_name)
        if f is None or f.value is None or f.confidence < self.min_confidence:
            return self.on_missing

        if self.op == "truthy":
            return bool(f.value)
        if self.op == "falsy":
            return not bool(f.value)

        compare = _OPS.get(self.op)
        if compare is None:
            return self.on_missing
        try:
            return bool(compare(f.value, self.value))
        except TypeError:
            # Comparing a string field against a number, usually a config typo.
            return self.on_missing

    def describe(self) -> str:
        """Human-readable form, written into the trace so decisions are explainable."""
        if self.kind == "const":
            return "always" if self.const else "never"
        if self.kind == "not":
            return f"not({self.children[0].describe()})"
        if self.kind in ("all", "any"):
            joiner = " and " if self.kind == "all" else " or "
            return "(" + joiner.join(c.describe() for c in self.children) + ")"
        if self.op == "truthy":
            return self.field_name
        if self.op == "falsy":
            return f"!{self.field_name}"
        return f"{self.field_name} {self.op} {self.value}"

    def fields(self) -> set[str]:
        """Every field this condition reads. Used to validate a profile at startup."""
        if self.kind == "leaf":
            return {self.field_name}
        out: set[str] = set()
        for child in self.children:
            out |= child.fields()
        return out


def parse_condition(spec: Any) -> Condition:
    """Build a `Condition` from YAML/JSON.

    Accepted forms::

        true / false                       -> always / never
        "player.in_combat"                 -> truthy
        "!player.casting"                  -> falsy
        {field: player.hp_frac, op: "<", value: 0.3}
        {all: [ ... ]} / {any: [ ... ]} / {not: { ... }}
        {buff: firestarter} / {ready: fire4}
    """
    if spec is None:
        return Condition.always()
    if isinstance(spec, bool):
        return Condition.always() if spec else Condition.never()
    if isinstance(spec, Condition):
        return spec
    if isinstance(spec, str):
        text = spec.strip()
        if text.startswith("!"):
            return Condition.falsy(text[1:])
        return Condition.truthy(text)
    if isinstance(spec, (list, tuple)):
        return Condition.all_(*(parse_condition(s) for s in spec))

    if not isinstance(spec, dict):
        raise ValueError(f"cannot parse condition from {spec!r}")

    if "all" in spec:
        return Condition.all_(*(parse_condition(s) for s in spec["all"]))
    if "any" in spec:
        return Condition.any_(*(parse_condition(s) for s in spec["any"]))
    if "not" in spec:
        return Condition.not_(parse_condition(spec["not"]))
    if "buff" in spec:
        return Condition.buff(str(spec["buff"]))
    if "no_buff" in spec:
        return Condition.no_buff(str(spec["no_buff"]))
    if "ready" in spec:
        return Condition.ready(str(spec["ready"]))
    if "field" in spec:
        return Condition.field(
            str(spec["field"]),
            spec.get("op", "truthy"),
            spec.get("value"),
            on_missing=bool(spec.get("on_missing", False)),
            min_confidence=float(spec.get("min_confidence", 0.5)),
        )
    raise ValueError(f"unrecognised condition keys: {sorted(spec)}")


@dataclass(slots=True)
class ConditionSet:
    """Named conditions, so a profile can define one and reference it several times."""

    named: dict[str, Condition] = field(default_factory=dict)

    def define(self, name: str, condition: Condition) -> None:
        self.named[name] = condition

    def resolve(self, spec: Any) -> Condition:
        if isinstance(spec, str) and spec in self.named:
            return self.named[spec]
        return parse_condition(spec)
