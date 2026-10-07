"""White Mage — the healer case.

The point of this file is that it contains **no healing machinery**. It is a
`RotationProfile` like any other; heals are abilities whose `target` names an ally and
whose `condition` reads party HP. Triage is list order.

That falls out of one design decision — putting a `TargetSpec` on `Ability` — and it is
worth stating why it matters. The obvious alternative is a separate `HealPolicy` running
beside the rotation, which immediately needs its own GCD arbitration, its own commitment
handling, its own cast-time logic, and a rule for who wins. All of that already exists once.

Two things a healer needs that a DPS does not:

* **Reactive targeting.** `TargetSpec.mouseover_lowest_ally(hp_ceiling=...)` resolves at
  press time. When nobody is below the ceiling it fails to resolve, the ability is
  skipped, and the list falls through to damage. "Do not heal when nobody is hurt" needs
  no code.
* **Emergency preemption.** A tank about to die cannot wait for the next GCD decision, so
  that one lives in the reflex layer, which already preempts.

Heals use **mouseover** rather than party-slot keys, and this is not a stylistic choice. A
slot key changes your hard target, so the next Glare lands on the ally you just healed and
does nothing — you would have to re-target the boss after every heal, costing a keypress
and a window where the target is wrong. Mouseover leaves the boss selected throughout,
which is exactly why it is the standard among FFXIV healers. Where no mouseover locator is
configured the resolver falls back to slot keys and says so in its reason string.

> **Accuracy caveat.** The structure is right; specific ability names, potencies and
> thresholds change with patches. Validate against a current guide. Everything here is
> data, so correcting it is editing a list.
"""

from __future__ import annotations

from player.policy.condition import Condition
from player.policy.reflex import Reflex
from player.policy.rotation import Ability, AbilityKind, RotationProfile
from player.policy.targeting import TargetResolver, TargetSpec
from player.act.timeline import Plan, Priority
from player.geometry import ScreenPoint

DEFAULT_KEYS: dict[str, str] = {
    "glare": "1",
    "dia": "2",
    "cure2": "3",
    "medica2": "4",
    "benediction": "5",
    "tetragrammaton": "6",
    "afflatus_solace": "7",
    "afflatus_rapture": "8",
    "assize": "9",
    "lucid_dreaming": "0",
}

DEFAULT_SLOTS: dict[int, str] = {
    0: "glare",
    1: "dia",
    2: "cure2",
    3: "medica2",
    4: "benediction",
    5: "tetragrammaton",
    6: "afflatus_solace",
    7: "afflatus_rapture",
    8: "assize",
    9: "lucid_dreaming",
}

RECASTS: dict[str, float] = {
    "glare": 2.5,
    "dia": 2.5,
    "cure2": 2.5,
    "medica2": 2.5,
    "benediction": 180.0,
    "tetragrammaton": 60.0,
    "afflatus_solace": 2.5,
    "afflatus_rapture": 2.5,
    "assize": 40.0,
    "lucid_dreaming": 60.0,
}

GCD_REFERENCE = "glare"
STATUS_IDS: tuple[str, ...] = ("dia_dot", "lily", "blood_lily")

# Thresholds, in one place so they can be tuned without touching the priority list.
EMERGENCY = 0.30
SINGLE_HEAL = 0.65
AOE_HEAL = 0.75
AOE_MIN_TARGETS = 3

# Every threshold the profile needs a `party.count_below_N` field for. The party sensor
# is built from this, so changing a threshold above cannot leave the rotation reading a
# field nothing produces — `GameProfile.validate()` catches the drift at startup.
PARTY_THRESHOLDS: tuple[float, ...] = (EMERGENCY, 0.55, SINGLE_HEAL, AOE_HEAL)


def build(keys: dict[str, str], gcd_recast_s: float = 2.5) -> RotationProfile:
    """Healer priority list: keep people alive, otherwise deal damage.

    Read top to bottom. Every heal is gated on someone actually needing it, so on a
    healthy party the whole healing section falls through and the profile behaves like a
    DPS rotation — which is what a good healer does.
    """
    abilities: list[Ability] = [
        # -- emergencies ---------------------------------------------------------
        Ability(
            id="benediction",
            key=keys["benediction"],
            kind=AbilityKind.OGCD,
            recast_s=RECASTS["benediction"],
            target=TargetSpec.mouseover_lowest_ally(hp_ceiling=EMERGENCY),
            condition=Condition.field("party.lowest_hp_frac", "<", EMERGENCY),
            comment="Full heal. Off-GCD, so it costs no cast — use it early, not late.",
        ),
        Ability(
            id="tetragrammaton",
            key=keys["tetragrammaton"],
            kind=AbilityKind.OGCD,
            recast_s=RECASTS["tetragrammaton"],
            target=TargetSpec.mouseover_lowest_ally(hp_ceiling=0.55),
            condition=Condition.field("party.lowest_hp_frac", "<", 0.55),
            comment="Instant off-GCD heal; free to weave, so it never costs damage.",
        ),
        # -- party-wide ----------------------------------------------------------
        Ability(
            id="afflatus_rapture",
            key=keys["afflatus_rapture"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["afflatus_rapture"],
            cast_time_s=0.0,
            condition=Condition.all_(
                Condition.truthy("buff.lily.active"),
                Condition.field(f"party.count_below_{int(AOE_HEAL * 100)}", ">=", AOE_MIN_TARGETS),
            ),
            comment="Free instant AoE heal when lilies are up.",
        ),
        Ability(
            id="medica2",
            key=keys["medica2"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["medica2"],
            cast_time_s=2.0,
            condition=Condition.field(
                f"party.count_below_{int(AOE_HEAL * 100)}", ">=", AOE_MIN_TARGETS
            ),
            comment="Hard-cast AoE regen. Skipped automatically when movement is coming.",
        ),
        # -- single target -------------------------------------------------------
        Ability(
            id="afflatus_solace",
            key=keys["afflatus_solace"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["afflatus_solace"],
            cast_time_s=0.0,
            target=TargetSpec.mouseover_lowest_ally(hp_ceiling=SINGLE_HEAL),
            condition=Condition.all_(
                Condition.truthy("buff.lily.active"),
                Condition.field("party.lowest_hp_frac", "<", SINGLE_HEAL),
            ),
            comment="Instant single-target heal. Preferred over Cure II — no cast to clip.",
        ),
        Ability(
            id="cure2",
            key=keys["cure2"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["cure2"],
            cast_time_s=2.0,
            target=TargetSpec.mouseover_lowest_ally(hp_ceiling=SINGLE_HEAL),
            condition=Condition.field("party.lowest_hp_frac", "<", SINGLE_HEAL),
            comment="Hard-cast single heal, the fallback when no lily is available.",
        ),
        # -- damage --------------------------------------------------------------
        Ability(
            id="dia",
            key=keys["dia"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["dia"],
            cast_time_s=0.0,
            condition=Condition.any_(
                Condition.no_buff("dia_dot"),
                Condition.field("buff.dia_dot.remaining_s", "<", 3.0),
            ),
            comment="DoT upkeep. Instant, so it is also the movement filler.",
        ),
        Ability(
            id="glare",
            key=keys["glare"],
            kind=AbilityKind.GCD,
            recast_s=RECASTS["glare"],
            cast_time_s=1.5,
            condition=Condition.truthy("target.exists"),
            comment="Filler damage. Everything above this is conditional, so this is the default.",
        ),
        # -- utility -------------------------------------------------------------
        Ability(
            id="assize",
            key=keys["assize"],
            kind=AbilityKind.OGCD,
            recast_s=RECASTS["assize"],
            condition=Condition.truthy("player.in_combat"),
            comment="Damage and a party heal on the same button; never worth holding.",
        ),
        Ability(
            id="lucid_dreaming",
            key=keys["lucid_dreaming"],
            kind=AbilityKind.OGCD,
            recast_s=RECASTS["lucid_dreaming"],
            condition=Condition.field("player.mp_frac", "<", 0.7),
        ),
    ]

    return RotationProfile(
        id="whm.healer",
        abilities=abilities,
        gcd_recast_s=gcd_recast_s,
        description="White Mage: heal when needed, damage otherwise. Validate against a guide.",
    )


def build_emergency_reflexes(
    keys: dict[str, str], targets: TargetResolver | None = None
) -> list[Reflex]:
    """Heals that cannot wait for the next rotation decision.

    Everything in the priority list above happens on a GCD boundary, which can be two and
    a half seconds away. Someone at 15% does not have two and a half seconds, so the
    genuine emergency lives in the reflex layer — which already preempts, already has
    cooldown handling, and already flushes the timeline.

    These use `Priority.RECOVERY`, not `REFLEX`, and do not preempt. A dodge outranks a
    heal: being alive and unhealed beats being healed inside an AoE.
    """
    resolver = targets or TargetResolver()

    return [
        Reflex(
            id="emergency_benediction",
            group="emergency_heal",
            condition=Condition.all_(
                Condition.truthy("player.in_combat"),
                Condition.field("party.lowest_hp_frac", "<", 0.20),
                Condition.ready("benediction"),
            ),
            plan=_targeted_emergency(
                resolver,
                TargetSpec.mouseover_lowest_ally(hp_ceiling=0.20),
                keys["benediction"],
                name="emergency:benediction",
            ),
            priority=Priority.RECOVERY,
            cooldown_ms=180_000,
            preempt=False,
        )
    ]


def _targeted_emergency(
    resolver: TargetResolver, spec: TargetSpec, key: str, *, name: str
):
    """Build the plan at fire time so the target is resolved from live HP.

    Resolving at construction time would aim the heal at whoever was lowest when the
    profile was loaded. Returning `None` when nobody qualifies is also how the reflex
    declines to fire — the layer already treats a `None` plan as "not now".
    """

    def build(state):
        resolution = resolver.resolve(spec, state)
        if resolution.failed:
            return None
        plan = Plan.single(key, name=name, priority=Priority.RECOVERY, expires_in_ms=800)
        if resolution.point is not None:
            return plan.hovered(ScreenPoint(*resolution.point))
        return plan.targeted(resolution.keys)

    return build
