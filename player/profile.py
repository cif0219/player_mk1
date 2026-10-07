"""What a game profile has to supply.

This is the seam between the generic substrate and one specific game. Everything
FFXIV-specific lives in `games/ffxiv/`; `player/` imports from here and never from there.
Adding a second game means writing another `GameProfile` and changing nothing under
`player/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .perceive.sensor import SensorBundle
from .policy.reflex import ReflexLayer
from .policy.rotation import RotationProfile
from .policy.targeting import TargetResolver


@dataclass(slots=True)
class GameProfile:
    """Everything the runtime needs to play one game."""

    name: str
    window_title: str
    sensors: SensorBundle
    reflexes: ReflexLayer
    rotations: dict[str, RotationProfile] = field(default_factory=dict)
    default_rotation: str = ""
    # Keybinds for target acquisition. Shared by the rotation and by any reflex that
    # aims at an ally, so both agree on which key selects which party slot.
    targets: TargetResolver | None = None
    # Fields the safety gate treats as load-bearing. Absence or low confidence in any of
    # these stops dispatch, because deciding without them is guessing.
    critical_fields: tuple[str, ...] = ()
    # Names and defaults the director may set. Anything not listed here is unreachable
    # from a directive, which is the point.
    parameters: dict[str, Any] = field(default_factory=dict)
    detector_backend: str = "none"
    # API control (player/api): when set, the runtime takes frames from the transport's
    # source and dispatches through its backend instead of capturing a window.
    transport: Any = None

    def rotation(self, profile_id: str) -> RotationProfile | None:
        return self.rotations.get(profile_id)

    def required_fields(self) -> set[str]:
        """Every field any policy in this profile reads."""
        out: set[str] = set(self.critical_fields)
        out |= self.reflexes.required_fields()
        for rotation in self.rotations.values():
            out |= rotation.required_fields()
        return out

    def validate(self) -> list[str]:
        """Startup checks. Returns problems; an empty list means good to run.

        Catching a policy that reads a field nothing produces here — rather than reading
        `None` sixty times a second and behaving oddly for reasons nobody can see — is
        most of the value of declaring `provides` on sensors at all.
        """
        problems: list[str] = []

        if self.default_rotation and self.default_rotation not in self.rotations:
            problems.append(
                f"default_rotation {self.default_rotation!r} is not in "
                f"{sorted(self.rotations)}"
            )

        missing = self.sensors.check_requirements(self.required_fields())
        if missing:
            problems.append(f"no sensor provides: {', '.join(missing)}")

        for rid, rotation in self.rotations.items():
            seen: set[str] = set()
            for ability in rotation.abilities:
                if ability.id in seen:
                    problems.append(f"rotation {rid}: duplicate ability id {ability.id!r}")
                seen.add(ability.id)
                if not ability.key:
                    problems.append(f"rotation {rid}: ability {ability.id!r} has no key")

        return problems

    def describe(self) -> str:
        if self.transport is not None:
            return (
                f"{self.name}: {self.transport.describe()} "
                f"sensors={len(self.sensors.sensors)} reflexes={len(self.reflexes)} "
                f"rotations={sorted(self.rotations)}"
            )
        return (
            f"{self.name}: window={self.window_title!r} "
            f"sensors={len(self.sensors.sensors)} "
            f"reflexes={len(self.reflexes)} "
            f"rotations={sorted(self.rotations)} "
            f"detector={self.detector_backend}"
        )
