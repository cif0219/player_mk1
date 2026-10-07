"""The FantCraft game profile: everything the runtime needs, assembled.

No ToS disclaimer here, for once — FantCraft is our own game and this player is
its automated playtester (see games/fancraft/__init__.py for the arrangement).
Run the game's server with FANCRAFT_PLAYTEST=1 and the client with ?playtest=1
so the ground-truth oracle records alongside our session; then
`python -m tools.fancraft_report` joins the two and scores this player's
perception against what the server actually knew.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from player.act.keymap import validate_keymap
from player.profile import GameProfile

from .combat import (
    MOVEMENT_KEYS,
    RECASTS,
    build_guardian,
    build_luminary,
    build_pipeline,
    keys_from_layout,
)
from .layout import Layout
from .reflexes import build_reflexes
from .sensors import build_bundle

# The game runs in a browser; the tab title is client/index.html's <title>.
WINDOW_TITLE = "FantCraft"


@dataclass(slots=True)
class FancraftConfig:
    # Title substring to find the browser window. Any browser showing the game
    # works; run it fullscreen (F11) so page viewport == window client area —
    # the manifest's regions assume it (games/fancraft/layout.py).
    window_title: str = WINDOW_TITLE
    # The HUD manifest the game's playtest bridge wrote:
    #   <fancraft>/var/playtest/<session>/manifest.json
    # Absent -> estimated defaults, good enough only for a first smoke run.
    manifest_path: str | None = None
    # Which job the character has, so the fallback hotbar mapping is right.
    job: str = "guardian"
    keys: dict[str, str] = field(default_factory=dict)
    rotation: str = "fancraft.pipeline"
    retreat_hp_threshold: float = 0.25
    # On by default since the game grew a readable radial sweep and the probe
    # became self-calibrating (games/fancraft/sensors.py AdaptiveCooldownProbe).
    hotbar_probes: bool = True

    # ── API control (games/fancraft/api.py, player/api) ──
    # "screen": read the HUD off pixels and press keys (the playtest of the
    # interface). "api": read the game's agent gateway and send intents (the
    # playtest of the game's rules, with exact positions and telegraph timing).
    control: str = "screen"
    gateway_url: str = "ws://127.0.0.1:10190"
    username: str = ""
    password: str = "agent-pass-123"
    character: str = ""
    zone: str = "trial_warden"
    room_id: str = ""
    setup_commands: tuple[str, ...] = ("/job guardian", "/skill * 30")
    # Warden tactics: "parry" (a fresh guard before each cut) or "dodge" (sprint out of reach).
    tactic: str = "parry"
    # Share of far-reaching cuts (Wind Cutter's line, Great Serpent's lightning) actually
    # left; the rest are taken, to measure the boss rather than the tester.
    dodge_chance: float = 1.0
    # Items bought at the Trial Warden's free counter before entering (id → quantity).
    shopping: dict[str, int] = field(default_factory=lambda: {"holy_water_supreme": 3})
    # Where the counter is: the playtest teleport that lands beside Warden Oswin.
    shop_teleport: str = "/tp 133 113"
    # Summoned test helper (fancraft docs/PLAYTEST.md §7): register with the gateway under
    # this name, bound to the requester, and start on this task (games/fancraft/tasks.py).
    helper: dict[str, str] = field(default_factory=dict)
    task: str = "full"


def build(config: FancraftConfig | None = None) -> GameProfile:
    """The profile for the configured control path."""
    config = config or FancraftConfig()
    if config.control == "api":
        return build_api(config)
    if config.control != "screen":
        raise ValueError(f"fancraft.control must be 'screen' or 'api', got {config.control!r}")
    return build_screen(config)


def build_api(config: FancraftConfig | None = None) -> GameProfile:
    """Play through the agent gateway: no window, no probes, exact state."""
    from player.api import ApiBackend, ApiSource, ApiTransport, GatewayClient
    from player.api.source import SessionSpec

    from .api import FightLedger, build_api_bundle, build_key_actions
    from .tasks import ALL_GROUPS, task as helper_task
    from .warden import WardenMemory, WardenTuning, build_warden_reflexes

    config = config or FancraftConfig(control="api")
    if config.tactic not in ("parry", "dodge"):
        raise ValueError(f"fancraft.tactic must be 'parry' or 'dodge', got {config.tactic!r}")
    if not 0.0 <= config.dodge_chance <= 1.0:
        raise ValueError(f"fancraft.dodge_chance must be in [0, 1], got {config.dodge_chance!r}")
    gateway = GatewayClient(config.gateway_url)
    helper = {k: str(v) for k, v in (config.helper or {}).items()}
    session = SessionSpec(
        username=config.username, password=config.password, character=config.character,
        zone=config.zone, room_id=config.room_id, setup_commands=tuple(config.setup_commands),
        shopping=tuple((k, int(v)) for k, v in config.shopping.items() if int(v) > 0),
        shop_teleport=config.shop_teleport,
        helper_name=helper.get("name", ""), helper_requester=helper.get("requester", ""),
    )
    source = ApiSource(gateway, session)
    backend = ApiBackend(gateway, build_key_actions(gateway))
    ledger = FightLedger()
    tuning = WardenTuning(tactic=config.tactic, dodge_chance=config.dodge_chance)
    memory = WardenMemory()
    reflexes = build_warden_reflexes(config.tactic, memory, tuning)
    current = {"task": helper.get("task", config.task)}

    def apply_task(task_id: str) -> None:
        # A task is the set of reflex groups that stay on (games/fancraft/tasks.py).
        spec = helper_task(task_id)
        for group in ALL_GROUPS:
            reflexes.set_group_enabled(group, group in spec.groups)
        tuning.tactic = spec.tactic if config.tactic == "parry" else config.tactic
        current["task"] = spec.id

    apply_task(current["task"])
    from .samurai import install_samurai_reflex
    samurai_tactics = install_samurai_reflex(reflexes, gateway, lambda: current["task"])

    def on_event(event) -> None:
        # Our entity id changes with every room (the counter in the overworld, then the
        # dojo; a helper following its requester), so the ledger re-reads it each event.
        latest = gateway.latest()
        if latest is not None:
            ledger.self_id = int((latest.data.get("self") or {}).get("entityId", 0)) or ledger.self_id
        ledger.on_event(event)
        if event.type == "helper_task":
            apply_task(str(event.data.get("task", "full")))
            ledger.notices.append((event.received_at - ledger.started_at, f"task → {current['task']}"))
        elif event.type == "helper_follow":
            source.request_room(str(event.data.get("zone", "")), str(event.data.get("roomId", "")))
        elif event.type == "helper_dismiss":
            source.dismiss()

    gateway.listeners.append(on_event)
    transport = ApiTransport(
        gateway=gateway, source=source, backend=backend,
        report=lambda: ledger.report() + [
            f"task: {current['task']}; tuning: tactic={tuning.tactic} dodge_chance={tuning.dodge_chance}; memory: {memory.stats()}",
            f"api backend: {backend.stats()}", f"gateway: {gateway.stats()}",
            f"samurai visible-state tactics: {samurai_tactics.stats}",
        ],
        apply_task=apply_task,
    )

    return GameProfile(
        name="fancraft",
        window_title=config.window_title,
        sensors=build_api_bundle(),
        reflexes=reflexes,
        rotations={},
        default_rotation="",
        critical_fields=("player.hp_frac", "player.downed"),
        parameters={"warden.tactic": config.tactic, "warden.dodge_chance": config.dodge_chance, "helper.task": current["task"]},
        detector_backend="none",
        transport=transport,
    )


def build_screen(config: FancraftConfig | None = None) -> GameProfile:
    """Play from pixels: the HUD manifest, probes and SendInput (the original path)."""
    config = config or FancraftConfig()
    layout = Layout.load_or_default(config.manifest_path)

    ability_keys = keys_from_layout(layout, config.job)
    keys = {**MOVEMENT_KEYS, **ability_keys, **config.keys}

    problems = validate_keymap(keys)
    if problems:
        raise ValueError("invalid keybinds:\n  " + "\n  ".join(problems))

    rotations = {
        "fancraft.pipeline": build_pipeline(keys),
        "fancraft.guardian": build_guardian(keys),
        "fancraft.luminary": build_luminary(keys),
    }

    bundle = build_bundle(layout, RECASTS, hotbar_probes=config.hotbar_probes)
    # The approach reflex watches the primary damage ability's range wash:
    # melee reach for the guardian, spell reach for the luminary.
    approach_ability = "starfire" if config.job == "luminary" else "iron_cleave"
    reflexes = build_reflexes(
        keys,
        retreat_hp_threshold=config.retreat_hp_threshold,
        approach_ability=approach_ability,
    )

    return GameProfile(
        name="fancraft",
        window_title=config.window_title,
        sensors=bundle,
        reflexes=reflexes,
        rotations=rotations,
        default_rotation=config.rotation,
        # HP gates survival; MP gates every luminary decision. The hotbar
        # probes degrade gracefully (an unreadable slot just falls through the
        # priority list), so they are deliberately not load-bearing.
        critical_fields=("player.hp_frac",),
        parameters={
            "rotation.enabled": True,
            "retreat.hp_threshold": config.retreat_hp_threshold,
        },
        detector_backend="none",
    )


def config_from_dict(data: dict[str, Any]) -> FancraftConfig:
    return FancraftConfig(
        window_title=data.get("window_title", WINDOW_TITLE),
        manifest_path=data.get("manifest_path"),
        job=data.get("job", "guardian"),
        keys=dict(data.get("keys", {})),
        rotation=data.get("rotation", "fancraft.pipeline"),
        retreat_hp_threshold=float(data.get("retreat_hp_threshold", 0.25)),
        hotbar_probes=bool(data.get("hotbar_probes", True)),
        control=str(data.get("control", "screen")),
        gateway_url=str(data.get("gateway_url", "ws://127.0.0.1:10190")),
        username=str(data.get("username", "") or ""),
        password=str(data.get("password", "agent-pass-123")),
        character=str(data.get("character", "") or ""),
        zone=str(data.get("zone", "trial_warden")),
        room_id=str(data.get("room_id", "") or ""),
        setup_commands=tuple(str(c) for c in (data.get("setup_commands") or ("/job guardian", "/skill * 30"))),
        tactic=str(data.get("tactic", "parry")),
        dodge_chance=float(data.get("dodge_chance", 1.0)),
        shopping={str(k): int(v) for k, v in (data.get("shopping") if data.get("shopping") is not None else {"holy_water_supreme": 3}).items()},
        shop_teleport=str(data.get("shop_teleport", "/tp 133 113")),
        helper={str(k): str(v) for k, v in (data.get("helper") or {}).items()},
        task=str(data.get("task", "full")),
    )
