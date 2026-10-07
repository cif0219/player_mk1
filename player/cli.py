"""Command line entry points.

    player run       --config config/ffxiv.yaml [--dry-run]
    player replay    --session sessions/xyz [--dry-run]
    player calibrate --game ffxiv --out config/ffxiv.layout.json
    player inspect   --session sessions/xyz
    player check     --config config/ffxiv.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from .act.backend import NullBackend
from .clock import now
from .config import AppConfig
from .profile import GameProfile
from .record.recorder import Recorder, RecorderConfig
from .runtime import Runtime, RuntimeConfig
from .safety.guards import (
    ConfidenceGuard,
    ForegroundGuard,
    KillSwitchGuard,
    RateGuard,
    SafetyGate,
    StalenessGuard,
    TakeoverGuard,
)
from .safety.killswitch import KillSwitch
from .strategy.director import Director, DirectorConfig


def build_game_profile(config: AppConfig) -> GameProfile:
    """Resolve the configured game to its profile. The one place games are dispatched."""
    if config.game == "ffxiv":
        from games.ffxiv import build, config_from_dict

        return build(config_from_dict(config.game_config))
    if config.game == "genshin":
        from games.genshin import build, config_from_dict

        return build(config_from_dict(config.game_config))
    if config.game == "fancraft":
        from games.fancraft import build, config_from_dict

        return build(config_from_dict(config.game_config))
    raise ValueError(f"unknown game {config.game!r}")


def build_gate(
    config: AppConfig,
    killswitch: KillSwitch,
    tracker,
    critical_fields: tuple[str, ...],
    require_foreground: bool,
) -> SafetyGate:
    s = config.safety
    return SafetyGate(
        [
            KillSwitchGuard(killswitch),
            ForegroundGuard(tracker, required=require_foreground),
            StalenessGuard(max_age_ms=s.max_state_age_ms),
            ConfidenceGuard(critical_fields, min_confidence=s.min_confidence),
            RateGuard(max_per_sec=s.max_actions_per_sec),
            TakeoverGuard(killswitch, cooldown_ms=s.takeover_cooldown_ms),
        ]
    )


def build_backend(dry_run: bool):
    if dry_run:
        return NullBackend()
    from .act.win_sendinput import SendInputBackend

    return SendInputBackend()


def build_director(config: AppConfig, runtime_ref: list[Runtime]) -> Director | None:
    d = config.director
    if not d.get("enabled", False):
        return None
    cfg = DirectorConfig(
        enabled=True,
        model=d.get("model", DirectorConfig.model),
        effort=d.get("effort", "low"),
        min_interval_s=float(d.get("min_interval_s", 5.0)),
        screenshot_every_n_calls=int(d.get("screenshot_every_n_calls", 3)),
        max_output_tokens_per_session=int(d.get("max_output_tokens_per_session", 60_000)),
        provider=d.get("provider", "anthropic"),
        base_url=d.get("base_url", ""),
        api_key=d.get("api_key", ""),
        disable_thinking=bool(d.get("disable_thinking", True)),
        request_timeout_s=float(d.get("request_timeout_s", 30.0)),
    )
    # Late binding: the director needs the runtime's `apply_directive`, and the runtime
    # needs the director. A one-element list is the smallest honest way to tie the knot.
    return Director(cfg, apply=lambda directive: runtime_ref[0].apply_directive(directive))


# -- commands ---------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from .capture.screen import ScreenSource
    from .capture.window import WindowTracker

    config = AppConfig.load(args.config)
    problems = config.validate()
    if problems:
        _fail("configuration problems", problems)
        return 2

    dry_run = args.dry_run or config.runtime.get("dry_run", True)
    if config.game_config.get("party"):
        # Several characters at once (API control only): one runtime per member, one
        # kill switch, one clock. The first member opens the room, the rest join it.
        return _run_party(config, dry_run, args)
    profile = build_game_profile(config)

    tracker = WindowTracker(profile.window_title)
    killswitch = KillSwitch(config.safety.kill_key, config.safety.pause_key)
    killswitch.start()

    transport = getattr(profile, "transport", None)
    if transport is not None:
        # API control: frames come from the game's gateway and intents go back through
        # it. There is no window to keep foreground and no pixels worth recording.
        return _run(
            config, profile, transport.source, tracker, killswitch, dry_run, args,
            require_foreground=False,
            backend=None if dry_run else transport.backend,
            extra_report=transport.report,
            record_frames=False,
        )

    source = ScreenSource(
        tracker,
        downscale=config.capture.downscale,
        max_fps=int(config.runtime.get("capture_fps", 60)),
    )
    return _run(config, profile, source, tracker, killswitch, dry_run, args)


def cmd_replay(args: argparse.Namespace) -> int:
    from .capture.replay import ReplaySource
    from .capture.window import WindowTracker

    config = AppConfig.load(args.config) if args.config else AppConfig()
    profile = build_game_profile(config)

    source = ReplaySource(args.session, realtime=args.realtime, loop=False, speed=args.speed)
    tracker = WindowTracker(profile.window_title)
    killswitch = KillSwitch(config.safety.kill_key, config.safety.pause_key)

    # Replay never dispatches and the window is not involved, so the foreground guard has
    # nothing meaningful to check — leaving it on would block every tick and produce a
    # session where nothing happens for a reason that is not the one under test.
    # Recording is off by default (replaying a session should not silently mint another
    # one) but --record-trace opts in: scoring perception against an external oracle
    # needs the replayed run's own state/dispatch trace on disk.
    return _run(
        config,
        profile,
        source,
        tracker,
        killswitch,
        dry_run=True,
        args=args,
        require_foreground=False,
        record=bool(getattr(args, "record_trace", False)),
    )


def party_member_usernames(base: dict, members: list[dict]) -> list[str]:
    """One account per party member: the member's own `username`, else `<party>_<name>`."""
    names = []
    for i, member in enumerate(members):
        name = str(member.get("name", f"member{i + 1}"))
        names.append(str(member.get("username") or f"{base.get('username', 'party')}_{name}"))
    return names


def _run_party(config: AppConfig, dry_run: bool, args: argparse.Namespace) -> int:
    """Play a party: `fancraft.party` lists members, each a set of overrides on the
    game config (`name`, `task`, `username`, `tactic`, `dodge_chance`, `shopping`…).

    Members share one gateway URL and one room: the first member's session
    creates (or joins) the zone, and every later member joins that room by id,
    so a two-character answer to the Warden — one on the man, one on the horse
    — runs from a single command. Reports print per member.
    """
    from .capture.window import WindowTracker
    from .profile import GameProfile

    members = list(config.game_config.get("party") or [])
    if not members:
        _fail("configuration problems", ["fancraft.party is empty"])
        return 2
    base = {k: v for k, v in config.game_config.items() if k != "party"}
    base["control"] = "api"
    profiles: list[tuple[str, GameProfile]] = []
    for i, member in enumerate(members):
        overrides = dict(member)
        name = str(overrides.pop("name", f"member{i + 1}"))
        merged = {**base, **overrides}
        # Each member is its own account (and so its own character) unless the member
        # names one: the shared `username` is only the party's prefix.
        if "username" not in overrides:
            merged["username"] = f"{base.get('username', 'party')}_{name}"
        member_config = AppConfig(
            game=config.game, runtime=config.runtime, capture=config.capture, safety=config.safety,
            record=config.record, director={"enabled": False}, game_config=merged,
        )
        profiles.append((name, build_game_profile(member_config)))
    for name, profile in profiles:
        if getattr(profile, "transport", None) is None:
            _fail("configuration problems", [f"party member {name!r} is not API-controlled"])
            return 2

    killswitch = KillSwitch(config.safety.kill_key, config.safety.pause_key)
    killswitch.start()
    tracker = WindowTracker(profiles[0][1].window_title)
    runtimes: list[tuple[str, Runtime]] = []
    for name, profile in profiles:
        transport = profile.transport
        gate = build_gate(config, killswitch, tracker, profile.critical_fields, False)
        for guard in gate.guards:
            if isinstance(guard, TakeoverGuard):
                guard.enabled = False
        rec_cfg = config.record
        recorder = Recorder(RecorderConfig(
            enabled=bool(rec_cfg.get("enabled", False)), root=Path(rec_cfg.get("root", "sessions")),
            name=f"{time.strftime('%Y%m%d-%H%M%S')}-{name}", frame_every_n=1, record_frames=False,
        ))
        runtime = Runtime(
            profile=profile, source=transport.source, backend=None if dry_run else transport.backend, gate=gate,
            recorder=recorder, killswitch=killswitch,
            config=RuntimeConfig(dry_run=dry_run, status_interval_s=float(config.runtime.get("status_interval_s", 5.0)),
                                 max_runtime_s=args.time or config.runtime.get("max_runtime_s")),
        )
        problems = runtime.preflight()
        if problems:
            _fail(f"preflight failed for {name}", problems)
            return 2
        runtimes.append((name, runtime))

    print(f"party of {len(runtimes)}: " + ", ".join(f"{n} ({p.parameters.get('helper.task', 'full')})" for n, p in profiles))
    print(f"mode: {'DRY RUN (no input dispatched)' if dry_run else 'LIVE'}")
    print(killswitch.status())
    print()

    # The first member opens the room; the others follow it in by id.
    started: list[tuple[str, Runtime]] = []
    try:
        for i, (name, runtime) in enumerate(runtimes):
            source = runtime.profile.transport.source
            if i > 0:
                lead = runtimes[0][1].profile.transport.source
                room_id = str(lead.entered.get("roomId", "") or "")
                zone = str(lead.entered.get("zone", "") or source.session.zone)
                if room_id:
                    source.session.room_id = room_id
                    source.session.zone = zone
            runtime.start()
            started.append((name, runtime))
            print(f"{name}: {source.describe()}")
    except Exception as exc:
        _fail("party start failed", [str(exc)])
        for _, runtime in started:
            runtime.stop()
        return 2

    signal.signal(signal.SIGINT, lambda *_: [r.request_stop() for _, r in started])
    begun = now()
    last_status = 0.0
    try:
        while any(r.running for _, r in started):
            elapsed = now() - begun
            limit = started[0][1].config.max_runtime_s
            if limit and elapsed >= limit:
                break
            if elapsed - last_status >= started[0][1].config.status_interval_s:
                for name, runtime in started:
                    print(f"[{name}] {runtime.status_line()}")
                last_status = elapsed
            started[0][1]._stop.wait(0.25)  # noqa: SLF001 - the CLI owns these runtimes
    except KeyboardInterrupt:
        pass
    finally:
        for _, runtime in started:
            runtime.stop()

    for name, runtime in started:
        print()
        print(f"=== {name} ===")
        for line in runtime.report():
            print(line)
        for line in runtime.profile.transport.report():
            print(line)
    return 0


def _run(
    config: AppConfig,
    profile: GameProfile,
    source,
    tracker,
    killswitch: KillSwitch,
    dry_run: bool,
    args: argparse.Namespace,
    require_foreground: bool | None = None,
    record: bool = True,
    backend=None,
    extra_report=None,
    record_frames: bool = True,
) -> int:
    if require_foreground is None:
        require_foreground = config.safety.require_foreground

    gate = build_gate(config, killswitch, tracker, profile.critical_fields, require_foreground)
    if getattr(profile, "transport", None) is not None:
        # API control shares no keyboard with the human: their typing is not a takeover,
        # and yielding to it would flush a dodge mid-fight. The kill switch still latches.
        for guard in gate.guards:
            if isinstance(guard, TakeoverGuard):
                guard.enabled = False

    rec_cfg = config.record
    recorder = Recorder(
        RecorderConfig(
            enabled=bool(rec_cfg.get("enabled", False)) and record,
            root=Path(rec_cfg.get("root", "sessions")),
            frame_every_n=int(rec_cfg.get("frame_every_n", 2)),
            jpeg_quality=int(rec_cfg.get("jpeg_quality", 80)),
            record_frames=record_frames,
        )
    )

    runtime_ref: list[Runtime] = []
    director = build_director(config, runtime_ref)

    runtime = Runtime(
        profile=profile,
        source=source,
        backend=backend if backend is not None else build_backend(dry_run),
        gate=gate,
        director=director,
        recorder=recorder,
        killswitch=killswitch,
        config=RuntimeConfig(
            dry_run=dry_run,
            capture_fps=int(config.runtime.get("capture_fps", 60)),
            status_interval_s=float(config.runtime.get("status_interval_s", 5.0)),
            max_runtime_s=args.time or config.runtime.get("max_runtime_s"),
        ),
    )
    runtime_ref.append(runtime)

    problems = runtime.preflight()
    if problems:
        _fail("preflight failed", problems)
        return 2

    print(profile.describe())
    print(f"mode: {'DRY RUN (no input dispatched)' if dry_run else 'LIVE'}")
    print(killswitch.status())
    print()

    signal.signal(signal.SIGINT, lambda *_: runtime.request_stop())

    runtime.start()
    started = now()
    last_status = 0.0
    try:
        while runtime.running:
            elapsed = now() - started
            limit = runtime.config.max_runtime_s
            if limit and elapsed >= limit:
                break
            if elapsed - last_status >= runtime.config.status_interval_s:
                print(runtime.status_line())
                last_status = elapsed
            runtime._stop.wait(0.25)  # noqa: SLF001 - the CLI owns this runtime
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()

    print()
    for line in runtime.report():
        print(line)
    if extra_report is not None:
        for line in extra_report():
            print(line)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from tools.calibrate import run_calibration

    out = args.out or f"config/{args.game}.layout.json"
    return run_calibration(game=args.game, out=out, session=args.session)


def cmd_inspect(args: argparse.Namespace) -> int:
    from .record.session import Session

    session = Session(args.session)
    print(session.summary())
    print(f"frames: {len(session.frame_paths())}")
    for key, value in sorted(session.meta.items()):
        print(f"  {key}: {value}")
    if args.states:
        for ev in list(session.states())[: args.limit]:
            print(f"  t={ev.t:7.3f}  {ev.data.get('fields')}")
    if args.dispatches:
        for ev in list(session.trace("dispatch"))[: args.limit]:
            print(f"  t={ev.t:7.3f}  {ev.data}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Validate config and profile without touching the screen or the game."""
    config = AppConfig.load(args.config)
    problems = config.validate()
    try:
        profile = build_game_profile(config)
        problems.extend(profile.validate())
    except Exception as exc:
        problems.append(f"profile build failed: {exc}")
        profile = None

    if problems:
        _fail("check failed", problems)
        return 1

    print("config ok")
    if profile is not None:
        print(profile.describe())
        print(f"fields required: {len(profile.required_fields())}")
        print(f"fields provided: {len(profile.sensors.provides)}")
    return 0


def _fail(headline: str, problems: list[str]) -> None:
    print(f"{headline}:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)


# -- argument parsing -------------------------------------------------------------


def cmd_journey(args: argparse.Namespace) -> int:
    import time as _time
    from pathlib import Path

    from games.fancraft.journey import run_journey
    from .api.gateway import GatewayClient

    gw = GatewayClient(args.gateway)
    gw.connect()
    username = args.username or f"journey_{int(_time.time()) % 100000}"
    out = Path(args.out) / f"{args.scenario}-{_time.strftime('%Y%m%d-%H%M%S')}"
    try:
        report = run_journey(gw, args.scenario, username, args.password, out, args.setup,
                             log=lambda m: print(m, flush=True), stop_on_failure=not args.keep_going,
                             character=args.character, companions=args.companions)
    finally:
        gw.close()
    summary = report.summary()
    print(f"journey {args.scenario}: {'FINISHED' if summary['finished'] else 'STOPPED'} "
          f"{summary['steps_ok']}/{summary['steps']} steps, report {out}", flush=True)
    return 0 if summary["finished"] else 1


def cmd_companion(args: argparse.Namespace) -> int:
    """An NPC companion launched by the agent gateway's helper desk (fancraft docs/COMBAT.md §7.3)."""
    from games.fancraft.companion import run_companion
    from .api.gateway import GatewayClient

    gw = GatewayClient(args.gateway)
    gw.connect()
    try:
        run_companion(gw, args.username, args.password, args.room, args.zone, args.requester, args.requester_name,
                      args.seconds, log=lambda m: print(m, flush=True))
    finally:
        gw.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="player", description="screen-perception game player")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="play live")
    run.add_argument("-c", "--config", default="config/ffxiv.yaml")
    run.add_argument("--dry-run", action="store_true", help="decide but dispatch nothing")
    run.add_argument("-t", "--time", type=float, help="stop after N seconds")
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="run the pipeline over a recorded session")
    replay.add_argument("-s", "--session", required=True)
    replay.add_argument("-c", "--config", default=None)
    replay.add_argument("--realtime", action="store_true", help="reproduce original timing")
    replay.add_argument("--speed", type=float, default=1.0)
    replay.add_argument("--dry-run", action="store_true", default=True)
    replay.add_argument("-t", "--time", type=float, default=None)
    replay.add_argument("--record-trace", action="store_true",
                        help="record the replayed run as a session (states + dry-run dispatches)")
    replay.set_defaults(func=cmd_replay)

    cal = sub.add_parser("calibrate", help="define HUD regions for your resolution")
    cal.add_argument("-g", "--game", default="ffxiv")
    # Default derived from the game at run time, so `--game genshin` with no --out does
    # not silently overwrite the FFXIV layout.
    cal.add_argument("-o", "--out", default=None)
    cal.add_argument("-s", "--session", default=None, help="calibrate from a recording")
    cal.set_defaults(func=cmd_calibrate)

    inspect = sub.add_parser("inspect", help="summarise a recorded session")
    inspect.add_argument("-s", "--session", required=True)
    inspect.add_argument("--states", action="store_true")
    inspect.add_argument("--dispatches", action="store_true")
    inspect.add_argument("--limit", type=int, default=20)
    inspect.set_defaults(func=cmd_inspect)

    journey = sub.add_parser("journey", help="play a story chapter through the agent gateway and report (FantCraft)")
    journey.add_argument("--gateway", default="ws://127.0.0.1:10190", help="agent gateway URL")
    journey.add_argument("--scenario", default="act3", help="prologue | act1 | act2 | act3 | campaign | luminara | luminara_alternate | luminara_phase2 | luminara_phase2_alternate")
    journey.add_argument("--username", default=None)
    journey.add_argument("--character", default=None, help="character name to create or pick (default: derived from the username)")
    journey.add_argument("--password", default="journey-pass-123")
    journey.add_argument("--out", default="var/journey")
    journey.add_argument("--setup", action="append", default=[], help="chat command before the first step (repeatable)")
    journey.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    journey.add_argument("--companions", type=int, default=0,
                         help="NPC companions to call into each instance through the party panel (capped by its headcount)")
    journey.set_defaults(func=cmd_journey)

    companion = sub.add_parser("companion", help="fight beside a player as an NPC companion (launched by the agent gateway)")
    companion.add_argument("--gateway", default="ws://127.0.0.1:10190")
    companion.add_argument("--username", required=True, help="the account name the game server gave this companion")
    companion.add_argument("--password", default="companion-pass-123")
    companion.add_argument("--room", required=True)
    companion.add_argument("--zone", required=True)
    companion.add_argument("--requester", required=True, help="character id of the player being served")
    companion.add_argument("--requester-name", required=True)
    companion.add_argument("--job", default=None, help="informational: the room gives the companion its job")
    companion.add_argument("--seconds", type=float, default=900)
    companion.add_argument("--out", default=None)
    companion.set_defaults(func=cmd_companion)

    check = sub.add_parser("check", help="validate config and profile, touching nothing")
    check.add_argument("-c", "--config", default="config/ffxiv.yaml")
    check.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
