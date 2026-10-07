# player_mk1

A general screen-perception game player, built around one concrete target so the
substrate is shaped by a real workload instead of guesses.

**Vertical slice:** execute a correct single-target combat rotation in Final Fantasy XIV
on a striking dummy, driven entirely by what is on screen, and react to a telegraphed
ground AoE by moving out of it. Success is measurable: rotation accuracy against a
reference priority list, and reaction latency from telegraph-visible to key-dispatched.

## What this is and is not

Perception is **screen pixels only**. There is no process injection, no memory reading,
no packet inspection, no anti-cheat evasion, and no input humanisation intended to make
synthetic input look human. See [docs/SAFETY.md](docs/SAFETY.md) — the non-goals there
are load-bearing, not decoration.

Square Enix's Final Fantasy XIV User Agreement prohibits third-party gameplay
automation, and accounts are banned for it. Running this against a live account is your
decision and your risk.

## Architecture in one screen

```
 capture thread            perceive+decide thread              dispatch thread
┌──────────────┐   frame  ┌───────────────────────────┐  plan ┌──────────────────┐
│ CaptureSource│ ───────▶ │ Sensors ──▶ WorldState    │ ────▶ │ InputTimeline    │
│  screen /    │  latest  │              │            │       │  (deadline queue)│
│  replay      │  slot    │              ▼            │       │        │         │
└──────────────┘          │  Reflexes ──▶ Rotation    │       │        ▼         │
                          └───────────────────────────┘       │ SafetyGate       │
                                        ▲                     │        │         │
                          directives    │                     │        ▼         │
                          ┌─────────────┴─────────────┐       │ InputBackend     │
                          │ Director (Claude, async)  │       │ (SendInput)      │
                          └───────────────────────────┘       └──────────────────┘
```

The load-bearing idea: **`WorldState` is the contract.** Perception's only job is to
produce it; policy's only input is to read it. Rules never touch pixels, so a UI layout
change is a perception change and a strategy change is a policy change, and neither
disturbs the other. Full rationale in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md);
field-by-field schema in [docs/CONTRACTS.md](docs/CONTRACTS.md).

For raid mechanics, see [docs/ENCOUNTERS.md](docs/ENCOUNTERS.md) — the short version is
that a guide is *compiled* into a script offline and executed deterministically, with no
model in the runtime loop.

## Quick start

Install (Python 3.12+):

```bash
pip install -e ".[dev]"
```

Run the test suite — it is fully headless, no game and no screen required:

```bash
pytest
```

Prove the pipeline is wired up with no game at all — generate a synthetic HUD session and
run the whole thing over it:

```bash
python -m tools.synth --name smoke --seconds 10
```

```bash
python -m player replay --session sessions/smoke --config config/ffxiv.yaml
```

That prints a latency breakdown per stage and the decisions the rotation made. Add
`--realtime` to reproduce the original pacing; without it frames stream as fast as they
decode, which compresses the game but not the planner's 2.5s GCD commitment, so you will
see fewer decisions than a live run would make.

Calibrate UI probe regions against your own resolution and HUD layout:

```bash
python -m player calibrate --game ffxiv --out config/ffxiv.layout.json
```

Run live in dry-run mode (perceives and decides, logs every intended keypress, dispatches
nothing):

```bash
python -m player run --config config/ffxiv.yaml --dry-run
```

Drop `--dry-run` to actually dispatch input. The kill switch (`F12` by default) is a
global low-level keyboard hook and works regardless of window focus.

## Second game: Genshin Impact

`games/genshin/` is the proof that the seam holds: a second game added by writing another
`GameProfile`, with nothing under `player/` changed except teaching the input layer that
a mouse button (`mouse1`) is a pressable key — Genshin's normal and charged attacks are
the left mouse button and cannot be rebound.

It is also the easier first target. Genshin's HUD is fixed rather than user-arrangeable,
so the default layout regions are close at any 16:9 resolution, and there is no GCD or
weave window to hit — the planner paces itself off its own clock. Start with the
`genshin.attack_only` rotation to prove the pipeline, then `genshin.solo`
(burst > skill > normal attack):

```bash
python -m player run --config config/genshin.yaml --dry-run
```

The same ToS reality applies: HoYoverse's Terms of Service prohibit third-party gameplay
automation and accounts are suspended for it. Same rule as above — your account, your
decision, made knowingly.

## Third game: FantCraft — the one we own

`games/fancraft/` plays FantCraft (`D:/projects/fancraft`), our own MMORPG, as its
automated **playtester** — no ToS caveat for once; automating it is the point. Owning
both sides changes the rules: the game ships a ground-truth oracle (its server records
what was actually true while the bot played, its client publishes a measured HUD
manifest the layout is *loaded* from instead of hand-calibrated), and
`tools/fancraft_report.py` joins truth, render, and perception to say which layer got a
run wrong. The full workflow lives in FantCraft's `docs/PLAYTEST.md`.

```bash
python -m player run --config config/fancraft.yaml --dry-run
```

### API control: the second way in

Owning the game also means we can hand the tester a door that is not the
screen. FantCraft ships an **agent gateway** (`scripts/agent-gateway.mts` over
there): an ordinary game client that publishes what it sees as JSON — exact
positions, every telegraph with its landing time, the boss phase machine — and
relays intents (move, face, guard, strike, skill, place) back as ordinary
client messages. `player/api/` plugs that in at both ends of the pipeline:
`ApiSource` makes a snapshot a frame, `SnapshotSensor` makes it fields, and
`ApiBackend` makes the timeline's key events intents. Everything between —
`WorldState`, reflexes, the deadline timeline, the guard chain — is untouched,
which is how a 100 ms parry lead scheduled against a telegraph's landing time
turns into `mouse2` at exactly the right moment.

The first fight it plays is the Ironclad Warden, FantCraft's difficulty-ceiling
boss (`games/fancraft/warden.py`): parries faced at the attacker, evades of the
two cuts that reach past their arcs, wall-aware lightning dodging, provoke on
the mechanical horse, cover built for the annihilation pulse.

```bash
# fancraft: npm run dev:server, then npm run agent:gateway
python -m player run --config config/fancraft-warden.yaml -t 240
```

`npm run smoke:warden-bot` in fancraft runs the whole thing against an
isolated server and asserts how far the fight got. The screen path stays: it
is the playtest of the interface, this is the playtest of the rules.

The same profile is also the game's **test helper**: FantCraft's Trial Warden
can summon a player_mk1 into a party and hand it a task — hold the horse,
tank, purify, cover — through the gateway's helper desk (`games/fancraft/tasks.py`
mirrors the game's task table; `ApiTransport.apply_task` switches reflex
groups live). `fancraft.dodge_chance` makes the tester deliberately take a
share of the boss's far-reaching cuts, so a run measures the boss and not
only the tester; the gateway records every session for the game's replay viewer.
A `party` list in the config runs several characters from one process — the
mount is a two-person mechanic, so `config/fancraft-warden-party.yaml` puts one
member on the Warden and one on the horse.

## Layout

| Path | What lives there |
| --- | --- |
| `player/capture/` | Frame acquisition: screen, replay, window tracking |
| `player/perceive/` | Sensors → `WorldState`. Probes (numpy) and detector (ONNX) |
| `player/policy/` | Reflexes and the rotation planner. Reads `WorldState`, emits plans |
| `player/act/` | Deadline-scheduled input timeline and input backends |
| `player/api/` | API control: gateway client, snapshot source and sensor, intent backend |
| `player/safety/` | Kill switch and the guards every dispatch passes through |
| `player/strategy/` | The Claude director and its typed directive vocabulary |
| `player/record/` | Session recording and playback |
| `games/ffxiv/` | Everything FFXIV-specific: layout, sensors, reflexes, job rotations |
| `games/genshin/` | Everything Genshin-specific: layout, sensors, reflexes, combat profiles |
| `games/fancraft/` | FantCraft playtester: manifest-driven layout, sensors, rotations |
| `tools/` | Calibration, synthetic sessions, weak-labelling for detector training |

Nothing under `player/` imports from `games/`. The dependency runs one way.
