# Contracts

The three types every layer boundary is expressed in. Change these deliberately; they are
the reason the layers can move independently.

## `WorldState` — perception → policy

Produced once per frame by `perceive/pipeline.py`. Read by everything downstream. Policy
never sees a frame.

```python
WorldState(
    tick=1834,                       # monotonic frame counter
    captured_at=91234.5061,          # time.perf_counter() at capture, not at parse
    perceived_at=91234.5138,         # when assembly finished; the age bound uses this
    fields={...},                    # dict[str, Field]
    entities=[...],                  # list[Entity] from the detector
)
```

### `Field`

Every scalar reading is a `Field`, never a bare float. The wrapper is the point: it is
what lets a guard refuse to act on a stale or unconfident value.

```python
Field(
    value=0.62,          # float | int | bool | str | None
    confidence=0.98,     # 0.0–1.0. Probes report match quality; detectors report score.
    source="probe:player_hp",
    updated_at=91234.5061,
)
```

`WorldState.get(name, default)` returns the value; `WorldState.field(name)` returns the
wrapper. Policy that cares about trust uses the latter. `Field.is_stale(now, max_age_s)`
is the single staleness predicate — do not reimplement it per call site.

### Field names for the FFXIV slice

Names are namespaced by subject. A game profile declares which it provides; the runtime
fails at startup if a policy references one that nothing produces, rather than silently
reading `None` at 60Hz.

| Field | Type | Produced by |
| --- | --- | --- |
| `player.hp_frac` | float 0–1 | bar probe |
| `player.mp_frac` | float 0–1 | bar probe |
| `player.in_combat` | bool | colour probe on the combat indicator |
| `player.casting` | bool | cast bar presence |
| `player.cast_progress` | float 0–1 | cast bar fill |
| `player.gcd_remaining_s` | float | hotbar cooldown-ring probe on the GCD slot |
| `target.exists` | bool | target frame presence |
| `target.hp_frac` | float 0–1 | target bar probe |
| `target.castbar_name` | str | OCR, low cadence |
| `action.<id>.ready` | bool | per-slot cooldown probe |
| `action.<id>.cooldown_s` | float | per-slot cooldown-ring probe |
| `buff.<id>.active` | bool | status-bar icon template match |
| `buff.<id>.remaining_s` | float | status-bar digit probe |

### Field names for the Genshin slice

Same namespacing discipline; a smaller vocabulary because Genshin's HUD exposes less.
The burst icon conflates "recharging energy" and "on cooldown" — both darken the icon —
so `action.burst.ready` means "can press Q now" without distinguishing why not.

| Field | Type | Produced by |
| --- | --- | --- |
| `player.hp_frac` | float 0–1 | overlay-tolerant bar probe (the HP numbers are drawn on the bar) |
| `boss.hp_frac` | float 0–1 | bar probe on the boss-bar slot; ~0 when no boss bar is up |
| `party.slot<N>.hp_frac` | float 0–1 | bar probes under the party portraits |
| `action.skill.progress` | float 0–1 | cooldown-sweep probe on the E icon |
| `action.burst.progress` | float 0–1 | darkness probe on the Q icon |
| `action.<id>.ready` | bool | derived |
| `action.<id>.cooldown_s` | float | derived |

One perception caveat worth knowing: the active HP bar changes colour as it drops
(green → orange → red), and the probe matches the healthy green. A red bar therefore
reads as 0.0 at reduced confidence — which policy treats the same as "very low", so the
failure direction is safe (dash and retreat, never press on).

### Field names for the FantCraft slice

Our own game (see its `docs/PLAYTEST.md`); the layout and hotbar fields come from the
manifest its client publishes, so the `action.*` names below are whatever abilities the
in-game hotbar currently holds.

| Field | Type | Produced by |
| --- | --- | --- |
| `player.hp_frac` | float 0–1 | overlay-tolerant bar probe ("100 / 100" is drawn on the bar) |
| `player.mp_frac` | float 0–1 | same, on the MP bar |
| `target.hp_frac` | float 0–1 | bar probe on the target frame; ~0 when the frame is hidden |
| `target.exists` | bool | derived: confident non-zero target HP reading |
| `action.<id>.progress` | float 0–1 | cooldown-overlay darkness per manifest hotbar slot (opt-in) |
| `action.<id>.ready` | bool | derived (opt-in) |
| `action.<id>.cooldown_s` | float | derived (opt-in) |
| `action.<id>.out_of_range` | bool | red-wash probe per slot — the game's range indicator |
| `action.<id>.combo_next` | bool | gold-glow probe per slot — the game's combo ring |

### Field names for the FantCraft API slice

The same game through its agent gateway instead of its screen (`player/api`,
`games/fancraft/api.py`). Fields come from a structured snapshot, so they are
exact where the screen slice's are estimates, and they add what no HUD shows:
`attack.land_in_ms` (countdown to the engine's landing tick), `attack.in_arc`
/ `attack.threatens` / `attack.guardable` / `attack.dodge_key`, `boss.phase`
(the boss challenge machine), `horse.*`, `marker.nearest_m` with an away
vector, `arena.edge_m`, `skill.<id>.ready`, `bag.<item>`, `ally.*` (the requester when
we are a summoned helper), `recent.*` (the boss's last landed blow, whose
follow-ups outlive its telegraph). The full table is the module
docstring of `games/fancraft/api.py`; the Warden tactics in
`games/fancraft/warden.py` are written against it. Confidence still means
something: our own position is dead-reckoned between corrections (0.9), a
skill the tree has not unlocked reads as not ready at 0.6.

### `Entity`

Detector output. Positions are `FramePoint` — converting to click targets goes through
`Geometry`, never by hand.

```python
Entity(
    kind="telegraph",        # telegraph | nameplate | marker
    bbox=(x, y, w, h),       # frame space
    confidence=0.87,
    attrs={"shape": "circle", "hostile": True},
)
```

## `Plan` — policy → act

What policy emits. Never a raw keypress: a plan is a *relative* schedule, and the timeline
resolves it against the current clock. That is what makes a plan replayable and testable.

```python
Plan(
    name="gcd:fire4",
    steps=[
        Step(at_ms=0,   action=Press(key="3", hold_ms=40)),
        Step(at_ms=620, action=Press(key="7", hold_ms=40)),   # weaved off-GCD
    ],
    priority=Priority.ROTATION,
    preempt=False,           # True flushes everything scheduled after now
    expires_in_ms=2500,      # dropped if not started by then; a stale plan is worse than none
)
```

A `Press` key may also name a mouse button — `mouse1`/`mouse2`/`mouse3` (aliases `lmb`,
`rmb`, `mmb`) — which the backend routes to button events with the same hold semantics.
That is what makes a Genshin charged attack expressible as `Press("mouse1", 700)`.

`Priority` ordering: `REFLEX` (100) > `RECOVERY` (75) > `ROTATION` (50) > `IDLE` (10).
A higher-priority plan with `preempt=True` cancels lower-priority scheduled events. Equal
priority never preempts — a rotation step does not cancel another rotation step.

`expires_in_ms` is what stops the "queued a dodge, dispatched it 400ms later, dodged into
the AoE" failure. A late action is not a slow success, it is a wrong action.

## `Directive` — strategy → policy

What Claude may say. Deliberately narrow: there is no directive that dispatches input.

```python
SetObjective(objective="maintain single-target rotation on dummy", reason=...)
SelectPlanProfile(profile_id="blm.single_target", reason=...)
SetParameter(key="opener.enabled", value=False, reason=...)
EnableReflexGroup(group="ground_aoe", enabled=True, reason=...)
Pause(reason=...)
```

Each carries a `reason` string. It is not decorative — it is what makes the session log
readable when the player does something surprising at 3am.

Directives are validated against the Pydantic union before reaching policy. `profile_id`
must exist in the registered catalog and `key` must be a declared parameter; unknown
values are rejected and logged, never coerced. An LLM that hallucinates a profile name
gets a dropped directive, not an undefined state.

## Coordinate spaces

Three distinct types in `player/geometry.py`, deliberately non-interchangeable:

```python
ScreenPoint(x, y)   # physical desktop pixels — what SendInput consumes
ClientPoint(x, y)   # relative to the game window client area — what HUD layout uses
FramePoint(x, y)    # captured buffer pixels — what perception computes on
```

Conversion only via `Geometry`, which holds the window client rect in screen space, the
DPI scale, and the capture downscale factor:

```python
geo.frame_to_client(p)   geo.client_to_screen(p)   geo.frame_to_screen(p)
geo.client_to_frame(p)   geo.screen_to_client(p)
```

HUD regions are authored as `RelRect(x, y, w, h)` with all four in 0–1 of client size, so
a calibration survives a resolution change at the same HUD scale.
