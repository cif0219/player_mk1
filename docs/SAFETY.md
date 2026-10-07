# Safety

## Non-goals, stated up front

These are deliberate exclusions, not unfinished work. Do not add them.

- **No process injection, memory reading, or DLL loading.** Perception is screen pixels.
- **No packet inspection or modification.**
- **No anti-cheat evasion.** No timing jitter whose purpose is to look human, no process
  name masking, no detection-vector research.
- **No account or credential handling.** The player never sees a login screen; if the
  client is not already logged in and in-world, the player refuses to start.
- **No multiplayer-affecting content in the slice.** The vertical slice targets a striking
  dummy in a private space. Whatever the player does badly, it does to a training dummy.

The first three exist because "make automation undetectable" is a different project with
a different ethics profile, and mixing it in here would compromise the parts that are
about perception and control.

## Terms of service

Square Enix's FFXIV User Agreement prohibits third-party software that automates
gameplay, and enforcement is account termination. This tool cannot be run against a live
account without violating it. That is the user's call to make about their own account —
it is documented here so the call is made knowingly rather than discovered later.

## The guard chain

Every dispatch passes through `SafetyGate` in `player/safety/guards.py`. Guards are
checked **at dispatch time**, not sampled into a flag that a worker reads later — the
window between "guard tripped" and "next keypress" has to be one dispatch, not one poll
interval.

| Guard | Trips when | Why |
| --- | --- | --- |
| `KillSwitchGuard` | Kill key pressed | Human override, must always win |
| `ForegroundGuard` | Target window is not foreground | Prevents typing into the browser you just alt-tabbed to |
| `StalenessGuard` | `WorldState` older than `max_age_ms` | Acting on a stale world is acting blind |
| `ConfidenceGuard` | Required fields below threshold | A misread bar is worse than no read |
| `RateGuard` | Dispatches exceed `max_per_sec` | Backstop against a runaway loop |
| `TakeoverGuard` | Human moved mouse or typed recently | The human is driving; get out of the way |

A tripped guard **stops dispatch and flushes the timeline**. It does not queue for later:
a plan built for a world state that has since expired is not worth executing when the
guard clears.

`SafetyGate.status()` reports which guards are currently blocking, and the runtime prints
this on the status line, so "why is it not doing anything" is answerable at a glance.

## The kill switch

`player/safety/killswitch.py` installs a Windows `WH_KEYBOARD_LL` low-level hook. This is
the only mechanism that works when the game has focus and is swallowing input — a
focus-scoped listener does not, which makes it useless for exactly the situation you need
it in.

Properties it must keep:

- **Latching.** Once tripped it stays tripped until explicitly resumed. A momentary trip
  that auto-clears is not a kill switch.
- **Independent of the decision loop.** The hook sets a flag on its own thread; a hung
  perception thread cannot prevent the kill switch from stopping dispatch.
- **Checked before every dispatch**, including events already scheduled on the timeline.

If the hook cannot be installed (non-Windows, permissions), the runtime **refuses to start
in live mode**. Running the dispatcher with no way to stop it is not a degraded mode worth
supporting. `--dry-run` still works everywhere.

## Human takeover

If the human moves the mouse or presses a key that the player did not dispatch,
`TakeoverGuard` blocks dispatch for `takeover_cooldown_ms`. Distinguishing the player's
own synthetic input from human input uses an expected-event set that the dispatcher
populates just before sending, matched by key and time window.

This is a usability feature more than a safety one, and it is the guard most likely to be
tuned per user. It is also the one that makes the player pleasant to supervise: you can
grab the mouse without racing it.

## Failure modes and intended responses

| Failure | Response |
| --- | --- |
| Capture returns no frame for `>500ms` | Staleness guard trips; timeline flushes |
| Window moves or resizes | `Geometry` recomputed from the window rect; probes reprojected |
| Window closes | Runtime stops |
| A required sensor throws | Field marked `confidence=0`; confidence guard trips if required |
| Detector unavailable (no ONNX runtime) | Entity list empty; reflexes depending on it disable themselves and say so at startup |
| LLM call fails or times out | Director keeps the previous objective; no directive applied |
| Token budget exhausted | Director stops; deterministic layers continue unchanged |

The pattern: degrade to *doing less*, never to *guessing more*.
