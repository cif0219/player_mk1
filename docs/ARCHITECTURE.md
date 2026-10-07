# Architecture

## The latency budget, measured rather than asserted

Design to the budget the game actually imposes, not to a round number.

FFXIV's global cooldown is ~2.5s. The tight constraint is the off-GCD weave window: after
a GCD you have roughly 600ms of animation lock, then ~600–700ms in which an off-GCD
ability must land or the weave is lost. Telegraphed ground AoEs give 1–3s from cast-start
to snapshot. So the real requirement is a **stable end-to-end path under ~100ms**, not a
sub-millisecond one.

| Stage | Budget | Where it is spent |
| --- | --- | --- |
| Capture | 5–10ms | `mss` grab of the client rect |
| Perception (probes) | 3–8ms | One numpy pass over the frame for all fixed-region probes |
| Perception (detector) | 5–15ms | ONNX inference, runs at a lower cadence than probes |
| Decide | <1ms | Reflex scan + rotation planner over `WorldState` |
| Dispatch | 1–2ms | Timeline pop + `SendInput` |
| **Controllable total** | **~15–35ms** | |
| Game render + server ack | 30–100ms | Not ours. Do not pretend to control it. |

`player/clock.py` records a histogram per stage and the CLI prints p50/p95/p99 on exit.
A budget you do not measure is a budget you do not have.

## Why `WorldState` is the contract

The obvious design has each rule check its own pixels: rule A samples the health bar,
rule B samples the same bar again, rule C samples a hotbar slot. That is the design the
earlier version of this project had, and it has three compounding problems.

1. **N rules means N passes over the frame.** Each rule re-slices the array. Cost scales
   with rule count instead of with frame size.
2. **Rules become untestable.** Testing "use a potion below 30% HP" requires synthesising
   an image with the right pixels in the right place, at the right resolution.
3. **UI layout leaks into strategy.** Move the hotbar and every rule that referenced it
   breaks, even though nothing about the strategy changed.

Instead, perception runs **once per frame** and produces a typed `WorldState`. Policy
reads fields off it. The consequences are worth the indirection:

- Swapping a pixel probe for an ML detector for a given field is invisible to policy.
- A reflex rule is a pure function of `WorldState`, so its test is three lines and no image.
- Replaying a recorded session re-runs perception and diffs the resulting `WorldState`,
  which is how perception regressions get caught before they reach the game.

`WorldState` carries `captured_at` and a per-field `confidence`. Policy that acts on a
stale or low-confidence field is a bug, and `SafetyGate` refuses to dispatch when the
state is older than the configured staleness bound.

## Three coordinate spaces, made unmixable

The single most productive bug source in screen automation is mixing up spaces. There are
three, and they are genuinely different:

- **Screen** — physical desktop pixels. What `SendInput` and the OS cursor use.
- **Client** — pixels relative to the game window's client area. What the HUD layout is
  defined in, and what survives the user moving the window.
- **Frame** — pixels in the captured buffer, which may be downscaled for the detector.
  What perception computes on.

`player/geometry.py` gives each its own type. They do not implicitly convert; you go
through a `Geometry` object that knows the window rect, DPI scale, and capture scale.
Passing a `FramePoint` where a `ScreenPoint` is expected is a type error, not a
mysteriously-off-by-a-scale-factor click.

HUD regions are authored in **client** space as fractions of client size, so a layout
calibrated at 2560×1440 still works at 1920×1080 as long as the HUD scale matches.

## The scheduled input timeline

The naive executor is a queue plus a worker that sleeps a fixed delay after each action.
That design has a fatal property: it serialises everything behind the slowest action, so
a "1ms reflex" is destroyed by the 30ms sleep left over from the previous keypress.

`player/act/timeline.py` is a **deadline-scheduled priority queue** instead. Every entry
is `(due_at_monotonic, priority, event)` and key-down / key-up are separate scheduled
events. This buys three things that matter for this workload:

- **Precise holds.** "Press `2` for 40ms" is two scheduled events, not a blocking sleep.
- **Weave windows fall out naturally.** The rotation planner schedules the GCD at `t`, the
  animation lock until `t+600ms`, and an off-GCD at `t+620ms`. The timeline just runs it.
- **Cancellation is meaningful.** A reflex firing with `preempt=True` flushes everything
  scheduled after `now` and inserts a dodge. With a sleeping worker there is nothing to
  flush — the sleep has already happened.

The dispatch thread only ever waits until the next deadline, so an empty timeline costs
nothing and a full one has no drift accumulation.

## Concurrency

Four threads, chosen so that no lock is held across an I/O boundary.

| Thread | Rate | Owns |
| --- | --- | --- |
| Capture | frame rate | The capture device. Writes newest frame into a triple-buffered slot. |
| Perceive+decide | frame rate | Sensors, `WorldState`, reflexes, rotation. Emits plans. |
| Dispatch | deadline-driven | The timeline and the input backend. |
| Strategy | event-triggered | The Claude director. Never blocks anything. |

Perception and decision share a thread deliberately: together they are well under 20ms,
and merging them removes a queue hop and a lock from the hot path. Splitting them would
buy parallelism this workload does not need and cost latency it cannot spare.

The frame slot holds only the newest frame. A list of the last N frames is the wrong
structure — a decision made on a frame two frames old is a worse decision, so there is
never a reason to read anything but the latest.

## Perception: probes for fixed things, a detector for found things

Both matter, for different reasons, and confusing them wastes effort in both directions.

**Probes** (`perceive/probes.py`) are numpy operations on fixed calibrated regions:
bar fill fraction, colour match, template correlation, seven-segment digit read. They
handle HP/MP, cast bar progress, target HP percentage, and hotbar cooldown/proc state.
These are at known positions and a trained model would be strictly worse — slower, less
accurate, and needing data to learn something a rectangle already knows.

**The detector** (`perceive/detector.py`) is an ONNX model for things at unknown
positions: ground AoE telegraphs anywhere in the 3D field, enemy nameplates, player
position relative to a boss. This is where ML earns its place.

The bootstrap for detector training data is the useful part. FFXIV telegraphs are
strongly saturated orange/red ground decals, so HSV segmentation produces **weak labels
for free** from any recorded session. Train on those, and the detector generalises to the
partially-occluded and edge-of-screen cases segmentation misses. `tools/label.py`
implements the weak-labelling pass over recordings; no manual annotation is needed to get
a first model.

The detector runs at 15Hz while probes run at frame rate. Telegraphs last 1–3s, so
inferring on every frame spends GPU for no information.

## Record and replay is not optional

You cannot iterate on a game bot without deterministic offline replay. This is the reason
the previous version of this project never played a game: every experiment required the
game running, in the right place, in the right state.

`player/record/` writes each session as frames plus a JSONL trace of `WorldState`,
decisions, and dispatched events, all on one monotonic clock. `ReplaySource` is a
`CaptureSource`, so the entire pipeline above it runs unchanged. That gives:

- **Perception regression tests.** Re-run sensors over recorded frames, diff `WorldState`
  against the recorded one, fail on drift.
- **Policy regression tests.** Feed the recorded `WorldState` sequence to the policy and
  assert the same decisions. No frames needed, so these run in milliseconds.
- **Latency forensics.** Every recorded event carries its stage timestamps.
- **Detector training data**, via the weak-labelling pass.

One caveat worth knowing before it confuses you: **policy timing is real-time, so
accelerated replay compresses the game but not the planner.** By default `ReplaySource`
delivers frames as fast as they decode, which means a 60-second session streams past in a
few seconds while the rotation planner is still holding a 2.5s GCD commitment — and you
see one decision where the live run made twenty. That is correct behaviour, not a bug.
Use `--realtime` to reproduce the original pacing, or drive the planner with an explicit
`at=` on a virtual clock, which is what the tests do.

There is also `tools/synth.py`, which renders a crude HUD into the layout's own regions
and records it as a normal session. It answers "is the pipeline wired up" separately from
"are the regions calibrated" — debugging both at once is much harder than either alone.

## The LLM is a director, not a driver

The strategic layer runs Claude on an **event trigger** — combat start/end, an unhandled
state, a stalled objective — rather than a fixed interval. A 5s timer burns tokens
describing a striking dummy that has not changed.

Two constraints make it safe and cheap:

**It cannot emit input.** The directive vocabulary (`strategy/directives.py`) lets it
select a plan from a registered catalog, set parameters, enable or disable a reflex group,
or set an objective. There is no directive that means "press this key." A hallucinated
directive fails schema validation and is dropped; the worst case is that the player keeps
doing what it was already doing.

**It is schema-enforced and budgeted.** Directives come back through Anthropic structured
outputs against a Pydantic model, so parsing cannot fail silently. A per-session token
ceiling stops the loop rather than quietly overspending, and the compact state summary
plus a downscaled screenshot keeps each call small.

Everything below the director is deterministic. That is what makes replay meaningful — a
non-deterministic policy cannot be regression-tested.
