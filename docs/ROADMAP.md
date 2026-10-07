# Roadmap

Phases are ordered so each one is independently useful and independently verifiable. The
slice is not "done" until phase 3 has numbers attached.

## Phase 0 — substrate (this commit)

Everything below policy, plus a runnable dry-run path.

- [x] Coordinate spaces, clock with per-stage latency histograms
- [x] `WorldState` with per-field confidence and staleness
- [x] Capture: screen source, replay source, window tracking
- [x] Record: session writer, session reader
- [x] Probes: bar, colour, template, cooldown ring
- [x] Deadline-scheduled input timeline; null and SendInput backends
- [x] Safety: kill switch, guard chain
- [x] Policy: reflexes, rotation planner with weave-window modelling
- [x] Strategy: typed directives, Claude director with a cost ceiling
- [x] FFXIV profile with one job rotation; calibration tool
- [x] Headless tests for geometry, timeline, policy, guards, replay

## Phase 1 — make perception real

Substrate is written against the FFXIV HUD but has not been fitted to it.

- [ ] Calibrate the HUD layout against a real client at 2560×1440 and 1920×1080
- [ ] Verify probe accuracy: record a session, hand-label 200 frames, measure per-field error
- [ ] Cooldown-ring probe validation — the hardest probe, and the one the rotation depends on
- [ ] Digit OCR for buff timers (template set per HUD scale)
- [ ] Perception regression test fixture from a recorded session

Exit criterion: `player.gcd_remaining_s` within ±50ms of ground truth on 95% of frames.

## Phase 2 — make the rotation correct

- [ ] Encode one job's single-target priority list completely
- [ ] Opener as an explicit scripted plan, separate from the steady-state priority list
- [ ] Weave-window scheduling validated against recorded animation-lock timings
- [ ] Rotation scorer: replay a session and diff the executed sequence against the reference

Exit criterion: ≥95% rotation accuracy against the reference list over a 5-minute dummy
parse, measured offline from a recording.

## Phase 3 — make reaction real

- [ ] Weak-label telegraphs from recordings (`tools/label.py`)
- [ ] Train the first detector on weak labels; evaluate on hand-labelled holdout
- [ ] Ground-AoE reflex: detect overlap with player position, emit a directional dodge
- [ ] Measure reaction latency: telegraph-visible timestamp → key-dispatched timestamp

Exit criterion: p95 reaction latency under 150ms, measured from recordings with the
telegraph appearance frame identified by the detector.

## Phase 4 — encounters

Executing scripted raid mechanics. This is the largest phase by a wide margin and has its
own design document: [ENCOUNTERS.md](ENCOUNTERS.md).

The short version: a raid guide is compiled **offline** into an `EncounterScript` and
executed deterministically, with no model in the runtime loop. The difficulty is not the
reasoning — it is world-space perception (minimap localisation, cast-bar OCR) and
navigation, neither of which exists yet.

- [ ] Minimap sensor, and **measure its accuracy in metres before building on it**
- [ ] Cast-bar OCR and the timeline tracker with cast-based resync
- [ ] `MovementController` — a second dispatch channel, not a timeline user
- [ ] `EncounterScript` runtime with two resolution kinds and the offline validator
- [ ] The guide compiler, once there is a runtime to compile *to*

Step 1 is the gate. If minimap localisation is not accurate enough, everything above it is
unbuildable, and that is much better to discover first than fifth.

## Phase 5 — generalise, carefully

Only after the slice works end to end. Generalising before that is how the previous
version of this project ended up with an architecture and no game.

- [ ] Second job profile, to find out what in the rotation engine was actually job-specific
- [ ] Second game profile, to find out what in `games/ffxiv/` was actually FFXIV-specific

## Phase C — FantCraft: API control and the Ironclad Warden (2026-09-23)

The game we own hands the tester a second door: its agent gateway. Same
`WorldState`, reflexes, timeline and guard chain; a snapshot instead of pixels,
intents instead of keystrokes.

- [x] `player/api`: gateway client, `ApiSource`, `SnapshotSensor`, `ApiBackend`, `ApiTransport`
- [x] `games/fancraft/api.py`: field contract (`attack.*`, `boss.*`, `horse.*`, `marker.*`, `arena.*`, `skill.*`), intent keys, fight ledger
- [x] `games/fancraft/warden.py`: the Ironclad Warden as reflexes — parry faced at the attacker, evades for Wind Cutter and Great Serpent, wall-aware marker dodging, the pulse wall, provoke on the horse
- [x] fancraft `npm run smoke:warden-bot`: clears phase 1, holds the horse, survives a set of phase 2
- [x] Found and fixed on the way: preemption dropped owed key releases (a stuck W); the engine measured the guard arc to the blade tip
- [ ] Three sets of phase 2 without a death (the cloud → slow → lightning chain)
- [x] The kneel: Supreme Holy Water bought at the free counter and thrown on the kneel (`purify` reflex)
- [x] Party play: the game summons helpers through the gateway's desk; tasks (`games/fancraft/tasks.py`) switch reflex groups; `hold_horse` kites the mount
- [x] `fancraft.party`: several characters from one process (one on the Warden, one on the horse)
- [x] `dodge_chance`: take a share of the far-reaching cuts on purpose, to measure the boss
- [ ] A helper that heals or shields the requester (no such kit on the guardian yet)

## Explicitly out of scope

- Anything under "Non-goals" in [SAFETY.md](SAFETY.md)
- Reinforcement learning. The action space is fine to write by hand and the reward signal
  is expensive to collect; a priority list is better engineering here.
- A GUI. The status line and the session log are the interface.
