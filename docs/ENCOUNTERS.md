# Encounters: turning a raid guide into something the player can execute

## The reframe

The instinct is to treat "solve the raid mechanic" as a reasoning problem: show the model
what is happening, let it work out what to do. That framing is wrong, and it is wrong in a
way that makes the whole thing intractable.

An FFXIV encounter is **a scripted program with bounded randomness**. The boss does the
same things in the same order every pull, within about half a second. What varies per pull
is a small set of parameters: which quadrant is safe, who got the marker, which pair got
which debuff. A raid guide is not an explanation of a puzzle — it is *source code for that
program*, written in prose.

So the job is not inference. It is **compilation**.

    guide prose ──[LLM, once, offline]──> EncounterScript ──[human review]──> committed YAML
                                                │
                                                ▼
                              deterministic runtime execution

The model reads the guide once, at development time, and emits a data structure. The
runtime executes that data structure with no model in the loop at all. Understanding
happens once; execution happens ten thousand times.

## Why the model must stay out of the runtime loop

The tempting objection is that there is time: a boss cast bar runs 3–8 seconds, and the
damage usually snapshots about a second after it completes. An API call fits in that
window, most of the time.

"Most of the time" is the problem, and latency is only the fourth-worst thing about it.

- **It is unrepeatable.** A non-deterministic policy cannot be regression-tested. The
  entire replay harness — the thing that makes this project developable at all — stops
  meaning anything the moment a decision depends on a sampled model response.
- **It is unreliable in the tail.** p50 is fine and p99 is a wipe. Network jitter, a rate
  limit, or one retry and the mechanic has resolved without you.
- **It is expensive in exactly the wrong shape.** Every mechanic, every pull, every wipe.
  Progging a fight means running the same twenty mechanics several hundred times.
- **The information was already available offline.** You are paying inference to
  rediscover, under time pressure, a fact that was written down in a guide last year.

And some mechanics genuinely are tight — 0.5 to 1.5 seconds from tell to snapshot. Those
are unservable by anything with a network hop in it.

The model's runtime role in an encounter is therefore **post-mortem, not real-time**: after
a pull, given the recorded trace, explain what the script got wrong. That is
latency-insensitive, genuinely useful, and plays to what a model is actually good at.

## The `EncounterScript` contract

The compiled artifact. One per fight, committed to the repo, reviewable as a diff.

```python
@dataclass(frozen=True)
class Mechanic:
    id: str                  # "radiant_plume_1"
    trigger: Trigger         # how we know it has started
    deadline_ms: int         # from trigger to snapshot — a hard bound, not a hint
    resolve: Resolution      # how to compute where to be
    category: str            # spread | stack | dodge | tower | tether | knockback | positional
    notes: str = ""          # the guide sentence this came from, verbatim
```

`notes` is not decoration. When a mechanic fails, the first question is always "did the
script get this wrong, or did the player execute it wrong" — and having the source
sentence sitting next to the compiled rule answers it in seconds instead of sending you
back to the guide.

### Triggers

```python
CastTrigger(name="Radiant Plume")        # OCR on the boss cast bar. Primary.
TimelineTrigger(at_s=182.4, window_s=2)  # seconds since pull, for silent mechanics
MarkerTrigger(kind="tower")              # a visual appeared
DebuffTrigger(id="light_resistance_down") # I got a thing
PhaseTrigger(boss_hp_below=0.5)          # phase transitions
```

Cast triggers do double duty: they also **re-anchor the timeline clock**. A pure
timeline-based script drifts — a slow kill, a phase transition on HP rather than time, a
stun — and drift compounds until every subsequent mechanic fires at the wrong moment.
Every named cast is a resync point, so drift never accumulates past one mechanic.

### Resolutions — a closed vocabulary

This is the load-bearing design decision. `Resolution` is a **small, fixed vocabulary the
model selects from**. It never emits code, never emits an expression, never emits
coordinates it made up.

```yaml
# Get out of the telegraphed area, whatever shape it is.
resolve: {kind: avoid_telegraphs, clearance_m: 2.0}

# Stand on a waymark.
resolve: {kind: waymark, id: "A"}

# Behind the boss, 3m out. Positional uptime.
resolve: {kind: relative_to_boss, angle_deg: 180, distance_m: 3.0}

# Get away from everyone.
resolve: {kind: spread, min_separation_m: 5.0, prefer: {kind: waymark, id: "A"}}

# Get on top of the marked player.
resolve: {kind: stack_on, target: marked_player}

# Nearest unsoaked tower.
resolve: {kind: nearest_entity, entity: tower, exclude: {debuff: already_soaked}}

# Knockback: brace, or pre-position to be knocked somewhere survivable.
resolve: {kind: knockback, from: boss, land_at: {kind: waymark, id: "1"}}
```

Each kind is implemented once, in Python, tested in isolation. Adding a kind is a
deliberate act by a human. That containment is the same principle as the directive
vocabulary in `docs/CONTRACTS.md`: **the model's output space is a menu, not a language.**
A hallucinated resolution fails schema validation. A hallucinated *destination* is
impossible, because destinations are computed from perception at runtime, not stated in
the script.

### Genuine puzzles

Some mechanics really are combinatorial — debuff-pair sorting, light/dark partner
assignment, priority-based tower assignment. These do not fit a positional vocabulary.

They get an escape hatch that is still closed:

```yaml
resolve:
  kind: solver
  solver: pair_by_opposing_debuff        # a name from a registry
  params:
    debuff_a: light_resistance_down
    debuff_b: dark_resistance_down
    destinations: ["A", "B", "C", "D"]
```

The solver registry is a dict of hand-written Python functions. The model **names** one and
supplies parameters; it does not author one. A fight that needs a solver nobody has written
yet is a fight the player cannot do — which is the correct answer, and an honest one,
rather than a plausible-looking guess.

## What perception has to grow

This is the real cost, and it is much larger than the scripting work. Everything in the
current build reads the HUD. Encounters need **world state**.

| New field | Source | Why it is hard |
| --- | --- | --- |
| `boss.cast_name` | **Closed-set classification**, not OCR — see below | Needs signatures learned from a recording |
| `encounter.elapsed_s` | Combat-start detection + resync | Easy once cast names work |
| `arena.player_pos` | Minimap | Low resolution — see below |
| `arena.camera_yaw` | Minimap compass | Needed for *every* movement command |
| `arena.waymarks` | Minimap + 3D view | The anchor for arena coordinates |
| `party.positions` | Minimap dots | Needed for stacks and spreads |
| `self.debuffs` | Status bar templates | Already scaffolded, not yet calibrated |
| `arena.markers` | Detector | Towers, tethers, headmarkers |

**The minimap is the single highest-value new sensor.** It is a fixed HUD region that gives
player position, facing, party dots, and arena bounds in one read. Nearly every positional
mechanic is servable from it.

### Measured, not assumed

`tools/measure_localization.py` sweeps positions, camera yaws and zoom levels against known
ground truth. Current synthetic result:

| | p50 | p95 | max |
| --- | --- | --- | --- |
| Position error | 0.014 m | 0.036 m | 0.119 m |
| Camera yaw error | 0.03° | 0.09° | 0.16° |

**TIER: TIGHT** — supports tight positionals and exact tower soaks. Far better than the
naive pixel-resolution estimate (a minimap is ~150px for a 40m arena, about 0.27 m/px)
suggests, because position is *fitted* from several correspondences at once rather than
read off one.

Three things had to be right to get there, and each was wrong first:

* **Rotation sign.** The renderer and the localiser disagreed about which way the minimap
  turns. Both now go through `CameraPose.to_camera_relative`, so there is one definition
  and a second cannot silently drift from it.
* **Clipped icons.** A waymark at the minimap edge is cut off, so its visible pixels are a
  biased sample and its centroid is pulled inward by metres. Edge-touching blobs are now
  discarded — one fewer correspondence beats one poisoned one.
* **Half-pixel bias.** Pixel *index* is not pixel *centre*. This one is invisible to the
  residual because it shifts every mark together: the fit stays clean while the answer is
  consistently wrong. Worth 0.2m on its own.

Treat these as a **lower bound**. A synthetic sweep measures the solver, not the game —
real minimaps have occlusion, party dots over waymarks, and varying terrain. Re-run with
`--session` against a real recording before trusting them.

### The cast bar is not an OCR problem

The instinct is OCR, and OCR is the wrong tool: stylised fonts, variable widths, a
translucent bar over a moving 3D scene, and a hard latency budget. General text recognition
is both the hardest option here and the least reliable.

But the question is not "what does this say". An `EncounterScript` **declares every cast
name it cares about**, so the real question is "which of these twelve known casts is this,
or none of them" — closed-set classification, which is a far easier problem.

The signature is a normalised column profile of the binarised text: how much ink stands in
each horizontal slice. Cheap, robust to the bar's own fill sweeping underneath, resampled
to a fixed width so one signature works across resolutions, and discriminative enough for a
dozen names because cast names differ in length and letter distribution far more than they
resemble each other.

Signatures are **learned from recordings**, not shipped — calibration is a replay, not a
data-entry exercise. Two refusals keep it honest: a match below the similarity floor is
rejected, and so is one that fails to beat its runner-up by a margin. Reporting nothing
lets `CastTrigger` fall through to the timeline trigger, which is late but correct; guessing
between two similar names fires the wrong mechanic.

### Waymark ambiguity is real

FFXIV's two waymark families **share colours**: A and 1 are both red, B and 2 both yellow.
A colour-keyed detector can narrow a blob to a pair and no further, so it emits
*candidates* and the localiser settles them by geometry — only one assignment admits a
consistent rotation and scale.

Except when it does not. The standard ring puts A/B/C/D on the cardinals and 1/2/3/4 on the
intercardinals, which makes the lettered set **the same shape rotated 45°**. Both
assignments fit *exactly*; the residual cannot choose.

Physics can. The character did not teleport and the camera did not spin 45° in one frame,
so the previous estimate breaks the tie. With no history to lean on, the result is flagged
`ambiguous` and its confidence drops sharply — an honest "I am not sure which way round
this is" rather than a confident coin flip.

Waymarks are the natural coordinate anchor precisely because players place them for the
same reason: they are deliberately positioned, visually distinct, and the guide is already
written in terms of them ("go to A", "stack on 1"). A script written in waymark space
needs no absolute arena calibration at all.

## Two loops, one body

The rotation loop and the mechanic loop run **at the same time** and contend for the same
character. How they are coupled is the single most consequential decision in this design.

The obvious coupling is preemption: when a mechanic needs to act, it interrupts the
rotation. It works, and it is purely reactive — the rotation starts a 2.8-second cast, the
mechanic fires 1.8 seconds later, the cast is cancelled and the damage is simply gone.

The better coupling is **anticipation**. A scripted encounter knows its own schedule
seconds ahead — that is what "scripted" means — so instead of interrupting, the mechanic
loop publishes claims on future time and the rotation plans inside the gaps.

```
mechanic loop ──publishes──▶  CommitmentBoard  ◀──reads──  rotation loop
                              MOVING 101.5–104.0s
                              GCD_RESERVED 102.2–103.0s (surecast)
                              CAMERA_AWAY 100.0–103.0s
```

The rotation asks one question — *how long can a cast safely run right now?* — and picks
the highest-priority ability that fits. If movement starts in 1.8 seconds it takes an
instant rather than a hard cast, and loses nothing. When the reserved window arrives it
hands the GCD over without being asked twice.

`CommitmentKind` covers what a claim actually prevents:

| Kind | Blocks |
| --- | --- |
| `MOVING` | Hard casts, which would be cancelled |
| `GCD_RESERVED` | The rotation spending this GCD; the mechanic's ability uses it |
| `OGCD_RESERVED` | Weaving into that window |
| `CAMERA_AWAY` | Anything depending on the boss being framed |
| `NO_TARGET` | Everything — phase transition, boss untargetable |

An empty board reports an infinite cast window, so a striking-dummy session behaves
exactly as it did before any of this existed. The coupling costs nothing when unused.

Two counters make the interaction visible in a session report: `clips_avoided` (hard casts
declined because movement was coming) and `yields` (GCDs handed to a mechanic). A rotation
quietly clipping every cast looks identical to one that is simply performing badly, unless
you count.

## The mechanic loop, phase by phase

    IDLE ──trigger──▶ OBSERVE ──framed──▶ RESOLVE ──target──▶ POSITION
                                                                  │
                IDLE ◀──uptime──── RECOVER ◀──ability──── ACT ◀───┘

Five phases, because a mechanic is five distinct problems and conflating them hides which
one failed. "Died to the tower" is not a bug report; "the camera never framed the tower, so
OBSERVE spent its budget and RESOLVE ran blind" is.

* **OBSERVE** — point the camera at the information. This is why `LookPlan` is part of the
  mechanic definition rather than an afterthought: in a 3D game you cannot resolve what you
  have not looked at, and looking costs time that has to be budgeted. The camera then holds
  still for `settle_s`, because a detector fed frames captured mid-slew does badly at
  exactly the moment its answer matters.
* **RESOLVE** — bind the mechanic's random parameters from what is now visible.
* **POSITION** — walk there, closed-loop, publishing `MOVING` so the rotation adapts.
* **ACT** — press what the mechanic requires. The GCD was reserved seconds ago.
* **RECOVER** — return to an uptime position so the rotation stops being penalised.

Phase transitions chain **within a single tick**. Three transitions at 60Hz would be 50ms,
and a mechanic with a one-second window cannot spend it changing its mind about which phase
it is in.

Each phase has a slice of the deadline, and blowing the budget advances anyway rather than
hanging — a mechanic that never frames its target should still try to resolve. Losing
localisation mid-mechanic abandons rather than guessing, because a mechanic resolved
against wrong coordinates walks into the thing it was dodging.

## Targeting: two problems that look like one

"Point at the right thing" splits into two problems with different machinery, and the
constraint they share is brutal: **a target swap has to be as fast as the next skill use.**
A healer whose tank drops to 20% cannot spend one GCD acquiring the tank and another
casting on them, and an add that must die in eight seconds cannot afford a swap costing two
of them.

So targeting is not a phase, a plan, or a state. It is a **prefix on the plan that needs
it** — the target key and the ability key are steps in the same schedule, tens of
milliseconds apart. They are preempted together, expire together, and cannot be separated
by a reflex firing in between, which is the failure that otherwise lands a heal on the boss.

    Plan.single("cure2").targeted(("f2",), settle_ms=60)
    ├─ t+0ms   press F2      (acquire party slot 1)
    └─ t+60ms  press cure2

Sixty milliseconds, against a 2500ms GCD.

### Planned: add phases

An add spawning on a known timeline is a script lookup. It publishes a `TARGET_OVERRIDE`
commitment, and — deliberately — **not** a GCD reservation: the rotation keeps choosing its
own abilities and simply aims them elsewhere. Taking the GCD would throw away the
rotation's knowledge of what actually does damage.

The override redirects abilities that would have hit the current target, and leaves heals
alone. Pointing a regen at the add is worse than useless.

### Reactive: healing

Who needs a heal depends on how badly everyone else played, so no guide can predict it.
This is a policy over live party state — and it needed no new machinery, because of one
design decision: `Ability` carries a `TargetSpec`.

That makes a healer profile **an ordinary priority list**. Heals are abilities whose target
names an ally and whose condition reads party HP; triage is list order. `party.lowest_hp_frac`
and `party.count_below_N` are derived fields, so a heal rule is an ordinary `Condition`.

The elegant part is what falls out for free: `mouseover_lowest_ally(hp_ceiling=0.65)`
**fails to resolve when nobody is below 65%**, the ability is skipped, and the list falls
through to damage. "Do not heal a healthy party" needs no code at all.

Two things still needed care:

* **Mouseover, not slot keys.** A slot key changes your *hard target*, so the next Glare
  lands on the ally you just healed and does nothing — you would have to re-target the boss
  after every heal. Mouseover leaves the boss selected throughout. This is exactly why it is
  the standard among FFXIV healers, and the model keeps the distinction (`Plan.hovered` vs
  `Plan.targeted`) rather than treating both as "targeting". Without a configured locator
  the resolver falls back to slot keys and says so in its reason string.
* **Emergencies bypass the GCD loop.** Someone at 15% cannot wait up to 2.5s for the next
  rotation decision, so that one lives in the reflex layer, which already preempts. It uses
  `RECOVERY` priority, not `REFLEX`: a dodge outranks a heal, because being alive and
  unhealed beats being healed inside an AoE.

Cast times interact correctly without special-casing. With movement 1.8s away the healer
takes the instant heal over the 2s hard cast; with only 1s and no instant available it
casts **nothing**, because saving the GCD beats wasting it.

## Navigation — the genuinely new subsystem

Every resolution above outputs *a place to be*. Nothing in the current build can get there.
This is the part that is not scripting, not perception, and not in the codebase.

```
target (arena space) ──> error vector ──> rotate by -camera_yaw ──> W/A/S/D
                             ▲                                        │
                             └──────────── minimap position ◀─────────┘
```

A proportional controller, closed every frame. FFXIV movement is camera-relative — `W` is
away from the camera, not north — which is why `camera_yaw` is load-bearing rather than
nice to have. Get it wrong by 90° and the player walks confidently into the wall.

**This does not fit the plan/timeline model, and pretending otherwise would be a mistake.**
A `Plan` is a discrete schedule of presses; movement is a continuously corrected held
state. So `MovementController` owns the four movement keys directly and reconciles them to
the desired direction each tick. It still passes through `SafetyGate` — every guard applies,
and a tripped guard releases the movement keys like any others — but it is a second channel
alongside the timeline, not a user of it.

That is an architectural addition, and it is the one place the current design has to open
up rather than extend.

## Validation: the part that makes this trustworthy

Here is where the record/replay infrastructure pays for itself.

**A compiled script can be validated against a recorded pull before it is ever used live.**

1. Record a pull — yours, or a friend's, or a VOD if the resolution is sufficient.
2. Replay it with the script loaded, in a dry-run mode that dispatches nothing.
3. For every mechanic: did its trigger fire, and did it fire at the right moment?
4. For every resolution: what destination did it compute, and where was the recording's
   player actually standing when it snapshotted?

That produces a per-mechanic report — trigger recall, trigger timing error, destination
error in metres — from a recording, with nobody in the instance. A script that scores badly
gets fixed at a desk instead of at the cost of seven other people's evening.

This is also the answer to "how do you know the LLM compiled the guide correctly": you do
not trust it, you test it. The compilation step is allowed to be imperfect because there is
a cheap, automatic, offline check downstream of it.

## Failure modes, and the honest ceiling

| Failure | What happens | Response |
| --- | --- | --- |
| Cast name OCR misses | Mechanic never triggers | Timeline trigger as fallback; log the miss |
| Timeline drifted | Right mechanic, wrong moment | Re-anchor on the next cast |
| Unknown mechanic | Nothing in the script matches | **Stop and hold position.** Do not improvise. |
| Minimap position wrong | Confidently walks somewhere wrong | Confidence guard on `arena.player_pos` |
| Deadline missed | Arrived late | Report it; a late arrival is a failure, not a partial success |
| Solver not implemented | Script will not compile | Refuse to load the encounter |

The "unknown mechanic" row is the important one. The correct behaviour when the player does
not recognise what is happening is to **stand still**, not to guess — same principle as an
uncalibrated rotation doing nothing rather than casting semi-randomly.

And the ceiling worth stating plainly: **this will never be good at progging.** It can
execute a script someone already wrote. It cannot work out a fight nobody has solved yet,
and it will be worse than a competent human at anything requiring adaptation mid-pull. The
target is farming known content, not clearing new content.

## Build order

Each phase is independently useful and independently verifiable. Phase 1 is the gate — if
minimap localisation is not accurate enough, everything above it is unbuildable and it is
much better to find that out first.

1. **Minimap sensor + accuracy measurement.** Position, camera yaw, party dots. Measure
   error in metres against known positions before writing anything that depends on it.
2. **Cast-bar OCR + timeline tracker.** Cast names, elapsed time, resync on cast.
3. **`MovementController`.** Navigate to a waymark on command, measure arrival error and
   time. Testable solo in an empty instance with no encounter at all.
4. **`EncounterScript` runtime**, with two resolution kinds only (`waymark`,
   `avoid_telegraphs`) and the offline validator.
5. **The compiler.** Guide text → script, with human review. Only worth building once the
   runtime can execute a hand-written script, so there is something to compile *to*.
6. **More resolution kinds**, driven by which mechanics actually fail in validation.
7. **Solvers**, one at a time, for specific named mechanics.

Note that the LLM shows up at step 5, not step 1. Everything before it is perception and
control, which is where the difficulty actually lives.

## One thing worth deciding deliberately

A striking dummy fails alone. A raid fails eight people.

Everything in `docs/SAFETY.md` about the User Agreement still applies unchanged, but there
is a second consideration here that does not exist for the dummy slice: when this goes
wrong in a party — and while it is being developed it will go wrong constantly — the cost
lands on other people who did not agree to it. That is a different question from the
account-risk one, and it is worth answering on purpose rather than by default.

Solo content, old content run solo, and a static that knows and agrees are all
straightforward. Party finder with strangers is the case worth thinking about before
building rather than after.
