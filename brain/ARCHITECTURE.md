# Player MK1 - Brain Architecture

## Three-Layer Decision System

```
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGIC LAYER                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LLM (slow, ~1-5s)                                   │   │
│  │  • Define high-level goals                           │   │
│  │  • Analyze game state periodically                   │   │
│  │  • Adjust tactical priorities                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▼                                 │
├─────────────────────────────────────────────────────────────┤
│                     TACTICAL LAYER                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Lightweight model / scripted logic (~10-100ms)      │   │
│  │  • Execute strategic goals step-by-step             │   │
│  │  • Routine operations (farming, navigation)         │   │
│  │  • Manage reaction rules:                           │   │
│  │    - Install new rules                              │   │
│  │    - Remove obsolete rules                          │   │
│  │    - Adjust rule priorities/parameters              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▼                                 │
├─────────────────────────────────────────────────────────────┤
│                    REACTION LAYER                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Pure math / hardcoded logic (<1ms)                  │   │
│  │  • Pattern matching on screen regions               │   │
│  │  • Color/pixel triggers                             │   │
│  │  • Immediate responses: dodge, block, combo         │   │
│  │  • No LLM, no network, no delay                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▼                                 │
│                      [OPERATOR]                             │
└─────────────────────────────────────────────────────────────┘
```

## Layer Details

### Reaction Layer (反應層)
**Speed**: <1ms  
**Tech**: NumPy, OpenCV, simple state machines

Responsibilities:
- Monitor specific screen regions for triggers
- Execute pre-programmed responses instantly
- Examples:
  - Health bar < 30% → use potion
  - Enemy attack animation detected → dodge
  - Cooldown ready → fire skill

Rules are defined by the Tactical Layer and stored as simple data structures:
```python
ReactionRule(
    id="dodge_on_red_flash",
    trigger=PixelTrigger(region=(100,100,50,50), color=(255,0,0), threshold=0.8),
    action={"type": "key", "key": "space"},
    priority=100,
    cooldown_ms=500
)
```

### Tactical Layer (戰術層)
**Speed**: 10-100ms  
**Tech**: Small local model, decision trees, behavior trees, or scripted FSM

Responsibilities:
- Break down strategic goals into executable steps
- Handle routine gameplay (move to X, attack target, collect loot)
- Dynamically manage reaction rules:
  - "Entering boss fight" → install dodge rules
  - "Boss defeated" → remove dodge rules, install loot rules
- React to tactical situations the reaction layer can't handle

### Strategic Layer (戰略層)
**Speed**: 1-10s (async, non-blocking)  
**Tech**: Full LLM (GPT-4V, Claude)

Responsibilities:
- Observe overall game state periodically
- Set high-level goals: "Clear this dungeon", "Level up to 50", "Complete quest X"
- Adjust tactical priorities based on situation
- Handle complex decisions requiring reasoning

Runs in background, doesn't block gameplay.

## Data Flow

```
[Screen] ──┬──▶ [Reaction Layer] ──immediate──▶ [Operator]
           │           ▲
           │           │ rules
           │           │
           ├──▶ [Tactical Layer] ──routine──▶ [Operator]
           │           ▲
           │           │ goals
           │           │
           └──▶ [Strategic Layer] (async)
```
