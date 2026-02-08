# Player MK1

General-purpose game player — watches, thinks, acts.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Vision    │────▶│   Brain     │────▶│  Operator   │
│  (Stream)   │     │  (Decide)   │     │  (Execute)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Modules

### 1. Vision (`vision/`)
Screen capture and preprocessing.
- Capture screen frames (full or region)
- Encode/compress for model input
- Optional: OCR, object detection hooks

### 2. Brain (`brain/`)
Decision engine — takes visual input, outputs actions.
- Multimodal LLM integration (GPT-4V, Claude, Gemini)
- Game-specific prompting / few-shot examples
- State tracking and memory

### 3. Operator (`operator/`)
Action execution layer.
- Mouse movement and clicks
- Keyboard input
- Gamepad/controller emulation (optional)
- Safety: action rate limiting, kill switch

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

## Config

See `config.yaml` for settings (capture region, model, action delays, etc.)
