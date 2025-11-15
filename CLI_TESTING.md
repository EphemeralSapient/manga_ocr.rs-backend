# CLI Testing Guide

Complete guide for testing the manga translation workflow using the 3-phase CLI.

## Table of Contents

- [Quick Start](#quick-start)
- [Phase 1: Detection](#phase-1-detection)
- [Phase 2: Translation](#phase-2-translation)
- [Phase 3: Rendering](#phase-3-rendering)
- [Full Workflow Example](#full-workflow-example)
- [Advanced Options](#advanced-options)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Build the project
cargo build

# 2. Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEYS

# 3. Run full pipeline on input_image.webp
cargo run -- phase1 --input input_image.webp --output test_output/
cargo run -- phase2 --input test_output/bubble_01.png --detections-json test_output/detections.json --output test_output/phase2.json
cargo run -- phase3 --input test_output/bubble_01.png --api-response test_output/phase2.json --detections-json test_output/detections.json --output test_output/final.png
```

## Phase 1: Detection

**Purpose:** Detect speech bubbles and text regions in manga pages using ONNX model.

### Basic Usage

```bash
cargo run -- phase1 --input <IMAGE> --output <OUTPUT_DIR>
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input manga page image | Required |
| `--output` | Output directory for results | Required |
| `--visualize` | Save visualization with bboxes | `true` |
| `--include-text-free` | Also detect text outside bubbles | `false` |

### Output Files

- **`bubble_01.png`, `bubble_02.png`, ...** - Cropped bubble images (Label 0)
- **`detections.json`** - Detection metadata with Label 1 text regions
- **`visualization.png`** - Full page with bounding boxes

## Phase 2: Translation

**Purpose:** Translate text using Gemini API with simplified schema.

### Basic Usage

```bash
cargo run -- phase2 \
  --input <BUBBLE_IMAGE> \
  --detections-json <DETECTIONS_JSON> \
  --output <OUTPUT_JSON> \
  --font-family arial
```

### Available Fonts

- `arial` (default) - Clean, professional
- `anime-ace` - Comic/manga style  
- `anime-ace-3` - Comic v3
- `comic-sans` - Casual

### Example

```bash
cargo run -- phase2 \
  --input test_output/bubble_01.png \
  --detections-json test_output/detections.json \
  --output test_output/phase2.json \
  --font-family arial
```

## Phase 3: Rendering

**Purpose:** Remove text using edge-based erosion + render translation with cosmic-text.

### Basic Usage

```bash
cargo run -- phase3 \
  --input <BUBBLE_IMAGE> \
  --api-response <PHASE2_JSON> \
  --detections-json <DETECTIONS_JSON> \
  --output <OUTPUT_IMAGE> \
  --debug-polygon
```

### Debug Polygon Output

- 🟩 **GREEN**: Label 1 text region
- 🔴 **RED**: Polygon outline (if present)
- 🔵 **BLUE**: Even vertices
- 🟡 **YELLOW**: Odd vertices

## Full Workflow Script

```bash
#!/bin/bash
OUTPUT_DIR="output_t1"
rm -rf $OUTPUT_DIR && mkdir -p $OUTPUT_DIR

# Phase 1
cargo run -- phase1 --input input_image.webp --output $OUTPUT_DIR/

# Phase 2  
cargo run -- phase2 \
  --input $OUTPUT_DIR/bubble_01.png \
  --detections-json $OUTPUT_DIR/detections.json \
  --output $OUTPUT_DIR/phase2.json

# Phase 3
cargo run -- phase3 \
  --input $OUTPUT_DIR/bubble_01.png \
  --api-response $OUTPUT_DIR/phase2.json \
  --detections-json $OUTPUT_DIR/detections.json \
  --output $OUTPUT_DIR/final.png

echo "Complete! Check $OUTPUT_DIR/final.png"
```

## Environment Variables

```bash
# Required
GEMINI_API_KEYS=key1,key2,key3

# Optional
CONFIDENCE_THRESHOLD=0.3
TRANSLATION_MODEL=gemini-flash-latest  
UPSCALE_FACTOR=3
LOG_LEVEL=INFO
```

## Troubleshooting

**No bubbles detected:** Lower `CONFIDENCE_THRESHOLD` in .env

**Translation fails:** Check `GEMINI_API_KEYS` and internet connection

**Poor rendering:** Try different `--font-family` or check `--debug-polygon`

**Text removal incomplete:** Expected - system uses edge-based erosion constrained to Label 1

See [CLAUDE.md](CLAUDE.md) for architecture details.
