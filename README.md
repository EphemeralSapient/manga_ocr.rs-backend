# Manga Translation Workflow

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-2.0-green.svg)](https://onnxruntime.ai/)

Production-ready Rust application for automated manga/comic translation using AI-powered text detection, OCR, translation, and intelligent rendering.

## Overview

Three-stage pipeline for manga translation:
1. **Detection** - ONNX neural network identifies text bubbles and regions
2. **Translation** - Google Gemini API performs OCR and translation with background complexity analysis
3. **Rendering** - Edge-based text removal + high-quality typography rendering via cosmic-text

## Features

**Core**
- ONNX-based text region detection (speech bubbles, interior text, exterior text)
- Gemini 2.5 Flash translation with optimized line breaks
- Edge-based text removal for simple backgrounds, AI inpainting for complex backgrounds
- SHA1-based translation cache (30-40% API cost reduction)
- Hardware acceleration support (CUDA, TensorRT, DirectML, CoreML, OpenVINO)

**Production**
- RESTful API with synchronous (v1) and asynchronous job queue (v2) endpoints
- CLI tool with three-phase testing workflow
- API key pool with health tracking and automatic failover
- Circuit breaker pattern and configurable retry logic
- Structured logging with tracing

## Architecture

| Phase | Technology | Purpose |
|-------|-----------|---------|
| **Detection** | ONNX Runtime | Detect text bubbles and regions |
| **Translation** | Gemini 2.5 Flash | OCR, translation, background analysis |
| **Rendering** | cosmic-text + OpenCV | Text removal and rendering |

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev clang libclang-dev pkg-config

# macOS
brew install opencv pkg-config
```

### Build

```bash
# Copy environment template
cp .env.example .env

# Add Gemini API keys to .env
# GEMINI_API_KEYS=key1,key2,key3

# Build (CPU-only)
cargo build --release

# Build with GPU acceleration
cargo build --release --features cuda
```

### Usage

```bash
# Start API server (port 1420)
cargo run --release

# CLI three-phase workflow
cargo run --release -- --image input.png --output-dir ./output
```

## Configuration

Key environment variables (see `.env.example` for complete list):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEYS` | **(required)** | Comma-separated API keys |
| `SERVER_PORT` | `1420` | HTTP server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `INFERENCE_BACKEND` | `AUTO` | ONNX backend: `AUTO`, `CPU`, `CUDA`, `TENSORRT`, etc. |
| `CONFIDENCE_THRESHOLD` | `0.3` | Detection confidence (0.0-1.0) |
| `TRANSLATION_MODEL` | `gemini-2.5-flash` | Gemini translation model |
| `CACHE_DIR` | `.cache` | Cache directory |
| `BATCH_SIZE` | `15` | Max images per batch |

## API Reference

### Endpoints

**Health & Status**
- `GET /health` - System health and statistics
- `GET /cache-stats` - Cache performance metrics

**Translation (v1 - Synchronous)**
- `POST /v1/translate-batch` - Process images synchronously

**Translation (v2 - Asynchronous)**
- `POST /v2/jobs` - Submit translation job
- `GET /v2/jobs/:id` - Get job status
- `GET /v2/jobs/:id/result` - Retrieve result
- `DELETE /v2/jobs/:id` - Cancel job

### Request Format

`multipart/form-data` with:
- `images`: PNG/JPEG files (max 50MB each, max 10000×10000px)
- `config` (optional): JSON configuration

```json
{
  "translation_model": "gemini-2.5-flash",
  "font_family": "arial"
}
```

### Response Format

```json
{
  "total": 3,
  "successful": 3,
  "processing_time_ms": 8234.5,
  "analytics": {
    "total_bubbles_detected": 12,
    "cache_hit_rate": 41.7
  },
  "results": [...]
}
```

## Font Support

Available font families:

| Font | Style |
|------|-------|
| Arial Unicode | Clean, modern (default) |
| Anime Ace | Comic-style |
| Comic Shanns | Hand-drawn |
| Noto Sans Mono CJK | Monospace |
| Microsoft YaHei | CJK support |

**Credits**: Fonts sourced from [zyddnys](https://github.com/zyddnys)

## Detection Model

**Model**: `models/detector.onnx` (168MB INT8 quantized)

**Source**: [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)

**Detection Labels**:
- Label 0: Speech bubble (full)
- Label 1: Text region (interior)
- Label 2: Text outside bubbles

**Performance** (640px target size):
- CPU: ~500ms/page
- CUDA: ~150ms/page
- TensorRT: ~80ms/page

## CLI Usage

**Three-Phase Workflow**:
```bash
cargo run --release -- --image input.png --output-dir ./output
```

Phases:
1. Detection - Extracts bubbles to `output/bubbles/`
2. Translation - Generates JSON metadata
3. Rendering - Creates final translated image

**Interactive Mode**:
```bash
cargo run --release -- --interactive
```

## Performance

| Backend | Pages/min | Cost/page* |
|---------|-----------|------------|
| CPU | ~8 | $0.002 |
| CUDA | ~24 | $0.002 |
| TensorRT | ~40 | $0.002 |
| +40% cache | ~32 | $0.0012 |

\* Gemini 2.5 Flash pricing

## Project Structure

```
src/
├── main.rs              # Entry point
├── detection.rs         # ONNX detector
├── translation.rs       # Gemini API client
├── rendering.rs         # Text removal + rendering
├── cosmic_renderer.rs   # Text shaping
├── pipeline/            # Orchestration
├── api/                 # HTTP handlers
├── middleware/          # Key pool, circuit breaker
└── job_queue/           # Async processing

models/detector.onnx     # Detection model (168MB)
fonts/                   # TrueType fonts
```

## Development

```bash
# Run tests
cargo test

# Build for production
cargo build --release

# Build with GPU support
cargo build --release --features cuda
```

## Troubleshooting

**OpenCV not found**
```bash
sudo apt-get install libopencv-dev  # Linux
brew install opencv                  # macOS
```

**API quota exceeded** - Add multiple keys to `.env`:
```bash
GEMINI_API_KEYS=key1,key2,key3,key4
```

**Detection sensitivity** - Adjust threshold in `.env`:
```bash
CONFIDENCE_THRESHOLD=0.5  # Higher = fewer detections
CONFIDENCE_THRESHOLD=0.2  # Lower = more detections
```

## Credits

**Detection Model**: [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) - ONNX speech bubble and text region detector

**Fonts**: [zyddnys](https://github.com/zyddnys) - Manga and comic-optimized font collection

**Technologies**: ONNX Runtime, cosmic-text, Axum, OpenCV, Google Gemini API

## License

MIT License - see [LICENSE](LICENSE) file for details
