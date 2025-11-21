# Manga Text Processor

High-performance Rust backend for manga text detection, translation, and rendering using ONNX models and Google Gemini API.

## Overview

This service implements a 4-phase pipeline for processing manga pages:

### Phase 1: Detection & Categorization
- Detects text regions using ONNX models (detector + segmentation)
- Generates segmentation masks for precise text removal
- Categorizes regions by background complexity (simple vs complex)

### Phase 2: Translation
- **Simple backgrounds**: OCR + text translation via batched API calls
- **Complex backgrounds**: Image-to-image translation (banana mode, optional)
- Persistent LRU cache to minimize API calls and costs

### Phase 3: Mask-based Text Removal
- Removes original text using segmentation masks
- Creates cleaned region images ready for new text
- Skips regions already processed by banana mode

### Phase 4: Text Insertion & Compositing
- Renders translated text onto cleaned regions using cosmic-text
- Supports multiple fonts (Anime Ace, Arial Unicode, Comic Sans, Noto Sans Mono CJK, Microsoft YaHei)
- Composites all regions back into final translated image

## Technical Stack

- **Server**: Axum HTTP server with Tokio async runtime
- **ML Inference**: ONNX Runtime with GPU acceleration support
  - CUDA (NVIDIA)
  - TensorRT (NVIDIA)
  - DirectML (Windows, supports iGPU)
  - CoreML (Apple Silicon)
  - OpenVINO (Intel, CPU only)
- **Image Processing**: OpenCV, imageproc
- **Text Rendering**: cosmic-text, ab_glyph
- **API**: Google Gemini 2.5-flash/pro
- **Concurrency**: Rayon for parallel processing

## Features

- Automatic text bubble detection and segmentation
- Batch processing with configurable concurrency limits
- Translation caching with persistent storage
- Circuit breaker pattern for API resilience
- RESTful API with health checks and metrics endpoints
- GPU acceleration support across platforms

## Installation

<details>
<summary><h3>Compile & Instructions</h3></summary>

<details>
<summary><b>Linux: Install Rust</b></summary>

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version
```

</details>

<details>
<summary><b>macOS: Install Rust</b></summary>

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version
```

</details>

<details>
<summary><b>Windows: Install Rust (MSVC required, not GNU)</b></summary>

```powershell
# Install Rust via winget
winget install --id=Rustlang.Rustup -e

# During installation, select MSVC toolchain (default)
# If prompted, install Visual Studio Build Tools

# Verify MSVC target
rustup default stable-msvc
rustc --version
```

**Note:** This project requires MSVC toolchain. GNU toolchain will not work.

</details>

**Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev clang pkg-config

# macOS
brew install opencv pkg-config llvm

# Windows - Download and install official OpenCV
# https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-windows.exe
# Then set environment variables:
# OPENCV_LINK_LIBS=opencv_world4120
# OPENCV_LINK_PATHS=C:\opencv\build\x64\vc16\lib
# OPENCV_INCLUDE_PATHS=C:\opencv\build\include
```

**Build:**
```bash
cargo build --release

# With GPU acceleration
cargo build --release --features cuda       # NVIDIA GPUs (Linux/Windows)
cargo build --release --features tensorrt   # NVIDIA TensorRT (Linux)
cargo build --release --features directml   # DirectML (Windows, supports iGPU)
cargo build --release --features coreml     # Apple Silicon (macOS)
cargo build --release --features openvino   # Intel OpenVINO (Linux, CPU only)
```

Note: OpenVINO provides CPU-only acceleration. For Intel integrated GPUs, use DirectML on Windows.

**Configuration:**
```bash
cp .env.example .env
cp .env.local.example .env.local
```

Add your Gemini API keys to `.env.local`:
```env
GEMINI_API_KEYS=key1,key2,key3
```

**Run:**
```bash
cargo run --release
# Server starts on http://localhost:1420
```

### Docker

**Quick start:**
```bash
# Setup
cp .env.local.example .env.local
# Add GEMINI_API_KEYS to .env.local

# Run
docker-compose up -d

# Check status
curl http://localhost:1420/health
```

**Manual build:**
```bash
DOCKER_BUILDKIT=1 docker build -t manga_workflow .

docker run -d \
  -p 1420:1420 \
  -v $(pwd)/.env.local:/app/.env.local:ro \
  -v manga_cache:/app/.cache \
  --memory=4g \
  manga_workflow
```

**Volume mounts:**

| Path | Container | Purpose |
|------|-----------|---------|
| `.env.local` | `/app/.env.local` | API keys (read-only) |
| `manga_cache` | `/app/.cache` | Translation cache |
| `fonts/` | `/app/fonts` | Custom fonts (optional) |

**GPU support:**
```bash
docker run -d --gpus all -p 1420:1420 \
  -v $(pwd)/.env.local:/app/.env.local:ro \
  manga_workflow
```

</details>

## Usage

### Python CLI Client

The included client provides both interactive and command-line modes:

```bash
pip install -r requirements.txt

# Interactive mode
python client.py

# Process single image
python client.py -i page.webp -o output/

# Process folder
python client.py -i manga_pages/ -o translated/

# Custom font
python client.py -i page.webp -f comic-sans
```

### Chrome Extension

For browser-based usage: https://github.com/EphemeralSapient/manga_ocr.rs-frontend

## Configuration

All configuration options are documented in `.env.example`. Copy it to `.env` and adjust settings as needed. API keys should be placed in `.env.local`.

## Performance & Monitoring

**Resource requirements:**
- Memory: 2-4GB minimum, 8GB recommended for large batches
- CPU: 4+ cores recommended
- GPU: Optional but significantly improves inference speed

**Monitoring endpoints:**
- `/health` - Health check
- `/metrics` - Prometheus metrics
- `/stats` - Detailed statistics

## Troubleshooting

**Slow processing:**
- Enable GPU acceleration via `INFERENCE_BACKEND` setting
- Increase `MAX_CONCURRENT_BATCHES` for higher throughput
- Reduce `BATCH_SIZE_N` if running out of memory

**API errors:**
- Check API key validity at https://aistudio.google.com/apikey
- Review rate limits in Gemini API console
- Enable `LOG_LEVEL=DEBUG` for detailed error messages

**Docker issues:**
- Ensure BuildKit is enabled: `export DOCKER_BUILDKIT=1`
- Check logs: `docker logs manga_workflow`
- Verify volume mounts: `docker inspect manga_workflow`

## Model Credits

- **Text/Bubble Detector**: [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)
- **Segmentation Mask**: [kitsumed/yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)

## Disclaimer

This software is provided for educational and personal use only. It is not intended for commercial use or distribution. Users are responsible for ensuring they have appropriate rights to any content they process. This project is not associated with or endorsed by any copyright holders. Use at your own risk and respect intellectual property rights.
