# Production Manga Bubble Segmentation

**Optimized ONNX Pipeline - Best Performance**

## Performance
- **5758ms per bubble** (avg)
- **Fastest configuration** - ONNX defaults beat manual optimization
- CPU optimized for ARM64

## Contents

### Models (310MB total)
1. **RT-DETR** (164MB) - `comic-detector/`
   - Detects speech bubbles (label 0)
   - ONNX format: `detector.onnx`

2. **SAM2.1 Tiny** (149MB) - `sam2.1_tiny/`
   - Preprocess (encoder): `sam2.1_tiny_preprocess.onnx` (128MB)
   - Decoder: `sam2.1_tiny.onnx` (20MB)

### Script
- `onnx_pipeline.py` - Production inference script

### Test Image
- `japanese OCR old manga.webp` - Sample manga page

## Usage

```bash
python3 onnx_pipeline.py
```

### Output
- `onnx_output/bubble_N_white_fill.jpg` - Cleaned bubbles
- `onnx_output/bubble_N_mask.jpg` - Segmentation masks

## Requirements

```bash
pip install onnxruntime opencv-python numpy pillow transformers torch
```

## Architecture

```
Input Image
    ↓
RT-DETR Detection (ONNX)
    ↓
Padded Crop (25% padding)
    ↓
SAM2.1 Tiny Encoder (ONNX)
    ↓
SAM2.1 Tiny Decoder (ONNX)
    ↓
White Fill Background
    ↓
Output Images
```

## Why This Configuration?

### ONNX vs PyTorch
- ONNX: 5758ms/bubble
- PyTorch: 9633ms/bubble
- **1.67x faster with ONNX**

### No Session Optimizations
- Default ONNX Runtime settings: 5758ms
- With optimizations: 8423ms
- **Defaults are 46% faster!**

### Quantization Results
- FP32: baseline
- FP16: type errors (incompatible)
- INT8: 19% faster but needs retesting without session opts

## Notes

- Session options (graph optimization, threading) hurt performance on ARM
- ONNX Runtime defaults are already optimized
- Over-optimization is counterproductive
- Simple is faster
