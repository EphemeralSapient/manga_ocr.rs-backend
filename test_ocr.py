#!/usr/bin/env python3
"""Quick OCR test - compare with Rust implementation"""

import sys
import os
sys.path.insert(0, 'custom_ocr_setup_ref')

import cv2
import numpy as np
from pathlib import Path

# Use sample image from reference
TEST_IMAGE = "custom_ocr_setup_ref/sample.png"
if not os.path.exists(TEST_IMAGE):
    print(f"Sample image not found: {TEST_IMAGE}")
    sys.exit(1)

# Load image
image = cv2.imread(TEST_IMAGE)
if image is None:
    print(f"Failed to load {TEST_IMAGE}")
    sys.exit(1)

print(f"Image shape: {image.shape}")

# Try oneocr_v2
try:
    from oneocr_v2 import OneOCR, ProcessOptions

    models_dir = "custom_ocr_setup_ref/models"
    if not os.path.exists(models_dir):
        print(f"Models not found at {models_dir}")
        sys.exit(1)

    print("\n=== OneOCR v2 (Python) ===")
    ocr = OneOCR(models_dir, recognition_model=0, use_gpu=False)  # CJK model

    # Test with vertical text detection
    result = ocr.run(image, detect_lines=True, vertical=True)

    print(f"Found {len(result.lines)} lines:")
    for i, line in enumerate(result.lines):
        print(f"  {i+1}. [{line.confidence:.2f}] {line.content}")

except Exception as e:
    print(f"OneOCR error: {e}")
    import traceback
    traceback.print_exc()

# Also test raw model inference
print("\n=== Raw ONNX inference ===")
try:
    import onnxruntime as ort

    # Load model
    model_path = "models/ocr/ocr.onnx"
    if not os.path.exists(model_path):
        model_path = "custom_ocr_setup_ref/models/oneocr_model_00.onnx"

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    print(f"Model: {model_path}")
    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    print("Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")

    # Preprocess - resize to 60px height
    h, w = image.shape[:2]
    target_height = 60
    scale = target_height / h
    new_w = int(w * scale)

    resized = cv2.resize(image, (new_w, target_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize and transpose
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
    tensor = tensor[np.newaxis, :, :, :]  # Add batch dim

    print(f"Input tensor shape: {tensor.shape}")

    seq_len = new_w // 4
    seq_lengths = np.array([seq_len], dtype=np.int32)

    # Run inference
    outputs = session.run(None, {
        'data': tensor,
        'seq_lengths': seq_lengths
    })

    logits = outputs[0]
    print(f"Output shape: {logits.shape}")

    # Load vocab
    vocab_path = "models/ocr/cjk_vocab.txt"
    if not os.path.exists(vocab_path):
        vocab_path = "custom_ocr_setup_ref/vocabs/cjk_vocab.txt"

    vocab = {}
    blank_idx = 0
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                idx, char = line.split('\t', 1)
                idx = int(idx)
                if char == '<blank>':
                    blank_idx = idx
                elif char == '<space>':
                    vocab[idx] = ' '
                else:
                    vocab[idx] = char

    print(f"Vocab size: {len(vocab)}, blank_idx: {blank_idx}")

    # CTC decode
    if len(logits.shape) == 3:
        if logits.shape[1] == 1:
            logits = logits[:, 0, :]  # [S, V]
        else:
            logits = logits[0, :, :]  # [S, V]

    decoded = []
    prev_idx = -1
    for t in range(logits.shape[0]):
        best_idx = np.argmax(logits[t])
        if best_idx != blank_idx and best_idx != prev_idx:
            decoded.append(best_idx)
        prev_idx = best_idx

    text = ''.join(vocab.get(idx, '?') for idx in decoded)
    print(f"Decoded indices: {decoded[:10]}...")
    print(f"Result: '{text}'")

except Exception as e:
    print(f"Raw inference error: {e}")
    import traceback
    traceback.print_exc()
