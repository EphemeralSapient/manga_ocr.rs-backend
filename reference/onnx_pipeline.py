#!/usr/bin/env python3
"""
Full ONNX Pipeline - SAM2.1 Tiny
Using rectlabel ONNX models
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

def get_providers():
    """Get best ONNX providers"""
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        print("ONNX Runtime: CUDA")
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif 'DmlExecutionProvider' in available:
        print("ONNX Runtime: DirectML")
        return ['DmlExecutionProvider', 'CPUExecutionProvider']
    else:
        print("ONNX Runtime: CPU")
        return ['CPUExecutionProvider']

def detect_bubbles(image_path, padding=0.25):
    """RT-DETR detection"""
    print("=" * 60)
    print("RT-DETR Detection")
    print("=" * 60)

    processor = RTDetrImageProcessor.from_pretrained("../comic-detector")
    model = RTDetrForObjectDetection.from_pretrained("../comic-detector")
    model.eval()

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    start = time.time()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3
    )[0]
    det_time = (time.time() - start) * 1000

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    bubbles = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label.item() == 0:
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * padding), int(h * padding)
            x1_p = max(0, x1 - pad_w)
            y1_p = max(0, y1 - pad_h)
            x2_p = min(img_w, x2 + pad_w)
            y2_p = min(img_h, y2 + pad_h)
            crop = image_cv[y1_p:y2_p, x1_p:x2_p]
            bubbles.append({'index': len(bubbles), 'score': score.item(), 'crop': crop})

    print(f"Detected {len(bubbles)} bubbles in {det_time:.1f}ms")
    return bubbles

def segment_with_sam2_onnx(crop, preprocess_sess, decoder_sess):
    """SAM2.1 Tiny ONNX segmentation"""
    h, w = crop.shape[:2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Resize to 1024x1024
    img_1024 = cv2.resize(crop_rgb, (1024, 1024))
    img_array = img_1024.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    # Run preprocess (encoder)
    start = time.time()
    preprocess_outputs = preprocess_sess.run(None, {'input': img_array})
    image_embeddings = preprocess_outputs[0]
    high_res_features1 = preprocess_outputs[1]
    high_res_features2 = preprocess_outputs[2]
    enc_time = (time.time() - start) * 1000

    # Create prompts - multiple points on bubble background
    points = []
    for y_off in [-0.2, -0.1, 0, 0.1, 0.2]:
        for x_off in [-0.15, 0, 0.15]:
            px = 512 + int(512 * x_off)
            py = 512 + int(512 * y_off)
            points.append([px, py])

    point_coords = np.array([points], dtype=np.float32)
    point_labels = np.array([[1] * len(points)], dtype=np.float32)

    # Empty mask input
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.array([0], dtype=np.float32)
    orig_im_size = np.array([h, w], dtype=np.int64)

    # Run decoder
    start = time.time()
    decoder_inputs = {
        'image_embeddings': image_embeddings,
        'high_res_features1': high_res_features1,
        'high_res_features2': high_res_features2,
        'point_coords': point_coords,
        'point_labels': point_labels,
        'mask_input': mask_input,
        'has_mask_input': has_mask_input,
        'orig_im_size': orig_im_size
    }
    decoder_outputs = decoder_sess.run(None, decoder_inputs)
    masks = decoder_outputs[0]
    iou_predictions = decoder_outputs[1]
    dec_time = (time.time() - start) * 1000

    # Take best mask
    best_idx = np.argmax(iou_predictions[0])
    mask = masks[0][best_idx]

    # Convert to uint8
    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Fill with white
    result = crop.copy()
    result[mask_clean > 127] = [255, 255, 255]

    total_time = enc_time + dec_time
    score = iou_predictions[0][best_idx]

    return result, mask_clean, total_time, score

def main():
    input_image = "japanese OCR old manga.webp"
    preprocess_path = "sam2.1_tiny/sam2.1_tiny_preprocess.onnx"
    decoder_path = "sam2.1_tiny/sam2.1_tiny.onnx"
    output_dir = "onnx_output"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("FULL ONNX PIPELINE - SAM2.1 Tiny")
    print("=" * 60)
    print(f"Preprocess: {os.path.getsize(preprocess_path)/1024/1024:.1f}MB")
    print(f"Decoder: {os.path.getsize(decoder_path)/1024/1024:.1f}MB")
    print()

    # Detect bubbles
    bubbles = detect_bubbles(input_image, padding=0.25)

    if not bubbles:
        return

    # Load ONNX models
    print("\nLoading SAM2.1 Tiny ONNX...")
    providers = get_providers()
    preprocess_sess = ort.InferenceSession(preprocess_path, providers=providers)
    decoder_sess = ort.InferenceSession(decoder_path, providers=providers)

    print("\n" + "=" * 60)
    print("SAM2.1 Tiny ONNX Segmentation")
    print("=" * 60)

    total_time = 0
    for bubble in bubbles:
        i = bubble['index']
        result, mask, sam_time, score = segment_with_sam2_onnx(
            bubble['crop'], preprocess_sess, decoder_sess
        )
        total_time += sam_time

        cv2.imwrite(f"{output_dir}/bubble_{i}_white_fill.jpg", result)
        cv2.imwrite(f"{output_dir}/bubble_{i}_mask.jpg", mask)

        print(f"Bubble {i}: {sam_time:.1f}ms, score={score:.3f}")

    avg_time = total_time / len(bubbles)

    print("\n" + "=" * 60)
    print("RESULTS - FULL ONNX")
    print("=" * 60)
    print(f"Processed: {len(bubbles)} bubbles")
    print(f"Avg SAM2 ONNX time: {avg_time:.1f}ms per bubble")
    print(f"Total SAM2 time: {total_time:.1f}ms")
    print(f"\nOutput: {output_dir}/")

if __name__ == "__main__":
    main()
