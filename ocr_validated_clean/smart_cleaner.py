"""
OCR-Validated Text Cleaner

Smart text removal that only cleans regions where OCR confirms actual text exists.
This prevents over-cleaning of art/characters that text detector misidentifies.

Flow:
1. Text detector (Model 01 FPN) finds potential text regions
2. OCR (Model 00 CJK) validates each region
3. Only clean regions where OCR returns valid text

Usage:
    python smart_cleaner.py input.png --output output.png
"""

import argparse
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TextRegion:
    """Detected text region with OCR validation"""
    x: int
    y: int
    width: int
    height: int
    mask: np.ndarray  # Binary mask for this region
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    has_valid_text: bool = False


class SmartTextCleaner:
    """Text cleaner with OCR validation to prevent over-cleaning"""

    def __init__(self,
                 detector_model: str = None,
                 ocr_model: str = None,
                 vocab_path: str = None):
        """
        Initialize with both detector and OCR models

        Args:
            detector_model: Path to FPN text detector (model 01)
            ocr_model: Path to CJK OCR model (model 00)
            vocab_path: Path to CJK vocabulary file
        """
        script_dir = Path(__file__).parent.parent

        # Default paths
        if detector_model is None:
            detector_model = script_dir / "clean_text_ref" / "oneocr_model_01_fixed.onnx"
        if ocr_model is None:
            ocr_model = script_dir / "custom_ocr_setup_ref" / "models" / "oneocr_model_00.onnx"
        if vocab_path is None:
            vocab_path = Path("/.ram/jap_ocr/vocabs/cjk_vocab.txt")
            if not vocab_path.exists():
                vocab_path = script_dir / "custom_ocr_setup_ref" / "vocabs" / "cjk_vocab.txt"

        print(f"Loading detector: {detector_model}")
        self.detector = ort.InferenceSession(str(detector_model), providers=['CPUExecutionProvider'])

        print(f"Loading OCR: {ocr_model}")
        self.ocr = ort.InferenceSession(str(ocr_model), providers=['CPUExecutionProvider'])

        # Load vocabulary
        self.vocab = self._load_vocab(vocab_path)
        self.blank_index = len(self.vocab)  # Usually last index
        print(f"Loaded vocab: {len(self.vocab)} entries, blank_index={self.blank_index}")

    def _load_vocab(self, vocab_path: str) -> dict:
        """Load CJK vocabulary"""
        vocab = {}
        # Default ASCII
        for i in range(97):
            if 32 <= i < 127:
                vocab[i] = chr(i)
            elif i == 0:
                vocab[i] = ' '

        if Path(vocab_path).exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '\t' in line:
                        idx, char = line.split('\t', 1)
                        idx = int(idx)
                        if char == '<blank>':
                            self.blank_index = idx
                            continue
                        if char == '<space>':
                            char = ' '
                        vocab[idx] = char
        return vocab

    def _preprocess_detector(self, image: np.ndarray, target_size: int = 640) -> Tuple:
        """Preprocess image for text detector"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        scale = target_size / max(orig_h, orig_w)
        new_h = ((int(orig_h * scale) + 63) // 64) * 64
        new_w = ((int(orig_w * scale) + 63) // 64) * 64
        new_h = max(new_h, 256)
        new_w = max(new_w, 256)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        data = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        im_info = np.array([[new_h, new_w, scale]], dtype=np.float32)
        featuremap_cond = np.array([False], dtype=np.bool_)

        return data, im_info, featuremap_cond, scale, (orig_h, orig_w), (new_h, new_w)

    def _preprocess_ocr(self, image: np.ndarray, target_height: int = 60) -> Tuple[np.ndarray, int]:
        """Preprocess image region for OCR"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        scale = target_height / h
        new_w = max(int(w * scale), 10)

        resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]

        seq_length = new_w // 4
        return tensor, seq_length

    def _ctc_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        """CTC greedy decode"""
        if len(logits.shape) == 3:
            if logits.shape[1] == 1:
                logits = logits[:, 0, :]
            else:
                logits = logits[0, :, :]

        seq_len, vocab_size = logits.shape
        decoded = []
        confidences = []
        prev_idx = -1

        for t in range(seq_len):
            best_idx = np.argmax(logits[t])
            best_val = logits[t, best_idx]

            if best_idx != self.blank_index and best_idx != prev_idx:
                decoded.append(best_idx)
                confidences.append(min(np.exp(best_val), 1.0))
            prev_idx = best_idx

        text = ''.join(self.vocab.get(i, '?') for i in decoded)
        avg_conf = np.mean(confidences) if confidences else 0.0

        return text, float(avg_conf)

    def _run_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Run OCR on image region, trying rotated if needed"""
        # Try rotated first (vertical Japanese text)
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        tensor_rot, seq_len_rot = self._preprocess_ocr(rotated)

        outputs_rot = self.ocr.run(None, {
            'data': tensor_rot,
            'seq_lengths': np.array([seq_len_rot], dtype=np.int32)
        })
        text_rot, conf_rot = self._ctc_decode(outputs_rot[0])

        # Also try original orientation
        tensor_orig, seq_len_orig = self._preprocess_ocr(image)
        outputs_orig = self.ocr.run(None, {
            'data': tensor_orig,
            'seq_lengths': np.array([seq_len_orig], dtype=np.int32)
        })
        text_orig, conf_orig = self._ctc_decode(outputs_orig[0])

        # Count CJK characters
        cjk_rot = sum(1 for c in text_rot if ord(c) > 0x3000)
        cjk_orig = sum(1 for c in text_orig if ord(c) > 0x3000)

        # Prefer result with more CJK content
        if cjk_rot > cjk_orig or (cjk_rot == cjk_orig and conf_rot >= conf_orig):
            return text_rot, conf_rot
        return text_orig, conf_orig

    def _is_valid_text(self, text: str, confidence: float) -> bool:
        """Check if OCR result indicates valid text"""
        if not text or len(text.strip()) == 0:
            return False

        # Count CJK characters
        cjk_count = sum(1 for c in text if ord(c) > 0x3000)

        # If has CJK, accept with low threshold
        if cjk_count > 0:
            return confidence >= 0.2

        # Pure ASCII single char is garbage
        if len(text) <= 2 and cjk_count == 0:
            return False

        return confidence >= 0.4

    def detect_text_regions(self, image: np.ndarray,
                            threshold: float = 0.3,
                            min_area: int = 100) -> List[TextRegion]:
        """Detect potential text regions using FPN detector"""
        data, im_info, featuremap_cond, scale, orig_size, new_size = self._preprocess_detector(image)

        outputs = self.detector.run(None, {
            'data': data,
            'im_info': im_info,
            'featuremap_cond': featuremap_cond
        })

        # Build output dict
        output_names = [out.name for out in self.detector.get_outputs()]
        result = {name: output for name, output in zip(output_names, outputs)}

        h, w = orig_size
        new_h, new_w = new_size

        # Create combined heatmap
        heatmap = np.zeros((new_h, new_w), dtype=np.float32)
        for level in ['fpn2', 'fpn3', 'fpn4']:
            scores_hori = result[f'scores_hori_{level}'][0, 0]
            scores_vert = result[f'scores_vert_{level}'][0, 0]
            hori_resized = cv2.resize(scores_hori, (new_w, new_h))
            vert_resized = cv2.resize(scores_vert, (new_w, new_h))
            heatmap = np.maximum(heatmap, hori_resized)
            heatmap = np.maximum(heatmap, vert_resized)

        # Resize to original size
        heatmap = cv2.resize(heatmap, (w, h))

        # Threshold and find contours
        binary = (heatmap > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            if area < min_area:
                continue

            # Create mask for this region
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Dilate slightly
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            regions.append(TextRegion(
                x=x, y=y, width=cw, height=ch, mask=mask
            ))

        return regions

    def validate_regions_with_ocr(self, image: np.ndarray,
                                   regions: List[TextRegion]) -> List[TextRegion]:
        """Run OCR on each region to validate if it contains real text"""
        for region in regions:
            # Extract region with padding
            pad = 5
            x1 = max(0, region.x - pad)
            y1 = max(0, region.y - pad)
            x2 = min(image.shape[1], region.x + region.width + pad)
            y2 = min(image.shape[0], region.y + region.height + pad)

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Run OCR
            text, confidence = self._run_ocr(crop)
            region.ocr_text = text
            region.ocr_confidence = confidence
            region.has_valid_text = self._is_valid_text(text, confidence)

        return regions

    def clean_image(self, image: np.ndarray,
                    threshold: float = 0.3,
                    fill_color: Tuple[int, int, int] = (255, 255, 255),
                    validate_with_ocr: bool = True) -> Tuple[np.ndarray, np.ndarray, List[TextRegion]]:
        """
        Clean text from image with optional OCR validation

        Args:
            image: Input BGR image
            threshold: Detection threshold
            fill_color: Color to fill text regions
            validate_with_ocr: If True, only clean regions where OCR finds valid text

        Returns:
            (cleaned_image, combined_mask, regions)
        """
        h, w = image.shape[:2]

        # Step 1: Detect potential text regions
        print(f"Detecting text regions (threshold={threshold})...")
        regions = self.detect_text_regions(image, threshold)
        print(f"Found {len(regions)} potential regions")

        # Step 2: Validate with OCR if enabled
        if validate_with_ocr:
            print("Validating regions with OCR...")
            regions = self.validate_regions_with_ocr(image, regions)
            valid_count = sum(1 for r in regions if r.has_valid_text)
            print(f"OCR validated: {valid_count}/{len(regions)} regions contain text")

        # Step 3: Create combined mask (only validated regions)
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            if not validate_with_ocr or region.has_valid_text:
                combined_mask = cv2.bitwise_or(combined_mask, region.mask)

        # Step 4: Apply mask to clean image
        result = image.copy()
        result[combined_mask > 0] = fill_color

        return result, combined_mask, regions


def main():
    parser = argparse.ArgumentParser(description='OCR-validated text cleaner')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='Detection threshold (default: 0.3)')
    parser.add_argument('--no-ocr-validation', action='store_true',
                        help='Disable OCR validation (clean all detected regions)')
    parser.add_argument('--save-mask', action='store_true',
                        help='Save the text mask')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print OCR results for each region')

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load {args.image}")
        return

    print(f"Loaded: {args.image} ({image.shape[1]}x{image.shape[0]})")

    # Initialize cleaner
    cleaner = SmartTextCleaner()

    # Clean image
    cleaned, mask, regions = cleaner.clean_image(
        image,
        threshold=args.threshold,
        validate_with_ocr=not args.no_ocr_validation
    )

    # Print region details if verbose
    if args.verbose:
        print("\nRegion details:")
        for i, r in enumerate(regions):
            status = "CLEAN" if r.has_valid_text else "SKIP"
            print(f"  {i+1}. [{status}] '{r.ocr_text}' (conf: {r.ocr_confidence:.2f})")

    # Determine output path
    input_path = Path(args.image)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_smart_cleaned{input_path.suffix}"

    # Save results
    cv2.imwrite(str(output_path), cleaned)
    print(f"\nSaved: {output_path}")

    if args.save_mask:
        mask_path = output_path.parent / f"{output_path.stem}_mask{output_path.suffix}"
        cv2.imwrite(str(mask_path), mask)
        print(f"Saved mask: {mask_path}")

    # Stats
    cleaned_count = sum(1 for r in regions if r.has_valid_text)
    skipped_count = len(regions) - cleaned_count
    coverage = (np.sum(mask > 0) / mask.size) * 100
    print(f"\nStats: {cleaned_count} cleaned, {skipped_count} skipped, {coverage:.1f}% coverage")


if __name__ == '__main__':
    main()
