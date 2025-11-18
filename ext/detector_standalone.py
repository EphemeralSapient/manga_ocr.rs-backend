#!/usr/bin/env python3
"""
Standalone RT-DETR v2 ONNX detector - no comic-translate dependency needed.
"""
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TextBlock:
    """Represents a detected text block with optional bubble."""
    xyxy: np.ndarray  # Text region [x1, y1, x2, y2]
    bubble_xyxy: Optional[np.ndarray]  # Bubble region [x1, y1, x2, y2]
    text_class: str  # 'text_bubble' or 'text_free'
    score: float  # Confidence score


class RTDetrV2ONNXDetection:
    """RT-DETR v2 ONNX detector for speech bubbles and text regions."""

    def __init__(self):
        self.session = None
        self.input_size = 640
        self.confidence_threshold = 0.3

    def initialize(self, device='cpu', confidence_threshold=0.3):
        """
        Initialize the ONNX model.

        Args:
            device: 'cpu' or 'cuda' (only cpu supported in this standalone version)
            confidence_threshold: Detection confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(
            'detector.onnx',
            sess_options=sess_options,
            providers=providers
        )

        print(f"RT-DETR model initialized (confidence={confidence_threshold})")

    def preprocess(self, image: np.ndarray):
        """
        Preprocess image for RT-DETR.

        Args:
            image: RGB image as numpy array [H, W, 3]

        Returns:
            Tuple of (preprocessed_tensor, original_size)
        """
        original_h, original_w = image.shape[:2]

        # Resize to 640x640
        from PIL import Image
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((self.input_size, self.input_size), Image.BILINEAR)

        # Convert to numpy and normalize
        img_array = np.array(pil_img).astype(np.float32) / 255.0

        # HWC to CHW
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, (original_h, original_w)

    def postprocess(self, outputs, original_size):
        """
        Postprocess RT-DETR outputs to get detections.

        Args:
            outputs: Model outputs dictionary
            original_size: (height, width) of original image

        Returns:
            List of TextBlock objects
        """
        labels = outputs['labels']  # [1, 300]
        boxes = outputs['boxes']    # [1, 300, 4]
        scores = outputs['scores']  # [1, 300]

        # Remove batch dimension
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]

        # Filter by confidence
        mask = scores >= self.confidence_threshold
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]

        # Boxes are already in original image space (thanks to orig_target_sizes)
        # Just clamp to image boundaries
        original_h, original_w = original_size
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h)

        # Group detections: label 0=bubble, 1=text_bubble, 2=text_free
        bubbles = []
        texts = []

        for i in range(len(labels)):
            label = int(labels[i])
            box = boxes[i]
            score = float(scores[i])

            if label == 0:  # bubble
                bubbles.append((box, score))
            elif label == 1:  # text_bubble
                texts.append((box, score, 'text_bubble'))
            elif label == 2:  # text_free
                texts.append((box, score, 'text_free'))

        # Match text regions with bubbles using IoU
        text_blocks = []

        for text_box, text_score, text_class in texts:
            best_bubble = None
            best_iou = 0.0

            for bubble_box, _ in bubbles:
                iou = self.calculate_iou(text_box, bubble_box)
                if iou > best_iou:
                    best_iou = iou
                    best_bubble = bubble_box

            # Only include text_bubble class if it has a matching bubble
            if text_class == 'text_bubble' and best_bubble is not None:
                text_blocks.append(TextBlock(
                    xyxy=text_box,
                    bubble_xyxy=best_bubble,
                    text_class=text_class,
                    score=text_score
                ))
            elif text_class == 'text_free':
                text_blocks.append(TextBlock(
                    xyxy=text_box,
                    bubble_xyxy=None,
                    text_class=text_class,
                    score=text_score
                ))

        return text_blocks

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def detect(self, image: np.ndarray) -> List[TextBlock]:
        """
        Detect text blocks and bubbles in image.

        Args:
            image: RGB image as numpy array [H, W, 3]

        Returns:
            List of TextBlock objects
        """
        if self.session is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Preprocess
        img_tensor, original_size = self.preprocess(image)

        # Prepare orig_target_sizes input
        orig_target_sizes = np.array([[original_size[0], original_size[1]]], dtype=np.int64)

        # Run inference
        outputs = self.session.run(
            None,
            {
                'images': img_tensor,
                'orig_target_sizes': orig_target_sizes
            }
        )

        # Convert outputs to dictionary
        output_names = [output.name for output in self.session.get_outputs()]
        outputs_dict = {name: output for name, output in zip(output_names, outputs)}

        # Postprocess
        text_blocks = self.postprocess(outputs_dict, original_size)

        return text_blocks
