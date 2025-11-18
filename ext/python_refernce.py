#!/usr/bin/env python3
"""
Complete in-memory Korean webtoon translation pipeline.
No intermediate files saved - everything in RAM until final output.

Pipeline: Load Image → Detect Bubbles → Extract & Translate (in-memory) → Render → Save Final
"""
import sys
import os
import io
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

# Add comic-translate to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'comic-translate'))

from modules.detection.rtdetr_v2_onnx import RTDetrV2ONNXDetection


# ============================================================================
# GEMINI TRANSLATION SETUP
# ============================================================================

class ComicTranslation(BaseModel):
    """Structured output for Korean comic text translation."""
    bubble_number: int = Field(description="The bubble number (1-indexed)")
    korean_text: str = Field(description="The exact Korean text extracted from the image")
    english_translation: str = Field(description="Natural English translation suitable for comic dialogue")
    tone: str = Field(description="The tone/emotion of the text (e.g., casual, formal, angry, excited)")
    context_notes: Optional[str] = Field(default=None, description="Any cultural or contextual notes for translators")


class BatchComicTranslation(BaseModel):
    """Batch translation response for multiple speech bubbles."""
    translations: List[ComicTranslation] = Field(description="List of translations for each speech bubble, in order")


SYSTEM_INSTRUCTION = """You are an expert Korean to English translator specializing in webtoon and comic dialogue translation.

Your responsibilities:
1. You will receive MULTIPLE speech bubble images numbered sequentially
2. For EACH bubble, extract ALL Korean text EXACTLY as written (including spacing, line breaks)
3. Translate each to natural, conversational English that flows well in comic speech bubbles
4. Preserve the speaker's tone, emotion, and personality for each bubble
5. Adapt Korean speech patterns to natural English equivalents
6. Keep translations concise and suitable for limited bubble space

Translation guidelines:
- Use natural English expressions, NOT literal word-for-word translations
- Adapt Korean idioms and cultural references to English equivalents when appropriate
- Preserve emotional emphasis (!, ?, ..., —)
- Convert Korean honorifics to English tone/formality (e.g., 반말 → casual, 존댓말 → polite)
- Maintain character consistency (informal speakers stay informal, formal stay formal)
- For sound effects (의성어/의태어), use English comic conventions (e.g., "쿵" → "THUD", "휴" → "sigh")
- Keep dialogue punchy and readable - comics need brevity

Tone identification:
- Identify the emotional tone: casual, formal, angry, excited, sad, sarcastic, etc.
- This helps maintain character voice consistency across translations

Context notes:
- Only provide notes if there are important cultural references, wordplay, or untranslatable nuances
- Keep notes brief and actionable for editors

Output format:
- Return translations in the SAME ORDER as the input bubbles
- Use bubble_number to match each translation to its input (1 for first bubble, 2 for second, etc.)"""


# ============================================================================
# TEXT RENDERING FUNCTIONS
# ============================================================================

def find_optimal_font_size(
    text: str,
    max_width: int,
    max_height: int,
    font_path: str,
    min_size: int = 12,
    max_size: int = 72
) -> Tuple[int, List[str]]:
    """Find optimal font size and word wrapping for text to fit in region."""
    for size in range(max_size, min_size - 1, -2):
        try:
            font = ImageFont.truetype(font_path, size)
        except:
            font = ImageFont.load_default()

        lines = wrap_text(text, max_width, font)
        line_height = get_text_height("Ay", font)
        total_height = len(lines) * line_height * 1.2

        if total_height <= max_height:
            return size, lines

    font = ImageFont.truetype(font_path, min_size) if os.path.exists(font_path) else ImageFont.load_default()
    lines = wrap_text(text, max_width, font)
    return min_size, lines


def wrap_text(text: str, max_width: int, font: ImageFont.FreeTypeFont) -> List[str]:
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)

    if current_line:
        lines.append(' '.join(current_line))

    return lines if lines else [text]


def get_text_height(text: str, font: ImageFont.FreeTypeFont) -> int:
    """Get height of text with given font."""
    bbox = font.getbbox(text)
    return bbox[3] - bbox[1]


def get_text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    """Get width of text with given font."""
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def clear_text_region(image: np.ndarray, text_bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Clear text region by filling with white."""
    img = image.copy()
    tx1, ty1, tx2, ty2 = text_bbox
    img[ty1:ty2, tx1:tx2] = 255  # White fill
    return img


def render_text_in_region(
    image: np.ndarray,
    text_bbox: Tuple[int, int, int, int],
    text: str,
    font_path: str,
    padding: int = 10
) -> np.ndarray:
    """Render translated text in text region."""
    tx1, ty1, tx2, ty2 = text_bbox
    text_width = tx2 - tx1
    text_height = ty2 - ty1
    available_width = text_width - (padding * 2)
    available_height = text_height - (padding * 2)

    # Find optimal font size and wrap text
    font_size, wrapped_lines = find_optimal_font_size(
        text, available_width, available_height, font_path
    )

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Convert to PIL for drawing
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    # Calculate total text height
    line_height = get_text_height("Ay", font)
    line_spacing = int(line_height * 0.2)
    total_height = len(wrapped_lines) * line_height + (len(wrapped_lines) - 1) * line_spacing

    # Starting Y position (vertically centered)
    start_y = ty1 + (text_height - total_height) // 2

    # Draw each line centered
    current_y = start_y
    for line in wrapped_lines:
        line_width = get_text_width(line, font)
        x = tx1 + (text_width - line_width) // 2
        draw.text((x, current_y), line, font=font, fill=(0, 0, 0))
        current_y += line_height + line_spacing

    return np.array(pil_img)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def translate_webtoon_pipeline(
    input_image: str,
    output_image: str,
    api_key: str,
    model: str = "gemini-flash-lite-latest",
    font_path: Optional[str] = None,
    padding: int = 10,
    confidence: float = 0.3
):
    """
    Complete in-memory translation pipeline.

    Args:
        input_image: Input image path
        output_image: Output image path
        api_key: Gemini API key
        model: Gemini model to use
        font_path: Path to TTF font (optional)
        padding: Padding from text region edges
        confidence: Detection confidence threshold
    """
    print("="*70)
    print("KOREAN WEBTOON TRANSLATION PIPELINE")
    print("="*70)
    print()

    # Step 1: Load image into memory
    print("[Step 1/4] Loading image into memory...")
    pil_image = Image.open(input_image).convert('RGB')
    image_array = np.array(pil_image)
    print(f"  Image size: {image_array.shape[1]}×{image_array.shape[0]} pixels")
    print(f"  Memory: ~{image_array.nbytes / 1024 / 1024:.1f} MB")
    print()

    # Step 2: Detect speech bubbles (in memory)
    print("[Step 2/4] Detecting speech bubbles...")
    detector = RTDetrV2ONNXDetection()
    detector.initialize(device='cpu', confidence_threshold=confidence)
    text_blocks = detector.detect(image_array)

    bubble_count = sum(1 for b in text_blocks if b.bubble_xyxy is not None)
    print(f"  Detected {len(text_blocks)} text regions")
    print(f"  Found {bubble_count} speech bubbles")
    print()

    # Step 3: Translate ALL bubbles in single API call (batch)
    print("[Step 3/4] Translating with Gemini AI (batch mode)...")
    print(f"  Model: {model}")
    print(f"  Sending {bubble_count} bubbles in single API call...")

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Prepare all bubble images and metadata
    bubble_blocks = [b for b in text_blocks if b.bubble_xyxy is not None]

    if not bubble_blocks:
        print("  ⚠ No bubbles to translate")
        return

    # Build contents list with all bubble images
    contents = []

    for i, block in enumerate(bubble_blocks, 1):
        # Crop text region in memory
        tx1, ty1, tx2, ty2 = map(int, block.xyxy)
        cropped_img = image_array[ty1:ty2, tx1:tx2]

        # Convert to bytes (in memory)
        pil_crop = Image.fromarray(cropped_img)
        img_buffer = io.BytesIO()
        pil_crop.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()

        # Add to contents
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type='image/png'))
        print(f"  [{i}/{bubble_count}] Added bubble at ({tx1},{ty1}) to batch")

    # Add instruction as final part
    contents.append(
        f'Extract and translate the Korean text from all {len(bubble_blocks)} comic speech bubble images above. '
        f'Provide translations in the same order as the images (bubble 1, 2, 3, etc.).'
    )

    translations = []
    try:
        print(f"\n  Sending batch request to Gemini...")

        # Single API call for all bubbles
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=BatchComicTranslation,
                temperature=0.3,
            )
        )

        # Parse batch response
        batch_result = BatchComicTranslation.model_validate_json(response.text)

        print(f"  ✓ Received {len(batch_result.translations)} translations\n")

        # Match translations to blocks
        for i, (block, translation) in enumerate(zip(bubble_blocks, batch_result.translations)):
            translations.append({
                'bubble_bbox': tuple(map(int, block.bubble_xyxy)),
                'text_bbox': tuple(map(int, block.xyxy)),
                'translation': translation
            })

            print(f"  [Bubble {translation.bubble_number}]")
            print(f"    Korean: {translation.korean_text[:50]}...")
            print(f"    English: {translation.english_translation}")

    except Exception as e:
        print(f"  ❌ Batch translation error: {e}")
        print(f"  Error details: {str(e)}")
        return

    print(f"\n  ✓ Successfully translated {len(translations)}/{bubble_count} bubbles in single API call")
    print()

    # Step 4: Render translations back onto image (in memory)
    print("[Step 4/4] Rendering English text...")

    # Find best font
    if font_path is None:
        font_candidates = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ]
        for candidate in font_candidates:
            if os.path.exists(candidate):
                font_path = candidate
                break

    if font_path and os.path.exists(font_path):
        print(f"  Font: {os.path.basename(font_path)}")
    else:
        print(f"  Font: System default")
        font_path = None

    # Process each translation (in memory)
    for i, trans in enumerate(translations, 1):
        text_bbox = trans['text_bbox']
        english = trans['translation'].english_translation

        print(f"  [{i}/{len(translations)}] Rendering: {english[:40]}...")

        # Clear original text
        image_array = clear_text_region(image_array, text_bbox)

        # Render English text
        if font_path:
            image_array = render_text_in_region(
                image_array, text_bbox, english, font_path, padding
            )

    print(f"  ✓ Rendered {len(translations)} translations")
    print()

    # Step 5: Save final result
    print("[Step 5/4] Saving final image...")
    result_img = Image.fromarray(image_array)
    result_img.save(output_image)
    file_size = os.path.getsize(output_image) / 1024
    print(f"  Output: {output_image}")
    print(f"  Size: {file_size:.1f} KB")
    print()

    # Summary
    print("="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print(f"Translated {len(translations)} speech bubbles")
    print(f"No intermediate files created - all processing in RAM")
    print(f"Final output: {output_image}")
    print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate Korean webtoon - complete in-memory pipeline"
    )
    parser.add_argument(
        "--input",
        default="image.png",
        help="Input image (default: image.png)"
    )
    parser.add_argument(
        "--output",
        default="image_translated.png",
        help="Output image (default: image_translated.png)"
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        default="gemini-flash-lite-latest",
        help="Gemini model (default: gemini-flash-lite-latest)"
    )
    parser.add_argument(
        "--font",
        help="Path to TTF font file"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding from text edges (default: 10)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)"
    )

    args = parser.parse_args()

    # Check API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: No API key provided")
        print("   Set GEMINI_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Check input exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input image not found: {args.input}")
        sys.exit(1)

    # Run pipeline
    translate_webtoon_pipeline(
        input_image=args.input,
        output_image=args.output,
        api_key=api_key,
        model=args.model,
        font_path=args.font,
        padding=args.padding,
        confidence=args.confidence
    )


if __name__ == "__main__":
    main()