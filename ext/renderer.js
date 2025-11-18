// Text Renderer - Canvas-based text rendering with word wrapping
class TextRenderer {
  constructor() {
    this.fontFamily = 'Arial, sans-serif';
    this.fontWeight = 'bold';
    this.fontSize = 24;
    this.textColor = '#000000';
    this.padding = 10;
    console.log('[RENDERER] TextRenderer instance created');
    console.log(`[RENDERER] Default font: ${this.fontWeight} ${this.fontFamily}`);
    console.log(`[RENDERER] Text color: ${this.textColor}`);
    console.log(`[RENDERER] Padding: ${this.padding}px`);
  }

  async renderTranslations(imageElement, detections, translations) {
    console.log('[RENDERER] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log(`[RENDERER] Starting rendering of ${translations.length} translations...`);

    const renderStartTime = performance.now();

    // Create canvas from original image
    console.log('[RENDERER] Creating canvas...');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const width = imageElement.naturalWidth || imageElement.width;
    const height = imageElement.naturalHeight || imageElement.height;
    canvas.width = width;
    canvas.height = height;

    console.log(`[RENDERER] Image natural size: ${imageElement.naturalWidth}x${imageElement.naturalHeight}`);
    console.log(`[RENDERER] Image display size: ${imageElement.width}x${imageElement.height}`);
    console.log(`[RENDERER] Canvas size: ${width}x${height}px`);

    // Draw original image
    console.log('[RENDERER] Drawing original image to canvas...');
    ctx.drawImage(imageElement, 0, 0);
    console.log('[RENDERER] ✓ Original image drawn');

    // Render each translation
    console.log(`[RENDERER] Rendering ${translations.length} text overlays...`);
    for (let i = 0; i < translations.length; i++) {
      const detection = detections[i];
      const translation = translations[i];

      console.log(`[RENDERER] ━━━ Bubble ${i + 1}/${translations.length} ━━━`);
      console.log(`[RENDERER] Korean: "${translation.korean_text}"`);
      console.log(`[RENDERER] English: "${translation.english_translation}"`);

      // Clear text region (white rectangle)
      const [x1, y1, x2, y2] = detection.textBox;
      const [bx1, by1, bx2, by2] = detection.bubbleBox;
      const textWidth = x2 - x1;
      const textHeight = y2 - y1;
      const bubbleWidth = bx2 - bx1;
      const bubbleHeight = by2 - by1;

      console.log(`[RENDERER] Text box: (${x1},${y1}) to (${x2},${y2}) [${textWidth}x${textHeight}px]`);
      console.log(`[RENDERER] Bubble box: (${bx1},${by1}) to (${bx2},${by2}) [${bubbleWidth}x${bubbleHeight}px]`);
      console.log(`[RENDERER] Using TEXT box for rendering`);

      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(x1, y1, textWidth, textHeight);
      console.log(`[RENDERER] ✓ Text region cleared with white (${textWidth}x${textHeight}px)`);

      // Render text
      console.log(`[RENDERER] Rendering text...`);
      this.renderText(ctx, translation.english_translation, detection.textBox);
      console.log(`[RENDERER] ✓ Bubble ${i + 1} complete`);
    }

    // Convert to data URL
    console.log('[RENDERER] Converting canvas to data URL...');
    const dataUrl = canvas.toDataURL('image/png');
    const dataUrlSize = dataUrl.length;
    console.log(`[RENDERER] ✓ Data URL created (${(dataUrlSize / 1024).toFixed(2)} KB)`);

    const renderEndTime = performance.now();
    const renderTime = renderEndTime - renderStartTime;

    console.log('[RENDERER] ━━━ Rendering Summary ━━━');
    console.log(`[RENDERER] Total bubbles rendered: ${translations.length}`);
    console.log(`[RENDERER] Total time: ${renderTime.toFixed(0)}ms (${(renderTime/1000).toFixed(2)}s)`);
    console.log(`[RENDERER] Average time per bubble: ${(renderTime / translations.length).toFixed(0)}ms`);
    console.log('[RENDERER] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

    return dataUrl;
  }

  renderText(ctx, text, box) {
    const [x1, y1, x2, y2] = box;
    const width = x2 - x1;
    const height = y2 - y1;

    const availableWidth = width - this.padding * 2;
    const availableHeight = height - this.padding * 2;

    console.log(`[RENDERER]   Available space: ${availableWidth}x${availableHeight}px (with ${this.padding}px padding)`);

    // Find optimal font size
    console.log(`[RENDERER]   Finding optimal font size for text: "${text}"`);
    const { fontSize, lines } = this.findOptimalFontSize(
      ctx,
      text,
      availableWidth,
      availableHeight
    );

    console.log(`[RENDERER]   ✓ Optimal font size: ${fontSize}px`);
    console.log(`[RENDERER]   ✓ Text wrapped into ${lines.length} line(s)`);
    lines.forEach((line, idx) => {
      console.log(`[RENDERER]     Line ${idx + 1}: "${line}"`);
    });

    // Set font
    ctx.font = `${this.fontWeight} ${fontSize}px ${this.fontFamily}`;
    ctx.fillStyle = this.textColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // Calculate line height and spacing (matches Python implementation)
    const lineHeight = fontSize;
    const lineSpacing = Math.floor(fontSize * 0.2);
    const totalHeight = lines.length * lineHeight + (lines.length - 1) * lineSpacing;

    console.log(`[RENDERER]   Line height: ${lineHeight}px, spacing: ${lineSpacing}px`);
    console.log(`[RENDERER]   Total text height: ${totalHeight}px`);

    // Starting Y position (vertically centered)
    let currentY = y1 + Math.floor((height - totalHeight) / 2);
    const verticalOffset = currentY - y1;
    console.log(`[RENDERER]   Vertical centering offset: ${verticalOffset}px`);

    // Draw each line
    for (const line of lines) {
      const centerX = x1 + Math.floor(width / 2);
      ctx.fillText(line, centerX, currentY);
      currentY += lineHeight + lineSpacing;
    }

    console.log(`[RENDERER]   ✓ All ${lines.length} lines drawn`);
  }

  findOptimalFontSize(ctx, text, maxWidth, maxHeight) {
    // Match Python: range from 72 down to 12
    for (let size = 72; size >= 12; size -= 2) {
      ctx.font = `${this.fontWeight} ${size}px ${this.fontFamily}`;

      const lines = this.wrapText(ctx, text, maxWidth);
      const lineHeight = size;
      const lineSpacing = Math.floor(size * 0.2);
      const totalHeight = lines.length * lineHeight + (lines.length - 1) * lineSpacing;

      if (totalHeight <= maxHeight) {
        return { fontSize: size, lines };
      }
    }

    // Fallback: use minimum size
    ctx.font = `${this.fontWeight} 12px ${this.fontFamily}`;
    return { fontSize: 12, lines: this.wrapText(ctx, text, maxWidth) };
  }

  wrapText(ctx, text, maxWidth) {
    const words = text.split(' ');
    const lines = [];
    let currentLine = '';

    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      const metrics = ctx.measureText(testLine);

      if (metrics.width > maxWidth && currentLine) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    }

    if (currentLine) {
      lines.push(currentLine);
    }

    return lines.length > 0 ? lines : [text];
  }
}

window.TextRenderer = TextRenderer;