// Content script - Batch image translator with extensive logging
console.log('═══════════════════════════════════════════════════════════');
console.log('[CONTENT] Webtoon Translator: Content script loaded');
console.log('[CONTENT] Mode: Batch processing (all images on page)');
console.log('[CONTENT] Hotkey: Ctrl+Q');
console.log('═══════════════════════════════════════════════════════════');

let isProcessing = false;
let processedImages = new Map(); // Track which images have been translated

// Initialize components
console.log('[CONTENT] Initializing components...');
const detector = new window.BubbleDetector();
const geminiClient = new window.GeminiClient();
const renderer = new window.TextRenderer();
console.log('[CONTENT] ✓ Components created');

// Initialize detector on load
console.log('[CONTENT] Starting detector initialization...');
detector.initialize()
  .then(() => {
    console.log('[CONTENT] ✓ Detector initialized successfully');
  })
  .catch(err => {
    console.error('[CONTENT] ✗ Failed to initialize detector:', err);
  });

// Listen for messages from background/popup (chrome.commands only works in background.js)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log(`[CONTENT] Message received:`, request);

  if (request.action === 'translate-all') {
    processAllImages();
    sendResponse({ status: 'started' });
  } else if (request.action === 'restore-all') {
    restoreAllImages();
    sendResponse({ status: 'restored' });
  }
  return true;
});

/**
 * Main function: Find and process all images on the page
 */
async function processAllImages() {
  if (isProcessing) {
    console.warn('[CONTENT] ⚠ Translation already in progress, ignoring request');
    showNotification('Translation already in progress...', 'info');
    return;
  }

  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║          BATCH IMAGE PROCESSING STARTED                  ║');
  console.log('╚═══════════════════════════════════════════════════════════╝');

  isProcessing = true;
  const startTime = performance.now();

  try {
    // Step 1: Find all images on page
    console.log('\n[STEP 1] Finding all images on page...');
    const allImages = document.querySelectorAll('img');
    console.log(`[STEP 1] Found ${allImages.length} total <img> elements on page`);

    // Filter visible images
    const visibleImages = Array.from(allImages).filter(img => {
      const rect = img.getBoundingClientRect();
      const isVisible = rect.width > 50 && rect.height > 50 &&
                       img.offsetParent !== null;
      if (!isVisible) {
        console.log(`[STEP 1] Skipping image (too small or hidden): ${img.src.substring(0, 60)}...`);
      }
      return isVisible;
    });

    console.log(`[STEP 1] Filtered to ${visibleImages.length} visible images (>50x50px)`);

    if (visibleImages.length === 0) {
      throw new Error('No visible images found on page');
    }

    showNotification(`Found ${visibleImages.length} images. Scanning for speech bubbles...`, 'info');

    // Step 2: Process each image
    console.log('\n[STEP 2] Processing images for speech bubble detection...');

    const results = [];
    let processedCount = 0;
    let successCount = 0;
    let errorCount = 0;
    let noBubbleCount = 0;

    for (let i = 0; i < visibleImages.length; i++) {
      const img = visibleImages[i];
      processedCount++;

      console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
      console.log(`[IMAGE ${i + 1}/${visibleImages.length}] Processing image...`);
      console.log(`[IMAGE ${i + 1}] URL: ${img.src.substring(0, 80)}...`);
      console.log(`[IMAGE ${i + 1}] Dimensions: ${img.naturalWidth}x${img.naturalHeight}px`);
      console.log(`[IMAGE ${i + 1}] Display size: ${img.width}x${img.height}px`);

      // Skip already translated images
      if (processedImages.has(img.src)) {
        console.log(`[IMAGE ${i + 1}] ⏭ Skipping (already translated)`);
        continue;
      }

      try {
        // Update progress notification
        showNotification(
          `Processing image ${processedCount}/${visibleImages.length}...`,
          'info'
        );

        // Detect bubbles in this image
        console.log(`[IMAGE ${i + 1}] Starting bubble detection...`);
        const detections = await detector.detect(img);
        console.log(`[IMAGE ${i + 1}] ✓ Detection complete`);
        console.log(`[IMAGE ${i + 1}] Found ${detections.length} speech bubbles`);

        if (detections.length === 0) {
          console.log(`[IMAGE ${i + 1}] ⊘ No bubbles detected, skipping translation`);
          noBubbleCount++;
          continue;
        }

        // Log detected bubble details
        detections.forEach((det, idx) => {
          const [x1, y1, x2, y2] = det.textBox;
          const width = x2 - x1;
          const height = y2 - y1;
          console.log(`[IMAGE ${i + 1}] Bubble ${idx + 1}: Position=(${x1},${y1}) Size=${width}x${height}px Score=${det.score.toFixed(3)}`);
        });

        // Translate this image
        console.log(`[IMAGE ${i + 1}] Starting translation of ${detections.length} bubbles...`);
        const translations = await geminiClient.translateBubbles(img, detections);
        console.log(`[IMAGE ${i + 1}] ✓ Translation complete, received ${translations.length} results`);

        // Render translations on image
        console.log(`[IMAGE ${i + 1}] Rendering translations onto image...`);
        const translatedDataUrl = await renderer.renderTranslations(img, detections, translations);
        console.log(`[IMAGE ${i + 1}] ✓ Rendering complete`);

        // Store original and replace
        console.log(`[IMAGE ${i + 1}] Replacing original image with translated version...`);
        img.dataset.originalSrc = img.src;
        img.dataset.translated = 'true';
        img.src = translatedDataUrl;
        processedImages.set(img.dataset.originalSrc, translatedDataUrl);

        // Add visual indicator
        addTranslatedBadge(img, detections.length);

        successCount++;
        console.log(`[IMAGE ${i + 1}] ✓ Successfully translated ${translations.length} bubbles`);

        results.push({
          imageIndex: i + 1,
          src: img.dataset.originalSrc,
          bubbleCount: detections.length,
          status: 'success'
        });

      } catch (error) {
        errorCount++;
        console.error(`[IMAGE ${i + 1}] ✗ Error processing image:`, error);
        console.error(`[IMAGE ${i + 1}] Error details:`, error.stack);

        results.push({
          imageIndex: i + 1,
          src: img.src,
          status: 'error',
          error: error.message
        });
      }
    }

    // Step 3: Summary
    const endTime = performance.now();
    const duration = ((endTime - startTime) / 1000).toFixed(2);

    console.log('\n╔═══════════════════════════════════════════════════════════╗');
    console.log('║          BATCH PROCESSING COMPLETE                       ║');
    console.log('╚═══════════════════════════════════════════════════════════╝');
    console.log(`[SUMMARY] Total images found: ${visibleImages.length}`);
    console.log(`[SUMMARY] Successfully translated: ${successCount}`);
    console.log(`[SUMMARY] No bubbles detected: ${noBubbleCount}`);
    console.log(`[SUMMARY] Errors: ${errorCount}`);
    console.log(`[SUMMARY] Total time: ${duration} seconds`);
    console.log(`[SUMMARY] Average time per image: ${(duration / visibleImages.length).toFixed(2)}s`);
    console.log('═══════════════════════════════════════════════════════════\n');

    // Show final notification
    if (successCount > 0) {
      showNotification(
        `✓ Translated ${successCount} image(s) with speech bubbles! (${duration}s)`,
        'success',
        5000
      );
    } else if (noBubbleCount === visibleImages.length) {
      showNotification(
        'No speech bubbles detected in any images on this page',
        'error'
      );
    } else {
      showNotification(
        `Processing complete. Success: ${successCount}, Errors: ${errorCount}`,
        'info'
      );
    }

  } catch (error) {
    console.error('\n[CONTENT] ✗ Fatal error during batch processing:', error);
    console.error('[CONTENT] Error stack:', error.stack);
    showNotification(`Error: ${error.message}`, 'error');
  } finally {
    isProcessing = false;
    console.log('[CONTENT] Processing flag reset, ready for next request\n');
  }
}

/**
 * Restore all translated images to their originals
 */
function restoreAllImages() {
  console.log('\n[RESTORE] Starting restoration of all translated images...');

  const translatedImages = document.querySelectorAll('img[data-translated="true"]');
  console.log(`[RESTORE] Found ${translatedImages.length} translated images`);

  let restoredCount = 0;
  translatedImages.forEach((img, index) => {
    if (img.dataset.originalSrc) {
      console.log(`[RESTORE] Restoring image ${index + 1}/${translatedImages.length}: ${img.dataset.originalSrc.substring(0, 60)}...`);
      img.src = img.dataset.originalSrc;
      delete img.dataset.translated;
      delete img.dataset.originalSrc;

      // Remove badge
      const badge = img.parentElement?.querySelector('.translation-badge');
      if (badge) {
        badge.remove();
        console.log(`[RESTORE] Removed badge from image ${index + 1}`);
      }

      restoredCount++;
    }
  });

  // Clear processed images map
  processedImages.clear();
  console.log(`[RESTORE] ✓ Restored ${restoredCount} images`);
  console.log('[RESTORE] Cleared processed images cache\n');

  showNotification(`Restored ${restoredCount} images to original`, 'success');
}

/**
 * Add a visual badge to translated images
 */
function addTranslatedBadge(img, bubbleCount) {
  console.log(`[BADGE] Adding translation badge to image (${bubbleCount} bubbles)`);

  // Remove existing badge
  const existingBadge = img.parentElement?.querySelector('.translation-badge');
  if (existingBadge) existingBadge.remove();

  // Create badge
  const badge = document.createElement('div');
  badge.className = 'translation-badge';
  badge.textContent = `Translated (${bubbleCount})`;
  badge.style.cssText = `
    position: absolute;
    top: 10px;
    right: 10px;
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    color: white;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: bold;
    z-index: 10000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    font-family: Arial, sans-serif;
    border: 2px solid rgba(255,255,255,0.3);
  `;

  // Ensure parent is positioned
  const parent = img.parentElement;
  if (parent) {
    const position = getComputedStyle(parent).position;
    if (position === 'static') {
      parent.style.position = 'relative';
      console.log(`[BADGE] Set parent position to relative`);
    }
    parent.appendChild(badge);
    console.log(`[BADGE] ✓ Badge added`);
  } else {
    console.warn(`[BADGE] ⚠ Cannot add badge: image has no parent element`);
  }
}

/**
 * Show notification to user
 */
function showNotification(message, type = 'info', duration = 3000) {
  console.log(`[NOTIFICATION] ${type.toUpperCase()}: ${message}`);

  // Remove existing notification
  const existing = document.getElementById('webtoon-translator-notification');
  if (existing) existing.remove();

  // Create notification
  const notification = document.createElement('div');
  notification.id = 'webtoon-translator-notification';
  notification.textContent = message;

  const colors = {
    info: '#2196F3',
    success: '#4CAF50',
    error: '#f44336',
  };

  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: ${colors[type] || colors.info};
    color: white;
    padding: 16px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    z-index: 100000;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    animation: slideIn 0.3s ease-out;
    font-family: Arial, sans-serif;
    max-width: 400px;
  `;

  // Add animation
  if (!document.getElementById('webtoon-translator-styles')) {
    const style = document.createElement('style');
    style.id = 'webtoon-translator-styles';
    style.textContent = `
      @keyframes slideIn {
        from {
          transform: translateX(400px);
          opacity: 0;
        }
        to {
          transform: translateX(0);
          opacity: 1;
        }
      }
    `;
    document.head.appendChild(style);
  }

  document.body.appendChild(notification);

  // Auto-remove
  setTimeout(() => {
    notification.style.animation = 'slideIn 0.3s ease-out reverse';
    setTimeout(() => notification.remove(), 300);
  }, duration);
}

// Log when page is fully loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    console.log('[CONTENT] ✓ DOM fully loaded, ready to process images');
  });
} else {
  console.log('[CONTENT] ✓ DOM already loaded, ready to process images');
}

console.log('[CONTENT] Content script initialization complete');
console.log('[CONTENT] Press Ctrl+Q to translate all images with speech bubbles');
console.log('═══════════════════════════════════════════════════════════\n');
