// Content script - Server-based image translator with toggle mode
console.log('═══════════════════════════════════════════════════════════');
console.log('[CONTENT] Webtoon Translator: Content script loaded');
console.log('[CONTENT] Mode: Toggle-based translation (Ctrl+Q)');
console.log('[CONTENT] Architecture:');
console.log('[CONTENT]   1. Extension injects CORS headers (bypasses security)');
console.log('[CONTENT]   2. Find all <img> tags on ANY page (dynamic)');
console.log('[CONTENT]   3. SHA1-based caching (persistent across sessions)');
console.log('[CONTENT]   4. Extract pixel data IN PARALLEL from browser memory (NO re-download!)');
console.log('[CONTENT]   5. Send ALL images in ONE batch request → server processes in PARALLEL');
console.log('[CONTENT] Press Ctrl+Q to toggle translation ON/OFF');
console.log('═══════════════════════════════════════════════════════════');

// State management
let translationEnabled = false; // Toggle state: OFF by default
let isProcessing = false;
let translationCache = new Map(); // In-memory cache: sha1 -> translatedDataUrl
let originalImages = new Map(); // Track: img element -> original src
const CACHE_STORAGE_KEY = 'webtoon_translation_cache'; // Persistent storage key

// Server configuration
const SERVER_URL = 'http://localhost:1420/translate';
const SERVER_URL_BATCH = 'http://localhost:1420/translate-batch';
const SERVER_URL_MODE = 'http://localhost:1420/translate-url';
const SERVER_URL_LOCAL = 'http://localhost:1420/translate-local';
const SERVER_URL_LOCAL_BATCH = 'http://localhost:1420/translate-local-batch';

/**
 * Calculate SHA1 checksum of image blob
 */
async function calculateSHA1(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest('SHA-1', arrayBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  return hashHex;
}

/**
 * Calculate SHA1 checksum of string (for local file paths)
 */
async function calculateStringSHA1(str) {
  const encoder = new TextEncoder();
  const data = encoder.encode(str);
  const hashBuffer = await crypto.subtle.digest('SHA-1', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  return hashHex;
}

/**
 * Load translation cache from chrome.storage.local
 */
async function loadCacheFromStorage() {
  try {
    const result = await chrome.storage.local.get(CACHE_STORAGE_KEY);
    if (result[CACHE_STORAGE_KEY]) {
      const cacheData = result[CACHE_STORAGE_KEY];
      translationCache = new Map(Object.entries(cacheData));
      console.log(`[CACHE] Loaded ${translationCache.size} cached translations from storage`);
    }
  } catch (error) {
    console.error('[CACHE] Failed to load cache:', error);
  }
}

/**
 * Save translation cache to chrome.storage.local
 * Falls back to saving to a file for local files if storage quota is exceeded
 */
async function saveCacheToStorage() {
  try {
    const cacheData = Object.fromEntries(translationCache);
    await chrome.storage.local.set({ [CACHE_STORAGE_KEY]: cacheData });
    console.log(`[CACHE] Saved ${translationCache.size} translations to storage`);
  } catch (error) {
    console.error('[CACHE] Failed to save cache:', error);

    // If storage quota is exceeded, save local file translations to disk
    if (error.message && error.message.includes('quota')) {
      console.log('[CACHE] Storage quota exceeded, saving local file translations to disk...');
      await saveLocalFileCacheToDisk();
    }
  }
}

/**
 * Request file system access permission for cache saving
 */
async function requestFileSystemAccess() {
  try {
    // Check if file system access is available
    if (!window.showDirectoryPicker) {
      console.warn('[CACHE] File System API not available in this browser');
      return false;
    }

    // Request file system access permission
    const hasPermission = await navigator.permissions.query({ name: 'file-system-access' });

    if (hasPermission.state === 'granted') {
      console.log('[CACHE] File system access already granted');
      return true;
    }

    // Request permission
    const permission = await navigator.permissions.request({ name: 'file-system-access' });
    console.log(`[CACHE] File system access permission: ${permission.state}`);
    return permission.state === 'granted';

  } catch (error) {
    console.warn('[CACHE] File system access permission check failed:', error);
    return false;
  }
}

/**
 * Save local file translations to disk files as fallback
 */
async function saveLocalFileCacheToDisk() {
  try {
    // First try to request permission if not already granted
    const hasAccess = await requestFileSystemAccess();
    if (!hasAccess) {
      console.log('[CACHE] File system access not available, skipping disk cache save');
      showNotification('Storage quota exceeded. File system access permission needed for disk caching.', 'warning');
      return;
    }

    // Request directory selection
    const fileHandle = await window.showDirectoryPicker({
      mode: 'readwrite',
      startIn: 'downloads'
    });

    console.log(`[CACHE] Selected directory: ${fileHandle.name}`);

    let savedCount = 0;
    let failedCount = 0;

    // Save cache entries one by one to avoid overwhelming the system
    for (const [sha1, dataUrl] of translationCache.entries()) {
      try {
        // Extract base64 data
        const base64Data = dataUrl.split(',')[1];
        const binaryData = atob(base64Data);
        const bytes = new Uint8Array(binaryData.length);

        for (let i = 0; i < binaryData.length; i++) {
          bytes[i] = binaryData.charCodeAt(i);
        }

        // Create cache filename (use first 8 chars to keep filenames reasonable)
        const cacheFileName = `cache_${sha1.substring(0, 8)}.png`;
        const cacheFileHandle = await fileHandle.getFileHandle(cacheFileName, { create: true });
        const writable = await cacheFileHandle.createWritable();
        await writable.write(bytes);
        await writable.close();

        savedCount++;

        // Log progress every 10 files
        if (savedCount % 10 === 0) {
          console.log(`[CACHE] Progress: ${savedCount}/${translationCache.size} files saved`);
        }

      } catch (fileError) {
        failedCount++;
        console.warn(`[CACHE] Failed to save cache file for ${sha1.substring(0, 8)}:`, fileError.message);
      }
    }

    console.log(`[CACHE] Cache save complete: ${savedCount}/${translationCache.size} files saved, ${failedCount} failed`);

    // Save metadata file with cache mapping
    const metadata = {
      savedAt: new Date().toISOString(),
      totalEntries: translationCache.size,
      savedCount: savedCount,
      failedCount: failedCount,
      cacheMapping: Array.from(translationCache.entries()).map(([sha1, dataUrl]) => ({
        sha1: sha1,
        fileName: `cache_${sha1.substring(0, 8)}.png`,
        size: dataUrl.length
      }))
    };

    const metadataHandle = await fileHandle.getFileHandle('translation_cache_metadata.json', { create: true });
    const metadataWritable = await metadataHandle.createWritable();
    await metadataWritable.write(JSON.stringify(metadata, null, 2));
    await metadataWritable.close();

    console.log('[CACHE] Cache metadata saved to disk');
    showNotification(`Cache saved: ${savedCount} files to disk`, 'success');

  } catch (error) {
    console.error('[CACHE] Failed to save cache to disk:', error);

    // Handle specific errors
    if (error.name === 'AbortError') {
      console.log('[CACHE] User cancelled directory selection');
    } else if (error.name === 'NotAllowedError') {
      showNotification('Storage quota exceeded. File access denied for disk caching.', 'warning');
    } else {
      showNotification('Storage quota exceeded. Consider clearing browser cache.', 'warning');
    }
  }
}

// Load cache on startup
loadCacheFromStorage();

// Listen for messages from background/popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log(`[CONTENT] Message received:`, request);

  if (request.action === 'translate-all') {
    toggleTranslation();
    sendResponse({ status: 'toggled', enabled: translationEnabled });
  }
  return true;
});

/**
 * Toggle translation ON/OFF
 */
async function toggleTranslation() {
  if (isProcessing) {
    console.warn('[CONTENT] ⚠ Translation in progress, ignoring toggle request');
    showNotification('Translation in progress...', 'info');
    return;
  }

  // Toggle state
  translationEnabled = !translationEnabled;

  console.log(`\n[TOGGLE] Translation mode: ${translationEnabled ? 'ON' : 'OFF'}`);

  if (translationEnabled) {
    // Mode ON: Translate all images
    showNotification('🌐 Translation mode: ON', 'info');
    await translateAllImages();
  } else {
    // Mode OFF: Restore all images
    showNotification('📖 Translation mode: OFF', 'info');
    restoreAllImages();
  }

  // Update badge indicator
  updateToggleBadge();
}

/**
 * Translate all large images on the page
 */
async function translateAllImages() {
  if (isProcessing) return;

  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║          TRANSLATING IMAGES (TOGGLE MODE ON)            ║');
  console.log('╚═══════════════════════════════════════════════════════════╝');

  isProcessing = true;
  const startTime = performance.now();

  try {
    // Find all large images (likely comic images)
    console.log('\n[STEP 1] Finding comic images on page...');
    console.log(`[STEP 1] Page URL: ${window.location.href}`);
    const allImages = document.querySelectorAll('img');
    console.log(`[STEP 1] Found ${allImages.length} total <img> elements`);

    // Debug: Log all images found
    console.log('\n[DEBUG] All images on page:');
    allImages.forEach((img, idx) => {
      const rect = img.getBoundingClientRect();
      console.log(`  [${idx + 1}] URL: ${img.src}`);
      console.log(`      Size: ${img.naturalWidth}x${img.naturalHeight}px (display: ${Math.round(rect.width)}x${Math.round(rect.height)}px)`);
      console.log(`      Visible: ${img.offsetParent !== null}, Width OK: ${rect.width > 100}, Height OK: ${rect.height > 100}`);
    });

    // Filter: visible, large enough for comic content
    const comicImages = Array.from(allImages).filter(img => {
      const rect = img.getBoundingClientRect();
      const isVisible = rect.width > 100 && rect.height > 100 && img.offsetParent !== null;
      return isVisible;
    });

    console.log(`\n[STEP 1] Filtered to ${comicImages.length} comic-sized images (>100x100px)`);
    console.log('[DEBUG] Images selected for translation:');
    comicImages.forEach((img, idx) => {
      console.log(`  [${idx + 1}] ${img.src}`);
    });

    if (comicImages.length === 0) {
      showNotification('No comic images found on page', 'info');
      translationEnabled = false;
      isProcessing = false;
      return;
    }

    showNotification(`Translating ${comicImages.length} image(s)...`, 'info');

    // Process images via BATCH request (all at once!)
    console.log('\n[STEP 2] Preparing BATCH translation request...');
    let successCount = 0;
    let errorCount = 0;
    let skippedCount = 0;

    // Convert all images to blobs and calculate SHA1 checksums
    console.log(`\n[STEP 2] Converting ${comicImages.length} images to blobs and calculating checksums IN PARALLEL...`);

    // Create conversion + SHA1 tasks (parallel execution!)
    const conversionTasks = comicImages.map(async (img, i) => {
      const index = i + 1;
      const originalSrc = img.src;
      console.log(`[IMAGE ${index}] Starting blob conversion + SHA1...`);

      try {
        const result = await imageToBlob(img);

        // Handle local files
        if (result && result.isLocalFile) {
          console.log(`[IMAGE ${index}] ✓ Local file detected: ${result.localPath}`);
          // For local files, we need to calculate SHA1 from image content
          // Try to get content-based hash from server
          const sha1 = await getLocalFileHash(result.localPath, img.naturalWidth, img.naturalHeight);
          return { img, originalSrc, isLocalFile: true, localPath: result.localPath, sha1, index, success: true };
        } else {
          // Regular image blob
          const blob = result;
          const sha1 = await calculateSHA1(blob);
          console.log(`[IMAGE ${index}] ✓ Blob: ${(blob.size / 1024).toFixed(2)} KB, SHA1: ${sha1.substring(0, 12)}...`);
          return { img, originalSrc, blob, sha1, index, success: true };
        }
      } catch (error) {
        console.error(`[IMAGE ${index}] ✗ Conversion failed:`, error.message);
        return { img, originalSrc, index, success: false, error };
      }
    });

    // Wait for ALL conversions to complete in parallel
    const conversionResults = await Promise.all(conversionTasks);

    console.log(`\n[STEP 3] ✓ Converted ${conversionResults.filter(r => r.success).length}/${comicImages.length} images`);

    // Separate cached vs uncached (check by SHA1)
    const uncachedImages = [];
    let cacheHits = 0;

    for (const result of conversionResults) {
      if (!result.success) {
        errorCount++;
        continue;
      }

      const { img, originalSrc, sha1, index } = result;

      // Check cache by SHA1
      if (translationCache.has(sha1)) {
        console.log(`[IMAGE ${index}] ✓ Cache HIT (SHA1: ${sha1.substring(0, 12)}...)`);
        const cachedDataUrl = translationCache.get(sha1);
        originalImages.set(img, originalSrc);
        img.src = cachedDataUrl;
        img.dataset.translated = 'true';
        successCount++;
        cacheHits++;
      } else {
        console.log(`[IMAGE ${index}] Cache MISS - needs translation`);
        uncachedImages.push(result);
      }
    }

    console.log(`[STEP 3] Cache: ${cacheHits} hits, ${uncachedImages.length} misses`);

    if (uncachedImages.length === 0) {
      console.log('[STEP 3] All images cached, done!');
      return;
    }

    const uncachedData = uncachedImages;

    // Separate local files and regular images
    const localFiles = uncachedData.filter(d => d.isLocalFile);
    const regularImages = uncachedData.filter(d => !d.isLocalFile);

    console.log(`\n[STEP 4] Processing uncached images: ${localFiles.length} local files, ${regularImages.length} regular images`);

    let batchResults = { results: [] };

    try {
      // Process regular images in batch
      if (regularImages.length > 0) {
        console.log(`[BATCH] Sending regular images batch: ${regularImages.length} images`);
        console.log(`[BATCH] Server endpoint: ${SERVER_URL_BATCH}`);
        console.log(`[BATCH] Payload size: ~${(regularImages.reduce((sum, d) => sum + d.blob.size, 0) / 1024).toFixed(2)} KB total`);

        const regularResults = await translateBatch(regularImages.map(d => d.blob));
        batchResults.results.push(...regularResults.results);
      }

      // Process local files in batch using the new endpoint
      if (localFiles.length > 0) {
        console.log(`[LOCAL] Processing local files in batch: ${localFiles.length} files`);

        try {
          const localPaths = localFiles.map(f => f.localPath);
          const localBatchResults = await translateLocalFilesBatch(localPaths);
          batchResults.results.push(...localBatchResults.results);
        } catch (error) {
          console.error(`[LOCAL] Failed to process local batch: ${error.message}`);
          // Fallback to individual processing if batch fails
          console.log(`[LOCAL] Falling back to individual processing...`);

          for (let i = 0; i < localFiles.length; i++) {
            const localFile = localFiles[i];
            console.log(`[LOCAL] Processing file ${i + 1}/${localFiles.length}: ${localFile.localPath}`);

            try {
              const localResult = await translateLocalFile(localFile.localPath);
              batchResults.results.push(localResult);
            } catch (error) {
              console.error(`[LOCAL] Failed to process local file: ${error.message}`);
              batchResults.results.push({
                success: false,
                error: error.message,
                filename: localFile.localPath,
                index: localFile.index
              });
            }
          }
        }
      }

      console.log(`\n[BATCH] ✓ Received ${batchResults.results.length} results`);

      // Verify we got expected number of results
      const expectedResults = regularImages.length + localFiles.length;
      if (batchResults.results.length !== expectedResults) {
        console.warn(`[BATCH] ⚠ Result count mismatch: expected ${expectedResults}, got ${batchResults.results.length}`);
      }

      // Apply results to images using MAPPING
      // We need to map results back to the correct images since we processed them separately
      let newCacheEntries = 0;
      let resultIndex = 0;

      // First, process regular images results
      for (let i = 0; i < regularImages.length; i++) {
        const image = regularImages[i];
        const result = batchResults.results[resultIndex++];

        if (result && result.success) {
          console.log(`[IMAGE ${image.index}] ✓ Translation successful: ${(result.size / 1024).toFixed(2)} KB, ${result.bubbles} bubbles`);

          // Cache by SHA1 (content-based, works across URLs)
          translationCache.set(image.sha1, result.dataUrl);
          newCacheEntries++;

          // Display translated image
          originalImages.set(image.img, image.originalSrc);
          image.img.src = result.dataUrl;
          image.img.dataset.translated = 'true';
          successCount++;
        } else {
          console.error(`[IMAGE ${image.index}] ✗ Translation failed: ${result ? result.error : 'No result received'}`);
          errorCount++;
        }
      }

      // Then, process local files results
      for (let i = 0; i < localFiles.length; i++) {
        const localFile = localFiles[i];
        const result = batchResults.results[resultIndex++];

        if (result && result.success) {
          console.log(`[IMAGE ${localFile.index}] ✓ Local file translation successful: ${(result.size / 1024).toFixed(2)} KB, ${result.bubbles} bubbles`);

          // Cache by SHA1 (file path-based)
          translationCache.set(localFile.sha1, result.dataUrl);
          newCacheEntries++;

          // Display translated image
          originalImages.set(localFile.img, localFile.originalSrc);
          localFile.img.src = result.dataUrl;
          localFile.img.dataset.translated = 'true';
          successCount++;
        } else {
          console.error(`[IMAGE ${localFile.index}] ✗ Local file translation failed: ${result ? result.error : 'No result received'}`);
          errorCount++;
        }
      }

      console.log(`[BATCH] ✓ Applied all results using positional mapping (order preserved)`);
      console.log(`[BATCH] ✓ Added ${newCacheEntries} new entries to cache (total: ${translationCache.size})`);

      if (batchResults.results.length > 0 && batchResults.results[0].index) {
        console.log(`[BATCH] Server confirmed: results returned in same order as input (indices 1-${batchResults.results.length})`);
      }

      // Save cache to persistent storage
      if (newCacheEntries > 0) {
        await saveCacheToStorage();
      }

    } catch (error) {
      console.error('[BATCH] ✗ Batch request failed:', error.message);
      errorCount += blobsData.length;
    }

    // Summary
    const endTime = performance.now();
    const duration = ((endTime - startTime) / 1000).toFixed(2);

    console.log('\n╔═══════════════════════════════════════════════════════════╗');
    console.log('║          TRANSLATION COMPLETE                            ║');
    console.log('╚═══════════════════════════════════════════════════════════╝');
    console.log(`[SUMMARY] Total images: ${comicImages.length}`);
    console.log(`[SUMMARY] Translated: ${successCount} (${cacheHits} from cache, ${successCount - cacheHits} new)`);
    console.log(`[SUMMARY] Skipped (no bubbles): ${skippedCount}`);
    console.log(`[SUMMARY] Errors: ${errorCount}`);
    console.log(`[SUMMARY] Cache size: ${translationCache.size} entries`);
    console.log(`[SUMMARY] Duration: ${duration}s`);
    if (cacheHits > 0) {
      const hitRate = ((cacheHits / comicImages.length) * 100).toFixed(1);
      console.log(`[SUMMARY] Cache hit rate: ${hitRate}% (saved ${cacheHits} API calls!)`);
    }
    console.log('═══════════════════════════════════════════════════════════\n');

    if (successCount > 0) {
      const cacheMsg = cacheHits > 0 ? ` (${cacheHits} cached)` : '';
      showNotification(`✓ Translated ${successCount} image(s)${cacheMsg} in ${duration}s`, 'success');
    } else if (skippedCount > 0) {
      showNotification('No speech bubbles detected in images', 'info');
    } else if (errorCount > 0) {
      showNotification(`Failed to translate images`, 'error');
    }

  } catch (error) {
    console.error('[CONTENT] Fatal error:', error);
    showNotification(`Error: ${error.message}`, 'error');
    translationEnabled = false;
  } finally {
    isProcessing = false;
  }
}

/**
 * Restore all images to their originals
 */
function restoreAllImages() {
  console.log('\n[RESTORE] Restoring all images to originals...');

  let restoredCount = 0;

  // Restore from our tracking map
  originalImages.forEach((originalSrc, img) => {
    if (img && img.parentElement) { // Check if element still exists
      console.log(`[RESTORE] Restoring: ${originalSrc.substring(0, 60)}...`);
      img.src = originalSrc;
      delete img.dataset.translated;
      restoredCount++;
    }
  });

  originalImages.clear();

  console.log(`[RESTORE] ✓ Restored ${restoredCount} images`);

  if (restoredCount > 0) {
    showNotification(`Restored ${restoredCount} images`, 'info');
  }
}

/**
 * Check if an image URL is a local file
 */
function isLocalFile(url) {
  return url.startsWith('file://') || url.startsWith('file:/');
}

/**
 * Convert img element to blob or handle local files
 * CORS headers are injected by extension, so canvas method works for web images!
 * Local files are handled by sending the file path to server.
 */
async function imageToBlob(img) {
  const imageUrl = img.src;

  // Handle local files differently
  if (isLocalFile(imageUrl)) {
    console.log(`[DEBUG] Detected local file: ${imageUrl}`);
    // For local files, we'll handle them in the batch processing
    return { isLocalFile: true, localPath: imageUrl };
  }

  // Ensure image has crossorigin attribute for CORS
  if (!img.crossOrigin) {
    console.log(`[DEBUG] Setting crossOrigin attribute on image...`);
    // Clone the image with crossorigin attribute
    const newImg = new Image();
    newImg.crossOrigin = 'anonymous';

    await new Promise((resolve, reject) => {
      newImg.onload = resolve;
      newImg.onerror = () => {
        console.log(`[DEBUG] Failed to reload with crossOrigin, trying original...`);
        resolve(); // Continue anyway
      };
      newImg.src = img.src;

      // Timeout after 5 seconds
      setTimeout(() => {
        console.log(`[DEBUG] Reload timeout, using original image...`);
        resolve();
      }, 5000);
    });

    // Try the new image first, fall back to original
    if (newImg.complete && newImg.naturalWidth > 0) {
      console.log(`[DEBUG] Using reloaded image with crossOrigin`);
      return await imageToBlobViaCanvas(newImg);
    }
  }

  // Try canvas method (CORS headers injected by extension make this work)
  try {
    return await imageToBlobViaCanvas(img);
  } catch (error) {
    console.log(`[DEBUG] Canvas failed: ${error.message}`);
    console.log(`[DEBUG] This shouldn't happen with CORS headers injected. Check extension permissions.`);
    throw error;
  }
}

/**
 * Method 1: Canvas (fast, same-origin only)
 */
async function imageToBlobViaCanvas(img) {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;

    try {
      ctx.drawImage(img, 0, 0);
      canvas.toBlob(blob => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to convert image to blob'));
        }
      }, 'image/png');
    } catch (error) {
      reject(new Error(`Canvas error: ${error.message}`));
    }
  });
}

/**
 * Method 2: Fetch via background script (bypasses CORS)
 */
async function imageToBlobViaFetch(imageUrl) {
  console.log(`[DEBUG] Fetching image via background script (CORS bypass): ${imageUrl}`);

  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(
      { action: 'fetch-image', url: imageUrl },
      (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(`Background fetch error: ${chrome.runtime.lastError.message}`));
          return;
        }

        if (response.success) {
          console.log(`[DEBUG] Background fetched successfully: ${(response.size / 1024).toFixed(2)} KB`);

          // Convert data URL back to blob
          fetch(response.dataUrl)
            .then(res => res.blob())
            .then(blob => resolve(blob))
            .catch(err => reject(new Error(`Data URL conversion failed: ${err.message}`)));
        } else {
          reject(new Error(`Background fetch failed: ${response.error}`));
        }
      }
    );
  });
}

/**
 * Send multiple images to server for BATCH translation (ONE request for ALL images)
 */
async function translateBatch(imageBlobs) {
  const formData = new FormData();

  // Add all images to form data
  imageBlobs.forEach((blob, index) => {
    formData.append('files', blob, `image_${index}.png`);
  });

  const response = await fetch(SERVER_URL_BATCH, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Server error (${response.status}): ${errorText}`);
  }

  return await response.json();
}

/**
 * Send image URL to server for translation (server fetches image, bypasses CORS)
 */
async function translateViaServerUrl(imageUrl) {
  const url = new URL(SERVER_URL_MODE);
  url.searchParams.append('url', imageUrl);

  const response = await fetch(url.toString(), {
    method: 'POST'
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Server error (${response.status}): ${errorText}`);
  }

  return await response.blob();
}

/**
 * Send multiple local file paths to server for batch translation
 */
async function translateLocalFilesBatch(localPaths) {
  const response = await fetch(SERVER_URL_LOCAL_BATCH, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ filePaths: localPaths })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Server error (${response.status}): ${errorText}`);
  }

  return await response.json();
}

/**
 * Get local file hash from server for caching
 */
async function getLocalFileHash(localPath, naturalWidth, naturalHeight) {
  try {
    const response = await fetch(`${SERVER_URL_LOCAL}/hash`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ filePath: localPath })
    });

    if (!response.ok) {
      // If hash endpoint fails, fall back to path-based hash
      console.log(`[CACHE] Hash endpoint failed for ${localPath}, using path-based hash`);
      const imgDimensions = `${naturalWidth}x${naturalHeight}`;
      const proxyId = `${localPath}_${imgDimensions}`;
      return await calculateStringSHA1(proxyId);
    }

    const result = await response.json();
    return result.hash;
  } catch (error) {
    console.log(`[CACHE] Error getting file hash: ${error.message}, using path-based hash`);
    const imgDimensions = `${naturalWidth}x${naturalHeight}`;
    const proxyId = `${localPath}_${imgDimensions}`;
    return await calculateStringSHA1(proxyId);
  }
}

/**
 * Send local file path to server for translation
 */
async function translateLocalFile(localPath) {
  const response = await fetch(SERVER_URL_LOCAL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ filePath: localPath })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Server error (${response.status}): ${errorText}`);
  }

  const responseBlob = await response.blob();

  // Convert blob to data URL to match the expected format
  return {
    success: true,
    dataUrl: await blobToDataUrl(responseBlob),
    size: responseBlob.size,
    bubbles: 'unknown', // We don't have this info from local file processing
    filename: localPath.split('/').pop() || localPath.split('\\').pop() || 'local_file'
  };
}

/**
 * Send image blob to server for translation (legacy method)
 */
async function translateViaServer(imageBlob) {
  const formData = new FormData();
  formData.append('file', imageBlob, 'image.png');

  const response = await fetch(SERVER_URL, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Server error (${response.status}): ${errorText}`);
  }

  return await response.blob();
}

/**
 * Convert blob to data URL
 */
async function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Update toggle indicator badge
 */
function updateToggleBadge() {
  // Remove existing badge
  const existingBadge = document.querySelector('.webtoon-translator-toggle-badge');
  if (existingBadge) {
    existingBadge.remove();
  }

  if (!translationEnabled) return;

  // Create toggle indicator
  const badge = document.createElement('div');
  badge.className = 'webtoon-translator-toggle-badge';
  badge.innerHTML = `
    <div style="font-weight: bold; font-size: 14px;">🌐 Translation: ON</div>
    <div style="font-size: 11px; opacity: 0.9;">Press Ctrl+Q to toggle OFF</div>
  `;
  badge.style.cssText = `
    position: fixed;
    top: 20px;
    left: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
    z-index: 999999;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    pointer-events: none;
    animation: slideInFromLeft 0.3s ease-out;
  `;

  // Add animation
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideInFromLeft {
      from { transform: translateX(-300px); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
  `;
  document.head.appendChild(style);

  document.body.appendChild(badge);
}

/**
 * Show notification to user
 */
function showNotification(message, type = 'info') {
  console.log(`[NOTIFICATION] ${type.toUpperCase()}: ${message}`);

  // Create notification element
  const notification = document.createElement('div');
  notification.textContent = message;

  const colors = {
    info: 'rgba(0, 120, 215, 0.95)',
    success: 'rgba(0, 200, 0, 0.95)',
    error: 'rgba(220, 0, 0, 0.95)'
  };

  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: ${colors[type] || colors.info};
    color: white;
    padding: 16px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    z-index: 999999;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    max-width: 400px;
    animation: slideIn 0.3s ease-out;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  `;

  // Add animation
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideIn {
      from { transform: translateX(400px); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
  `;
  document.head.appendChild(style);

  document.body.appendChild(notification);

  // Auto-remove after 3 seconds
  setTimeout(() => {
    notification.style.animation = 'slideIn 0.3s ease-out reverse';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    console.log('[CONTENT] ✓ DOM loaded, ready for translation');
  });
} else {
  console.log('[CONTENT] ✓ DOM already loaded, ready for translation');
}

console.log('[CONTENT] Content script initialization complete');
console.log('[CONTENT] 📖 Translation mode: OFF (default)');
console.log('[CONTENT] Press Ctrl+Q to toggle translation ON/OFF');
console.log('═══════════════════════════════════════════════════════════\n');
