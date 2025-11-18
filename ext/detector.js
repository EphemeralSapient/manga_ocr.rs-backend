// RT-DETR Bubble Detector - Client-Side ONNX Inference
class BubbleDetector {
  constructor() {
    this.session = null;
    this.modelLoaded = false;
    this.inputSize = 640; // RT-DETR input size
    console.log('[DETECTOR] BubbleDetector instance created');
    console.log(`[DETECTOR] Configuration: inputSize=${this.inputSize}px`);
  }

  async initialize() {
    if (this.modelLoaded) {
      console.log('[DETECTOR] Model already loaded, skipping initialization');
      return;
    }

    console.log('[DETECTOR] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('[DETECTOR] Starting RT-DETR model initialization...');

    const startTime = performance.now();

    try {
      // Configure ONNX Runtime WASM paths for Chrome Extension
      console.log('[DETECTOR] Configuring ONNX Runtime WASM backend...');
      if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.getURL) {
        // Configure for multi-threaded CPU execution (no GPU/JSEP to avoid AveragePool error)
        ort.env.wasm.numThreads = 4; // Enable 4 threads for performance
        ort.env.wasm.simd = true; // Enable SIMD instructions
        ort.env.wasm.proxy = false; // Disable web worker proxy for lower overhead

        // Set base path for WASM files (ONNX Runtime will auto-select non-JSEP version)
        ort.env.wasm.wasmPaths = chrome.runtime.getURL('');

        console.log(`[DETECTOR] ✓ WASM base path: ${ort.env.wasm.wasmPaths}`);
        console.log('[DETECTOR] ✓ Multi-threaded SIMD WASM (CPU-only, no JSEP/GPU)');
        console.log('[DETECTOR] ✓ Threads: 4, SIMD: enabled, Proxy: disabled');
      } else {
        console.warn('[DETECTOR] ⚠ Chrome runtime not available, using default WASM paths');
      }

      // Load ONNX model from local file (bundled with extension)
      const modelUrl = chrome.runtime.getURL('detector.onnx');
      console.log(`[DETECTOR] Model URL: ${modelUrl}`);
      console.log('[DETECTOR] Loading from local file (161MB, bundled with extension)');
      console.log('[DETECTOR] Creating ONNX Runtime inference session...');

      // Create ONNX Runtime session
      // NOTE: Using CPU-only WASM because model has AveragePool with ceil_mode
      // which is not yet supported in WebGPU/WebGL backends
      console.log('[DETECTOR] Creating session with multi-threaded WASM (CPU)...');
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        executionMode: 'parallel', // Enable parallel execution
        interOpNumThreads: 4, // Use 4 threads for ops
        intraOpNumThreads: 4, // Use 4 threads within ops
      });

      const endTime = performance.now();
      const loadTime = ((endTime - startTime) / 1000).toFixed(2);

      this.modelLoaded = true;
      console.log('[DETECTOR] ✓ RT-DETR model loaded successfully');
      console.log(`[DETECTOR] Load time: ${loadTime} seconds`);
      console.log(`[DETECTOR] Execution provider: WASM (CPU, multi-threaded)`);
      console.log(`[DETECTOR] Threads: 4 inter-op + 4 intra-op`);
      console.log(`[DETECTOR] SIMD: enabled, Proxy: disabled`);
      console.log(`[DETECTOR] Graph optimization: all`);
      console.log('[DETECTOR] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    } catch (error) {
      const endTime = performance.now();
      const attemptTime = ((endTime - startTime) / 1000).toFixed(2);
      console.error('[DETECTOR] ✗ Failed to load RT-DETR model');
      console.error(`[DETECTOR] Attempt duration: ${attemptTime} seconds`);
      console.error('[DETECTOR] Error details:', error);
      console.error('[DETECTOR] Error stack:', error.stack);
      throw error;
    }
  }

  async detect(imageElement) {
    console.log('[DETECTOR] ━━━ Starting bubble detection ━━━');

    if (!this.modelLoaded) {
      console.log('[DETECTOR] Model not loaded, initializing first...');
      await this.initialize();
    }

    const detectStartTime = performance.now();

    try {
      // Preprocess image
      console.log('[DETECTOR] Step 1/3: Preprocessing image...');
      const preprocessStart = performance.now();
      const { tensor, originalSize } = await this.preprocessImage(imageElement);
      const preprocessEnd = performance.now();
      console.log(`[DETECTOR] ✓ Preprocessing complete (${(preprocessEnd - preprocessStart).toFixed(0)}ms)`);
      console.log(`[DETECTOR] Original image size: ${originalSize.width}x${originalSize.height}px`);
      console.log(`[DETECTOR] Tensor shape: [${tensor.dims.join(', ')}]`);

      // Run inference
      console.log('[DETECTOR] Step 2/3: Running ONNX inference...');
      const inferenceStart = performance.now();

      // RT-DETR requires orig_target_sizes input: [batch_size, 2] containing [height, width]
      const origTargetSizes = new ort.Tensor(
        'int64',
        new BigInt64Array([BigInt(originalSize.height), BigInt(originalSize.width)]),
        [1, 2]
      );
      console.log(`[DETECTOR] Original target sizes: [${originalSize.height}, ${originalSize.width}]`);

      const feeds = {
        images: tensor,
        orig_target_sizes: origTargetSizes
      };

      const results = await this.session.run(feeds);
      const inferenceEnd = performance.now();
      const inferenceTime = inferenceEnd - inferenceStart;
      console.log(`[DETECTOR] ✓ Inference complete (${inferenceTime.toFixed(0)}ms)`);

      // Postprocess results
      console.log('[DETECTOR] Step 3/3: Postprocessing results...');
      const postprocessStart = performance.now();
      const detections = this.postprocess(results, originalSize);
      const postprocessEnd = performance.now();
      console.log(`[DETECTOR] ✓ Postprocessing complete (${(postprocessEnd - postprocessStart).toFixed(0)}ms)`);

      const totalTime = performance.now() - detectStartTime;
      console.log(`[DETECTOR] ━━━ Detection Summary ━━━`);
      console.log(`[DETECTOR] Total bubbles found: ${detections.length}`);
      console.log(`[DETECTOR] Total detection time: ${totalTime.toFixed(0)}ms (${(totalTime/1000).toFixed(2)}s)`);
      console.log(`[DETECTOR] ━━━━━━━━━━━━━━━━━━━━━━━━`);

      return detections;
    } catch (error) {
      console.error('[DETECTOR] ✗ Detection failed');
      console.error('[DETECTOR] Error:', error);
      throw error;
    }
  }

  async preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Get original dimensions
    const originalWidth = imageElement.naturalWidth || imageElement.width;
    const originalHeight = imageElement.naturalHeight || imageElement.height;

    // Resize to model input size (640x640)
    canvas.width = this.inputSize;
    canvas.height = this.inputSize;

    // Draw and resize image
    ctx.drawImage(imageElement, 0, 0, this.inputSize, this.inputSize);

    // Get image data
    const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
    const { data } = imageData;

    // Convert to RGB tensor [1, 3, 640, 640]
    const tensorData = new Float32Array(1 * 3 * this.inputSize * this.inputSize);

    // Normalize: (pixel / 255.0)
    for (let i = 0; i < this.inputSize * this.inputSize; i++) {
      // R channel
      tensorData[i] = data[i * 4] / 255.0;
      // G channel
      tensorData[this.inputSize * this.inputSize + i] = data[i * 4 + 1] / 255.0;
      // B channel
      tensorData[2 * this.inputSize * this.inputSize + i] = data[i * 4 + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', tensorData, [1, 3, this.inputSize, this.inputSize]);

    return {
      tensor,
      originalSize: { width: originalWidth, height: originalHeight },
    };
  }

  postprocess(results, originalSize) {
    console.log('[DETECTOR] Postprocessing: Extracting detection results...');
    console.log('[DETECTOR] Output keys:', Object.keys(results));

    // RT-DETR v2 outputs: labels [batch, num_queries], boxes [batch, num_queries, 4], scores [batch, num_queries]
    // Extract the tensors
    const labelsOutput = results.labels || results.pred_logits;
    const boxesOutput = results.boxes || results.pred_boxes;
    const scoresOutput = results.scores;

    if (!labelsOutput || !boxesOutput || !scoresOutput) {
      console.error('[DETECTOR] Missing required outputs. Available:', Object.keys(results));
      throw new Error('Model output format unexpected. Check model outputs.');
    }

    const labels = labelsOutput.data;
    const boxes = boxesOutput.data; // [batch * num_queries * 4]
    const scores = scoresOutput.data;

    console.log(`[DETECTOR] Raw detections: ${scores.length} candidates`);
    console.log(`[DETECTOR] Labels shape:`, labelsOutput.dims);
    console.log(`[DETECTOR] Boxes shape:`, boxesOutput.dims);
    console.log(`[DETECTOR] Scores shape:`, scoresOutput.dims);

    const detections = [];

    // RT-DETR with orig_target_sizes returns boxes ALREADY in original image space!
    // Evidence: raw box x2=673.84 > 640 (not in 640x640) and > 546 (can extend beyond bounds)
    console.log(`[DETECTOR] Boxes are in original image space: ${originalSize.width}x${originalSize.height}`);
    console.log(`[DETECTOR] NO SCALING needed (orig_target_sizes handles it)`);

    // Find max score to see what we're getting
    let maxScore = -Infinity;
    let maxScoreIndex = -1;
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIndex = i;
      }
    }
    console.log(`[DETECTOR] Max score in results: ${maxScore.toFixed(4)} at index ${maxScoreIndex}`);
    if (maxScoreIndex >= 0) {
      console.log(`[DETECTOR] Max score label: ${labels[maxScoreIndex]}`);
    }

    // Show score distribution
    const scoreRanges = { '0.0-0.1': 0, '0.1-0.2': 0, '0.2-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-1.0': 0 };
    for (let i = 0; i < scores.length; i++) {
      const s = scores[i];
      if (s < 0.1) scoreRanges['0.0-0.1']++;
      else if (s < 0.2) scoreRanges['0.1-0.2']++;
      else if (s < 0.3) scoreRanges['0.2-0.3']++;
      else if (s < 0.5) scoreRanges['0.3-0.5']++;
      else if (s < 0.7) scoreRanges['0.5-0.7']++;
      else scoreRanges['0.7-1.0']++;
    }
    console.log('[DETECTOR] Score distribution:', scoreRanges);

    // Filter by confidence threshold
    const confidenceThreshold = 0.3;
    console.log(`[DETECTOR] Confidence threshold: ${confidenceThreshold}`);

    let filteredCount = 0;
    let labelCounts = {};
    for (let i = 0; i < scores.length; i++) {
      const label = labels ? labels[i] : 0;
      labelCounts[label] = (labelCounts[label] || 0) + 1;

      if (scores[i] >= confidenceThreshold) {
        const boxStart = i * 4;

        // Boxes are already in original image space - just round and clamp
        const x1 = Math.max(0, Math.round(boxes[boxStart]));
        const y1 = Math.max(0, Math.round(boxes[boxStart + 1]));
        const x2 = Math.min(Math.round(boxes[boxStart + 2]), originalSize.width);
        const y2 = Math.min(Math.round(boxes[boxStart + 3]), originalSize.height);

        // Skip invalid boxes
        if (x2 <= x1 || y2 <= y1) {
          continue;
        }

        // Convert label to Number (handles BigInt from Int64Array)
        const labelNum = Number(label);

        detections.push({
          box: [x1, y1, x2, y2],
          score: scores[i],
          label: labelNum,
        });
        filteredCount++;

        // Log first few detections
        if (filteredCount <= 5) {
          const rawBox = [boxes[boxStart], boxes[boxStart + 1], boxes[boxStart + 2], boxes[boxStart + 3]];
          console.log(`[DETECTOR] Detection ${filteredCount}:`);
          console.log(`[DETECTOR]   Label: ${labelNum}, Score: ${scores[i].toFixed(4)}`);
          console.log(`[DETECTOR]   Raw box (original space): [${rawBox.map(v => v.toFixed(2)).join(', ')}]`);
          console.log(`[DETECTOR]   Clamped box: [${x1}, ${y1}, ${x2}, ${y2}]`);
        }
      }
    }

    console.log('[DETECTOR] Label distribution:', labelCounts);

    console.log(`[DETECTOR] Filtered to ${filteredCount} detections (score >= ${confidenceThreshold})`);
    console.log(`[DETECTOR] Detection types: bubbles and text_bubbles`);

    // Group text with bubbles
    console.log('[DETECTOR] Grouping text regions with speech bubbles...');
    const grouped = this.groupTextWithBubbles(detections);
    console.log(`[DETECTOR] Final grouped detections: ${grouped.length}`);

    return grouped;
  }

  groupTextWithBubbles(detections) {
    // Label mapping: 0 = bubble, 1 = text_bubble (check your model's label map)
    // Changed from string comparison to numeric comparison

    // Debug: log first detection's label type
    if (detections.length > 0) {
      console.log(`[DETECTOR] Sample detection label: ${detections[0].label} (type: ${typeof detections[0].label})`);
    }

    const bubbles = detections.filter(d => d.label === 0);
    const texts = detections.filter(d => d.label === 1);

    console.log(`[DETECTOR] Found ${bubbles.length} bubbles (label=0) and ${texts.length} text regions (label=1)`);

    const grouped = [];

    // If we have text regions, pair them with bubbles
    for (const textDet of texts) {
      // Find overlapping bubble
      let bestBubble = null;
      let bestIoU = 0;

      for (const bubbleDet of bubbles) {
        const iou = this.calculateIoU(textDet.box, bubbleDet.box);
        if (iou > bestIoU) {
          bestIoU = iou;
          bestBubble = bubbleDet;
        }
      }

      grouped.push({
        textBox: textDet.box,
        bubbleBox: bestBubble ? bestBubble.box : textDet.box,
        score: textDet.score,
      });
    }

    // If no text regions detected but we have bubbles, use bubbles directly
    if (grouped.length === 0 && bubbles.length > 0) {
      console.log('[DETECTOR] No text regions found, using bubble regions directly');
      for (const bubbleDet of bubbles) {
        grouped.push({
          textBox: bubbleDet.box,
          bubbleBox: bubbleDet.box,
          score: bubbleDet.score,
        });
      }
    }

    return grouped;
  }

  calculateIoU(box1, box2) {
    const [x1_1, y1_1, x2_1, y2_1] = box1;
    const [x1_2, y1_2, x2_2, y2_2] = box2;

    const xi1 = Math.max(x1_1, x1_2);
    const yi1 = Math.max(y1_1, y1_2);
    const xi2 = Math.min(x2_1, x2_2);
    const yi2 = Math.min(y2_1, y2_2);

    const interArea = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);

    const box1Area = (x2_1 - x1_1) * (y2_1 - y1_1);
    const box2Area = (x2_2 - x1_2) * (y2_2 - y1_2);

    const unionArea = box1Area + box2Area - interArea;

    return unionArea > 0 ? interArea / unionArea : 0;
  }
}

// Export for use in content script
window.BubbleDetector = BubbleDetector;