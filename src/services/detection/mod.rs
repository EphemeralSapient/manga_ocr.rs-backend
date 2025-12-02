use crate::core::config::Config;
use crate::core::types::BubbleDetection;
use crate::services::onnx_builder::{DynamicSessionPool, build_session_with_acceleration, try_forced_backend};
use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array2, Array4};
use ort::session::Session;
use ort::value::Value;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::{info, debug, trace};

// Embed the ONNX model at compile time (INT8 quantized, ~42MB)
// If LFS not checked out, this will be a small stub - load from runtime path instead
static MODEL_BYTES: &[u8] = include_bytes!("../../../models/detector.onnx");

/// Load model bytes from embedded or runtime path
fn load_model_bytes(config: &Config) -> Result<Vec<u8>> {
    // Check if embedded model is real (>10KB means it's not an LFS stub)
    if MODEL_BYTES.len() > 10_000 {
        debug!("Using embedded detector model ({} MB)", MODEL_BYTES.len() as f64 / 1_048_576.0);
        Ok(MODEL_BYTES.to_vec())
    } else {
        // Load from runtime path
        let path = &config.detection.detector_model_path;
        debug!("Loading detector model from: {}", path);
        std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("Failed to load detector model from {}: {}", path, e))
    }
}

/// Detection service with single session (detector.onnx is 160MB)
pub struct DetectionService {
    session_pool: Arc<DynamicSessionPool>,
    config: Arc<Config>,
    device_type: String,
    max_sessions: usize,
}

impl DetectionService {
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        Self::new_with_limit(config, None).await
    }

    /// Create detection service (single session only - 160MB model)
    pub async fn new_with_limit(config: Arc<Config>, _session_limit: Option<usize>) -> Result<Self> {
        // detector.onnx is 160MB - use only 1 session to avoid RAM bloat
        let max_sessions = 1;

        info!("🚀 Initializing detection service (single session - 160MB model)");

        // Load model bytes (160MB)
        let model_bytes = load_model_bytes(&config)?;
        info!("Loaded detector model ({:.1} MB)", model_bytes.len() as f64 / 1_048_576.0);

        // Create single session
        let (device_type, first_session) = Self::create_session_with_acceleration(&config, &model_bytes)?;

        // Create session pool with max 1 session
        let session_pool = DynamicSessionPool::new(max_sessions);
        session_pool.add_session(first_session);

        let session_pool = Arc::new(session_pool);

        let service = Self {
            session_pool,
            config,
            device_type: device_type.clone(),
            max_sessions,
        };

        // Warmup inference to trigger JIT compilation
        info!("Running warmup inference for detection...");
        let warmup_start = std::time::Instant::now();
        service.warmup().await?;
        info!("✓ Detection warmup completed in {:.2}ms", warmup_start.elapsed().as_secs_f64() * 1000.0);

        info!("✓ Detection: {} (single session, ~160 MB)", device_type);

        Ok(service)
    }

    /// Run warmup inference to trigger JIT compilation and memory allocation
    async fn warmup(&self) -> Result<()> {
        let target_size = self.config.target_size();

        // Create a small dummy image (just needs to match expected dimensions)
        let dummy_img = image::DynamicImage::new_rgb8(target_size, target_size);

        // Run inference to trigger optimization
        let (preprocessed, original_size) = self.preprocess_image(&dummy_img, None);

        let images_value = Value::from_array(preprocessed)?;
        let sizes_value = Value::from_array(original_size)?;

        // Acquire session and run dummy inference
        // CRITICAL: Must release session even on error to avoid deadlock
        let mut session = self.session_pool.acquire();
        let result: Result<()> = (|| {
            let _outputs = session.run(ort::inputs![
                "images" => images_value,
                "orig_target_sizes" => sizes_value
            ])?;
            Ok(())
        })();
        
        // Always release session back to pool (even on error)
        self.session_pool.release(session);
        
        // Now propagate any error
        result
    }

    /// No-op: detector.onnx uses single session to avoid 160MB RAM bloat per session
    #[inline]
    fn expand_if_needed(&self) {
        // Intentionally disabled - detector.onnx is 160MB, we use single session only
    }

    /// Create a session with hardware acceleration using pre-loaded model bytes
    /// Delegates to the unified onnx_builder module
    fn create_session_with_acceleration(config: &Config, model_bytes: &[u8]) -> Result<(String, Session)> {
        let model_size_mb = model_bytes.len() as f32 / 1_048_576.0;

        // Check if a specific backend is forced via config
        if let Some(ref backend) = config.detection.inference_backend {
            if backend.to_uppercase() != "AUTO" {
                return try_forced_backend(backend, model_bytes, "detector", model_size_mb);
            }
        }

        // Use automatic detection from unified builder
        build_session_with_acceleration(model_bytes, "detector", model_size_mb)
    }

    #[allow(dead_code)]
    pub fn device_type(&self) -> &str {
        &self.device_type
    }

    /// Check if using DirectML backend
    pub fn is_directml(&self) -> bool {
        self.device_type.contains("DirectML")
    }

    /// Keep session alive - detector uses single 160MB session that should persist across requests
    /// (Previously drained sessions which caused deadlock on second request)
    pub fn cleanup_sessions(&self) {
        // Keep session alive - draining would cause next request to block forever
        // since we only have 1 session and no expansion
        debug!("Detection: Keeping single session alive (160MB model)");
    }

    fn preprocess_image(&self, img: &DynamicImage, target_size_override: Option<u32>) -> (Array4<f32>, Array2<i64>) {
        // Determine target size: use override if provided, otherwise use config default
        let target_size = match target_size_override {
            Some(0) => {
                // 0 means use source resolution (no resizing)
                debug!("Using source image resolution: {}x{} (no resizing)", img.width(), img.height());
                // For square target, use the maximum dimension to avoid distortion
                img.width().max(img.height())
            }
            Some(size) => {
                debug!("Using override target size: {}", size);
                size
            }
            None => {
                debug!("Using config default target size: {}", self.config.target_size());
                self.config.target_size()
            }
        };

        trace!("Preprocessing image: {}x{} → {}x{}",
            img.width(), img.height(),
            target_size, target_size);

        let original_size = Array2::from_shape_vec(
            (1, 2),
            vec![img.width() as i64, img.height() as i64],
        )
        .unwrap();

        let resized = img.resize_exact(
            target_size,
            target_size,
            image::imageops::FilterType::Triangle,
        );
        let rgb_img = resized.to_rgb8();

        let target = target_size as usize;
        let mut array = Array4::<f32>::zeros((1, 3, target, target));

        for y in 0..target {
            for x in 0..target {
                let pixel = rgb_img.get_pixel(x as u32, y as u32);
                array[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
                array[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
                array[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }

        debug!("✓ Image preprocessed: array shape=[1, 3, {}, {}]", target, target);
        (array, original_size)
    }

    fn calculate_iou(box1: &[i32; 4], box2: &[i32; 4]) -> f32 {
        let x1 = box1[0].max(box2[0]);
        let y1 = box1[1].max(box2[1]);
        let x2 = box1[2].min(box2[2]);
        let y2 = box1[3].min(box2[3]);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = ((x2 - x1) * (y2 - y1)) as f32;
        let area1 = ((box1[2] - box1[0]) * (box1[3] - box1[1])) as f32;
        let area2 = ((box2[2] - box2[0]) * (box2[3] - box2[1])) as f32;
        let union = area1 + area2 - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    fn nms(&self, detections: Vec<BubbleDetection>) -> Vec<BubbleDetection> {
        if detections.is_empty() {
            debug!("NMS: No detections to filter");
            return vec![];
        }

        let iou_threshold = self.config.iou_threshold();
        trace!("NMS: Processing {} detections with IoU threshold={}",
            detections.len(), iou_threshold);

        let mut sorted = detections;
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal));

        let mut keep = Vec::new();
        let mut suppressed = vec![false; sorted.len()];
        let mut suppressed_count = 0;

        for i in 0..sorted.len() {
            if suppressed[i] {
                continue;
            }

            keep.push(sorted[i].clone());

            for j in (i + 1)..sorted.len() {
                if !suppressed[j] {
                    let iou = Self::calculate_iou(&sorted[i].bbox, &sorted[j].bbox);
                    if iou > iou_threshold {
                        suppressed[j] = true;
                        suppressed_count += 1;
                        trace!("NMS: Suppressed detection {} (IoU={:.3} with detection {})",
                            j, iou, i);
                    }
                }
            }
        }

        debug!("NMS: Kept {}/{} detections (suppressed {})",
            keep.len(), sorted.len(), suppressed_count);
        keep
    }

    /// Detect ALL regions (all labels) in a single inference pass
    /// This is MUCH more efficient than calling detect_with_label 3 times
    ///
    /// # Arguments
    /// * `target_size_override` - Optional target size override. If None, uses config default.
    ///                            If Some(0), uses source image resolution (no resizing).
    pub async fn detect_all_labels(
        &self,
        img: &DynamicImage,
        page_index: usize,
        target_size_override: Option<u32>,
    ) -> Result<(Vec<BubbleDetection>, Vec<BubbleDetection>, Vec<BubbleDetection>)> {
        debug!("🔍 [DETECTION] Starting detection for page {}", page_index);
        let detection_start = std::time::Instant::now();

        let (preprocessed, original_size) = self.preprocess_image(img, target_size_override);

        let images_value = Value::from_array(preprocessed)?;
        let sizes_value = Value::from_array(original_size)?;

        debug!("Running ONNX inference on {}...", self.device_type);
        let inference_start = std::time::Instant::now();

        // Expand pool if needed before acquiring
        self.expand_if_needed();

        // Acquire session from pool and run inference
        // CRITICAL: Must release session even on error to avoid deadlock
        let mut session = self.session_pool.acquire();
        let result = (|| -> Result<_> {
            let outputs = session.run(ort::inputs![
                "images" => images_value,
                "orig_target_sizes" => sizes_value
            ])?;

            let inference_time = inference_start.elapsed();
            debug!("✓ Inference completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);

            // Extract tensor references
            let (labels_shape, labels_data) = outputs["labels"].try_extract_tensor::<i64>()?;
            let (_boxes_shape, boxes_data) = outputs["boxes"].try_extract_tensor::<f32>()?;
            let (_scores_shape, scores_data) = outputs["scores"].try_extract_tensor::<f32>()?;

            let num_detections = labels_shape[1] as usize;
            trace!("Raw detections from model: {}", num_detections);

            let confidence_threshold = self.config.confidence_threshold();
            let mut label_0_detections = Vec::new();
            let mut label_1_detections = Vec::new();
            let mut label_2_detections = Vec::new();

            // Process tensors in-place - only clone filtered detections
            for i in 0..num_detections {
                let label = labels_data[i];
                let score = scores_data[i];

                // Early filter - skip low-confidence detections
                if score < confidence_threshold {
                    continue;
                }

                let bbox = [
                    boxes_data[i * 4] as i32,
                    boxes_data[i * 4 + 1] as i32,
                    boxes_data[i * 4 + 2] as i32,
                    boxes_data[i * 4 + 3] as i32,
                ];

                let detection = BubbleDetection {
                    bbox,
                    confidence: score,
                    page_index,
                    bubble_index: i,
                    text_regions: vec![],
                };

                match label {
                    0 => label_0_detections.push(detection),
                    1 => label_1_detections.push(detection),
                    2 => label_2_detections.push(detection),
                    _ => {}
                }
            }

            Ok((label_0_detections, label_1_detections, label_2_detections))
        })();
        
        // Always release session back to pool (even on error)
        self.session_pool.release(session);
        
        let (label_0_detections, label_1_detections, label_2_detections) = result?;

        let total_time = detection_start.elapsed();
        debug!(
            "✓ Detection completed in {:.2}ms: {} L0, {} L1, {} L2",
            total_time.as_secs_f64() * 1000.0,
            label_0_detections.len(),
            label_1_detections.len(),
            label_2_detections.len()
        );

        Ok((label_0_detections, label_1_detections, label_2_detections))
    }

    /// Detect ALL regions for multiple images in a single batched inference pass
    /// This is more efficient than calling detect_all_labels multiple times
    ///
    /// # Arguments
    /// * `images` - Vector of (image, page_index) tuples
    /// * `target_size_override` - Optional target size override. If None, uses config default.
    ///                            If Some(0), uses max dimension from batch (no fixed resizing).
    ///
    /// # Returns
    /// Vector of (label_0, label_1, label_2) detection tuples, one per input image
    pub async fn detect_all_labels_batch(
        &self,
        images: &[(&DynamicImage, usize)],
        target_size_override: Option<u32>,
    ) -> Result<Vec<(Vec<BubbleDetection>, Vec<BubbleDetection>, Vec<BubbleDetection>)>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = images.len();
        debug!("🔍 [BATCH] Starting batched detection for {} images", batch_size);
        let detection_start = std::time::Instant::now();

        // Determine target size
        let target_size = match target_size_override {
            Some(0) => {
                let max_dim = images.iter()
                    .map(|(img, _)| img.width().max(img.height()))
                    .max()
                    .unwrap_or(self.config.target_size());
                debug!("Using source resolution (batch max dimension): {}", max_dim);
                max_dim
            }
            Some(size) => {
                debug!("Using override target size for batch: {}", size);
                size
            }
            None => {
                debug!("Using config default target size: {}", self.config.target_size());
                self.config.target_size()
            }
        };
        let target = target_size as usize;

        // Preprocess all images in parallel and stack into batch tensor
        let mut batch_array = Array4::<f32>::zeros((batch_size, 3, target, target));
        let mut sizes_array = Array2::<i64>::zeros((batch_size, 2));

        // Parallel preprocessing - each image is independent
        let preprocessed: Vec<_> = images
            .par_iter()
            .map(|(img, _page_index)| {
                let resized = img.resize_exact(
                    target_size,
                    target_size,
                    image::imageops::FilterType::Triangle,
                );
                let rgb_img = resized.to_rgb8();

                // Create per-image buffer
                let mut img_data = vec![0.0f32; 3 * target * target];
                for y in 0..target {
                    for x in 0..target {
                        let pixel = rgb_img.get_pixel(x as u32, y as u32);
                        let base = y * target + x;
                        img_data[base] = pixel[0] as f32 / 255.0;
                        img_data[target * target + base] = pixel[1] as f32 / 255.0;
                        img_data[2 * target * target + base] = pixel[2] as f32 / 255.0;
                    }
                }
                (img.width() as i64, img.height() as i64, img_data)
            })
            .collect();

        // Copy preprocessed data into batch tensor (sequential but fast memcpy)
        for (i, (w, h, img_data)) in preprocessed.into_iter().enumerate() {
            sizes_array[[i, 0]] = w;
            sizes_array[[i, 1]] = h;

            for c in 0..3 {
                for y in 0..target {
                    for x in 0..target {
                        batch_array[[i, c, y, x]] = img_data[c * target * target + y * target + x];
                    }
                }
            }
        }

        debug!("✓ Batch preprocessed (parallel): {} images into [{}, 3, {}, {}]", batch_size, batch_size, target, target);

        let images_value = Value::from_array(batch_array)?;
        let sizes_value = Value::from_array(sizes_array)?;

        debug!("Running batched ONNX inference on {}...", self.device_type);
        let inference_start = std::time::Instant::now();

        // Expand pool if needed
        self.expand_if_needed();

        // Run batched inference
        // CRITICAL: Must release session even on error to avoid deadlock
        let mut session = self.session_pool.acquire();
        let result = (|| -> Result<_> {
            let outputs = session.run(ort::inputs![
                "images" => images_value,
                "orig_target_sizes" => sizes_value
            ])?;

            let inference_time = inference_start.elapsed();
            debug!("✓ Batch inference completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);

            // Extract tensor data
            let (labels_shape, labels_data) = outputs["labels"].try_extract_tensor::<i64>()?;
            let (boxes_shape, boxes_data) = outputs["boxes"].try_extract_tensor::<f32>()?;
            let (_scores_shape, scores_data) = outputs["scores"].try_extract_tensor::<f32>()?;

            let num_detections = labels_shape[1] as usize;
            let confidence_threshold = self.config.confidence_threshold();

            let mut all_results = Vec::with_capacity(batch_size);

            for batch_idx in 0..batch_size {
                let page_index = images[batch_idx].1;
                let mut label_0 = Vec::new();
                let mut label_1 = Vec::new();
                let mut label_2 = Vec::new();

                for det_idx in 0..num_detections {
                    let flat_idx = batch_idx * num_detections + det_idx;
                    let label = labels_data[flat_idx];
                    let score = scores_data[flat_idx];

                    if score < confidence_threshold {
                        continue;
                    }

                    let box_base = if boxes_shape.len() == 3 {
                        (batch_idx * num_detections + det_idx) * 4
                    } else {
                        flat_idx * 4
                    };

                    let bbox = [
                        boxes_data[box_base] as i32,
                        boxes_data[box_base + 1] as i32,
                        boxes_data[box_base + 2] as i32,
                        boxes_data[box_base + 3] as i32,
                    ];

                    let detection = BubbleDetection {
                        bbox,
                        confidence: score,
                        page_index,
                        bubble_index: det_idx,
                        text_regions: vec![],
                    };

                    match label {
                        0 => label_0.push(detection),
                        1 => label_1.push(detection),
                        2 => label_2.push(detection),
                        _ => {}
                    }
                }

                all_results.push((label_0, label_1, label_2));
            }

            Ok(all_results)
        })();
        
        // Always release session back to pool (even on error)
        self.session_pool.release(session);
        
        let results = result?;

        let total_time = detection_start.elapsed();
        debug!(
            "✓ Batch detection completed in {:.2}ms for {} images ({:.2}ms/image)",
            total_time.as_secs_f64() * 1000.0,
            batch_size,
            total_time.as_secs_f64() * 1000.0 / batch_size as f64
        );

        Ok(results)
    }

    /// Detect regions with a specific label
    /// NOTE: This is kept for backward compatibility, but detect_all_labels() is more efficient
    pub async fn detect_with_label(
        &self,
        img: &DynamicImage,
        page_index: usize,
        target_label: i64,
    ) -> Result<Vec<BubbleDetection>> {
        let label_name = match target_label {
            0 => "bubble",
            1 => "text_bubble",
            2 => "text_free",
            _ => "unknown",
        };
        debug!("🔍 [DETECTION] Starting detection for label {} ({}) on page {}",
            target_label, label_name, page_index);
        let detection_start = std::time::Instant::now();

        let (preprocessed, original_size) = self.preprocess_image(img, None);

        let images_value = Value::from_array(preprocessed)?;
        let sizes_value = Value::from_array(original_size)?;

        debug!("Running ONNX inference on {}...", self.device_type);
        let inference_start = std::time::Instant::now();

        // Expand pool if needed
        self.expand_if_needed();

        // Acquire session from pool and run inference
        // CRITICAL: Must release session even on error to avoid deadlock
        let mut session = self.session_pool.acquire();
        let result = (|| -> Result<_> {
            let outputs = session.run(ort::inputs![
                "images" => images_value,
                "orig_target_sizes" => sizes_value
            ])?;

            let inference_time = inference_start.elapsed();
            debug!("✓ Inference completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);

            // Extract tensor references
            let (labels_shape, labels_data) = outputs["labels"].try_extract_tensor::<i64>()?;
            let (_boxes_shape, boxes_data) = outputs["boxes"].try_extract_tensor::<f32>()?;
            let (_scores_shape, scores_data) = outputs["scores"].try_extract_tensor::<f32>()?;

            let num_detections = labels_shape[1] as usize;
            trace!("Raw detections from model: {}", num_detections);

            let confidence_threshold = self.config.confidence_threshold();
            let mut text_detections = Vec::new();

            for i in 0..num_detections {
                let label = labels_data[i];
                let score = scores_data[i];

                if label != target_label || score < confidence_threshold {
                    continue;
                }

                let bbox = [
                    boxes_data[i * 4] as i32,
                    boxes_data[i * 4 + 1] as i32,
                    boxes_data[i * 4 + 2] as i32,
                    boxes_data[i * 4 + 3] as i32,
                ];

                trace!("Detection {}: bbox=[{}, {}, {}, {}], conf={:.3}",
                    i, bbox[0], bbox[1], bbox[2], bbox[3], score);

                text_detections.push(BubbleDetection {
                    bbox,
                    confidence: score,
                    page_index,
                    bubble_index: i,
                    text_regions: Vec::new(),
                });
            }

            Ok(text_detections)
        })();
        
        // Always release session back to pool (even on error)
        self.session_pool.release(session);
        
        let text_detections = result?;

        debug!("Filtered {} detections above confidence threshold {:.2}",
            text_detections.len(), self.config.confidence_threshold());

        let filtered = self.nms(text_detections);

        // Sort by reading order (top-to-bottom, left-to-right)
        let mut sorted = filtered;
        sorted.sort_by_key(|d| (d.bbox[1], d.bbox[0]));

        let total_time = detection_start.elapsed();
        debug!("✅ [DETECTION] Completed for page {}: {} {} detected in {:.2}ms",
            page_index, sorted.len(), label_name, total_time.as_secs_f64() * 1000.0);

        Ok(sorted)
    }
}

