use crate::core::config::Config;
use crate::core::types::BubbleDetection;
use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array2, Array4};
use ort::execution_providers::CPUExecutionProvider;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;

#[cfg(feature = "openvino")]
use ort::execution_providers::OpenVINOExecutionProvider;

#[cfg(all(target_os = "windows", feature = "directml"))]
use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(all(target_os = "macos", feature = "coreml"))]
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::sync::Arc;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tracing::{info, debug, trace};

// Embed the ONNX model at compile time (INT8 quantized, ~42MB)
static MODEL_BYTES: &[u8] = include_bytes!("../../../models/detector.onnx");

/// Session pool for concurrent inference
pub struct SessionPool {
    sender: Sender<Session>,
    receiver: Arc<tokio::sync::Mutex<Receiver<Session>>>,
}

impl SessionPool {
    async fn acquire(&self) -> Session {
        let mut receiver = self.receiver.lock().await;
        receiver.recv().await.expect("Session pool exhausted")
    }

    async fn release(&self, session: Session) {
        self.sender.send(session).await.expect("Failed to return session to pool");
    }
}

pub struct DetectionService {
    session_pool: Arc<SessionPool>,
    config: Arc<Config>,
    device_type: String,
}

impl DetectionService {
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        // Determine pool size: use num_cpus or MAX_CONCURRENT_BATCHES (whichever is smaller)
        let pool_size = std::cmp::min(num_cpus::get(), config.max_concurrent_batches());
        debug!("Creating detection session pool with {} sessions", pool_size);

        // Create first session to determine device type
        let (device_type, first_session) = Self::initialize_with_acceleration(&config)?;

        // Create channel for session pool
        let (sender, receiver) = channel(pool_size);

        // Send first session to pool
        sender.send(first_session).await
            .map_err(|_| anyhow::anyhow!("Failed to initialize session pool"))?;

        // Create remaining sessions IN PARALLEL for faster startup
        if pool_size > 1 {
            let mut tasks = Vec::new();

            for i in 1..pool_size {
                let config_clone = Arc::clone(&config);
                let task = tokio::task::spawn_blocking(move || {
                    debug!("Creating session {} of {}", i + 1, pool_size);
                    Self::initialize_with_acceleration(&config_clone)
                });
                tasks.push(task);
            }

            // Wait for all sessions to be created
            for task in tasks {
                let (_, session) = task.await
                    .map_err(|e| anyhow::anyhow!("Failed to spawn session creation: {}", e))??;
                sender.send(session).await
                    .map_err(|_| anyhow::anyhow!("Failed to add session to pool"))?;
            }
        }

        let session_pool = Arc::new(SessionPool {
            sender,
            receiver: Arc::new(tokio::sync::Mutex::new(receiver)),
        });

        info!("✓ Detection: {} ({} sessions)", device_type, pool_size);

        Ok(Self {
            session_pool,
            config,
            device_type,
        })
    }

    fn try_forced_backend(backend: &str) -> Result<(String, Session)> {
        match backend {
            #[cfg(feature = "tensorrt")]
            "TENSORRT" => {
                info!("Forcing TensorRT backend...");
                let session = Session::builder()?
                    .with_execution_providers([TensorRTExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MODEL_BYTES)?;
                info!("✓ Successfully initialized TensorRT backend");
                Ok(("TensorRT (forced)".to_string(), session))
            }
            #[cfg(not(feature = "tensorrt"))]
            "TENSORRT" => {
                anyhow::bail!("TensorRT backend not available. Rebuild with: cargo build --features tensorrt")
            }

            #[cfg(feature = "cuda")]
            "CUDA" => {
                info!("Forcing CUDA backend...");
                let session = Session::builder()?
                    .with_execution_providers([CUDAExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MODEL_BYTES)?;
                info!("✓ Successfully initialized CUDA backend");
                Ok(("CUDA (forced)".to_string(), session))
            }
            #[cfg(not(feature = "cuda"))]
            "CUDA" => {
                anyhow::bail!("CUDA backend not available. Rebuild with: cargo build --features cuda")
            }

            #[cfg(feature = "openvino")]
            "OPENVINO" => {
                info!("Forcing OpenVINO backend...");
                let session = Session::builder()?
                    .with_execution_providers([
                        OpenVINOExecutionProvider::default()
                            .with_device_type("CPU")
                            .build()
                    ])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MODEL_BYTES)?;
                info!("✓ Successfully initialized OpenVINO backend");
                Ok(("OpenVINO-CPU (forced)".to_string(), session))
            }
            #[cfg(not(feature = "openvino"))]
            "OPENVINO" => {
                anyhow::bail!("OpenVINO backend not available. Rebuild with: cargo build --features openvino")
            }

            #[cfg(all(target_os = "windows", feature = "directml"))]
            "DIRECTML" => {
                info!("Forcing DirectML backend...");
                let session = Session::builder()?
                    .with_execution_providers([DirectMLExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MODEL_BYTES)?;
                info!("✓ Successfully initialized DirectML backend");
                Ok(("DirectML (forced)".to_string(), session))
            }
            #[cfg(not(all(target_os = "windows", feature = "directml")))]
            "DIRECTML" => {
                anyhow::bail!("DirectML backend not available. Rebuild with: cargo build --features directml (Windows only)")
            }

            #[cfg(all(target_os = "macos", feature = "coreml"))]
            "COREML" => {
                info!("Forcing CoreML backend...");
                let session = Session::builder()?
                    .with_execution_providers([CoreMLExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MODEL_BYTES)?;
                info!("✓ Successfully initialized CoreML backend");
                Ok(("CoreML (forced)".to_string(), session))
            }
            #[cfg(not(all(target_os = "macos", feature = "coreml")))]
            "COREML" => {
                anyhow::bail!("CoreML backend not available. Rebuild with: cargo build --features coreml (macOS only)")
            }

            "CPU" => {
                info!("Forcing CPU backend...");
                let session = Session::builder()?
                    .with_execution_providers([CPUExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MODEL_BYTES)?;
                info!("✓ Successfully initialized CPU backend");
                Ok(("CPU (forced)".to_string(), session))
            }
            _ => {
                anyhow::bail!(
                    "Unknown inference backend '{}'. \
                    Valid options: TENSORRT, CUDA, OPENVINO, DIRECTML, COREML, CPU, AUTO",
                    backend
                )
            }
        }
    }

    fn initialize_with_acceleration(config: &Config) -> Result<(String, Session)> {
        info!("Loading embedded ONNX model ({} bytes)...", MODEL_BYTES.len());

        // Check if a specific backend is forced via config
        if let Some(ref backend) = config.detection.inference_backend {
            match backend.as_str() {
                "AUTO" => {
                    info!("INFERENCE_BACKEND=AUTO, using automatic detection");
                }
                forced_backend => {
                    info!("INFERENCE_BACKEND={}, forcing specific backend", forced_backend);
                    return Self::try_forced_backend(forced_backend);
                }
            }
        }

        // Try hardware acceleration in order of preference
        // Only attempt providers that are compiled in via Cargo features

        // Try TensorRT (if feature enabled)
        #[cfg(feature = "tensorrt")]
        {
            if let Ok(session) = Session::builder()
                .and_then(|b| b.with_execution_providers([TensorRTExecutionProvider::default().build()]))
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
                .and_then(|b| b.with_intra_threads(num_cpus::get()))
                .and_then(|b| b.commit_from_memory(MODEL_BYTES))
            {
                info!("✓ Using TensorRT acceleration");
                return Ok(("TensorRT".to_string(), session));
            }
        }

        // Try CUDA (if feature enabled)
        #[cfg(feature = "cuda")]
        {
            if let Ok(session) = Session::builder()
                .and_then(|b| b.with_execution_providers([CUDAExecutionProvider::default().build()]))
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
                .and_then(|b| b.with_intra_threads(num_cpus::get()))
                .and_then(|b| b.commit_from_memory(MODEL_BYTES))
            {
                info!("✓ Using CUDA acceleration");
                return Ok(("CUDA".to_string(), session));
            }
        }

        // Try CoreML (Apple Silicon, if feature enabled)
        #[cfg(all(target_os = "macos", feature = "coreml"))]
        {
            if let Ok(session) = Session::builder()
                .and_then(|b| b.with_execution_providers([CoreMLExecutionProvider::default().build()]))
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
                .and_then(|b| b.with_intra_threads(num_cpus::get()))
                .and_then(|b| b.commit_from_memory(MODEL_BYTES))
            {
                info!("✓ Using CoreML acceleration (Apple Neural Engine)");
                return Ok(("CoreML".to_string(), session));
            }
        }

        // Try DirectML (Windows, if feature enabled)
        #[cfg(all(target_os = "windows", feature = "directml"))]
        {
            if let Ok(session) = Session::builder()
                .and_then(|b| b.with_execution_providers([DirectMLExecutionProvider::default().build()]))
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
                .and_then(|b| b.with_intra_threads(num_cpus::get()))
                .and_then(|b| b.commit_from_memory(MODEL_BYTES))
            {
                info!("✓ Using DirectML acceleration");
                return Ok(("DirectML".to_string(), session));
            }
        }

        // Try OpenVINO (Intel CPU optimization, if feature enabled)
        #[cfg(feature = "openvino")]
        {
            if let Ok(session) = Session::builder()
                .and_then(|b| b.with_execution_providers([
                    OpenVINOExecutionProvider::default()
                        .with_device_type("CPU")
                        .build()
                ]))
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
                .and_then(|b| b.with_intra_threads(num_cpus::get()))
                .and_then(|b| b.commit_from_memory(MODEL_BYTES))
            {
                info!("✓ Using OpenVINO acceleration (Intel CPU optimizations)");
                return Ok(("OpenVINO-CPU".to_string(), session));
            }
        }

        // Final fallback: Pure CPU (no acceleration)
        let session = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_memory(MODEL_BYTES)?;

        info!("✓ Using CPU (no hardware acceleration)");
        Ok(("CPU".to_string(), session))
    }

    #[allow(dead_code)]
    pub fn device_type(&self) -> &str {
        &self.device_type
    }

    fn preprocess_image(&self, img: &DynamicImage) -> (Array4<f32>, Array2<i64>) {
        let target_size = self.config.target_size();
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
    pub async fn detect_all_labels(
        &self,
        img: &DynamicImage,
        page_index: usize,
    ) -> Result<(Vec<BubbleDetection>, Vec<BubbleDetection>, Vec<BubbleDetection>)> {
        debug!("🔍 [DETECTION] Starting detection for ALL labels on page {}", page_index);
        let detection_start = std::time::Instant::now();

        let (preprocessed, original_size) = self.preprocess_image(img);

        let images_value = Value::from_array(preprocessed)?;
        let sizes_value = Value::from_array(original_size)?;

        debug!("Running ONNX inference on {}...", self.device_type);
        let inference_start = std::time::Instant::now();

        // Acquire session from pool and run inference
        let (labels_shape_owned, labels_data, boxes_data, scores_data) = {
            let mut session = self.session_pool.acquire().await;
            let outputs = session.run(ort::inputs![
                "images" => images_value,
                "orig_target_sizes" => sizes_value
            ])?;

            // Extract and immediately clone all data while session is borrowed
            let (labels_shape, labels_data) = outputs["labels"].try_extract_tensor::<i64>()?;
            let labels_shape_owned = labels_shape.to_vec();
            let labels_data_owned = labels_data.to_vec();

            let (_boxes_shape, boxes_data) = outputs["boxes"].try_extract_tensor::<f32>()?;
            let boxes_data_owned = boxes_data.to_vec();

            let (_scores_shape, scores_data) = outputs["scores"].try_extract_tensor::<f32>()?;
            let scores_data_owned = scores_data.to_vec();

            // Drop outputs and return session to pool
            drop(outputs);
            self.session_pool.release(session).await;

            (labels_shape_owned, labels_data_owned, boxes_data_owned, scores_data_owned)
        };

        let inference_time = inference_start.elapsed();
        debug!("✓ Inference completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);

        let num_detections = labels_shape_owned[1] as usize;
        trace!("Raw detections from model: {}", num_detections);

        let confidence_threshold = self.config.confidence_threshold();
        let mut label_0_detections = Vec::new();
        let mut label_1_detections = Vec::new();
        let mut label_2_detections = Vec::new();

        // Split detections by label in a single pass
        for i in 0..num_detections {
            let label = labels_data[i];
            let score = scores_data[i];

            if score >= confidence_threshold {
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
                    text_regions: vec![], // Will be populated later in Phase1
                };

                match label {
                    0 => label_0_detections.push(detection),
                    1 => label_1_detections.push(detection),
                    2 => label_2_detections.push(detection),
                    _ => {} // Ignore unknown labels
                }
            }
        }

        let total_time = detection_start.elapsed();
        debug!(
            "✓ Detection completed in {:.2}ms: {} label 0, {} label 1, {} label 2",
            total_time.as_secs_f64() * 1000.0,
            label_0_detections.len(),
            label_1_detections.len(),
            label_2_detections.len()
        );

        Ok((label_0_detections, label_1_detections, label_2_detections))
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

        let (preprocessed, original_size) = self.preprocess_image(img);

        let images_value = Value::from_array(preprocessed)?;
        let sizes_value = Value::from_array(original_size)?;

        debug!("Running ONNX inference on {}...", self.device_type);
        let inference_start = std::time::Instant::now();

        // Acquire session from pool and run inference
        let (labels_shape_owned, labels_data, boxes_data, scores_data) = {
            let mut session = self.session_pool.acquire().await;
            let outputs = session.run(ort::inputs![
                "images" => images_value,
                "orig_target_sizes" => sizes_value
            ])?;

            // Extract and immediately clone all data while session is borrowed
            let (labels_shape, labels_data) = outputs["labels"].try_extract_tensor::<i64>()?;
            let labels_shape_owned = labels_shape.to_vec();
            let labels_data_owned = labels_data.to_vec();

            let (_boxes_shape, boxes_data) = outputs["boxes"].try_extract_tensor::<f32>()?;
            let boxes_data_owned = boxes_data.to_vec();

            let (_scores_shape, scores_data) = outputs["scores"].try_extract_tensor::<f32>()?;
            let scores_data_owned = scores_data.to_vec();

            // Drop outputs and return session to pool
            drop(outputs);
            self.session_pool.release(session).await;

            (labels_shape_owned, labels_data_owned, boxes_data_owned, scores_data_owned)
        };

        let inference_time = inference_start.elapsed();
        debug!("✓ Inference completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);

        let num_detections = labels_shape_owned[1] as usize;
        trace!("Raw detections from model: {}", num_detections);

        let confidence_threshold = self.config.confidence_threshold();
        let mut text_detections = Vec::new();

        for i in 0..num_detections {
            let label = labels_data[i];
            let score = scores_data[i];

            if label == target_label && score >= confidence_threshold {
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
                    text_regions: Vec::new(), // Will be filled later by detect_bubbles()
                });
            }
        }

        debug!("Filtered {} detections above confidence threshold {:.2}",
            text_detections.len(), confidence_threshold);

        let filtered = self.nms(text_detections);

        // Sort by reading order (top-to-bottom, left-to-right)
        let mut sorted = filtered;
        sorted.sort_by_key(|d| (d.bbox[1], d.bbox[0]));

        let total_time = detection_start.elapsed();
        debug!("✅ [DETECTION] Completed for page {}: {} {} detected in {:.2}ms",
            page_index, sorted.len(), label_name, total_time.as_secs_f64() * 1000.0);

        Ok(sorted)
    }

    /// Detect bubbles (label 0) with text regions (label 1) - main workflow method
    #[allow(dead_code)]
    pub async fn detect_bubbles(
        &self,
        img: &DynamicImage,
        page_index: usize,
    ) -> Result<Vec<BubbleDetection>> {
        debug!("🔍 Detecting bubbles (label 0) AND text regions (label 1)");

        // Step 1: Detect full bubbles (label 0)
        let mut bubbles = self.detect_with_label(img, page_index, 0).await?;

        // Step 2: Detect text regions inside bubbles (label 1)
        let text_regions = self.detect_with_label(img, page_index, 1).await?;

        debug!("Found {} bubbles and {} text regions", bubbles.len(), text_regions.len());

        // Step 3: Match text regions to their parent bubbles and refine them
        for bubble in &mut bubbles {
            let mut matched_regions = Vec::new();

            for text_region in &text_regions {
                // Check if text region is inside this bubble
                if Self::bbox_contains(&bubble.bbox, &text_region.bbox) {
                    // Refine the text region bbox to fit tighter around actual text
                    let refined_bbox = Self::refine_text_region_bbox(img, text_region.bbox);
                    matched_regions.push(refined_bbox);
                }
            }

            bubble.text_regions = matched_regions;
            debug!("Bubble at [{},{},{},{}] contains {} text regions",
                bubble.bbox[0], bubble.bbox[1], bubble.bbox[2], bubble.bbox[3],
                bubble.text_regions.len());
        }

        Ok(bubbles)
    }

    /// Refine text region bbox by analyzing actual pixel content
    /// Shrinks the bbox to fit tighter around actual text pixels
    #[allow(dead_code)]
    fn refine_text_region_bbox(img: &DynamicImage, bbox: [i32; 4]) -> [i32; 4] {
        let x1 = bbox[0].max(0) as u32;
        let y1 = bbox[1].max(0) as u32;
        let x2 = bbox[2].min(img.width() as i32) as u32;
        let y2 = bbox[3].min(img.height() as i32) as u32;

        if x2 <= x1 || y2 <= y1 {
            return bbox; // Invalid bbox, return as-is
        }

        // Convert to grayscale for text detection
        let gray_img = img.to_luma8();

        // Find the actual bounds of dark pixels (text)
        // Text is typically dark (low luminance values)
        // Lower threshold to avoid detecting halftone/screentone patterns
        let threshold = 120u8; // Pixels darker than this are considered text (black text only)

        let mut min_x = x2;
        let mut max_x = x1;
        let mut min_y = y2;
        let mut max_y = y1;
        let mut found_text = false;

        for y in y1..y2 {
            for x in x1..x2 {
                let pixel = gray_img.get_pixel(x, y);
                if pixel[0] < threshold {
                    // Found a dark pixel (likely text)
                    found_text = true;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        if !found_text {
            // No text found, return original bbox
            return bbox;
        }

        // Add small padding (5% of dimensions) for safety
        let width = (max_x - min_x) as f32;
        let height = (max_y - min_y) as f32;
        let padding_x = (width * 0.05).max(2.0) as i32;
        let padding_y = (height * 0.05).max(2.0) as i32;

        let refined_x1 = (min_x as i32 - padding_x).max(bbox[0]);
        let refined_y1 = (min_y as i32 - padding_y).max(bbox[1]);
        let refined_x2 = (max_x as i32 + padding_x + 1).min(bbox[2]);
        let refined_y2 = (max_y as i32 + padding_y + 1).min(bbox[3]);

        let original_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
        let refined_area = (refined_x2 - refined_x1) * (refined_y2 - refined_y1);
        let reduction_pct = (1.0 - refined_area as f32 / original_area as f32) * 100.0;

        debug!("Refined text bbox: [{},{},{},{}] → [{},{},{},{}] ({:.1}% reduction)",
            bbox[0], bbox[1], bbox[2], bbox[3],
            refined_x1, refined_y1, refined_x2, refined_y2,
            reduction_pct);

        [refined_x1, refined_y1, refined_x2, refined_y2]
    }

    /// Check if bbox1 contains bbox2
    #[allow(dead_code)]
    fn bbox_contains(bbox1: &[i32; 4], bbox2: &[i32; 4]) -> bool {
        bbox2[0] >= bbox1[0] && bbox2[1] >= bbox1[1] &&
        bbox2[2] <= bbox1[2] && bbox2[3] <= bbox1[3]
    }
}

