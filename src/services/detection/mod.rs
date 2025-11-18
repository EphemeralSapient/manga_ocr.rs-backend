use crate::core::config::Config;
use crate::core::types::BubbleDetection;
use crate::services::onnx_builder::{self, OnnxSessionPool};
use anyhow::{Result, Context};
use image::DynamicImage;
use ndarray::{Array2, Array4};
use ort::execution_providers::CPUExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
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

pub struct DetectionService {
    session_pool: Arc<OnnxSessionPool>,
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

        // Create generic ONNX session pool (lock-free, high performance)
        let session_pool = OnnxSessionPool::new(pool_size);

        // Send first session to pool (synchronous - no await needed)
        session_pool.sender().send(first_session)
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
                session_pool.sender().send(session)
                    .map_err(|_| anyhow::anyhow!("Failed to add session to pool"))?;
            }
        }

        // Wrap in Arc for sharing across threads
        let session_pool = Arc::new(session_pool);

        info!("✓ Detection: {} ({} sessions)", device_type, pool_size);

        Ok(Self {
            session_pool,
            config,
            device_type,
        })
    }

    fn try_forced_backend(backend: &str, model_bytes: &[u8]) -> Result<(String, Session)> {
        match backend {
            #[cfg(feature = "tensorrt")]
            "TENSORRT" => {
                info!("Forcing TensorRT backend...");
                let session = Session::builder()?
                    .with_execution_providers([TensorRTExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(model_bytes)?;
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
                    .commit_from_memory(model_bytes)?;
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
                    .commit_from_memory(model_bytes)?;
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
                    .commit_from_memory(model_bytes)?;
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
                    .commit_from_memory(model_bytes)?;
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
                    .commit_from_memory(model_bytes)?;
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
        // Load model bytes (embedded or from runtime path)
        let model_bytes = load_model_bytes(config)?;
        info!("Loaded detector model ({:.1} MB)", model_bytes.len() as f64 / 1_048_576.0);

        // Validate ONNX model header
        if model_bytes.len() < 100 {
            anyhow::bail!(
                "Model file is too small ({} bytes). This might be a Git LFS stub. \
                Please ensure Git LFS is installed and the model is properly checked out.",
                model_bytes.len()
            );
        }

        // Check for valid ONNX protobuf header (should start with model version info)
        if model_bytes.len() >= 4 {
            debug!("Model header bytes: {:02x} {:02x} {:02x} {:02x}",
                model_bytes[0], model_bytes[1], model_bytes[2], model_bytes[3]);
        }

        // Log environment info
        info!("Platform: {}/{}", std::env::consts::OS, std::env::consts::ARCH);

        // Check if a specific backend is forced via config
        if let Some(ref backend) = config.detection.inference_backend {
            match backend.as_str() {
                "AUTO" => {
                    info!("INFERENCE_BACKEND=AUTO, using automatic detection");
                }
                forced_backend => {
                    info!("INFERENCE_BACKEND={}, forcing specific backend", forced_backend);
                    return Self::try_forced_backend(forced_backend, &model_bytes);
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
                .and_then(|b| b.commit_from_memory(&model_bytes))
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
                .and_then(|b| b.commit_from_memory(&model_bytes))
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
                .and_then(|b| b.commit_from_memory(&model_bytes))
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
                .and_then(|b| b.commit_from_memory(&model_bytes))
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
                .and_then(|b| b.commit_from_memory(&model_bytes))
            {
                info!("✓ Using OpenVINO acceleration (Intel CPU optimizations)");
                return Ok(("OpenVINO-CPU".to_string(), session));
            }
        }

        // Final fallback: Pure CPU (no acceleration)
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .context("Failed to configure CPU execution provider")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set graph optimization level")?
            .with_intra_threads(num_cpus::get())
            .context("Failed to configure intra-op threads")?
            .commit_from_memory(&model_bytes)
            .context(format!(
                "Failed to load ONNX model from memory ({:.1} MB). \
                This usually indicates:\n  \
                1. Model file corruption during transfer (verify with: ./verify_models.sh or verify_models.ps1)\n  \
                2. ONNX Runtime version/platform mismatch\n  \
                3. Model created with incompatible ONNX opset version\n  \
                4. Platform-specific binary incompatibility ({}/{})\n  \
                Solutions:\n  \
                - Run checksum verification: ./verify_models.sh (Linux/Mac) or .\\verify_models.ps1 (Windows)\n  \
                - Re-download/re-transfer the models if checksums fail\n  \
                - Ensure the model was created with ONNX opset <= 18\n  \
                - Check that the binary matches your platform",
                model_bytes.len() as f64 / 1_048_576.0,
                std::env::consts::OS,
                std::env::consts::ARCH
            ))?;

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
        // OPTIMIZED: Process tensors in-place and only clone filtered results
        let (label_0_detections, label_1_detections, label_2_detections, inference_time) = {
            let mut session = self.session_pool.acquire();  // No await - crossbeam is sync!
            let outputs = session.run(ort::inputs![
                "images" => images_value,
                "orig_target_sizes" => sizes_value
            ])?;

            let inference_time = inference_start.elapsed();

            // Extract tensor references (no cloning yet)
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

                // Early filter - skip low-confidence detections without allocating
                if score < confidence_threshold {
                    continue;
                }

                // Only clone bbox data for high-confidence detections
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

            // Drop outputs and return session to pool
            drop(outputs);
            self.session_pool.release(session);  // No await - crossbeam is sync!

            (label_0_detections, label_1_detections, label_2_detections, inference_time)
        };

        debug!("✓ Inference completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);

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
        // OPTIMIZED: Process tensors in-place and only clone filtered results
        let (text_detections, inference_time) = {
            let mut session = self.session_pool.acquire();  // No await - crossbeam is sync!
            let outputs = session.run(ort::inputs![
                "images" => images_value,
                "orig_target_sizes" => sizes_value
            ])?;

            let inference_time = inference_start.elapsed();

            // Extract tensor references (no cloning yet)
            let (labels_shape, labels_data) = outputs["labels"].try_extract_tensor::<i64>()?;
            let (_boxes_shape, boxes_data) = outputs["boxes"].try_extract_tensor::<f32>()?;
            let (_scores_shape, scores_data) = outputs["scores"].try_extract_tensor::<f32>()?;

            let num_detections = labels_shape[1] as usize;
            trace!("Raw detections from model: {}", num_detections);

            let confidence_threshold = self.config.confidence_threshold();
            let mut text_detections = Vec::new();

            // Process tensors in-place - only clone filtered detections
            for i in 0..num_detections {
                let label = labels_data[i];
                let score = scores_data[i];

                // Early filter - skip non-matching or low-confidence detections
                if label != target_label || score < confidence_threshold {
                    continue;
                }

                // Only clone bbox data for matching detections
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

            // Drop outputs and return session to pool
            drop(outputs);
            self.session_pool.release(session);  // No await - crossbeam is sync!

            (text_detections, inference_time)
        };

        debug!("✓ Inference completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);

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

