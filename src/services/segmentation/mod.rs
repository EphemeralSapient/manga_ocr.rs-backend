// Segmentation service using mask.onnx model

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::{Array2, Array4, Axis};
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
use std::sync::Arc;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tracing::{debug, info, instrument};

use crate::core::config::Config;

// Embed the mask model at compile time
static MASK_MODEL_BYTES: &[u8] = include_bytes!("../../../models/mask.onnx");

/// Session pool for concurrent inference
pub struct SegmentationSessionPool {
    sender: Sender<Session>,
    receiver: Arc<tokio::sync::Mutex<Receiver<Session>>>,
}

impl SegmentationSessionPool {
    async fn acquire(&self) -> Session {
        let mut receiver = self.receiver.lock().await;
        receiver.recv().await.expect("Session pool exhausted")
    }

    async fn release(&self, session: Session) {
        self.sender.send(session).await.expect("Failed to return session to pool");
    }
}

/// Segmentation service for generating text region masks
pub struct SegmentationService {
    session_pool: Arc<SegmentationSessionPool>,
    target_size: u32,
    #[allow(dead_code)]
    device_type: String,
}

impl SegmentationService {
    /// Create a new segmentation service
    ///
    /// # Arguments
    /// * `config` - Application configuration
    ///
    /// # Returns
    /// Result containing the segmentation service or an error
    #[instrument(skip(config), fields(target_size = config.target_size()))]
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        // Determine pool size: use num_cpus or MAX_CONCURRENT_BATCHES (whichever is smaller)
        let pool_size = std::cmp::min(num_cpus::get(), config.max_concurrent_batches());
        debug!("Creating segmentation session pool with {} sessions", pool_size);

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

        let session_pool = Arc::new(SegmentationSessionPool {
            sender,
            receiver: Arc::new(tokio::sync::Mutex::new(receiver)),
        });

        info!("✓ Segmentation: {} ({} sessions)", device_type, pool_size);

        Ok(Self {
            session_pool,
            target_size: config.target_size(),
            device_type,
        })
    }

    fn initialize_with_acceleration(config: &Config) -> Result<(String, Session)> {
        info!("Loading embedded ONNX model ({} bytes)...", MASK_MODEL_BYTES.len());

        // Check if a specific backend is forced via config
        if let Some(ref backend) = config.detection.inference_backend {
            match backend.as_str() {
                "AUTO" => {
                    info!("INFERENCE_BACKEND=AUTO, using automatic detection");
                }
                forced_backend => {
                    info!("INFERENCE_BACKEND={}, forcing specific backend for segmentation", forced_backend);
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
                .and_then(|b| b.commit_from_memory(MASK_MODEL_BYTES))
            {
                info!("✓ Using TensorRT acceleration for segmentation");
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
                .and_then(|b| b.commit_from_memory(MASK_MODEL_BYTES))
            {
                info!("✓ Using CUDA acceleration for segmentation");
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
                .and_then(|b| b.commit_from_memory(MASK_MODEL_BYTES))
            {
                info!("✓ Using CoreML acceleration for segmentation (Apple Neural Engine)");
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
                .and_then(|b| b.commit_from_memory(MASK_MODEL_BYTES))
            {
                info!("✓ Using DirectML acceleration for segmentation");
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
                .and_then(|b| b.commit_from_memory(MASK_MODEL_BYTES))
            {
                info!("✓ Using OpenVINO acceleration for segmentation (Intel CPU optimizations)");
                return Ok(("OpenVINO-CPU".to_string(), session));
            }
        }

        // Final fallback: Pure CPU (no acceleration)
        let session = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_memory(MASK_MODEL_BYTES)?;

        info!("✓ Using CPU for segmentation (no hardware acceleration)");
        Ok(("CPU".to_string(), session))
    }

    fn try_forced_backend(backend: &str) -> Result<(String, Session)> {
        match backend {
            #[cfg(feature = "tensorrt")]
            "TENSORRT" => {
                info!("Forcing TensorRT backend for segmentation...");
                let session = Session::builder()?
                    .with_execution_providers([TensorRTExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MASK_MODEL_BYTES)?;
                info!("✓ Successfully initialized TensorRT backend for segmentation");
                Ok(("TensorRT (forced)".to_string(), session))
            }
            #[cfg(not(feature = "tensorrt"))]
            "TENSORRT" => {
                anyhow::bail!("TensorRT backend not available. Rebuild with: cargo build --features tensorrt")
            }

            #[cfg(feature = "cuda")]
            "CUDA" => {
                info!("Forcing CUDA backend for segmentation...");
                let session = Session::builder()?
                    .with_execution_providers([CUDAExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MASK_MODEL_BYTES)?;
                info!("✓ Successfully initialized CUDA backend for segmentation");
                Ok(("CUDA (forced)".to_string(), session))
            }
            #[cfg(not(feature = "cuda"))]
            "CUDA" => {
                anyhow::bail!("CUDA backend not available. Rebuild with: cargo build --features cuda")
            }

            #[cfg(all(target_os = "macos", feature = "coreml"))]
            "COREML" => {
                info!("Forcing CoreML backend for segmentation...");
                let session = Session::builder()?
                    .with_execution_providers([CoreMLExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MASK_MODEL_BYTES)?;
                info!("✓ Successfully initialized CoreML backend for segmentation");
                Ok(("CoreML (forced)".to_string(), session))
            }
            #[cfg(not(all(target_os = "macos", feature = "coreml")))]
            "COREML" => {
                anyhow::bail!("CoreML backend not available. Only available on macOS with: cargo build --features coreml")
            }

            #[cfg(all(target_os = "windows", feature = "directml"))]
            "DIRECTML" => {
                info!("Forcing DirectML backend for segmentation...");
                let session = Session::builder()?
                    .with_execution_providers([DirectMLExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MASK_MODEL_BYTES)?;
                info!("✓ Successfully initialized DirectML backend for segmentation");
                Ok(("DirectML (forced)".to_string(), session))
            }
            #[cfg(not(all(target_os = "windows", feature = "directml")))]
            "DIRECTML" => {
                anyhow::bail!("DirectML backend not available. Only available on Windows with: cargo build --features directml")
            }

            #[cfg(feature = "openvino")]
            "OPENVINO" => {
                info!("Forcing OpenVINO backend for segmentation...");
                let session = Session::builder()?
                    .with_execution_providers([
                        OpenVINOExecutionProvider::default()
                            .with_device_type("CPU")
                            .build()
                    ])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MASK_MODEL_BYTES)?;
                info!("✓ Successfully initialized OpenVINO backend for segmentation");
                Ok(("OpenVINO (forced)".to_string(), session))
            }
            #[cfg(not(feature = "openvino"))]
            "OPENVINO" => {
                anyhow::bail!("OpenVINO backend not available. Rebuild with: cargo build --features openvino")
            }

            "CPU" => {
                info!("Forcing CPU backend for segmentation...");
                let session = Session::builder()?
                    .with_execution_providers([CPUExecutionProvider::default().build()])?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(num_cpus::get())?
                    .commit_from_memory(MASK_MODEL_BYTES)?;
                info!("✓ Successfully initialized CPU backend for segmentation");
                Ok(("CPU (forced)".to_string(), session))
            }

            _ => {
                anyhow::bail!(
                    "Unknown inference backend: {}. Supported: CUDA, TENSORRT, DIRECTML, COREML, OPENVINO, CPU",
                    backend
                )
            }
        }
    }

    /// Generate segmentation mask for an image
    ///
    /// # Arguments
    /// * `img` - Input image
    ///
    /// # Returns
    /// Result containing flattened binary mask (0 or 255) of shape (h*w)
    ///
    /// # Implementation
    /// Based on minimal_code.py lines 55-72:
    /// - Resize image to 640x640
    /// - Normalize to [0, 1] and convert to NCHW
    /// - Run segmenter to get detection output and proto masks
    /// - For each detection with conf > 0.25:
    ///   - Apply sigmoid(coeffs @ proto_masks)
    ///   - Resize to original size
    ///   - Threshold at 0.5
    /// - Combine all masks with bitwise OR
    #[instrument(skip(self, img), fields(width = img.width(), height = img.height()))]
    pub async fn generate_mask(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();
        let orig_width = img.width();
        let orig_height = img.height();

        debug!(
            "Generating segmentation mask for {}x{} image",
            orig_width, orig_height
        );

        // Preprocessing: Resize and normalize
        let (input_tensor, _) = self.preprocess_image(img)?;

        // Run inference
        let images_value = ort::value::Value::from_array(input_tensor)?;

        // Acquire session from pool, run inference, and extract data
        let (det_output, proto_masks) = {
            let mut session = self.session_pool.acquire().await;
            let outputs = session
                .run(ort::inputs!["images" => images_value])?;

            // Extract outputs while session is borrowed
            // det_output: [1, 37, num_detections] where 37 = 4 (bbox) + 1 (conf) + 32 (mask coeffs)
            // proto_masks: [1, 32, mask_h, mask_w]
            let (_shape0, det_data) = outputs["output0"]
                .try_extract_tensor::<f32>()?;

            // Calculate number of detections: total elements / 37
            let num_detections = det_data.len() / 37;
            let det_output = ndarray::Array3::from_shape_vec(
                (1, 37, num_detections),
                det_data.to_vec(),
            )?;

            let (_shape1, proto_data) = outputs["output1"]
                .try_extract_tensor::<f32>()?;
            // Assuming proto masks are [1, 32, 160, 160] based on YOLOv8 segmentation
            let proto_masks = ndarray::Array4::from_shape_vec(
                (1, 32, 160, 160),
                proto_data.to_vec(),
            )?;

            // Drop outputs and return session to pool
            drop(outputs);
            self.session_pool.release(session).await;

            (det_output, proto_masks)
        };

        debug!(
            "Detection output shape: {:?}, Proto masks shape: {:?}",
            det_output.shape(),
            proto_masks.shape()
        );

        // Process detections and generate masks
        let mask = self.process_segmentation(
            det_output,
            proto_masks,
            orig_width,
            orig_height,
        )?;

        debug!(
            "Segmentation completed in {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(mask)
    }

    /// Preprocess image for model input
    fn preprocess_image(&self, img: &DynamicImage) -> Result<(Array4<f32>, Array2<i64>)> {
        use image::imageops::FilterType;

        // Resize to target size (640x640)
        let resized = img.resize_exact(
            self.target_size,
            self.target_size,
            FilterType::Triangle,
        );

        // Convert to RGB and normalize to [0, 1]
        let rgb = resized.to_rgb8();
        let (width, height) = rgb.dimensions();

        // Create NCHW tensor [1, 3, 640, 640]
        let mut input_array = Array4::<f32>::zeros((1, 3, height as usize, width as usize));

        for (x, y, pixel) in rgb.enumerate_pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            input_array[[0, 0, y as usize, x as usize]] = r;
            input_array[[0, 1, y as usize, x as usize]] = g;
            input_array[[0, 2, y as usize, x as usize]] = b;
        }

        // Original size for output
        let orig_size = Array2::<i64>::from_shape_vec(
            (1, 2),
            vec![img.width() as i64, img.height() as i64],
        )?;

        Ok((input_array, orig_size))
    }

    /// Process segmentation outputs to generate combined mask
    ///
    /// Based on minimal_code.py lines 62-72
    fn process_segmentation(
        &self,
        det_output: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>,
        proto_masks: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>,
        orig_width: u32,
        orig_height: u32,
    ) -> Result<Vec<u8>> {
        let det_output = det_output.index_axis(Axis(0), 0); // Remove batch dim: [37, num_detections]
        let proto_masks = proto_masks.index_axis(Axis(0), 0); // Remove batch dim: [32, mask_h, mask_w]

        let num_detections = det_output.shape()[1];
        let (num_protos, mask_h, mask_w) = (
            proto_masks.shape()[0],
            proto_masks.shape()[1],
            proto_masks.shape()[2],
        );

        debug!(
            "Processing {} detections with proto masks {}x{}x{}",
            num_detections, num_protos, mask_h, mask_w
        );

        // Initialize combined mask
        let mut all_seg_mask =
            ImageBuffer::<Luma<u8>, Vec<u8>>::new(orig_width, orig_height);

        let confidence_threshold = 0.25;
        let sigmoid_threshold = 0.5;
        let mut valid_detections = 0;
        let loop_start = std::time::Instant::now();

        // First pass: count valid detections
        debug!("Scanning {} detections for confidence > {}", num_detections, confidence_threshold);
        let scan_start = std::time::Instant::now();

        for i in 0..num_detections {
            // Get confidence (5th element, index 4)
            let conf = det_output[[4, i]];
            if conf >= confidence_threshold {
                valid_detections += 1;
            }
        }

        debug!("Found {} valid detections in {:.2}ms",
            valid_detections,
            scan_start.elapsed().as_secs_f64() * 1000.0);

        // Second pass: process only valid detections
        let mut processed = 0;
        for i in 0..num_detections {
            // Get confidence (5th element, index 4)
            let conf = det_output[[4, i]];
            if conf < confidence_threshold {
                continue;
            }

            processed += 1;

            if processed == 1 {
                debug!("Processing first valid detection (#{} out of {}) - this should be fast!", i, num_detections);
            }

            // Get mask coefficients (elements 5-36, indices 5-36)
            // Python: coeffs = det_output[5:37, i]
            let coeffs = det_output.slice(ndarray::s![5..37, i]);

            // Generate mask: sigmoid(sum(coeffs * proto_masks, axis=0))
            // Python: mask = 1 / (1 + np.exp(-np.sum(proto_masks * coeffs[:, None, None], axis=0)))
            // Use direct weighted sum approach like Python
            let mut mask = Array2::<f32>::zeros((mask_h, mask_w));

            // Weighted sum: for each position (y,x), compute sum_k(coeffs[k] * proto_masks[k,y,x])
            for k in 0..num_protos {
                let coeff = coeffs[k];
                let proto_slice = proto_masks.slice(ndarray::s![k, .., ..]);

                // Add weighted proto mask to result
                use ndarray::Zip;
                Zip::from(&mut mask)
                    .and(&proto_slice)
                    .for_each(|m, &p| *m += coeff * p);
            }

            // Apply sigmoid element-wise
            mask.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));

            // Resize mask to original size using image crate
            let mask_img = ImageBuffer::<Luma<f32>, Vec<f32>>::from_vec(
                mask_w as u32,
                mask_h as u32,
                mask.iter().copied().collect(),
            )
            .context("Failed to create mask image")?;

            let resized_mask = image::imageops::resize(
                &mask_img,
                orig_width,
                orig_height,
                image::imageops::FilterType::Triangle,
            );

            // Threshold and combine with OR
            for (x, y, pixel) in resized_mask.enumerate_pixels() {
                if pixel[0] > sigmoid_threshold {
                    all_seg_mask.put_pixel(x, y, Luma([255u8]));
                }
            }
        }

        // Convert to flattened vector
        let flat_mask: Vec<u8> = all_seg_mask.into_raw();

        debug!(
            "Processed {} valid detections (out of {}) in {:.2}ms, generated mask with {} pixels masked",
            valid_detections,
            num_detections,
            loop_start.elapsed().as_secs_f64() * 1000.0,
            flat_mask.iter().filter(|&&p| p > 0).count()
        );

        Ok(flat_mask)
    }
}
