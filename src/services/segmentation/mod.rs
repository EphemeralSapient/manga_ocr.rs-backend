// Segmentation service with mode switching: Fast (YOLOv8) vs Accuracy (SAM 2.1)

mod sam21_tiny;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::{Array2, Array4, Axis};
use ort::session::Session;
use std::sync::Arc;
use tracing::{debug, info, instrument};

use crate::core::config::Config;
use crate::services::onnx_builder::{self};

pub use sam21_tiny::{RegionDetection, Sam21TinyService};

/// Segmentation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    /// Fast mode using YOLOv8-seg (104MB, ~100ms per image, 640x640)
    Fast,
    /// Accuracy mode using SAM 2.1-tiny (149MB, ~5s per bubble, 1024x1024)
    Accuracy,
}

impl MaskMode {
    /// Parse from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "fast" => Ok(MaskMode::Fast),
            "accuracy" | "accurate" => Ok(MaskMode::Accuracy),
            _ => anyhow::bail!("Invalid mask mode: '{}'. Expected 'fast', 'accurate', or 'accuracy'", s),
        }
    }
}

// Embed the mask model at compile time
// If LFS not checked out, this will be a small stub - load from runtime path instead
static MASK_MODEL_BYTES: &[u8] = include_bytes!("../../../models/mask.onnx");

/// Load model bytes from embedded or runtime path
fn load_model_bytes(config: &Config) -> Result<Vec<u8>> {
    // Check if embedded model is real (>10KB means it's not an LFS stub)
    if MASK_MODEL_BYTES.len() > 10_000 {
        debug!("Using embedded mask model ({} MB)", MASK_MODEL_BYTES.len() as f64 / 1_048_576.0);
        Ok(MASK_MODEL_BYTES.to_vec())
    } else {
        // Load from runtime path
        let path = &config.detection.mask_model_path;
        debug!("Loading mask model from: {}", path);
        std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("Failed to load mask model from {}: {}", path, e))
    }
}

use crate::services::onnx_builder::DynamicSessionPool;
use tokio::sync::OnceCell;

/// Segmentation service for generating text region masks
/// Supports two modes: Fast (YOLOv8-seg) and Accuracy (SAM 2.1-tiny)
/// Both modes are lazily initialized on first use
pub struct SegmentationService {
    target_size: u32,
    max_sessions: usize,
    config: Arc<Config>,
    // Lazy-initialized Fast mode (YOLOv8-seg)
    fast_session_pool: Arc<OnceCell<Arc<DynamicSessionPool>>>,
    fast_device_type: Arc<OnceCell<String>>,
    // Lazy-initialized Accuracy mode (SAM 2.1-tiny)
    sam_service: Arc<OnceCell<Arc<Sam21TinyService>>>,
}

impl SegmentationService {
    /// Create a new segmentation service with lazy initialization
    /// Models are NOT loaded at startup - they initialize on first use
    ///
    /// # Arguments
    /// * `config` - Application configuration
    ///
    /// # Returns
    /// Segmentation service ready for lazy initialization
    #[instrument(skip(config), fields(target_size = config.target_size()))]
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        let max_sessions = config.onnx_pool_size();

        info!("🚀 Creating segmentation service with LAZY initialization");
        info!("   Fast mode (YOLOv8): Will load on first request with mask_mode=fast");
        info!("   Accuracy mode (SAM 2.1): Will load on first request with mask_mode=accurate");

        Ok(Self {
            target_size: config.target_size(),
            max_sessions,
            config,
            fast_session_pool: Arc::new(OnceCell::new()),
            fast_device_type: Arc::new(OnceCell::new()),
            sam_service: Arc::new(OnceCell::new()),
        })
    }

    /// Create segmentation service with optional session limit override
    pub async fn new_with_limit(config: Arc<Config>, session_limit: Option<usize>) -> Result<Self> {
        let max_sessions = session_limit.unwrap_or_else(|| config.onnx_pool_size());

        info!("🚀 Creating segmentation service with LAZY initialization (session limit: {})", max_sessions);

        Ok(Self {
            target_size: config.target_size(),
            max_sessions,
            config,
            fast_session_pool: Arc::new(OnceCell::new()),
            fast_device_type: Arc::new(OnceCell::new()),
            sam_service: Arc::new(OnceCell::new()),
        })
    }

    /// Lazy initialize Fast mode (YOLOv8-seg) on first use
    async fn ensure_fast_initialized(&self) -> Result<()> {
        // Check if already initialized
        if self.fast_session_pool.get().is_some() {
            return Ok(());
        }

        info!("🔄 [LAZY INIT] Initializing Fast segmentation (YOLOv8) on first use...");
        let init_start = std::time::Instant::now();

        // Create first session to determine device type
        let (device_type, first_session) = Self::initialize_with_acceleration(&self.config)?;

        // Create dynamic session pool with capacity for max_sessions
        let session_pool = DynamicSessionPool::new(self.max_sessions);
        session_pool.add_session(first_session);
        let session_pool = Arc::new(session_pool);

        // Warmup inference
        info!("Running warmup inference for Fast segmentation...");
        let warmup_start = std::time::Instant::now();
        self.warmup_fast(&session_pool).await?;
        info!("✓ Fast segmentation warmup completed in {:.2}ms", warmup_start.elapsed().as_secs_f64() * 1000.0);

        // Store in OnceCell (this can only succeed once)
        let _ = self.fast_session_pool.set(session_pool);
        let _ = self.fast_device_type.set(device_type.clone());

        let is_directml = device_type.contains("DirectML");
        if is_directml {
            info!("✓ Fast segmentation initialized: {} (1/1 sessions, ~40 MB used) [DirectML: pool expansion disabled] in {:.2}ms",
                  device_type, init_start.elapsed().as_secs_f64() * 1000.0);
        } else {
            info!("✓ Fast segmentation initialized: {} (1/{} sessions allocated, ~40 MB used, {} MB max) in {:.2}ms",
                  device_type, self.max_sessions, self.max_sessions * 40, init_start.elapsed().as_secs_f64() * 1000.0);
        }

        Ok(())
    }

    /// Lazy initialize Accuracy mode (SAM 2.1-tiny) on first use
    async fn ensure_sam_initialized(&self) -> Result<()> {
        // Check if already initialized
        if self.sam_service.get().is_some() {
            return Ok(());
        }

        info!("🔄 [LAZY INIT] Initializing Accuracy segmentation (SAM 2.1-tiny) on first use...");
        let init_start = std::time::Instant::now();

        // Create SAM service
        let sam = Arc::new(Sam21TinyService::new(Arc::clone(&self.config)).await?);

        // Store in OnceCell (this can only succeed once)
        let _ = self.sam_service.set(sam);

        info!("✓ Accuracy segmentation initialized: SAM 2.1-tiny (~155 MB per session pair) in {:.2}ms",
              init_start.elapsed().as_secs_f64() * 1000.0);

        Ok(())
    }

    /// Run warmup inference for Fast mode to trigger JIT compilation and memory allocation
    async fn warmup_fast(&self, session_pool: &DynamicSessionPool) -> Result<()> {
        // Create a small dummy image (just needs to match expected dimensions)
        let dummy_img = image::DynamicImage::new_rgb8(self.target_size, self.target_size);

        // Run inference on one session to trigger optimization
        // Preprocess: resize and normalize
        let resized = dummy_img.resize_exact(
            self.target_size,
            self.target_size,
            image::imageops::FilterType::Triangle,
        );
        let rgb_img = resized.to_rgb8();

        let target = self.target_size as usize;
        let mut array = Array4::<f32>::zeros((1, 3, target, target));

        for y in 0..target {
            for x in 0..target {
                let pixel = rgb_img.get_pixel(x as u32, y as u32);
                array[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
                array[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
                array[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }

        let input_value = ort::value::Value::from_array(array)?;

        // Acquire session and run dummy inference
        let mut session = session_pool.acquire();
        {
            let _outputs = session.run(ort::inputs!["images" => input_value])?;
            // outputs dropped here before releasing session
        }
        session_pool.release(session);

        Ok(())
    }

    fn initialize_with_acceleration(config: &Config) -> Result<(String, Session)> {
        // Load model bytes (embedded or from runtime path)
        let model_bytes = load_model_bytes(config)?;
        info!("Loaded mask model ({:.1} MB)", model_bytes.len() as f64 / 1_048_576.0);

        // Validate ONNX model header
        if model_bytes.len() < 100 {
            anyhow::bail!(
                "Mask model file is too small ({} bytes). This might be a Git LFS stub. \
                Please ensure Git LFS is installed and the model is properly checked out.",
                model_bytes.len()
            );
        }

        // Check for valid ONNX protobuf header
        if model_bytes.len() >= 4 {
            debug!("Mask model header bytes: {:02x} {:02x} {:02x} {:02x}",
                model_bytes[0], model_bytes[1], model_bytes[2], model_bytes[3]);
        }

        // Use shared ONNX session builder (eliminates ~220 lines of duplication)
        let model_size_mb = model_bytes.len() as f32 / 1_048_576.0;
        let (backend, session) = onnx_builder::build_session_with_acceleration(
            &model_bytes,
            "segmentation",
            model_size_mb
        )?;


        Ok((backend, session))
    }

    /// Check if using DirectML backend (for Fast mode)
    pub fn is_directml(&self) -> bool {
        if let Some(device_type) = self.fast_device_type.get() {
            device_type.contains("DirectML")
        } else {
            false  // Not initialized yet, assume false
        }
    }

    /// Expand session pool if under capacity (lazy allocation)
    /// Expands aggressively when high contention detected
    ///
    /// IMPORTANT: DirectML processes sequentially, so pool expansion is disabled
    /// Only applicable to Fast mode (YOLOv8)
    fn expand_if_needed(&self) {
        // Only applicable to Fast mode
        let session_pool = match self.fast_session_pool.get() {
            Some(pool) => pool,
            None => return,  // Not initialized yet
        };

        // DirectML can only use 1 session at a time (sequential processing)
        if self.is_directml() {
            return;
        }

        let available = session_pool.available();
        let in_use = session_pool.in_use();
        let total = available + in_use;

        // Aggressive expansion: create multiple sessions if under pressure
        let sessions_to_create = if available == 0 && total < self.max_sessions {
            let headroom = self.max_sessions - total;
            std::cmp::min(3, headroom) // Create up to 3 sessions at once
        } else {
            0
        };

        if sessions_to_create > 0 {
            debug!("🔄 [LAZY EXPANSION] Segmentation pool under pressure: {}/{} sessions, creating {} new sessions",
                   total, self.max_sessions, sessions_to_create);

            let mut created = 0;
            for _ in 0..sessions_to_create {
                if let Ok((_, new_session)) = Self::initialize_with_acceleration(&self.config) {
                    session_pool.add_session(new_session);
                    created += 1;
                } else {
                    debug!("⚠️  Failed to create additional segmentation session");
                    break;
                }
            }

            if created > 0 {
                info!("✓ Expanded segmentation pool: {}/{} sessions ({} MB used)",
                      total + created, self.max_sessions, (total + created) * 40);
            }
        }
    }

    /// Drain all sessions and free memory (call after phase 1 complete)
    ///
    /// IMPORTANT: DirectML keeps persistent sessions (no cleanup)
    /// Only applicable to Fast mode (YOLOv8)
    pub fn cleanup_sessions(&self) {
        // Only applicable to Fast mode
        let session_pool = match self.fast_session_pool.get() {
            Some(pool) => pool,
            None => {
                debug!("Fast mode not initialized yet - nothing to cleanup");
                return;
            }
        };

        // DirectML: Keep persistent session (no cleanup/recreation overhead)
        if self.is_directml() {
            debug!("DirectML: Keeping persistent segmentation session (no cleanup)");
            return;
        }

        let sessions = session_pool.drain_all();
        let count = sessions.len();
        drop(sessions); // Explicit drop to ensure memory is freed
        info!("🧹 [CLEANUP] Dropped {} segmentation sessions (~{} MB freed)",
              count, count * 40);
    }

    /// Generate segmentation mask for an image with dynamic mode selection
    ///
    /// # Arguments
    /// * `img` - Input image
    /// * `mode` - Segmentation mode to use (Fast or Accuracy)
    /// * `regions` - Optional detected regions (required for Accuracy mode)
    ///
    /// # Returns
    /// Result containing flattened binary mask (0 or 255) of shape (h*w)
    #[instrument(skip(self, img, regions), fields(width = img.width(), height = img.height(), mode = ?mode))]
    pub async fn generate_mask(&self, img: &DynamicImage, mode: MaskMode, regions: Option<&[RegionDetection]>) -> Result<Vec<u8>> {
        match mode {
            MaskMode::Fast => {
                // Lazy initialize Fast mode if not already initialized
                self.ensure_fast_initialized().await?;
                self.generate_mask_fast(img).await
            }
            MaskMode::Accuracy => {
                // Lazy initialize SAM if not already initialized
                self.ensure_sam_initialized().await?;
                let regions = regions.ok_or_else(|| {
                    anyhow::anyhow!("Accuracy mode requires detected regions for prompting")
                })?;
                self.generate_mask_sam(img, regions).await
            }
        }
    }

    /// Generate mask using Fast mode (YOLOv8-seg)
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
    async fn generate_mask_fast(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        let session_pool = self.fast_session_pool.get()
            .ok_or_else(|| anyhow::anyhow!("Fast mode session pool not initialized - call ensure_fast_initialized() first"))?;
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

        // Lazy expand pool if needed
        self.expand_if_needed();

        // Acquire session from pool, run inference, and extract data
        let (det_output, proto_masks) = {
            let mut session = session_pool.acquire();
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
            session_pool.release(session);  // No await - crossbeam is sync!

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
    ///
    /// OPTIMIZATIONS applied:
    /// 1. FilterType::Nearest instead of Triangle (10-30x faster for masks)
    /// 2. Optimized matrix multiplication using ndarray operations
    /// 3. Vectorized threshold and mask combination
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

        let confidence_threshold = 0.25;
        let sigmoid_threshold = 0.5;
        let loop_start = std::time::Instant::now();

        // OPTIMIZATION: Reshape proto_masks for efficient matrix multiplication
        // From [32, 160, 160] to [32, 25600] for coeffs @ proto_flat
        let proto_flat = proto_masks
            .to_owned()
            .into_shape_with_order((num_protos, mask_h * mask_w))
            .context("Failed to reshape proto masks")?;

        // MAJOR OPTIMIZATION: Combine all masks at LOW resolution (160x160) first,
        // then resize ONCE to full resolution. This reduces 60 resizes to 1 resize.
        // Previous: 60 * resize(160x160 -> 1337x1920) = 150M pixels
        // Now: combine at 160x160, then 1 * resize = 2.5M pixels
        let mut combined_mask_low_res = vec![0u8; mask_h * mask_w];
        let mut valid_detections = 0;

        for i in 0..num_detections {
            // Get confidence (5th element, index 4)
            let conf = det_output[[4, i]];
            if conf < confidence_threshold {
                continue; // Early skip - avoid processing low-confidence detections
            }

            valid_detections += 1;

            if valid_detections == 1 {
                debug!("Processing first valid detection (#{} out of {}) with confidence {:.3}", i, num_detections, conf);
            }

            // Get mask coefficients (elements 5-36, indices 5-36)
            let coeffs = det_output.slice(ndarray::s![5..37, i]);

            // Matrix multiplication: coeffs [32] @ proto_flat [32, 25600] -> mask_flat [25600]
            let mask_flat = coeffs.dot(&proto_flat);

            // Vectorized sigmoid and threshold at low resolution
            for (idx, &val) in mask_flat.iter().enumerate() {
                let sigmoid_val = 1.0 / (1.0 + (-val).exp());
                if sigmoid_val > sigmoid_threshold {
                    combined_mask_low_res[idx] = 255;
                }
            }
        }

        // Single resize of combined mask from 160x160 to original size
        let combined_img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(
            mask_w as u32,
            mask_h as u32,
            combined_mask_low_res,
        )
        .context("Failed to create combined mask image")?;

        let resized_mask = image::imageops::resize(
            &combined_img,
            orig_width,
            orig_height,
            image::imageops::FilterType::Nearest,
        );

        let all_seg_mask = resized_mask.into_raw();

        debug!(
            "Processed {} valid detections (out of {}) in {:.2}ms, generated mask with {} pixels masked",
            valid_detections,
            num_detections,
            loop_start.elapsed().as_secs_f64() * 1000.0,
            all_seg_mask.iter().filter(|&&p| p > 0).count()
        );

        Ok(all_seg_mask)
    }

    /// Generate mask using Accuracy mode (SAM 2.1-tiny)
    async fn generate_mask_sam(&self, img: &DynamicImage, regions: &[RegionDetection]) -> Result<Vec<u8>> {
        let sam_service = self.sam_service.get()
            .ok_or_else(|| anyhow::anyhow!("SAM service not initialized - call ensure_sam_initialized() first"))?;

        sam_service.generate_mask(img, regions).await
    }
}
