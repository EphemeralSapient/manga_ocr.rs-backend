// Simplified Segmentation Service - YOLOv8-seg only (SAM removed for performance)
//
// This module provides fast text region segmentation using YOLOv8-seg.
// The slower SAM 2.1 accuracy mode has been removed to improve overall performance.

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::{Array2, Array4, Axis};
use ort::session::Session;
use std::sync::Arc;
use tracing::{debug, info, instrument};

use crate::core::config::Config;
use crate::services::onnx_builder::{self, DynamicSessionPool};

use tokio::sync::OnceCell;

// Embed the mask model at compile time
static MASK_MODEL_BYTES: &[u8] = include_bytes!("../../../models/mask.onnx");

/// Load model bytes from embedded or runtime path
fn load_model_bytes(config: &Config) -> Result<Vec<u8>> {
    if MASK_MODEL_BYTES.len() > 10_000 {
        debug!("Using embedded mask model ({:.1} MB)", MASK_MODEL_BYTES.len() as f64 / 1_048_576.0);
        Ok(MASK_MODEL_BYTES.to_vec())
    } else {
        let path = &config.detection.mask_model_path;
        debug!("Loading mask model from: {}", path);
        std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("Failed to load mask model from {}: {}", path, e))
    }
}

/// Simplified Segmentation Service - YOLOv8 only
///
/// Performance: ~80-120ms per image (CPU), ~20-40ms (GPU)
/// Memory: ~40 MB per session
pub struct SegmentationService {
    target_size: u32,
    max_sessions: usize,
    config: Arc<Config>,
    session_pool: Arc<OnceCell<Arc<DynamicSessionPool>>>,
    device_type: Arc<OnceCell<String>>,
}

impl SegmentationService {
    /// Create new segmentation service with lazy initialization
    #[instrument(skip(config), fields(target_size = config.target_size()))]
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        let max_sessions = config.onnx_pool_size();

        info!("🚀 Creating segmentation service (YOLOv8-seg, lazy init)");

        Ok(Self {
            target_size: config.target_size(),
            max_sessions,
            config,
            session_pool: Arc::new(OnceCell::new()),
            device_type: Arc::new(OnceCell::new()),
        })
    }

    /// Create with optional session limit override
    pub async fn new_with_limit(config: Arc<Config>, session_limit: Option<usize>) -> Result<Self> {
        let max_sessions = session_limit.unwrap_or_else(|| config.onnx_pool_size());

        Ok(Self {
            target_size: config.target_size(),
            max_sessions,
            config,
            session_pool: Arc::new(OnceCell::new()),
            device_type: Arc::new(OnceCell::new()),
        })
    }

    /// Lazy initialize on first use
    async fn ensure_initialized(&self) -> Result<()> {
        if self.session_pool.get().is_some() {
            return Ok(());
        }

        info!("🔄 Initializing YOLOv8-seg on first use...");
        let init_start = std::time::Instant::now();

        let (device_type, first_session) = Self::create_session(&self.config)?;

        let pool = DynamicSessionPool::new(self.max_sessions);
        pool.add_session(first_session);
        let pool = Arc::new(pool);

        // Warmup
        self.warmup(&pool).await?;

        let _ = self.session_pool.set(pool);
        let _ = self.device_type.set(device_type.clone());

        info!("✓ Segmentation ready: {} ({:.0}ms)", device_type, init_start.elapsed().as_secs_f64() * 1000.0);

        Ok(())
    }

    fn create_session(config: &Config) -> Result<(String, Session)> {
        let model_bytes = load_model_bytes(config)?;

        if model_bytes.len() < 100 {
            anyhow::bail!("Mask model too small - check Git LFS");
        }

        let model_size_mb = model_bytes.len() as f32 / 1_048_576.0;
        onnx_builder::build_session_with_acceleration(&model_bytes, "segmentation", model_size_mb)
    }

    async fn warmup(&self, pool: &DynamicSessionPool) -> Result<()> {
        let dummy = DynamicImage::new_rgb8(self.target_size, self.target_size);
        let (input_tensor, _) = self.preprocess_image(&dummy)?;
        let input_value = ort::value::Value::from_array(input_tensor)?;

        let mut session = pool.acquire();
        let _ = session.run(ort::inputs!["images" => input_value])?;
        pool.release(session);

        Ok(())
    }

    /// Check if using DirectML
    pub fn is_directml(&self) -> bool {
        self.device_type.get().map(|s| s.contains("DirectML")).unwrap_or(false)
    }

    /// Expand pool if under pressure
    fn expand_if_needed(&self) {
        let pool = match self.session_pool.get() {
            Some(p) => p,
            None => return,
        };

        if self.is_directml() {
            return; // DirectML = sequential only
        }

        let available = pool.available();
        let in_use = pool.in_use();
        let total = available + in_use;

        if available == 0 && total < self.max_sessions {
            let to_create = std::cmp::min(3, self.max_sessions - total);
            for _ in 0..to_create {
                if let Ok((_, session)) = Self::create_session(&self.config) {
                    pool.add_session(session);
                }
            }
        }
    }

    /// Cleanup sessions to free memory
    pub fn cleanup_sessions(&self) {
        if let Some(pool) = self.session_pool.get() {
            if !self.is_directml() {
                let sessions = pool.drain_all();
                info!("🧹 Dropped {} segmentation sessions", sessions.len());
            }
        }
    }

    /// Generate segmentation mask for an image
    #[instrument(skip(self, img), fields(w = img.width(), h = img.height()))]
    pub async fn generate_mask(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        self.ensure_initialized().await?;

        let pool = self.session_pool.get().unwrap();
        let start = std::time::Instant::now();
        let (orig_w, orig_h) = (img.width(), img.height());

        // Preprocess
        let (input_tensor, _) = self.preprocess_image(img)?;
        let images_value = ort::value::Value::from_array(input_tensor)?;

        self.expand_if_needed();

        // Run inference
        let (det_output, proto_masks) = {
            let mut session = pool.acquire();
            let outputs = session.run(ort::inputs!["images" => images_value])?;

            let (_, det_data) = outputs["output0"].try_extract_tensor::<f32>()?;
            let num_det = det_data.len() / 37;
            let det = ndarray::Array3::from_shape_vec((1, 37, num_det), det_data.to_vec())?;

            let (_, proto_data) = outputs["output1"].try_extract_tensor::<f32>()?;
            let proto = ndarray::Array4::from_shape_vec((1, 32, 160, 160), proto_data.to_vec())?;

            drop(outputs);
            pool.release(session);
            (det, proto)
        };

        // Process to mask
        let mask = self.process_detections(det_output, proto_masks, orig_w, orig_h)?;

        debug!("Segmentation done in {:.0}ms", start.elapsed().as_secs_f64() * 1000.0);
        Ok(mask)
    }

    fn preprocess_image(&self, img: &DynamicImage) -> Result<(Array4<f32>, Array2<i64>)> {
        let resized = img.resize_exact(self.target_size, self.target_size, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let (w, h) = rgb.dimensions();

        let mut input = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
        for (x, y, pixel) in rgb.enumerate_pixels() {
            input[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            input[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            input[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }

        let orig_size = Array2::<i64>::from_shape_vec((1, 2), vec![img.width() as i64, img.height() as i64])?;
        Ok((input, orig_size))
    }

    fn process_detections(
        &self,
        det_output: ndarray::Array3<f32>,
        proto_masks: ndarray::Array4<f32>,
        orig_w: u32,
        orig_h: u32,
    ) -> Result<Vec<u8>> {
        let det = det_output.index_axis(Axis(0), 0);
        let proto = proto_masks.index_axis(Axis(0), 0);

        let num_det = det.shape()[1];
        let (num_protos, mask_h, mask_w) = (proto.shape()[0], proto.shape()[1], proto.shape()[2]);

        // Reshape proto for matrix multiply
        let proto_flat = proto.to_owned().into_shape_with_order((num_protos, mask_h * mask_w))?;

        // Combine masks at low resolution first
        let mut combined = vec![0u8; mask_h * mask_w];
        let conf_threshold = 0.25f32;
        let sigmoid_threshold = 0.5f32;

        for i in 0..num_det {
            let conf = det[[4, i]];
            if conf < conf_threshold {
                continue;
            }

            let coeffs = det.slice(ndarray::s![5..37, i]);
            let mask_flat = coeffs.dot(&proto_flat);

            for (idx, &val) in mask_flat.iter().enumerate() {
                let sigmoid = 1.0 / (1.0 + (-val).exp());
                if sigmoid > sigmoid_threshold {
                    combined[idx] = 255;
                }
            }
        }

        // Single resize to original size
        let mask_img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(mask_w as u32, mask_h as u32, combined)
            .context("Failed to create mask image")?;

        let resized = image::imageops::resize(&mask_img, orig_w, orig_h, image::imageops::FilterType::Nearest);
        Ok(resized.into_raw())
    }
}
