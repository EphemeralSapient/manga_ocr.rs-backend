// Text Segmentation Service - FPN-based text detection for mask generation
//
// Uses FPN text detection model to identify text regions.
// CPU-only inference for consistent performance across platforms.

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage};
use ndarray::{Array1, Array2, Array4};
use ort::{session::{Session, SessionOutputs}, value::Value};
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::{debug, info, instrument};

use crate::core::config::Config;

// Embed the text cleaner model at compile time (CPU-only, ~11MB)
static TEXT_CLEANER_MODEL_BYTES: &[u8] = include_bytes!("../../../models/text_cleaner.onnx");

/// FPN output levels for text detection
const FPN_LEVELS: [&str; 3] = ["fpn2", "fpn3", "fpn4"];

/// Text Segmentation Service using FPN-based text detection
///
/// This uses an FPN text detector which outputs horizontal and vertical
/// text score maps at multiple scales. These are combined into a unified mask.
///
/// CPU-only for consistent cross-platform behavior.
pub struct SegmentationService {
    session: Mutex<Session>,
    config: Arc<Config>,
    /// Detection threshold (0.0-1.0, lower = more aggressive detection)
    threshold: f32,
    /// Dilation iterations to expand mask coverage
    dilate_iterations: u32,
    /// Dilation kernel size
    dilate_kernel_size: u32,
}

impl SegmentationService {
    /// Create new segmentation service
    #[instrument(skip(config))]
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        info!("🚀 Initializing text segmentation service (FPN text detector, CPU-only)");
        let init_start = std::time::Instant::now();

        // Verify model is properly embedded
        if TEXT_CLEANER_MODEL_BYTES.len() < 100_000 {
            anyhow::bail!(
                "Text cleaner model too small ({} bytes) - check Git LFS or model embedding",
                TEXT_CLEANER_MODEL_BYTES.len()
            );
        }

        let model_size_mb = TEXT_CLEANER_MODEL_BYTES.len() as f64 / 1_048_576.0;
        info!("Loading embedded text cleaner model ({:.1} MB)", model_size_mb);

        // Create CPU-only session (no GPU acceleration)
        let session = Session::builder()?
            .with_intra_threads(num_cpus::get().min(8))?
            .with_inter_threads(1)?
            .commit_from_memory(TEXT_CLEANER_MODEL_BYTES)
            .context("Failed to load text cleaner ONNX model")?;

        info!(
            "✓ Text segmentation ready: CPU ({:.0}ms)",
            init_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(Self {
            session: Mutex::new(session),
            config,
            threshold: 0.3,        // Default detection threshold
            dilate_iterations: 3,  // Expand mask to ensure full text coverage
            dilate_kernel_size: 5, // 5x5 dilation kernel
        })
    }

    /// Create with optional session limit override (compatibility method)
    pub async fn new_with_limit(config: Arc<Config>, _session_limit: Option<usize>) -> Result<Self> {
        // Session limit not used for CPU-only single session
        Self::new(config).await
    }

    /// Check if using DirectML (always false for CPU-only)
    pub fn is_directml(&self) -> bool {
        false
    }

    /// Cleanup sessions (no-op for single session)
    pub fn cleanup_sessions(&self) {
        // Single session, nothing to cleanup dynamically
    }

    /// Preprocess image for model input
    ///
    /// Returns (data tensor, im_info tensor, featuremap_cond tensor, scale, orig_size, new_size)
    fn preprocess(&self, img: &DynamicImage, target_size: u32) -> Result<PreprocessResult> {
        let rgb = img.to_rgb8();
        let (orig_w, orig_h) = (rgb.width(), rgb.height());

        // Calculate scale to fit target size
        let scale = target_size as f32 / orig_w.max(orig_h) as f32;
        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;

        // Round to multiple of 64 for model compatibility
        let new_w = ((new_w + 63) / 64) * 64;
        let new_h = ((new_h + 63) / 64) * 64;

        // Ensure minimum size
        let new_w = new_w.max(256);
        let new_h = new_h.max(256);

        // Resize image
        let resized = image::imageops::resize(
            &rgb,
            new_w,
            new_h,
            image::imageops::FilterType::Triangle,
        );

        // Create data tensor [1, 3, H, W] - raw float values (not normalized to 0-1)
        let mut data = Array4::<f32>::zeros((1, 3, new_h as usize, new_w as usize));
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                data[[0, 0, y, x]] = pixel[0] as f32; // R
                data[[0, 1, y, x]] = pixel[1] as f32; // G
                data[[0, 2, y, x]] = pixel[2] as f32; // B
            }
        }

        // im_info: [height, width, scale]
        let im_info = Array2::<f32>::from_shape_vec(
            (1, 3),
            vec![new_h as f32, new_w as f32, scale],
        )?;

        // featuremap_cond: always false
        let featuremap_cond = Array1::<bool>::from_vec(vec![false]);

        Ok(PreprocessResult {
            data,
            im_info,
            featuremap_cond,
            scale,
            orig_size: (orig_h, orig_w),
            new_size: (new_h, new_w),
        })
    }

    /// Extract FPN outputs from session outputs into owned data
    fn extract_fpn_outputs(outputs: &SessionOutputs) -> Result<Vec<FpnLevelOutput>> {
        let mut fpn_outputs = Vec::new();

        for level in FPN_LEVELS {
            let hori_key = format!("scores_hori_{}", level);
            let vert_key = format!("scores_vert_{}", level);

            if let (Ok(hori), Ok(vert)) = (
                outputs[hori_key.as_str()].try_extract_tensor::<f32>(),
                outputs[vert_key.as_str()].try_extract_tensor::<f32>(),
            ) {
                let (hori_shape, hori_data) = hori;
                let (vert_shape, vert_data) = vert;

                // Shape is [1, 1, H, W]
                fpn_outputs.push(FpnLevelOutput {
                    hori_data: hori_data.to_vec(),
                    hori_h: hori_shape[2] as u32,
                    hori_w: hori_shape[3] as u32,
                    vert_data: vert_data.to_vec(),
                    vert_h: vert_shape[2] as u32,
                    vert_w: vert_shape[3] as u32,
                });
            }
        }

        Ok(fpn_outputs)
    }

    /// Create text mask from extracted FPN outputs
    fn create_text_mask(
        &self,
        fpn_outputs: &[FpnLevelOutput],
        orig_size: (u32, u32),
        new_size: (u32, u32),
    ) -> Result<Vec<u8>> {
        let (orig_h, orig_w) = orig_size;
        let (new_h, new_w) = new_size;

        // Combine heatmaps from all FPN levels
        let mut heatmap = vec![0.0f32; (new_h * new_w) as usize];

        for fpn in fpn_outputs {
            // Resize horizontal scores and combine
            self.resize_and_combine_heatmap(
                &fpn.hori_data,
                fpn.hori_h,
                fpn.hori_w,
                new_h,
                new_w,
                &mut heatmap,
            );
            // Resize vertical scores and combine
            self.resize_and_combine_heatmap(
                &fpn.vert_data,
                fpn.vert_h,
                fpn.vert_w,
                new_h,
                new_w,
                &mut heatmap,
            );
        }

        // Apply threshold and create binary mask at model resolution
        let mut mask_small: Vec<u8> = heatmap
            .iter()
            .map(|&v| if v > self.threshold { 255 } else { 0 })
            .collect();

        // Dilate mask to ensure complete text coverage
        if self.dilate_iterations > 0 {
            mask_small = self.dilate_mask(
                &mask_small,
                new_w as usize,
                new_h as usize,
                self.dilate_kernel_size as usize,
                self.dilate_iterations as usize,
            );
        }

        // Resize to original image size
        let mask_img = GrayImage::from_raw(new_w, new_h, mask_small)
            .context("Failed to create mask image")?;

        let resized = image::imageops::resize(
            &mask_img,
            orig_w,
            orig_h,
            image::imageops::FilterType::Nearest,
        );

        Ok(resized.into_raw())
    }

    /// Resize heatmap and combine using max operation
    fn resize_and_combine_heatmap(
        &self,
        src: &[f32],
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
        dst: &mut [f32],
    ) {
        // Simple bilinear resize and max combine
        let scale_y = src_h as f32 / dst_h as f32;
        let scale_x = src_w as f32 / dst_w as f32;

        for y in 0..dst_h as usize {
            for x in 0..dst_w as usize {
                let src_y = (y as f32 * scale_y).min((src_h - 1) as f32);
                let src_x = (x as f32 * scale_x).min((src_w - 1) as f32);

                let y0 = src_y.floor() as usize;
                let x0 = src_x.floor() as usize;
                let y1 = (y0 + 1).min(src_h as usize - 1);
                let x1 = (x0 + 1).min(src_w as usize - 1);

                let fy = src_y.fract();
                let fx = src_x.fract();

                // Bilinear interpolation
                let v00 = src[y0 * src_w as usize + x0];
                let v01 = src[y0 * src_w as usize + x1];
                let v10 = src[y1 * src_w as usize + x0];
                let v11 = src[y1 * src_w as usize + x1];

                let value = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                let idx = y * dst_w as usize + x;
                dst[idx] = dst[idx].max(value);
            }
        }
    }

    /// Dilate binary mask to expand text regions
    fn dilate_mask(
        &self,
        mask: &[u8],
        width: usize,
        height: usize,
        kernel_size: usize,
        iterations: usize,
    ) -> Vec<u8> {
        let mut current = mask.to_vec();
        let mut next = vec![0u8; width * height];
        let half_k = kernel_size / 2;

        for _ in 0..iterations {
            for y in 0..height {
                for x in 0..width {
                    let mut max_val = 0u8;

                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let ny = y as isize + ky as isize - half_k as isize;
                            let nx = x as isize + kx as isize - half_k as isize;

                            if ny >= 0 && ny < height as isize && nx >= 0 && nx < width as isize {
                                let idx = ny as usize * width + nx as usize;
                                max_val = max_val.max(current[idx]);
                            }
                        }
                    }

                    next[y * width + x] = max_val;
                }
            }
            std::mem::swap(&mut current, &mut next);
        }

        current
    }

    /// Generate segmentation mask for an image
    ///
    /// Returns Vec<u8> where 255 = text region, 0 = background
    #[instrument(skip(self, img), fields(w = img.width(), h = img.height()))]
    pub async fn generate_mask(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();
        let target_size = self.config.target_size();

        // Preprocess
        let prep = self.preprocess(img, target_size)?;

        // Create ONNX values
        let data_value = Value::from_array(prep.data)?;
        let im_info_value = Value::from_array(prep.im_info)?;
        let featuremap_cond_value = Value::from_array(prep.featuremap_cond)?;

        // Run inference and extract outputs inside the session block
        let fpn_outputs = {
            let mut session = self.session.lock();
            let outputs = session.run(ort::inputs![
                "data" => data_value,
                "im_info" => im_info_value,
                "featuremap_cond" => featuremap_cond_value
            ])?;
            Self::extract_fpn_outputs(&outputs)?
        };

        // Create mask from extracted outputs
        let mask = self.create_text_mask(&fpn_outputs, prep.orig_size, prep.new_size)?;

        debug!(
            "Text segmentation done in {:.0}ms ({}x{})",
            start.elapsed().as_secs_f64() * 1000.0,
            img.width(),
            img.height()
        );

        Ok(mask)
    }
}

/// Preprocessing result struct
struct PreprocessResult {
    data: Array4<f32>,
    im_info: Array2<f32>,
    featuremap_cond: Array1<bool>,
    #[allow(dead_code)]
    scale: f32,
    orig_size: (u32, u32),
    new_size: (u32, u32),
}

/// Extracted FPN level output with owned data
struct FpnLevelOutput {
    hori_data: Vec<f32>,
    hori_h: u32,
    hori_w: u32,
    vert_data: Vec<f32>,
    vert_h: u32,
    vert_w: u32,
}
