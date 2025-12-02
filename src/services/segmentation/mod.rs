// Text Cleaner Service - FPN-based text detection and removal
//
// Uses FPN text detection model to identify text regions and fill with white.
// OCR validation: Only cleans regions where OCR confirms actual text exists.
// This prevents over-cleaning of art/characters that text detector misidentifies.
//
// OPTIMIZATION: Uses DynamicSessionPool for parallel inference (like DetectionService).
// Multiple regions can be processed simultaneously with separate ONNX sessions.

use anyhow::{Context, Result};
use image::{DynamicImage, Rgba};
use ndarray::{Array1, Array2, Array4};
use ort::{session::{Session, SessionOutputs}, value::Value};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

use crate::core::config::Config;
use crate::services::onnx_builder::DynamicSessionPool;

/// Embedded text cleaner model bytes (only for local builds, not CI)
#[cfg(text_cleaner_model_embedded)]
const TEXT_CLEANER_MODEL_BYTES: &[u8] = include_bytes!("../../../models/text_cleaner.onnx");

/// Placeholder for CI builds (model loaded from disk)
#[cfg(not(text_cleaner_model_embedded))]
const TEXT_CLEANER_MODEL_BYTES: &[u8] = &[];

/// FPN output levels for text detection
const FPN_LEVELS: [&str; 3] = ["fpn2", "fpn3", "fpn4"];

/// Text Cleaner Service using FPN-based text detection
///
/// This uses an FPN text detector which outputs horizontal and vertical
/// text score maps at multiple scales. Text regions are filled with white.
///
/// OPTIMIZATION: Uses DynamicSessionPool for parallel inference across regions.
pub struct SegmentationService {
    session_pool: Arc<DynamicSessionPool>,
    config: Arc<Config>,
    max_sessions: usize,
    /// Detection threshold (0.0-1.0, lower = more aggressive detection)
    threshold: f32,
    /// Dilation iterations to expand mask coverage
    dilate_iterations: u32,
    /// Dilation kernel size
    dilate_kernel_size: u32,
}

impl SegmentationService {
    /// Create new text cleaner service with parallel session pool
    /// First tries embedded model (local builds), then falls back to disk (CI builds)
    #[instrument(skip(config))]
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        Self::new_with_limit(config, None).await
    }

    /// Create new text cleaner service with custom models directory
    pub async fn new_with_models_dir(config: Arc<Config>, models_dir: &Path) -> Result<Self> {
        Self::new_with_models_dir_and_limit(config, models_dir, None).await
    }

    /// Create with optional session limit override
    pub async fn new_with_limit(config: Arc<Config>, session_limit: Option<usize>) -> Result<Self> {
        Self::new_with_models_dir_and_limit(config, Path::new("models"), session_limit).await
    }

    /// Create text cleaner service with custom models directory and session limit
    pub async fn new_with_models_dir_and_limit(
        config: Arc<Config>,
        models_dir: &Path,
        session_limit: Option<usize>,
    ) -> Result<Self> {
        let max_sessions = session_limit.unwrap_or_else(|| config.onnx_pool_size());
        
        info!("🚀 Initializing text cleaner service (FPN text detector, {} parallel sessions)", max_sessions);
        let init_start = std::time::Instant::now();

        let model_path = models_dir.join("text_cleaner.onnx");
        
        // Get threshold from config or environment variable
        let threshold = std::env::var("TEXT_CLEANER_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.3);  // Default 0.3

        // Determine model bytes source
        let model_bytes: Vec<u8> = if TEXT_CLEANER_MODEL_BYTES.len() > 100_000 {
            let model_size_mb = TEXT_CLEANER_MODEL_BYTES.len() as f64 / 1_048_576.0;
            info!("Loading text cleaner model from embedded bytes ({:.1} MB)", model_size_mb);
            TEXT_CLEANER_MODEL_BYTES.to_vec()
        } else {
            // Fall back to disk loading (CI builds or embedded not available)
            if !model_path.exists() {
                anyhow::bail!(
                    "Text cleaner model not found at: {}. Text cleaning feature is unavailable.",
                    model_path.display()
                );
            }

            let model_size = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
            if model_size < 100_000 {
                anyhow::bail!(
                    "Text cleaner model too small ({} bytes) at {} - check Git LFS",
                    model_size,
                    model_path.display()
                );
            }

            let model_size_mb = model_size as f64 / 1_048_576.0;
            info!("Loading text cleaner model from disk: {} ({:.1} MB)", model_path.display(), model_size_mb);
            std::fs::read(&model_path).context("Failed to read text cleaner model from disk")?
        };

        // Create session pool and pre-allocate all sessions upfront
        // text_cleaner.onnx is small (~11.5MB) so pre-allocation is memory-efficient
        let session_pool = Arc::new(DynamicSessionPool::new(max_sessions));
        
        info!("Pre-allocating {} text cleaner sessions...", max_sessions);
        for i in 0..max_sessions {
            let session = Self::create_session(&model_bytes)?;
            session_pool.add_session(session);
            if i == 0 {
                info!("  Session 1/{} created", max_sessions);
            }
        }

        info!(
            "✓ Text cleaner ready: {}/{} sessions pre-allocated, threshold={:.2} ({:.0}ms)",
            max_sessions, max_sessions, threshold,
            init_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(Self {
            session_pool,
            config,
            max_sessions,
            threshold,
            dilate_iterations: 2,  // Match Python: 2 iterations
            dilate_kernel_size: 3, // Match Python: 3x3 kernel
        })
    }

    /// Create an ONNX session from model bytes
    fn create_session(model_bytes: &[u8]) -> Result<Session> {
        use ort::session::builder::GraphOptimizationLevel;
        use ort::execution_providers::CPUExecutionProvider;
        
        Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_memory(model_bytes)
            .context("Failed to load text cleaner ONNX model")
    }

    /// Check if using DirectML (always false for CPU-only FPN model)
    pub fn is_directml(&self) -> bool {
        false
    }
    
    /// Get current detection threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Keep sessions alive - text cleaner sessions are pre-allocated and should persist across requests
    /// (Previously drained sessions which caused deadlock on second request)
    pub fn cleanup_sessions(&self) {
        // Keep sessions alive - draining would cause next request to block forever
        // Sessions are pre-allocated at startup for parallel processing
        debug!("Text cleaner: Keeping {} pre-allocated sessions alive", self.max_sessions);
    }
    
    /// Expand session pool if under capacity (lazy allocation for parallel processing)
    fn expand_pool_if_needed(&self, model_bytes: Option<&[u8]>) {
        let available = self.session_pool.available();
        let in_use = self.session_pool.in_use();
        let total = available + in_use;
        
        // Create more sessions if under pressure
        let sessions_to_create = if available == 0 && total < self.max_sessions {
            let headroom = self.max_sessions - total;
            std::cmp::min(2, headroom)  // Create up to 2 at a time
        } else {
            0
        };
        
        if sessions_to_create > 0 {
            debug!("🔄 Expanding text cleaner pool: {}/{} sessions, creating {} new", total, self.max_sessions, sessions_to_create);
            
            // Get model bytes - try embedded first, then fall back to disk
            let bytes_owned;
            let bytes = if let Some(b) = model_bytes {
                b
            } else if TEXT_CLEANER_MODEL_BYTES.len() > 100_000 {
                TEXT_CLEANER_MODEL_BYTES
            } else {
                // Try loading from disk
                match std::fs::read("models/text_cleaner.onnx") {
                    Ok(b) => {
                        bytes_owned = b;
                        &bytes_owned
                    }
                    Err(_) => return, // Can't expand without model
                }
            };
            
            for _ in 0..sessions_to_create {
                if let Ok(session) = Self::create_session(bytes) {
                    self.session_pool.add_session(session);
                }
            }
            
            info!("✓ Expanded text cleaner pool: {}/{} sessions", total + sessions_to_create, self.max_sessions);
        }
    }

    /// Clean text from a cropped region
    ///
    /// Takes a cropped image region, detects text, and fills text areas with white.
    /// Returns the cleaned image as PNG bytes.
    ///
    /// # Arguments
    /// * `crop` - Cropped image region (label 1 or label 2 bbox)
    ///
    /// # Returns
    /// PNG bytes of the cleaned region
    #[instrument(skip(self, crop), fields(w = crop.width(), h = crop.height()))]
    pub async fn clean_region(&self, crop: &DynamicImage) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        // Skip tiny regions - just return original as PNG
        if crop.width() < 16 || crop.height() < 16 {
            return self.encode_png(crop);
        }

        // Generate mask for text areas
        let mask = self.generate_mask_internal(crop).await?;

        // Apply mask: fill text areas with white
        let mut rgba = crop.to_rgba8();
        let (w, h) = rgba.dimensions();

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if idx < mask.len() && mask[idx] > 0 {
                    rgba.put_pixel(x, y, Rgba([255, 255, 255, 255]));
                }
            }
        }

        debug!(
            "Text cleaning done in {:.0}ms ({}x{})",
            start.elapsed().as_secs_f64() * 1000.0,
            w, h
        );

        self.encode_png(&DynamicImage::ImageRgba8(rgba))
    }

    /// Clean multiple regions and return cleaned PNGs with their bboxes
    ///
    /// Takes list of (region_id, cropped_image, bbox) and returns list of (region_id, cleaned_png_bytes, bbox)
    #[instrument(skip(self, regions), fields(count = regions.len()))]
    pub async fn clean_regions_batch(
        &self,
        regions: &[(usize, DynamicImage, [i32; 4])],
    ) -> Result<Vec<(usize, Vec<u8>, [i32; 4])>> {
        // Clean all regions (no filtering)
        self.clean_regions_batch_filtered(regions, None).await
    }

    /// Clean multiple regions with optional filtering by valid region IDs
    ///
    /// OPTIMIZATION: Uses parallel processing with session pool.
    /// Multiple regions are cleaned concurrently up to max_sessions limit.
    ///
    /// Only regions with IDs in `valid_region_ids` are cleaned.
    /// Regions not in the set are returned as-is (original image, not cleaned).
    /// If `valid_region_ids` is None, all regions are cleaned.
    ///
    /// # Arguments
    /// * `regions` - List of (region_id, cropped_image, bbox)
    /// * `valid_region_ids` - Optional set of region IDs that should be cleaned.
    ///   Regions not in this set will be returned without text removal.
    #[instrument(skip(self, regions, valid_region_ids), fields(count = regions.len()))]
    pub async fn clean_regions_batch_filtered(
        &self,
        regions: &[(usize, DynamicImage, [i32; 4])],
        valid_region_ids: Option<&std::collections::HashSet<usize>>,
    ) -> Result<Vec<(usize, Vec<u8>, [i32; 4])>> {
        let start = std::time::Instant::now();
        
        // Separate regions into those to clean and those to skip
        let mut to_clean: Vec<(usize, DynamicImage, [i32; 4])> = Vec::new();
        let mut to_skip: Vec<(usize, &DynamicImage, [i32; 4])> = Vec::new();
        
        for (region_id, crop, bbox) in regions {
            let should_clean = valid_region_ids
                .map(|ids| ids.contains(region_id))
                .unwrap_or(true);
            
            if should_clean {
                to_clean.push((*region_id, crop.clone(), *bbox));
            } else {
                to_skip.push((*region_id, crop, *bbox));
            }
        }
        
        let cleaned_count = to_clean.len();
        let skipped_count = to_skip.len();
        
        // Pre-expand pool for parallel processing
        if cleaned_count > 1 {
            self.expand_pool_if_needed(None);
        }
        
        // Use rayon for TRUE parallel CPU-bound processing
        // Each thread acquires its own session from the pool
        use rayon::prelude::*;
        
        let cleaned_results: Vec<_> = to_clean
            .into_par_iter()
            .filter_map(|(region_id, crop, bbox)| {
                // Run mask generation synchronously (blocking is OK in rayon thread)
                let mask_result = self.generate_mask_sync(&crop);
                
                match mask_result {
                    Ok(mask) => {
                        let mut rgba = crop.to_rgba8();
                        let (w, h) = rgba.dimensions();
                        for y in 0..h {
                            for x in 0..w {
                                let idx = (y * w + x) as usize;
                                if idx < mask.len() && mask[idx] > 0 {
                                    rgba.put_pixel(x, y, Rgba([255, 255, 255, 255]));
                                }
                            }
                        }
                        match self.encode_png(&DynamicImage::ImageRgba8(rgba)) {
                            Ok(png_bytes) => Some((region_id, png_bytes, bbox)),
                            Err(e) => {
                                warn!("Failed to encode region {}: {:?}", region_id, e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to clean region {}: {:?}", region_id, e);
                        None
                    }
                }
            })
            .collect();
        
        // Encode skipped regions (just PNG encode, no cleaning)
        let mut skipped_results: Vec<(usize, Vec<u8>, [i32; 4])> = Vec::with_capacity(skipped_count);
        for (region_id, crop, bbox) in to_skip {
            debug!("Skipping cleaning for region {} (not in valid set)", region_id);
            if let Ok(bytes) = self.encode_png(crop) {
                skipped_results.push((region_id, bytes, bbox));
            }
        }
        
        // Combine results in original order
        let mut results = Vec::with_capacity(regions.len());
        results.extend(cleaned_results);
        results.extend(skipped_results);
        
        // Sort by region_id to maintain order
        results.sort_by_key(|(id, _, _)| *id);

        info!(
            "⏱️  Cleaning: {:.0}ms ({} cleaned, {} skipped, {} parallel sessions)",
            start.elapsed().as_secs_f64() * 1000.0,
            cleaned_count,
            skipped_count,
            self.max_sessions
        );

        Ok(results)
    }

    /// Generate mask for full image (legacy compatibility)
    #[instrument(skip(self, img), fields(w = img.width(), h = img.height()))]
    pub async fn generate_mask(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        self.generate_mask_internal(img).await
    }

    /// Generate mask for regions (legacy compatibility)
    pub async fn generate_mask_for_regions(
        &self,
        img: &DynamicImage,
        bboxes: &[[i32; 4]],
    ) -> Result<Vec<u8>> {
        let (img_w, img_h) = (img.width(), img.height());
        let mut full_mask = vec![0u8; (img_w * img_h) as usize];

        if bboxes.is_empty() {
            return Ok(full_mask);
        }

        for bbox in bboxes {
            let [x1, y1, x2, y2] = *bbox;
            let x1 = x1.max(0).min(img_w as i32) as u32;
            let y1 = y1.max(0).min(img_h as i32) as u32;
            let x2 = x2.max(0).min(img_w as i32) as u32;
            let y2 = y2.max(0).min(img_h as i32) as u32;

            if x2 <= x1 || y2 <= y1 || (x2 - x1) < 16 || (y2 - y1) < 16 {
                continue;
            }

            let cropped = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
            let crop_mask = self.generate_mask_internal(&cropped).await?;

            for cy in 0..(y2 - y1) {
                for cx in 0..(x2 - x1) {
                    let crop_idx = (cy * (x2 - x1) + cx) as usize;
                    let full_idx = ((y1 + cy) * img_w + (x1 + cx)) as usize;
                    if crop_idx < crop_mask.len() && full_idx < full_mask.len() {
                        full_mask[full_idx] = full_mask[full_idx].max(crop_mask[crop_idx]);
                    }
                }
            }
        }

        Ok(full_mask)
    }

    /// Encode image as PNG
    fn encode_png(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        let mut png_bytes = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
        img.write_with_encoder(encoder)
            .context("Failed to encode PNG")?;
        Ok(png_bytes)
    }

    /// Preprocess image for model input
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

        // Combine heatmaps from all FPN levels at model resolution
        let mut heatmap_small = vec![0.0f32; (new_h * new_w) as usize];

        for fpn in fpn_outputs {
            self.resize_and_combine_heatmap(
                &fpn.hori_data,
                fpn.hori_h,
                fpn.hori_w,
                new_h,
                new_w,
                &mut heatmap_small,
            );
            self.resize_and_combine_heatmap(
                &fpn.vert_data,
                fpn.vert_h,
                fpn.vert_w,
                new_h,
                new_w,
                &mut heatmap_small,
            );
        }

        // Resize heatmap to original size BEFORE thresholding (like Python)
        // This preserves more detail than thresholding then resizing binary mask
        let mut heatmap = vec![0.0f32; (orig_h * orig_w) as usize];
        self.resize_heatmap_bilinear(
            &heatmap_small,
            new_h,
            new_w,
            orig_h,
            orig_w,
            &mut heatmap,
        );

        // Apply threshold at original resolution
        let mut mask: Vec<u8> = heatmap
            .iter()
            .map(|&v| if v > self.threshold { 255 } else { 0 })
            .collect();

        // Dilate mask to ensure complete text coverage
        if self.dilate_iterations > 0 {
            mask = self.dilate_mask(
                &mask,
                orig_w as usize,
                orig_h as usize,
                self.dilate_kernel_size as usize,
                self.dilate_iterations as usize,
            );
        }

        Ok(mask)
    }

    /// Resize heatmap using bilinear interpolation (for upscaling before threshold)
    fn resize_heatmap_bilinear(
        &self,
        src: &[f32],
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
        dst: &mut [f32],
    ) {
        let scale_y = (src_h as f32 - 1.0) / (dst_h as f32 - 1.0).max(1.0);
        let scale_x = (src_w as f32 - 1.0) / (dst_w as f32 - 1.0).max(1.0);

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

                let v00 = src[y0 * src_w as usize + x0];
                let v01 = src[y0 * src_w as usize + x1];
                let v10 = src[y1 * src_w as usize + x0];
                let v11 = src[y1 * src_w as usize + x1];

                let value = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                dst[y * dst_w as usize + x] = value;
            }
        }
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

    /// Internal mask generation using session pool for parallel access (async version)
    async fn generate_mask_internal(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        self.generate_mask_sync(img)
    }
    
    /// Synchronous mask generation for use in rayon parallel iterators
    /// This is the actual implementation that acquires a session and runs inference
    fn generate_mask_sync(&self, img: &DynamicImage) -> Result<Vec<u8>> {
        // Expand pool if under pressure (safe to call from any thread)
        self.expand_pool_if_needed(None);
        
        let target_size = self.config.target_size();

        let prep = self.preprocess(img, target_size)?;

        let data_value = Value::from_array(prep.data)?;
        let im_info_value = Value::from_array(prep.im_info)?;
        let featuremap_cond_value = Value::from_array(prep.featuremap_cond)?;

        // Acquire session from pool (blocks if none available)
        // CRITICAL: Must release session even on error to avoid deadlock
        let mut session = self.session_pool.acquire();
        
        let result = (|| -> Result<_> {
            let outputs = session.run(ort::inputs![
                "data" => data_value,
                "im_info" => im_info_value,
                "featuremap_cond" => featuremap_cond_value
            ])?;
            Self::extract_fpn_outputs(&outputs)
        })();
        
        // Always release session back to pool (even on error)
        self.session_pool.release(session);
        
        let fpn_result = result?;

        self.create_text_mask(&fpn_result, prep.orig_size, prep.new_size)
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
