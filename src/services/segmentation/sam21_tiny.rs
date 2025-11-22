// SAM 2.1-tiny segmentation service for high-accuracy edge detection
// Uses two-model pipeline: encoder (preprocessor) + decoder (mask generator)

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use ort::session::Session;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

use crate::core::config::Config;
use crate::services::onnx_builder::{self, DynamicSessionPool};

// Embed SAM models at compile time
// If LFS not checked out, these will be small stubs - load from runtime path instead
static SAM_ENCODER_BYTES: &[u8] = include_bytes!("../../../models/sam2.1_tiny_preprocess.onnx");
static SAM_DECODER_BYTES: &[u8] = include_bytes!("../../../models/sam2.1_tiny.onnx");

/// Load encoder model bytes from embedded or runtime path
fn load_encoder_bytes(config: &Config) -> Result<Vec<u8>> {
    // Check if embedded model is real (>10KB means it's not an LFS stub)
    if SAM_ENCODER_BYTES.len() > 10_000 {
        debug!("Using embedded SAM encoder model ({:.1} MB)", SAM_ENCODER_BYTES.len() as f64 / 1_048_576.0);
        Ok(SAM_ENCODER_BYTES.to_vec())
    } else {
        // Load from runtime path
        let path = &config.detection.sam_encoder_model_path;
        debug!("Loading SAM encoder model from: {}", path);
        std::fs::read(path)
            .with_context(|| format!("Failed to load SAM encoder model from {}", path))
    }
}

/// Load decoder model bytes from embedded or runtime path
fn load_decoder_bytes(config: &Config) -> Result<Vec<u8>> {
    // Check if embedded model is real (>10KB means it's not an LFS stub)
    if SAM_DECODER_BYTES.len() > 10_000 {
        debug!("Using embedded SAM decoder model ({:.1} MB)", SAM_DECODER_BYTES.len() as f64 / 1_048_576.0);
        Ok(SAM_DECODER_BYTES.to_vec())
    } else {
        // Load from runtime path
        let path = &config.detection.sam_decoder_model_path;
        debug!("Loading SAM decoder model from: {}", path);
        std::fs::read(path)
            .with_context(|| format!("Failed to load SAM decoder model from {}", path))
    }
}

/// SAM 2.1-tiny segmentation service for high-accuracy edge detection
///
/// Architecture:
/// - Encoder (129MB): Processes 1024x1024 crops -> embeddings
/// - Decoder (20MB): Embeddings + prompts -> masks
///
/// Per-bubble processing with 25% bbox padding for edge accuracy
pub struct Sam21TinyService {
    encoder_pool: Arc<DynamicSessionPool>,
    decoder_pool: Arc<DynamicSessionPool>,
    config: Arc<Config>,
    device_type: String,
}

/// Region detection info needed for SAM prompting
#[derive(Debug, Clone)]
pub struct RegionDetection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub label: u8,
    pub confidence: f32,
}

/// Embeddings from encoder to pass to decoder
#[derive(Clone)]
struct SamEmbeddings {
    image_embeddings: Array4<f32>,      // [1, 256, 64, 64]
    high_res_features1: Array4<f32>,    // [1, 32, 256, 256]
    high_res_features2: Array4<f32>,    // [1, 64, 128, 128]
}

impl Sam21TinyService {
    /// Create new SAM service with lazy initialization
    #[instrument(skip(config))]
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        info!("🚀 Creating SAM 2.1-tiny service with lazy session initialization");

        // Initialize encoder
        let (device_type, encoder_session) = Self::initialize_encoder(&config)?;
        let encoder_pool = Arc::new(DynamicSessionPool::new(8)); // Max 8 encoder sessions
        encoder_pool.add_session(encoder_session);

        // Initialize decoder
        let (_decoder_device, decoder_session) = Self::initialize_decoder(&config)?;
        let decoder_pool = Arc::new(DynamicSessionPool::new(8)); // Max 8 decoder sessions
        decoder_pool.add_session(decoder_session);

        let service = Self {
            encoder_pool,
            decoder_pool,
            config,
            device_type: device_type.clone(),
        };

        info!("✓ SAM 2.1-tiny: {} (encoder: 1/8, decoder: 1/8, ~155MB per pair)", device_type);

        Ok(service)
    }

    /// Initialize encoder session
    fn initialize_encoder(config: &Config) -> Result<(String, Session)> {
        let model_bytes = load_encoder_bytes(config)?;

        info!("Loading SAM encoder model ({:.1} MB)", model_bytes.len() as f64 / 1_048_576.0);

        let (backend, session) = onnx_builder::build_session_with_acceleration(
            &model_bytes,
            "sam-encoder",
            model_bytes.len() as f32 / 1_048_576.0
        )?;

        Ok((backend, session))
    }

    /// Initialize decoder session
    fn initialize_decoder(config: &Config) -> Result<(String, Session)> {
        let model_bytes = load_decoder_bytes(config)?;

        info!("Loading SAM decoder model ({:.1} MB)", model_bytes.len() as f64 / 1_048_576.0);

        let (backend, session) = onnx_builder::build_session_with_acceleration(
            &model_bytes,
            "sam-decoder",
            model_bytes.len() as f32 / 1_048_576.0
        )?;

        Ok((backend, session))
    }

    /// Generate segmentation mask for image with detected regions
    ///
    /// # Arguments
    /// * `img` - Full input image
    /// * `regions` - Detected regions from RT-DETR (label 0 = bubbles)
    ///
    /// # Returns
    /// Flattened binary mask (0 or 255) of shape (h*w)
    #[instrument(skip(self, img, regions), fields(width = img.width(), height = img.height(), num_regions = regions.len()))]
    pub async fn generate_mask(
        &self,
        img: &DynamicImage,
        regions: &[RegionDetection],
    ) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();
        let (orig_width, orig_height) = img.dimensions();

        debug!("SAM: Processing {} regions on {}x{} image", regions.len(), orig_width, orig_height);

        // Filter for label 0 regions (speech bubbles) only
        let bubbles: Vec<&RegionDetection> = regions
            .iter()
            .filter(|r| r.label == 0)
            .collect();

        if bubbles.is_empty() {
            debug!("No label 0 (bubble) regions found, returning empty mask");
            return Ok(vec![0u8; (orig_width * orig_height) as usize]);
        }

        info!("SAM: Processing {} bubbles (filtered from {} total regions)", bubbles.len(), regions.len());

        // Initialize combined mask
        let mut combined_mask = vec![0u8; (orig_width * orig_height) as usize];
        let mut processed_count = 0;
        let mut skipped_count = 0;

        // Process each bubble separately
        for (idx, bubble) in bubbles.iter().enumerate() {
            match self.process_single_bubble(img, bubble).await {
                Ok(Some(region_mask)) => {
                    // Merge this bubble's mask into combined mask
                    for (i, &val) in region_mask.iter().enumerate() {
                        if val > 127 {
                            combined_mask[i] = 255;
                        }
                    }
                    processed_count += 1;

                    if processed_count == 1 {
                        debug!("SAM: Successfully processed first bubble (#{}/{})", idx + 1, bubbles.len());
                    }
                }
                Ok(None) => {
                    skipped_count += 1;
                    debug!("SAM: Skipped bubble #{}/{} (low confidence)", idx + 1, bubbles.len());
                }
                Err(e) => {
                    skipped_count += 1;
                    warn!("SAM: Failed to process bubble #{}/{}: {}", idx + 1, bubbles.len(), e);
                }
            }
        }

        info!(
            "SAM: Completed in {:.2}ms - processed {}/{} bubbles ({} skipped)",
            start.elapsed().as_secs_f64() * 1000.0,
            processed_count,
            bubbles.len(),
            skipped_count
        );

        Ok(combined_mask)
    }

    /// Process a single bubble region with SAM
    ///
    /// Returns None if mask confidence is too low (skip bubble)
    async fn process_single_bubble(
        &self,
        img: &DynamicImage,
        bubble: &RegionDetection,
    ) -> Result<Option<Vec<u8>>> {
        let (img_width, img_height) = img.dimensions();

        // Apply 25% padding to bbox (reference: onnx_pipeline.py lines 29-67)
        let (padded_crop, crop_x1, crop_y1, crop_w, crop_h) =
            self.extract_padded_crop(img, bubble, img_width, img_height)?;

        // Run encoder on 1024x1024 crop
        let embeddings = self.run_encoder(&padded_crop).await?;

        // Generate 15-point grid prompts (reference: lines 88-97)
        let (point_coords, point_labels) = self.generate_grid_prompts();

        // Run decoder with embeddings + prompts
        let (masks, iou_scores) = self.run_decoder(
            &embeddings,
            &point_coords,
            &point_labels,
            crop_w,
            crop_h,
        ).await?;

        // Select best mask based on IoU score (reference: line 121)
        let best_idx = iou_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let best_iou = iou_scores[best_idx];

        // Skip if confidence too low
        if best_iou < 0.5 {
            debug!("SAM: Skipping bubble with low IoU score: {:.3}", best_iou);
            return Ok(None);
        }

        // Extract and process best mask
        let mask = self.extract_and_process_mask(
            &masks,
            best_idx,
            crop_w,
            crop_h,
            img_width,
            img_height,
            crop_x1,
            crop_y1,
        )?;

        Ok(Some(mask))
    }

    /// Extract padded crop from image (25% padding around bbox)
    fn extract_padded_crop(
        &self,
        img: &DynamicImage,
        bubble: &RegionDetection,
        img_width: u32,
        img_height: u32,
    ) -> Result<(DynamicImage, u32, u32, u32, u32)> {
        let padding = 0.25; // 25% padding as in reference

        let w = bubble.x2 - bubble.x1;
        let h = bubble.y2 - bubble.y1;
        let pad_w = (w * padding) as u32;
        let pad_h = (h * padding) as u32;

        // Clamp to image bounds
        let x1 = (bubble.x1 as i32 - pad_w as i32).max(0) as u32;
        let y1 = (bubble.y1 as i32 - pad_h as i32).max(0) as u32;
        let x2 = ((bubble.x2 as u32 + pad_w).min(img_width)) as u32;
        let y2 = ((bubble.y2 as u32 + pad_h).min(img_height)) as u32;

        let crop_w = x2 - x1;
        let crop_h = y2 - y1;

        // Crop image
        let crop = img.crop_imm(x1, y1, crop_w, crop_h);

        Ok((crop, x1, y1, crop_w, crop_h))
    }

    /// Generate 15-point grid prompts (5x3 grid centered in 1024x1024 space)
    /// Reference: onnx_pipeline.py lines 88-97
    fn generate_grid_prompts(&self) -> (Array3<f32>, Array2<f32>) {
        let mut points = Vec::new();

        // 5 vertical positions x 3 horizontal positions = 15 points
        for y_off in [-0.2, -0.1, 0.0, 0.1, 0.2] {
            for x_off in [-0.15, 0.0, 0.15] {
                let px = 512.0 + (512.0 * x_off);
                let py = 512.0 + (512.0 * y_off);
                points.push([px, py]);
            }
        }

        // All points labeled as foreground (1)
        let point_coords = Array3::from_shape_vec(
            (1, 15, 2),
            points.into_iter().flatten().collect(),
        ).expect("Failed to create point coords array");

        let point_labels = Array2::from_shape_vec(
            (1, 15),
            vec![1.0f32; 15],
        ).expect("Failed to create point labels array");

        (point_coords, point_labels)
    }

    /// Run encoder on 1024x1024 crop
    async fn run_encoder(&self, crop: &DynamicImage) -> Result<SamEmbeddings> {
        // Resize to 1024x1024
        let resized = crop.resize_exact(
            1024,
            1024,
            image::imageops::FilterType::Triangle,
        );

        // Convert to RGB and normalize [0, 1], NCHW format
        let rgb = resized.to_rgb8();
        let mut input_array = Array4::<f32>::zeros((1, 3, 1024, 1024));

        for (x, y, pixel) in rgb.enumerate_pixels() {
            input_array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            input_array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            input_array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }

        let input_value = ort::value::Value::from_array(input_array)?;

        // Acquire encoder session and run
        let mut session = self.encoder_pool.acquire();
        let outputs = session.run(ort::inputs!["input" => input_value])?;

        // Extract embeddings (3 output tensors)
        let (_, img_emb_data) = outputs["image_embeddings"]
            .try_extract_tensor::<f32>()?;
        let image_embeddings = Array4::from_shape_vec(
            (1, 256, 64, 64),
            img_emb_data.to_vec(),
        )?;

        let (_, hr1_data) = outputs["high_res_features_0"]
            .try_extract_tensor::<f32>()?;
        let high_res_features1 = Array4::from_shape_vec(
            (1, 32, 256, 256),
            hr1_data.to_vec(),
        )?;

        let (_, hr2_data) = outputs["high_res_features_1"]
            .try_extract_tensor::<f32>()?;
        let high_res_features2 = Array4::from_shape_vec(
            (1, 64, 128, 128),
            hr2_data.to_vec(),
        )?;

        drop(outputs);
        self.encoder_pool.release(session);

        Ok(SamEmbeddings {
            image_embeddings,
            high_res_features1,
            high_res_features2,
        })
    }

    /// Run decoder with embeddings and prompts
    async fn run_decoder(
        &self,
        embeddings: &SamEmbeddings,
        point_coords: &Array3<f32>,
        point_labels: &Array2<f32>,
        orig_crop_w: u32,
        orig_crop_h: u32,
    ) -> Result<(Array4<f32>, Vec<f32>)> {
        // Prepare decoder inputs
        let img_emb_value = ort::value::Value::from_array(embeddings.image_embeddings.clone())?;
        let hr1_value = ort::value::Value::from_array(embeddings.high_res_features1.clone())?;
        let hr2_value = ort::value::Value::from_array(embeddings.high_res_features2.clone())?;
        let coords_value = ort::value::Value::from_array(point_coords.clone())?;
        let labels_value = ort::value::Value::from_array(point_labels.clone())?;

        // Empty mask input (first iteration)
        let mask_input = Array4::<f32>::zeros((1, 1, 256, 256));
        let mask_input_value = ort::value::Value::from_array(mask_input)?;

        // has_mask_input = 0 (no previous mask)
        let has_mask = Array1::<f32>::from_vec(vec![0.0]);
        let has_mask_value = ort::value::Value::from_array(has_mask)?;

        // Original crop size
        let orig_size = Array1::<i64>::from_vec(vec![orig_crop_h as i64, orig_crop_w as i64]);
        let orig_size_value = ort::value::Value::from_array(orig_size)?;

        // Acquire decoder session and run
        let mut session = self.decoder_pool.acquire();
        let outputs = session.run(ort::inputs![
            "image_embeddings" => img_emb_value,
            "high_res_features_0" => hr1_value,
            "high_res_features_1" => hr2_value,
            "point_coords" => coords_value,
            "point_labels" => labels_value,
            "mask_input" => mask_input_value,
            "has_mask_input" => has_mask_value,
            "orig_im_size" => orig_size_value,
        ])?;

        // Extract masks and IoU predictions
        let (masks_shape, masks_data) = outputs["masks"]
            .try_extract_tensor::<f32>()?;

        // Masks shape: [1, 4, H, W] where H,W are original crop dimensions
        let shape_slice: &[i64] = masks_shape.as_ref();
        let masks = Array4::from_shape_vec(
            (shape_slice[0] as usize, shape_slice[1] as usize, shape_slice[2] as usize, shape_slice[3] as usize),
            masks_data.to_vec(),
        )?;

        let (_, iou_data) = outputs["iou_predictions"]
            .try_extract_tensor::<f32>()?;
        let iou_scores = iou_data.to_vec();

        drop(outputs);
        self.decoder_pool.release(session);

        Ok((masks, iou_scores))
    }

    /// Extract best mask and apply morphological cleanup
    fn extract_and_process_mask(
        &self,
        masks: &Array4<f32>,
        best_idx: usize,
        _crop_w: u32,
        _crop_h: u32,
        img_width: u32,
        img_height: u32,
        crop_x1: u32,
        crop_y1: u32,
    ) -> Result<Vec<u8>> {
        // Get mask shape [1, 4, H, W]
        let mask_shape = masks.shape();
        let mask_h = mask_shape[2];
        let mask_w = mask_shape[3];

        // Extract best mask (batch 0, channel best_idx)
        let batch_0 = masks.index_axis(Axis(0), 0);
        let mask_slice = batch_0.index_axis(Axis(0), best_idx);

        // Binarize mask (threshold at 0.0 as in reference)
        let mut binary_mask = vec![0u8; mask_h * mask_w];
        for (i, &val) in mask_slice.iter().enumerate() {
            binary_mask[i] = if val > 0.0 { 255 } else { 0 };
        }

        // Apply morphological closing (3x3 kernel, 2 iterations)
        // Reference: onnx_pipeline.py lines 129-131
        let mask_img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(
            mask_w as u32,
            mask_h as u32,
            binary_mask,
        ).context("Failed to create mask image")?;

        let cleaned = self.morphological_close(mask_img, 2);

        // Map crop mask back to full image coordinates
        let mut full_mask = vec![0u8; (img_width * img_height) as usize];

        for y in 0..mask_h {
            for x in 0..mask_w {
                let mask_val = cleaned.get_pixel(x as u32, y as u32)[0];
                if mask_val > 127 {
                    // Map to full image coordinates
                    let full_x = crop_x1 + x as u32;
                    let full_y = crop_y1 + y as u32;

                    if full_x < img_width && full_y < img_height {
                        let idx = (full_y * img_width + full_x) as usize;
                        full_mask[idx] = 255;
                    }
                }
            }
        }

        Ok(full_mask)
    }

    /// Apply morphological closing (dilate then erode) for cleanup
    /// Uses 3x3 kernel as in reference
    fn morphological_close(
        &self,
        img: ImageBuffer<Luma<u8>, Vec<u8>>,
        iterations: u32,
    ) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut result = img;

        for _ in 0..iterations {
            // Dilate
            result = self.dilate(&result);
            // Erode
            result = self.erode(&result);
        }

        result
    }

    /// Simple 3x3 dilation
    fn dilate(&self, img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let (width, height) = img.dimensions();
        let mut output = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut max_val = 0u8;

                // 3x3 kernel
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx).max(0).min(width as i32 - 1) as u32;
                        let ny = (y as i32 + dy).max(0).min(height as i32 - 1) as u32;
                        max_val = max_val.max(img.get_pixel(nx, ny)[0]);
                    }
                }

                output.put_pixel(x, y, Luma([max_val]));
            }
        }

        output
    }

    /// Simple 3x3 erosion
    fn erode(&self, img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let (width, height) = img.dimensions();
        let mut output = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut min_val = 255u8;

                // 3x3 kernel
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx).max(0).min(width as i32 - 1) as u32;
                        let ny = (y as i32 + dy).max(0).min(height as i32 - 1) as u32;
                        min_val = min_val.min(img.get_pixel(nx, ny)[0]);
                    }
                }

                output.put_pixel(x, y, Luma([min_val]));
            }
        }

        output
    }
}
