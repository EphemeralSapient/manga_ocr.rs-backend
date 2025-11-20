// Phase 4: Text Insertion Pipeline

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use rayon::prelude::*;
use std::sync::Arc;
use tracing::instrument;

use crate::core::config::Config;
use crate::services::rendering::CosmicTextRenderer;
use crate::core::types::{
    BananaResult, CategorizedRegion, ImageData, OCRTranslation, Phase1Output, Phase2Output,
    Phase3Output, Phase4Output,
};

/// Phase 4 pipeline: Text insertion
pub struct Phase4Pipeline {
    config: Arc<Config>,
    renderer: Arc<CosmicTextRenderer>,
}

// Font sizing constants for optimal comic readability
const MIN_FONT_SIZE: f32 = 18.0; // Minimum font size for readability
const MAX_FONT_SIZE: f32 = 32.0; // Maximum font size to prevent oversized text

impl Phase4Pipeline {
    /// Create new Phase 4 pipeline
    pub fn new(config: Arc<Config>) -> Self {
        let renderer = Arc::new(CosmicTextRenderer::new());
        Self { config, renderer }
    }

    /// Execute Phase 4: Insert translated text
    ///
    /// # Steps:
    /// 1. Load original image
    /// 2. For each region:
    ///    a. If banana-processed: Use banana result image directly
    ///    b. Otherwise: Use cleaned image from Phase 3 + render text from Phase 2
    /// 3. Composite all regions back into final image
    ///
    /// # Arguments:
    /// * `image_data` - Original image data
    /// * `phase1_output` - Detection/categorization
    /// * `phase2_output` - Translation results
    /// * `phase3_output` - Cleaned regions
    /// * `font_family` - Font family for rendering text (e.g., "arial", "comic-sans")
    /// * `text_stroke` - Whether to add stroke/outline to text
    ///
    /// # Returns:
    /// Phase4Output with final rendered image
    #[instrument(skip(self, image_data, phase1_output, phase2_output, phase3_output), fields(
        page_index = phase1_output.page_index,
        font_family = font_family
    ))]
    pub async fn execute(
        &self,
        image_data: &ImageData,
        phase1_output: &Phase1Output,
        phase2_output: &Phase2Output,
        phase3_output: &Phase3Output,
        font_family: &str,
        text_stroke: bool,
    ) -> Result<Phase4Output> {
        // Use pre-decoded image if available, otherwise load from bytes
        // OPTIMIZATION: Pre-decoded image eliminates redundant decoding across phases
        // Use Arc reference instead of cloning the entire image (saves ~8MB per phase!)
        let img_owned;
        let img: &DynamicImage = if let Some(ref decoded) = image_data.decoded_image {
            decoded.as_ref()
        } else {
            img_owned = image::load_from_memory(&image_data.image_bytes)
                .context("Failed to load image")?;
            &img_owned
        };
        let mut final_image = img.to_rgba8();

        // Create lookup maps for faster access (usize keys for zero-cost lookups)
        let mut simple_translations_map = std::collections::HashMap::new();
        for (region_id, translation) in &phase2_output.simple_bg_translations {
            simple_translations_map.insert(*region_id, translation);
        }

        let mut complex_translations_map = std::collections::HashMap::new();
        for (region_id, translation) in &phase2_output.complex_bg_translations {
            complex_translations_map.insert(*region_id, translation);
        }

        let mut banana_map = std::collections::HashMap::new();
        for banana in &phase2_output.complex_bg_bananas {
            banana_map.insert(banana.region_id, banana);
        }

        let mut cleaned_map = std::collections::HashMap::new();
        for (region_id, cleaned_bytes) in &phase3_output.cleaned_regions {
            cleaned_map.insert(*region_id, cleaned_bytes);
        }

        // Process each region
        for region in &phase1_output.regions {
            let region_id = region.region_id;

            // Check if this region was processed with banana
            if let Some(banana) = banana_map.get(&region_id) {
                self.composite_banana_result(&mut final_image, region, banana)?;
                continue;
            }

            // Otherwise, composite cleaned image + rendered text
            let translation = simple_translations_map
                .get(&region_id)
                .or_else(|| complex_translations_map.get(&region_id))
                .context(format!("No translation found for region {}", region_id))?;

            let cleaned_bytes = cleaned_map
                .get(&region_id)
                .context(format!("No cleaned image found for region {}", region_id))?;

            // Convert label_1_regions from absolute to local coordinates
            let [x1, y1, _, _] = region.bbox;
            let local_label_1_regions: Vec<[i32; 4]> = region
                .label_1_regions
                .iter()
                .map(|&[lx1, ly1, lx2, ly2]| {
                    [lx1 - x1, ly1 - y1, lx2 - x1, ly2 - y1]
                })
                .collect();

            self.composite_rendered_text(
                &mut final_image,
                region,
                cleaned_bytes,
                translation,
                &local_label_1_regions,
                font_family,
                text_stroke,
            )
            .await
            .context("Failed to composite rendered text")?;
        }

        // Convert final image to PNG bytes
        let mut png_bytes = Vec::new();
        DynamicImage::ImageRgba8(final_image)
            .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
            .context("Failed to encode final image")?;

        Ok(Phase4Output {
            page_index: phase1_output.page_index,
            final_image_bytes: png_bytes,
        })
    }

    /// Composite banana result directly onto final image
    fn composite_banana_result(
        &self,
        final_image: &mut RgbaImage,
        region: &CategorizedRegion,
        banana: &BananaResult,
    ) -> Result<()> {
        // Load banana result image
        let banana_img = image::load_from_memory(&banana.translated_image_bytes)
            .context("Failed to load banana result")?
            .to_rgba8();

        // Get region bbox
        let [x1, y1, x2, y2] = region.bbox;
        let width = (x2 - x1).max(1) as u32;
        let height = (y2 - y1).max(1) as u32;

        // Resize banana result to fit bbox if needed
        let banana_resized = if banana_img.width() != width || banana_img.height() != height {
            image::imageops::resize(
                &banana_img,
                width,
                height,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            banana_img
        };

        // Paste onto final image
        image::imageops::overlay(final_image, &banana_resized, x1 as i64, y1 as i64);

        Ok(())
    }

    /// Composite cleaned region + rendered text onto final image
    async fn composite_rendered_text(
        &self,
        final_image: &mut RgbaImage,
        region: &CategorizedRegion,
        cleaned_bytes: &[u8],
        translation: &OCRTranslation,
        local_label_1_regions: &[[i32; 4]],
        font_family: &str,
        text_stroke: bool,
    ) -> Result<()> {
        // Load cleaned region image
        let mut cleaned_img = image::load_from_memory(cleaned_bytes)
            .context("Failed to load cleaned region")?
            .to_rgba8();

        let [x1, y1, x2, y2] = region.bbox;
        let width = (x2 - x1).max(1) as u32;
        let height = (y2 - y1).max(1) as u32;

        // If text is empty, just paste cleaned image
        if translation.translated_text.trim().is_empty() {
            image::imageops::overlay(final_image, &cleaned_img, x1 as i64, y1 as i64);
            return Ok(());
        }

        // Render text on transparent canvas using local coordinates
        let text_canvas = self.render_text_for_region(
            width,
            height,
            local_label_1_regions,
            &translation.translated_text,
            font_family,
            text_stroke,
        ).await?;

        // OPTIMIZATION: Parallelize alpha blending using rayon with pre-allocated buffer
        // For 640x640 region, this is 409,600 pixels. Parallel processing
        // provides ~10-15% speedup on multi-core systems.
        // Pre-allocate buffer and process rows in parallel
        let total_pixels = (width * height) as usize;
        let mut img_data = Vec::with_capacity(total_pixels * 4);

        let row_data: Vec<Vec<u8>> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut row_bytes = Vec::with_capacity((width * 4) as usize);
                for x in 0..width {
                    let text_pixel = text_canvas.get_pixel(x, y);
                    if text_pixel[3] > 0 {
                        // Has alpha - blend it
                        let bg_pixel = cleaned_img.get_pixel(x, y);
                        let alpha = text_pixel[3] as f32 / 255.0;

                        row_bytes.push(((text_pixel[0] as f32 * alpha) + (bg_pixel[0] as f32 * (1.0 - alpha))) as u8);
                        row_bytes.push(((text_pixel[1] as f32 * alpha) + (bg_pixel[1] as f32 * (1.0 - alpha))) as u8);
                        row_bytes.push(((text_pixel[2] as f32 * alpha) + (bg_pixel[2] as f32 * (1.0 - alpha))) as u8);
                        row_bytes.push(255u8);
                    } else {
                        // No alpha - keep background
                        let bg = cleaned_img.get_pixel(x, y);
                        row_bytes.push(bg[0]);
                        row_bytes.push(bg[1]);
                        row_bytes.push(bg[2]);
                        row_bytes.push(bg[3]);
                    }
                }
                row_bytes
            })
            .collect();

        // Efficiently flatten into pre-allocated buffer
        for row in row_data {
            img_data.extend_from_slice(&row);
        }

        // Replace cleaned_img pixels with blended result
        cleaned_img = ImageBuffer::from_vec(width, height, img_data)
            .context("Failed to create blended image buffer")?;

        // Paste composited result onto final image
        image::imageops::overlay(final_image, &cleaned_img, x1 as i64, y1 as i64);

        Ok(())
    }

    /// Render text on transparent canvas using cosmic renderer
    ///
    /// # Arguments:
    /// * `canvas_width` - Width of the region
    /// * `canvas_height` - Height of the region
    /// * `label_1_regions` - Text regions in local coordinates (relative to region bbox)
    /// * `text` - Text to render
    /// * `font_family` - Font family to use
    /// * `text_stroke` - Whether to add stroke/outline to text
    async fn render_text_for_region(
        &self,
        canvas_width: u32,
        canvas_height: u32,
        label_1_regions: &[[i32; 4]],
        text: &str,
        font_family: &str,
        text_stroke: bool,
    ) -> Result<RgbaImage> {
        // Create transparent canvas for early return only
        let empty_canvas = ImageBuffer::from_pixel(
            canvas_width,
            canvas_height,
            Rgba([0, 0, 0, 0]),
        );

        // Calculate text rendering area (union of all label 1 regions)
        if label_1_regions.is_empty() {
            return Ok(empty_canvas);
        }

        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;

        for bbox in label_1_regions {
            let [x1, y1, x2, y2] = *bbox;
            min_x = min_x.min(x1);
            min_y = min_y.min(y1);
            max_x = max_x.max(x2);
            max_y = max_y.max(y2);
        }

        let text_width = (max_x - min_x).max(1) as f32;
        let text_height = (max_y - min_y).max(1) as f32;

        // Add padding for comic-style text (10% on each side)
        let padding = 0.1;
        let available_width = text_width * (1.0 - padding * 2.0);
        let available_height = text_height * (1.0 - padding * 2.0);

        // Use cosmic-text's built-in optimal font size finder
        // Optimized for comic readability
        let font_size = self.renderer.find_optimal_font_size(
            text,
            font_family,
            available_width,
            available_height,
            MIN_FONT_SIZE,
            MAX_FONT_SIZE,
        ).await?;

        let text_color = Rgba([0u8, 0u8, 0u8, 255u8]); // Black text

        // Create upscaled canvas for better quality
        let upscale_factor = self.config.upscale_factor();
        let upscaled_width = canvas_width * upscale_factor;
        let upscaled_height = canvas_height * upscale_factor;

        let mut upscaled_canvas = ImageBuffer::from_pixel(
            upscaled_width,
            upscaled_height,
            Rgba([0, 0, 0, 0]),
        );

        // Measure actual text dimensions for centering
        let (actual_text_width, actual_text_height) = self.renderer.measure_text(
            text,
            font_family,
            font_size,
            Some(text_width),
        ).await?;

        // Calculate centered position
        let center_offset_x = ((text_width - actual_text_width) / 2.0).max(0.0);
        let center_offset_y = ((text_height - actual_text_height) / 2.0).max(0.0);

        // Render text on upscaled canvas at centered position
        let scaled_x = ((min_x as f32 + center_offset_x) * upscale_factor as f32) as i32;
        let scaled_y = ((min_y as f32 + center_offset_y) * upscale_factor as f32) as i32;
        let scaled_font_size = font_size * upscale_factor as f32;
        let scaled_max_width = Some(text_width * upscale_factor as f32);

        // Use stroke if enabled
        let stroke_width = if text_stroke {
            Some(self.config.text_stroke_width() * upscale_factor as i32)
        } else {
            None
        };

        self.renderer.render_text(
            &mut upscaled_canvas,
            text,
            font_family,
            scaled_font_size,
            text_color,
            scaled_x,
            scaled_y,
            scaled_max_width,
            stroke_width,
        ).await?;

        // Downscale back
        let final_canvas = image::imageops::resize(
            &upscaled_canvas,
            canvas_width,
            canvas_height,
            image::imageops::FilterType::Lanczos3,
        );

        Ok(final_canvas)
    }
}
