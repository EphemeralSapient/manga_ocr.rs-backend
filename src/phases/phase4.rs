// Phase 4: Parallel Text Rendering and Compositing
//
// This phase renders translated text onto cleaned regions and composites them
// back into the final image. Region rendering is done in parallel for speed.

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, instrument};

use crate::core::config::Config;
use crate::services::rendering::CosmicTextRenderer;
use crate::core::types::{
    BananaResult, CategorizedRegion, ImageData, OCRTranslation, Phase1Output, Phase2Output,
    Phase3Output, Phase4Output,
};

/// Phase 4 pipeline: Parallel text rendering
pub struct Phase4Pipeline {
    config: Arc<Config>,
    renderer: Arc<CosmicTextRenderer>,
}

impl Phase4Pipeline {
    pub fn new(config: Arc<Config>) -> Self {
        let renderer = Arc::new(CosmicTextRenderer::new());
        Self { config, renderer }
    }

    pub fn load_google_font(&self, font_data: Vec<u8>, family_name: &str) -> Result<()> {
        self.renderer.load_google_font(font_data, family_name)
    }

    /// Execute Phase 4: Render text and composite
    ///
    /// NEW APPROACH: Create transparent text overlay, composite onto cleaned image
    /// - Phase 1 (non-mask) or Phase 3 (mask) provides the cleaned base image
    /// - Phase 4 creates a transparent canvas with text at bubble positions
    /// - Alpha-blend text overlay onto cleaned image
    #[instrument(skip(self, image_data, phase1_output, phase2_output, phase3_output), fields(
        page = phase1_output.page_index,
        font = font_family
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
        let start = std::time::Instant::now();

        // STEP 1: Reconstruct cleaned image from Phase 3 regions (always using segmentation mask)
        let mut final_image = self.create_cleaned_base_mask(image_data, phase1_output, phase3_output)?;

        let (img_width, img_height) = final_image.dimensions();

        // Build lookup maps
        let translations_map = build_translations_map(phase2_output);
        let banana_map = build_banana_map(phase2_output);

        debug!("Phase 4: {} regions, translations_map has {} entries",
            phase1_output.regions.len(),
            translations_map.len()
        );

        // Debug: print region IDs and translation IDs
        let region_ids: Vec<_> = phase1_output.regions.iter().map(|r| r.region_id).collect();
        let translation_ids: Vec<_> = translations_map.keys().copied().collect();
        debug!("Phase 4: Region IDs: {:?}", region_ids);
        debug!("Phase 4: Translation IDs: {:?}", translation_ids);

        // STEP 2: Render all text in PARALLEL onto individual canvases
        // Each returns (region_bbox, text_canvas) for compositing
        let text_renders: Vec<_> = phase1_output.regions
            .par_iter()
            .filter_map(|region| {
                let region_id = region.region_id;

                // Skip banana regions
                if banana_map.contains_key(&region_id) {
                    return None;
                }

                // Get translation
                let translation = match translations_map.get(&region_id) {
                    Some(t) => t,
                    None => {
                        debug!("Phase 4: No translation for region_id {}", region_id);
                        return None;
                    }
                };

                // Skip empty text
                if translation.translated_text.trim().is_empty() {
                    debug!("Phase 4: Empty translation for region_id {}", region_id);
                    return None;
                }

                // Skip if no label_1 regions
                if region.label_1_regions.is_empty() {
                    debug!("Phase 4: No label_1 regions for region_id {}", region_id);
                    return None;
                }

                debug!("Phase 4: Rendering text for region {}: '{}'",
                    region_id,
                    translation.translated_text.chars().take(30).collect::<String>()
                );

                // Render text for this region
                match self.render_text_canvas(region, &translation.translated_text, font_family, text_stroke) {
                    Ok((paste_x, paste_y, canvas)) => {
                        debug!("Phase 4: Rendered region {} at ({}, {}), canvas {}x{}",
                            region_id, paste_x, paste_y, canvas.width(), canvas.height());
                        Some((paste_x, paste_y, canvas))
                    },
                    Err(e) => {
                        debug!("Failed to render text for region {}: {:?}", region_id, e);
                        None
                    }
                }
            })
            .collect();

        debug!("Phase 4: {} text renders to composite", text_renders.len());

        // STEP 3: Composite all text canvases onto the cleaned image
        for (paste_x, paste_y, text_canvas) in text_renders {
            for (tx, ty, text_px) in text_canvas.enumerate_pixels() {
                if text_px[3] > 0 {
                    let fx = (paste_x + tx as i32).max(0) as u32;
                    let fy = (paste_y + ty as i32).max(0) as u32;

                    if fx < img_width && fy < img_height {
                        let bg_px = final_image.get_pixel(fx, fy);
                        let alpha = text_px[3] as f32 / 255.0;

                        let blended = Rgba([
                            blend(text_px[0], bg_px[0], alpha),
                            blend(text_px[1], bg_px[1], alpha),
                            blend(text_px[2], bg_px[2], alpha),
                            255,
                        ]);

                        final_image.put_pixel(fx, fy, blended);
                    }
                }
            }
        }

        // STEP 4: Composite banana results
        for region in &phase1_output.regions {
            if let Some(banana) = banana_map.get(&region.region_id) {
                composite_banana(&mut final_image, region, banana)?;
            }
        }

        // Encode final image
        let mut png_bytes = Vec::new();
        DynamicImage::ImageRgba8(final_image)
            .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
            .context("Failed to encode final image")?;

        debug!("Phase 4 completed in {:.0}ms", start.elapsed().as_secs_f64() * 1000.0);

        Ok(Phase4Output {
            page_index: phase1_output.page_index,
            final_image_bytes: png_bytes,
        })
    }

    /// Create cleaned base image using segmentation mask
    /// Composites Phase 3 cleaned regions onto original image
    fn create_cleaned_base_mask(
        &self,
        image_data: &ImageData,
        _phase1_output: &Phase1Output,
        phase3_output: &Phase3Output,
    ) -> Result<RgbaImage> {
        let img: DynamicImage = if let Some(ref decoded) = image_data.decoded_image {
            (**decoded).clone()
        } else {
            image::load_from_memory(&image_data.image_bytes)
                .context("Failed to load image")?
        };

        let mut final_image = img.to_rgba8();
        let cleaned_list = build_cleaned_map(phase3_output);

        // Composite each cleaned label_1 region onto the base image
        for (cleaned_bytes, bbox) in cleaned_list {
            let cleaned_img = image::load_from_memory(cleaned_bytes)
                .context("Failed to load cleaned region")?
                .to_rgba8();

            let [x1, y1, _, _] = bbox;
            image::imageops::overlay(&mut final_image, &cleaned_img, x1 as i64, y1 as i64);
        }

        debug!("Phase 4: Created cleaned base ({} label_1 regions)", phase3_output.cleaned_regions.len());
        Ok(final_image)
    }

    /// Render text for a single region onto a transparent canvas
    /// Returns (paste_x, paste_y, canvas) for later compositing
    ///
    /// SIMPLIFIED: Removed double-margin penalty and heuristic min_lines.
    /// Uses single fixed-pixel margin and lets find_optimal_font_size handle sizing.
    fn render_text_canvas(
        &self,
        region: &CategorizedRegion,
        text: &str,
        font_family: &str,
        text_stroke: bool,
    ) -> Result<(i32, i32, RgbaImage)> {
        // Calculate the bounding box of all label_1 regions (text area)
        let (min_x, min_y, max_x, max_y) = region.label_1_regions.iter().fold(
            (i32::MAX, i32::MAX, i32::MIN, i32::MIN),
            |(mx1, my1, mx2, my2), &[x1, y1, x2, y2]| {
                (mx1.min(x1), my1.min(y1), mx2.max(x2), my2.max(y2))
            },
        );

        let text_area_width = (max_x - min_x).max(1) as f32;
        let text_area_height = (max_y - min_y).max(1) as f32;

        // Margin scales with box size: larger boxes get proportionally more margin
        // Min 2px, max 8px, scales at 2% of smaller dimension
        let margin_px = (text_area_width.min(text_area_height) * 0.02).clamp(2.0, 8.0);
        let available_width = (text_area_width - margin_px * 2.0).max(10.0);
        let available_height = (text_area_height - margin_px * 2.0).max(10.0);

        // Font size bounds
        let min_font = 7.0f32;
        // Max font: use available_height as upper bound (single line max)
        // The algorithm will find optimal size considering line wrapping
        let max_font = available_height.min(72.0).max(min_font + 1.0);

        let stroke_width_f = if text_stroke { Some(self.config.text_stroke_width() as f32) } else { None };

        // STEP 1: Find optimal font size that fits (algorithm handles line estimation)
        // Now sync - no async/block_on needed
        let font_size = self.renderer.find_optimal_font_size(
            text,
            font_family,
            available_width,
            available_height,
            min_font,
            max_font,
            stroke_width_f,
        ).unwrap_or(min_font);

        // STEP 2: Measure actual rendered text dimensions at this font size
        let (actual_text_width, actual_text_height) = self.renderer.measure_text(
            text,
            font_family,
            font_size,
            Some(available_width),
        ).unwrap_or((available_width, font_size * 1.4));

        // STEP 3: Calculate centered position within the text area
        let offset_x = ((available_width - actual_text_width) / 2.0 + margin_px).max(margin_px);
        let offset_y = ((available_height - actual_text_height) / 2.0 + margin_px).max(margin_px);

        // STEP 4: Create upscaled canvas - canvas = text_area (NO extra margin on canvas)
        let upscale = self.config.upscale_factor();
        let canvas_width = text_area_width as u32;
        let canvas_height = text_area_height as u32;
        let upscaled_width = canvas_width * upscale;
        let upscaled_height = canvas_height * upscale;

        let mut text_canvas = ImageBuffer::from_pixel(upscaled_width, upscaled_height, Rgba([0, 0, 0, 0]));

        // Scale positions for upscaled canvas
        let scaled_x = (offset_x * upscale as f32) as i32;
        let scaled_y = (offset_y * upscale as f32) as i32;
        let scaled_font = font_size * upscale as f32;
        let scaled_max_width = available_width * upscale as f32;

        let text_color = Rgba([0u8, 0u8, 0u8, 255u8]);
        let stroke_width_render = if text_stroke {
            Some(self.config.text_stroke_width() * upscale as i32)
        } else {
            None
        };

        // STEP 5: Render text at the centered position (now sync)
        let render_result = self.renderer.render_text(
            &mut text_canvas,
            text,
            font_family,
            scaled_font,
            text_color,
            scaled_x,
            scaled_y,
            Some(scaled_max_width),
            stroke_width_render,
            None,
        );

        if let Err(e) = render_result {
            debug!("Text render warning: {:?}", e);
        }

        // STEP 6: Downscale to final size
        let final_canvas = image::imageops::resize(
            &text_canvas,
            canvas_width,
            canvas_height,
            image::imageops::FilterType::Lanczos3,
        );

        // Position canvas at text area origin (no extra margin offset needed)
        let paste_x = min_x;
        let paste_y = min_y;

        Ok((paste_x, paste_y, final_canvas))
    }
}

fn blend(fg: u8, bg: u8, alpha: f32) -> u8 {
    ((fg as f32 * alpha) + (bg as f32 * (1.0 - alpha))) as u8
}

/// Composite banana result onto final image
fn composite_banana(
    final_image: &mut RgbaImage,
    region: &CategorizedRegion,
    banana: &BananaResult,
) -> Result<()> {
    let banana_img = image::load_from_memory(&banana.translated_image_bytes)
        .context("Failed to load banana result")?
        .to_rgba8();

    let [x1, y1, x2, y2] = region.bbox;
    let width = (x2 - x1).max(1) as u32;
    let height = (y2 - y1).max(1) as u32;

    let resized = if banana_img.width() != width || banana_img.height() != height {
        image::imageops::resize(&banana_img, width, height, image::imageops::FilterType::Lanczos3)
    } else {
        banana_img
    };

    image::imageops::overlay(final_image, &resized, x1 as i64, y1 as i64);
    Ok(())
}

// Helper functions to build lookup maps
fn build_translations_map(phase2: &Phase2Output) -> HashMap<usize, &OCRTranslation> {
    let mut map = HashMap::new();
    for (id, t) in &phase2.simple_bg_translations {
        map.insert(*id, t);
    }
    for (id, t) in &phase2.complex_bg_translations {
        map.insert(*id, t);
    }
    map
}

fn build_banana_map(phase2: &Phase2Output) -> HashMap<usize, &BananaResult> {
    phase2.complex_bg_bananas.iter().map(|b| (b.region_id, b)).collect()
}

fn build_cleaned_map(phase3: &Phase3Output) -> Vec<(&Vec<u8>, [i32; 4])> {
    phase3.cleaned_regions.iter().map(|(_, bytes, bbox)| (bytes, *bbox)).collect()
}
