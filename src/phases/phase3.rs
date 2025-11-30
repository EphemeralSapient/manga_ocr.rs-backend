// Phase 3: Simplified Parallel Text Removal
//
// This phase removes text from detected regions using segmentation masks.
// All regions are processed in parallel using rayon for maximum throughput.

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Luma, Rgba};
use rayon::prelude::*;
use std::sync::Arc;
use tracing::{debug, instrument};

use crate::core::config::Config;
use crate::core::types::{CategorizedRegion, ImageData, Phase1Output, Phase3Output};

/// Phase 3 pipeline: Parallel text removal
pub struct Phase3Pipeline {
    config: Arc<Config>,
}

impl Phase3Pipeline {
    pub fn new(config: Arc<Config>) -> Self {
        Self { config }
    }

    /// Execute Phase 3: Remove text from all regions in parallel
    ///
    /// # Arguments
    /// * `image_data` - Original image data
    /// * `phase1_output` - Phase 1 detection results
    /// * `banana_processed_ids` - Region IDs already processed by banana mode (skip)
    /// * `blur_free_text` - Blur label 2 regions instead of white fill
    /// * `use_mask` - Use segmentation mask (false = fill entire label1 with white)
    #[instrument(skip(self, image_data, phase1_output, banana_processed_ids), fields(
        page = phase1_output.page_index,
        regions = phase1_output.regions.len()
    ))]
    pub async fn execute(
        &self,
        image_data: &ImageData,
        phase1_output: &Phase1Output,
        banana_processed_ids: &[usize],
        blur_free_text: bool,
        use_mask: bool,
    ) -> Result<Phase3Output> {
        let start = std::time::Instant::now();

        // Load image once
        let img: Arc<DynamicImage> = if let Some(ref decoded) = image_data.decoded_image {
            Arc::clone(decoded)
        } else {
            Arc::new(image::load_from_memory(&image_data.image_bytes)
                .context("Failed to load image")?)
        };

        // Reconstruct segmentation mask
        let seg_mask = Arc::new(
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                phase1_output.width,
                phase1_output.height,
                phase1_output.segmentation_mask.clone(),
            ).context("Failed to reconstruct segmentation mask")?
        );

        let blur_radius = self.config.blur_radius();

        // Filter regions to process (exclude banana-processed)
        let regions_to_process: Vec<_> = phase1_output.regions.iter()
            .filter(|r| !banana_processed_ids.contains(&r.region_id))
            .collect();

        debug!("Phase 3: Processing {} regions in parallel", regions_to_process.len());

        // PARALLEL: Process all regions concurrently
        let cleaned_regions: Result<Vec<_>> = regions_to_process
            .par_iter()
            .map(|region| {
                let cleaned_bytes = clean_region(
                    &img,
                    region,
                    &seg_mask,
                    blur_free_text,
                    use_mask,
                    blur_radius,
                )?;
                Ok((region.region_id, cleaned_bytes))
            })
            .collect();

        let cleaned_regions = cleaned_regions?;

        debug!("Phase 3 completed in {:.0}ms", start.elapsed().as_secs_f64() * 1000.0);

        Ok(Phase3Output {
            page_index: phase1_output.page_index,
            cleaned_regions,
        })
    }
}

/// Clean a single region by removing text
///
/// # Strategy:
/// - Label 2 (free text): Fill with white or blur entire region
/// - Label 0/1 (bubbles): Use segmentation mask to remove text, or fill label1 areas with white
fn clean_region(
    img: &DynamicImage,
    region: &CategorizedRegion,
    seg_mask: &ImageBuffer<Luma<u8>, Vec<u8>>,
    blur_free_text: bool,
    use_mask: bool,
    blur_radius: f32,
) -> Result<Vec<u8>> {
    let [x1, y1, x2, y2] = region.bbox;
    let width = (x2 - x1).max(1) as u32;
    let height = (y2 - y1).max(1) as u32;

    // Crop region from original
    let mut cropped = img.crop_imm(x1 as u32, y1 as u32, width, height).to_rgba8();

    // Label 2 (free text): Special handling - fill or blur entire region
    if region.label == 2 {
        return clean_free_text(&cropped, blur_free_text, blur_radius);
    }

    // Label 0/1 (speech bubbles)
    if !use_mask {
        // Simple mode: Fill all label1 areas with white
        fill_label1_regions_white(&mut cropped, region, x1, y1, width, height);
    } else {
        // Mask mode: Use segmentation mask to remove text
        apply_mask_removal(&mut cropped, region, seg_mask, x1, y1, width, height);
    }

    // Encode to PNG
    encode_png(cropped)
}

/// Fill all label 1 regions with white (no mask mode)
fn fill_label1_regions_white(
    cropped: &mut image::RgbaImage,
    region: &CategorizedRegion,
    x1: i32,
    y1: i32,
    width: u32,
    height: u32,
) {
    for l1_bbox in &region.label_1_regions {
        let [l1_x1, l1_y1, l1_x2, l1_y2] = *l1_bbox;
        // Convert to local coordinates
        let local_x1 = ((l1_x1 - x1).max(0)) as u32;
        let local_y1 = ((l1_y1 - y1).max(0)) as u32;
        let local_x2 = ((l1_x2 - x1).min(width as i32)) as u32;
        let local_y2 = ((l1_y2 - y1).min(height as i32)) as u32;

        for y in local_y1..local_y2 {
            for x in local_x1..local_x2 {
                if x < width && y < height {
                    cropped.put_pixel(x, y, Rgba([255, 255, 255, 255]));
                }
            }
        }
    }
}

/// Apply segmentation mask to remove text
fn apply_mask_removal(
    cropped: &mut image::RgbaImage,
    region: &CategorizedRegion,
    seg_mask: &ImageBuffer<Luma<u8>, Vec<u8>>,
    x1: i32,
    y1: i32,
    width: u32,
    height: u32,
) {
    // Create label1 constraint mask
    let mut label1_mask = vec![0u8; (width * height) as usize];

    for l1_bbox in &region.label_1_regions {
        let [l1_x1, l1_y1, l1_x2, l1_y2] = *l1_bbox;
        let local_x1 = ((l1_x1 - x1).max(0)) as u32;
        let local_y1 = ((l1_y1 - y1).max(0)) as u32;
        let local_x2 = ((l1_x2 - x1).min(width as i32)) as u32;
        let local_y2 = ((l1_y2 - y1).min(height as i32)) as u32;

        for y in local_y1..local_y2 {
            for x in local_x1..local_x2 {
                if x < width && y < height {
                    label1_mask[(y * width + x) as usize] = 255;
                }
            }
        }
    }

    // Apply: where (label1_mask AND seg_mask) == white, fill with white
    let src_x1 = x1 as u32;
    let src_y1 = y1 as u32;

    for y in 0..height {
        for x in 0..width {
            let label1_val = label1_mask[(y * width + x) as usize];
            if label1_val == 0 {
                continue; // Outside label1 area
            }

            // Get seg mask value at this global position
            let global_x = src_x1 + x;
            let global_y = src_y1 + y;

            if global_x < seg_mask.width() && global_y < seg_mask.height() {
                let seg_val = seg_mask.get_pixel(global_x, global_y)[0];
                if seg_val > 0 {
                    cropped.put_pixel(x, y, Rgba([255, 255, 255, 255]));
                }
            }
        }
    }
}

/// Clean free text region (label 2)
fn clean_free_text(
    img: &image::RgbaImage,
    blur_free_text: bool,
    blur_radius: f32,
) -> Result<Vec<u8>> {
    let cleaned = if blur_free_text {
        image::imageops::blur(img, blur_radius)
    } else {
        // Fill with white
        ImageBuffer::from_pixel(img.width(), img.height(), Rgba([255, 255, 255, 255]))
    };

    encode_png(cleaned)
}

/// Encode image to PNG bytes
fn encode_png(img: image::RgbaImage) -> Result<Vec<u8>> {
    let mut png_bytes = Vec::new();
    DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .context("Failed to encode PNG")?;
    Ok(png_bytes)
}
