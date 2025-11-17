// Phase 3: Mask-based Text Removal Pipeline

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Luma, Rgba};
use opencv::core::{Mat, Size, BORDER_DEFAULT};
use opencv::imgproc::{self, MORPH_RECT};
use opencv::prelude::*;
use std::sync::Arc;
use tracing::{debug, instrument};

use crate::core::config::Config;
use crate::core::types::{CategorizedRegion, ImageData, Phase1Output, Phase3Output};

/// Phase 3 pipeline: Mask-based text removal
pub struct Phase3Pipeline {
    #[allow(dead_code)]
    config: Arc<Config>,
}

impl Phase3Pipeline {
    /// Create new Phase 3 pipeline
    pub fn new(config: Arc<Config>) -> Self {
        Self { config }
    }

    /// Execute Phase 3: Remove text using segmentation masks
    ///
    /// # Steps (based on minimal_code.py lines 74-85):
    /// 1. For each region (skip banana-processed complex bg):
    ///    a. Extract label 1,2 areas
    ///    b. Create label 1,2 mask (eroded)
    ///    c. Intersect with segmentation mask
    ///    d. Replace masked pixels with white (255)
    ///
    /// # Arguments:
    /// * `image_data` - Original image data
    /// * `phase1_output` - Phase 1 detection/categorization results
    /// * `banana_processed_region_ids` - IDs of regions processed with banana (skip these)
    ///
    /// # Returns:
    /// Phase3Output with cleaned region images
    #[instrument(skip(self, image_data, phase1_output, banana_processed_region_ids), fields(
        page_index = phase1_output.page_index,
        regions = phase1_output.regions.len()
    ))]
    pub async fn execute(
        &self,
        image_data: &ImageData,
        phase1_output: &Phase1Output,
        banana_processed_region_ids: &[String],
    ) -> Result<Phase3Output> {
        debug!(
            "Phase 3: Text removal for page {}",
            phase1_output.page_index
        );

        // Load image
        let img = image::load_from_memory(&image_data.image_bytes)
            .context("Failed to load image")?;

        let width = phase1_output.width;
        let height = phase1_output.height;

        // Reconstruct segmentation mask
        let seg_mask = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
            width,
            height,
            phase1_output.segmentation_mask.clone(),
        )
        .context("Failed to reconstruct segmentation mask")?;

        let mut cleaned_regions = Vec::new();

        for region in &phase1_output.regions {
            // Skip banana-processed regions
            if banana_processed_region_ids.contains(&region.region_id) {
                debug!(
                    "Skipping region {} (processed with banana)",
                    region.region_id
                );
                continue;
            }

            // Only process simple backgrounds and complex backgrounds without banana
            debug!(
                "Cleaning region {} (label {}, {:?})",
                region.region_id, region.label, region.background_type
            );

            let cleaned_bytes = self
                .clean_region(&img, region, &seg_mask)
                .context("Failed to clean region")?;

            cleaned_regions.push((region.region_id.clone(), cleaned_bytes));
        }

        Ok(Phase3Output {
            page_index: phase1_output.page_index,
            cleaned_regions,
        })
    }

    /// Clean a single region using mask-based text removal
    ///
    /// # Implementation (minimal_code.py lines 46-85):
    /// 1. Create label 1 mask for the region
    /// 2. Erode mask for tighter fit (7x7 kernel, 4 iterations)
    /// 3. Intersect with segmentation mask
    /// 4. Replace masked pixels with white
    fn clean_region(
        &self,
        img: &DynamicImage,
        region: &CategorizedRegion,
        seg_mask: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<Vec<u8>> {
        let [x1, y1, x2, y2] = region.bbox;
        let width = (x2 - x1).max(1) as u32;
        let height = (y2 - y1).max(1) as u32;

        // Crop region from original image
        let mut cropped = img.crop_imm(x1 as u32, y1 as u32, width, height).to_rgba8();

        // Special handling for label 2 (free text): fill or blur entire bbox
        if region.label == 2 {
            return self.clean_free_text_region(cropped);
        }

        // Create label 1 mask for this region
        let mut label_1_mask = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width, height);

        for l1_bbox in &region.label_1_regions {
            let [l1_x1, l1_y1, l1_x2, l1_y2] = *l1_bbox;
            // Convert to region-local coordinates
            let local_x1 = ((l1_x1 - x1).max(0)) as u32;
            let local_y1 = ((l1_y1 - y1).max(0)) as u32;
            let local_x2 = ((l1_x2 - x1).min(width as i32)) as u32;
            let local_y2 = ((l1_y2 - y1).min(height as i32)) as u32;

            // Fill label 1 region with 255
            for y in local_y1..local_y2 {
                for x in local_x1..local_x2 {
                    if x < width && y < height {
                        label_1_mask.put_pixel(x, y, Luma([255u8]));
                    }
                }
            }
        }

        // Erode label 1 mask (minimal_code.py lines 47-48)
        let erosion_kernel = 7;
        let erosion_iterations = 4;
        let eroded_label_1_mask = self.opencv_erode(&label_1_mask, erosion_kernel, erosion_iterations)?;

        // Crop segmentation mask to region
        let mut seg_mask_cropped = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let src_x = (x1 as u32 + x).min(seg_mask.width() - 1);
                let src_y = (y1 as u32 + y).min(seg_mask.height() - 1);
                let pixel = seg_mask.get_pixel(src_x, src_y);
                seg_mask_cropped.put_pixel(x, y, *pixel);
            }
        }

        // Final mask = eroded_label_1 AND segmentation (minimal_code.py line 75)
        let mut final_mask = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let label1_val = eroded_label_1_mask.get_pixel(x, y)[0];
                let seg_val = seg_mask_cropped.get_pixel(x, y)[0];
                // Bitwise AND
                final_mask.put_pixel(x, y, Luma([if label1_val > 0 && seg_val > 0 { 255 } else { 0 }]));
            }
        }

        // Apply mask: Replace masked pixels with white (minimal_code.py lines 83-84)
        for y in 0..height {
            for x in 0..width {
                if final_mask.get_pixel(x, y)[0] > 0 {
                    cropped.put_pixel(x, y, Rgba([255, 255, 255, 255]));
                }
            }
        }

        // Convert to PNG bytes
        let mut png_bytes = Vec::new();
        DynamicImage::ImageRgba8(cropped)
            .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
            .context("Failed to encode cleaned region")?;

        Ok(png_bytes)
    }

    /// OpenCV erode operation (minimal_code.py line 48)
    fn opencv_erode(
        &self,
        img: &ImageBuffer<Luma<u8>, Vec<u8>>,
        kernel_size: i32,
        iterations: i32,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let (width, height) = img.dimensions();

        // Convert to OpenCV Mat
        let mut src = Mat::new_rows_cols_with_default(
            height as i32,
            width as i32,
            opencv::core::CV_8UC1,
            opencv::core::Scalar::all(0.0),
        )?;

        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y)[0];
                *src.at_2d_mut::<u8>(y as i32, x as i32)? = pixel;
            }
        }

        // Create kernel
        let kernel = imgproc::get_structuring_element(
            MORPH_RECT,
            Size::new(kernel_size, kernel_size),
            opencv::core::Point::new(-1, -1),
        )?;

        // Erode
        let mut dst = Mat::default();
        imgproc::erode(
            &src,
            &mut dst,
            &kernel,
            opencv::core::Point::new(-1, -1),
            iterations,
            BORDER_DEFAULT,
            imgproc::morphology_default_border_value()?,
        )?;

        // Convert back to ImageBuffer
        let mut result = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let pixel = *dst.at_2d::<u8>(y as i32, x as i32)?;
                result.put_pixel(x, y, Luma([pixel]));
            }
        }

        Ok(result)
    }

    /// Clean free text (label 2) region: white fill or blur entire bbox
    fn clean_free_text_region(&self, mut img: image::RgbaImage) -> Result<Vec<u8>> {
        if self.config.blur_free_text() {
            // Blur the entire region
            let blur_radius = self.config.blur_radius();
            img = image::imageops::blur(&img, blur_radius);
        } else {
            // Fill entire region with white
            for pixel in img.pixels_mut() {
                *pixel = Rgba([255, 255, 255, 255]);
            }
        }

        // Convert to PNG bytes
        let mut png_bytes = Vec::new();
        DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
            .context("Failed to encode cleaned free text region")?;

        Ok(png_bytes)
    }
}
