// Phase 1: Detection & Categorization Pipeline

use anyhow::{Context, Result};
use image::DynamicImage;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, instrument, warn};

use crate::core::config::Config;
use crate::services::detection::DetectionService;
use crate::services::segmentation::SegmentationService;
use crate::core::types::{
    BackgroundType, CategorizedRegion, ImageData, Phase1Output, RegionDetection,
};

/// Phase 1 pipeline: Detection & Categorization
pub struct Phase1Pipeline {
    config: Arc<Config>,
    detector: Arc<DetectionService>,
    segmenter: Arc<SegmentationService>,
}

impl Phase1Pipeline {
    /// Create new Phase 1 pipeline
    pub fn new(
        config: Arc<Config>,
        detector: Arc<DetectionService>,
        segmenter: Arc<SegmentationService>,
    ) -> Self {
        Self {
            config,
            detector,
            segmenter,
        }
    }

    /// Execute Phase 1 on a single image
    ///
    /// # Steps:
    /// 1. Run detector for labels 0, 1, 2
    /// 2. Run mask model for segmentation
    /// 3. Validate label 1 regions are within label 0
    /// 4. Categorize simple vs complex backgrounds
    ///
    /// # Returns
    /// Phase1Output with categorized regions
    #[instrument(skip(self, image_data), fields(
        page_index = image_data.index,
        filename = %image_data.filename
    ))]
    pub async fn execute(&self, image_data: &ImageData) -> Result<Phase1Output> {
        debug!(
            "Phase 1: Processing page {} ({})",
            image_data.index, image_data.filename
        );

        // Load image
        let img = image::load_from_memory(&image_data.image_bytes)
            .context("Failed to load image")?;

        // Step 1 & 2: Run detection and segmentation IN PARALLEL
        // This is a MASSIVE optimization - both models run simultaneously!
        debug!("Running detection and segmentation in parallel...");
        let parallel_start = std::time::Instant::now();

        let (detection_result, segmentation_result) =
            tokio::join!(
                self.detector.detect_all_labels(&img, image_data.index),
                self.segmenter.generate_mask(&img)
            );

        let (label_0_raw, label_1_raw, label_2_raw) =
            detection_result.context("Failed to detect regions")?;
        let segmentation_mask = segmentation_result.context("Failed to generate segmentation mask")?;

        // Convert BubbleDetection to RegionDetection
        let label_0_detections: Vec<RegionDetection> = label_0_raw
            .into_iter()
            .map(|d| RegionDetection { bbox: d.bbox, label: 0, confidence: d.confidence })
            .collect();
        let label_1_detections: Vec<RegionDetection> = label_1_raw
            .into_iter()
            .map(|d| RegionDetection { bbox: d.bbox, label: 1, confidence: d.confidence })
            .collect();
        let label_2_detections: Vec<RegionDetection> = label_2_raw
            .into_iter()
            .map(|d| RegionDetection { bbox: d.bbox, label: 2, confidence: d.confidence })
            .collect();

        debug!(
            "✓ Parallel inference completed in {:.2}ms - Detected: {} label 0, {} label 1, {} label 2",
            parallel_start.elapsed().as_secs_f64() * 1000.0,
            label_0_detections.len(),
            label_1_detections.len(),
            label_2_detections.len()
        );

        // Step 3 & 4: Validate and categorize
        let (regions, validation_warnings) = self.categorize_regions(
            &img,
            &label_0_detections,
            &label_1_detections,
            &label_2_detections,
            image_data.index,
        )?;

        Ok(Phase1Output {
            page_index: image_data.index,
            filename: image_data.filename.clone(),
            width: image_data.width,
            height: image_data.height,
            regions,
            segmentation_mask,
            validation_warnings,
        })
    }

    /// Categorize regions and validate label 1 within label 0
    ///
    /// # Returns
    /// (categorized_regions, validation_warnings)
    fn categorize_regions(
        &self,
        img: &DynamicImage,
        label_0: &[RegionDetection],
        label_1: &[RegionDetection],
        label_2: &[RegionDetection],
        page_index: usize,
    ) -> Result<(Vec<CategorizedRegion>, Vec<String>)> {
        let mut regions = Vec::new();
        let mut warnings = Vec::new();

        // Process label 0 (speech bubbles) IN PARALLEL with rayon
        let rgb_img = img.to_rgb8();
        let threshold = self.config.simple_bg_white_threshold();
        let region_id_counter = AtomicUsize::new(page_index * 10000); // Base ID on page index for uniqueness

        let l0_results: Vec<_> = label_0.par_iter().map(|l0| {
            // Find all label 1 regions within this label 0
            let mut children_l1 = Vec::new();
            let mut local_warnings = Vec::new();

            for l1 in label_1 {
                if self.is_bbox_within(l1.bbox, l0.bbox) {
                    children_l1.push(l1.bbox);
                } else if self.bboxes_overlap(l1.bbox, l0.bbox) {
                    local_warnings.push(format!(
                        "Label 1 region {:?} not fully within label 0 {:?}",
                        l1.bbox, l0.bbox
                    ));
                }
            }

            if children_l1.is_empty() {
                local_warnings.push(format!("Label 0 region {:?} has no label 1 children", l0.bbox));
            }

            // Categorize background (parallel pixel counting)
            let bg_type = Self::classify_background_fast(&rgb_img, l0.bbox, &children_l1, threshold);

            (CategorizedRegion {
                region_id: region_id_counter.fetch_add(1, Ordering::Relaxed),
                page_index,
                label: 0,
                bbox: l0.bbox,
                background_type: bg_type,
                label_1_regions: children_l1,
            }, local_warnings)
        }).collect();

        // Collect results
        for (region, mut local_warnings) in l0_results {
            regions.push(region);
            warnings.append(&mut local_warnings);
        }

        // Process standalone label 1 (not within any label 0) - skip them with warning
        for l1 in label_1 {
            let within_any_l0 = label_0.iter().any(|l0| self.is_bbox_within(l1.bbox, l0.bbox));
            if !within_any_l0 {
                warn!(
                    "Label 1 region {:?} not within any label 0 - discarding",
                    l1.bbox
                );
                warnings.push(format!(
                    "Label 1 region {:?} not within any label 0",
                    l1.bbox
                ));
            }
        }

        // Process label 2 (free text) - always complex
        for l2 in label_2 {
            regions.push(CategorizedRegion {
                region_id: region_id_counter.fetch_add(1, Ordering::Relaxed),
                page_index,
                label: 2,
                bbox: l2.bbox,
                background_type: BackgroundType::Complex,
                label_1_regions: vec![l2.bbox], // Use itself as text region
            });
        }

        Ok((regions, warnings))
    }

    /// Check if bbox1 is fully within bbox2
    fn is_bbox_within(&self, bbox1: [i32; 4], bbox2: [i32; 4]) -> bool {
        let [x1_1, y1_1, x2_1, y2_1] = bbox1;
        let [x1_2, y1_2, x2_2, y2_2] = bbox2;

        x1_1 >= x1_2 && y1_1 >= y1_2 && x2_1 <= x2_2 && y2_1 <= y2_2
    }

    /// Check if two bboxes overlap
    fn bboxes_overlap(&self, bbox1: [i32; 4], bbox2: [i32; 4]) -> bool {
        let [x1_1, y1_1, x2_1, y2_1] = bbox1;
        let [x1_2, y1_2, x2_2, y2_2] = bbox2;

        !(x2_1 < x1_2 || x2_2 < x1_1 || y2_1 < y1_2 || y2_2 < y1_1)
    }

    /// Fast parallel background classification (static method for rayon)
    ///
    /// # Logic:
    /// - Simple: >= threshold% of pixels in label 0,1 region are white (>= 240)
    /// - Complex: otherwise
    fn classify_background_fast(
        rgb: &image::RgbImage,
        label_0_bbox: [i32; 4],
        label_1_regions: &[[i32; 4]],
        threshold: f32,
    ) -> BackgroundType {
        let [x1, y1, x2, y2] = label_0_bbox;
        let img_width = rgb.width() as i32;
        let img_height = rgb.height() as i32;

        // Count white pixels in label 1 regions (parallel iteration with optimized bounds)
        let (white_pixels, total_pixels): (usize, usize) = label_1_regions.par_iter()
            .map(|l1_bbox| {
                let [l1_x1, l1_y1, l1_x2, l1_y2] = *l1_bbox;

                // Pre-calculate and clamp bounds to valid range (eliminates per-pixel bounds checking)
                let crop_x1 = l1_x1.max(x1).max(0).min(img_width) as u32;
                let crop_y1 = l1_y1.max(y1).max(0).min(img_height) as u32;
                let crop_x2 = l1_x2.min(x2).max(0).min(img_width) as u32;
                let crop_y2 = l1_y2.min(y2).max(0).min(img_height) as u32;

                // Early exit for invalid regions
                if crop_x2 <= crop_x1 || crop_y2 <= crop_y1 {
                    return (0, 0);
                }

                let mut local_white = 0usize;
                let mut local_total = 0usize;

                // No bounds checking needed - already validated above
                for y in crop_y1..crop_y2 {
                    for x in crop_x1..crop_x2 {
                        local_total += 1;
                        let pixel = rgb.get_pixel(x, y);
                        // Optimized: single comparison for all channels (white = RGB all >= 240)
                        if pixel[0] >= 240 && pixel[1] >= 240 && pixel[2] >= 240 {
                            local_white += 1;
                        }
                    }
                }
                (local_white, local_total)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

        if total_pixels == 0 {
            return BackgroundType::Complex;
        }

        let white_ratio = white_pixels as f32 / total_pixels as f32;
        debug!("Simple background: {:.1}% white ({}/{})",
            white_ratio * 100.0, white_pixels, total_pixels);

        if white_ratio >= threshold {
            BackgroundType::Simple
        } else {
            BackgroundType::Complex
        }
    }

    /// Legacy classify_background (kept for compatibility if needed)
    #[allow(dead_code)]
    fn classify_background(
        &self,
        img: &DynamicImage,
        label_0_bbox: [i32; 4],
        label_1_regions: &[[i32; 4]],
    ) -> Result<BackgroundType> {
        let [x1, y1, x2, y2] = label_0_bbox;
        let width = (x2 - x1).max(1) as u32;
        let height = (y2 - y1).max(1) as u32;

        // Crop to label 0 region
        let cropped = img.crop_imm(x1 as u32, y1 as u32, width, height);
        let rgb = cropped.to_rgb8();

        // Count white pixels in label 1 regions
        let mut total_pixels = 0;
        let mut white_pixels = 0;
        let white_threshold = 240u8;

        for l1_bbox in label_1_regions {
            let [l1_x1, l1_y1, l1_x2, l1_y2] = *l1_bbox;
            // Convert to cropped coordinates
            let crop_x1 = ((l1_x1 - x1).max(0)) as u32;
            let crop_y1 = ((l1_y1 - y1).max(0)) as u32;
            let crop_x2 = ((l1_x2 - x1).min(width as i32)) as u32;
            let crop_y2 = ((l1_y2 - y1).min(height as i32)) as u32;

            for y in crop_y1..crop_y2 {
                for x in crop_x1..crop_x2 {
                    if x < width && y < height {
                        total_pixels += 1;
                        let pixel = rgb.get_pixel(x, y);
                        if pixel[0] >= white_threshold
                            && pixel[1] >= white_threshold
                            && pixel[2] >= white_threshold
                        {
                            white_pixels += 1;
                        }
                    }
                }
            }
        }

        if total_pixels == 0 {
            return Ok(BackgroundType::Complex);
        }

        let white_ratio = white_pixels as f32 / total_pixels as f32;
        let threshold = self.config.simple_bg_white_threshold();

        if white_ratio >= threshold {
            debug!(
                "Simple background: {:.1}% white ({}/{})",
                white_ratio * 100.0,
                white_pixels,
                total_pixels
            );
            Ok(BackgroundType::Simple)
        } else {
            debug!(
                "Complex background: {:.1}% white ({}/{})",
                white_ratio * 100.0,
                white_pixels,
                total_pixels
            );
            Ok(BackgroundType::Complex)
        }
    }
}
