// Phase 1: Detection & Categorization Pipeline
//
// Simplified: Only uses YOLOv8-seg for segmentation (SAM removed for performance)

use anyhow::{Context, Result};
use image::{DynamicImage, Rgba};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, info, instrument, warn};

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

    /// Cleanup ONNX sessions after Phase 1 completes to free memory
    pub fn cleanup_sessions(&self) {
        self.detector.cleanup_sessions();
        self.segmenter.cleanup_sessions();
    }

    /// Cleanup only detection sessions (use when segmentation is still running)
    pub fn cleanup_detection_sessions(&self) {
        self.detector.cleanup_sessions();
    }

    /// Cleanup only segmentation sessions
    pub fn cleanup_segmentation_sessions(&self) {
        self.segmenter.cleanup_sessions();
    }

    /// Check if using DirectML backend
    pub fn is_directml(&self) -> bool {
        self.detector.is_directml()
    }

    /// Execute Phase 1 on a single image
    ///
    /// # Steps:
    /// 1. Run detector for labels 0, 1, 2
    /// 2. Run YOLOv8-seg for segmentation (if use_mask is true)
    /// 3. Validate label 1 regions are within label 0
    /// 4. Categorize simple vs complex backgrounds
    /// 5. If use_mask=false, also clean regions early (skips Phase 3)
    ///
    /// # Arguments
    /// * `use_mask` - Whether to use segmentation mask
    /// * `mask_mode` - Ignored (only "fast" mode is supported)
    /// * `target_size_override` - Optional target size override for detection model
    /// * `filter_orphan_regions` - If true, discard label 1 regions not within any label 0
    /// * `blur_free_text` - Whether to blur free text (label 2) backgrounds
    #[instrument(skip(self, image_data), fields(
        page_index = image_data.index,
        filename = %image_data.filename
    ))]
    pub async fn execute(
        &self,
        image_data: &ImageData,
        use_mask: bool,
        _mask_mode: Option<&str>, // Ignored - only fast mode is supported
        target_size_override: Option<u32>,
        filter_orphan_regions: bool,
        blur_free_text: bool,
    ) -> Result<Phase1Output> {
        debug!("Phase 1: Processing page {} ({})", image_data.index, image_data.filename);

        // Use pre-decoded image if available
        let img_owned;
        let img: &DynamicImage = if let Some(ref decoded) = image_data.decoded_image {
            decoded.as_ref()
        } else {
            img_owned = image::load_from_memory(&image_data.image_bytes)
                .context("Failed to load image")?;
            &img_owned
        };

        let start = std::time::Instant::now();

        // Run detection and segmentation in parallel (if mask enabled)
        let (label_0_raw, label_1_raw, label_2_raw, segmentation_mask) = if use_mask {
            debug!("Running detection + segmentation in parallel...");
            let (detection_result, segmentation_result) = tokio::join!(
                self.detector.detect_all_labels(img, image_data.index, target_size_override),
                self.segmenter.generate_mask(img)
            );

            let (l0, l1, l2) = detection_result.context("Failed to detect regions")?;
            let mask = segmentation_result.context("Failed to generate segmentation mask")?;
            (l0, l1, l2, mask)
        } else {
            debug!("Running detection only (mask disabled)...");
            let (l0, l1, l2) = self.detector.detect_all_labels(img, image_data.index, target_size_override).await
                .context("Failed to detect regions")?;
            let empty_mask = vec![0u8; (img.width() * img.height()) as usize];
            (l0, l1, l2, empty_mask)
        };

        // Convert to RegionDetection
        let label_0: Vec<RegionDetection> = label_0_raw.into_iter()
            .map(|d| RegionDetection { bbox: d.bbox, label: 0, confidence: d.confidence })
            .collect();
        let label_1: Vec<RegionDetection> = label_1_raw.into_iter()
            .map(|d| RegionDetection { bbox: d.bbox, label: 1, confidence: d.confidence })
            .collect();
        let label_2: Vec<RegionDetection> = label_2_raw.into_iter()
            .map(|d| RegionDetection { bbox: d.bbox, label: 2, confidence: d.confidence })
            .collect();

        debug!(
            "✓ Detection completed in {:.0}ms - {} label 0, {} label 1, {} label 2",
            start.elapsed().as_secs_f64() * 1000.0,
            label_0.len(), label_1.len(), label_2.len()
        );

        // Validate and categorize
        let (regions, warnings) = self.categorize_regions(
            img, &label_0, &label_1, &label_2, image_data.index, filter_orphan_regions,
        )?;

        // When use_mask=false, clean regions early (skips Phase 3)
        let early_cleaned_regions = if !use_mask && !regions.is_empty() {
            let blur_radius = self.config.blur_radius();
            info!("🧹 Early cleaning {} regions (mask disabled, skipping Phase 3)", regions.len());
            let cleaned = Self::clean_regions_early(img, &regions, blur_free_text, blur_radius);
            Some(cleaned)
        } else {
            None
        };

        Ok(Phase1Output {
            page_index: image_data.index,
            filename: image_data.filename.clone(),
            width: image_data.width,
            height: image_data.height,
            regions,
            segmentation_mask,
            mask_mode: "fast".to_string(), // Always fast mode
            validation_warnings: warnings,
            early_cleaned_regions,
        })
    }

    /// Execute Phase 1 detection only (without segmentation)
    /// Returns detection results + decoded images for later segmentation
    ///
    /// This allows Phase 2 to start immediately while segmentation runs in background.
    /// If use_mask=false, also performs early region cleaning (skips Phase 3).
    ///
    /// # Arguments
    /// * `target_size_override` - Optional target size override for detection model
    /// * `filter_orphan_regions` - If true, discard label 1 regions not within any label 0
    /// * `use_mask` - Whether segmentation mask will be used (if false, early cleaning is done)
    /// * `blur_free_text` - Whether to blur free text backgrounds (for early cleaning)
    pub async fn execute_detection_only(
        &self,
        images: &[ImageData],
        merge_img: bool,
        target_size_override: Option<u32>,
        filter_orphan_regions: bool,
        use_mask: bool,
        blur_free_text: bool,
    ) -> Result<(Vec<Phase1Output>, Vec<DynamicImage>)> {
        if images.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        if merge_img {
            debug!("Phase 1 DETECTION: Processing {} images with SINGLE batched ONNX inference", images.len());
        } else {
            debug!("Phase 1 DETECTION: Processing {} images with INDIVIDUAL inferences", images.len());
        }
        let detection_start = std::time::Instant::now();

        // Prepare images
        let mut decoded_images: Vec<DynamicImage> = Vec::with_capacity(images.len());
        for image_data in images {
            let img = if let Some(ref decoded) = image_data.decoded_image {
                (**decoded).clone()
            } else {
                image::load_from_memory(&image_data.image_bytes)
                    .context("Failed to load image")?
            };
            decoded_images.push(img);
        }

        // Run detection: batched or parallel based on merge_img setting
        let batch_detections: Vec<(Vec<_>, Vec<_>, Vec<_>)> = if merge_img {
            debug!("Running BATCHED detection (single ONNX call for {} images)...", images.len());
            let batch_refs: Vec<(&DynamicImage, usize)> = decoded_images
                .iter()
                .zip(images.iter())
                .map(|(img, data)| (img, data.index))
                .collect();

            self.detector.detect_all_labels_batch(&batch_refs, target_size_override).await
                .context("Batch detection failed")?
        } else {
            debug!("Running PARALLEL detection ({} individual ONNX calls)...", images.len());
            let detection_tasks: Vec<_> = decoded_images
                .iter()
                .zip(images.iter())
                .map(|(img, data)| {
                    let detector = Arc::clone(&self.detector);
                    let img_ref = img;
                    let index = data.index;
                    async move {
                        detector.detect_all_labels(img_ref, index, target_size_override).await
                    }
                })
                .collect();

            futures::future::join_all(detection_tasks)
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()?
        };

        debug!("✓ Detection completed in {:.2}ms (mode: {})",
               detection_start.elapsed().as_secs_f64() * 1000.0,
               if merge_img { "BATCH" } else { "PARALLEL" });

        // Get blur radius for early cleaning
        let blur_radius = self.config.blur_radius();

        // Create Phase1Output with empty segmentation masks (will be filled later if use_mask=true)
        let mut outputs = Vec::with_capacity(images.len());

        for (i, (label_0_raw, label_1_raw, label_2_raw)) in batch_detections.into_iter().enumerate() {
            let image_data = &images[i];
            let img = &decoded_images[i];

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

            // Categorize regions
            let (regions, validation_warnings) = self.categorize_regions(
                img,
                &label_0_detections,
                &label_1_detections,
                &label_2_detections,
                image_data.index,
                filter_orphan_regions,
            )?;

            // Empty segmentation mask placeholder
            let empty_mask = vec![0u8; (image_data.width * image_data.height) as usize];

            // When use_mask=false, clean regions early (skips Phase 3)
            let early_cleaned_regions = if !use_mask && !regions.is_empty() {
                Some(Self::clean_regions_early(img, &regions, blur_free_text, blur_radius))
            } else {
                None
            };

            outputs.push(Phase1Output {
                page_index: image_data.index,
                filename: image_data.filename.clone(),
                width: image_data.width,
                height: image_data.height,
                regions,
                segmentation_mask: empty_mask,
                mask_mode: "fast".to_string(), // Placeholder, will be updated by complete_segmentation
                validation_warnings,
                early_cleaned_regions,
            });
        }

        // Log early cleaning if performed
        if !use_mask {
            let total_regions: usize = outputs.iter().map(|o| o.regions.len()).sum();
            if total_regions > 0 {
                info!("🧹 Early cleaned {} regions across {} images (mask disabled, skipping Phase 3)",
                      total_regions, outputs.len());
            }
        }

        Ok((outputs, decoded_images))
    }

    /// Run segmentation and update Phase1Output with masks
    /// This is called after Phase 2 starts, to fill in the segmentation masks before Phase 3
    pub async fn complete_segmentation(
        &self,
        outputs: &mut [Phase1Output],
        decoded_images: &[DynamicImage],
        use_mask: bool,
        _mask_mode: Option<&str>, // Ignored - only fast mode supported
    ) -> Result<()> {
        if !use_mask {
            debug!("Segmentation skipped (mask disabled)");
            return Ok(());
        }

        let seg_start = std::time::Instant::now();

        if self.segmenter.is_directml() {
            // DirectML: Sequential processing
            debug!("Running segmentation for {} images sequentially (DirectML)...", decoded_images.len());

            for (i, img) in decoded_images.iter().enumerate() {
                let mask = self.segmenter.generate_mask(img).await
                    .context("Failed to generate segmentation mask")?;
                outputs[i].segmentation_mask = mask;
                outputs[i].mask_mode = "fast".to_string();
            }
        } else {
            // Parallel processing
            debug!("Running segmentation for {} images in parallel...", decoded_images.len());

            let tasks: Vec<_> = decoded_images.iter()
                .map(|img| self.segmenter.generate_mask(img))
                .collect();

            let results = futures::future::join_all(tasks).await;

            for (i, result) in results.into_iter().enumerate() {
                let mask = result.context("Failed to generate segmentation mask")?;
                outputs[i].segmentation_mask = mask;
                outputs[i].mask_mode = "fast".to_string();
            }
        }

        debug!("✓ Segmentation completed in {:.0}ms", seg_start.elapsed().as_secs_f64() * 1000.0);
        Ok(())
    }

    /// Execute Phase 1 on multiple images using batch inference
    ///
    /// This is more efficient than calling execute() multiple times when merge_img is enabled.
    /// Batch execution: supports both batched and parallel modes
    /// - merge_img=true: Single batched ONNX inference (faster for multiple images)
    /// - merge_img=false: Individual ONNX inferences in parallel (uses multiple sessions)
    /// - use_mask: Enable/disable segmentation mask generation
    /// - blur_free_text: Whether to blur free text backgrounds (for early cleaning when use_mask=false)
    ///
    /// # Arguments
    /// * `target_size_override` - Optional target size override for detection model
    /// * `filter_orphan_regions` - If true, discard label 1 regions not within any label 0
    /// * `blur_free_text` - Whether to blur free text backgrounds
    pub async fn execute_batch(&self, images: &[ImageData], use_mask: bool, merge_img: bool, target_size_override: Option<u32>, filter_orphan_regions: bool, blur_free_text: bool) -> Result<Vec<Phase1Output>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        if merge_img {
            debug!("Phase 1 BATCH MODE: Processing {} images with SINGLE batched ONNX inference", images.len());
        } else {
            debug!("Phase 1 PARALLEL MODE: Processing {} images with INDIVIDUAL inferences", images.len());
        }
        let batch_start = std::time::Instant::now();

        // Prepare images
        let mut decoded_images: Vec<DynamicImage> = Vec::with_capacity(images.len());
        for image_data in images {
            let img = if let Some(ref decoded) = image_data.decoded_image {
                (**decoded).clone()
            } else {
                image::load_from_memory(&image_data.image_bytes)
                    .context("Failed to load image")?
            };
            decoded_images.push(img);
        }

        // Run detection: batched or parallel based on merge_img setting
        let detection_start = std::time::Instant::now();
        let batch_detections: Vec<(Vec<_>, Vec<_>, Vec<_>)> = if merge_img {
            // BATCH MODE: Single ONNX inference for all images
            debug!("Running BATCHED detection (single ONNX call for {} images)...", images.len());
            let batch_refs: Vec<(&DynamicImage, usize)> = decoded_images
                .iter()
                .zip(images.iter())
                .map(|(img, data)| (img, data.index))
                .collect();

            self.detector.detect_all_labels_batch(&batch_refs, target_size_override).await
                .context("Batch detection failed")?
        } else {
            // PARALLEL MODE: Individual ONNX inferences (uses multiple sessions)
            debug!("Running PARALLEL detection ({} individual ONNX calls)...", images.len());
            let detection_tasks: Vec<_> = decoded_images
                .iter()
                .zip(images.iter())
                .map(|(img, data)| {
                    let detector = Arc::clone(&self.detector);
                    let img_ref = img;
                    let index = data.index;
                    async move {
                        detector.detect_all_labels(img_ref, index, target_size_override).await
                    }
                })
                .collect();

            futures::future::join_all(detection_tasks)
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()?
        };

        debug!("✓ Detection completed in {:.2}ms (mode: {})",
               detection_start.elapsed().as_secs_f64() * 1000.0,
               if merge_img { "BATCH" } else { "PARALLEL" });

        // Run segmentation for each image (only if use_mask is true)
        let segmentation_results: Vec<Result<Vec<u8>, anyhow::Error>> = if use_mask {
            let seg_start = std::time::Instant::now();
            let is_directml = self.segmenter.is_directml();

            let results = if is_directml {
                // DirectML: Sequential processing
                debug!("Running segmentation for {} images sequentially (DirectML)...", images.len());

                let mut masks = Vec::with_capacity(decoded_images.len());
                for img in decoded_images.iter() {
                    masks.push(self.segmenter.generate_mask(img).await);
                }
                masks
            } else {
                // Parallel processing
                debug!("Running segmentation for {} images in parallel...", images.len());

                let segmentation_tasks: Vec<_> = decoded_images.iter()
                    .map(|img| self.segmenter.generate_mask(img))
                    .collect();

                futures::future::join_all(segmentation_tasks).await
            };

            debug!("✓ Segmentation completed in {:.2}ms", seg_start.elapsed().as_secs_f64() * 1000.0);
            results
        } else {
            debug!("Segmentation skipped (mask disabled)");
            // Return empty masks for each image
            decoded_images.iter().map(|img| {
                Ok(vec![0u8; (img.width() * img.height()) as usize])
            }).collect()
        };

        // Get blur radius for early cleaning
        let blur_radius = self.config.blur_radius();

        // Process results for each image
        let mut outputs = Vec::with_capacity(images.len());

        for (i, ((label_0_raw, label_1_raw, label_2_raw), seg_result)) in
            batch_detections.into_iter().zip(segmentation_results.into_iter()).enumerate()
        {
            let image_data = &images[i];
            let img = &decoded_images[i];
            let segmentation_mask = seg_result.context("Failed to generate segmentation mask")?;

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

            // Categorize regions
            let (regions, validation_warnings) = self.categorize_regions(
                img,
                &label_0_detections,
                &label_1_detections,
                &label_2_detections,
                image_data.index,
                filter_orphan_regions,
            )?;

            // When use_mask=false, clean regions early (skips Phase 3)
            let early_cleaned_regions = if !use_mask && !regions.is_empty() {
                Some(Self::clean_regions_early(img, &regions, blur_free_text, blur_radius))
            } else {
                None
            };

            outputs.push(Phase1Output {
                page_index: image_data.index,
                filename: image_data.filename.clone(),
                width: image_data.width,
                height: image_data.height,
                regions,
                segmentation_mask,
                mask_mode: "fast".to_string(), // execute_batch always uses Fast mode
                validation_warnings,
                early_cleaned_regions,
            });
        }

        // Log early cleaning if performed
        if !use_mask {
            let total_regions: usize = outputs.iter().map(|o| o.regions.len()).sum();
            if total_regions > 0 {
                info!("🧹 Early cleaned {} regions across {} images (mask disabled, skipping Phase 3)",
                      total_regions, outputs.len());
            }
        }

        debug!(
            "✓ Phase 1 BATCH completed in {:.2}ms for {} images ({:.2}ms/image)",
            batch_start.elapsed().as_secs_f64() * 1000.0,
            images.len(),
            batch_start.elapsed().as_secs_f64() * 1000.0 / images.len() as f64
        );

        Ok(outputs)
    }

    /// Categorize regions and validate label 1 within label 0
    ///
    /// # Arguments
    /// * `filter_orphan_regions` - If true, discard label 1 regions not within any label 0
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
        filter_orphan_regions: bool,
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

        // Process standalone label 1 (not within any label 0)
        // Behavior depends on filter_orphan_regions flag
        for l1 in label_1 {
            let within_any_l0 = label_0.iter().any(|l0| self.is_bbox_within(l1.bbox, l0.bbox));
            if !within_any_l0 {
                if filter_orphan_regions {
                    // Filter enabled: Discard orphan region with warning
                    warn!(
                        "Label 1 region {:?} not within any label 0 - discarding",
                        l1.bbox
                    );
                    warnings.push(format!(
                        "Label 1 region {:?} not within any label 0",
                        l1.bbox
                    ));
                } else {
                    // Filter disabled (default): Keep orphan region as standalone text region
                    debug!(
                        "Label 1 region {:?} not within any label 0 - keeping as standalone region",
                        l1.bbox
                    );
                    regions.push(CategorizedRegion {
                        region_id: region_id_counter.fetch_add(1, Ordering::Relaxed),
                        page_index,
                        label: 1,
                        bbox: l1.bbox,
                        background_type: BackgroundType::Complex,
                        label_1_regions: vec![l1.bbox],
                    });
                }
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

        // OPTIMIZATION: Adaptive parallelization based on total region area
        // Rayon has overhead - only use it for large total areas
        let total_area: i32 = label_1_regions
            .iter()
            .map(|bbox| {
                let width = (bbox[2] - bbox[0]).max(0);
                let height = (bbox[3] - bbox[1]).max(0);
                width * height
            })
            .sum();

        const PARALLEL_THRESHOLD: i32 = 10_000; // ~100x100 pixels

        let (white_pixels, total_pixels): (usize, usize) = if total_area > PARALLEL_THRESHOLD {
            // Large regions: Use parallel processing
            label_1_regions.par_iter()
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
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            // Small regions: Use sequential processing (less overhead)
            let mut white = 0usize;
            let mut total = 0usize;

            for l1_bbox in label_1_regions {
                let [l1_x1, l1_y1, l1_x2, l1_y2] = *l1_bbox;

                let crop_x1 = l1_x1.max(x1).max(0).min(img_width) as u32;
                let crop_y1 = l1_y1.max(y1).max(0).min(img_height) as u32;
                let crop_x2 = l1_x2.min(x2).max(0).min(img_width) as u32;
                let crop_y2 = l1_y2.min(y2).max(0).min(img_height) as u32;

                if crop_x2 <= crop_x1 || crop_y2 <= crop_y1 {
                    continue;
                }

                for y in crop_y1..crop_y2 {
                    for x in crop_x1..crop_x2 {
                        total += 1;
                        let pixel = rgb.get_pixel(x, y);
                        if pixel[0] >= 240 && pixel[1] >= 240 && pixel[2] >= 240 {
                            white += 1;
                        }
                    }
                }
            }

            (white, total)
        };

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

    /// Clean regions by white-filling label1 areas (for use_mask=false mode)
    /// This allows skipping Phase 3 entirely when mask mode is disabled.
    ///
    /// Runs in parallel using rayon for performance.
    fn clean_regions_early(
        img: &DynamicImage,
        regions: &[CategorizedRegion],
        blur_free_text: bool,
        blur_radius: f32,
    ) -> Vec<(usize, Vec<u8>)> {
        regions.par_iter()
            .filter_map(|region| {
                match Self::clean_single_region(img, region, blur_free_text, blur_radius) {
                    Ok(bytes) => Some((region.region_id, bytes)),
                    Err(e) => {
                        debug!("Failed to clean region {}: {:?}", region.region_id, e);
                        None
                    }
                }
            })
            .collect()
    }

    /// Clean a single region by white-filling label1 areas
    fn clean_single_region(
        img: &DynamicImage,
        region: &CategorizedRegion,
        blur_free_text: bool,
        blur_radius: f32,
    ) -> Result<Vec<u8>> {
        let [x1, y1, x2, y2] = region.bbox;
        let width = (x2 - x1).max(1) as u32;
        let height = (y2 - y1).max(1) as u32;

        // Crop region from original
        let mut cropped = img.crop_imm(x1 as u32, y1 as u32, width, height).to_rgba8();

        // Label 2 (free text): blur or fill entirely with white
        if region.label == 2 {
            let cleaned = if blur_free_text {
                image::imageops::blur(&cropped, blur_radius)
            } else {
                image::ImageBuffer::from_pixel(width, height, Rgba([255, 255, 255, 255]))
            };
            return Self::encode_png_rgba(cleaned);
        }

        // Label 0/1: Fill all label1 regions with white
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

        Self::encode_png_rgba(cropped)
    }

    /// Encode RGBA image to PNG bytes
    fn encode_png_rgba(img: image::RgbaImage) -> Result<Vec<u8>> {
        let mut png_bytes = Vec::new();
        DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
            .context("Failed to encode PNG")?;
        Ok(png_bytes)
    }
}
