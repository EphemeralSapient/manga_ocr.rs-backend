// Phase 1: Detection & Categorization Pipeline
//
// Uses FPN text detector for segmentation (CPU-only)

use anyhow::{Context, Result};
use image::DynamicImage;
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
    /// 2. Run FPN text detector for segmentation
    /// 3. Validate label 1 regions are within label 0
    /// 4. Categorize simple vs complex backgrounds
    ///
    /// # Arguments
    /// * `_use_mask` - Ignored (segmentation always runs)
    /// * `_mask_mode` - Ignored (only "fast" mode is supported)
    /// * `target_size_override` - Optional target size override for detection model
    /// * `filter_orphan_regions` - If true, discard label 1 regions not within any label 0
    /// * `_blur_free_text` - Ignored (no early cleaning)
    #[instrument(skip(self, image_data), fields(
        page_index = image_data.index,
        filename = %image_data.filename
    ))]
    pub async fn execute(
        &self,
        image_data: &ImageData,
        _use_mask: bool,
        _mask_mode: Option<&str>,
        target_size_override: Option<u32>,
        filter_orphan_regions: bool,
        _blur_free_text: bool,
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

        // Step 1: Run detection first
        let detection_start = std::time::Instant::now();
        let (label_0_raw, label_1_raw, label_2_raw) = self.detector
            .detect_all_labels(img, image_data.index, target_size_override)
            .await
            .context("Failed to detect regions")?;
        let detection_ms = detection_start.elapsed().as_secs_f64() * 1000.0;

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

        // Step 2: Validate and categorize
        let (regions, warnings) = self.categorize_regions(
            img, &label_0, &label_1, &label_2, image_data.index, filter_orphan_regions,
        )?;

        // Step 3: Clean each label_1 region (crop, detect text, fill with white)
        // Only clean within label_1_regions bounds, not the entire region bbox
        let cleaning_start = std::time::Instant::now();
        let mut regions_to_clean: Vec<(usize, DynamicImage, [i32; 4])> = Vec::new();

        for region in &regions {
            // For each label_1 bbox within this region, crop and clean
            for l1_bbox in &region.label_1_regions {
                let [x1, y1, x2, y2] = *l1_bbox;
                let x1 = x1.max(0) as u32;
                let y1 = y1.max(0) as u32;
                let x2 = (x2 as u32).min(img.width());
                let y2 = (y2 as u32).min(img.height());

                if x2 > x1 && y2 > y1 {
                    let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
                    regions_to_clean.push((region.region_id, crop, *l1_bbox));
                }
            }
        }

        let cleaned_regions = self.segmenter
            .clean_regions_batch(&regions_to_clean)
            .await
            .context("Failed to clean regions")?;
        let cleaning_ms = cleaning_start.elapsed().as_secs_f64() * 1000.0;

        info!(
            "⏱️  Detection: {:.0}ms | Cleaning: {:.0}ms ({} regions) | L0={}, L1={}, L2={}",
            detection_ms, cleaning_ms, regions.len(),
            label_0.len(), label_1.len(), label_2.len()
        );

        Ok(Phase1Output {
            page_index: image_data.index,
            filename: image_data.filename.clone(),
            width: image_data.width,
            height: image_data.height,
            regions,
            cleaned_regions,
            ocr_results: Vec::new(),
            validation_warnings: warnings,
        })
    }

    /// Execute Phase 1 detection only (without segmentation)
    /// Returns detection results + decoded images for later segmentation
    ///
    /// This allows Phase 2 to start immediately while segmentation runs in background.
    ///
    /// # Arguments
    /// * `target_size_override` - Optional target size override for detection model
    /// * `filter_orphan_regions` - If true, discard label 1 regions not within any label 0
    /// * `_use_mask` - Ignored (segmentation always runs later)
    /// * `_blur_free_text` - Ignored (no early cleaning)
    pub async fn execute_detection_only(
        &self,
        images: &[ImageData],
        merge_img: bool,
        target_size_override: Option<u32>,
        filter_orphan_regions: bool,
        _use_mask: bool,
        _blur_free_text: bool,
    ) -> Result<(Vec<Phase1Output>, Vec<DynamicImage>)> {
        if images.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let mode_str = if merge_img { "BATCH" } else { "PARALLEL" };
        debug!("Phase 1 DETECTION: Processing {} images ({})", images.len(), mode_str);

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

        // Run detection: batched or parallel
        let batch_detections: Vec<(Vec<_>, Vec<_>, Vec<_>)> = if merge_img {
            // Batch mode: single inference for all images
            let batch_refs: Vec<(&DynamicImage, usize)> = decoded_images
                .iter()
                .zip(images.iter())
                .map(|(img, data)| (img, data.index))
                .collect();

            self.detector.detect_all_labels_batch(&batch_refs, target_size_override).await
                .context("Batch detection failed")?
        } else {
            // Parallel mode: multiple concurrent inferences
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

        let detection_ms = detection_start.elapsed().as_secs_f64() * 1000.0;
        info!("⏱️  Detection ({}): {:.0}ms for {} images ({:.1}ms/img)",
              mode_str, detection_ms, images.len(), detection_ms / images.len() as f64);

        // Create Phase1Output with empty segmentation masks (will be filled later)
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

            // Empty cleaned_regions placeholder (filled later by complete_cleaning)
            outputs.push(Phase1Output {
                page_index: image_data.index,
                filename: image_data.filename.clone(),
                width: image_data.width,
                height: image_data.height,
                regions,
                cleaned_regions: Vec::new(),
                ocr_results: Vec::new(),
                validation_warnings,
            });
        }

        Ok((outputs, decoded_images))
    }

    /// Run text cleaning and update Phase1Output with cleaned region images
    /// This is called after detection to fill in the cleaned_regions
    /// Only cleans within label_1_regions bounds, not entire region bbox
    ///
    /// # Arguments
    /// * `valid_region_ids` - Optional set of region IDs that should be cleaned.
    ///   Only regions with OCR-validated text should be cleaned. If None, all regions are cleaned.
    pub async fn complete_cleaning(
        &self,
        outputs: &mut [Phase1Output],
        decoded_images: &[DynamicImage],
        valid_region_ids: Option<&std::collections::HashSet<usize>>,
    ) -> Result<()> {
        let clean_start = std::time::Instant::now();
        let mut total_regions = 0usize;

        // Process each image - crop and clean each label_1 region
        for (i, img) in decoded_images.iter().enumerate() {
            let mut regions_to_clean: Vec<(usize, DynamicImage, [i32; 4])> = Vec::new();

            for region in &outputs[i].regions {
                // Clean each label_1 bbox within this region
                for l1_bbox in &region.label_1_regions {
                    let [x1, y1, x2, y2] = *l1_bbox;
                    let x1 = x1.max(0) as u32;
                    let y1 = y1.max(0) as u32;
                    let x2 = (x2 as u32).min(img.width());
                    let y2 = (y2 as u32).min(img.height());

                    if x2 > x1 && y2 > y1 {
                        let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
                        regions_to_clean.push((region.region_id, crop, *l1_bbox));
                    }
                }
            }

            total_regions += regions_to_clean.len();

            // Use filtered cleaning if valid_region_ids provided
            let cleaned = self.segmenter
                .clean_regions_batch_filtered(&regions_to_clean, valid_region_ids)
                .await
                .context("Failed to clean regions")?;

            outputs[i].cleaned_regions = cleaned;
        }

        let clean_ms = clean_start.elapsed().as_secs_f64() * 1000.0;
        info!("⏱️  Cleaning: {:.0}ms for {} images ({} label_1 regions, {:.1}ms/img)",
              clean_ms, decoded_images.len(), total_regions, clean_ms / decoded_images.len().max(1) as f64);

        Ok(())
    }

    /// Run text cleaning AND OCR in parallel, updating Phase1Output with both
    /// 
    /// OPTIMIZATION: OCR and cleaning run in parallel instead of sequentially.
    /// OCR results are stored in Phase1Output for Phase 2 to use directly,
    /// avoiding redundant image OCR via Gemini API.
    /// 
    /// # Arguments
    /// * `models_dir` - Path to models directory containing OCR model
    /// * `valid_region_ids` - Optional set of region IDs to filter (used for cleaning)
    /// 
    /// # Flow:
    /// 1. Crop all label_1 regions from images
    /// 2. Fork into parallel tasks:
    ///    a) Run OCR on each cropped region → store Japanese text
    ///    b) Run text_cleaner on each region → store cleaned PNG
    /// 3. Combine results into Phase1Output
    pub async fn complete_cleaning_with_ocr(
        &self,
        outputs: &mut [Phase1Output],
        decoded_images: &[DynamicImage],
        models_dir: &std::path::Path,
        valid_region_ids: Option<&std::collections::HashSet<usize>>,
    ) -> Result<()> {
        use crate::services::ocr::{get_ocr_service, is_ocr_available};

        let start = std::time::Instant::now();

        // Check OCR availability
        let ocr_available = is_ocr_available(models_dir);
        if !ocr_available {
            warn!("OCR models not available, falling back to cleaning-only mode");
            return self.complete_cleaning(outputs, decoded_images, valid_region_ids).await;
        }

        let ocr_service = get_ocr_service(models_dir)?;
        let mut total_regions = 0usize;
        let mut ocr_success_count = 0usize;

        // Process each image
        for (i, img) in decoded_images.iter().enumerate() {
            // Collect all regions to process with their crops
            let mut region_crops: Vec<(usize, DynamicImage, [i32; 4])> = Vec::new();

            for region in &outputs[i].regions {
                for l1_bbox in &region.label_1_regions {
                    let [x1, y1, x2, y2] = *l1_bbox;
                    let x1 = x1.max(0) as u32;
                    let y1 = y1.max(0) as u32;
                    let x2 = (x2 as u32).min(img.width());
                    let y2 = (y2 as u32).min(img.height());

                    if x2 > x1 && y2 > y1 {
                        let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
                        region_crops.push((region.region_id, crop, *l1_bbox));
                    }
                }
            }

            total_regions += region_crops.len();

            // PARALLEL PROCESSING: Run OCR and cleaning simultaneously
            // Clone crops for OCR (cleaning consumes the crops)
            let ocr_crops: Vec<(usize, DynamicImage, [i32; 4])> = region_crops
                .iter()
                .map(|(id, crop, bbox)| (*id, crop.clone(), *bbox))
                .collect();

            // Fork into parallel tasks
            let (cleaned_result, ocr_results) = tokio::join!(
                // Task 1: Text cleaning
                async {
                    self.segmenter
                        .clean_regions_batch_filtered(&region_crops, valid_region_ids)
                        .await
                        .context("Failed to clean regions")
                },
                // Task 2: OCR recognition (run on blocking threadpool)
                async {
                    let ocr_service_clone = ocr_service.clone();
                    tokio::task::spawn_blocking(move || {
                        let mut results = Vec::new();
                        for (region_id, crop, _bbox) in ocr_crops {
                            match ocr_service_clone.recognize(&crop) {
                                Ok((text, confidence)) => {
                                    if !text.trim().is_empty() {
                                        results.push((region_id, text, confidence));
                                    }
                                }
                                Err(e) => {
                                    debug!("OCR failed for region {}: {:?}", region_id, e);
                                }
                            }
                        }
                        results
                    })
                    .await
                    .context("OCR task panicked")
                }
            );

            outputs[i].cleaned_regions = cleaned_result?;

            let ocr_results = ocr_results?;
            ocr_success_count += ocr_results.len();
            outputs[i].ocr_results = ocr_results;
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "⏱️  Parallel OCR+Cleaning: {:.0}ms for {} images ({} regions, {} OCR success)",
            elapsed_ms, decoded_images.len(), total_regions, ocr_success_count
        );

        Ok(())
    }

    /// Execute Phase 1 on multiple images using batch inference
    ///
    /// This is more efficient than calling execute() multiple times when merge_img is enabled.
    /// Batch execution: supports both batched and parallel modes
    /// - merge_img=true: Single batched ONNX inference (faster for multiple images)
    /// - merge_img=false: Individual ONNX inferences in parallel (uses multiple sessions)
    ///
    /// # Arguments
    /// * `_use_mask` - Ignored (segmentation always runs)
    /// * `target_size_override` - Optional target size override for detection model
    /// * `filter_orphan_regions` - If true, discard label 1 regions not within any label 0
    /// * `_blur_free_text` - Ignored (no early cleaning)
    pub async fn execute_batch(
        &self,
        images: &[ImageData],
        _use_mask: bool,
        merge_img: bool,
        target_size_override: Option<u32>,
        filter_orphan_regions: bool,
        _blur_free_text: bool,
    ) -> Result<Vec<Phase1Output>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mode_str = if merge_img { "BATCH" } else { "PARALLEL" };
        debug!("Phase 1 {} MODE: Processing {} images", mode_str, images.len());
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

        // Run detection: batched or parallel
        let detection_start = std::time::Instant::now();
        let batch_detections: Vec<(Vec<_>, Vec<_>, Vec<_>)> = if merge_img {
            // Batch mode: single inference for all images
            let batch_refs: Vec<(&DynamicImage, usize)> = decoded_images
                .iter()
                .zip(images.iter())
                .map(|(img, data)| (img, data.index))
                .collect();

            self.detector.detect_all_labels_batch(&batch_refs, target_size_override).await
                .context("Batch detection failed")?
        } else {
            // Parallel mode: multiple concurrent inferences
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
        let detection_ms = detection_start.elapsed().as_secs_f64() * 1000.0;

        // Process detections and categorize regions
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

            // Create output with empty cleaned_regions (filled next)
            outputs.push(Phase1Output {
                page_index: image_data.index,
                filename: image_data.filename.clone(),
                width: image_data.width,
                height: image_data.height,
                regions,
                cleaned_regions: Vec::new(),
                ocr_results: Vec::new(),
                validation_warnings,
            });
        }

        // Clean each label_1 region (not entire region bbox)
        let clean_start = std::time::Instant::now();
        let mut total_regions = 0usize;

        for (i, img) in decoded_images.iter().enumerate() {
            let mut regions_to_clean: Vec<(usize, DynamicImage, [i32; 4])> = Vec::new();

            for region in &outputs[i].regions {
                // Clean each label_1 bbox within this region
                for l1_bbox in &region.label_1_regions {
                    let [x1, y1, x2, y2] = *l1_bbox;
                    let x1 = x1.max(0) as u32;
                    let y1 = y1.max(0) as u32;
                    let x2 = (x2 as u32).min(img.width());
                    let y2 = (y2 as u32).min(img.height());

                    if x2 > x1 && y2 > y1 {
                        let crop = img.crop_imm(x1, y1, x2 - x1, y2 - y1);
                        regions_to_clean.push((region.region_id, crop, *l1_bbox));
                    }
                }
            }

            total_regions += regions_to_clean.len();

            let cleaned = self.segmenter
                .clean_regions_batch(&regions_to_clean)
                .await
                .context("Failed to clean regions")?;

            outputs[i].cleaned_regions = cleaned;
        }
        let cleaning_ms = clean_start.elapsed().as_secs_f64() * 1000.0;

        info!("⏱️  Detection ({}): {:.0}ms | Cleaning: {:.0}ms ({} label_1 regions) | Total: {:.0}ms for {} images",
              mode_str, detection_ms, cleaning_ms, total_regions,
              batch_start.elapsed().as_secs_f64() * 1000.0, images.len());

        debug!(
            "✓ Phase 1 completed in {:.0}ms for {} images ({:.1}ms/image)",
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
        let region_id_counter = AtomicUsize::new(page_index * 10000);

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

            // Categorize background
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
        for l1 in label_1 {
            let within_any_l0 = label_0.iter().any(|l0| self.is_bbox_within(l1.bbox, l0.bbox));
            if !within_any_l0 {
                if filter_orphan_regions {
                    warn!(
                        "Label 1 region {:?} not within any label 0 - discarding",
                        l1.bbox
                    );
                    warnings.push(format!(
                        "Label 1 region {:?} not within any label 0",
                        l1.bbox
                    ));
                } else {
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
                label_1_regions: vec![l2.bbox],
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

    /// Fast parallel background classification
    fn classify_background_fast(
        rgb: &image::RgbImage,
        label_0_bbox: [i32; 4],
        label_1_regions: &[[i32; 4]],
        threshold: f32,
    ) -> BackgroundType {
        let [x1, y1, x2, y2] = label_0_bbox;
        let img_width = rgb.width() as i32;
        let img_height = rgb.height() as i32;

        let total_area: i32 = label_1_regions
            .iter()
            .map(|bbox| {
                let width = (bbox[2] - bbox[0]).max(0);
                let height = (bbox[3] - bbox[1]).max(0);
                width * height
            })
            .sum();

        const PARALLEL_THRESHOLD: i32 = 10_000;

        let (white_pixels, total_pixels): (usize, usize) = if total_area > PARALLEL_THRESHOLD {
            label_1_regions.par_iter()
                .map(|l1_bbox| {
                    let [l1_x1, l1_y1, l1_x2, l1_y2] = *l1_bbox;

                    let crop_x1 = l1_x1.max(x1).max(0).min(img_width) as u32;
                    let crop_y1 = l1_y1.max(y1).max(0).min(img_height) as u32;
                    let crop_x2 = l1_x2.min(x2).max(0).min(img_width) as u32;
                    let crop_y2 = l1_y2.min(y2).max(0).min(img_height) as u32;

                    if crop_x2 <= crop_x1 || crop_y2 <= crop_y1 {
                        return (0, 0);
                    }

                    let mut local_white = 0usize;
                    let mut local_total = 0usize;

                    for y in crop_y1..crop_y2 {
                        for x in crop_x1..crop_x2 {
                            local_total += 1;
                            let pixel = rgb.get_pixel(x, y);
                            if pixel[0] >= 240 && pixel[1] >= 240 && pixel[2] >= 240 {
                                local_white += 1;
                            }
                        }
                    }
                    (local_white, local_total)
                })
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
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
}
