// Batch Orchestrator: Main workflow coordinator

use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine};
use futures::future::join_all;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::{debug, info, instrument, warn};

use crate::services::translation::api_client::ApiClient;
use crate::core::config::Config;
use crate::services::detection::DetectionService;
use crate::services::font_manager::FontManager;
use crate::phases::phase1::Phase1Pipeline;
use crate::phases::phase2::Phase2Pipeline;
use crate::phases::phase3::Phase3Pipeline;
use crate::phases::phase4::Phase4Pipeline;
use crate::services::segmentation::SegmentationService;
use crate::services::translation::cache::TranslationCache;
use crate::core::types::{
    BatchAnalytics, BatchResult, ImageBatch, ImageData, PageResult, PerformanceMetrics,
    ProcessingConfig,
};
use crate::utils::image_ops::draw_debug_bboxes;

/// Main batch orchestrator
pub struct BatchOrchestrator {
    config: Arc<Config>,
    phase1: Arc<Phase1Pipeline>,
    phase2: Arc<Phase2Pipeline>,
    phase3: Arc<Phase3Pipeline>,
    phase4: Arc<Phase4Pipeline>,
    font_manager: Arc<FontManager>,
    batch_semaphore: Arc<Semaphore>,
    backend_type: String,
}

impl BatchOrchestrator {
    /// Create new batch orchestrator
    #[instrument(skip(config))]
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        info!("Initializing services...");

        // Initialize ONNX services (with warmup)
        let detector = Arc::new(DetectionService::new(config.clone()).await?);
        let segmenter = Arc::new(SegmentationService::new(config.clone()).await?);

        // Warmup OCR model (single instance, kept ready)
        let models_dir = std::path::Path::new("models");
        if crate::services::ocr::is_ocr_available(models_dir) {
            info!("🔄 Warming up OCR model...");
            if let Err(e) = crate::services::ocr::warmup_ocr_service(models_dir) {
                warn!("OCR warmup failed (will retry on first use): {:?}", e);
            } else {
                info!("✓ OCR model ready");
            }
        }

        let api_client = Arc::new(ApiClient::new(config.clone(), None, None)?);
        let cache = Arc::new(TranslationCache::new(config.cache_dir(), None, None, None).await?);
        let font_manager = Arc::new(FontManager::new(config.cache_dir())?);

        // Log cache stats
        let (_cache_entries, _cache_size_mb) = cache.stats().await;

        // Store backend type for health endpoint
        let backend_type = detector.device_type().to_string();

        // Initialize phase pipelines
        let phase1 = Arc::new(Phase1Pipeline::new(
            config.clone(),
            detector,
            segmenter,
        ));
        let phase2 = Arc::new(Phase2Pipeline::new(config.clone(), api_client, cache));
        let phase3 = Arc::new(Phase3Pipeline::new(config.clone()));
        let phase4 = Arc::new(Phase4Pipeline::new(config.clone()));

        let batch_semaphore = Arc::new(Semaphore::new(config.max_concurrent_batches()));

        info!("✓ All models ready (detector ~42MB, text cleaner ~11MB, OCR ~13MB)");

        Ok(Self {
            config,
            phase1,
            phase2,
            phase3,
            phase4,
            font_manager,
            batch_semaphore,
            backend_type,
        })
    }

    /// Get the backend type (e.g., "DirectML+CPU", "CUDA", "TensorRT", "CPU")
    pub fn backend_type(&self) -> &str {
        &self.backend_type
    }

    /// Format ProcessingConfig for logging without exposing sensitive data
    fn format_config_for_logging(config: &ProcessingConfig) -> String {
        format!(
            "model={}, banana_mode={}, cache={}, merge_img={}, sessions={}, include_free_text={}, text_stroke={}, blur_bg={}, target_size={}, filter_orphans={}, api_keys={}, local_ocr={}, use_cerebras={}",
            config.ocr_translation_model.as_deref().unwrap_or("default"),
            config.banana_mode.unwrap_or(false),
            config.cache_enabled.unwrap_or(true),
            config.merge_img.unwrap_or(false),
            config.session_limit.map(|s| s.to_string()).unwrap_or_else(|| "default".to_string()),
            config.include_free_text.unwrap_or(false),
            config.text_stroke.unwrap_or(false),
            config.blur_free_text_bg.unwrap_or(false),
            config.target_size.map(|s| if s == 0 { "source".to_string() } else { s.to_string() }).unwrap_or_else(|| "default".to_string()),
            config.filter_orphan_regions.unwrap_or(false),
            config.api_keys.as_ref().map(|keys| format!("[{} keys]", keys.len())).unwrap_or_else(|| "[default]".to_string()),
            config.ocr_enabled.unwrap_or(false),
            config.use_cerebras.unwrap_or(false)
        )
    }

    /// Process multiple batches of images
    ///
    /// # Workflow:
    /// 1. Divide images into batches of N (batch_size_n)
    /// 2. Process batches in parallel (up to max_concurrent_batches)
    /// 3. Within each batch, process images sequentially through phases 1-4
    ///
    /// # Arguments:
    /// * `images` - All images to process
    /// * `config` - Processing configuration
    ///
    /// # Returns:
    /// BatchResult with all results and analytics
    #[instrument(skip(self, images, config), fields(total_images = images.len()))]
    pub async fn process_batch(
        &self,
        images: Vec<ImageData>,
        config: &ProcessingConfig,
    ) -> Result<BatchResult> {
        let start_time = Instant::now();
        let total_images = images.len();

        info!("Processing {} images", total_images);
        info!("Config: {}", Self::format_config_for_logging(config));

        // Divide images into batches of N
        let batch_size_n = self.config.batch_size_n();
        let batches: Vec<ImageBatch> = images
            .chunks(batch_size_n)
            .enumerate()
            .map(|(i, chunk)| ImageBatch {
                batch_id: format!("batch_{}", i),
                images: chunk.to_vec(),
            })
            .collect();

        info!(
            "Created {} batches of {} images each",
            batches.len(),
            batch_size_n
        );

        // Process batches in parallel
        // NEW ARCHITECTURE: Process all batches phase-by-phase globally
        // Phase 1 for all → Phase 2 for all (split across keys) → Phase 3 for all → Phase 4 for all

        let config = Arc::new(config.clone());
        let app_config = Arc::clone(&self.config);
        let merge_img = config.merge_img.unwrap_or(false);
        let use_mask = config.use_mask.unwrap_or(true);
        let include_free_text = config.include_free_text.unwrap_or(false);
        let filter_orphan_regions = config.filter_orphan_regions.unwrap_or(false);
        let blur_free_text = config.blur_free_text_bg.unwrap_or(app_config.blur_free_text());

        info!("Processing {} batches phase-by-phase: Phase 1 → Phase 2 (global) → Phase 3{} → Phase 4",
              batches.len(),
              if use_mask { "" } else { " (skipped - early cleaned)" });

        // ===== PHASE 1: DETECTION ONLY (Segmentation runs in background) =====
        info!("Starting Phase 1 detection for all {} batches", batches.len());

        let phase1_start = Instant::now();
        let mut all_phase1_data: Vec<(ImageData, crate::core::types::Phase1Output)> = Vec::new();
        let mut all_decoded_images: Vec<image::DynamicImage> = Vec::new();
        let mut phase1_metrics = PerformanceMetrics::default();

        // Check if using DirectML - if so, use sequential processing
        let is_directml = self.phase1.is_directml();

        if is_directml {
            // DirectML: Sequential processing through single session (no parallelism)
            info!("🔄 DirectML detected: Processing {} batches sequentially (detection only)", batches.len());

            for batch in batches.into_iter() {
                match self.phase1.execute_detection_only(&batch.images, merge_img, config.target_size, filter_orphan_regions, use_mask, blur_free_text).await {
                    Ok((mut outputs, decoded_images)) => {
                        for (i, output) in outputs.iter_mut().enumerate() {
                            // Filter out label 2 if not included
                            if !include_free_text {
                                output.regions.retain(|r| r.label != 2);
                            }

                            // Count metrics
                            phase1_metrics.total_regions += output.regions.len();
                            for region in &output.regions {
                                match region.label {
                                    0 => phase1_metrics.label_0_count += 1,
                                    1 => phase1_metrics.label_1_count += 1,
                                    2 => phase1_metrics.label_2_count += 1,
                                    _ => {}
                                }
                                if region.background_type == crate::core::types::BackgroundType::Simple {
                                    phase1_metrics.simple_bg_count += 1;
                                } else {
                                    phase1_metrics.complex_bg_count += 1;
                                }
                            }

                            all_phase1_data.push((batch.images[i].clone(), output.clone()));
                        }
                        all_decoded_images.extend(decoded_images);
                    }
                    Err(e) => {
                        tracing::error!("Phase 1 detection failed: {:?}", e);
                    }
                }
            }
        } else {
            // Non-DirectML: Parallel processing
            let num_batches = batches.len();
            if num_batches > 1 {
                info!("⚡ Non-DirectML backend: Processing {} batches in parallel (detection only)", num_batches);
            }

            let mut phase1_tasks = Vec::new();
            for batch in batches {
                let phase1 = Arc::clone(&self.phase1);
                let images = batch.images.clone();
                let target_size = config.target_size;

                let task = tokio::spawn(async move {
                    let (outputs, decoded_images) = phase1.execute_detection_only(&images, merge_img, target_size, filter_orphan_regions, use_mask, blur_free_text).await?;
                    Ok::<_, anyhow::Error>((images, outputs, decoded_images))
                });

                phase1_tasks.push(task);
            }

            // Wait for all Phase 1 detection to complete
            for task in phase1_tasks {
                match task.await {
                    Ok(Ok((images, mut outputs, decoded_images))) => {
                        for (i, output) in outputs.iter_mut().enumerate() {
                            // Filter out label 2 if not included
                            if !include_free_text {
                                output.regions.retain(|r| r.label != 2);
                            }

                            // Count metrics
                            phase1_metrics.total_regions += output.regions.len();
                            for region in &output.regions {
                                match region.label {
                                    0 => phase1_metrics.label_0_count += 1,
                                    1 => phase1_metrics.label_1_count += 1,
                                    2 => phase1_metrics.label_2_count += 1,
                                    _ => {}
                                }
                                if region.background_type == crate::core::types::BackgroundType::Simple {
                                    phase1_metrics.simple_bg_count += 1;
                                } else {
                                    phase1_metrics.complex_bg_count += 1;
                                }
                            }

                            all_phase1_data.push((images[i].clone(), output.clone()));
                        }
                        all_decoded_images.extend(decoded_images);
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Phase 1 detection failed: {:?}", e);
                    }
                    Err(e) => {
                        tracing::error!("Phase 1 detection task panicked: {:?}", e);
                    }
                }
            }
        }

        phase1_metrics.phase1_time = phase1_start.elapsed();
        info!(
            "Phase 1 detection complete for all {} pages in {:.2}ms",
            all_phase1_data.len(),
            phase1_metrics.phase1_time.as_secs_f64() * 1000.0
        );

        // ===== L1_DEBUG MODE: Early return with bbox visualization =====
        if config.l1_debug.unwrap_or(false) {
            info!("🔍 L1_DEBUG mode enabled - returning bbox visualization (use_mask={})", use_mask);

            let mut debug_results = Vec::new();

            for (i, (image_data, phase1_output)) in all_phase1_data.iter().enumerate() {
                // Get the decoded image
                let mut decoded_img: image::DynamicImage = if let Some(img) = all_decoded_images.get(i) {
                    img.clone()
                } else if let Some(ref img) = image_data.decoded_image {
                    (**img).clone()
                } else {
                    match image::load_from_memory(&image_data.image_bytes) {
                        Ok(img) => img,
                        Err(e) => {
                            tracing::error!("Failed to decode image for debug: {:?}", e);
                            continue;
                        }
                    }
                };

                // Collect label 0 and label 1 bboxes
                let mut label_0_bboxes: Vec<[i32; 4]> = Vec::new();
                let mut label_1_bboxes: Vec<[i32; 4]> = Vec::new();

                for region in &phase1_output.regions {
                    match region.label {
                        0 => {
                            label_0_bboxes.push(region.bbox);
                            // Also add all label_1_regions contained in this label 0
                            for l1_bbox in &region.label_1_regions {
                                label_1_bboxes.push(*l1_bbox);
                            }
                        }
                        1 => {
                            label_1_bboxes.push(region.bbox);
                        }
                        _ => {}
                    }
                }

                // For non-mask mode: Apply label 1 white-fill cleaning before drawing bboxes
                // This shows what the cleaning would look like
                if !use_mask {
                    let mut rgba_img = decoded_img.to_rgba8();
                    let (img_width, img_height) = rgba_img.dimensions();

                    // Fill all label 1 regions with white
                    for l1_bbox in &label_1_bboxes {
                        let [x1, y1, x2, y2] = *l1_bbox;
                        let x1 = x1.max(0).min(img_width as i32 - 1) as u32;
                        let y1 = y1.max(0).min(img_height as i32 - 1) as u32;
                        let x2 = x2.max(0).min(img_width as i32) as u32;
                        let y2 = y2.max(0).min(img_height as i32) as u32;

                        for y in y1..y2 {
                            for x in x1..x2 {
                                rgba_img.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
                            }
                        }
                    }

                    decoded_img = image::DynamicImage::ImageRgba8(rgba_img);
                    info!("🔍 Applied label 1 white-fill cleaning for debug visualization");
                }

                // Draw bboxes on the image (cleaned for non-mask, original for mask mode)
                match draw_debug_bboxes(&decoded_img, &label_0_bboxes, &label_1_bboxes) {
                    Ok(debug_bytes) => {
                        let base64_image = general_purpose::STANDARD.encode(&debug_bytes);
                        let data_url = format!("data:image/png;base64,{}", base64_image);

                        debug_results.push(PageResult {
                            index: image_data.index,
                            filename: image_data.filename.clone(),
                            success: true,
                            processing_time_ms: phase1_metrics.phase1_time.as_secs_f64() * 1000.0,
                            error: None,
                            data_url: Some(data_url),
                        });

                        info!(
                            "🔍 Page {}: {} label0 boxes (BLUE), {} label1 boxes (RED)",
                            image_data.index,
                            label_0_bboxes.len(),
                            label_1_bboxes.len()
                        );
                    }
                    Err(e) => {
                        tracing::error!("Failed to draw debug bboxes: {:?}", e);
                        debug_results.push(PageResult {
                            index: image_data.index,
                            filename: image_data.filename.clone(),
                            success: false,
                            processing_time_ms: 0.0,
                            error: Some(format!("Debug bbox drawing failed: {}", e)),
                            data_url: None,
                        });
                    }
                }
            }

            // Cleanup ONNX sessions
            self.phase1.cleanup_detection_sessions();

            let total_time = start_time.elapsed();
            let successful = debug_results.iter().filter(|r| r.success).count();
            let failed = debug_results.len() - successful;

            info!(
                "🔍 L1_DEBUG complete: {} pages visualized in {:.2}s",
                successful,
                total_time.as_secs_f64()
            );

            return Ok(BatchResult {
                total: total_images,
                successful,
                failed,
                processing_time_ms: total_time.as_secs_f64() * 1000.0,
                average_time_per_page_ms: total_time.as_secs_f64() * 1000.0 / total_images.max(1) as f64,
                analytics: BatchAnalytics {
                    total_images,
                    total_regions: phase1_metrics.total_regions,
                    simple_bg_count: phase1_metrics.simple_bg_count,
                    complex_bg_count: phase1_metrics.complex_bg_count,
                    label_0_count: phase1_metrics.label_0_count,
                    label_1_count: phase1_metrics.label_1_count,
                    label_2_count: phase1_metrics.label_2_count,
                    validation_warnings: 0,
                    api_calls_simple: 0,
                    api_calls_complex: 0,
                    api_calls_banana: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_hits: 0,
                    cache_misses: 0,
                    phase1_time_ms: phase1_metrics.phase1_time.as_secs_f64() * 1000.0,
                    phase2_time_ms: 0.0,
                    phase3_time_ms: 0.0,
                    phase4_time_ms: 0.0,
                    total_time_ms: total_time.as_secs_f64() * 1000.0,
                    inference_time_ms: 0.0,
                    api_wait_time_ms: 0.0,
                },
                results: debug_results,
            });
        }

        // ===== SEQUENTIAL: OCR first, then cleaning (single OCR pass) =====
        // Phase 2 OCR runs first to determine which regions have valid text.
        // Cleaning only processes regions with OCR-validated text (prevents over-cleaning).
        info!("🔄 Starting OCR → Cleaning (single OCR pass)...");
        let ocr_clean_start = Instant::now();

        // Prepare Phase 2 config
        let ocr_model_override = config.ocr_translation_model.as_deref().map(|s| s.to_string());
        let banana_model_override = config.banana_image_model.as_deref().map(|s| s.to_string());
        let banana_mode = config.banana_mode.unwrap_or(app_config.banana_mode_enabled());
        let cache_enabled = config.cache_enabled.unwrap_or(true);
        let custom_api_keys = config.api_keys.clone();
        let target_language = config.target_language.as_deref().map(|s| s.to_string());
        let reuse_factor = config.reuse_factor.unwrap_or(4).clamp(1, 8);
        let ocr_enabled = config.ocr_enabled.unwrap_or(false);
        let use_cerebras = config.use_cerebras.unwrap_or(false);
        let cerebras_api_key = config.cerebras_api_key.as_deref().map(|s| s.to_string());

        // Warn if Cerebras is requested but local OCR is not enabled
        if use_cerebras && !ocr_enabled {
            warn!("⚠️  use_cerebras=true requires ocr_enabled=true. Falling back to Gemini for OCR+translation.");
        }

        // Step 1: Run Phase 2 (OCR + Translation) FIRST
        // This determines which regions have valid text
        let phase2_outputs = if ocr_enabled {
            let models_dir = std::path::Path::new("models");
            self.phase2
                .execute_batch_with_local_ocr(
                    &all_phase1_data,
                    models_dir,
                    use_cerebras,
                    cerebras_api_key.as_deref(),
                    target_language.as_deref(),
                    cache_enabled,
                    custom_api_keys.as_deref(),
                )
                .await
        } else {
            self.phase2
                .execute_batch(
                    &all_phase1_data,
                    ocr_model_override.as_deref(),
                    banana_model_override.as_deref(),
                    banana_mode,
                    cache_enabled,
                    custom_api_keys.as_deref(),
                    target_language.as_deref(),
                    reuse_factor,
                )
                .await
        };

        // Step 2: Extract valid region IDs from Phase 2 output
        // These are regions that have OCR text and translations
        let valid_region_ids: std::collections::HashSet<usize> = match &phase2_outputs {
            Ok(outputs) => {
                outputs
                    .iter()
                    .flat_map(|o| {
                        // Collect from both simple and complex bg translations
                        o.simple_bg_translations.iter().map(|(id, _)| *id)
                            .chain(o.complex_bg_translations.iter().map(|(id, _)| *id))
                    })
                    .collect()
            }
            Err(_) => std::collections::HashSet::new(),
        };
        debug!("OCR validated {} regions with text", valid_region_ids.len());

        // Step 3: Run cleaning with valid_region_ids filter
        // Only regions with OCR text are cleaned (prevents over-cleaning of art)
        let mut phase1_outputs_for_clean: Vec<_> = all_phase1_data
            .iter()
            .map(|(_, output)| output.clone())
            .collect();

        let cleaning_result = self.phase1
            .complete_cleaning(
                &mut phase1_outputs_for_clean,
                &all_decoded_images,
                if valid_region_ids.is_empty() { None } else { Some(&valid_region_ids) },
            )
            .await;

        // Process cleaning result
        let updated_phase1_outputs = match cleaning_result {
            Ok(()) => phase1_outputs_for_clean,
            Err(e) => {
                tracing::error!("Text cleaning failed: {:?}", e);
                all_phase1_data.iter().map(|(_, p1)| p1.clone()).collect()
            }
        };

        let ocr_clean_ms = ocr_clean_start.elapsed().as_secs_f64() * 1000.0;
        info!("✓ OCR + Cleaning completed in {:.2}ms", ocr_clean_ms);

        // CLEANUP: Free detection and text cleaner ONNX sessions
        self.phase1.cleanup_detection_sessions();
        self.phase1.cleanup_segmentation_sessions();

        phase1_metrics.phase2_time = ocr_clean_start.elapsed();
        info!(
            "Phase 2 complete for all {} pages in {:.2}ms",
            all_phase1_data.len(),
            phase1_metrics.phase2_time.as_secs_f64() * 1000.0
        );

        // Combine Phase 1 (with completed segmentation) and Phase 2 outputs
        let mut all_phase2_data: Vec<(ImageData, crate::core::types::Phase1Output, crate::core::types::Phase2Output)> = Vec::new();

        match phase2_outputs {
            Ok(outputs) => {
                // Collect metrics
                let mut total_simple_translations = 0;
                for (i, phase2_output) in outputs.into_iter().enumerate() {
                    let (image_data, _) = &all_phase1_data[i];
                    let updated_phase1 = &updated_phase1_outputs[i];

                    total_simple_translations += phase2_output.simple_bg_translations.len();
                    phase1_metrics.api_calls_banana += phase2_output.complex_bg_bananas.len();
                    if !phase2_output.complex_bg_translations.is_empty() {
                        phase1_metrics.api_calls_complex += 1;
                    }

                    all_phase2_data.push((image_data.clone(), updated_phase1.clone(), phase2_output));
                }

                // Count as 1 logical batch (split across keys)
                if total_simple_translations > 0 {
                    phase1_metrics.api_calls_simple = 1;
                }
            }
            Err(e) => {
                tracing::error!("Phase 2 global failed: {:?}", e);
                return Err(e);
            }
        }

        // ===== PHASE 3: ALL PAGES =====
        let phase3_start = Instant::now();
        let mut all_phase3_data = Vec::new();

        info!("Starting Phase 3 for all {} pages", all_phase2_data.len());
        {

            let mut phase3_tasks = Vec::new();
            for (image_data, phase1_output, phase2_output) in all_phase2_data.iter() {
                let phase3 = Arc::clone(&self.phase3);
                let config = Arc::clone(&config);
                let app_config = Arc::clone(&app_config);
                let image_data = image_data.clone();
                let phase1_output = phase1_output.clone();
                let phase2_output = phase2_output.clone();

                let task = tokio::spawn(async move {
                    let blur_free_text = config.blur_free_text_bg.unwrap_or(app_config.blur_free_text());
                    let use_mask = config.use_mask.unwrap_or(true);

                    let banana_region_ids: Vec<usize> = phase2_output
                        .complex_bg_bananas
                        .iter()
                        .map(|b| b.region_id)
                        .collect();

                    let phase3_output = phase3
                        .execute(&image_data, &phase1_output, &banana_region_ids, blur_free_text, use_mask)
                        .await?;

                    Ok::<_, anyhow::Error>((image_data, phase1_output, phase2_output, phase3_output))
                });

                phase3_tasks.push(task);
            }

            for task in phase3_tasks {
                match task.await {
                    Ok(Ok(data)) => all_phase3_data.push(data),
                    Ok(Err(e)) => tracing::error!("Phase 3 error: {:?}", e),
                    Err(e) => tracing::error!("Phase 3 task error: {:?}", e),
                }
            }
        }

        phase1_metrics.phase3_time = phase3_start.elapsed();
        info!(
            "Phase 3 complete for all {} pages in {:.2}ms",
            all_phase3_data.len(),
            phase1_metrics.phase3_time.as_secs_f64() * 1000.0
        );

        // ===== PHASE 4: ALL PAGES =====
        info!("Starting Phase 4 for all {} pages", all_phase3_data.len());
        let phase4_start = Instant::now();

        let mut phase4_tasks = Vec::new();
        for (image_data, phase1_output, phase2_output, phase3_output) in all_phase3_data.iter() {
            let phase4 = Arc::clone(&self.phase4);
            let config = Arc::clone(&config);
            let app_config = Arc::clone(&app_config);
            let image_data = image_data.clone();
            let phase1_output = phase1_output.clone();
            let phase2_output = phase2_output.clone();
            let phase3_output = phase3_output.clone();

            let task = tokio::spawn(async move {
                let text_stroke = config.text_stroke.unwrap_or(app_config.text_stroke_enabled());
                let font_family = config
                    .font_family
                    .as_deref()
                    .unwrap_or("arial");

                let phase4_output = phase4
                    .execute(&image_data, &phase1_output, &phase2_output, &phase3_output, font_family, text_stroke)
                    .await?;

                Ok::<_, anyhow::Error>((image_data, phase4_output))
            });

            phase4_tasks.push(task);
        }

        let mut all_results = Vec::new();
        for task in phase4_tasks {
            match task.await {
                Ok(Ok((image_data, phase4_output))) => {
                    // Convert image bytes to data URL
                    let base64_image = general_purpose::STANDARD.encode(&phase4_output.final_image_bytes);
                    let data_url = format!("data:image/png;base64,{}", base64_image);

                    all_results.push(crate::core::types::PageResult {
                        index: image_data.index,
                        filename: image_data.filename.clone(),
                        success: true,
                        processing_time_ms: 0.0, // Will be calculated
                        error: None,
                        data_url: Some(data_url),
                    });
                }
                Ok(Err(e)) => {
                    tracing::error!("Phase 4 error: {:?}", e);
                }
                Err(e) => {
                    tracing::error!("Phase 4 task error: {:?}", e);
                }
            }
        }

        phase1_metrics.phase4_time = phase4_start.elapsed();
        info!(
            "Phase 4 complete for all {} pages in {:.2}ms",
            all_results.len(),
            phase1_metrics.phase4_time.as_secs_f64() * 1000.0
        );

        let all_metrics = phase1_metrics;

        let total_time = start_time.elapsed();

        // Compute analytics
        let successful = all_results.iter().filter(|r| r.success).count();
        let failed = all_results.len() - successful;

        let analytics = BatchAnalytics {
            total_images,
            total_regions: all_metrics.total_regions,
            simple_bg_count: all_metrics.simple_bg_count,
            complex_bg_count: all_metrics.complex_bg_count,
            label_0_count: all_metrics.label_0_count,
            label_1_count: all_metrics.label_1_count,
            label_2_count: all_metrics.label_2_count,
            validation_warnings: all_metrics.validation_warnings,
            api_calls_simple: all_metrics.api_calls_simple,
            api_calls_complex: all_metrics.api_calls_complex,
            api_calls_banana: all_metrics.api_calls_banana,
            input_tokens: all_metrics.input_tokens,
            output_tokens: all_metrics.output_tokens,
            cache_hits: all_metrics.cache_hits,
            cache_misses: all_metrics.cache_misses,
            phase1_time_ms: all_metrics.phase1_time.as_secs_f64() * 1000.0,
            phase2_time_ms: all_metrics.phase2_time.as_secs_f64() * 1000.0,
            phase3_time_ms: all_metrics.phase3_time.as_secs_f64() * 1000.0,
            phase4_time_ms: all_metrics.phase4_time.as_secs_f64() * 1000.0,
            total_time_ms: total_time.as_secs_f64() * 1000.0,
            inference_time_ms: all_metrics.inference_time.as_secs_f64() * 1000.0,
            api_wait_time_ms: all_metrics.api_wait_time.as_secs_f64() * 1000.0,
        };

        info!(
            "Batch processing complete: {} successful, {} failed in {:.2}s",
            successful,
            failed,
            total_time.as_secs_f64()
        );

        Ok(BatchResult {
            total: total_images,
            successful,
            failed,
            processing_time_ms: total_time.as_secs_f64() * 1000.0,
            average_time_per_page_ms: total_time.as_secs_f64() * 1000.0
                / total_images.max(1) as f64,
            analytics,
            results: all_results,
        })
    }

    /// Process images with STREAMING PIPELINE architecture
    ///
    /// Each image flows through all phases independently:
    /// Image → Detection → [OCR + Cleaning in parallel] → Rendering → Done
    ///
    /// This provides better latency than batch processing since images don't
    /// wait for all other images to complete each phase.
    ///
    /// # Arguments:
    /// * `images` - All images to process
    /// * `config` - Processing configuration
    /// * `max_concurrent` - Maximum concurrent image pipelines (default: 4)
    #[instrument(skip(self, images, config), fields(total_images = images.len()))]
    pub async fn process_streaming(
        &self,
        images: Vec<ImageData>,
        config: &ProcessingConfig,
        max_concurrent: Option<usize>,
    ) -> Result<BatchResult> {
        let start_time = Instant::now();
        let total_images = images.len();
        let max_concurrent = max_concurrent.unwrap_or(4).max(1);

        info!("🚀 STREAMING PIPELINE: Processing {} images with max {} concurrent", total_images, max_concurrent);
        info!("Config: {}", Self::format_config_for_logging(config));

        let config = Arc::new(config.clone());
        let app_config = Arc::clone(&self.config);

        // Concurrency limiter
        let semaphore = Arc::new(Semaphore::new(max_concurrent));

        // Spawn a pipeline task for each image
        let mut tasks = Vec::new();

        for image_data in images {
            let phase1 = Arc::clone(&self.phase1);
            let phase2 = Arc::clone(&self.phase2);
            let phase3 = Arc::clone(&self.phase3);
            let phase4 = Arc::clone(&self.phase4);
            let config = Arc::clone(&config);
            let app_config = Arc::clone(&app_config);
            let semaphore = Arc::clone(&semaphore);

            let task = tokio::spawn(async move {
                // Acquire permit (limits concurrency)
                let _permit = semaphore.acquire().await.unwrap();
                let page_start = Instant::now();

                Self::process_single_image_pipeline(
                    phase1, phase2, phase3, phase4,
                    image_data, &config, &app_config,
                ).await.map(|result| (result, page_start.elapsed()))
            });

            tasks.push(task);
        }

        // Collect all results
        let mut all_results = Vec::new();
        let total_regions = 0;
        let mut simple_bg_count = 0;
        let mut complex_bg_count = 0;

        for task in tasks {
            match task.await {
                Ok(Ok((page_result, _duration))) => {
                    all_results.push(page_result);
                }
                Ok(Err(e)) => {
                    tracing::error!("Pipeline error: {:?}", e);
                }
                Err(e) => {
                    tracing::error!("Pipeline task panicked: {:?}", e);
                }
            }
        }

        // Cleanup ONNX sessions
        self.phase1.cleanup_detection_sessions();
        self.phase1.cleanup_segmentation_sessions();

        let total_time = start_time.elapsed();
        let successful = all_results.iter().filter(|r| r.success).count();
        let failed = all_results.len() - successful;

        info!(
            "🚀 STREAMING complete: {} successful, {} failed in {:.2}s ({:.0}ms/page avg)",
            successful, failed, total_time.as_secs_f64(),
            total_time.as_secs_f64() * 1000.0 / total_images.max(1) as f64
        );

        Ok(BatchResult {
            total: total_images,
            successful,
            failed,
            processing_time_ms: total_time.as_secs_f64() * 1000.0,
            average_time_per_page_ms: total_time.as_secs_f64() * 1000.0 / total_images.max(1) as f64,
            analytics: BatchAnalytics {
                total_images,
                total_regions,
                simple_bg_count,
                complex_bg_count,
                label_0_count: 0,
                label_1_count: 0,
                label_2_count: 0,
                validation_warnings: 0,
                api_calls_simple: 0,
                api_calls_complex: 0,
                api_calls_banana: 0,
                input_tokens: 0,
                output_tokens: 0,
                cache_hits: 0,
                cache_misses: 0,
                phase1_time_ms: 0.0,
                phase2_time_ms: 0.0,
                phase3_time_ms: 0.0,
                phase4_time_ms: 0.0,
                total_time_ms: total_time.as_secs_f64() * 1000.0,
                inference_time_ms: 0.0,
                api_wait_time_ms: 0.0,
            },
            results: all_results,
        })
    }

    /// Process a single image through the entire pipeline
    /// Detection → [OCR + Cleaning in parallel] → Phase3 → Rendering
    async fn process_single_image_pipeline(
        phase1: Arc<Phase1Pipeline>,
        phase2: Arc<Phase2Pipeline>,
        phase3: Arc<Phase3Pipeline>,
        phase4: Arc<Phase4Pipeline>,
        image_data: ImageData,
        config: &ProcessingConfig,
        app_config: &Config,
    ) -> Result<PageResult> {
        let page_start = Instant::now();
        let page_index = image_data.index;
        let filename = image_data.filename.clone();

        // Extract config values
        let include_free_text = config.include_free_text.unwrap_or(false);
        let use_mask = config.use_mask.unwrap_or(true);
        let filter_orphan_regions = config.filter_orphan_regions.unwrap_or(false);
        let blur_free_text = config.blur_free_text_bg.unwrap_or(app_config.blur_free_text());
        let banana_mode = config.banana_mode.unwrap_or(app_config.banana_mode_enabled());
        let cache_enabled = config.cache_enabled.unwrap_or(true);
        let text_stroke = config.text_stroke.unwrap_or(app_config.text_stroke_enabled());
        let font_family = config.font_family.as_deref().unwrap_or("arial");
        let ocr_model = config.ocr_translation_model.as_deref();
        let banana_model = config.banana_image_model.as_deref();
        let target_language = config.target_language.as_deref();

        // Decode image once
        let decoded_image = if let Some(ref img) = image_data.decoded_image {
            Arc::clone(img)
        } else {
            Arc::new(image::load_from_memory(&image_data.image_bytes)
                .context("Failed to decode image")?)
        };

        let mut optimized_image_data = image_data.clone();
        optimized_image_data.decoded_image = Some(Arc::clone(&decoded_image));

        // ===== STEP 1: DETECTION =====
        debug!("Page {}: Starting detection", page_index);
        let detection_start = Instant::now();

        let (mut phase1_outputs, decoded_images) = phase1
            .execute_detection_only(
                &[optimized_image_data.clone()],
                false, // merge_img
                config.target_size,
                filter_orphan_regions,
                use_mask,
                blur_free_text,
            )
            .await
            .context("Detection failed")?;

        let mut phase1_output = phase1_outputs.pop().unwrap();

        // Filter out label 2 if not included
        if !include_free_text {
            phase1_output.regions.retain(|r| r.label != 2);
        }

        debug!("Page {}: Detection complete in {:.0}ms, {} regions",
            page_index, detection_start.elapsed().as_secs_f64() * 1000.0, phase1_output.regions.len());

        // ===== STEP 2: OCR + CLEANING IN PARALLEL =====
        debug!("Page {}: Starting OCR + Cleaning in parallel", page_index);
        let parallel_start = Instant::now();

        // Clone data for parallel tasks
        let phase1_for_ocr = phase1_output.clone();
        let phase1_for_clean = phase1_output.clone();
        let image_for_ocr = optimized_image_data.clone();
        let decoded_for_clean = decoded_images.into_iter().next().unwrap_or_else(|| (*decoded_image).clone());

        // Run OCR and Cleaning in PARALLEL
        let (ocr_result, clean_result) = tokio::join!(
            // OCR Task
            async {
                phase2.execute(
                    &image_for_ocr,
                    &phase1_for_ocr,
                    ocr_model,
                    banana_model,
                    banana_mode,
                    cache_enabled,
                    target_language,
                ).await
            },
            // Cleaning Task
            async {
                let mut outputs = vec![phase1_for_clean];
                phase1.complete_cleaning(&mut outputs, &[decoded_for_clean], None).await?;
                Ok::<_, anyhow::Error>(outputs.pop().unwrap())
            }
        );

        let phase2_output = ocr_result.context("OCR failed")?;
        let cleaned_phase1 = clean_result.context("Cleaning failed")?;

        debug!("Page {}: OCR + Cleaning complete in {:.0}ms",
            page_index, parallel_start.elapsed().as_secs_f64() * 1000.0);

        // ===== STEP 3: PHASE 3 (pass-through) =====
        let banana_region_ids: Vec<usize> = phase2_output
            .complex_bg_bananas
            .iter()
            .map(|b| b.region_id)
            .collect();

        let phase3_output = phase3
            .execute(&optimized_image_data, &cleaned_phase1, &banana_region_ids, blur_free_text, use_mask)
            .await
            .context("Phase 3 failed")?;

        // ===== STEP 4: RENDERING =====
        debug!("Page {}: Starting rendering", page_index);
        let render_start = Instant::now();

        let phase4_output = phase4
            .execute(&optimized_image_data, &cleaned_phase1, &phase2_output, &phase3_output, font_family, text_stroke)
            .await
            .context("Rendering failed")?;

        debug!("Page {}: Rendering complete in {:.0}ms",
            page_index, render_start.elapsed().as_secs_f64() * 1000.0);

        // Convert to data URL
        let base64_image = general_purpose::STANDARD.encode(&phase4_output.final_image_bytes);
        let data_url = format!("data:image/png;base64,{}", base64_image);

        let total_time = page_start.elapsed();
        info!("Page {}: Pipeline complete in {:.0}ms", page_index, total_time.as_secs_f64() * 1000.0);

        Ok(PageResult {
            index: page_index,
            filename,
            success: true,
            processing_time_ms: total_time.as_secs_f64() * 1000.0,
            error: None,
            data_url: Some(data_url),
        })
    }
}

/// Process a single batch with PHASE-LEVEL SYNCHRONIZATION
/// All pages complete Phase 1 before any start Phase 2, etc.
async fn process_single_batch(
    phase1: Arc<Phase1Pipeline>,
    phase2: Arc<Phase2Pipeline>,
    phase3: Arc<Phase3Pipeline>,
    phase4: Arc<Phase4Pipeline>,
    batch: ImageBatch,
    config: &ProcessingConfig,
    app_config: &Config,
) -> Result<(Vec<PageResult>, PerformanceMetrics)> {
    let config = Arc::new(config.clone());
    let app_config = Arc::new(app_config.clone());
    let page_starts: Vec<_> = batch.images.iter().map(|_| Instant::now()).collect();

    // Batch-level metrics
    let mut batch_metrics = PerformanceMetrics::default();

    // Check config settings
    let merge_img = config.merge_img.unwrap_or(false);
    let include_free_text = config.include_free_text.unwrap_or(false);
    let use_mask = config.use_mask.unwrap_or(true);
    let filter_orphan_regions = config.filter_orphan_regions.unwrap_or(false);
    let blur_free_text = config.blur_free_text_bg.unwrap_or(app_config.blur_free_text());

    // ===== PHASE 1: Detection & Categorization =====
    let phase1_start = Instant::now();

    // Collect Phase 1 results - either batched or parallel individual
    let phase1_results: Vec<Result<Result<(ImageData, crate::core::types::Phase1Output), anyhow::Error>, tokio::task::JoinError>> = if merge_img {
        // BATCH MODE: Run detection for all images in single ONNX inference
        // Execute batch Phase 1
        let batch_outputs = phase1.execute_batch(&batch.images, use_mask, merge_img, config.target_size, filter_orphan_regions, blur_free_text).await;

        match batch_outputs {
            Ok(outputs) => {
                // Convert to expected format
                outputs.into_iter().enumerate().map(|(i, mut output)| {
                    // Filter out label 2 if not included
                    if !include_free_text {
                        output.regions.retain(|r| r.label != 2);
                    }
                    let image_data = batch.images[i].clone();
                    Ok(Ok((image_data, output)))
                }).collect()
            }
            Err(e) => {
                // All images failed - return error for each
                batch.images.iter().map(|_| {
                    Ok(Err(anyhow::anyhow!("Batch detection failed: {}", e)))
                }).collect()
            }
        }
    } else {
        // PARALLEL MODE: Run detection for each image independently
        let phase1_tasks: Vec<_> = batch.images.iter().map(|image_data| {
            let phase1 = Arc::clone(&phase1);
            let config = Arc::clone(&config);
            let app_config = Arc::clone(&app_config);
            let image_data = image_data.clone();

            tokio::spawn(async move {
                let include_free_text = config.include_free_text.unwrap_or(false);
                let use_mask = config.use_mask.unwrap_or(true);
                let blur_free_text = config.blur_free_text_bg.unwrap_or(app_config.blur_free_text());

                // Load decoded image
                let decoded_image = if let Some(ref img) = image_data.decoded_image {
                    Arc::clone(img)
                } else {
                    Arc::new(image::load_from_memory(&image_data.image_bytes)?)
                };

                let mut optimized_image_data = image_data.clone();
                optimized_image_data.decoded_image = Some(decoded_image);

                let mut phase1_output = phase1.execute(&optimized_image_data, use_mask, config.mask_mode.as_deref(), config.target_size, filter_orphan_regions, blur_free_text).await?;

                // Filter out label 2 if not included
                if !include_free_text {
                    phase1_output.regions.retain(|r| r.label != 2);
                }

                Ok::<_, anyhow::Error>((optimized_image_data, phase1_output))
            })
        }).collect();

        join_all(phase1_tasks).await
    };
    batch_metrics.phase1_time = phase1_start.elapsed();

    // Collect Phase 1 outputs and metrics
    let mut phase1_data: Vec<(ImageData, crate::core::types::Phase1Output)> = Vec::new();
    let mut failed_pages: Vec<(usize, String, String)> = Vec::new();

    for (i, result) in phase1_results.into_iter().enumerate() {
        match result {
            Ok(Ok(data)) => {
                // Collect metrics from phase1_output
                let phase1_output = &data.1;
                batch_metrics.total_regions += phase1_output.regions.len();
                batch_metrics.validation_warnings += phase1_output.validation_warnings.len();

                // Count labels and background types
                for region in &phase1_output.regions {
                    match region.label {
                        0 => batch_metrics.label_0_count += 1,
                        1 => batch_metrics.label_1_count += 1,
                        2 => batch_metrics.label_2_count += 1,
                        _ => {}
                    }
                    if region.background_type == crate::core::types::BackgroundType::Simple {
                        batch_metrics.simple_bg_count += 1;
                    } else {
                        batch_metrics.complex_bg_count += 1;
                    }
                }

                phase1_data.push(data);
            }
            Ok(Err(e)) => {
                let img = &batch.images[i];
                failed_pages.push((img.index, img.filename.clone(), e.to_string()));
            }
            Err(e) => {
                let img = &batch.images[i];
                failed_pages.push((img.index, img.filename.clone(), e.to_string()));
            }
        }
    }

    // ===== PHASE 2: BATCHED API calls for all pages =====
    let phase2_start = Instant::now();

    // Use batched Phase 2 to combine all regions into fewer API calls
    let ocr_model_override = config.ocr_translation_model.as_deref();
    let banana_model_override = config.banana_image_model.as_deref();
    let banana_mode = config.banana_mode.unwrap_or(app_config.banana_mode_enabled());
    let cache_enabled = config.cache_enabled.unwrap_or(true);
    let custom_api_keys = config.api_keys.as_deref();
    let target_language = config.target_language.as_deref();
    let reuse_factor = config.reuse_factor.unwrap_or(4).clamp(1, 8);

    let phase2_outputs = phase2
        .execute_batch(&phase1_data, ocr_model_override, banana_model_override, banana_mode, cache_enabled, custom_api_keys, target_language, reuse_factor)
        .await;

    batch_metrics.phase2_time = phase2_start.elapsed();

    // Collect Phase 2 outputs and metrics
    let mut phase2_data: Vec<(ImageData, crate::core::types::Phase1Output, crate::core::types::Phase2Output)> = Vec::new();

    match phase2_outputs {
        Ok(outputs) => {
            // Track if we made any API calls for simple regions (batched)
            let mut total_simple_translations = 0;

            for (i, phase2_output) in outputs.into_iter().enumerate() {
                let (image_data, phase1_output) = &phase1_data[i];

                // Count regions for metrics
                let simple_regions_count = phase1_output.regions.iter()
                    .filter(|r| r.background_type == crate::core::types::BackgroundType::Simple)
                    .count();

                // Track translations vs cache hits
                let translations_count = phase2_output.simple_bg_translations.len();
                total_simple_translations += translations_count;

                // Cache hits = total simple regions for this page - translations returned
                // Note: translations include both cache misses AND cached results now
                // The Phase2 batch returns all results, so we can't distinguish here
                // We'll count total at the end
                batch_metrics.cache_misses += translations_count;
                let cache_hits_for_page = simple_regions_count.saturating_sub(translations_count);
                batch_metrics.cache_hits += cache_hits_for_page;

                // Complex background handling
                batch_metrics.api_calls_banana += phase2_output.complex_bg_bananas.len();
                if !phase2_output.complex_bg_translations.is_empty() {
                    batch_metrics.api_calls_complex += 1;
                }

                phase2_data.push((image_data.clone(), phase1_output.clone(), phase2_output));
            }

            // Count as ONE logical batch (split across multiple keys for parallelism)
            // e.g. 35 regions with 6 keys = 6 parallel API calls instead of 1 sequential
            if total_simple_translations > 0 {
                batch_metrics.api_calls_simple = 1;
            }
        }
        Err(e) => {
            tracing::error!("Phase 2 batch error: {:?}", e);
            // Return empty results - all pages failed
            return Ok((
                batch.images.iter().map(|img| crate::core::types::PageResult {
                    index: img.index,
                    filename: img.filename.clone(),
                    success: false,
                    processing_time_ms: 0.0,
                    error: Some(format!("Phase 2 failed: {}", e)),
                    data_url: None,
                }).collect(),
                batch_metrics,
            ));
        }
    }

    // ===== PHASE 3: All pages in parallel =====
    let phase3_start = Instant::now();

    let phase3_data: Vec<(ImageData, crate::core::types::Phase1Output, crate::core::types::Phase2Output, crate::core::types::Phase3Output)> = {
        let phase3_tasks: Vec<_> = phase2_data.iter().map(|(image_data, phase1_output, phase2_output)| {
            let phase3 = Arc::clone(&phase3);
            let config = Arc::clone(&config);
            let app_config = Arc::clone(&app_config);
            let image_data = image_data.clone();
            let phase1_output = phase1_output.clone();
            let phase2_output = phase2_output.clone();

            tokio::spawn(async move {
                let blur_free_text = config.blur_free_text_bg.unwrap_or(app_config.blur_free_text());
                let use_mask = config.use_mask.unwrap_or(true);

                let banana_region_ids: Vec<usize> = phase2_output
                    .complex_bg_bananas
                    .iter()
                    .map(|b| b.region_id)
                    .collect();

                let phase3_output = phase3
                    .execute(&image_data, &phase1_output, &banana_region_ids, blur_free_text, use_mask)
                    .await?;

                Ok::<_, anyhow::Error>((image_data, phase1_output, phase2_output, phase3_output))
            })
        }).collect();

        let phase3_results = join_all(phase3_tasks).await;

        // Collect Phase 3 outputs
        let mut data = Vec::new();
        for result in phase3_results {
            match result {
                Ok(Ok(d)) => data.push(d),
                Ok(Err(e)) => tracing::error!("Phase 3 error: {:?}", e),
                Err(e) => tracing::error!("Phase 3 task error: {:?}", e),
            }
        }
        data
    };

    batch_metrics.phase3_time = phase3_start.elapsed();

    // ===== PHASE 4: All pages in parallel =====
    let phase4_start = Instant::now();
    let phase4_tasks: Vec<_> = phase3_data.into_iter().enumerate().map(|(i, (image_data, phase1_output, phase2_output, phase3_output))| {
        let phase4 = Arc::clone(&phase4);
        let config = Arc::clone(&config);
        let app_config = Arc::clone(&app_config);
        let page_start = page_starts[i];

        tokio::spawn(async move {
            let font_family = config.font_family.clone().unwrap_or_else(|| "arial".to_string());
            let text_stroke = config.text_stroke.unwrap_or(app_config.text_stroke_enabled());

            let phase4_output = phase4
                .execute(&image_data, &phase1_output, &phase2_output, &phase3_output, &font_family, text_stroke)
                .await?;

            let processing_time_ms = page_start.elapsed().as_secs_f64() * 1000.0;

            // Convert to base64 data URL
            let base64_image = general_purpose::STANDARD.encode(&phase4_output.final_image_bytes);
            let data_url = format!("data:image/png;base64,{}", base64_image);

            Ok::<_, anyhow::Error>(PageResult {
                index: image_data.index,
                filename: image_data.filename,
                success: true,
                processing_time_ms,
                error: None,
                data_url: Some(data_url),
            })
        })
    }).collect();

    let phase4_results = join_all(phase4_tasks).await;
    batch_metrics.phase4_time = phase4_start.elapsed();

    // Collect final results
    let mut results = Vec::new();

    for result in phase4_results {
        match result {
            Ok(Ok(page_result)) => {
                results.push(page_result);
            }
            Ok(Err(e)) => tracing::error!("Phase 4 error: {:?}", e),
            Err(e) => tracing::error!("Phase 4 task error: {:?}", e),
        }
    }

    // Add failed pages
    for (index, filename, error) in failed_pages {
        results.push(PageResult {
            index,
            filename,
            success: false,
            processing_time_ms: 0.0,
            error: Some(error),
            data_url: None,
        });
    }

    // Sort results by index to maintain order
    results.sort_by_key(|r| r.index);

    Ok((results, batch_metrics))
}

/// Process a single page through all phases
async fn process_single_page(
    phase1: &Phase1Pipeline,
    phase2: &Phase2Pipeline,
    phase3: &Phase3Pipeline,
    phase4: &Phase4Pipeline,
    font_manager: &FontManager,
    image_data: &ImageData,
    config: &ProcessingConfig,
    app_config: &Config,
) -> Result<(PerformanceMetrics, Vec<u8>)> {
    let mut metrics = PerformanceMetrics::default();

    // Extract config with defaults from app_config
    let font_source = config.font_source.clone().unwrap_or_else(|| "builtin".to_string());
    let font_family = config.font_family.clone().unwrap_or_else(|| "arial".to_string());
    let google_font_family = config.google_font_family.clone();
    let ocr_model_override = config.ocr_translation_model.as_deref();
    let banana_model_override = config.banana_image_model.as_deref();
    let include_free_text = config.include_free_text.unwrap_or(false);
    let banana_mode = config.banana_mode.unwrap_or(app_config.banana_mode_enabled());
    let text_stroke = config.text_stroke.unwrap_or(app_config.text_stroke_enabled());
    let blur_free_text = config.blur_free_text_bg.unwrap_or(app_config.blur_free_text());
    let cache_enabled = config.cache_enabled.unwrap_or(true);
    let use_mask = config.use_mask.unwrap_or(true);
    let filter_orphan_regions = config.filter_orphan_regions.unwrap_or(false);

    // OPTIMIZATION: Load image once for all phases to avoid redundant decoding
    // Image decoding is expensive (5-50ms), and we were doing it 4 times per image.
    // This single decode + Arc sharing saves ~15-20% total processing time.
    let decoded_image = if let Some(ref img) = image_data.decoded_image {
        Arc::clone(img)
    } else {
        Arc::new(
            image::load_from_memory(&image_data.image_bytes)
                .context("Failed to decode image")?,
        )
    };

    // Create optimized image data with pre-decoded image
    let mut optimized_image_data = image_data.clone();
    optimized_image_data.decoded_image = Some(decoded_image);

    // Phase 1: Detection & Categorization
    let p1_start = Instant::now();
    let mut phase1_output = phase1
        .execute(&optimized_image_data, use_mask, config.mask_mode.as_deref(), config.target_size, filter_orphan_regions, blur_free_text)
        .await
        .context("Phase 1 failed")?;

    // Filter out label 2 (free text) if not included
    if !include_free_text {
        phase1_output.regions.retain(|r| r.label != 2);
    }

    metrics.phase1_time = p1_start.elapsed();
    metrics.total_regions = phase1_output.regions.len();
    metrics.simple_bg_count = phase1_output
        .regions
        .iter()
        .filter(|r| r.background_type == crate::core::types::BackgroundType::Simple)
        .count();
    metrics.complex_bg_count = phase1_output.regions.len() - metrics.simple_bg_count;
    metrics.validation_warnings = phase1_output.validation_warnings.len();

    // Phase 2: API Calls
    let p2_start = Instant::now();
    let target_language = config.target_language.as_deref();
    let phase2_output = phase2
        .execute(
            &optimized_image_data,
            &phase1_output,
            ocr_model_override,
            banana_model_override,
            banana_mode,
            cache_enabled,
            target_language,
        )
        .await
        .context("Phase 2 failed")?;
    metrics.phase2_time = p2_start.elapsed();
    metrics.api_calls_simple = phase2_output.simple_bg_translations.len() / 5; // Approximation (batch size M)
    metrics.api_calls_banana = phase2_output.complex_bg_bananas.len();
    metrics.api_calls_complex = phase2_output.complex_bg_translations.len() / 5; // Approximation

    // Collect banana-processed region IDs
    let banana_region_ids: Vec<usize> = phase2_output
        .complex_bg_bananas
        .iter()
        .map(|b| b.region_id)
        .collect();

    // Phase 3: Text Removal
    let p3_start = Instant::now();
    let phase3_output = phase3
        .execute(&optimized_image_data, &phase1_output, &banana_region_ids, blur_free_text, use_mask)
        .await
        .context("Phase 3 failed")?;
    metrics.phase3_time = p3_start.elapsed();

    // Handle Google Fonts if needed
    let final_font_family = if font_source == "google" {
        if let Some(ref google_font) = google_font_family {
            // Download and load Google Font
            match font_manager.get_google_font(google_font).await {
                Ok(font_data) => {
                    // Load font into Phase4's renderer
                    if let Err(e) = phase4.load_google_font(font_data, google_font).await {
                        tracing::warn!("Failed to load Google Font '{}': {}. Falling back to built-in font.", google_font, e);
                        font_family
                    } else {
                        google_font.clone()
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to download Google Font '{}': {}. Falling back to built-in font.", google_font, e);
                    font_family
                }
            }
        } else {
            tracing::warn!("Google Fonts selected but no font family specified. Using default built-in font.");
            font_family
        }
    } else {
        font_family
    };

    // Phase 4: Text Insertion
    let p4_start = Instant::now();
    let phase4_output = phase4
        .execute(
            &optimized_image_data,
            &phase1_output,
            &phase2_output,
            &phase3_output,
            &final_font_family,
            text_stroke,
        )
        .await
        .context("Phase 4 failed")?;
    metrics.phase4_time = p4_start.elapsed();

    Ok((metrics, phase4_output.final_image_bytes))
}
