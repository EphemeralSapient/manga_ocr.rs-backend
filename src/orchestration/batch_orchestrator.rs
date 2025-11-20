// Batch Orchestrator: Main workflow coordinator

use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine};
use futures::future::join_all;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::{info, instrument};

use crate::services::translation::api_client::ApiClient;
use crate::core::config::Config;
use crate::services::detection::DetectionService;
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

/// Main batch orchestrator
pub struct BatchOrchestrator {
    config: Arc<Config>,
    phase1: Arc<Phase1Pipeline>,
    phase2: Arc<Phase2Pipeline>,
    phase3: Arc<Phase3Pipeline>,
    phase4: Arc<Phase4Pipeline>,
    batch_semaphore: Arc<Semaphore>,
}

impl BatchOrchestrator {
    /// Create new batch orchestrator
    #[instrument(skip(config))]
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        info!("Initializing services...");

        // Initialize services
        let detector = Arc::new(DetectionService::new(config.clone()).await?);
        let segmenter = Arc::new(SegmentationService::new(config.clone()).await?);
        let api_client = Arc::new(ApiClient::new(config.clone(), None, None)?);
        let cache = Arc::new(TranslationCache::new(config.cache_dir(), None, None, None).await?);

        // Log cache stats
        let (cache_entries, cache_size_mb) = cache.stats().await;
        debug!("Cache: {} entries ({:.2} MB)", cache_entries, cache_size_mb);

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

        info!("✓ Ready (batches: {}, ONNX pool: {} sessions, ~{} MB)",
            config.max_concurrent_batches(),
            config.onnx_pool_size(),
            config.onnx_pool_size() * (42 + 40));

        Ok(Self {
            config,
            phase1,
            phase2,
            phase3,
            phase4,
            batch_semaphore,
        })
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
    #[instrument(skip(self, images), fields(total_images = images.len()))]
    pub async fn process_batch(
        &self,
        images: Vec<ImageData>,
        config: &ProcessingConfig,
    ) -> Result<BatchResult> {
        let start_time = Instant::now();
        let total_images = images.len();

        info!("Processing {} images", total_images);

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

        info!("Processing {} batches phase-by-phase: Phase 1 → Phase 2 (global) → Phase 3 → Phase 4", batches.len());

        // ===== PHASE 1: ALL BATCHES =====
        info!("Starting Phase 1 for all {} batches", batches.len());

        let phase1_start = Instant::now();
        let mut all_phase1_data: Vec<(ImageData, crate::core::types::Phase1Output)> = Vec::new();
        let mut phase1_metrics = PerformanceMetrics::default();

        // Check if using DirectML - if so, use sequential processing
        let is_directml = self.phase1.is_directml();

        if is_directml {
            // DirectML: Sequential processing through single session (no parallelism)
            info!("🔄 DirectML detected: Processing {} batches sequentially (single session)", batches.len());

            for (batch_idx, batch) in batches.into_iter().enumerate() {
                match self.phase1.execute_batch(&batch.images, use_mask, merge_img).await {
                    Ok(outputs) => {
                        for (i, mut output) in outputs.into_iter().enumerate() {
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

                            all_phase1_data.push((batch.images[i].clone(), output));
                        }
                    }
                    Err(e) => {
                        tracing::error!("Phase 1 batch {} failed: {:?}", batch_idx, e);
                    }
                }
            }
        } else {
            // Non-DirectML: Parallel processing (original behavior)
            let num_batches = batches.len();
            if num_batches > 1 {
                info!("⚡ Non-DirectML backend: Processing {} batches in parallel", num_batches);
            }

            let mut phase1_tasks = Vec::new();
            for batch in batches {
                let phase1 = Arc::clone(&self.phase1);
                let images = batch.images.clone();

                let task = tokio::spawn(async move {
                    let batch_outputs = phase1.execute_batch(&images, use_mask, merge_img).await?;
                    Ok::<_, anyhow::Error>((images, batch_outputs))
                });

                phase1_tasks.push(task);
            }

            // Wait for all Phase 1 to complete
            for task in phase1_tasks {
                match task.await {
                    Ok(Ok((images, outputs))) => {
                        for (i, mut output) in outputs.into_iter().enumerate() {
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

                            all_phase1_data.push((images[i].clone(), output));
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Phase 1 batch failed: {:?}", e);
                    }
                    Err(e) => {
                        tracing::error!("Phase 1 task panicked: {:?}", e);
                    }
                }
            }
        }

        phase1_metrics.phase1_time = phase1_start.elapsed();
        info!(
            "Phase 1 complete for all {} pages in {:.2}ms",
            all_phase1_data.len(),
            phase1_metrics.phase1_time.as_secs_f64() * 1000.0
        );

        // CLEANUP: Free ONNX sessions after Phase 1 to minimize memory footprint
        // Detection and segmentation are only needed in Phase 1
        // Phases 2-4 use API calls and image processing (no ONNX)
        info!("🧹 [MEMORY OPTIMIZATION] Cleaning up ONNX sessions after Phase 1");
        self.phase1.cleanup_sessions();
        info!("✓ ONNX memory freed, proceeding with translation & rendering");

        // ===== PHASE 2: GLOBAL (ALL PAGES, SPLIT ACROSS KEYS) =====
        info!("Starting Phase 2 GLOBAL for all {} pages", all_phase1_data.len());
        let phase2_start = Instant::now();

        let ocr_model_override = config.ocr_translation_model.as_deref();
        let banana_model_override = config.banana_image_model.as_deref();
        let banana_mode = config.banana_mode.unwrap_or(app_config.banana_mode_enabled());
        let cache_enabled = config.cache_enabled.unwrap_or(true);

        let phase2_outputs = self.phase2
            .execute_batch(&all_phase1_data, ocr_model_override, banana_model_override, banana_mode, cache_enabled)
            .await;

        phase1_metrics.phase2_time = phase2_start.elapsed();
        info!(
            "Phase 2 complete for all {} pages in {:.2}ms",
            all_phase1_data.len(),
            phase1_metrics.phase2_time.as_secs_f64() * 1000.0
        );

        let mut all_phase2_data: Vec<(ImageData, crate::core::types::Phase1Output, crate::core::types::Phase2Output)> = Vec::new();

        match phase2_outputs {
            Ok(outputs) => {
                // Collect metrics
                let mut total_simple_translations = 0;
                for (i, phase2_output) in outputs.into_iter().enumerate() {
                    let (image_data, phase1_output) = &all_phase1_data[i];

                    total_simple_translations += phase2_output.simple_bg_translations.len();
                    phase1_metrics.api_calls_banana += phase2_output.complex_bg_bananas.len();
                    if !phase2_output.complex_bg_translations.is_empty() {
                        phase1_metrics.api_calls_complex += 1;
                    }

                    all_phase2_data.push((image_data.clone(), phase1_output.clone(), phase2_output));
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
        info!("Starting Phase 3 for all {} pages", all_phase2_data.len());
        let phase3_start = Instant::now();

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

        let mut all_phase3_data = Vec::new();
        for task in phase3_tasks {
            match task.await {
                Ok(Ok(data)) => all_phase3_data.push(data),
                Ok(Err(e)) => tracing::error!("Phase 3 error: {:?}", e),
                Err(e) => tracing::error!("Phase 3 task error: {:?}", e),
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
    debug!("Processing batch {} with {} images (phase-synchronized)", batch.batch_id, batch.images.len());

    let config = Arc::new(config.clone());
    let app_config = Arc::new(app_config.clone());
    let page_starts: Vec<_> = batch.images.iter().map(|_| Instant::now()).collect();

    // Batch-level metrics
    let mut batch_metrics = PerformanceMetrics::default();

    // Check config settings
    let merge_img = config.merge_img.unwrap_or(false);
    let include_free_text = config.include_free_text.unwrap_or(false);
    let use_mask = config.use_mask.unwrap_or(true);

    // ===== PHASE 1: Detection & Categorization =====
    let phase1_start = Instant::now();

    // Collect Phase 1 results - either batched or parallel individual
    let phase1_results: Vec<Result<Result<(ImageData, crate::core::types::Phase1Output), anyhow::Error>, tokio::task::JoinError>> = if merge_img {
        // BATCH MODE: Run detection for all images in single ONNX inference
        debug!("Batch {}: Starting Phase 1 BATCH MODE for {} pages", batch.batch_id, batch.images.len());

        // Execute batch Phase 1
        let batch_outputs = phase1.execute_batch(&batch.images, use_mask, merge_img).await;

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
        debug!("Batch {}: Starting Phase 1 for all {} pages", batch.batch_id, batch.images.len());
        let phase1_tasks: Vec<_> = batch.images.iter().map(|image_data| {
            let phase1 = Arc::clone(&phase1);
            let config = Arc::clone(&config);
            let image_data = image_data.clone();

            tokio::spawn(async move {
                let include_free_text = config.include_free_text.unwrap_or(false);
                let use_mask = config.use_mask.unwrap_or(true);

                // Load decoded image
                let decoded_image = if let Some(ref img) = image_data.decoded_image {
                    Arc::clone(img)
                } else {
                    Arc::new(image::load_from_memory(&image_data.image_bytes)?)
                };

                let mut optimized_image_data = image_data.clone();
                optimized_image_data.decoded_image = Some(decoded_image);

                let mut phase1_output = phase1.execute(&optimized_image_data, use_mask).await?;

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
    debug!("Batch {}: Phase 1 complete for all pages in {:.2}ms", batch.batch_id, batch_metrics.phase1_time.as_secs_f64() * 1000.0);

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
    debug!("Batch {}: Starting Phase 2 BATCHED for {} pages", batch.batch_id, phase1_data.len());

    // Use batched Phase 2 to combine all regions into fewer API calls
    let ocr_model_override = config.ocr_translation_model.as_deref();
    let banana_model_override = config.banana_image_model.as_deref();
    let banana_mode = config.banana_mode.unwrap_or(app_config.banana_mode_enabled());
    let cache_enabled = config.cache_enabled.unwrap_or(true);

    let phase2_outputs = phase2
        .execute_batch(&phase1_data, ocr_model_override, banana_model_override, banana_mode, cache_enabled)
        .await;

    batch_metrics.phase2_time = phase2_start.elapsed();
    debug!("Batch {}: Phase 2 complete for all pages in {:.2}ms", batch.batch_id, batch_metrics.phase2_time.as_secs_f64() * 1000.0);

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
    debug!("Batch {}: Starting Phase 3 for {} pages", batch.batch_id, phase2_data.len());
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
    batch_metrics.phase3_time = phase3_start.elapsed();
    debug!("Batch {}: Phase 3 complete for all pages in {:.2}ms", batch.batch_id, batch_metrics.phase3_time.as_secs_f64() * 1000.0);

    // Collect Phase 3 outputs
    let mut phase3_data: Vec<(ImageData, crate::core::types::Phase1Output, crate::core::types::Phase2Output, crate::core::types::Phase3Output)> = Vec::new();

    for result in phase3_results {
        match result {
            Ok(Ok(data)) => phase3_data.push(data),
            Ok(Err(e)) => tracing::error!("Phase 3 error: {:?}", e),
            Err(e) => tracing::error!("Phase 3 task error: {:?}", e),
        }
    }

    // ===== PHASE 4: All pages in parallel =====
    let phase4_start = Instant::now();
    debug!("Batch {}: Starting Phase 4 for {} pages", batch.batch_id, phase3_data.len());
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
    debug!("Batch {}: Phase 4 complete for all pages in {:.2}ms", batch.batch_id, batch_metrics.phase4_time.as_secs_f64() * 1000.0);

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
    image_data: &ImageData,
    config: &ProcessingConfig,
    app_config: &Config,
) -> Result<(PerformanceMetrics, Vec<u8>)> {
    let mut metrics = PerformanceMetrics::default();

    // Extract config with defaults from app_config
    let font_family = config.font_family.clone().unwrap_or_else(|| "arial".to_string());
    let ocr_model_override = config.ocr_translation_model.as_deref();
    let banana_model_override = config.banana_image_model.as_deref();
    let include_free_text = config.include_free_text.unwrap_or(false);
    let banana_mode = config.banana_mode.unwrap_or(app_config.banana_mode_enabled());
    let text_stroke = config.text_stroke.unwrap_or(app_config.text_stroke_enabled());
    let blur_free_text = config.blur_free_text_bg.unwrap_or(app_config.blur_free_text());
    let cache_enabled = config.cache_enabled.unwrap_or(true);
    let use_mask = config.use_mask.unwrap_or(true);

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
        .execute(&optimized_image_data, use_mask)
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
    let phase2_output = phase2
        .execute(
            &optimized_image_data,
            &phase1_output,
            ocr_model_override,
            banana_model_override,
            banana_mode,
            cache_enabled,
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

    // Phase 4: Text Insertion
    let p4_start = Instant::now();
    let phase4_output = phase4
        .execute(
            &optimized_image_data,
            &phase1_output,
            &phase2_output,
            &phase3_output,
            &font_family,
            text_stroke,
        )
        .await
        .context("Phase 4 failed")?;
    metrics.phase4_time = p4_start.elapsed();

    Ok((metrics, phase4_output.final_image_bytes))
}
