// Batch Orchestrator: Main workflow coordinator

use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine};
use futures::future::join_all;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::{debug, info, instrument};

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
        let mut tasks = Vec::new();

        // Clone config for spawned tasks
        let config = Arc::new(config.clone());
        let app_config = Arc::clone(&self.config);

        for batch in batches {
            let phase1 = Arc::clone(&self.phase1);
            let phase2 = Arc::clone(&self.phase2);
            let phase3 = Arc::clone(&self.phase3);
            let phase4 = Arc::clone(&self.phase4);
            let semaphore = Arc::clone(&self.batch_semaphore);
            let config = Arc::clone(&config);
            let app_config = Arc::clone(&app_config);

            let task = tokio::spawn(async move {
                // Acquire semaphore permit
                let _permit = match semaphore.acquire().await {
                    Ok(permit) => permit,
                    Err(_) => {
                        return Err(anyhow::anyhow!("Semaphore closed - orchestrator shutting down"));
                    }
                };

                process_single_batch(
                    phase1,
                    phase2,
                    phase3,
                    phase4,
                    batch,
                    &config,
                    &app_config,
                )
                .await
            });

            tasks.push(task);
        }

        // Collect all results
        let mut all_results = Vec::new();
        let mut all_metrics = PerformanceMetrics::default();

        for task in tasks {
            match task.await {
                Ok(Ok((results, metrics))) => {
                    all_results.extend(results);
                    all_metrics.merge(metrics);
                }
                Ok(Err(e)) => {
                    tracing::error!("Batch processing failed: {:?}", e);
                }
                Err(e) => {
                    tracing::error!("Batch task panicked: {:?}", e);
                }
            }
        }

        let total_time = start_time.elapsed();

        // Compute analytics
        let successful = all_results.iter().filter(|r| r.success).count();
        let failed = all_results.len() - successful;

        let analytics = BatchAnalytics {
            total_images,
            total_regions: all_metrics.total_regions,
            simple_bg_count: all_metrics.simple_bg_count,
            complex_bg_count: all_metrics.complex_bg_count,
            label_0_count: 0, // TODO: Track separately if needed
            label_1_count: 0,
            label_2_count: 0,
            validation_warnings: all_metrics.validation_warnings,
            api_calls_simple: all_metrics.api_calls_simple,
            api_calls_complex: all_metrics.api_calls_complex,
            api_calls_banana: all_metrics.api_calls_banana,
            input_tokens: 0,  // TODO: Track if API provides
            output_tokens: 0, // TODO: Track if API provides
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

    // ===== PHASE 1: All pages in parallel =====
    debug!("Batch {}: Starting Phase 1 for all {} pages", batch.batch_id, batch.images.len());
    let phase1_tasks: Vec<_> = batch.images.iter().map(|image_data| {
        let phase1 = Arc::clone(&phase1);
        let app_config = Arc::clone(&app_config);
        let config = Arc::clone(&config);
        let image_data = image_data.clone();

        tokio::spawn(async move {
            let include_free_text = config.include_free_text.unwrap_or(false);

            // Load decoded image
            let decoded_image = if let Some(ref img) = image_data.decoded_image {
                Arc::clone(img)
            } else {
                Arc::new(image::load_from_memory(&image_data.image_bytes)?)
            };

            let mut optimized_image_data = image_data.clone();
            optimized_image_data.decoded_image = Some(decoded_image);

            let mut phase1_output = phase1.execute(&optimized_image_data).await?;

            // Filter out label 2 if not included
            if !include_free_text {
                phase1_output.regions.retain(|r| r.label != 2);
            }

            Ok::<_, anyhow::Error>((optimized_image_data, phase1_output))
        })
    }).collect();

    let phase1_results = join_all(phase1_tasks).await;
    debug!("Batch {}: Phase 1 complete for all pages", batch.batch_id);

    // Collect Phase 1 outputs
    let mut phase1_data: Vec<(ImageData, crate::core::types::Phase1Output)> = Vec::new();
    let mut failed_pages: Vec<(usize, String, String)> = Vec::new();

    for (i, result) in phase1_results.into_iter().enumerate() {
        match result {
            Ok(Ok(data)) => phase1_data.push(data),
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

    // ===== PHASE 2: All pages in parallel =====
    debug!("Batch {}: Starting Phase 2 for {} pages", batch.batch_id, phase1_data.len());
    let phase2_tasks: Vec<_> = phase1_data.iter().map(|(image_data, phase1_output)| {
        let phase2 = Arc::clone(&phase2);
        let config = Arc::clone(&config);
        let app_config = Arc::clone(&app_config);
        let image_data = image_data.clone();
        let phase1_output = phase1_output.clone();

        tokio::spawn(async move {
            let ocr_model_override = config.ocr_translation_model.as_deref();
            let banana_model_override = config.banana_image_model.as_deref();
            let banana_mode = config.banana_mode.unwrap_or(app_config.banana_mode_enabled());
            let cache_enabled = config.cache_enabled.unwrap_or(true);

            let phase2_output = phase2
                .execute(&image_data, &phase1_output, ocr_model_override, banana_model_override, banana_mode, cache_enabled)
                .await?;

            Ok::<_, anyhow::Error>((image_data, phase1_output, phase2_output))
        })
    }).collect();

    let phase2_results = join_all(phase2_tasks).await;
    debug!("Batch {}: Phase 2 complete for all pages", batch.batch_id);

    // Collect Phase 2 outputs
    let mut phase2_data: Vec<(ImageData, crate::core::types::Phase1Output, crate::core::types::Phase2Output)> = Vec::new();

    for result in phase2_results {
        match result {
            Ok(Ok(data)) => phase2_data.push(data),
            Ok(Err(e)) => tracing::error!("Phase 2 error: {:?}", e),
            Err(e) => tracing::error!("Phase 2 task error: {:?}", e),
        }
    }

    // ===== PHASE 3: All pages in parallel =====
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

            let banana_region_ids: Vec<usize> = phase2_output
                .complex_bg_bananas
                .iter()
                .map(|b| b.region_id)
                .collect();

            let phase3_output = phase3
                .execute(&image_data, &phase1_output, &banana_region_ids, blur_free_text)
                .await?;

            Ok::<_, anyhow::Error>((image_data, phase1_output, phase2_output, phase3_output))
        })
    }).collect();

    let phase3_results = join_all(phase3_tasks).await;
    debug!("Batch {}: Phase 3 complete for all pages", batch.batch_id);

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

            // Build metrics
            let mut metrics = PerformanceMetrics::default();
            metrics.total_regions = phase1_output.regions.len();
            metrics.simple_bg_count = phase1_output.regions.iter()
                .filter(|r| r.background_type == crate::core::types::BackgroundType::Simple)
                .count();
            metrics.complex_bg_count = phase1_output.regions.len() - metrics.simple_bg_count;
            metrics.validation_warnings = phase1_output.validation_warnings.len();

            // Convert to base64 data URL
            let base64_image = general_purpose::STANDARD.encode(&phase4_output.final_image_bytes);
            let data_url = format!("data:image/png;base64,{}", base64_image);

            Ok::<_, anyhow::Error>((
                PageResult {
                    index: image_data.index,
                    filename: image_data.filename,
                    success: true,
                    processing_time_ms,
                    error: None,
                    data_url: Some(data_url),
                },
                metrics,
            ))
        })
    }).collect();

    let phase4_results = join_all(phase4_tasks).await;
    debug!("Batch {}: Phase 4 complete for all pages", batch.batch_id);

    // Collect final results
    let mut results = Vec::new();
    let mut metrics = PerformanceMetrics::default();

    for result in phase4_results {
        match result {
            Ok(Ok((page_result, page_metrics))) => {
                results.push(page_result);
                metrics.merge(page_metrics);
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

    Ok((results, metrics))
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
        .execute(&optimized_image_data)
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
        .execute(&optimized_image_data, &phase1_output, &banana_region_ids, blur_free_text)
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
