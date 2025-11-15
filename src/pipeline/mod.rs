// Pipeline module - Core business logic for manga translation
//
// This module contains the orchestration logic for processing manga pages.
// It's separated from HTTP handlers to enable:
// - Testability (can test with mocked services)
// - Reusability (can use from CLI, gRPC, Lambda, etc.)
// - Clear separation of concerns

use crate::cache::Cache;
use crate::rendering::RenderingService;
use crate::services::traits::{BubbleDetector, BubbleRenderer, CacheStore, Translator};
use crate::types::{BatchConfig, ImageData, PageResult, PerformanceMetrics};
use anyhow::Result;
use base64::{engine::general_purpose, Engine};
use image::RgbaImage;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;
// Note: Mutex no longer needed - Cache has interior mutability
use tracing::{error, info, debug, instrument, Span};

/// Processing pipeline that orchestrates manga translation workflow
pub struct ProcessingPipeline {
    pub detector: Arc<dyn BubbleDetector>,
    pub translator: Arc<dyn Translator>,
    pub renderer: Arc<dyn BubbleRenderer>,
    pub cache: Arc<dyn CacheStore>,
}

impl ProcessingPipeline {
    /// Create a new processing pipeline with the given services
    pub fn new(
        detector: Arc<dyn BubbleDetector>,
        translator: Arc<dyn Translator>,
        renderer: Arc<dyn BubbleRenderer>,
        cache: Arc<dyn CacheStore>,
    ) -> Self {
        Self {
            detector,
            translator,
            renderer,
            cache,
        }
    }

    /// Process a batch of images in parallel
    ///
    /// # Arguments
    /// * `images` - Vector of images to process
    /// * `batch_config` - Configuration for this batch (optional model overrides)
    ///
    /// # Returns
    /// Tuple of (results, aggregated_metrics)
    #[instrument(skip(self, images, batch_config), fields(image_count = images.len()))]
    pub async fn process_batch(
        &self,
        images: Vec<ImageData>,
        batch_config: &BatchConfig,
    ) -> Result<(Vec<PageResult>, PerformanceMetrics)> {
        let mut results = Vec::new();
        let mut metrics = PerformanceMetrics::default();

        // Process all images in parallel
        let mut tasks = Vec::new();

        for img_data in images {
            let detector = Arc::clone(&self.detector);
            let translator = Arc::clone(&self.translator);
            let renderer = Arc::clone(&self.renderer);
            let cache = Arc::clone(&self.cache);
            let config = batch_config.clone();

            let task = tokio::spawn(async move {
                process_single_page(detector, translator, renderer, cache, img_data, config).await
            });
            tasks.push(task);
        }

        // Collect results
        for task in tasks {
            match task.await {
                Ok(Ok((result, page_metrics))) => {
                    results.push(result);
                    metrics.merge(&page_metrics);
                }
                Ok(Err(e)) => {
                    error!("Page processing failed: {}", e);
                }
                Err(e) => {
                    error!("Task join failed: {}", e);
                }
            }
        }

        // Save cache (cache has interior mutability, no lock needed)
        self.cache.save()?;

        Ok((results, metrics))
    }
}

/// Process a single page (image) with all its bubbles
#[instrument(skip(detector, translator, renderer, cache, img_data, batch_config), fields(page_index = img_data.index, filename = %img_data.filename))]
async fn process_single_page(
    detector: Arc<dyn BubbleDetector>,
    translator: Arc<dyn Translator>,
    renderer: Arc<dyn BubbleRenderer>,
    cache: Arc<dyn CacheStore>,
    img_data: ImageData,
    batch_config: BatchConfig,
) -> Result<(PageResult, PerformanceMetrics)> {
    let page_start = Instant::now();
    let mut metrics = PerformanceMetrics::default();

    info!("[PAGE {}] Processing {}...", img_data.index, img_data.filename);

    // Load image
    let img = image::load_from_memory(&img_data.image_bytes)?;
    let mut result_img = img.to_rgba8();

    // Detect bubbles
    let detection_start = Instant::now();
    let detections = detector.detect_bubbles(&img, img_data.index).await?;
    metrics.detection_time = detection_start.elapsed();

    Span::current().record("bubbles_detected", detections.len());
    info!("[PAGE {}] Detected {} bubbles", img_data.index, detections.len());

    if detections.is_empty() {
        // No bubbles, return original
        let output = encode_image_to_data_url(&img.to_rgba8())?;
        return Ok((
            PageResult {
                index: img_data.index,
                filename: img_data.filename,
                success: true,
                bubbles_detected: 0,
                bubbles_translated: 0,
                cache_hit: false,
                processing_time_ms: page_start.elapsed().as_secs_f64() * 1000.0,
                error: None,
                data_url: Some(output),
            },
            metrics,
        ));
    }

    // Process bubbles in parallel
    let mut bubble_tasks = Vec::new();

    for detection in detections.iter() {
        let translator = Arc::clone(&translator);
        let renderer = Arc::clone(&renderer);
        let cache = Arc::clone(&cache);
        let detection = detection.clone();
        let img_data = img_data.clone();
        let batch_config = batch_config.clone();

        let task = tokio::spawn(async move {
            process_single_bubble(translator, renderer, cache, img_data, detection, batch_config)
                .await
        });

        bubble_tasks.push(task);
    }

    // Collect bubble results
    let translation_start = Instant::now();
    let mut translated_count = 0;

    for (idx, task) in bubble_tasks.into_iter().enumerate() {
        match task.await {
            Ok(Ok((bubble_img, cache_hit, api_saved, is_simple_bg))) => {
                // Composite bubble onto result image
                renderer.composite_bubble_onto_page(
                    &mut result_img,
                    &bubble_img,
                    &detections[idx],
                );
                translated_count += 1;

                if cache_hit {
                    metrics.cache_hits += 1;
                } else {
                    metrics.cache_misses += 1;
                }

                if api_saved {
                    metrics.api_calls_saved += 1;
                }

                if is_simple_bg {
                    metrics.simple_backgrounds += 1;
                } else {
                    metrics.background_redraws += 1;
                }
            }
            Ok(Err(e)) => {
                error!("[PAGE {}] Bubble {} failed: {}", img_data.index, idx + 1, e);
            }
            Err(e) => {
                error!("[PAGE {}] Bubble task {} panicked: {}", img_data.index, idx + 1, e);
            }
        }
    }

    metrics.translation_time = translation_start.elapsed();

    // Encode result
    let output = encode_image_to_data_url(&result_img)?;

    let processing_time_ms = page_start.elapsed().as_secs_f64() * 1000.0;

    info!(
        "[PAGE {}] ✓ Complete in {:.2}ms ({} bubbles translated)",
        img_data.index, processing_time_ms, translated_count
    );

    Ok((
        PageResult {
            index: img_data.index,
            filename: img_data.filename,
            success: true,
            bubbles_detected: detections.len(),
            bubbles_translated: translated_count,
            cache_hit: metrics.cache_hits > 0,
            processing_time_ms,
            error: None,
            data_url: Some(output),
        },
        metrics,
    ))
}

/// Process a single bubble within a page
#[instrument(skip(translator, renderer, cache, img_data, detection, batch_config), fields(page_index = img_data.index, bubble_index = detection.bubble_index))]
async fn process_single_bubble(
    translator: Arc<dyn Translator>,
    renderer: Arc<dyn BubbleRenderer>,
    cache: Arc<dyn CacheStore>,
    img_data: ImageData,
    detection: crate::types::BubbleDetection,
    batch_config: BatchConfig,
) -> Result<(image::DynamicImage, bool, bool, bool)> {
    // Extract bubble region
    let img = image::load_from_memory(&img_data.image_bytes)?;
    let x1 = detection.bbox[0].max(0) as u32;
    let y1 = detection.bbox[1].max(0) as u32;
    let x2 = detection.bbox[2].min(img.width() as i32) as u32;
    let y2 = detection.bbox[3].min(img.height() as i32) as u32;

    let bubble_img = img.crop_imm(x1, y1, x2 - x1, y2 - y1);

    // Convert to bytes for cache lookup
    let mut bubble_bytes = Vec::new();
    bubble_img
        .write_to(&mut Cursor::new(&mut bubble_bytes), image::ImageFormat::Png)?;

    // Check cache
    let checksum = Cache::compute_checksum(&bubble_bytes);
    let cached_translation = cache.get(&checksum);

    let (translation, cache_hit) = if let Some(ref cached_text) = cached_translation {
        // Use cached translation
        Span::current().record("cache_hit", true);
        debug!("Cache hit for bubble");
        (serde_json::from_str(cached_text)?, true)
    } else {
        // Call API for translation
        Span::current().record("cache_hit", false);
        debug!("Cache miss, calling translation API");
        let trans = translator
            .translate_bubble(&bubble_bytes, batch_config.translation_model.as_deref())
            .await?;
        (trans, false)
    };

    // OPTIMIZATION: Perform local background complexity analysis
    // This can override the AI's redraw_bg_required flag if we detect a simple background locally
    // Saves expensive AI image generation API calls (20-30% reduction)
    let bubble_rgba = bubble_img.to_rgba8();
    let (is_complex_local, _detected_color) =
        RenderingService::analyze_background_complexity(&bubble_rgba);

    // Final decision: Use local analysis as a "veto" - if local analysis says simple, trust it
    let needs_ai_redraw = translation.redraw_bg_required && is_complex_local;

    let api_saved = !needs_ai_redraw;
    let is_simple_bg = !needs_ai_redraw;

    // Render bubble (may fail)
    let rendered_bubble = if needs_ai_redraw {
        // Complex background confirmed: need AI image generation
        tracing::debug!("Using AI redraw (AI said YES, local analysis CONFIRMED)");

        let cleaned_bytes = translator
            .remove_text_from_image(&bubble_bytes, batch_config.image_gen_model.as_deref())
            .await?;

        renderer
            .render_bubble_complex_background(&cleaned_bytes, &detection, &translation)
            .await?
    } else {
        // Simple background (either AI said so, or local analysis overrode AI's decision)
        if translation.redraw_bg_required && !is_complex_local {
            tracing::info!(
                "⚡ API call saved! AI wanted redraw, but local analysis detected simple background"
            );
        }

        renderer
            .render_bubble_simple_background(&bubble_rgba, &detection, &translation)
            .await?
    };

    // Only cache if rendering was successful (we reached this point without error)
    if !cache_hit {
        let trans_json = serde_json::to_string(&translation)?;
        cache.insert(checksum.clone(), trans_json);
    }

    Ok((rendered_bubble, cache_hit, api_saved, is_simple_bg))
}

/// Encode an image to a data URL for embedding in JSON responses
fn encode_image_to_data_url(img: &RgbaImage) -> Result<String> {
    let mut bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)?;
    let base64_data = general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:image/png;base64,{}", base64_data))
}
