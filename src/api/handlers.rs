// HTTP request handlers for batch translation
//
// This module contains the business logic for handling HTTP requests,
// separated from routing concerns in main.rs

use axum::{
    extract::{Multipart, State},
    response::Json,
};
use image;
use std::sync::Arc;
use std::time::Instant;
use tracing::{error, info};

use crate::api::ApiError;
use crate::types::{BatchAnalytics, BatchConfig, BatchResult, ImageData, ModelInfo, PageResult, PerformanceMetrics};
use crate::AppState;

// Security limits for image uploads
const MAX_IMAGE_SIZE_BYTES: usize = 50 * 1024 * 1024; // 50MB
const MAX_IMAGE_DIMENSION: u32 = 10_000; // 10000x10000 pixels

/// Handle batch translation request
///
/// Parses multipart form data, processes images through the pipeline,
/// and returns comprehensive results with analytics.
///
/// # Arguments
/// * `state` - Application state with pipeline and configuration
/// * `multipart` - Multipart form data containing images and optional config
///
/// # Returns
/// JSON response with processing results and metrics
///
/// # Errors
/// Returns `ApiError` for invalid inputs or processing failures
pub async fn handle_translate_batch(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<BatchResult>, ApiError> {
    let start = Instant::now();

    info!("📨 Received batch translation request");

    // Extract images and config from multipart
    let mut images = Vec::new();
    let mut index = 1;
    let mut batch_config = BatchConfig {
        translation_model: None,
        image_gen_model: None,
        font_family: None,
    };

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::BadRequest(format!("Invalid multipart data: {}", e)))?
    {
        let field_name = field.name().map(|s| s.to_string());

        if field_name.as_deref() == Some("config") {
            // Parse JSON config
            if let Ok(data) = field.text().await {
                if let Ok(config) = serde_json::from_str::<BatchConfig>(&data) {
                    batch_config = config;
                    info!("📋 Using custom models: translation={:?}, image_gen={:?}",
                        batch_config.translation_model, batch_config.image_gen_model);
                }
            }
        } else if let Some(filename) = field.file_name() {
            let filename = filename.to_string();
            let data = field
                .bytes()
                .await
                .map_err(|e| ApiError::BadRequest(format!("Failed to read image data: {}", e)))?;

            // Validate image size
            if data.len() > MAX_IMAGE_SIZE_BYTES {
                return Err(ApiError::BadRequest(format!(
                    "Image '{}' exceeds maximum size of {}MB (got {:.2}MB)",
                    filename,
                    MAX_IMAGE_SIZE_BYTES / (1024 * 1024),
                    data.len() as f64 / (1024.0 * 1024.0)
                )));
            }

            if let Ok(img) = image::load_from_memory(&data) {
                // Validate image dimensions
                if img.width() > MAX_IMAGE_DIMENSION || img.height() > MAX_IMAGE_DIMENSION {
                    return Err(ApiError::BadRequest(format!(
                        "Image '{}' dimensions {}x{} exceed maximum of {}x{}",
                        filename, img.width(), img.height(),
                        MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION
                    )));
                }

                images.push(ImageData {
                    index,
                    filename,
                    image_bytes: Arc::new(data.to_vec()),
                    width: img.width(),
                    height: img.height(),
                });
                index += 1;
            }
        }
    }

    if images.is_empty() {
        return Err(ApiError::BadRequest(
            "No valid images provided in multipart request".to_string()
        ));
    }

    info!("📥 Loaded {} images for processing", images.len());

    // Process in batches
    let batch_size = state.config.batch_size;
    let mut all_results = Vec::new();
    let mut metrics = PerformanceMetrics::default();

    for (batch_idx, chunk) in images.chunks(batch_size).enumerate() {
        info!(
            "[BATCH {}/{}] Processing {} images...",
            batch_idx + 1,
            (images.len() + batch_size - 1) / batch_size,
            chunk.len()
        );

        match state.pipeline.process_batch(chunk.to_vec(), &batch_config).await {
            Ok((results, batch_metrics)) => {
                all_results.extend(results);
                metrics.merge(&batch_metrics);
            }
            Err(e) => {
                error!("[BATCH {}] Failed: {}", batch_idx + 1, e);
                // Add error results
                for img in chunk {
                    all_results.push(PageResult {
                        index: img.index,
                        filename: img.filename.clone(),
                        success: false,
                        bubbles_detected: 0,
                        bubbles_translated: 0,
                        cache_hit: false,
                        processing_time_ms: 0.0,
                        error: Some(e.to_string()),
                        data_url: None,
                    });
                }
            }
        }
    }

    let total_time = start.elapsed().as_secs_f64() * 1000.0;
    let successful = all_results.iter().filter(|r| r.success).count();

    info!(
        "🎉 Batch complete: {}/{} successful in {:.2}ms",
        successful,
        images.len(),
        total_time
    );

    // Detailed analytics logging
    let total_bubbles_detected: usize = all_results.iter().map(|r| r.bubbles_detected).sum();
    let total_bubbles_translated: usize = all_results.iter().map(|r| r.bubbles_translated).sum();

    info!("{}", "=".repeat(70));
    info!("📊 DETAILED ANALYTICS");
    info!("{}", "=".repeat(70));
    info!("Bubble Statistics:");
    info!("  Total bubbles detected: {}", total_bubbles_detected);
    info!("  Total bubbles translated: {}", total_bubbles_translated);
    info!("  Translation success rate: {:.1}%",
        if total_bubbles_detected > 0 {
            (total_bubbles_translated as f64 / total_bubbles_detected as f64) * 100.0
        } else { 0.0 }
    );
    info!("");
    info!("Cache Performance:");
    info!("  Cache hits: {}", metrics.cache_hits);
    info!("  Cache misses: {}", metrics.cache_misses);
    info!("  Cache hit rate: {:.1}%",
        if metrics.cache_hits + metrics.cache_misses > 0 {
            (metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64) * 100.0
        } else { 0.0 }
    );
    info!("");
    info!("Background Processing:");
    info!("  Simple backgrounds (no AI redraw): {}", metrics.simple_backgrounds);
    info!("  Complex backgrounds (AI redraw): {}", metrics.background_redraws);
    info!("  API calls saved: {}", metrics.api_calls_saved);
    info!("  Cost optimization: {:.1}%",
        if metrics.simple_backgrounds + metrics.background_redraws > 0 {
            (metrics.simple_backgrounds as f64 / (metrics.simple_backgrounds + metrics.background_redraws) as f64) * 100.0
        } else { 0.0 }
    );
    info!("");
    info!("Timing Breakdown:");
    info!("  Avg detection time per page: {:.2}ms",
        if successful > 0 {
            metrics.detection_time.as_secs_f64() * 1000.0 / successful as f64
        } else { 0.0 }
    );
    info!("  Avg translation time per bubble: {:.2}ms",
        if total_bubbles_translated > 0 {
            metrics.translation_time.as_secs_f64() * 1000.0 / total_bubbles_translated as f64
        } else { 0.0 }
    );
    info!("  Total processing time: {:.2}ms", total_time);
    info!("  Avg time per page: {:.2}ms", total_time / images.len() as f64);
    info!("{}", "=".repeat(70));

    // Build analytics
    let cache_hit_rate = if metrics.cache_hits + metrics.cache_misses > 0 {
        metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64 * 100.0
    } else {
        0.0
    };

    let analytics = BatchAnalytics {
        total_bubbles_detected,
        total_bubbles_translated,
        cache_hit_rate,
        cache_hits: metrics.cache_hits,
        cache_misses: metrics.cache_misses,
        api_calls_made: metrics.api_calls_made,
        api_calls_saved: metrics.api_calls_saved,
        background_redraws: metrics.background_redraws,
        simple_backgrounds: metrics.simple_backgrounds,
        average_detection_time_ms: if successful > 0 {
            metrics.detection_time.as_secs_f64() * 1000.0 / successful as f64
        } else {
            0.0
        },
        average_translation_time_ms: if total_bubbles_translated > 0 {
            metrics.translation_time.as_secs_f64() * 1000.0 / total_bubbles_translated as f64
        } else {
            0.0
        },
        average_rendering_time_ms: if total_bubbles_translated > 0 {
            metrics.rendering_time.as_secs_f64() * 1000.0 / total_bubbles_translated as f64
        } else {
            0.0
        },
        model_used: ModelInfo {
            translation_model: batch_config.translation_model
                .unwrap_or_else(|| state.config.translation_model().to_string()),
            image_gen_model: batch_config.image_gen_model
                .unwrap_or_else(|| state.config.image_gen_model().to_string()),
        },
    };

    Ok(Json(BatchResult {
        total: images.len(),
        successful,
        failed: images.len() - successful,
        processing_time_ms: total_time,
        average_time_per_page_ms: total_time / images.len() as f64,
        analytics,
        results: all_results,
    }))
}
