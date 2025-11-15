// API v2 - New endpoints with async job queue support

use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use serde_json::json;
use std::sync::Arc;
use uuid::Uuid;

use crate::job_queue::JobQueue;
use crate::types::{BatchConfig, ImageData};

// Security limits for image uploads
const MAX_IMAGE_SIZE_BYTES: usize = 50 * 1024 * 1024; // 50MB
const MAX_IMAGE_DIMENSION: u32 = 10_000; // 10000x10000 pixels

/// Shared state for v2 API
#[derive(Clone)]
pub struct V2State {
    pub job_queue: Arc<JobQueue>,
}

/// Create v2 router with async job endpoints
pub fn create_v2_router() -> Router<V2State> {
    Router::new()
        .route("/jobs", post(submit_job))
        .route("/jobs/:job_id", get(get_job_status))
        .route("/jobs/:job_id/result", get(get_job_result))
        .route("/jobs/:job_id", delete(cancel_job))
        .route("/jobs/stats", get(job_stats))
}

/// Submit a new translation job
async fn submit_job(
    State(state): State<V2State>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Parse multipart data (same as v1)
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
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Multipart error: {}", e)))?
    {
        let field_name = field.name().map(|s| s.to_string());

        if field_name.as_deref() == Some("config") {
            // Parse JSON config
            if let Ok(data) = field.text().await {
                if let Ok(config) = serde_json::from_str::<BatchConfig>(&data) {
                    batch_config = config;
                }
            }
        } else if let Some(filename) = field.file_name() {
            let filename = filename.to_string();
            let data = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Read error: {}", e)))?;

            // Validate image size
            if data.len() > MAX_IMAGE_SIZE_BYTES {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "Image '{}' exceeds maximum size of {}MB (got {:.2}MB)",
                        filename,
                        MAX_IMAGE_SIZE_BYTES / (1024 * 1024),
                        data.len() as f64 / (1024.0 * 1024.0)
                    )
                ));
            }

            if let Ok(img) = image::load_from_memory(&data) {
                // Validate image dimensions
                if img.width() > MAX_IMAGE_DIMENSION || img.height() > MAX_IMAGE_DIMENSION {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!(
                            "Image '{}' dimensions {}x{} exceed maximum of {}x{}",
                            filename, img.width(), img.height(),
                            MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION
                        )
                    ));
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
        return Err((StatusCode::BAD_REQUEST, "No images provided".to_string()));
    }

    // Submit job
    let job_id = state.job_queue.submit(images, batch_config).await;

    Ok(Json(json!({
        "job_id": job_id,
        "status": "pending",
        "message": "Job submitted successfully"
    })))
}

/// Get job status
async fn get_job_status(
    State(state): State<V2State>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let job = state
        .job_queue
        .get_job(job_id)
        .await
        .ok_or((StatusCode::NOT_FOUND, "Job not found".to_string()))?;

    let status_str = match job.status {
        crate::job_queue::JobStatus::Pending => "pending",
        crate::job_queue::JobStatus::Processing => "processing",
        crate::job_queue::JobStatus::Completed => "completed",
        crate::job_queue::JobStatus::Failed => "failed",
    };

    Ok(Json(json!({
        "job_id": job.id,
        "status": status_str,
        "progress": job.progress,
        "created_at": job.created_at.elapsed().as_secs(),
        "processing_time_ms": job.processing_time_ms(),
        "error": job.error,
    })))
}

/// Get job result
async fn get_job_result(
    State(state): State<V2State>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let job = state
        .job_queue
        .get_job(job_id)
        .await
        .ok_or((StatusCode::NOT_FOUND, "Job not found".to_string()))?;

    match job.status {
        crate::job_queue::JobStatus::Completed => {
            let result = job
                .result
                .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Result not available".to_string()))?;
            Ok(Json(serde_json::to_value(result).unwrap()))
        }
        crate::job_queue::JobStatus::Failed => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            job.error.unwrap_or_else(|| "Job failed".to_string()),
        )),
        crate::job_queue::JobStatus::Processing => Err((
            StatusCode::ACCEPTED,
            "Job still processing".to_string(),
        )),
        crate::job_queue::JobStatus::Pending => Err((
            StatusCode::ACCEPTED,
            "Job pending".to_string(),
        )),
    }
}

/// Cancel a job
async fn cancel_job(
    State(state): State<V2State>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let cancelled = state.job_queue.cancel(job_id).await;

    if cancelled {
        Ok(Json(json!({
            "job_id": job_id,
            "status": "cancelled"
        })))
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            "Job cannot be cancelled (already processing or completed)".to_string(),
        ))
    }
}

/// Get job queue statistics
async fn job_stats(
    State(state): State<V2State>,
) -> Json<serde_json::Value> {
    let (pending, processing, completed, failed) = state.job_queue.stats().await;

    Json(json!({
        "pending": pending,
        "processing": processing,
        "completed": completed,
        "failed": failed,
        "total": pending + processing + completed + failed,
    }))
}
