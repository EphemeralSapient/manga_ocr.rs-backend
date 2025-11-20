// Main entry point for the manga text processing workflow

use manga_workflow::{
    core::{Config, types::*},
    orchestration::batch_orchestrator::BatchOrchestrator,
    utils::Metrics,
};

use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    orchestrator: Arc<BatchOrchestrator>,
    metrics: Metrics,
    /// Global mask mode setting (true = use segmentation mask, false = fill with white)
    mask_enabled: Arc<AtomicBool>,
    /// Current session limit per model (detection and segmentation each get this many)
    session_limit: Arc<AtomicUsize>,
    /// Global batch inference mode (true = merge images into single tensor)
    merge_img_enabled: Arc<AtomicBool>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Arc::new(Config::new().expect("Failed to load configuration"));

    // Initialize logging
    use tracing_subscriber::EnvFilter;

    let level_str = match config.log_level() {
        tracing::Level::TRACE => "trace",
        tracing::Level::DEBUG => "debug",
        tracing::Level::INFO => "info",
        tracing::Level::WARN => "warn",
        tracing::Level::ERROR => "error",
    };

    // Set global default to configured level, then override noisy dependencies
    let filter = EnvFilter::new(format!(
        "{},ort=warn,h2=warn,tower_http=warn,hyper=warn,tokio=info,runtime=warn,xnnpack=warn,cosmic_text=info",
        level_str
    ));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)  // Show module paths
        .with_thread_ids(false)  // Cleaner output
        .init();

    info!("Logging initialized at level: {}", level_str.to_uppercase());

    info!("=== MANGA TEXT PROCESSOR - OPTIMIZED ===");
    info!("Config: N={} M={} Batches={} Banana={}",
        config.batch_size_n(),
        config.api_batch_size_m(),
        config.max_concurrent_batches(),
        if config.banana_mode_enabled() { "ON" } else { "OFF" }
    );

    // Initialize metrics
    let metrics = Metrics::new();

    // Initialize batch orchestrator
    info!("Initializing batch orchestrator...");
    let orchestrator = Arc::new(BatchOrchestrator::new(config.clone()).await?);
    let initial_session_limit = config.onnx_pool_size();
    let state = AppState {
        config: config.clone(),
        orchestrator,
        metrics,
        mask_enabled: Arc::new(AtomicBool::new(true)), // Default: mask enabled
        session_limit: Arc::new(AtomicUsize::new(initial_session_limit)),
        merge_img_enabled: Arc::new(AtomicBool::new(false)), // Default: batch mode disabled
    };

    // Setup CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Create router with monitoring endpoints
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/health/api-keys", get(health_api_keys))
        .route("/metrics", get(metrics_endpoint))
        .route("/stats", get(stats_endpoint))
        .route("/process", post(process_images))
        .route("/mask-toggle", post(mask_toggle))
        .route("/mask-status", get(mask_status))
        .route("/sessions-limit", post(sessions_limit))
        .route("/sessions-status", get(sessions_status))
        .route("/mergeimg-toggle", post(mergeimg_toggle))
        .route("/mergeimg-status", get(mergeimg_status))
        .with_state(state)
        .layer(DefaultBodyLimit::max(200 * 1024 * 1024)) // 200MB for large batches
        .layer(cors);

    let addr = format!("{}:{}", config.server_host(), config.server_port());
    info!("{}", "=".repeat(70));
    info!("Server starting on http://{}", addr);
    info!("{}", "-".repeat(70));
    info!("Endpoints:");
    info!("  GET  /                - Root endpoint");
    info!("  GET  /health          - Health check");
    info!("  GET  /health/api-keys - API key health status");
    info!("  GET  /metrics         - Prometheus metrics");
    info!("  GET  /stats           - Detailed statistics");
    info!("  POST /process         - Process images (multipart/form-data)");
    info!("  POST /mask-toggle     - Toggle mask segmentation mode");
    info!("  GET  /mask-status     - Get current mask mode status");
    info!("  POST /sessions-limit  - Set session limit (requires restart)");
    info!("  GET  /sessions-status - Get current session configuration");
    info!("  POST /mergeimg-toggle - Toggle batch inference mode");
    info!("  GET  /mergeimg-status - Get batch inference mode status");
    info!("{}", "=".repeat(70));

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn root() -> &'static str {
    "Manga Text Processing Workflow - Optimized Rust Version"
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mask_enabled = state.mask_enabled.load(Ordering::SeqCst);
    let merge_img_enabled = state.merge_img_enabled.load(Ordering::SeqCst);
    let session_limit = state.session_limit.load(Ordering::SeqCst);

    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "config": {
            "mask_enabled": mask_enabled,
            "merge_img_enabled": merge_img_enabled,
            "session_limit": session_limit,
            "onnx_pool_size": state.config.onnx_pool_size(),
            "batch_size_n": state.config.batch_size_n(),
            "max_concurrent_batches": state.config.max_concurrent_batches(),
        }
    }))
}

/// API key health status endpoint
async fn health_api_keys(State(state): State<AppState>) -> Json<serde_json::Value> {
    // TODO: Get actual API key health from orchestrator
    Json(serde_json::json!({
        "status": "healthy",
        "total_keys": state.config.api_keys().len(),
        "healthy_keys": state.config.api_keys().len(),
        "degraded_keys": 0,
        "unhealthy_keys": 0,
    }))
}

/// Prometheus metrics endpoint
async fn metrics_endpoint(State(state): State<AppState>) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("Content-Type", "text/plain; version=0.0.4")],
        state.metrics.to_prometheus(),
    )
}

/// Detailed statistics endpoint (JSON)
async fn stats_endpoint(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let snapshot = state.metrics.snapshot();
    serde_json::to_value(snapshot)
        .map(Json)
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to serialize metrics: {}", e),
            )
        })
}

/// Toggle mask segmentation mode
///
/// When mask is enabled (default): Uses segmentation mask for precise text removal
/// When mask is disabled: Fills entire label 1 region with white background
///
/// Label 2 (free text) always fills entire region (never uses mask)
async fn mask_toggle(State(state): State<AppState>) -> Json<serde_json::Value> {
    let current = state.mask_enabled.load(Ordering::SeqCst);
    let new_value = !current;
    state.mask_enabled.store(new_value, Ordering::SeqCst);

    info!("Mask mode toggled: {} -> {}",
        if current { "enabled" } else { "disabled" },
        if new_value { "enabled" } else { "disabled" }
    );

    Json(serde_json::json!({
        "mask_enabled": new_value,
        "message": if new_value {
            "Mask mode enabled - using segmentation mask for text removal"
        } else {
            "Mask mode disabled - filling entire label 1 region with white"
        }
    }))
}

/// Get current mask mode status
async fn mask_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let enabled = state.mask_enabled.load(Ordering::SeqCst);
    Json(serde_json::json!({
        "mask_enabled": enabled,
        "description": if enabled {
            "Using segmentation mask for precise text removal"
        } else {
            "Filling entire label 1 region with white background"
        }
    }))
}

/// Session limit request body
#[derive(serde::Deserialize)]
struct SessionLimitRequest {
    /// Number of sessions per model (detection and segmentation each)
    limit: usize,
}

/// Set session limit (requires server restart to take effect)
///
/// # Arguments
/// - limit: Number of sessions per model (1-16 recommended)
///
/// Note: Full dynamic resizing is not yet implemented.
/// This sets the desired limit which requires restart to apply.
async fn sessions_limit(
    State(state): State<AppState>,
    Json(payload): Json<SessionLimitRequest>,
) -> Json<serde_json::Value> {
    let old_limit = state.session_limit.load(Ordering::SeqCst);
    let new_limit = payload.limit.clamp(1, 32); // Reasonable bounds

    state.session_limit.store(new_limit, Ordering::SeqCst);

    info!("Session limit set: {} -> {} (restart required to apply)", old_limit, new_limit);

    // Calculate memory estimates
    let detection_mem_mb = new_limit * 42; // ~42MB per detection session
    let segmentation_mem_mb = new_limit * 40; // ~40MB per segmentation session
    let total_mem_mb = detection_mem_mb + segmentation_mem_mb;

    Json(serde_json::json!({
        "previous_limit": old_limit,
        "new_limit": new_limit,
        "detection_sessions": new_limit,
        "segmentation_sessions": new_limit,
        "estimated_memory_mb": total_mem_mb,
        "status": "pending_restart",
        "message": format!(
            "Session limit set to {}. Restart server with ONNX_POOL_SIZE={} to apply. \
            Estimated memory: {} MB ({} detection + {} segmentation)",
            new_limit, new_limit, total_mem_mb, detection_mem_mb, segmentation_mem_mb
        )
    }))
}

/// Get current session configuration and status
async fn sessions_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let current_limit = state.config.onnx_pool_size();
    let desired_limit = state.session_limit.load(Ordering::SeqCst);
    let mask_enabled = state.mask_enabled.load(Ordering::SeqCst);

    // Calculate memory usage
    let detection_mem_mb = current_limit * 42;
    let segmentation_mem_mb = if mask_enabled { current_limit * 40 } else { 0 };
    let total_mem_mb = detection_mem_mb + segmentation_mem_mb;

    Json(serde_json::json!({
        "current": {
            "detection_sessions": current_limit,
            "segmentation_sessions": if mask_enabled { current_limit } else { 0 },
            "total_sessions": if mask_enabled { current_limit * 2 } else { current_limit },
            "estimated_memory_mb": total_mem_mb
        },
        "desired": {
            "limit": desired_limit,
            "needs_restart": desired_limit != current_limit
        },
        "mask_enabled": mask_enabled,
        "note": if !mask_enabled {
            "Segmentation sessions not used when mask is disabled"
        } else {
            "Both detection and segmentation sessions active"
        }
    }))
}

/// Toggle batch inference mode (mergeImg)
///
/// When enabled: Multiple images are batched into single ONNX inference
/// When disabled: Each image processed with separate inference (default)
async fn mergeimg_toggle(State(state): State<AppState>) -> Json<serde_json::Value> {
    let current = state.merge_img_enabled.load(Ordering::SeqCst);
    let new_value = !current;
    state.merge_img_enabled.store(new_value, Ordering::SeqCst);

    info!("Batch inference mode toggled: {} -> {}",
        if current { "enabled" } else { "disabled" },
        if new_value { "enabled" } else { "disabled" }
    );

    Json(serde_json::json!({
        "merge_img_enabled": new_value,
        "message": if new_value {
            "Batch inference enabled - images will be merged into single ONNX tensor for faster GPU processing"
        } else {
            "Batch inference disabled - each image processed with separate inference"
        }
    }))
}

/// Get current batch inference mode status
async fn mergeimg_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let enabled = state.merge_img_enabled.load(Ordering::SeqCst);
    Json(serde_json::json!({
        "merge_img_enabled": enabled,
        "description": if enabled {
            "Images are batched into single ONNX inference for better GPU utilization"
        } else {
            "Each image is processed with separate ONNX inference"
        }
    }))
}

/// Process images endpoint
///
/// # Request Format:
/// - multipart/form-data
/// - Field "images": One or more image files (PNG/JPEG)
/// - Field "config" (optional): JSON configuration
///
/// # Response:
/// - BatchResult JSON with all results and analytics
async fn process_images(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<BatchResult>, (StatusCode, String)> {
    let start_time = std::time::Instant::now();

    info!("Received process request");

    let mut images = Vec::new();
    let mut config = ProcessingConfig::default();

    // Parse multipart form
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Multipart error: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "images" => {
                let filename = field.file_name().unwrap_or("unknown.png").to_string();

                let data = field
                    .bytes()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Read error: {}", e)))?;

                // OPTIMIZATION: Load image once and reuse for both dimensions and processing
                // This eliminates redundant decoding in phases (saves ~15-20% processing time)
                let img = image::load_from_memory(&data)
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid image: {}", e)))?;

                images.push(ImageData {
                    index: images.len(),
                    filename,
                    image_bytes: Arc::new(data.to_vec()),
                    width: img.width(),
                    height: img.height(),
                    decoded_image: Some(Arc::new(img)),
                });
            }
            "config" => {
                let config_data = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Config read error: {}", e)))?;

                config = serde_json::from_str(&config_data).map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Invalid config JSON: {}", e),
                    )
                })?;
            }
            _ => {}
        }
    }

    if images.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "No images provided".to_string(),
        ));
    }

    info!("Processing {} images", images.len());

    // Apply global settings if not overridden in request config
    if config.use_mask.is_none() {
        config.use_mask = Some(state.mask_enabled.load(Ordering::SeqCst));
    }
    if config.merge_img.is_none() {
        config.merge_img = Some(state.merge_img_enabled.load(Ordering::SeqCst));
    }

    // Process batch
    let result = state
        .orchestrator
        .process_batch(images, &config)
        .await
        .map_err(|e| {
            error!("Batch processing failed: {:?}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Processing failed: {}", e),
            )
        })?;

    info!(
        "Request completed in {:.2}s: {} successful, {} failed",
        start_time.elapsed().as_secs_f64(),
        result.successful,
        result.failed
    );

    Ok(Json(result))
}
