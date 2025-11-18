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
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    orchestrator: Arc<BatchOrchestrator>,
    metrics: Metrics,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Arc::new(Config::new().expect("Failed to load configuration"));

    // Initialize logging
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::new(format!(
        "manga_workflow={},ort=off",
        match config.log_level() {
            tracing::Level::TRACE => "trace",
            tracing::Level::DEBUG => "debug",
            tracing::Level::INFO => "info",
            tracing::Level::WARN => "warn",
            tracing::Level::ERROR => "error",
        }
    ));

    tracing_subscriber::fmt().with_env_filter(filter).init();

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
    let state = AppState {
        config: config.clone(),
        orchestrator,
        metrics,
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
    info!("{}", "=".repeat(70));

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn root() -> &'static str {
    "Manga Text Processing Workflow - Optimized Rust Version"
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
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

                // Load image to get dimensions
                let img = image::load_from_memory(&data)
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid image: {}", e)))?;

                images.push(ImageData {
                    index: images.len(),
                    filename,
                    image_bytes: Arc::new(data.to_vec()),
                    width: img.width(),
                    height: img.height(),
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
