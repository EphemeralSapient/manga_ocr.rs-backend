mod api;
mod cache;
mod cli;
mod cli_interactive;
mod config;
mod cosmic_renderer;
mod detection;
mod errors;
mod job_queue;
mod middleware;
mod pipeline;
mod rendering;
mod schema;
mod services;
mod translation;
mod types;

use anyhow::Result;
use axum::{
    extract::{Multipart, State, DefaultBodyLimit},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use cache::Cache;
use clap::Parser;
use cli::{Cli, Commands};
use config::Config;
use detection::DetectionService;
use job_queue::JobQueue;
use pipeline::ProcessingPipeline;
use rendering::RenderingService;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use translation::TranslationService;
use types::BatchResult;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub pipeline: Arc<ProcessingPipeline>,
    pub job_queue: Arc<JobQueue>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Load config first to get log level
    let config = Arc::new(Config::new().expect("Failed to load configuration"));

    // Initialize logging with configured level, but completely silence ort library logs
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

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    // Route to appropriate mode
    match cli.command {
        Some(Commands::Interactive) => {
            // Interactive mode with prompts
            return cli_interactive::execute_interactive(config).await;
        }
        Some(Commands::Phase1 { input, output, visualize, include_text_free }) => {
            // Phase 1: Detection only
            return cli::execute_phase1(input, output, visualize, include_text_free, config).await;
        }
        Some(Commands::Phase2 { input, detections_json, output, translation_model, font_family }) => {
            // Phase 2: Translation only
            return cli::execute_phase2(input, detections_json, output, translation_model, font_family, config).await;
        }
        Some(Commands::Phase3 { input, api_response, detections_json, output, image_gen_model, font_family, debug_polygon, speech, insertion }) => {
            // Phase 3: Rendering only
            return cli::execute_phase3(input, api_response, detections_json, output, image_gen_model, font_family, debug_polygon, speech, insertion, config).await;
        }
        Some(Commands::Server) | None => {
            // Default: Run HTTP server
        }
    }

    info!("{}", "=".repeat(70));
    info!("MANGA TRANSLATION WORKFLOW SERVER - RUST VERSION");
    info!("{}", "=".repeat(70));
    info!("Log level: {:?}", config.log_level());

    info!("Initializing services...");
    let detection_service = Arc::new(DetectionService::new(config.clone()).await?);
    info!("✓ Detection service ready ({})", detection_service.device_type());

    let translation_service = Arc::new(TranslationService::new(config.clone()));
    info!("✓ Translation service ready ({} API keys configured)", config.api_keys().len());

    let rendering_service = Arc::new(RenderingService::new(config.clone()));
    info!("✓ Rendering service ready ({}x upscaling)", config.upscale_factor());

    let cache_instance = Cache::new(config.cache_dir())?;
    let (cache_entries, cache_size_mb) = cache_instance.stats();
    info!("✓ Cache ready ({} entries, {:.2} MB)", cache_entries, cache_size_mb);

    // Cache is internally thread-safe, no external Mutex needed
    let cache: Arc<dyn services::traits::CacheStore> = Arc::new(cache_instance);

    // Create the processing pipeline with trait objects
    let pipeline = Arc::new(ProcessingPipeline::new(
        detection_service as Arc<dyn services::traits::BubbleDetector>,
        translation_service as Arc<dyn services::traits::Translator>,
        rendering_service as Arc<dyn services::traits::BubbleRenderer>,
        cache,
    ));

    // Create job queue
    let job_queue = Arc::new(JobQueue::new(Arc::clone(&pipeline)));
    info!("✓ Job queue initialized");

    // Create rate limiter
    let rate_limiter_config = api::rate_limit::RateLimiterConfig::from_config(&config);
    let rate_limiter = Arc::new(api::rate_limit::RateLimiter::new(rate_limiter_config.clone()));
    if rate_limiter_config.enabled {
        info!("✓ Rate limiting enabled ({} requests per {} seconds)",
            rate_limiter_config.max_requests,
            rate_limiter_config.window.as_secs()
        );
    } else {
        info!("✓ Rate limiting disabled");
    }

    let state = AppState {
        config: config.clone(),
        pipeline,
        job_queue: Arc::clone(&job_queue),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Create v2 state
    let v2_state = api::v2::V2State {
        job_queue: Arc::clone(&job_queue),
    };

    let mut app = Router::new()
        .route("/", get(root))
        // Legacy endpoints (no version prefix)
        .route("/health", get(health))
        .route("/translate-batch", post(translate_batch))
        .route("/cache-stats", get(cache_stats))
        // v1 endpoints (explicit versioning)
        .nest("/v1", api::v1::create_v1_router())
        // v2 endpoints (with async job queue)
        .nest("/v2", api::v2::create_v2_router().with_state(v2_state))
        .with_state(state);

    // Add body size limit (100MB to allow multiple 50MB images in a batch)
    app = app.layer(DefaultBodyLimit::max(100 * 1024 * 1024));

    // Conditionally add rate limiting middleware
    if rate_limiter_config.enabled {
        use axum::middleware;
        app = app.layer(middleware::from_fn_with_state(
            rate_limiter,
            api::rate_limit::rate_limit_middleware,
        ));
    }

    // Add CORS after everything else
    let app = app.layer(cors);

    let addr = format!("{}:{}", config.server_host(), config.server_port());
    info!("{}", "=".repeat(70));
    info!("Server starting on http://{}", addr);
    if rate_limiter_config.enabled {
        info!("Rate Limiting: ENABLED ({} req/{} sec per IP)",
            rate_limiter_config.max_requests,
            rate_limiter_config.window.as_secs()
        );
    } else {
        info!("Rate Limiting: DISABLED");
    }
    info!("{}", "-".repeat(70));
    info!("Endpoints:");
    info!("  Legacy (no version):");
    info!("    POST /translate-batch  - Batch translate images (sync)");
    info!("    GET  /health           - Health check");
    info!("    GET  /cache-stats      - Cache statistics");
    info!("  API v1 (explicit versioning):");
    info!("    POST /v1/translate-batch  - Batch translate images (sync)");
    info!("    GET  /v1/health           - Health check");
    info!("    GET  /v1/cache-stats      - Cache statistics");
    info!("  API v2 (async job queue):");
    info!("    POST   /v2/jobs           - Submit translation job");
    info!("    GET    /v2/jobs/:id       - Get job status");
    info!("    GET    /v2/jobs/:id/result - Get job result");
    info!("    DELETE /v2/jobs/:id       - Cancel job");
    info!("    GET    /v2/jobs/stats     - Job queue statistics");
    info!("{}", "-".repeat(70));
    info!("CLI Testing Modes:");
    info!("  cargo run -- phase1 --input <image>         - Detect bubbles only");
    info!("  cargo run -- phase2 --input <bubble>        - Translate bubble only");
    info!("  cargo run -- phase3 --input <bubble> --api-response <json> - Render only");
    info!("  cargo run -- --help                          - Show all CLI options");
    info!("{}", "=".repeat(70));

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

pub async fn root() -> &'static str {
    "Manga Translation Workflow Server - Rust Version"
}

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let (cache_entries, cache_size_mb) = state.pipeline.cache.stats();

    Json(serde_json::json!({
        "status": "healthy",
        "detection_device": state.pipeline.detector.device_type(),
        "api_keys": state.config.api_keys().len(),
        "cache_entries": cache_entries,
        "cache_size_mb": cache_size_mb,
        "batch_size": state.config.batch_size,
    }))
}

pub async fn cache_stats(State(state): State<AppState>) -> impl IntoResponse {
    let (entries, size_mb) = state.pipeline.cache.stats();

    Json(serde_json::json!({
        "entries": entries,
        "size_mb": size_mb,
    }))
}

/// Legacy endpoint wrapper - delegates to api::handlers::handle_translate_batch
pub async fn translate_batch(
    state: State<AppState>,
    multipart: Multipart,
) -> Result<Json<BatchResult>, api::ApiError> {
    api::handlers::handle_translate_batch(state, multipart).await
}
