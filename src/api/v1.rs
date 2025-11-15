// API v1 - Legacy endpoints for backward compatibility

use axum::{
    routing::{get, post},
    Router,
};
use crate::AppState;

/// Create v1 router with legacy endpoints
pub fn create_v1_router() -> Router<AppState> {
    Router::new()
        .route("/translate-batch", post(crate::translate_batch))
        .route("/health", get(crate::health))
        .route("/cache-stats", get(crate::cache_stats))
}
