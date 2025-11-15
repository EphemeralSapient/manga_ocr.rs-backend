// API error types with proper HTTP response handling
//
// Provides type-safe error handling for HTTP endpoints with automatic
// status code mapping and JSON error responses.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::fmt;

/// API error type that automatically converts to HTTP responses
///
/// This enum covers common HTTP error scenarios and provides
/// automatic status code mapping and JSON error bodies.
#[derive(Debug)]
pub enum ApiError {
    /// 400 Bad Request - Invalid input from client
    BadRequest(String),

    /// 422 Unprocessable Entity - Valid input but unable to process
    UnprocessableEntity(String),

    /// 500 Internal Server Error - Unexpected server-side error
    InternalError(String),

    /// 503 Service Unavailable - Service temporarily unavailable
    ServiceUnavailable(String),
}

/// JSON error response body
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<String>,
}

impl ApiError {
    /// Get HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::UnprocessableEntity(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ApiError::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    /// Get error type name for logging
    pub fn error_type(&self) -> &'static str {
        match self {
            ApiError::BadRequest(_) => "BAD_REQUEST",
            ApiError::UnprocessableEntity(_) => "UNPROCESSABLE_ENTITY",
            ApiError::InternalError(_) => "INTERNAL_ERROR",
            ApiError::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE",
        }
    }

    /// Get error message
    pub fn message(&self) -> &str {
        match self {
            ApiError::BadRequest(msg) => msg,
            ApiError::UnprocessableEntity(msg) => msg,
            ApiError::InternalError(msg) => msg,
            ApiError::ServiceUnavailable(msg) => msg,
        }
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.error_type(), self.message())
    }
}

impl std::error::Error for ApiError {}

/// Implement IntoResponse for automatic HTTP response conversion
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_type = self.error_type();
        let message = self.message().to_string();

        // Log error for server-side tracking
        match &self {
            ApiError::InternalError(_) | ApiError::ServiceUnavailable(_) => {
                tracing::error!("[API ERROR] {}: {}", error_type, message);
            }
            _ => {
                tracing::warn!("[API ERROR] {}: {}", error_type, message);
            }
        }

        let body = ErrorResponse {
            error: error_type.to_string(),
            details: Some(message),
        };

        (status, Json(body)).into_response()
    }
}

/// Convert from anyhow::Error to ApiError (defaults to InternalError)
impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError::InternalError(err.to_string())
    }
}

/// Convert from std::io::Error to ApiError
impl From<std::io::Error> for ApiError {
    fn from(err: std::io::Error) -> Self {
        ApiError::InternalError(format!("I/O error: {}", err))
    }
}

/// Convert from serde_json::Error to ApiError
impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        ApiError::BadRequest(format!("Invalid JSON: {}", err))
    }
}

/// Convert from image::ImageError to ApiError
impl From<image::ImageError> for ApiError {
    fn from(err: image::ImageError) -> Self {
        ApiError::UnprocessableEntity(format!("Image processing error: {}", err))
    }
}
