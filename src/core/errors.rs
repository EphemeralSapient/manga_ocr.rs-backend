// Custom error types for better error handling and debugging
//
// Using thiserror for ergonomic error definitions with:
// - Context preservation
// - Type-safe error matching
// - Automatic Display/Error trait implementations
// - Source error chaining

use thiserror::Error;

/// Detection service errors
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum DetectionError {
    #[error("ONNX inference failed: {0}")]
    InferenceFailed(#[from] ort::Error),

    #[error("Image preprocessing failed: {0}")]
    PreprocessingFailed(String),

    #[error("No bubbles detected with confidence >= {threshold} (page {page_index})")]
    NoBubbles { page_index: usize, threshold: f32 },

    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidImageSize { width: u32, height: u32 },
}

/// Translation service errors
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum TranslationError {
    #[error("All {key_count} API keys exhausted after {attempts} attempts")]
    AllKeysFailed { key_count: usize, attempts: usize },

    #[error("API request failed: {0}")]
    ApiRequestFailed(#[from] reqwest::Error),

    #[error("Translation validation failed: {reason}")]
    ValidationFailed { reason: String },

    #[error("Invalid response format: {0}")]
    InvalidResponse(String),

    #[error("Rate limit exceeded for API key (retry after {0}s)")]
    RateLimited(u64),

    #[error("API key {key_index} is unhealthy (consecutive failures: {failures})")]
    UnhealthyApiKey { key_index: usize, failures: usize },
}

/// Rendering service errors
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum RenderingError {
    #[error("Font not found: {family} (searched paths: {paths:?})")]
    FontNotFound {
        family: String,
        paths: Vec<String>,
    },

    #[error("Text doesn't fit in bubble (text: {text_width}x{text_height}, bubble: {bubble_width}x{bubble_height})")]
    TextOverflow {
        text_width: u32,
        text_height: u32,
        bubble_width: u32,
        bubble_height: u32,
    },

    #[error("Image processing failed: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Font loading failed: {0}")]
    FontLoadError(String),

    #[error("Invalid bubble dimensions: {0}")]
    InvalidBubbleDimensions(String),
}

/// Cache storage errors
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum CacheError {
    #[error("Failed to load cache from {path}: {source}")]
    LoadFailed {
        path: String,
        source: std::io::Error,
    },

    #[error("Failed to save cache to {path}: {source}")]
    SaveFailed {
        path: String,
        source: std::io::Error,
    },

    #[error("Cache deserialization failed: {0}")]
    DeserializationFailed(#[from] serde_json::Error),

    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    #[error("Cache directory creation failed: {0}")]
    DirectoryCreationFailed(std::io::Error),
}

/// Pipeline orchestration errors
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum PipelineError {
    #[error("Detection failed on page {page_index}: {source}")]
    DetectionFailed {
        page_index: usize,
        #[source]
        source: DetectionError,
    },

    #[error("Translation failed for bubble {bubble_index} on page {page_index}: {source}")]
    TranslationFailed {
        page_index: usize,
        bubble_index: usize,
        #[source]
        source: TranslationError,
    },

    #[error("Rendering failed for bubble {bubble_index} on page {page_index}: {source}")]
    RenderingFailed {
        page_index: usize,
        bubble_index: usize,
        #[source]
        source: RenderingError,
    },

    #[error("Cache operation failed: {0}")]
    CacheFailed(#[from] CacheError),

    #[error("Image loading failed for page {page_index}: {source}")]
    ImageLoadFailed {
        page_index: usize,
        source: image::ImageError,
    },

    #[error("Batch processing failed: {0}")]
    BatchFailed(String),

    #[error("Task join failed: {0}")]
    TaskJoinFailed(String),
}

/// Configuration errors
#[derive(Debug, Error)]
#[allow(dead_code)]
pub enum ConfigError {
    #[error("No API keys configured (set GEMINI_API_KEYS environment variable)")]
    NoApiKeys,

    #[error("Invalid detection config: {0}")]
    InvalidDetectionConfig(String),

    #[error("Invalid translation config: {0}")]
    InvalidTranslationConfig(String),

    #[error("Invalid rendering config: {0}")]
    InvalidRenderingConfig(String),

    #[error("Invalid cache path: {0}")]
    InvalidCachePath(String),

    #[error("Confidence threshold must be in [0.0, 1.0], got {0}")]
    InvalidConfidenceThreshold(f32),

    #[error("IoU threshold must be in [0.0, 1.0], got {0}")]
    InvalidIoUThreshold(f32),

    #[error("Batch size must be > 0, got {0}")]
    InvalidBatchSize(usize),

    #[error("Environment variable parsing failed: {0}")]
    EnvVarError(String),
}

// Convenience type aliases for Results
pub type DetectionResult<T> = Result<T, DetectionError>;
pub type TranslationResult<T> = Result<T, TranslationError>;
pub type RenderingResult<T> = Result<T, RenderingError>;
#[allow(dead_code)]
pub type CacheResult<T> = Result<T, CacheError>;
#[allow(dead_code)]
pub type PipelineResult<T> = Result<T, PipelineError>;
#[allow(dead_code)]
pub type ConfigResult<T> = Result<T, ConfigError>;

// Helper trait for adding context to errors
#[allow(dead_code)]
pub trait ErrorContext<T> {
    fn with_page_context(self, page_index: usize) -> Result<T, PipelineError>;
    fn with_bubble_context(
        self,
        page_index: usize,
        bubble_index: usize,
    ) -> Result<T, PipelineError>;
}

impl<T> ErrorContext<T> for DetectionResult<T> {
    fn with_page_context(self, page_index: usize) -> Result<T, PipelineError> {
        self.map_err(|e| PipelineError::DetectionFailed {
            page_index,
            source: e,
        })
    }

    fn with_bubble_context(
        self,
        _page_index: usize,
        _bubble_index: usize,
    ) -> Result<T, PipelineError> {
        unreachable!("Detection errors don't have bubble context")
    }
}

impl<T> ErrorContext<T> for TranslationResult<T> {
    fn with_page_context(self, _page_index: usize) -> Result<T, PipelineError> {
        unreachable!("Translation errors should use bubble context")
    }

    fn with_bubble_context(
        self,
        page_index: usize,
        bubble_index: usize,
    ) -> Result<T, PipelineError> {
        self.map_err(|e| PipelineError::TranslationFailed {
            page_index,
            bubble_index,
            source: e,
        })
    }
}

impl<T> ErrorContext<T> for RenderingResult<T> {
    fn with_page_context(self, _page_index: usize) -> Result<T, PipelineError> {
        unreachable!("Rendering errors should use bubble context")
    }

    fn with_bubble_context(
        self,
        page_index: usize,
        bubble_index: usize,
    ) -> Result<T, PipelineError> {
        self.map_err(|e| PipelineError::RenderingFailed {
            page_index,
            bubble_index,
            source: e,
        })
    }
}
