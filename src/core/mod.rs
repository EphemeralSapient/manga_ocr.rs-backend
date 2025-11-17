pub mod config;
pub mod errors;
pub mod types;

// Re-export commonly used items for convenience
pub use config::Config;
pub use errors::{
    CacheError, ConfigError, DetectionError, PipelineError, RenderingError, TranslationError,
};
pub use types::{
    BackgroundType, CategorizedRegion, ImageData, Phase1Output, Phase2Output, Phase3Output,
    Phase4Output, PerformanceMetrics,
};
