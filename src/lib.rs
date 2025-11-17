// Library exports for the manga text processing workflow
//
// Reorganized module structure for better maintainability and performance

// Core modules
pub mod core;
pub mod middleware;
pub mod orchestration;
pub mod phases;
pub mod services;
pub mod utils;

// Re-export commonly used types and functions
pub use core::{
    config::Config,
    errors::{CacheError, ConfigError, DetectionError, PipelineError, RenderingError, TranslationError},
    types::{
        BackgroundType, CategorizedRegion, ImageData,
        Phase1Output, Phase2Output, Phase3Output, Phase4Output, PerformanceMetrics,
    },
};

pub use middleware::{ApiKeyPool, CircuitBreaker, CircuitBreakerConfig, CircuitState};

pub use orchestration::batch_orchestrator::{BatchOrchestrator};



pub use services::{ApiClient, CosmicTextRenderer, DetectionService, SegmentationService, TranslationCache};

pub use utils::{Metrics, crop_and_encode_png_async, load_image_from_memory_async};
