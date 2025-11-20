use crate::core::errors::ConfigError;
use std::env;
use std::path::Path;
use tracing::Level;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
    pub log_level: Level,
}

/// Detection configuration
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub target_size: u32,
    pub inference_backend: Option<String>,
    pub detector_model_path: String,
    pub mask_model_path: String,
}

/// API configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub api_keys: Vec<String>,
    pub ocr_translation_model: String,
    pub banana_image_model: String,
    pub banana_mode_enabled: bool,
    pub max_retries: u32,
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// N: Number of images per batch (processed sequentially)
    pub batch_size_n: usize,
    /// M: Number of images per API call for simple backgrounds
    pub api_batch_size_m: usize,
    /// Maximum number of batches to process concurrently
    pub max_concurrent_batches: usize,
    /// Number of ONNX sessions per model (controls inference parallelism)
    pub onnx_pool_size: usize,
}

/// Background classification configuration
#[derive(Debug, Clone)]
pub struct BackgroundConfig {
    /// Threshold for simple background (0.0-1.0, default 0.6 = 60% white)
    pub simple_bg_white_threshold: f32,
}

/// Rendering configuration
#[derive(Debug, Clone)]
pub struct RenderingConfig {
    pub upscale_factor: u32,
    pub text_stroke_enabled: bool,
    pub text_stroke_width: i32,
    pub blur_free_text: bool,
    pub blur_radius: f32,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub cache_dir: String,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub max_requests: u32,
    pub window_seconds: u64,
}

/// Main application configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub detection: DetectionConfig,
    pub api: ApiConfig,
    pub batch: BatchConfig,
    pub background: BackgroundConfig,
    pub rendering: RenderingConfig,
    pub cache: CacheConfig,
    pub rate_limit: RateLimitConfig,
}

impl Config {
    pub fn new() -> Result<Self, ConfigError> {
        // Load .env file if it exists
        let _ = dotenvy::dotenv();

        let config = Self::load_from_env()?;
        config.validate()?;
        Ok(config)
    }

    fn load_from_env() -> Result<Self, ConfigError> {
        // Load API keys from environment (comma-separated) or use empty vec
        let api_keys = env::var("GEMINI_API_KEYS")
            .ok()
            .map(|keys| {
                keys.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        // Parse log level
        let log_level = env::var("LOG_LEVEL")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "trace" => Some(Level::TRACE),
                "debug" => Some(Level::DEBUG),
                "info" => Some(Level::INFO),
                "warn" | "warning" => Some(Level::WARN),
                "error" => Some(Level::ERROR),
                _ => None,
            })
            .unwrap_or(Level::INFO);

        Ok(Self {
            server: ServerConfig {
                port: env::var("SERVER_PORT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1420),
                host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                log_level,
            },
            detection: DetectionConfig {
                confidence_threshold: env::var("CONFIDENCE_THRESHOLD")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.3),
                iou_threshold: env::var("IOU_THRESHOLD")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.7),
                target_size: env::var("TARGET_SIZE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(640),
                inference_backend: env::var("INFERENCE_BACKEND")
                    .ok()
                    .map(|s| s.trim().to_uppercase())
                    .filter(|s| !s.is_empty()),
                detector_model_path: env::var("DETECTOR_MODEL_PATH")
                    .unwrap_or_else(|_| "models/detector.onnx".to_string()),
                mask_model_path: env::var("MASK_MODEL_PATH")
                    .unwrap_or_else(|_| "models/mask.onnx".to_string()),
            },
            api: ApiConfig {
                api_keys,
                ocr_translation_model: env::var("OCR_TRANSLATION_MODEL")
                    .unwrap_or_else(|_| "gemini-2.5-flash".to_string()),
                banana_image_model: env::var("BANANA_IMAGE_MODEL")
                    .unwrap_or_else(|_| "gemini-2.5-flash-image".to_string()),
                banana_mode_enabled: env::var("BANANA_MODE_ENABLED")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(false),
                max_retries: env::var("MAX_RETRIES")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3),
            },
            batch: BatchConfig {
                batch_size_n: env::var("BATCH_SIZE_N")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5),
                api_batch_size_m: env::var("API_BATCH_SIZE_M")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5),
                max_concurrent_batches: env::var("MAX_CONCURRENT_BATCHES")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(100),
                onnx_pool_size: env::var("ONNX_POOL_SIZE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        // Intelligent default: max(half the cores, 8)
                        // This balances performance with memory usage
                        let cores = num_cpus::get();
                        std::cmp::max(cores / 2, 8)
                    }),
            },
            background: BackgroundConfig {
                simple_bg_white_threshold: env::var("SIMPLE_BG_WHITE_THRESHOLD")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.6),
            },
            rendering: RenderingConfig {
                upscale_factor: env::var("UPSCALE_FACTOR")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3),
                text_stroke_enabled: env::var("TEXT_STROKE_ENABLED")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(false),
                text_stroke_width: env::var("TEXT_STROKE_WIDTH")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2),
                blur_free_text: env::var("BLUR_FREE_TEXT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(false),
                blur_radius: env::var("BLUR_RADIUS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10.0),
            },
            cache: CacheConfig {
                cache_dir: env::var("CACHE_DIR").unwrap_or_else(|_| ".cache".to_string()),
            },
            rate_limit: RateLimitConfig {
                enabled: env::var("RATE_LIMIT_ENABLED")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(false), // Disabled by default
                max_requests: env::var("RATE_LIMIT_MAX_REQUESTS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(100),
                window_seconds: env::var("RATE_LIMIT_WINDOW_SECONDS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(60),
            },
        })
    }

    fn validate(&self) -> Result<(), ConfigError> {
        // Note: API keys validation removed - keys can be provided via API request from extension
        // if not set in .env

        // Validate detection thresholds
        if !(0.0..=1.0).contains(&self.detection.confidence_threshold) {
            return Err(ConfigError::InvalidConfidenceThreshold(
                self.detection.confidence_threshold,
            ));
        }

        if !(0.0..=1.0).contains(&self.detection.iou_threshold) {
            return Err(ConfigError::InvalidIoUThreshold(
                self.detection.iou_threshold,
            ));
        }

        // Validate target size
        if !(320..=2048).contains(&self.detection.target_size) {
            return Err(ConfigError::InvalidDetectionConfig(format!(
                "target_size must be between 320 and 2048, got {}",
                self.detection.target_size
            )));
        }

        // Validate batch sizes
        if self.batch.batch_size_n == 0 {
            return Err(ConfigError::InvalidBatchSize(self.batch.batch_size_n));
        }
        if self.batch.api_batch_size_m == 0 {
            return Err(ConfigError::InvalidBatchSize(self.batch.api_batch_size_m));
        }
        if self.batch.max_concurrent_batches == 0 {
            return Err(ConfigError::InvalidBatchSize(self.batch.max_concurrent_batches));
        }

        // Validate background threshold
        if !(0.0..=1.0).contains(&self.background.simple_bg_white_threshold) {
            return Err(ConfigError::InvalidDetectionConfig(format!(
                "simple_bg_white_threshold must be between 0.0 and 1.0, got {}",
                self.background.simple_bg_white_threshold
            )));
        }

        // Validate cache directory parent exists
        let cache_path = Path::new(&self.cache.cache_dir);
        if let Some(parent) = cache_path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(ConfigError::InvalidCachePath(format!(
                    "Parent directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        // Validate upscale factor
        if !(1..=10).contains(&self.rendering.upscale_factor) {
            return Err(ConfigError::InvalidRenderingConfig(format!(
                "upscale_factor must be between 1 and 10, got {}",
                self.rendering.upscale_factor
            )));
        }

        // Validate rate limit config
        if self.rate_limit.enabled {
            if self.rate_limit.max_requests == 0 {
                return Err(ConfigError::InvalidDetectionConfig(
                    "rate_limit_max_requests must be > 0 when rate limiting is enabled".to_string(),
                ));
            }
            if self.rate_limit.window_seconds == 0 {
                return Err(ConfigError::InvalidDetectionConfig(
                    "rate_limit_window_seconds must be > 0 when rate limiting is enabled".to_string(),
                ));
            }
        }

        Ok(())
    }

    // Legacy accessors for backward compatibility during migration
    pub fn server_port(&self) -> u16 {
        self.server.port
    }

    pub fn server_host(&self) -> &str {
        &self.server.host
    }

    pub fn log_level(&self) -> Level {
        self.server.log_level
    }

    pub fn confidence_threshold(&self) -> f32 {
        self.detection.confidence_threshold
    }

    pub fn iou_threshold(&self) -> f32 {
        self.detection.iou_threshold
    }

    pub fn target_size(&self) -> u32 {
        self.detection.target_size
    }

    pub fn api_keys(&self) -> &[String] {
        &self.api.api_keys
    }

    pub fn ocr_translation_model(&self) -> &str {
        &self.api.ocr_translation_model
    }

    pub fn banana_image_model(&self) -> &str {
        &self.api.banana_image_model
    }

    pub fn banana_mode_enabled(&self) -> bool {
        self.api.banana_mode_enabled
    }

    pub fn max_retries(&self) -> u32 {
        self.api.max_retries
    }

    pub fn batch_size_n(&self) -> usize {
        self.batch.batch_size_n
    }

    pub fn api_batch_size_m(&self) -> usize {
        self.batch.api_batch_size_m
    }

    pub fn max_concurrent_batches(&self) -> usize {
        self.batch.max_concurrent_batches
    }

    pub fn onnx_pool_size(&self) -> usize {
        self.batch.onnx_pool_size
    }

    pub fn simple_bg_white_threshold(&self) -> f32 {
        self.background.simple_bg_white_threshold
    }

    pub fn upscale_factor(&self) -> u32 {
        self.rendering.upscale_factor
    }

    pub fn cache_dir(&self) -> &str {
        &self.cache.cache_dir
    }

    pub fn text_stroke_enabled(&self) -> bool {
        self.rendering.text_stroke_enabled
    }

    pub fn text_stroke_width(&self) -> i32 {
        self.rendering.text_stroke_width
    }

    pub fn blur_free_text(&self) -> bool {
        self.rendering.blur_free_text
    }

    pub fn blur_radius(&self) -> f32 {
        self.rendering.blur_radius
    }
}

// Note: No Default implementation because Config::new() can fail
// Users should explicitly call Config::new()? and handle errors
