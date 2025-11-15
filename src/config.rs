use crate::errors::ConfigError;
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
}

/// Translation configuration
#[derive(Debug, Clone)]
pub struct TranslationConfig {
    pub api_keys: Vec<String>,
    pub translation_model: String,
    pub image_gen_model: String,
    pub max_retries: u32,
}

/// Rendering configuration
#[derive(Debug, Clone)]
pub struct RenderingConfig {
    pub upscale_factor: u32,
    pub padding_ratio: f32,
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
    pub translation: TranslationConfig,
    pub rendering: RenderingConfig,
    pub cache: CacheConfig,
    pub rate_limit: RateLimitConfig,
    pub batch_size: usize,
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
            .unwrap_or_else(Vec::new);

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
            },
            translation: TranslationConfig {
                api_keys,
                translation_model: env::var("TRANSLATION_MODEL")
                    .unwrap_or_else(|_| "gemini-2.5-flash".to_string()),
                image_gen_model: env::var("IMAGE_GEN_MODEL")
                    .unwrap_or_else(|_| "gemini-2.0-flash-preview-image-generation".to_string()),
                max_retries: env::var("MAX_RETRIES")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3),
            },
            rendering: RenderingConfig {
                upscale_factor: env::var("UPSCALE_FACTOR")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3),
                padding_ratio: env::var("PADDING_RATIO")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.05),
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
            batch_size: env::var("BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(15),
        })
    }

    fn validate(&self) -> Result<(), ConfigError> {
        // Validate API keys
        if self.translation.api_keys.is_empty() {
            return Err(ConfigError::NoApiKeys);
        }

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

        // Validate batch size
        if self.batch_size == 0 {
            return Err(ConfigError::InvalidBatchSize(self.batch_size));
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
        &self.translation.api_keys
    }

    pub fn translation_model(&self) -> &str {
        &self.translation.translation_model
    }

    pub fn image_gen_model(&self) -> &str {
        &self.translation.image_gen_model
    }

    pub fn max_retries(&self) -> u32 {
        self.translation.max_retries
    }

    pub fn upscale_factor(&self) -> u32 {
        self.rendering.upscale_factor
    }

    pub fn cache_dir(&self) -> &str {
        &self.cache.cache_dir
    }

    pub fn inference_backend(&self) -> Option<&str> {
        self.detection.inference_backend.as_deref()
    }
}

// Note: No Default implementation because Config::new() can fail
// Users should explicitly call Config::new()? and handle errors
