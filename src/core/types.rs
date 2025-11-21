// New types for the redesigned workflow

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Forward declarations
use crate::core::config::Config;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Global configuration (used via method calls in API handlers - false positive warning)
    #[allow(dead_code)]
    pub config: Arc<Config>,
    pub orchestrator: Arc<crate::orchestration::batch_orchestrator::BatchOrchestrator>,
}

/// Request configuration for batch processing
/// All fields are optional - if not provided, defaults from .env are used
#[derive(Debug, Deserialize, Clone, Default)]
pub struct ProcessingConfig {
    /// Font source: "builtin" or "google"
    /// If not provided, defaults to "builtin"
    pub font_source: Option<String>,

    /// Font family for built-in fonts (e.g., "arial", "comic-sans", "anime-ace", "ms-yahei", "noto-sans-mono-cjk")
    /// Used when font_source is "builtin"
    pub font_family: Option<String>,

    /// Google Font family name (e.g., "Roboto", "Open Sans", "Noto Sans")
    /// Used when font_source is "google"
    pub google_font_family: Option<String>,

    /// Target language for translation (e.g., "English", "Spanish", "French", "Japanese", "Chinese")
    /// If not provided, defaults to "English"
    #[serde(rename = "targetLanguage")]
    pub target_language: Option<String>,

    /// OCR/Translation model override (e.g., "gemini-2.5-flash")
    /// If not provided, uses the model from environment config
    pub ocr_translation_model: Option<String>,

    /// Banana mode image model override (e.g., "gemini-2.5-flash-image")
    /// If not provided, uses the model from environment config
    pub banana_image_model: Option<String>,

    /// API keys override (array of Gemini API keys)
    /// If not provided, uses GEMINI_API_KEYS from .env
    #[serde(rename = "apiKeys")]
    pub api_keys: Option<Vec<String>>,

    /// Include free-standing text (label 2) in processing
    /// If not provided, defaults to false
    #[serde(rename = "includeFreeText")]
    pub include_free_text: Option<bool>,

    /// Enable banana mode for complex backgrounds
    /// If not provided, uses BANANA_MODE_ENABLED from .env
    #[serde(rename = "bananaMode")]
    pub banana_mode: Option<bool>,

    /// Enable text stroke/outline for better readability
    /// If not provided, uses TEXT_STROKE_ENABLED from .env
    #[serde(rename = "textStroke")]
    pub text_stroke: Option<bool>,

    /// Apply blur to free text backgrounds
    /// If not provided, uses BLUR_FREE_TEXT from .env
    #[serde(rename = "blurFreeTextBg")]
    pub blur_free_text_bg: Option<bool>,

    /// Enable translation cache
    /// If not provided, defaults to true
    #[serde(rename = "cache")]
    pub cache_enabled: Option<bool>,

    /// Include detailed metrics in response
    /// If not provided, defaults to true
    #[serde(rename = "metricsDetail")]
    pub metrics_detail: Option<bool>,

    /// Use segmentation mask for text removal
    /// If false, fills entire label 1 region with white instead of using mask
    /// If not provided, uses global mask mode setting
    #[serde(rename = "useMask")]
    pub use_mask: Option<bool>,

    /// Enable batch inference mode (merge multiple images into single tensor)
    /// Processes N images in one ONNX inference pass for better GPU utilization
    /// If not provided, defaults to false
    #[serde(rename = "mergeImg")]
    pub merge_img: Option<bool>,

    /// Maximum number of ONNX sessions per model (controls inference parallelism)
    /// Sessions are allocated dynamically on demand up to this limit
    /// If not provided, uses config default (max(cores/2, 8))
    /// Range: 1-32, each session uses ~82MB (42MB detection + 40MB segmentation)
    #[serde(rename = "sessionLimit")]
    pub session_limit: Option<usize>,

    /// Apply tighter bounds (erosion) to label 1 mask for more precise text removal
    /// When enabled, erodes the mask by 7x7 kernel for 4 iterations
    /// If not provided, defaults to true (enabled)
    #[serde(rename = "tighterBounds")]
    pub tighter_bounds: Option<bool>,
}

/// Detection label for region classification
/// OPTIMIZATION: Type-safe enum replaces magic numbers, prevents label mixups at compile time
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DetectionLabel {
    /// Speech bubble / outer region (label 0)
    SpeechBubble = 0,
    /// Inner text region within bubble (label 1)
    InnerText = 1,
    /// Free-standing text outside bubbles (label 2)
    FreeText = 2,
}

impl DetectionLabel {
    /// Convert from u8 to DetectionLabel
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::SpeechBubble),
            1 => Some(Self::InnerText),
            2 => Some(Self::FreeText),
            _ => None,
        }
    }

    /// Convert from i64 (ONNX output type) to DetectionLabel
    pub fn from_i64(value: i64) -> Option<Self> {
        Self::from_u8(value as u8)
    }

    /// Get the u8 value
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl From<DetectionLabel> for u8 {
    fn from(label: DetectionLabel) -> Self {
        label as u8
    }
}

/// Detection result for a single region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionDetection {
    pub bbox: [i32; 4],
    pub label: u8, // Keeping as u8 for now for backward compatibility, but prefer DetectionLabel
    pub confidence: f32,
}

/// Legacy bubble detection result (used by detection.rs)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BubbleDetection {
    pub bbox: [i32; 4],
    pub confidence: f32,
    pub page_index: usize,
    pub bubble_index: usize,
    pub text_regions: Vec<[i32; 4]>, // Text regions inside this bubble (label 1)
}

/// Legacy translation response (used by detection.rs)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct TextTranslation {
    pub original_text: String,
    pub english_translation: String,
    pub font_family: String,
    pub font_color: String,
    pub redraw_bg_required: bool,
    pub background_color: Option<String>,
}

/// Classification of a region background
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackgroundType {
    Simple,  // >=60% white in label 0,1 region
    Complex, // Otherwise, or label 2
}

/// Categorized region for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategorizedRegion {
    pub region_id: usize, // Unique sequential ID for this region (eliminates UUID heap allocations)
    pub page_index: usize,
    pub label: u8,
    pub bbox: [i32; 4],
    pub background_type: BackgroundType,
    pub label_1_regions: Vec<[i32; 4]>, // For label 0, contains its label 1 children
}

/// Phase 1 output: Categorized regions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase1Output {
    pub page_index: usize,
    pub filename: String,
    pub width: u32,
    pub height: u32,
    pub regions: Vec<CategorizedRegion>,
    pub segmentation_mask: Vec<u8>, // Flattened (h*w)
    pub validation_warnings: Vec<String>, // e.g., label 1 not in label 0
}

/// OCR/Translation result for simple background
/// Uses Arc<str> to avoid expensive string cloning on cache access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRTranslation {
    #[serde(serialize_with = "serialize_arc_str", deserialize_with = "deserialize_arc_str")]
    pub original_text: Arc<str>,
    #[serde(serialize_with = "serialize_arc_str", deserialize_with = "deserialize_arc_str")]
    pub translated_text: Arc<str>,
}

// Serde helpers for Arc<str>
fn serialize_arc_str<S>(arc_str: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(arc_str)
}

fn deserialize_arc_str<'de, D>(deserializer: D) -> Result<Arc<str>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    String::deserialize(deserializer).map(|s| Arc::from(s.as_str()))
}

/// Banana mode result for complex background
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BananaResult {
    pub region_id: usize,
    pub translated_image_bytes: Vec<u8>, // Translated image from API
}

/// Phase 2 output: API call results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase2Output {
    pub page_index: usize,
    pub simple_bg_translations: Vec<(usize, OCRTranslation)>, // (region_id, translation)
    pub complex_bg_bananas: Vec<BananaResult>, // Only if banana mode enabled
    pub complex_bg_translations: Vec<(usize, OCRTranslation)>, // Only if banana mode disabled
}

/// Phase 3 output: Text removed images
#[derive(Debug, Clone)]
pub struct Phase3Output {
    #[allow(dead_code)]
    pub page_index: usize,
    pub cleaned_regions: Vec<(usize, Vec<u8>)>, // (region_id, cleaned image bytes)
}

/// Phase 4 output: Final rendered image
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Phase4Output {
    pub page_index: usize,
    pub final_image_bytes: Vec<u8>,
}

/// Image data for processing
#[derive(Clone)]
pub struct ImageData {
    pub index: usize,
    pub filename: String,
    pub image_bytes: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    /// Pre-decoded image to avoid redundant decoding in each phase
    /// OPTIMIZATION: Image decoding is expensive (5-50ms per image).
    /// By decoding once and sharing via Arc, we eliminate 3-4 redundant
    /// decode operations per image, saving ~15-20% total processing time.
    pub decoded_image: Option<Arc<image::DynamicImage>>,
}

/// Batch of images to process together
#[derive(Clone)]
pub struct ImageBatch {
    pub batch_id: String,
    pub images: Vec<ImageData>,
}

/// Comprehensive batch analytics
#[derive(Debug, Clone, Serialize)]
pub struct BatchAnalytics {
    pub total_images: usize,
    pub total_regions: usize,
    pub simple_bg_count: usize,
    pub complex_bg_count: usize,
    pub label_0_count: usize,
    pub label_1_count: usize,
    pub label_2_count: usize,
    pub validation_warnings: usize,
    pub api_calls_simple: usize,
    pub api_calls_complex: usize,
    pub api_calls_banana: usize,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub phase1_time_ms: f64,
    pub phase2_time_ms: f64,
    pub phase3_time_ms: f64,
    pub phase4_time_ms: f64,
    pub total_time_ms: f64,
    pub inference_time_ms: f64, // Model inference time
    pub api_wait_time_ms: f64,  // Time waiting for API responses
}

/// Individual page result
#[derive(Debug, Clone, Serialize)]
pub struct PageResult {
    pub index: usize,
    pub filename: String,
    pub success: bool,
    pub processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_url: Option<String>,
}

/// Batch processing result
#[derive(Debug, Clone, Serialize)]
pub struct BatchResult {
    pub total: usize,
    pub successful: usize,
    pub failed: usize,
    pub processing_time_ms: f64,
    pub average_time_per_page_ms: f64,
    pub analytics: BatchAnalytics,
    pub results: Vec<PageResult>,
}

/// Performance metrics for tracking
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub phase1_time: Duration,
    pub phase2_time: Duration,
    pub phase3_time: Duration,
    pub phase4_time: Duration,
    pub inference_time: Duration,
    pub api_wait_time: Duration,
    pub total_regions: usize,
    pub simple_bg_count: usize,
    pub complex_bg_count: usize,
    pub validation_warnings: usize,
    pub api_calls_simple: usize,
    pub api_calls_complex: usize,
    pub api_calls_banana: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    // Label counts
    pub label_0_count: usize,
    pub label_1_count: usize,
    pub label_2_count: usize,
    // Token usage
    pub input_tokens: usize,
    pub output_tokens: usize,
}

impl PerformanceMetrics {
    /// Merge another PerformanceMetrics into this one
    /// OPTIMIZED: Pass by value since all fields are Copy
    pub fn merge(&mut self, other: PerformanceMetrics) {
        self.phase1_time += other.phase1_time;
        self.phase2_time += other.phase2_time;
        self.phase3_time += other.phase3_time;
        self.phase4_time += other.phase4_time;
        self.inference_time += other.inference_time;
        self.api_wait_time += other.api_wait_time;
        self.total_regions += other.total_regions;
        self.simple_bg_count += other.simple_bg_count;
        self.complex_bg_count += other.complex_bg_count;
        self.validation_warnings += other.validation_warnings;
        self.api_calls_simple += other.api_calls_simple;
        self.api_calls_complex += other.api_calls_complex;
        self.api_calls_banana += other.api_calls_banana;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        // Label counts
        self.label_0_count += other.label_0_count;
        self.label_1_count += other.label_1_count;
        self.label_2_count += other.label_2_count;
        // Token usage
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}
