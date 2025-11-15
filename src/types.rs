use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Request configuration for batch processing
#[derive(Debug, Deserialize, Clone)]
pub struct BatchConfig {
    #[serde(default)]
    pub translation_model: Option<String>,
    #[serde(default)]
    pub image_gen_model: Option<String>,
    #[serde(default)]
    pub font_family: Option<String>,
}

/// Detection result for a single bubble
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BubbleDetection {
    pub bbox: [i32; 4],
    pub confidence: f32,
    pub page_index: usize,
    pub bubble_index: usize,
    pub text_regions: Vec<[i32; 4]>, // Text regions inside this bubble (label 1)
}

/// Translation response with new schema fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextTranslation {
    pub original_text: String,
    pub english_translation: String,
    #[serde(default = "default_font_family")]
    pub font_family: String,
    pub font_color: String,
    pub redraw_bg_required: bool,
    pub background_color: Option<String>,
}

fn default_font_family() -> String {
    "arial".to_string()
}

/// Image data for processing
#[derive(Clone)]
#[allow(dead_code)]
pub struct ImageData {
    pub index: usize,
    pub filename: String,
    pub image_bytes: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
}

/// Processed bubble with translation
#[derive(Clone)]
#[allow(dead_code)]
pub struct ProcessedBubble {
    pub detection: BubbleDetection,
    pub translation: Option<TextTranslation>,
    pub bubble_image: Arc<Vec<u8>>,
}

/// Batch processing result with analytics
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

/// Comprehensive batch analytics
#[derive(Debug, Clone, Serialize)]
pub struct BatchAnalytics {
    pub total_bubbles_detected: usize,
    pub total_bubbles_translated: usize,
    pub cache_hit_rate: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub api_calls_made: usize,
    pub api_calls_saved: usize,
    pub background_redraws: usize,
    pub simple_backgrounds: usize,
    pub average_detection_time_ms: f64,
    pub average_translation_time_ms: f64,
    pub average_rendering_time_ms: f64,
    pub model_used: ModelInfo,
}

/// Model information
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub translation_model: String,
    pub image_gen_model: String,
}

/// Individual page result
#[derive(Debug, Clone, Serialize)]
pub struct PageResult {
    pub index: usize,
    pub filename: String,
    pub success: bool,
    pub bubbles_detected: usize,
    pub bubbles_translated: usize,
    pub cache_hit: bool,
    pub processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_url: Option<String>,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub detection_time: Duration,
    pub translation_time: Duration,
    pub rendering_time: Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub api_calls_made: usize,
    pub api_calls_saved: usize,
    pub background_redraws: usize,
    pub simple_backgrounds: usize,
    pub total_bubbles: usize,
}

impl PerformanceMetrics {
    pub fn merge(&mut self, other: &PerformanceMetrics) {
        self.detection_time += other.detection_time;
        self.translation_time += other.translation_time;
        self.rendering_time += other.rendering_time;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.api_calls_made += other.api_calls_made;
        self.api_calls_saved += other.api_calls_saved;
        self.background_redraws += other.background_redraws;
        self.simple_backgrounds += other.simple_backgrounds;
        self.total_bubbles += other.total_bubbles;
    }
}
