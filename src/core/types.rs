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
#[derive(Debug, Deserialize, Clone, Default)]
pub struct ProcessingConfig {
    // Empty for now - will contain processing options in the future
}

/// Detection result for a single region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionDetection {
    pub bbox: [i32; 4],
    pub label: u8, // 0=speech bubble, 1=inner text, 2=free text
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
    pub region_id: String, // Unique ID for this region
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRTranslation {
    pub original_text: String,
    pub translated_text: String,
}

/// Banana mode result for complex background
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BananaResult {
    pub region_id: String,
    pub translated_image_bytes: Vec<u8>, // Translated image from API
}

/// Phase 2 output: API call results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase2Output {
    pub page_index: usize,
    pub simple_bg_translations: Vec<(String, OCRTranslation)>, // (region_id, translation)
    pub complex_bg_bananas: Vec<BananaResult>, // Only if banana mode enabled
    pub complex_bg_translations: Vec<(String, OCRTranslation)>, // Only if banana mode disabled
}

/// Phase 3 output: Text removed images
#[derive(Debug, Clone)]
pub struct Phase3Output {
    #[allow(dead_code)]
    pub page_index: usize,
    pub cleaned_regions: Vec<(String, Vec<u8>)>, // (region_id, cleaned image bytes)
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
}

impl PerformanceMetrics {
    pub fn merge(&mut self, other: &PerformanceMetrics) {
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
    }
}
