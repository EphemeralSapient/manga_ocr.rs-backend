// Trait abstractions for all services
//
// These traits enable:
// - Dependency injection
// - Mocking for tests
// - Multiple implementations (e.g., different translation APIs)
// - Decorator patterns (retry, circuit breaker, metrics)

use crate::types::{BubbleDetection, TextTranslation};
use anyhow::Result;
use async_trait::async_trait;
use image::{DynamicImage, Rgba, RgbaImage};

/// Trait for bubble detection services
///
/// Implementations can use different detection backends:
/// - ONNX Runtime (current implementation)
/// - TensorFlow
/// - PyTorch via tch-rs
/// - Cloud APIs (Google Vision, AWS Rekognition)
#[async_trait]
pub trait BubbleDetector: Send + Sync {
    /// Detect text bubbles in an image
    ///
    /// # Arguments
    /// * `img` - Input image to analyze
    /// * `page_index` - Page number (for logging/debugging)
    ///
    /// # Returns
    /// Vector of detected bubbles with bounding boxes and confidence scores
    async fn detect_bubbles(
        &self,
        img: &DynamicImage,
        page_index: usize,
    ) -> Result<Vec<BubbleDetection>>;

    /// Get the device type being used for detection
    ///
    /// # Returns
    /// String describing the device (e.g., "CUDA", "CPU", "TensorRT")
    fn device_type(&self) -> &str;
}

/// Trait for translation services
///
/// Implementations can use different translation providers:
/// - Gemini API (current implementation)
/// - OpenAI Vision API
/// - Azure Cognitive Services
/// - Local OCR + Translation models
#[async_trait]
pub trait Translator: Send + Sync {
    /// Translate text in a bubble image
    ///
    /// # Arguments
    /// * `bubble_bytes` - PNG bytes of the bubble image
    /// * `model` - Optional model name override
    ///
    /// # Returns
    /// Translation with text, font info, and background complexity flag
    async fn translate_bubble(
        &self,
        bubble_bytes: &[u8],
        model: Option<&str>,
    ) -> Result<TextTranslation>;

    /// Remove text from an image using AI inpainting
    ///
    /// # Arguments
    /// * `image_bytes` - PNG bytes of the bubble image with text
    /// * `model` - Optional model name override
    ///
    /// # Returns
    /// PNG bytes of the cleaned image (text removed)
    async fn remove_text_from_image(
        &self,
        image_bytes: &[u8],
        model: Option<&str>,
    ) -> Result<Vec<u8>>;
}

/// Trait for bubble rendering services
///
/// Implementations can use different rendering strategies:
/// - Current ab_glyph + imageproc implementation
/// - Cairo graphics
/// - Skia
/// - GPU-accelerated rendering
#[async_trait]
pub trait BubbleRenderer: Send + Sync {
    /// Render text on a simple background (solid color)
    ///
    /// # Arguments
    /// * `bubble_img` - Original bubble image (for color detection)
    /// * `detection` - Bubble detection with bounding box
    /// * `translation` - Translation with text and styling info
    ///
    /// # Returns
    /// Rendered bubble image ready for compositing
    async fn render_bubble_simple_background(
        &self,
        bubble_img: &RgbaImage,
        detection: &BubbleDetection,
        translation: &TextTranslation,
    ) -> Result<DynamicImage>;

    /// Render text on a complex background (AI-cleaned image)
    ///
    /// # Arguments
    /// * `cleaned_bubble_bytes` - PNG bytes from AI text removal
    /// * `detection` - Bubble detection with bounding box
    /// * `translation` - Translation with text and styling info
    ///
    /// # Returns
    /// Rendered bubble image with text on cleaned background
    async fn render_bubble_complex_background(
        &self,
        cleaned_bubble_bytes: &[u8],
        detection: &BubbleDetection,
        translation: &TextTranslation,
    ) -> Result<DynamicImage>;

    /// Composite a rendered bubble onto the page image
    ///
    /// # Arguments
    /// * `page_img` - Mutable page image to modify
    /// * `bubble_img` - Rendered bubble to composite
    /// * `detection` - Bubble detection with position info
    fn composite_bubble_onto_page(
        &self,
        page_img: &mut RgbaImage,
        bubble_img: &DynamicImage,
        detection: &BubbleDetection,
    );

    /// Analyze background complexity to determine if AI redrawing is needed
    ///
    /// Static method (no self) for flexibility in calling
    ///
    /// # Arguments
    /// * `img` - Bubble image to analyze
    ///
    /// # Returns
    /// Tuple of (is_complex, detected_color_if_simple)
    fn analyze_background_complexity(img: &RgbaImage) -> (bool, Option<Rgba<u8>>)
    where
        Self: Sized;
}

/// Trait for cache storage
///
/// Implementations can use different storage backends:
/// - File-based cache with gzip (current implementation)
/// - Redis
/// - Memcached
/// - In-memory (for testing)
/// - SQLite
/// - Distributed cache (etcd, Consul)
pub trait CacheStore: Send + Sync {
    /// Get a cached translation by checksum
    ///
    /// # Arguments
    /// * `checksum` - SHA1 checksum of the bubble image
    ///
    /// # Returns
    /// JSON string of TextTranslation if found, None otherwise
    fn get(&self, checksum: &str) -> Option<String>;

    /// Insert a translation into the cache
    ///
    /// Uses interior mutability to allow insertion with shared reference.
    /// Implementations should use Arc<RwLock<>> or similar for thread safety.
    ///
    /// # Arguments
    /// * `checksum` - SHA1 checksum of the bubble image
    /// * `translation` - JSON string of TextTranslation
    fn insert(&self, checksum: String, translation: String);

    /// Save the cache to persistent storage
    ///
    /// # Returns
    /// Result indicating success or failure
    fn save(&self) -> Result<()>;

    /// Get cache statistics
    ///
    /// # Returns
    /// Tuple of (entry_count, size_in_mb)
    fn stats(&self) -> (usize, f64);

    /// Compute SHA1 checksum of image bytes
    ///
    /// Static method for utility
    ///
    /// # Arguments
    /// * `image_bytes` - Raw image bytes
    ///
    /// # Returns
    /// Hex-encoded SHA1 checksum
    fn compute_checksum(image_bytes: &[u8]) -> String
    where
        Self: Sized;
}
