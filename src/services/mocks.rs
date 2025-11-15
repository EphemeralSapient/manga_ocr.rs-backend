// Mock implementations for testing
//
// These mocks allow testing business logic without:
// - Real ONNX sessions
// - Real API calls
// - File I/O
// - Hardware dependencies

use crate::services::traits::{BubbleDetector, Translator, BubbleRenderer, CacheStore};
use crate::types::{BubbleDetection, TextTranslation};
use anyhow::Result;
use async_trait::async_trait;
use image::{DynamicImage, Rgba, RgbaImage};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock bubble detector that returns predefined detections
///
/// Usage:
/// ```rust,ignore
/// let detector = MockDetector::new(vec![
///     BubbleDetection { bbox: [10, 10, 100, 50], confidence: 0.95, ... }
/// ]);
/// ```
#[allow(dead_code)]
pub struct MockDetector {
    pub detections: Vec<BubbleDetection>,
    pub device_type: String,
}

#[allow(dead_code)]
impl MockDetector {
    pub fn new(detections: Vec<BubbleDetection>) -> Self {
        Self {
            detections,
            device_type: "MockDevice".to_string(),
        }
    }

    pub fn empty() -> Self {
        Self::new(vec![])
    }
}

#[async_trait]
impl BubbleDetector for MockDetector {
    async fn detect_bubbles(
        &self,
        _img: &DynamicImage,
        _page_index: usize,
    ) -> Result<Vec<BubbleDetection>> {
        Ok(self.detections.clone())
    }

    fn device_type(&self) -> &str {
        &self.device_type
    }
}

/// Mock translator that returns predefined translations
///
/// Usage:
/// ```rust,ignore
/// let translator = MockTranslator::new(vec![
///     TextTranslation {
///         original_text: "テスト".to_string(),
///         english_translation: "Test".to_string(),
///         ...
///     }
/// ]);
/// ```
#[allow(dead_code)]
pub struct MockTranslator {
    pub translations: Arc<Mutex<Vec<TextTranslation>>>,
    pub removed_images: Arc<Mutex<Vec<Vec<u8>>>>,
    pub call_count: Arc<Mutex<usize>>,
}

#[allow(dead_code)]
impl MockTranslator {
    pub fn new(translations: Vec<TextTranslation>) -> Self {
        Self {
            translations: Arc::new(Mutex::new(translations)),
            removed_images: Arc::new(Mutex::new(vec![])),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn single(translation: TextTranslation) -> Self {
        Self::new(vec![translation])
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

#[async_trait]
impl Translator for MockTranslator {
    async fn translate_bubble(
        &self,
        _bubble_bytes: &[u8],
        _model: Option<&str>,
    ) -> Result<TextTranslation> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let translations = self.translations.lock().unwrap();
        let index = (*count - 1) % translations.len();
        Ok(translations[index].clone())
    }

    async fn remove_text_from_image(
        &self,
        image_bytes: &[u8],
        _model: Option<&str>,
    ) -> Result<Vec<u8>> {
        // Return a simple white image for testing
        let removed_images = self.removed_images.lock().unwrap();
        if !removed_images.is_empty() {
            Ok(removed_images[0].clone())
        } else {
            // Default: return original bytes (no-op)
            Ok(image_bytes.to_vec())
        }
    }
}

/// Mock renderer that creates simple colored images
#[allow(dead_code)]
pub struct MockRenderer {
    pub background_complexity_result: Option<(bool, Option<Rgba<u8>>)>,
}

#[allow(dead_code)]
impl MockRenderer {
    pub fn new() -> Self {
        Self {
            background_complexity_result: None,
        }
    }

    pub fn with_complexity(mut self, is_complex: bool, color: Option<Rgba<u8>>) -> Self {
        self.background_complexity_result = Some((is_complex, color));
        self
    }
}

impl Default for MockRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BubbleRenderer for MockRenderer {
    async fn render_bubble_simple_background(
        &self,
        _bubble_img: &RgbaImage,
        detection: &BubbleDetection,
        _translation: &TextTranslation,
    ) -> Result<DynamicImage> {
        // Create a simple white image with the bubble dimensions
        let width = (detection.bbox[2] - detection.bbox[0]) as u32;
        let height = (detection.bbox[3] - detection.bbox[1]) as u32;
        let img = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));
        Ok(DynamicImage::ImageRgba8(img))
    }

    async fn render_bubble_complex_background(
        &self,
        _cleaned_bubble_bytes: &[u8],
        detection: &BubbleDetection,
        _translation: &TextTranslation,
    ) -> Result<DynamicImage> {
        // Create a simple gray image with the bubble dimensions
        let width = (detection.bbox[2] - detection.bbox[0]) as u32;
        let height = (detection.bbox[3] - detection.bbox[1]) as u32;
        let img = RgbaImage::from_pixel(width, height, Rgba([200, 200, 200, 255]));
        Ok(DynamicImage::ImageRgba8(img))
    }

    fn composite_bubble_onto_page(
        &self,
        page_img: &mut RgbaImage,
        bubble_img: &DynamicImage,
        detection: &BubbleDetection,
    ) {
        let x1 = detection.bbox[0].max(0) as u32;
        let y1 = detection.bbox[1].max(0) as u32;

        let bubble_rgba = bubble_img.to_rgba8();

        for (dx, dy, pixel) in bubble_rgba.enumerate_pixels() {
            let px = x1 + dx;
            let py = y1 + dy;

            if px < page_img.width() && py < page_img.height() {
                page_img.put_pixel(px, py, *pixel);
            }
        }
    }

    fn analyze_background_complexity(img: &RgbaImage) -> (bool, Option<Rgba<u8>>) {
        // Default: assume simple white background
        let width = img.width();
        let height = img.height();

        if width == 0 || height == 0 {
            return (true, None);
        }

        // Sample center pixel as "dominant color"
        let center_pixel = *img.get_pixel(width / 2, height / 2);
        (false, Some(center_pixel))
    }
}

/// Mock cache store using in-memory HashMap
///
/// Useful for testing cache hit/miss behavior without file I/O
#[allow(dead_code)]
pub struct MockCache {
    pub data: Arc<Mutex<HashMap<String, String>>>,
    pub save_called: Arc<Mutex<bool>>,
}

#[allow(dead_code)]
impl MockCache {
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
            save_called: Arc::new(Mutex::new(false)),
        }
    }

    pub fn with_data(data: HashMap<String, String>) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            save_called: Arc::new(Mutex::new(false)),
        }
    }

    pub fn was_save_called(&self) -> bool {
        *self.save_called.lock().unwrap()
    }

    pub fn entry_count(&self) -> usize {
        self.data.lock().unwrap().len()
    }
}

impl Default for MockCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheStore for MockCache {
    fn get(&self, checksum: &str) -> Option<String> {
        self.data.lock().unwrap().get(checksum).cloned()
    }

    fn insert(&self, checksum: String, translation: String) {
        self.data.lock().unwrap().insert(checksum, translation);
    }

    fn save(&self) -> Result<()> {
        let mut called = self.save_called.lock().unwrap();
        *called = true;
        Ok(())
    }

    fn stats(&self) -> (usize, f64) {
        let count = self.data.lock().unwrap().len();
        // Mock size: assume ~1KB per entry
        let size_mb = (count as f64) / 1024.0;
        (count, size_mb)
    }

    fn compute_checksum(image_bytes: &[u8]) -> String {
        use sha1::{Digest, Sha1};
        let mut hasher = Sha1::new();
        hasher.update(image_bytes);
        format!("{:x}", hasher.finalize())
    }
}

// Helper functions for creating test data

/// Create a default test translation
#[allow(dead_code)]
pub fn test_translation() -> TextTranslation {
    TextTranslation {
        original_text: "テスト".to_string(),
        english_translation: "Test".to_string(),
        font_family: "sans-serif-bold".to_string(),
        font_color: "0,0,0".to_string(),
        redraw_bg_required: false,
        background_color: Some("255,255,255".to_string()),
    }
}

/// Create a test detection
#[allow(dead_code)]
pub fn test_detection(page_index: usize, bubble_index: usize) -> BubbleDetection {
    BubbleDetection {
        bbox: [10, 10, 100, 50],
        confidence: 0.95,
        page_index,
        bubble_index,
        text_regions: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_detector() {
        let detections = vec![test_detection(0, 0)];
        let detector = MockDetector::new(detections.clone());

        let result = detector.detect_bubbles(&DynamicImage::new_rgba8(100, 100), 0).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].bbox, detections[0].bbox);
    }

    #[tokio::test]
    async fn test_mock_translator() {
        let translation = test_translation();
        let translator = MockTranslator::single(translation.clone());

        let result = translator.translate_bubble(b"fake_image", None).await.unwrap();
        assert_eq!(result.english_translation, translation.english_translation);
        assert_eq!(translator.call_count(), 1);
    }

    #[test]
    fn test_mock_cache() {
        let mut cache = MockCache::new();

        let checksum = "test123".to_string();
        let translation = "test_translation".to_string();

        // Test insert and get
        cache.insert(checksum.clone(), translation.clone());
        assert_eq!(cache.get(&checksum), Some(translation));

        // Test save
        assert!(!cache.was_save_called());
        cache.save().unwrap();
        assert!(cache.was_save_called());
    }
}
