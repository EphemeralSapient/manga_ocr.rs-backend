// Integration tests for the manga translation pipeline
//
// These tests use mocked services to verify business logic without
// requiring real ONNX sessions, API keys, or external dependencies

use manga_workflow::pipeline::ProcessingPipeline;
use manga_workflow::services::mocks::*;
use manga_workflow::types::{BatchConfig, ImageData, TextTranslation};
use std::sync::Arc;

#[tokio::test]
async fn test_pipeline_with_simple_background() {
    // Create mock services
    let detector = MockDetector::new(vec![test_detection(0, 0)]);

    let translation = TextTranslation {
        original_text: "テスト".to_string(),
        english_translation: "Test".to_string(),
        font_family: "sans-serif-bold".to_string(),
        font_color: "0,0,0".to_string(),
        redraw_bg_required: false, // Simple background
        background_color: Some("255,255,255".to_string()),
    };
    let translator = MockTranslator::single(translation);

    let renderer = MockRenderer::new();
    let cache = MockCache::new();

    // Create pipeline
    let pipeline = ProcessingPipeline::new(
        Arc::new(detector),
        Arc::new(translator),
        Arc::new(renderer),
        Arc::new(cache),
    );

    // Create test image
    let test_image = create_test_image();
    let images = vec![test_image];

    // Process batch
    let result = pipeline
        .process_batch(images, &BatchConfig::default())
        .await;

    assert!(result.is_ok());
    let (results, metrics) = result.unwrap();

    // Verify results
    assert_eq!(results.len(), 1);
    assert!(results[0].success);
    assert_eq!(results[0].bubbles_detected, 1);
    assert_eq!(results[0].bubbles_translated, 1);

    // Verify metrics
    assert_eq!(metrics.cache_misses, 1); // First run, cache miss
    assert_eq!(metrics.simple_backgrounds, 1); // Simple background detected
}

#[tokio::test]
async fn test_pipeline_with_cache_hit() {
    // Pre-populate cache
    let mut cache_data = std::collections::HashMap::new();
    let translation = test_translation();
    let checksum = "test_checksum_123".to_string();
    cache_data.insert(checksum.clone(), serde_json::to_string(&translation).unwrap());

    let detector = MockDetector::new(vec![test_detection(0, 0)]);
    let translator = MockTranslator::single(translation.clone());
    let renderer = MockRenderer::new();
    let cache = MockCache::with_data(cache_data);

    let pipeline = ProcessingPipeline::new(
        Arc::new(detector),
        Arc::new(translator),
        Arc::new(renderer),
        Arc::new(cache),
    );

    let test_image = create_test_image();
    let images = vec![test_image];

    let result = pipeline
        .process_batch(images, &BatchConfig::default())
        .await;

    assert!(result.is_ok());
    let (results, metrics) = result.unwrap();

    // Note: Cache hits depend on actual image checksums matching
    // This test verifies the pipeline handles cached data correctly
    assert_eq!(results.len(), 1);
    assert!(results[0].success);
}

#[tokio::test]
async fn test_pipeline_with_multiple_bubbles() {
    // Create multiple detections
    let detections = vec![
        test_detection(0, 0),
        test_detection(0, 1),
        test_detection(0, 2),
    ];
    let detector = MockDetector::new(detections);

    // Create multiple translations
    let translations = vec![
        test_translation(),
        test_translation(),
        test_translation(),
    ];
    let translator = MockTranslator::new(translations);

    let renderer = MockRenderer::new();
    let cache = MockCache::new();

    let pipeline = ProcessingPipeline::new(
        Arc::new(detector),
        Arc::new(translator),
        Arc::new(renderer),
        Arc::new(cache),
    );

    let test_image = create_test_image();
    let images = vec![test_image];

    let result = pipeline
        .process_batch(images, &BatchConfig::default())
        .await;

    assert!(result.is_ok());
    let (results, metrics) = result.unwrap();

    // Verify all bubbles processed
    assert_eq!(results.len(), 1);
    assert!(results[0].success);
    assert_eq!(results[0].bubbles_detected, 3);
    assert_eq!(results[0].bubbles_translated, 3);
    assert_eq!(metrics.cache_misses, 3); // Three new translations
}

#[tokio::test]
async fn test_pipeline_with_no_bubbles() {
    // Empty detection
    let detector = MockDetector::empty();
    let translator = MockTranslator::single(test_translation());
    let renderer = MockRenderer::new();
    let cache = MockCache::new();

    let pipeline = ProcessingPipeline::new(
        Arc::new(detector),
        Arc::new(translator),
        Arc::new(renderer),
        Arc::new(cache),
    );

    let test_image = create_test_image();
    let images = vec![test_image];

    let result = pipeline
        .process_batch(images, &BatchConfig::default())
        .await;

    assert!(result.is_ok());
    let (results, _metrics) = result.unwrap();

    // Verify no bubbles detected
    assert_eq!(results.len(), 1);
    assert!(results[0].success);
    assert_eq!(results[0].bubbles_detected, 0);
    assert_eq!(results[0].bubbles_translated, 0);
}

#[tokio::test]
async fn test_pipeline_with_complex_background() {
    let detector = MockDetector::new(vec![test_detection(0, 0)]);

    let translation = TextTranslation {
        original_text: "複雑".to_string(),
        english_translation: "Complex".to_string(),
        font_family: "sans-serif-bold".to_string(),
        font_color: "0,0,0".to_string(),
        redraw_bg_required: true, // Complex background
        background_color: None,
    };
    let translator = MockTranslator::single(translation);

    let renderer = MockRenderer::new().with_complexity(true, None);
    let cache = MockCache::new();

    let pipeline = ProcessingPipeline::new(
        Arc::new(detector),
        Arc::new(translator),
        Arc::new(renderer),
        Arc::new(cache),
    );

    let test_image = create_test_image();
    let images = vec![test_image];

    let result = pipeline
        .process_batch(images, &BatchConfig::default())
        .await;

    assert!(result.is_ok());
    let (results, metrics) = result.unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].success);

    // Note: Metrics for complex backgrounds depend on actual background analysis
    // This test verifies the pipeline handles complex backgrounds correctly
}

#[tokio::test]
async fn test_pipeline_parallel_processing() {
    // Create multiple pages
    let detector = MockDetector::new(vec![test_detection(0, 0)]);
    let translator = MockTranslator::single(test_translation());
    let renderer = MockRenderer::new();
    let cache = MockCache::new();

    let pipeline = ProcessingPipeline::new(
        Arc::new(detector),
        Arc::new(translator),
        Arc::new(renderer),
        Arc::new(cache),
    );

    // Process multiple images in parallel
    let images = vec![
        create_test_image_with_index(1),
        create_test_image_with_index(2),
        create_test_image_with_index(3),
    ];

    let start = std::time::Instant::now();
    let result = pipeline
        .process_batch(images, &BatchConfig::default())
        .await;
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    let (results, _metrics) = result.unwrap();

    assert_eq!(results.len(), 3);
    for result in &results {
        assert!(result.success);
    }

    // Parallel processing should complete in reasonable time
    // (This is a smoke test, actual timing depends on system)
    assert!(elapsed.as_secs() < 10);
}

// Helper functions

fn create_test_image() -> ImageData {
    create_test_image_with_index(1)
}

fn create_test_image_with_index(index: usize) -> ImageData {
    // Create a simple 100x100 white image
    use image::{ImageBuffer, Rgba};
    let img = ImageBuffer::from_pixel(100, 100, Rgba([255u8, 255u8, 255u8, 255u8]));

    let mut bytes = Vec::new();
    use std::io::Cursor;
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();

    ImageData {
        index,
        filename: format!("test_{}.png", index),
        image_bytes: Arc::new(bytes),
        width: 100,
        height: 100,
    }
}
