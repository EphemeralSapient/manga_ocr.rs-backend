//! Quick OCR test binary - compare with Python implementation
//! Run with: cargo run --release --bin test_ocr -- <image_path>

use anyhow::Result;
use std::path::Path;
use tracing::info;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("manga_workflow::services::ocr=debug")
        .with_target(false)
        .init();

    // Get image path from args
    let args: Vec<String> = std::env::args().collect();
    let sample_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "test_sample.png".to_string()
    };

    if !Path::new(&sample_path).exists() {
        eprintln!("Image not found: {}", sample_path);
        std::process::exit(1);
    }

    info!("Loading image: {}", sample_path);
    let image = image::open(&sample_path)?;
    info!("Image dimensions: {}x{}", image.width(), image.height());

    // Initialize OCR service
    let models_dir = Path::new("models");
    info!("Initializing OCR service from: {}", models_dir.display());

    let ocr_service = manga_workflow::services::ocr::OcrService::new(models_dir)?;

    // Run OCR
    info!("\n=== Running Rust OCR ===");
    let (text, confidence) = ocr_service.recognize(&image)?;

    println!("\n=== Results ===");
    println!("Confidence: {:.2}", confidence);
    println!("Text:");
    if text.is_empty() {
        println!("  (empty)");
    } else {
        for (i, line) in text.lines().enumerate() {
            println!("  {}. {}", i + 1, line);
        }
    }

    Ok(())
}
