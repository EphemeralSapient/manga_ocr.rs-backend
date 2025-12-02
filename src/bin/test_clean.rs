/// Test binary for text cleaning with mask debug output
/// Compares Rust text cleaner output with Python reference
///
/// Usage: cargo run --release --bin test_clean -- input.png [--output dir] [--threshold 0.3]

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, Rgba};
use std::path::Path;
use std::sync::Arc;

use manga_workflow::core::config::Config;
use manga_workflow::services::segmentation::SegmentationService;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image.png> [--output dir] [--threshold 0.3]", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let mut output_dir = "/.ram/tmp".to_string();
    let mut _threshold = 0.3f32;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_dir = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--threshold" | "-t" => {
                if i + 1 < args.len() {
                    _threshold = args[i + 1].parse().unwrap_or(0.3);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    // Ensure output dir exists
    std::fs::create_dir_all(&output_dir)?;

    // Load image
    println!("Loading: {}", input_path);
    let img = image::open(input_path).context("Failed to load image")?;
    println!("Image size: {}x{}", img.width(), img.height());

    // Initialize segmentation service
    println!("Initializing text cleaner...");
    let config = Arc::new(Config::new().expect("Failed to load config"));
    let seg_service = SegmentationService::new(config).await?;

    // Generate mask
    println!("Generating text mask...");
    let mask_bytes = seg_service.generate_mask(&img).await?;

    // Save mask as image
    let input_stem = Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let mask_path = format!("{}/{}_rust_mask.png", output_dir, input_stem);
    let mask_img = GrayImage::from_raw(img.width(), img.height(), mask_bytes.clone())
        .context("Failed to create mask image")?;
    mask_img.save(&mask_path)?;
    println!("Saved mask: {}", mask_path);

    // Apply mask to create cleaned image
    let mut rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if idx < mask_bytes.len() && mask_bytes[idx] > 0 {
                rgba.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
    }

    // Save cleaned image
    let cleaned_path = format!("{}/{}_rust_cleaned.png", output_dir, input_stem);
    DynamicImage::ImageRgba8(rgba).save(&cleaned_path)?;
    println!("Saved cleaned: {}", cleaned_path);

    // Stats
    let mask_coverage = mask_bytes.iter().filter(|&&v| v > 0).count() as f64 / mask_bytes.len() as f64 * 100.0;
    println!("\nStats:");
    println!("  Mask coverage: {:.1}%", mask_coverage);
    println!("  Mask pixels: {} / {}", mask_bytes.iter().filter(|&&v| v > 0).count(), mask_bytes.len());

    Ok(())
}
