/// CLI module for phased testing workflow
///
/// This module provides independent testing of each pipeline phase:
/// - Phase 1: Detection (outputs cropped bubbles with bbox visualization)
/// - Phase 2: Translation (outputs API response JSON)
/// - Phase 3: Rendering (outputs final rendered bubble)

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use image::Rgba;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
// Note: ab_glyph import removed - now using cosmic-text for all text rendering
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, debug};

use crate::config::Config;
use crate::detection::DetectionService;
use crate::rendering::RenderingService;
use crate::translation::TranslationService;
use crate::types::{BubbleDetection, TextTranslation};

#[derive(Parser)]
#[command(name = "manga_workflow")]
#[command(about = "Manga translation workflow with phased testing", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the HTTP server (default mode)
    Server,

    /// Interactive mode - guided workflow with prompts
    #[command(name = "interactive")]
    Interactive,

    /// Phase 1: Detect speech bubbles and visualize bounding boxes
    #[command(name = "phase1")]
    Phase1 {
        /// Input manga page image path
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for cropped bubbles (default: phase_1_output)
        #[arg(short, long, default_value = "phase_1_output")]
        output: PathBuf,

        /// Also save full page with visualized bboxes
        #[arg(long, default_value = "true")]
        visualize: bool,

        /// Also detect and save text_free regions (text outside bubbles) for testing
        #[arg(long, default_value = "false")]
        include_text_free: bool,
    },

    /// Phase 2: Translate a speech bubble using API
    #[command(name = "phase2")]
    Phase2 {
        /// Input speech bubble image path (Label 0 - full bubble)
        #[arg(short, long)]
        input: PathBuf,

        /// Path to detections.json from Phase 1 (to extract Label 1 text region)
        #[arg(short, long)]
        detections_json: PathBuf,

        /// Output JSON file path (default: phase_2_output.json)
        #[arg(short, long, default_value = "phase_2_output.json")]
        output: PathBuf,

        /// Translation model override (default: gemini-2.0-flash-exp)
        #[arg(long)]
        translation_model: Option<String>,

        /// Font family for rendering (default: arial). Options: arial, anime-ace, anime-ace-3, comic-sans
        #[arg(long)]
        font_family: Option<String>,
    },

    /// Phase 3: Render translated text onto bubble
    #[command(name = "phase3")]
    Phase3 {
        /// Input speech bubble image path
        #[arg(short, long)]
        input: PathBuf,

        /// Translation JSON from phase 2
        #[arg(short, long)]
        api_response: PathBuf,

        /// Detections JSON from phase 1 (contains text_regions for cleaning)
        #[arg(short, long)]
        detections_json: PathBuf,

        /// Output image path (default: phase_3_output.png)
        #[arg(short, long, default_value = "phase_3_output.png")]
        output: PathBuf,

        /// Image generation model override (for complex backgrounds)
        #[arg(long)]
        image_gen_model: Option<String>,

        /// Font family for rendering (default: arial). Options: arial, anime-ace, anime-ace-3, comic-sans
        #[arg(long)]
        font_family: Option<String>,

        /// Save polygon visualization for debugging
        #[arg(long)]
        debug_polygon: bool,

        /// Save cleaned speech bubble (text removed, no translation rendered)
        #[arg(long)]
        speech: Option<PathBuf>,

        /// Insertion mode: Use pre-cleaned bubble (from --speech) as input to debug text rendering
        #[arg(long)]
        insertion: bool,
    },
}

/// Execute Phase 1: Detection with bbox visualization
pub async fn execute_phase1(
    input: PathBuf,
    output: PathBuf,
    visualize: bool,
    include_text_free: bool,
    config: Arc<Config>,
) -> Result<()> {
    info!("{}", "=".repeat(70));
    info!("PHASE 1: BUBBLE DETECTION");
    info!("{}", "=".repeat(70));
    info!("Input: {}", input.display());
    info!("Output: {}/", output.display());
    info!("");

    // Create output directory
    std::fs::create_dir_all(&output)
        .context("Failed to create output directory")?;

    // Load image
    info!("📖 Loading image...");
    let img = image::open(&input)
        .context("Failed to load input image")?;
    info!("✓ Image loaded: {}x{}", img.width(), img.height());
    info!("");

    // Initialize detection service
    info!("🔧 Initializing detection service...");
    let detector = DetectionService::new(config.clone()).await?;
    info!("✓ Detection service ready ({})", detector.device_type());
    info!("");

    // Run detection
    info!("🔍 Running bubble detection...");
    let detections = detector.detect_bubbles(&img, 0).await?;
    info!("✓ Detected {} bubbles", detections.len());
    info!("");

    if detections.is_empty() {
        info!("⚠️  No bubbles detected in image");
        return Ok(());
    }

    // Save cropped bubbles with bbox outlines
    info!("💾 Saving cropped bubbles...");
    for (idx, detection) in detections.iter().enumerate() {
        let x1 = detection.bbox[0].max(0) as u32;
        let y1 = detection.bbox[1].max(0) as u32;
        let x2 = detection.bbox[2].max(0).min(img.width() as i32) as u32;
        let y2 = detection.bbox[3].max(0).min(img.height() as i32) as u32;

        let bubble_img = img.crop_imm(x1, y1, x2 - x1, y2 - y1);

        let output_path = output.join(format!("bubble_{:02}.png", idx + 1));
        bubble_img.save(&output_path)
            .context(format!("Failed to save bubble {}", idx + 1))?;

        info!("  [{}] bbox=[{}, {}, {}, {}], conf={:.3}, size={}x{} → {}",
            idx + 1,
            detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3],
            detection.confidence,
            x2 - x1, y2 - y1,
            output_path.file_name().unwrap().to_str().unwrap()
        );
    }
    info!("");

    // Optionally detect and save text_free regions (text outside bubbles) for testing
    if include_text_free {
        info!("🔍 Detecting text_free regions (text outside bubbles) for testing...");

        let text_free_dir = output.join("text_free_output");
        std::fs::create_dir_all(&text_free_dir)?;

        // Run detection with label 2 (text_free)
        let text_free_detections = detector.detect_with_label(&img, 0, 2).await?;
        info!("✓ Detected {} text_free regions", text_free_detections.len());

        if text_free_detections.is_empty() {
            info!("  No text_free regions found");
        } else {
            info!("💾 Saving text_free regions...");
            for (idx, detection) in text_free_detections.iter().enumerate() {
                let x1 = detection.bbox[0].max(0) as u32;
                let y1 = detection.bbox[1].max(0) as u32;
                let x2 = detection.bbox[2].min(img.width() as i32) as u32;
                let y2 = detection.bbox[3].min(img.height() as i32) as u32;

                let text_free_img = img.crop_imm(x1, y1, x2 - x1, y2 - y1);

                let output_path = text_free_dir.join(format!("text_free_{:02}.png", idx + 1));
                text_free_img.save(&output_path)
                    .context(format!("Failed to save text_free {}", idx + 1))?;

                info!("  [{}] bbox=[{}, {}, {}, {}], conf={:.3}, size={}x{} → {}",
                    idx + 1,
                    detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3],
                    detection.confidence,
                    x2 - x1, y2 - y1,
                    output_path.file_name().unwrap().to_str().unwrap()
                );
            }
        }
        info!("");
    }

    // Visualize bboxes on full page
    if visualize {
        info!("🎨 Creating visualization with bounding boxes...");
        let mut viz_img = img.to_rgba8();

        for (idx, detection) in detections.iter().enumerate() {
            let x1 = detection.bbox[0].max(0);
            let y1 = detection.bbox[1].max(0);
            let x2 = detection.bbox[2].min(viz_img.width() as i32);
            let y2 = detection.bbox[3].min(viz_img.height() as i32);

            let width = (x2 - x1).max(0) as u32;
            let height = (y2 - y1).max(0) as u32;

            if width > 0 && height > 0 {
                // Draw bounding box (green)
                let rect = Rect::at(x1, y1).of_size(width, height);
                draw_hollow_rect_mut(&mut viz_img, rect, Rgba([0, 255, 0, 255]));

                // Draw thicker border (3px)
                if x1 > 0 && y1 > 0 && width > 2 && height > 2 {
                    let rect_inner = Rect::at(x1 + 1, y1 + 1).of_size(width - 2, height - 2);
                    draw_hollow_rect_mut(&mut viz_img, rect_inner, Rgba([0, 255, 0, 255]));
                    if width > 4 && height > 4 {
                        let rect_inner2 = Rect::at(x1 + 2, y1 + 2).of_size(width - 4, height - 4);
                        draw_hollow_rect_mut(&mut viz_img, rect_inner2, Rgba([0, 255, 0, 255]));
                    }
                }

                // Draw label with bubble number (only if there's space)
                if height > 30 {
                    let _label = format!("#{}", idx + 1);
                    // Note: For simplicity, we'll skip text drawing if font isn't available
                    // In production, you'd load a font properly
                    debug!("Bubble {} labeled at ({}, {})", idx + 1, x1, y1);
                }

                // Draw text regions (Label 1) in red
                for (text_idx, text_region) in detection.text_regions.iter().enumerate() {
                    let tx1 = text_region[0].max(0);
                    let ty1 = text_region[1].max(0);
                    let tx2 = text_region[2].min(viz_img.width() as i32);
                    let ty2 = text_region[3].min(viz_img.height() as i32);

                    let text_width = (tx2 - tx1).max(0) as u32;
                    let text_height = (ty2 - ty1).max(0) as u32;

                    if text_width > 0 && text_height > 0 {
                        // Draw text region box (red)
                        let text_rect = Rect::at(tx1, ty1).of_size(text_width, text_height);
                        draw_hollow_rect_mut(&mut viz_img, text_rect, Rgba([255, 0, 0, 255]));

                        // Draw thicker border (2px) for text regions
                        if tx1 > 0 && ty1 > 0 && text_width > 2 && text_height > 2 {
                            let text_rect_inner = Rect::at(tx1 + 1, ty1 + 1).of_size(text_width - 2, text_height - 2);
                            draw_hollow_rect_mut(&mut viz_img, text_rect_inner, Rgba([255, 0, 0, 255]));
                        }

                        debug!("Text region {} for bubble {} at ({}, {})", text_idx + 1, idx + 1, tx1, ty1);
                    }
                }
            }
        }

        let viz_path = output.join("visualization.png");
        viz_img.save(&viz_path)
            .context("Failed to save visualization")?;
        info!("✓ Visualization saved: {}", viz_path.display());
        info!("");
    }

    // Save detection data to JSON for Phase 3
    info!("💾 Saving detection data to JSON...");

    #[derive(serde::Serialize)]
    struct Phase1Detection {
        bbox: [i32; 4],
        confidence: f32,
        text_regions: Vec<[i32; 4]>,
    }

    #[derive(serde::Serialize)]
    struct Phase1Output {
        detections: Vec<Phase1Detection>,
    }

    let phase1_output = Phase1Output {
        detections: detections.iter().map(|d| Phase1Detection {
            bbox: d.bbox,
            confidence: d.confidence,
            text_regions: d.text_regions.clone(),
        }).collect(),
    };

    let json_path = output.join("detections.json");
    let json = serde_json::to_string_pretty(&phase1_output)
        .context("Failed to serialize detections")?;
    std::fs::write(&json_path, json)
        .context("Failed to write detections JSON")?;
    info!("✓ Detection data saved to: {}", json_path.display());
    info!("");

    // Print summary
    info!("{}", "=".repeat(70));
    info!("PHASE 1 SUMMARY");
    info!("{}", "=".repeat(70));
    info!("Total bubbles detected: {}", detections.len());
    info!("Cropped images saved to: {}/", output.display());
    info!("Detection data: {}/detections.json", output.display());
    if visualize {
        info!("Visualization: {}/visualization.png", output.display());
    }
    info!("");
    info!("Detection details:");
    for (idx, det) in detections.iter().enumerate() {
        let width = det.bbox[2] - det.bbox[0];
        let height = det.bbox[3] - det.bbox[1];
        info!("  Bubble {}: {}x{}px, confidence={:.3}, text_regions={}",
            idx + 1, width, height, det.confidence, det.text_regions.len());
    }
    info!("{}", "=".repeat(70));

    Ok(())
}

/// Execute Phase 2: Translation API call
pub async fn execute_phase2(
    input: PathBuf,
    detections_json: PathBuf,
    output: PathBuf,
    translation_model: Option<String>,
    _font_family: Option<String>,
    config: Arc<Config>,
) -> Result<()> {
    info!("{}", "=".repeat(70));
    info!("PHASE 2: TRANSLATION");
    info!("{}", "=".repeat(70));
    info!("Input: {}", input.display());
    info!("Output: {}", output.display());
    if let Some(ref model) = translation_model {
        info!("Translation model: {}", model);
    }
    info!("");

    // Load bubble image (Label 0)
    info!("📖 Loading bubble image (Label 0)...");
    let bubble_img = image::open(&input)
        .context("Failed to load input image")?;
    info!("✓ Bubble loaded: {}x{}", bubble_img.width(), bubble_img.height());
    info!("");

    // Load detections.json to get Label 1 text region
    info!("📖 Loading detections.json...");
    let detections_file = std::fs::File::open(&detections_json)
        .context("Failed to open detections.json")?;
    let detections: serde_json::Value = serde_json::from_reader(detections_file)
        .context("Failed to parse detections.json")?;

    // Extract bubble number from filename (e.g., "bubble_01.png" → 1, then -1 → 0 for array index)
    let bubble_num_1indexed: usize = input
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("bubble_"))
        .and_then(|s| s.parse().ok())
        .context("Failed to parse bubble number from filename")?;

    if bubble_num_1indexed == 0 {
        return Err(anyhow::anyhow!("Invalid bubble number: filenames start from bubble_01"));
    }

    let bubble_num = bubble_num_1indexed - 1; // Convert to 0-indexed array position
    info!("✓ Processing bubble_{:02} → detection index {}", bubble_num_1indexed, bubble_num);

    // Get Label 1 text regions for this bubble
    let detections_array = detections["detections"]
        .as_array()
        .context("detections.json missing 'detections' array")?;
    let bubble_detection = detections_array
        .get(bubble_num)
        .context(format!("Bubble {} not found in detections", bubble_num))?;

    // Get bubble bbox (Label 0) to calculate offset
    let bubble_bbox_array = bubble_detection["bbox"]
        .as_array()
        .context("Missing bbox in bubble detection")?;

    let bubble_x1 = bubble_bbox_array[0].as_i64().context("Invalid bubble x1")? as i32;
    let bubble_y1 = bubble_bbox_array[1].as_i64().context("Invalid bubble y1")? as i32;

    let text_regions = bubble_detection["text_regions"]
        .as_array()
        .context("Missing text_regions in detection")?;

    if text_regions.is_empty() {
        return Err(anyhow::anyhow!("No text regions found in bubble {}", bubble_num));
    }

    // Use first text region (Label 1) bbox for cropping
    // text_regions is an array of [x1, y1, x2, y2] arrays directly in ORIGINAL IMAGE coordinates
    let first_text_region = &text_regions[0];
    let label1_bbox_orig = first_text_region.as_array()
        .context("text_region is not an array")?;

    // Label 1 bbox in original image coordinates
    let label1_x1_orig = label1_bbox_orig[0].as_i64().context("Invalid x1")? as i32;
    let label1_y1_orig = label1_bbox_orig[1].as_i64().context("Invalid y1")? as i32;
    let label1_x2_orig = label1_bbox_orig[2].as_i64().context("Invalid x2")? as i32;
    let label1_y2_orig = label1_bbox_orig[3].as_i64().context("Invalid y2")? as i32;

    // Convert to bubble-relative coordinates (Label 1 relative to Label 0)
    let label1_x1 = (label1_x1_orig - bubble_x1).max(0) as u32;
    let label1_y1 = (label1_y1_orig - bubble_y1).max(0) as u32;
    let label1_x2 = (label1_x2_orig - bubble_x1).max(0) as u32;
    let label1_y2 = (label1_y2_orig - bubble_y1).max(0) as u32;

    let label1_width = label1_x2 - label1_x1;
    let label1_height = label1_y2 - label1_y1;

    info!("✓ Label 1 text region: [{}, {}, {}, {}] ({}x{})",
        label1_x1, label1_y1, label1_x2, label1_y2, label1_width, label1_height);
    info!("  (Label 1 will be used to CONSTRAIN polygon, not for API input)");
    info!("");

    // Convert full bubble (Label 0) to bytes - send FULL context to Gemini
    let mut bubble_bytes = Vec::new();
    bubble_img.write_to(
        &mut std::io::Cursor::new(&mut bubble_bytes),
        image::ImageFormat::Png
    )?;
    info!("📦 Full bubble (Label 0) encoded: {} bytes", bubble_bytes.len());
    info!("  Sending FULL bubble to API for better context");
    info!("");

    // Initialize translation service
    info!("🔧 Initializing translation service...");
    let translator = TranslationService::new(config.clone());
    info!("✓ Translation service ready ({} API keys)", config.api_keys().len());
    info!("");

    // Call translation API with full bubble (Label 0) for better context
    info!("🌐 Calling translation API with full bubble (Label 0)...");
    let start = std::time::Instant::now();
    let translation = translator
        .translate_bubble(&bubble_bytes, translation_model.as_deref())
        .await
        .context("Translation API call failed")?;
    let elapsed = start.elapsed();
    info!("✓ Translation received in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    info!("  (Polygon coordinates are in Label 0 space)");

    // Display translation details
    info!("{}", "=".repeat(70));
    info!("TRANSLATION RESULT");
    info!("{}", "=".repeat(70));
    info!("Original text: {}", translation.original_text);
    info!("English translation: {}", translation.english_translation);
    info!("Font family: {}", translation.font_family);
    info!("Font color: {}", translation.font_color);
    info!("Redraw background required: {}", translation.redraw_bg_required);
    if let Some(ref bg_color) = translation.background_color {
        info!("Background color: {}", bg_color);
    }
    info!("{}", "=".repeat(70));
    info!("");

    // Analyze background complexity locally on full bubble
    info!("🔬 Performing local background analysis...");
    let bubble_rgba = bubble_img.to_rgba8();
    let (is_complex_local, detected_color) =
        RenderingService::analyze_background_complexity(&bubble_rgba);

    info!("Local analysis results:");
    info!("  Complex background: {}", is_complex_local);
    if let Some(color) = detected_color {
        info!("  Detected color: RGB({}, {}, {})", color[0], color[1], color[2]);
    }
    info!("");

    // Determine final rendering strategy
    let needs_ai_redraw = translation.redraw_bg_required && is_complex_local;
    info!("Rendering strategy:");
    info!("  API says redraw: {}", translation.redraw_bg_required);
    info!("  Local says complex: {}", is_complex_local);
    info!("  Final decision: {}", if needs_ai_redraw { "Complex (AI)" } else { "Simple" });
    info!("");

    // Create detailed output JSON
    #[derive(serde::Serialize)]
    struct Phase2Output {
        translation: TextTranslation,
        metadata: Phase2Metadata,
    }

    #[derive(serde::Serialize)]
    struct Phase2Metadata {
        input_file: String,
        label0_size: (u32, u32),         // Full bubble size (Label 0)
        label1_bbox: [u32; 4],           // Text region bbox [x1, y1, x2, y2] in Label 0 coordinates
        label1_size: (u32, u32),         // Text region size (Label 1)
        processing_time_ms: f64,
        api_keys_available: usize,
        local_background_analysis: LocalAnalysis,
        recommended_rendering: String,
    }

    #[derive(serde::Serialize)]
    struct LocalAnalysis {
        is_complex: bool,
        detected_color: Option<[u8; 4]>,
    }

    let output_data = Phase2Output {
        translation,
        metadata: Phase2Metadata {
            input_file: input.to_string_lossy().to_string(),
            label0_size: (bubble_img.width(), bubble_img.height()),
            label1_bbox: [label1_x1, label1_y1, label1_x2, label1_y2],
            label1_size: (label1_width, label1_height),
            processing_time_ms: elapsed.as_secs_f64() * 1000.0,
            api_keys_available: config.api_keys().len(),
            local_background_analysis: LocalAnalysis {
                is_complex: is_complex_local,
                detected_color: detected_color.map(|c| c.0),
            },
            recommended_rendering: if needs_ai_redraw {
                "complex".to_string()
            } else {
                "simple".to_string()
            },
        },
    };

    // Save JSON
    info!("💾 Saving translation JSON...");
    let json = serde_json::to_string_pretty(&output_data)
        .context("Failed to serialize translation")?;
    std::fs::write(&output, json)
        .context("Failed to write output JSON")?;
    info!("✓ Saved to: {}", output.display());
    info!("");

    info!("{}", "=".repeat(70));
    info!("PHASE 2 COMPLETE");
    info!("{}", "=".repeat(70));
    info!("Use this JSON with Phase 3 to render the translated text:");
    info!("  cargo run -- phase3 --input <bubble.png> --api-response {} --detections-json <phase1_dir>/detections.json",
        output.display());
    info!("{}", "=".repeat(70));

    Ok(())
}

/// Execute Phase 3: Rendering
pub async fn execute_phase3(
    input: PathBuf,
    api_response: PathBuf,
    detections_json: PathBuf,
    output: PathBuf,
    image_gen_model: Option<String>,
    _font_family: Option<String>,
    debug_polygon: bool,
    speech: Option<PathBuf>,
    insertion: bool,
    config: Arc<Config>,
) -> Result<()> {
    info!("{}", "=".repeat(70));
    info!("PHASE 3: RENDERING{}", if insertion { " (INSERTION MODE)" } else { "" });
    info!("{}", "=".repeat(70));
    info!("Input bubble: {}", input.display());
    info!("API response: {}", api_response.display());
    info!("Detections JSON: {}", detections_json.display());
    info!("Output: {}", output.display());
    if insertion {
        info!("Mode: INSERTION (rendering text on pre-cleaned bubble)");
    }
    if let Some(ref model) = image_gen_model {
        info!("Image gen model: {}", model);
    }
    info!("");

    // Load detections JSON to get text_regions
    info!("📖 Loading detections JSON...");
    let detections_str = std::fs::read_to_string(&detections_json)
        .context("Failed to read detections JSON")?;

    #[derive(serde::Deserialize)]
    struct Phase1Detection {
        bbox: [i32; 4],
        confidence: f32,
        text_regions: Vec<[i32; 4]>,
    }

    #[derive(serde::Deserialize)]
    struct Phase1Output {
        detections: Vec<Phase1Detection>,
    }

    let phase1_output: Phase1Output = serde_json::from_str(&detections_str)
        .context("Failed to parse detections JSON")?;

    // Get first detection (assuming bubble_01 corresponds to detection 0)
    let phase1_detection = phase1_output.detections.get(0)
        .context("No detections found in Phase 1 output")?;

    info!("✓ Detection loaded: bbox={:?}, text_regions={}",
        phase1_detection.bbox, phase1_detection.text_regions.len());
    info!("");

    // Load API response JSON
    info!("📖 Loading translation JSON...");
    let json_str = std::fs::read_to_string(&api_response)
        .context("Failed to read API response JSON")?;

    #[derive(serde::Deserialize)]
    struct Phase2Metadata {
        label0_size: (u32, u32),
        label1_bbox: [u32; 4],
        label1_size: (u32, u32),
        #[serde(flatten)]
        _other: serde_json::Value,
    }

    #[derive(serde::Deserialize)]
    struct Phase2Output {
        translation: TextTranslation,
        metadata: Phase2Metadata,
    }

    let phase2_output: Phase2Output = serde_json::from_str(&json_str)
        .context("Failed to parse API response JSON")?;
    let translation = phase2_output.translation;
    let label1_bbox = phase2_output.metadata.label1_bbox;

    info!("✓ Metadata loaded:");
    info!("  Label 0 (full bubble) size: {}x{}", phase2_output.metadata.label0_size.0, phase2_output.metadata.label0_size.1);
    info!("  Label 1 (text region) bbox: [{}, {}, {}, {}]",
        label1_bbox[0], label1_bbox[1], label1_bbox[2], label1_bbox[3]);
    info!("  Label 1 size: {}x{}", phase2_output.metadata.label1_size.0, phase2_output.metadata.label1_size.1);

    info!("✓ Translation loaded");
    info!("  Text: {}", translation.english_translation);
    info!("  Redraw required: {}", translation.redraw_bg_required);
    info!("");

    // Note: Polygon constraining removed - we now use edge-based detection
    info!("✓ Using edge-based text region detection (no polygon data from Gemini)");
    info!("");

    // Load bubble image
    info!("📖 Loading bubble image...");
    let img = image::open(&input)
        .context("Failed to load input image")?;
    let bubble_rgba = img.to_rgba8();
    info!("✓ Bubble loaded: {}x{}", img.width(), img.height());
    info!("");

    // Debug polygon visualization
    if debug_polygon {
        info!("🎨 Creating polygon visualization...");

        use image::Rgba;
        let mut debug_img = bubble_rgba.clone();

        // Draw Label 1 text region bbox in GREEN
        let label1_x1 = label1_bbox[0];
        let label1_y1 = label1_bbox[1];
        let label1_x2 = label1_bbox[2];
        let label1_y2 = label1_bbox[3];

        let green = Rgba([0u8, 255u8, 0u8, 255u8]);

        // Draw Label 1 bbox (GREEN)
        for x in label1_x1..label1_x2 {
            if x < debug_img.width() && label1_y1 < debug_img.height() {
                debug_img.put_pixel(x, label1_y1, green);
                let y2_clamped = label1_y2.saturating_sub(1).min(debug_img.height() - 1);
                debug_img.put_pixel(x, y2_clamped, green);
            }
        }
        for y in label1_y1..label1_y2 {
            if y < debug_img.height() && label1_x1 < debug_img.width() {
                debug_img.put_pixel(label1_x1, y, green);
                let x2_clamped = label1_x2.saturating_sub(1).min(debug_img.width() - 1);
                debug_img.put_pixel(x2_clamped, y, green);
            }
        }

        // Note: Polygon visualization removed - no longer using polygon data from Gemini

        // Save debug visualization
        let debug_path = output.with_extension("").with_extension("debug.png");
        debug_img.save(&debug_path)
            .context("Failed to save debug visualization")?;

        info!("✓ Debug visualization saved to: {}", debug_path.display());
        info!("  GREEN = Label 1 text region bbox");
        info!("");
    }

    // Initialize services
    info!("🔧 Initializing services...");
    let translator = Arc::new(TranslationService::new(config.clone()));
    let renderer = Arc::new(RenderingService::new(config.clone()));
    info!("✓ Services ready");
    info!("");

    // Analyze background complexity
    info!("🔬 Analyzing background complexity...");
    let (is_complex_local, detected_color) =
        RenderingService::analyze_background_complexity(&bubble_rgba);
    info!("  Local analysis: complex={}", is_complex_local);
    if let Some(color) = detected_color {
        info!("  Detected color: RGB({}, {}, {})", color[0], color[1], color[2]);
    }

    let needs_ai_redraw = translation.redraw_bg_required && is_complex_local;
    info!("  Final decision: {}", if needs_ai_redraw { "Complex (AI)" } else { "Simple" });
    info!("");

    // Create detection with text_regions from Phase 1
    let detection = BubbleDetection {
        bbox: phase1_detection.bbox,
        confidence: phase1_detection.confidence,
        page_index: 0,
        bubble_index: 0,
        text_regions: phase1_detection.text_regions.clone(),
    };

    // In insertion mode, skip cleaning and use input as-is
    if !insertion {
        info!("🧹 Will clean {} text regions before rendering", detection.text_regions.len());
    } else {
        info!("📝 INSERTION MODE: Skipping cleaning, using pre-cleaned bubble as-is");
    }

    // Save cleaned bubble if --speech flag provided (text removed, no translation)
    if !insertion {
        if let Some(ref speech_path) = speech {
            info!("💾 Saving cleaned speech bubble (no translation)...");
            use crate::rendering::RenderingService;
            let cleaned = RenderingService::clean_text_from_bubble_public(
                &bubble_rgba,
                &detection.text_regions,
                &detection.bbox
            );
            cleaned.save(speech_path)
                .context("Failed to save cleaned speech bubble")?;
            info!("✓ Cleaned bubble saved to: {}", speech_path.display());
            info!("");
        }
    }

    // Render
    info!("🎨 Rendering translated text...");
    let start = std::time::Instant::now();

    let rendered = if insertion {
        // INSERTION MODE: Render directly on pre-cleaned bubble without cleaning
        info!("  INSERTION MODE: Rendering text on pre-cleaned bubble (skipping cleaning)...");
        info!("  Text will be constrained to Label 1 region: {:?}", detection.text_regions);
        

        // Keep text_regions for proper Label 1 constraint during rendering
        // Skip cleaning since bubble is already cleaned in insertion mode
        renderer
            .render_bubble_simple_background_skip_cleaning(&bubble_rgba, &detection, &translation)
            .await
            .context("Failed to render text in insertion mode")?
    } else if needs_ai_redraw {
        info!("  Using complex background rendering (AI text removal)...");

        // Convert to bytes for API call
        let mut bubble_bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bubble_bytes),
            image::ImageFormat::Png
        )?;

        info!("  Calling AI image generation API...");
        let api_start = std::time::Instant::now();
        let cleaned_bytes = translator
            .remove_text_from_image(&bubble_bytes, image_gen_model.as_deref())
            .await
            .context("Failed to remove text from image")?;
        let api_elapsed = api_start.elapsed();
        info!("  ✓ AI cleaning completed in {:.2}ms", api_elapsed.as_secs_f64() * 1000.0);

        renderer
            .render_bubble_complex_background(&cleaned_bytes, &detection, &translation)
            .await
            .context("Failed to render complex background")?
    } else {
        info!("  Using simple background rendering...");
        renderer
            .render_bubble_simple_background(&bubble_rgba, &detection, &translation)
            .await
            .context("Failed to render simple background")?
    };

    let elapsed = start.elapsed();
    info!("✓ Rendering completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    info!("");

    // Save result
    info!("💾 Saving rendered image...");
    rendered.save(&output)
        .context("Failed to save output image")?;
    info!("✓ Saved to: {}", output.display());
    info!("");

    // Print summary
    info!("{}", "=".repeat(70));
    info!("PHASE 3 SUMMARY");
    info!("{}", "=".repeat(70));
    info!("Input: {}x{}", img.width(), img.height());
    info!("Output: {}x{}", rendered.width(), rendered.height());
    info!("Rendering method: {}", if needs_ai_redraw { "Complex (AI)" } else { "Simple" });
    info!("Processing time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    info!("Text rendered: {}", translation.english_translation);
    info!("Font: {} ({})", translation.font_family, translation.font_color);
    info!("{}", "=".repeat(70));

    Ok(())
}
