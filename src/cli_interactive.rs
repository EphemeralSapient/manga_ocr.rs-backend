/// Interactive CLI mode with guided prompts
///
/// This module provides a user-friendly interactive interface for running
/// the manga translation pipeline with step-by-step prompts.

use anyhow::Result;
use console::style;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, MultiSelect, Select};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::cli;
use crate::config::Config;

/// Main interactive workflow
pub async fn execute_interactive(config: Arc<Config>) -> Result<()> {
    println!("\n{}", style("╔═══════════════════════════════════════════════════════╗").cyan().bold());
    println!("{}", style("║   🎨 MANGA TRANSLATION - INTERACTIVE MODE 🎨        ║").cyan().bold());
    println!("{}", style("╚═══════════════════════════════════════════════════════╝").cyan().bold());
    println!();

    let theme = ColorfulTheme::default();

    // Step 1: Choose workflow phases
    let phase_options = vec![
        "Phase 1: Detection (Detect & crop bubbles)",
        "Phase 2: Translation (Translate text via API)",
        "Phase 3: Rendering (Render translation on image)",
        "All phases (Full pipeline)",
    ];

    let phase_selection = Select::with_theme(&theme)
        .with_prompt("Which workflow do you want to run?")
        .items(&phase_options)
        .default(3)
        .interact()
        .map_err(|e| anyhow::anyhow!("Interactive prompt error: {}", e))?;

    let run_all = phase_selection == 3;
    let run_phase1 = run_all || phase_selection == 0;
    let run_phase2 = run_all || phase_selection == 1;
    let run_phase3 = run_all || phase_selection == 2;

    // Step 2: Input mode (single file or folder)
    let input_mode = Select::with_theme(&theme)
        .with_prompt("Input mode")
        .items(&["Single image file", "Folder (batch process)"])
        .default(0)
        .interact()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let is_batch = input_mode == 1;

    // Step 3: Get input path
    let input_path: PathBuf = Input::with_theme(&theme)
        .with_prompt(if is_batch {
            "Input folder path"
        } else {
            "Input image path"
        })
        .default(if is_batch {
            "input_images/".to_string()
        } else {
            "input_image.webp".to_string()
        })
        .interact_text()
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .into();

    // Step 4: Get output folder
    let output_base: PathBuf = Input::with_theme(&theme)
        .with_prompt("Output folder")
        .default(if run_all {
            "output/".to_string()
        } else if run_phase1 {
            "phase_1_output/".to_string()
        } else if run_phase2 {
            "phase_2_output/".to_string()
        } else {
            "phase_3_output/".to_string()
        })
        .interact_text()
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .into();

    // Step 5: Model selection (if running translation)
    let translation_model = if run_phase2 {
        let models = vec![
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemini-2.5-pro",
            "gemini-pro-latest",
            "gemini-2.5-flash-lite",
            "gemini-flash-lite-latest",
        ];

        let model_choice = Select::with_theme(&theme)
            .with_prompt("Translation model")
            .items(&models)
            .default(0)
            .interact()
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        Some(models[model_choice].to_string())
    } else {
        None
    };

    // Step 6: Font selection (if running rendering)
    let font_family = if run_phase3 {
        let fonts = vec![
            "arial (Default, clean)",
            "anime-ace (Comic style)",
            "anime-ace-3 (Comic style v3)",
            "comic-sans (Casual)",
            "ms-yahei (CJK support)",
            "noto-sans-mono-cjk (Monospace CJK)",
        ];

        let font_choice = Select::with_theme(&theme)
            .with_prompt("Font family")
            .items(&fonts)
            .default(0)
            .interact()
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        Some(match font_choice {
            0 => "arial",
            1 => "anime-ace",
            2 => "anime-ace-3",
            3 => "comic-sans",
            4 => "ms-yahei",
            5 => "noto-sans-mono-cjk",
            _ => "arial",
        }.to_string())
    } else {
        None
    };

    // Step 7: Additional options
    let mut options = vec![];
    if run_phase1 {
        options.push("Visualize bounding boxes");
        options.push("Include text-free regions");
    }
    if run_phase3 {
        options.push("Debug polygon visualization");
        options.push("Save cleaned bubble (--speech mode)");
        options.push("Insertion mode (skip cleaning)");
    }

    let selected_options = if !options.is_empty() {
        MultiSelect::with_theme(&theme)
            .with_prompt("Additional options (space to select, enter to confirm)")
            .items(&options)
            .interact()
            .map_err(|e| anyhow::anyhow!("{}", e))?
    } else {
        vec![]
    };

    let visualize = run_phase1 && selected_options.contains(&0);
    let include_text_free = run_phase1 && selected_options.iter().any(|&i| options.get(i) == Some(&"Include text-free regions"));
    let debug_polygon = run_phase3 && selected_options.iter().any(|&i| options.get(i) == Some(&"Debug polygon visualization"));
    let speech_mode = run_phase3 && selected_options.iter().any(|&i| options.get(i) == Some(&"Save cleaned bubble (--speech mode)"));
    let insertion = run_phase3 && selected_options.iter().any(|&i| options.get(i) == Some(&"Insertion mode (skip cleaning)"));

    // Step 8: Confirm and execute
    println!("\n{}", style("═".repeat(60)).cyan());
    println!("{}", style("SUMMARY").cyan().bold());
    println!("{}", style("═".repeat(60)).cyan());
    println!("  Workflow: {}", style(&phase_options[phase_selection]).green());
    println!("  Input: {}", style(input_path.display()).green());
    println!("  Output: {}", style(output_base.display()).green());
    if let Some(ref model) = translation_model {
        println!("  Model: {}", style(model).green());
    }
    if let Some(ref font) = font_family {
        println!("  Font: {}", style(font).green());
    }
    if !selected_options.is_empty() {
        println!("  Options: {}", style(format!("{} enabled", selected_options.len())).green());
    }
    println!("{}", style("═".repeat(60)).cyan());
    println!();

    let confirm = Confirm::with_theme(&theme)
        .with_prompt("Ready to process?")
        .default(true)
        .interact()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if !confirm {
        println!("\n{}", style("❌ Cancelled").red());
        return Ok(());
    }

    println!("\n{}", style("🚀 Starting workflow...").green().bold());
    println!();

    // Execute based on selection
    if run_all {
        execute_full_pipeline(
            input_path,
            output_base,
            translation_model,
            font_family,
            visualize,
            include_text_free,
            debug_polygon,
            speech_mode,
            insertion,
            config,
        ).await?;
    } else {
        if run_phase1 {
            cli::execute_phase1(input_path.clone(), output_base.clone(), visualize, include_text_free, config.clone()).await?;
        }
        if run_phase2 {
            let input = output_base.join("bubble_01.png");
            let detections_json = output_base.join("detections.json");
            let output_json = output_base.join("phase2.json");
            cli::execute_phase2(input, detections_json, output_json, translation_model, font_family.clone(), config.clone()).await?;
        }
        if run_phase3 {
            let input = output_base.join("bubble_01.png");
            let api_response = output_base.join("phase2.json");
            let detections_json = output_base.join("detections.json");
            let output_img = output_base.join("final.png");
            let speech_path = if speech_mode { Some(output_base.join("cleaned.png")) } else { None };
            cli::execute_phase3(
                input,
                api_response,
                detections_json,
                output_img,
                None, // image_gen_model
                font_family,
                debug_polygon,
                speech_path,
                insertion,
                config,
            ).await?;
        }
    }

    println!("\n{}", style("✅ Workflow completed successfully!").green().bold());
    Ok(())
}

/// Execute full pipeline (all 3 phases)
async fn execute_full_pipeline(
    input: PathBuf,
    output: PathBuf,
    translation_model: Option<String>,
    font_family: Option<String>,
    visualize: bool,
    include_text_free: bool,
    debug_polygon: bool,
    speech_mode: bool,
    insertion: bool,
    config: Arc<Config>,
) -> Result<()> {
    // Phase 1
    info!("📍 PHASE 1: Detection");
    let phase1_output = output.join("phase1");
    std::fs::create_dir_all(&phase1_output)?;
    cli::execute_phase1(input, phase1_output.clone(), visualize, include_text_free, config.clone()).await?;

    // Phase 2
    info!("\n📍 PHASE 2: Translation");
    let bubble_path = phase1_output.join("bubble_01.png");
    let detections_json = phase1_output.join("detections.json");
    let phase2_output = output.join("phase2.json");
    cli::execute_phase2(
        bubble_path.clone(),
        detections_json.clone(),
        phase2_output.clone(),
        translation_model,
        font_family.clone(),
        config.clone(),
    ).await?;

    // Phase 3
    info!("\n📍 PHASE 3: Rendering");
    let final_output = output.join("final.png");
    let speech_path = if speech_mode {
        Some(output.join("cleaned.png"))
    } else {
        None
    };
    cli::execute_phase3(
        bubble_path,
        phase2_output,
        detections_json,
        final_output,
        None,
        font_family,
        debug_polygon,
        speech_path,
        insertion,
        config,
    ).await?;

    Ok(())
}
