use std::env;

fn main() {
    // Ensure both model files exist at compile time
    let detector_path = "models/detector.onnx";
    let mask_path = "models/mask.onnx";

    // Check detector model
    if !std::path::Path::new(detector_path).exists() {
        panic!(
            "Detection model not found at {}. \n\
            Please copy the model file to this location before building.",
            detector_path
        );
    }

    // Check segmentation model
    if !std::path::Path::new(mask_path).exists() {
        panic!(
            "Segmentation model not found at {}. \n\
            Please copy the model file to this location before building.",
            mask_path
        );
    }

    // Trigger rebuild if models change
    println!("cargo:rerun-if-changed={}", detector_path);
    println!("cargo:rerun-if-changed={}", mask_path);

    // Show embedded model sizes
    let detector_size = std::fs::metadata(detector_path).map(|m| m.len()).unwrap_or(0);
    let mask_size = std::fs::metadata(mask_path).map(|m| m.len()).unwrap_or(0);
    let total_size = detector_size + mask_size;

    println!("cargo:warning=Embedding models into binary:");
    println!("cargo:warning=  - detector.onnx: {:.1} MB", detector_size as f64 / 1_048_576.0);
    println!("cargo:warning=  - mask.onnx: {:.1} MB", mask_size as f64 / 1_048_576.0);
    println!("cargo:warning=  Total: {:.1} MB", total_size as f64 / 1_048_576.0);

    // Detect enabled acceleration features
    let mut enabled_features = Vec::new();

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        enabled_features.push("CUDA");
    }
    if env::var("CARGO_FEATURE_TENSORRT").is_ok() {
        enabled_features.push("TensorRT");
    }
    if env::var("CARGO_FEATURE_DIRECTML").is_ok() {
        enabled_features.push("DirectML");
    }
    if env::var("CARGO_FEATURE_COREML").is_ok() {
        enabled_features.push("CoreML");
    }
    if env::var("CARGO_FEATURE_OPENVINO").is_ok() {
        enabled_features.push("OpenVINO");
    }

    if enabled_features.is_empty() {
        println!("cargo:warning=Building with CPU-only inference (no GPU acceleration)");
        println!("cargo:warning=To enable GPU: cargo build --features cuda (or directml on Windows)");
    } else {
        println!("cargo:warning=GPU acceleration enabled: {}", enabled_features.join(", "));
    }

    // Platform-specific warnings
    let target = env::var("TARGET").unwrap_or_default();

    if target.contains("windows-gnu") && enabled_features.contains(&"CUDA") {
        println!("cargo:warning=WARNING: CUDA binaries may not be available for Windows GNU target");
        println!("cargo:warning=Consider using DirectML instead: cargo build --features directml");
    }
}
