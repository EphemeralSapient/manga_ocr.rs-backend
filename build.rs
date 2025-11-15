use std::env;

fn main() {
    // Ensure model file exists at compile time
    let model_path = "models/detector.onnx";

    if !std::path::Path::new(model_path).exists() {
        panic!(
            "ONNX model not found at {}. \n\
            Please copy the model file to this location before building.",
            model_path
        );
    }

    println!("cargo:rerun-if-changed={}", model_path);
    println!("cargo:warning=Embedding ONNX model ({} bytes)",
        std::fs::metadata(model_path)
            .map(|m| m.len())
            .unwrap_or(0)
    );

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
