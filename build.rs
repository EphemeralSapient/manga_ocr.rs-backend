use std::env;

fn main() {
    let detector_path = "models/detector.onnx";
    let mask_path = "models/mask.onnx";

    // Trigger rebuild if models change
    println!("cargo:rerun-if-changed={}", detector_path);
    println!("cargo:rerun-if-changed={}", mask_path);

    // Check if models exist and are real (not LFS stubs)
    let detector_exists = std::path::Path::new(detector_path).exists();
    let mask_exists = std::path::Path::new(mask_path).exists();

    if detector_exists && mask_exists {
        let detector_size = std::fs::metadata(detector_path).map(|m| m.len()).unwrap_or(0);
        let mask_size = std::fs::metadata(mask_path).map(|m| m.len()).unwrap_or(0);

        // LFS stub files are tiny (~130 bytes), real models are 100MB+
        if detector_size > 10_000 && mask_size > 10_000 {
            let total_size = detector_size + mask_size;
            println!("cargo:warning=Embedding models into binary:");
            println!("cargo:warning=  - detector.onnx: {:.1} MB", detector_size as f64 / 1_048_576.0);
            println!("cargo:warning=  - mask.onnx: {:.1} MB", mask_size as f64 / 1_048_576.0);
            println!("cargo:warning=  Total: {:.1} MB", total_size as f64 / 1_048_576.0);
        } else {
            println!("cargo:warning=Models are LFS stubs - binary will load from runtime path");
            println!("cargo:warning=Binary will be small (~25MB). Models must be provided at runtime.");
        }
    } else {
        println!("cargo:warning=Models not found - binary will load from runtime path");
        println!("cargo:warning=Binary will be small (~25MB). Models must be provided at runtime.");
    }

    // Get target platform early (needed for DirectML DLL copying)
    let target = env::var("TARGET").unwrap_or_default();

    // Detect enabled acceleration features
    let mut gpu_features = Vec::new();
    let mut cpu_features = Vec::new();

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        gpu_features.push("CUDA");
    }
    if env::var("CARGO_FEATURE_TENSORRT").is_ok() {
        gpu_features.push("TensorRT");
    }
    if env::var("CARGO_FEATURE_DIRECTML").is_ok() {
        gpu_features.push("DirectML");

        // Copy DirectML.dll to output directory (fixes version mismatch with System32)
        // Windows ships old DirectML 1.8, ONNX Runtime needs 1.15.4+
        if target.contains("windows") {
            let out_dir = env::var("OUT_DIR").unwrap();
            let profile = env::var("PROFILE").unwrap();
            let dll_source = "libs/windows/x64/DirectML.dll";

            if std::path::Path::new(dll_source).exists() {
                // Calculate target directory (e.g., target/release or target/debug)
                let target_dir = std::path::Path::new(&out_dir)
                    .ancestors()
                    .nth(3)
                    .unwrap()
                    .join(&profile);

                let dll_target = target_dir.join("DirectML.dll");

                match std::fs::copy(dll_source, &dll_target) {
                    Ok(_) => println!("cargo:warning=Copied DirectML.dll v1.15.4 to output (fixes System32 v1.8 conflict)"),
                    Err(e) => println!("cargo:warning=Failed to copy DirectML.dll: {}", e),
                }
            } else {
                println!("cargo:warning=DirectML.dll not found in libs/windows/x64/");
                println!("cargo:warning=Download from: https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4");
            }
        }
    }
    if env::var("CARGO_FEATURE_COREML").is_ok() {
        gpu_features.push("CoreML");
    }
    if env::var("CARGO_FEATURE_OPENVINO").is_ok() {
        cpu_features.push("OpenVINO");
    }
    if env::var("CARGO_FEATURE_XNNPACK").is_ok() {
        cpu_features.push("XNNPACK");
    }

    // Print acceleration status
    if !gpu_features.is_empty() {
        println!("cargo:warning=GPU acceleration enabled: {}", gpu_features.join(", "));
    }
    if !cpu_features.is_empty() {
        println!("cargo:warning=CPU acceleration enabled: {}", cpu_features.join(", "));
    }
    if gpu_features.is_empty() && cpu_features.is_empty() {
        println!("cargo:warning=Building with CPU-only inference (no acceleration)");
        println!("cargo:warning=To enable acceleration: cargo build --features cuda (or directml/xnnpack)");
    }

    // Platform-specific warnings (target already defined above)

    if target.contains("windows-gnu") && gpu_features.contains(&"CUDA") {
        println!("cargo:warning=WARNING: CUDA binaries may not be available for Windows GNU target");
        println!("cargo:warning=Consider using DirectML instead: cargo build --features directml");
    }
}
