// Shared ONNX Runtime session builder with automatic hardware acceleration detection
//
// This module is the SINGLE SOURCE OF TRUTH for ONNX session creation.
// All services (detection, segmentation, OCR) should use these builders.

use anyhow::{Context, Result};
use crossbeam::channel::{bounded, Receiver, Sender};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::execution_providers::CPUExecutionProvider;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use tracing::{info, warn, debug};

/// Calculate optimal thread count for ONNX Runtime CPU inference.
///
/// Research shows that using all CPU cores can actually HURT performance on Windows
/// due to thread synchronization overhead. Capping at 6 threads showed 2x speedup
/// in benchmarks on 8-core Windows systems.
///
/// Reference: https://github.com/microsoft/onnxruntime/issues/3713
pub fn optimal_intra_op_threads() -> usize {
    let total_cores = num_cpus::get();

    // Platform-specific optimization:
    // - Windows: Cap at 6 threads due to synchronization overhead
    // - Linux/macOS: Use all physical cores for maximum throughput
    #[cfg(target_os = "windows")]
    let optimal = std::cmp::min(6, total_cores).max(1);

    #[cfg(not(target_os = "windows"))]
    let optimal = total_cores.max(1);

    debug!("CPU threads: {} total cores, using {} for inference", total_cores, optimal);
    optimal
}

/// Track if platform info has been logged (avoid spam)
static LOGGED_PLATFORM: AtomicBool = AtomicBool::new(false);

// Import acceleration providers based on features
#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(all(target_os = "macos", feature = "coreml"))]
use ort::execution_providers::CoreMLExecutionProvider;

#[cfg(all(target_os = "windows", feature = "directml"))]
use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(feature = "openvino")]
use ort::execution_providers::OpenVINOExecutionProvider;

#[cfg(feature = "xnnpack")]
use ort::execution_providers::XNNPACKExecutionProvider;

/// Generic session pool for ONNX Runtime sessions
/// OPTIMIZATION: Eliminates duplication between SessionPool and SegmentationSessionPool
/// Uses crossbeam bounded channel instead of tokio Mutex (15-25% faster under load)
pub struct OnnxSessionPool {
    sender: Sender<Session>,
    receiver: Receiver<Session>,
}

impl OnnxSessionPool {
    /// Create a new session pool with the given capacity
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        Self { sender, receiver }
    }

    /// Get sender for adding sessions to the pool
    pub fn sender(&self) -> &Sender<Session> {
        &self.sender
    }

    /// Acquire a session from the pool (blocks if pool is empty)
    /// crossbeam recv() is lock-free and much faster than tokio Mutex
    pub fn acquire(&self) -> Session {
        self.receiver.recv().expect("Session pool exhausted")
    }

    /// Release a session back to the pool
    /// crossbeam send() on bounded channel blocks if full (backpressure)
    pub fn release(&self, session: Session) {
        self.sender.send(session).expect("Failed to return session to pool");
    }
}

/// Dynamic session pool that supports runtime resizing
/// Uses Mutex<VecDeque> for flexibility at slight performance cost
pub struct DynamicSessionPool {
    sessions: Mutex<VecDeque<Session>>,
    capacity: AtomicUsize,
    in_use: AtomicUsize,
}

impl DynamicSessionPool {
    /// Create a new dynamic session pool
    pub fn new(capacity: usize) -> Self {
        Self {
            sessions: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity: AtomicUsize::new(capacity),
            in_use: AtomicUsize::new(0),
        }
    }

    /// Add a session to the pool
    pub fn add_session(&self, session: Session) {
        let mut sessions = self.sessions.lock();
        sessions.push_back(session);
    }

    /// Get current capacity
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::SeqCst)
    }

    /// Get number of available sessions
    pub fn available(&self) -> usize {
        self.sessions.lock().len()
    }

    /// Get number of sessions in use
    pub fn in_use(&self) -> usize {
        self.in_use.load(Ordering::SeqCst)
    }

    /// Acquire a session from the pool (blocks if pool is empty)
    pub fn acquire(&self) -> Session {
        loop {
            {
                let mut sessions = self.sessions.lock();
                if let Some(session) = sessions.pop_front() {
                    self.in_use.fetch_add(1, Ordering::SeqCst);
                    return session;
                }
            }
            // Yield to allow other threads to release sessions
            std::thread::yield_now();
        }
    }

    /// Try to acquire a session without blocking
    pub fn try_acquire(&self) -> Option<Session> {
        let mut sessions = self.sessions.lock();
        if let Some(session) = sessions.pop_front() {
            self.in_use.fetch_add(1, Ordering::SeqCst);
            Some(session)
        } else {
            None
        }
    }

    /// Release a session back to the pool
    pub fn release(&self, session: Session) {
        self.in_use.fetch_sub(1, Ordering::SeqCst);
        let mut sessions = self.sessions.lock();
        sessions.push_back(session);
    }

    /// Remove one session from the pool (for downsizing)
    /// Returns None if no sessions available
    pub fn remove_one(&self) -> Option<Session> {
        let mut sessions = self.sessions.lock();
        if let Some(session) = sessions.pop_back() {
            self.capacity.fetch_sub(1, Ordering::SeqCst);
            Some(session)
        } else {
            None
        }
    }

    /// Increase capacity (call add_session separately to add actual sessions)
    pub fn increase_capacity(&self, amount: usize) {
        self.capacity.fetch_add(amount, Ordering::SeqCst);
    }

    /// Drain all sessions from the pool
    pub fn drain_all(&self) -> Vec<Session> {
        let mut sessions = self.sessions.lock();
        self.capacity.store(0, Ordering::SeqCst);
        sessions.drain(..).collect()
    }
}

/// Build ONNX Runtime session with automatic hardware acceleration detection
///
/// Tries acceleration providers in this order:
/// 1. TensorRT (NVIDIA GPUs, best performance)
/// 2. CUDA (NVIDIA GPUs)
/// 3. CoreML (Apple Silicon M1/M2/M3)
/// 4. DirectML (Windows GPU acceleration)
/// 5. OpenVINO (Intel CPU optimizations)
/// 6. XNNPACK (ARM CPU optimizations - mobile, Raspberry Pi)
/// 7. CPU (fallback)
///
/// # Arguments
/// * `model_bytes` - ONNX model file bytes
/// * `model_name` - Name for logging (e.g., "detection", "segmentation")
/// * `model_size_mb` - Model size in MB for error messages
///
/// # Returns
/// (backend_name, Session)
pub fn build_session_with_acceleration(
    model_bytes: &[u8],
    model_name: &str,
    model_size_mb: f32,
) -> Result<(String, Session)> {
    // Log platform info once
    if !LOGGED_PLATFORM.swap(true, Ordering::Relaxed) {
        info!("Platform: {}/{}", std::env::consts::OS, std::env::consts::ARCH);
    }

    // Validate model bytes
    if model_bytes.len() < 100 {
        anyhow::bail!(
            "Model file is too small ({} bytes). This might be a Git LFS stub. \
            Please ensure Git LFS is installed and the model is properly checked out.",
            model_bytes.len()
        );
    }

    // Check for forced backend via environment variable
    if let Ok(forced_backend) = std::env::var("INFERENCE_BACKEND") {
        if !forced_backend.is_empty() && forced_backend.to_uppercase() != "AUTO" {
            info!("INFERENCE_BACKEND={}, forcing specific backend for {}", forced_backend, model_name);
            return try_forced_backend(&forced_backend, model_bytes, model_name, model_size_mb);
        }
    }

    // Try hardware acceleration in order of preference
    // Only attempt providers that are compiled in via Cargo features

    // Try TensorRT (if feature enabled)
    #[cfg(feature = "tensorrt")]
    {
        if let Ok(session) = Session::builder()
            .and_then(|b| b.with_execution_providers([TensorRTExecutionProvider::default().build()]))
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_intra_threads(optimal_intra_op_threads()))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_memory(model_bytes))
        {
            info!("✓ Using TensorRT acceleration for {}", model_name);
            return Ok(("TensorRT".to_string(), session));
        }
    }

    // Try CUDA (if feature enabled)
    #[cfg(feature = "cuda")]
    {
        if let Ok(session) = Session::builder()
            .and_then(|b| b.with_execution_providers([CUDAExecutionProvider::default().build()]))
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_intra_threads(optimal_intra_op_threads()))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_memory(model_bytes))
        {
            info!("✓ Using CUDA acceleration for {}", model_name);
            return Ok(("CUDA".to_string(), session));
        }
    }

    // Try CoreML (Apple Silicon, if feature enabled)
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    {
        if let Ok(session) = Session::builder()
            .and_then(|b| b.with_execution_providers([CoreMLExecutionProvider::default().build()]))
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_intra_threads(optimal_intra_op_threads()))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_memory(model_bytes))
        {
            info!("✓ Using CoreML acceleration for {} (Apple Neural Engine)", model_name);
            return Ok(("CoreML".to_string(), session));
        }
    }

    // Try DirectML (Windows, if feature enabled)
    #[cfg(all(target_os = "windows", feature = "directml"))]
    {
        // DirectML-only (NO CPU fallback)
        // IMPORTANT: DirectML requires specific settings for stability
        // - parallel_execution(false): Sequential execution required
        // - memory_pattern(false): Memory pattern must be disabled
        // - Level2 optimization: Balanced optimization for better performance
        if let Ok(session) = Session::builder()
            .and_then(|b| b.with_execution_providers([
                DirectMLExecutionProvider::default().build()  // GPU acceleration only
            ]))
            .and_then(|b| b.with_parallel_execution(false))  // REQUIRED: Sequential execution
            .and_then(|b| b.with_memory_pattern(false))      // Disable memory pattern for stability
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level2))
            .and_then(|b| b.with_intra_threads(optimal_intra_op_threads()))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_memory(model_bytes))
        {
            info!("✓ Using DirectML acceleration for {} (GPU-only, no CPU fallback)", model_name);
            return Ok(("DirectML".to_string(), session));
        }
    }

    // Try OpenVINO (Intel CPU optimization, if feature enabled)
    #[cfg(feature = "openvino")]
    {
        if let Ok(session) = Session::builder()
            .and_then(|b| b.with_execution_providers([
                OpenVINOExecutionProvider::default()
                    .with_device_type("CPU")
                    .build()
            ]))
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_intra_threads(optimal_intra_op_threads()))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_memory(model_bytes))
        {
            info!("✓ Using OpenVINO acceleration for {} (Intel CPU optimizations)", model_name);
            return Ok(("OpenVINO-CPU".to_string(), session));
        }
    }

    // Try XNNPACK (ARM CPU optimization, if feature enabled)
    #[cfg(feature = "xnnpack")]
    {
        if let Ok(session) = Session::builder()
            .and_then(|b| b.with_execution_providers([
                XNNPACKExecutionProvider::default().build()
            ]))
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_intra_threads(optimal_intra_op_threads()))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_memory(model_bytes))
        {
            info!("✓ Using XNNPACK acceleration for {} (ARM CPU optimizations)", model_name);
            return Ok(("XNNPACK".to_string(), session));
        }
    }

    // Final fallback: Pure CPU (no acceleration)
    let session = Session::builder()
        .context(format!("Failed to create ONNX session builder for {}", model_name))?
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .context(format!("Failed to configure CPU execution provider for {}", model_name))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .context(format!("Failed to set graph optimization level for {}", model_name))?
        .with_intra_threads(optimal_intra_op_threads())
        .context(format!("Failed to configure intra-op threads for {}", model_name))?
        .with_inter_threads(1)
        .context(format!("Failed to configure inter-op threads for {}", model_name))?
        .commit_from_memory(model_bytes)
        .context(format!(
            "Failed to load {} ONNX model from memory ({:.1} MB). \
            This usually indicates:\n  \
            1. Model file corruption during transfer\n  \
            2. ONNX Runtime version/platform mismatch\n  \
            3. Model created with incompatible ONNX opset version",
            model_name, model_size_mb
        ))?;

    warn!("⚠️  Using CPU-only inference for {} (no GPU acceleration available)", model_name);
    Ok(("CPU".to_string(), session))
}

/// Try to force a specific backend (for testing/debugging)
///
/// # Arguments
/// * `backend` - Backend name (case-insensitive): "cuda", "tensorrt", "openvino", "directml", "coreml", "xnnpack", "cpu"
/// * `model_bytes` - ONNX model bytes
/// * `model_name` - Name for logging
/// * `model_size_mb` - Model size for error messages
pub fn try_forced_backend(
    backend: &str,
    model_bytes: &[u8],
    model_name: &str,
    model_size_mb: f32,
) -> Result<(String, Session)> {
    let backend_upper = backend.to_uppercase();

    match backend_upper.as_str() {
        #[cfg(feature = "tensorrt")]
        "TENSORRT" => {
            info!("Forcing TensorRT backend for {}...", model_name);
            let session = Session::builder()
                .context("Failed to create session builder")?
                .with_execution_providers([TensorRTExecutionProvider::default().build()])
                .context("Failed to configure TensorRT provider")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .with_intra_threads(optimal_intra_op_threads())
                .context("Failed to configure intra-op threads")?
                .with_inter_threads(1)
                .context("Failed to configure inter-op threads")?
                .commit_from_memory(model_bytes)
                .context("Failed to load model with TensorRT")?;
            info!("✓ Forced TensorRT backend for {}", model_name);
            Ok(("TensorRT (forced)".to_string(), session))
        }
        #[cfg(not(feature = "tensorrt"))]
        "TENSORRT" => {
            anyhow::bail!("TensorRT backend not available. Rebuild with: cargo build --features tensorrt")
        }

        #[cfg(feature = "cuda")]
        "CUDA" => {
            info!("Forcing CUDA backend for {}...", model_name);
            let session = Session::builder()
                .context("Failed to create session builder")?
                .with_execution_providers([CUDAExecutionProvider::default().build()])
                .context("Failed to configure CUDA provider")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .with_intra_threads(optimal_intra_op_threads())
                .context("Failed to configure intra-op threads")?
                .with_inter_threads(1)
                .context("Failed to configure inter-op threads")?
                .commit_from_memory(model_bytes)
                .context("Failed to load model with CUDA")?;
            info!("✓ Forced CUDA backend for {}", model_name);
            Ok(("CUDA (forced)".to_string(), session))
        }
        #[cfg(not(feature = "cuda"))]
        "CUDA" => {
            anyhow::bail!("CUDA backend not available. Rebuild with: cargo build --features cuda")
        }

        #[cfg(feature = "openvino")]
        "OPENVINO" => {
            info!("Forcing OpenVINO backend for {}...", model_name);
            let session = Session::builder()
                .context("Failed to create session builder")?
                .with_execution_providers([
                    OpenVINOExecutionProvider::default()
                        .with_device_type("CPU")
                        .build()
                ])
                .context("Failed to configure OpenVINO provider")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .with_intra_threads(optimal_intra_op_threads())
                .context("Failed to configure intra-op threads")?
                .with_inter_threads(1)
                .context("Failed to configure inter-op threads")?
                .commit_from_memory(model_bytes)
                .context("Failed to load model with OpenVINO")?;
            info!("✓ Forced OpenVINO backend for {}", model_name);
            Ok(("OpenVINO-CPU (forced)".to_string(), session))
        }
        #[cfg(not(feature = "openvino"))]
        "OPENVINO" => {
            anyhow::bail!("OpenVINO backend not available. Rebuild with: cargo build --features openvino")
        }

        #[cfg(all(target_os = "windows", feature = "directml"))]
        "DIRECTML" => {
            info!("Forcing DirectML backend for {}...", model_name);
            let session = Session::builder()
                .context("Failed to create session builder")?
                .with_execution_providers([DirectMLExecutionProvider::default().build()])
                .context("Failed to configure DirectML provider")?
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?
                .with_optimization_level(GraphOptimizationLevel::Level2)
                .context("Failed to set optimization level")?
                .with_intra_threads(optimal_intra_op_threads())
                .context("Failed to configure intra-op threads")?
                .with_inter_threads(1)
                .context("Failed to configure inter-op threads")?
                .commit_from_memory(model_bytes)
                .context("Failed to load model with DirectML")?;
            info!("✓ Forced DirectML backend for {} (GPU-only)", model_name);
            Ok(("DirectML (forced)".to_string(), session))
        }
        #[cfg(not(all(target_os = "windows", feature = "directml")))]
        "DIRECTML" => {
            anyhow::bail!("DirectML backend not available. Rebuild with: cargo build --features directml (Windows only)")
        }

        #[cfg(all(target_os = "macos", feature = "coreml"))]
        "COREML" => {
            info!("Forcing CoreML backend for {}...", model_name);
            let session = Session::builder()
                .context("Failed to create session builder")?
                .with_execution_providers([CoreMLExecutionProvider::default().build()])
                .context("Failed to configure CoreML provider")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .with_intra_threads(optimal_intra_op_threads())
                .context("Failed to configure intra-op threads")?
                .with_inter_threads(1)
                .context("Failed to configure inter-op threads")?
                .commit_from_memory(model_bytes)
                .context("Failed to load model with CoreML")?;
            info!("✓ Forced CoreML backend for {}", model_name);
            Ok(("CoreML (forced)".to_string(), session))
        }
        #[cfg(not(all(target_os = "macos", feature = "coreml")))]
        "COREML" => {
            anyhow::bail!("CoreML backend not available. Rebuild with: cargo build --features coreml (macOS only)")
        }

        #[cfg(feature = "xnnpack")]
        "XNNPACK" => {
            info!("Forcing XNNPACK backend for {}...", model_name);
            let session = Session::builder()
                .context("Failed to create session builder")?
                .with_execution_providers([XNNPACKExecutionProvider::default().build()])
                .context("Failed to configure XNNPACK provider")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .with_intra_threads(optimal_intra_op_threads())
                .context("Failed to configure intra-op threads")?
                .with_inter_threads(1)
                .context("Failed to configure inter-op threads")?
                .commit_from_memory(model_bytes)
                .context("Failed to load model with XNNPACK")?;
            info!("✓ Forced XNNPACK backend for {}", model_name);
            Ok(("XNNPACK (forced)".to_string(), session))
        }
        #[cfg(not(feature = "xnnpack"))]
        "XNNPACK" => {
            anyhow::bail!("XNNPACK backend not available. Rebuild with: cargo build --features xnnpack")
        }

        "CPU" => {
            info!("Forcing CPU backend for {}...", model_name);
            let session = Session::builder()
                .context("Failed to create session builder")?
                .with_execution_providers([CPUExecutionProvider::default().build()])
                .context("Failed to configure CPU provider")?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .context("Failed to set optimization level")?
                .with_intra_threads(optimal_intra_op_threads())
                .context("Failed to configure intra-op threads")?
                .with_inter_threads(1)
                .context("Failed to configure inter-op threads")?
                .commit_from_memory(model_bytes)
                .context("Failed to load model with CPU")?;
            info!("✓ Forced CPU backend for {}", model_name);
            Ok(("CPU (forced)".to_string(), session))
        }

        "AUTO" => {
            // Explicit AUTO means use auto-detection
            build_session_with_acceleration(model_bytes, model_name, model_size_mb)
        }

        _ => {
            anyhow::bail!(
                "Unknown inference backend '{}'. \
                Valid options: TENSORRT, CUDA, OPENVINO, DIRECTML, COREML, XNNPACK, CPU, AUTO",
                backend
            )
        }
    }
}

/// Build a CPU-only session (for lightweight models like text_cleaner, OCR)
///
/// Uses minimal threads for small models where parallelism overhead isn't worth it.
pub fn build_cpu_session(model_bytes: &[u8], model_name: &str) -> Result<Session> {
    Session::builder()
        .context(format!("Failed to create session builder for {}", model_name))?
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .context(format!("Failed to configure CPU provider for {}", model_name))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .context(format!("Failed to set optimization level for {}", model_name))?
        .with_intra_threads(1)
        .context(format!("Failed to configure intra-op threads for {}", model_name))?
        .with_inter_threads(1)
        .context(format!("Failed to configure inter-op threads for {}", model_name))?
        .commit_from_memory(model_bytes)
        .context(format!("Failed to load {} ONNX model", model_name))
}
