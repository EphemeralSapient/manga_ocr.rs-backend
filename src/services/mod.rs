pub mod detection;
pub mod onnx_builder; // Shared ONNX session builder (eliminates ~280 lines of duplication)
pub mod rendering;
pub mod segmentation;
pub mod translation;

// Re-export commonly used services
pub use detection::DetectionService;
pub use rendering::CosmicTextRenderer;
pub use segmentation::SegmentationService;
pub use translation::{ApiClient, TranslationCache};
