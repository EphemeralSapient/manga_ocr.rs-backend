pub mod detection;
pub mod font_manager;
pub mod ocr;  // Local OCR service for Japanese text recognition
pub mod onnx_builder; // Shared ONNX session builder (eliminates ~280 lines of duplication)
pub mod rendering;
pub mod segmentation;
pub mod translation;

// Re-export commonly used services
pub use detection::DetectionService;
pub use ocr::{get_ocr_service, is_ocr_available, warmup_ocr_service, OcrService};
pub use rendering::CosmicTextRenderer;
pub use segmentation::SegmentationService;
pub use translation::{ApiClient, TranslationCache};
