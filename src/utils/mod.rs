pub mod image_ops;
pub mod metrics;

// Re-export commonly used items
pub use image_ops::{
    crop_and_encode_png_async,
    crop_and_encode_png_sync,
    crop_image_async,
    encode_png_async,
    load_image_from_memory_async,
    overlay_image_async,
    resize_image_async,
    // Visual numbering system for bubble identification
    add_number_to_region,
    add_numbers_to_regions_batch,
    NumberingConfig,
};
pub use metrics::Metrics;
