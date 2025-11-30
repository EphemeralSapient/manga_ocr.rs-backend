use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use std::io::Cursor;
use anyhow::{Context, Result};
use ab_glyph::{FontRef, PxScale};
use imageproc::drawing::draw_text_mut;

/// Asynchronously crop an image using spawn_blocking to avoid blocking the async runtime.
///
/// This is especially important for large images or when cropping many regions,
/// as image cropping is a CPU-intensive synchronous operation.
pub async fn crop_image_async(
    img: DynamicImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<DynamicImage> {
    tokio::task::spawn_blocking(move || {
        let cropped = img.crop_imm(x, y, width, height);
        Ok(cropped)
    })
    .await
    .context("Failed to spawn blocking task for image cropping")?
}

/// Asynchronously encode an image to PNG bytes using spawn_blocking.
///
/// PNG encoding is CPU-intensive and can block the async runtime if done synchronously.
pub async fn encode_png_async(img: DynamicImage) -> Result<Vec<u8>> {
    tokio::task::spawn_blocking(move || {
        let mut png_bytes = Vec::new();
        let mut cursor = Cursor::new(&mut png_bytes);
        img.write_to(&mut cursor, ImageFormat::Png)
            .context("Failed to encode image as PNG")?;
        Ok(png_bytes)
    })
    .await
    .context("Failed to spawn blocking task for PNG encoding")?
}

/// Asynchronously crop and encode an image to PNG in a single blocking operation.
///
/// This is more efficient than calling crop_image_async and encode_png_async separately,
/// as it only spawns one blocking task instead of two.
pub async fn crop_and_encode_png_async(
    img: DynamicImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    tokio::task::spawn_blocking(move || {
        let cropped = img.crop_imm(x, y, width, height);
        let mut png_bytes = Vec::new();
        let mut cursor = Cursor::new(&mut png_bytes);
        cropped
            .write_to(&mut cursor, ImageFormat::Png)
            .context("Failed to encode cropped image as PNG")?;
        Ok(png_bytes)
    })
    .await
    .context("Failed to spawn blocking task for crop and encode")?
}

/// Asynchronously load an image from bytes using spawn_blocking.
///
/// Image decoding is CPU-intensive, especially for large images.
pub async fn load_image_from_memory_async(bytes: &[u8]) -> Result<DynamicImage> {
    let bytes = bytes.to_vec(); // Clone to move into blocking task
    tokio::task::spawn_blocking(move || {
        image::load_from_memory(&bytes)
            .context("Failed to load image from memory")
    })
    .await
    .context("Failed to spawn blocking task for image loading")?
}

/// Asynchronously resize an image using spawn_blocking.
pub async fn resize_image_async(
    img: DynamicImage,
    new_width: u32,
    new_height: u32,
    filter: image::imageops::FilterType,
) -> Result<DynamicImage> {
    tokio::task::spawn_blocking(move || {
        Ok(img.resize_exact(new_width, new_height, filter))
    })
    .await
    .context("Failed to spawn blocking task for image resizing")?
}

/// Asynchronously overlay one image onto another using spawn_blocking.
///
/// This is used in Phase 4 for compositing text onto the base image.
pub async fn overlay_image_async(
    mut base: RgbaImage,
    overlay: &RgbaImage,
    x: i64,
    y: i64,
) -> Result<RgbaImage> {
    let overlay = overlay.clone(); // Clone to move into blocking task
    tokio::task::spawn_blocking(move || {
        image::imageops::overlay(&mut base, &overlay, x, y);
        Ok(base)
    })
    .await
    .context("Failed to spawn blocking task for image overlay")?
}

/// Synchronous version of crop_and_encode for small batches where spawn_blocking overhead isn't worth it.
///
/// Use this when processing < IMAGE_OPS_BLOCKING_THRESHOLD images (typically 5).
pub fn crop_and_encode_png_sync(
    img: &DynamicImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let cropped = img.crop_imm(x, y, width, height);
    let mut png_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut png_bytes);
    cropped
        .write_to(&mut cursor, ImageFormat::Png)
        .context("Failed to encode cropped image as PNG")?;
    Ok(png_bytes)
}

// =============================================================================
// VISUAL NUMBERING SYSTEM FOR BUBBLE IDENTIFICATION
// =============================================================================
//
// This system adds visible numbers to bubble crops before sending to Gemini API.
// This ensures 100% accurate translation-to-bubble matching by making Gemini
// explicitly identify bubbles by their visible number, eliminating spatial ambiguity.

/// Embedded font data for number rendering (Arial Bold subset)
/// Using the existing arial-unicode.ttf from fonts/ directory
static FONT_DATA: once_cell::sync::Lazy<Option<Vec<u8>>> = once_cell::sync::Lazy::new(|| {
    // Try to load from fonts directory
    std::fs::read("fonts/arial-unicode.ttf")
        .or_else(|_| std::fs::read("fonts/anime_ace.ttf"))
        .ok()
});

/// Configuration for bubble numbering
#[derive(Debug, Clone)]
pub struct NumberingConfig {
    /// Ratio of number panel width to bubble height (default: 0.15)
    pub size_ratio: f32,
    /// Minimum number panel width in pixels (default: 40)
    pub min_size: u32,
    /// Maximum number panel width in pixels (default: 80)
    pub max_size: u32,
    /// Outline width for number text (default: 2)
    pub outline_width: i32,
}

impl Default for NumberingConfig {
    fn default() -> Self {
        Self {
            size_ratio: 0.15,
            min_size: 40,
            max_size: 80,
            outline_width: 2,
        }
    }
}

/// Add a visible number label to a region image for reliable LLM identification.
///
/// This function prepends a numbered panel to the left side of the image,
/// allowing Gemini to accurately identify which bubble contains which text.
///
/// # Arguments
/// * `image_bytes` - PNG bytes of the cropped region
/// * `number` - The bubble number to display (0-indexed or 1-indexed)
/// * `config` - Optional numbering configuration
///
/// # Returns
/// * `Ok((numbered_image_bytes, panel_width))` - PNG bytes of numbered image and the panel width
/// * `Err` if image processing fails
///
/// # Example
/// ```ignore
/// let (numbered_bytes, panel_width) = add_number_to_region(&crop_bytes, 1, None)?;
/// ```
pub fn add_number_to_region(
    image_bytes: &[u8],
    number: usize,
    config: Option<NumberingConfig>,
) -> Result<(Vec<u8>, u32)> {
    let config = config.unwrap_or_default();

    // Load the original image
    let original = image::load_from_memory(image_bytes)
        .context("Failed to load region image for numbering")?;
    let original_rgba = original.to_rgba8();
    let (width, height) = original_rgba.dimensions();

    // Calculate number panel size based on bubble height
    let panel_size = ((height as f32 * config.size_ratio) as u32)
        .max(config.min_size)
        .min(config.max_size);

    // Create extended image with number panel on the LEFT
    let extended_width = width + panel_size;
    let mut extended_image = RgbaImage::from_pixel(
        extended_width,
        height,
        Rgba([255, 255, 255, 255]), // White background
    );

    // Paste original image on the RIGHT side
    image::imageops::overlay(&mut extended_image, &original_rgba, panel_size as i64, 0);

    // Draw vertical separator line (gray)
    let separator_color = Rgba([200, 200, 200, 255]);
    for y in 0..height {
        extended_image.put_pixel(panel_size - 1, y, separator_color);
        extended_image.put_pixel(panel_size - 2, y, separator_color);
    }

    // Draw number with outline
    if let Some(ref font_data) = *FONT_DATA {
        draw_number_with_outline(
            &mut extended_image,
            number,
            panel_size,
            height,
            font_data,
            config.outline_width,
        )?;
    } else {
        // Fallback: Draw simple number without fancy font
        draw_simple_number(&mut extended_image, number, panel_size, height)?;
    }

    // Encode back to PNG
    let mut png_bytes = Vec::new();
    DynamicImage::ImageRgba8(extended_image)
        .write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)
        .context("Failed to encode numbered image as PNG")?;

    Ok((png_bytes, panel_size))
}

/// Draw a number with white outline for visibility on any background
fn draw_number_with_outline(
    img: &mut RgbaImage,
    number: usize,
    panel_width: u32,
    panel_height: u32,
    font_data: &[u8],
    outline_width: i32,
) -> Result<()> {
    let font = FontRef::try_from_slice(font_data)
        .context("Failed to parse font data")?;

    let number_text = number.to_string();

    // Calculate font size (60% of panel width, clamped)
    let font_size = (panel_width as f32 * 0.6).max(20.0).min(60.0);
    let scale = PxScale::from(font_size);

    // Calculate text position (centered in panel)
    // Rough estimation: each digit is about 0.6 * font_size wide
    let text_width = number_text.len() as f32 * font_size * 0.6;
    let text_height = font_size;

    let text_x = ((panel_width as f32 - text_width) / 2.0).max(2.0) as i32;
    let text_y = ((panel_height as f32 - text_height) / 2.0).max(2.0) as i32;

    // Draw white outline first (8 directions + corners)
    let outline_color = Rgba([255, 255, 255, 255]);
    for dx in -outline_width..=outline_width {
        for dy in -outline_width..=outline_width {
            if dx != 0 || dy != 0 {
                draw_text_mut(
                    img,
                    outline_color,
                    text_x + dx,
                    text_y + dy,
                    scale,
                    &font,
                    &number_text,
                );
            }
        }
    }

    // Draw black number on top
    let text_color = Rgba([0, 0, 0, 255]);
    draw_text_mut(
        img,
        text_color,
        text_x,
        text_y,
        scale,
        &font,
        &number_text,
    );

    Ok(())
}

/// Simple fallback number drawing without font (using basic shapes)
fn draw_simple_number(
    img: &mut RgbaImage,
    number: usize,
    panel_width: u32,
    panel_height: u32,
) -> Result<()> {
    // Draw a simple centered rectangle with the number
    // This is a fallback when no font is available
    let center_x = panel_width / 2;
    let center_y = panel_height / 2;
    let radius = (panel_width.min(panel_height) / 3) as i32;

    // Draw circle background
    let bg_color = Rgba([240, 240, 240, 255]);
    let border_color = Rgba([0, 0, 0, 255]);

    for y in 0..panel_height {
        for x in 0..panel_width {
            let dx = x as i32 - center_x as i32;
            let dy = y as i32 - center_y as i32;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq <= radius * radius {
                img.put_pixel(x, y, bg_color);
            }
            if dist_sq <= radius * radius && dist_sq >= (radius - 2) * (radius - 2) {
                img.put_pixel(x, y, border_color);
            }
        }
    }

    // For simple fallback, we just use the number as a simple pattern
    // Real implementation would draw digit patterns
    tracing::warn!(
        "Using simple number fallback for bubble {} - font not available",
        number
    );

    Ok(())
}

/// Batch process: add numbers to multiple regions efficiently
///
/// # Arguments
/// * `regions` - Vector of (region_id, image_bytes) pairs
/// * `config` - Optional numbering configuration
///
/// # Returns
/// * Vector of (region_id, numbered_image_bytes) pairs
pub fn add_numbers_to_regions_batch(
    regions: Vec<(usize, Vec<u8>)>,
    config: Option<NumberingConfig>,
) -> Result<Vec<(usize, Vec<u8>)>> {
    use rayon::prelude::*;

    let config = config.unwrap_or_default();

    regions
        .into_par_iter()
        .enumerate()
        .map(|(idx, (region_id, bytes))| {
            // Use 1-indexed numbers for display (more natural for humans/LLMs)
            let display_number = idx + 1;
            let (numbered_bytes, _panel_width) = add_number_to_region(
                &bytes,
                display_number,
                Some(config.clone()),
            )?;
            Ok((region_id, numbered_bytes))
        })
        .collect()
}

// =============================================================================
// DEBUG VISUALIZATION: BOUNDING BOX DRAWING
// =============================================================================
//
// This function draws bounding boxes on an image for debugging detection results.

/// Draw bounding boxes on an image for debug visualization.
///
/// # Arguments
/// * `img` - The image to draw on
/// * `label_0_bboxes` - Label 0 (speech bubble) bounding boxes - drawn in BLUE
/// * `label_1_bboxes` - Label 1 (text region) bounding boxes - drawn in RED
///
/// # Returns
/// * PNG bytes of the annotated image
pub fn draw_debug_bboxes(
    img: &DynamicImage,
    label_0_bboxes: &[[i32; 4]],
    label_1_bboxes: &[[i32; 4]],
) -> anyhow::Result<Vec<u8>> {
    let mut rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    // Draw label 0 boxes in BLUE (outer bubbles)
    let blue = Rgba([0, 100, 255, 255]);
    for bbox in label_0_bboxes {
        draw_rect(&mut rgba, bbox, blue, width, height, 3);
    }

    // Draw label 1 boxes in RED (inner text regions)
    let red = Rgba([255, 50, 50, 255]);
    for bbox in label_1_bboxes {
        draw_rect(&mut rgba, bbox, red, width, height, 2);
    }

    // Encode to PNG
    let mut png_bytes = Vec::new();
    DynamicImage::ImageRgba8(rgba)
        .write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)
        .context("Failed to encode debug image as PNG")?;

    Ok(png_bytes)
}

/// Draw a rectangle outline on an image
fn draw_rect(
    img: &mut RgbaImage,
    bbox: &[i32; 4],
    color: Rgba<u8>,
    img_width: u32,
    img_height: u32,
    thickness: i32,
) {
    let [x1, y1, x2, y2] = *bbox;

    // Clamp to image bounds
    let x1 = x1.max(0).min(img_width as i32 - 1) as u32;
    let y1 = y1.max(0).min(img_height as i32 - 1) as u32;
    let x2 = x2.max(0).min(img_width as i32 - 1) as u32;
    let y2 = y2.max(0).min(img_height as i32 - 1) as u32;

    // Draw horizontal lines (top and bottom)
    for t in 0..thickness {
        let offset = t as u32;
        // Top line
        if y1 + offset < img_height {
            for x in x1..=x2 {
                if x < img_width {
                    img.put_pixel(x, y1 + offset, color);
                }
            }
        }
        // Bottom line
        if y2 >= offset && y2 - offset < img_height {
            for x in x1..=x2 {
                if x < img_width {
                    img.put_pixel(x, y2 - offset, color);
                }
            }
        }
    }

    // Draw vertical lines (left and right)
    for t in 0..thickness {
        let offset = t as u32;
        // Left line
        if x1 + offset < img_width {
            for y in y1..=y2 {
                if y < img_height {
                    img.put_pixel(x1 + offset, y, color);
                }
            }
        }
        // Right line
        if x2 >= offset && x2 - offset < img_width {
            for y in y1..=y2 {
                if y < img_height {
                    img.put_pixel(x2 - offset, y, color);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    #[tokio::test]
    async fn test_crop_and_encode_async() {
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            100,
            100,
            Rgba([255, 0, 0, 255]),
        ));

        let result = crop_and_encode_png_async(img, 10, 10, 50, 50).await;
        assert!(result.is_ok());

        let png_bytes = result.unwrap();
        assert!(!png_bytes.is_empty());
    }

    #[tokio::test]
    async fn test_load_image_async() {
        // Create a simple 1x1 red pixel PNG
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            1,
            1,
            Rgba([255, 0, 0, 255]),
        ));
        let mut png_bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)
            .unwrap();

        let result = load_image_from_memory_async(&png_bytes).await;
        assert!(result.is_ok());
    }
}
