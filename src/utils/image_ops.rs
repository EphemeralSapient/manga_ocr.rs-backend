use image::{DynamicImage, ImageFormat, RgbaImage};
use std::io::Cursor;
use anyhow::{Context, Result};

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
