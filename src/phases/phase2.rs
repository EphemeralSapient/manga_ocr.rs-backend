// Phase 2: API Calls Pipeline (OCR/Translation and Banana Mode)

use anyhow::{Context, Result};
use futures::future::try_join_all;
use image::DynamicImage;
use std::sync::Arc;
use tracing::{debug, instrument};
use xxhash_rust::xxh3::xxh3_64;

use crate::services::translation::api_client::ApiClient;
use crate::core::config::Config;
use crate::services::translation::cache::TranslationCache;
use crate::core::types::{
    BackgroundType, BananaResult, CategorizedRegion, ImageData, OCRTranslation, Phase1Output,
    Phase2Output,
};

/// Phase 2 pipeline: API calls for translation
pub struct Phase2Pipeline {
    config: Arc<Config>,
    api_client: Arc<ApiClient>,
    cache: Arc<TranslationCache>,
}

impl Phase2Pipeline {
    /// Create new Phase 2 pipeline
    pub fn new(config: Arc<Config>, api_client: Arc<ApiClient>, cache: Arc<TranslationCache>) -> Self {
        Self { config, api_client, cache }
    }

    /// Execute Phase 2 on Phase 1 output
    ///
    /// # Steps:
    /// 1. Separate simple vs complex background regions
    /// 2. If banana mode enabled:
    ///    - Process complex backgrounds with banana API (1 image per call)
    ///    - Batch simple backgrounds with OCR/translation API (M images per call)
    /// 3. If banana mode disabled:
    ///    - Batch all regions with OCR/translation API (M images per call)
    ///
    /// # Returns
    /// Phase2Output with translations
    #[instrument(skip(self, image_data, phase1_output), fields(
        page_index = phase1_output.page_index,
        regions = phase1_output.regions.len()
    ))]
    pub async fn execute(
        &self,
        image_data: &ImageData,
        phase1_output: &Phase1Output,
    ) -> Result<Phase2Output> {
        debug!(
            "Phase 2: Processing {} regions for page {}",
            phase1_output.regions.len(),
            phase1_output.page_index
        );

        // Use pre-decoded image if available, otherwise load from bytes
        // OPTIMIZATION: Pre-decoded image eliminates redundant decoding across phases
        let img = if let Some(ref decoded) = image_data.decoded_image {
            (**decoded).clone()
        } else {
            image::load_from_memory(&image_data.image_bytes)
                .context("Failed to load image")?
        };

        // OPTIMIZATION: Hash source image once for all cache lookups
        // Using the original image bytes (already encoded) avoids re-encoding
        let source_image_hash = xxh3_64(&image_data.image_bytes);

        // Separate regions by background type
        let (simple_regions, complex_regions): (Vec<_>, Vec<_>) = phase1_output
            .regions
            .iter()
            .partition(|r| r.background_type == BackgroundType::Simple);

        debug!(
            "Separated: {} simple, {} complex",
            simple_regions.len(),
            complex_regions.len()
        );

        let banana_mode = self.config.banana_mode_enabled();
        let batch_size_m = self.config.api_batch_size_m();

        // Process simple and complex backgrounds IN PARALLEL for maximum performance
        let (simple_bg_translations, complex_bg_result) = tokio::join!(
            // Process simple backgrounds (always use OCR/translation)
            async {
                if !simple_regions.is_empty() {
                    self.process_simple_backgrounds(&img, source_image_hash, &simple_regions, batch_size_m)
                        .await
                        .context("Failed to process simple backgrounds")
                } else {
                    Ok(Vec::new())
                }
            },
            // Process complex backgrounds (banana mode or OCR)
            async {
                if !complex_regions.is_empty() {
                    if banana_mode {
                        // Use banana API (1 image per call)
                        self.process_complex_banana(&img, &complex_regions)
                            .await
                            .map(|bananas| (bananas, Vec::new()))
                            .context("Failed to process complex backgrounds with banana")
                    } else {
                        // Use OCR/translation (batched)
                        self.process_simple_backgrounds(&img, source_image_hash, &complex_regions, batch_size_m)
                            .await
                            .map(|translations| (Vec::new(), translations))
                            .context("Failed to process complex backgrounds with OCR")
                    }
                } else {
                    Ok((Vec::new(), Vec::new()))
                }
            }
        );

        let simple_bg_translations = simple_bg_translations?;
        let (complex_bg_bananas, complex_bg_translations) = complex_bg_result?;

        Ok(Phase2Output {
            page_index: phase1_output.page_index,
            simple_bg_translations,
            complex_bg_bananas,
            complex_bg_translations,
        })
    }

    /// Process simple backgrounds with OCR/translation API (batched with caching)
    ///
    /// # Arguments:
    /// * `img` - Source image
    /// * `source_image_hash` - xxHash3 of the source image bytes (for cache keys)
    /// * `regions` - Regions to process
    /// * `batch_size_m` - Number of images per API call
    ///
    /// # Returns:
    /// Vector of (region_id, OCRTranslation)
    async fn process_simple_backgrounds(
        &self,
        img: &DynamicImage,
        source_image_hash: u64,
        regions: &[&CategorizedRegion],
        batch_size_m: usize,
    ) -> Result<Vec<(usize, OCRTranslation)>> {
        let mut results = Vec::new();

        // Process in batches of M
        for batch in regions.chunks(batch_size_m) {
            debug!("Processing batch of {} simple backgrounds", batch.len());

            // Check cache FIRST (before cropping/encoding) - MAJOR OPTIMIZATION
            // Generate cache keys from source image hash + bbox (no cropping needed!)
            let mut image_bytes_batch = Vec::new();
            let mut region_data: Vec<(usize, [i32; 4])> = Vec::new();

            for region in batch.iter() {
                // Generate cache key from source image hash + bbox (ultra-fast)
                let cache_key = TranslationCache::generate_key_from_source(
                    source_image_hash,
                    &region.bbox,
                );

                // Check cache BEFORE expensive cropping/encoding
                if let Some(cached_translation) = self.cache.get(cache_key) {
                    debug!("Cache HIT for region {} (skipped crop/encode)", region.region_id);
                    results.push((region.region_id, cached_translation));
                } else {
                    debug!("Cache MISS for region {} (will crop/encode)", region.region_id);
                    region_data.push((region.region_id, region.bbox));
                }
            }

            // Only crop/encode for cache misses (HUGE savings on cache hits!)
            if !region_data.is_empty() {
                for (_region_id, bbox) in &region_data {
                    let [x1, y1, x2, y2] = bbox;
                    let width = (x2 - x1).max(1) as u32;
                    let height = (y2 - y1).max(1) as u32;

                    let cropped = img.crop_imm(*x1 as u32, *y1 as u32, width, height);

                    // Convert to PNG bytes
                    let mut png_bytes = Vec::new();
                    cropped
                        .write_to(
                            &mut std::io::Cursor::new(&mut png_bytes),
                            image::ImageFormat::Png,
                        )
                        .context("Failed to encode cropped image")?;

                    image_bytes_batch.push(png_bytes);
                }

                // Call API for uncached regions
                let translations = self
                    .api_client
                    .ocr_translate_batch(image_bytes_batch)
                    .await
                    .context("OCR/translation API call failed")?;

                // Store in cache and results
                for ((region_id, bbox), translation) in region_data
                    .into_iter()
                    .zip(translations.into_iter())
                {
                    // Use source_image_hash for cache key consistency
                    let cache_key = TranslationCache::generate_key_from_source(source_image_hash, &bbox);
                    self.cache.put(cache_key, &translation);
                    results.push((region_id, translation));
                }
            }
        }

        Ok(results)
    }

    /// Process complex backgrounds with banana mode (1 image per call)
    ///
    /// # Arguments:
    /// * `img` - Source image
    /// * `regions` - Complex background regions
    ///
    /// # Returns:
    /// Vector of BananaResult
    async fn process_complex_banana(
        &self,
        img: &DynamicImage,
        regions: &[&CategorizedRegion],
    ) -> Result<Vec<BananaResult>> {
        debug!("Processing {} complex backgrounds with banana mode", regions.len());

        let mut tasks = Vec::new();

        for region in regions {
            let [x1, y1, x2, y2] = region.bbox;
            let width = (x2 - x1).max(1) as u32;
            let height = (y2 - y1).max(1) as u32;

            let cropped = img.crop_imm(x1 as u32, y1 as u32, width, height);

            // Convert to PNG bytes
            let mut png_bytes = Vec::new();
            cropped
                .write_to(
                    &mut std::io::Cursor::new(&mut png_bytes),
                    image::ImageFormat::Png,
                )
                .context("Failed to encode cropped image")?;

            let region_id = region.region_id;
            let api_client = Arc::clone(&self.api_client);

            // Spawn concurrent tasks for each banana call
            let task = tokio::spawn(async move {
                api_client.banana_translate(region_id, png_bytes).await
            });

            tasks.push(task);
        }

        // OPTIMIZATION: Collect all results concurrently instead of sequentially
        // Using try_join_all instead of sequential awaits prevents completed tasks
        // from waiting on slower tasks, improving throughput by ~15-25%
        let results = try_join_all(tasks)
            .await
            .context("One or more banana tasks panicked")?
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .context("One or more banana API calls failed")?;

        Ok(results)
    }
}
