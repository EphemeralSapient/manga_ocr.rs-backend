// Phase 2: API Calls Pipeline (OCR/Translation and Banana Mode)

use anyhow::{Context, Result};
use image::DynamicImage;
use std::sync::Arc;
use tracing::{debug, instrument};

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

        // Load image
        let img = image::load_from_memory(&image_data.image_bytes)
            .context("Failed to load image")?;

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
                    self.process_simple_backgrounds(&img, &simple_regions, batch_size_m)
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
                        self.process_simple_backgrounds(&img, &complex_regions, batch_size_m)
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
    /// * `regions` - Regions to process
    /// * `batch_size_m` - Number of images per API call
    ///
    /// # Returns:
    /// Vector of (region_id, OCRTranslation, cache_hit)
    async fn process_simple_backgrounds(
        &self,
        img: &DynamicImage,
        regions: &[&CategorizedRegion],
        batch_size_m: usize,
    ) -> Result<Vec<(String, OCRTranslation)>> {
        let mut results = Vec::new();

        // Process in batches of M
        for batch in regions.chunks(batch_size_m) {
            debug!("Processing batch of {} simple backgrounds", batch.len());

            // Extract image crops and check cache
            let mut image_bytes_batch = Vec::new();
            let mut region_data: Vec<(String, Vec<u8>, [i32; 4])> = Vec::new();

            for region in batch.iter() {
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

                // Generate cache key
                let cache_key = TranslationCache::generate_key(&png_bytes, &region.bbox);

                // Check cache
                if let Some(cached_translation) = self.cache.get(&cache_key) {
                    debug!("Cache HIT for region {}", region.region_id);
                    results.push((region.region_id.clone(), cached_translation));
                } else {
                    debug!("Cache MISS for region {}", region.region_id);
                    image_bytes_batch.push(png_bytes.clone());
                    region_data.push((region.region_id.clone(), png_bytes, region.bbox));
                }
            }

            // Call API only for uncached regions
            if !image_bytes_batch.is_empty() {
                let translations = self
                    .api_client
                    .ocr_translate_batch(image_bytes_batch)
                    .await
                    .context("OCR/translation API call failed")?;

                // Store in cache and results
                for ((region_id, png_bytes, bbox), translation) in
                    region_data.into_iter().zip(translations.into_iter())
                {
                    let cache_key = TranslationCache::generate_key(&png_bytes, &bbox);
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

            let region_id = region.region_id.clone();
            let api_client = Arc::clone(&self.api_client);

            // Spawn concurrent tasks for each banana call
            let task = tokio::spawn(async move {
                api_client.banana_translate(region_id, png_bytes).await
            });

            tasks.push(task);
        }

        // Collect all results
        let mut results = Vec::new();
        for task in tasks {
            let result = task
                .await
                .context("Banana task panicked")?
                .context("Banana API call failed")?;
            results.push(result);
        }

        Ok(results)
    }
}
