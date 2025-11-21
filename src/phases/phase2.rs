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
    /// # Arguments:
    /// * `ocr_model_override` - Optional OCR/translation model name override
    /// * `banana_model_override` - Optional banana mode image model name override
    /// * `banana_mode` - Whether to use banana mode for complex backgrounds
    /// * `cache_enabled` - Whether to use translation cache
    /// * `target_language` - Optional target language for translation (defaults to "English")
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
        ocr_model_override: Option<&str>,
        banana_model_override: Option<&str>,
        banana_mode: bool,
        cache_enabled: bool,
        target_language: Option<&str>,
    ) -> Result<Phase2Output> {
        debug!(
            "Phase 2: Processing {} regions for page {}",
            phase1_output.regions.len(),
            phase1_output.page_index
        );

        // Use pre-decoded image if available, otherwise load from bytes
        // OPTIMIZATION: Pre-decoded image eliminates redundant decoding across phases
        // Use Arc reference instead of cloning the entire image (saves ~8MB per phase!)
        let img_owned;
        let img: &DynamicImage = if let Some(ref decoded) = image_data.decoded_image {
            decoded.as_ref()
        } else {
            img_owned = image::load_from_memory(&image_data.image_bytes)
                .context("Failed to load image")?;
            &img_owned
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

        // Process simple and complex backgrounds IN PARALLEL for maximum performance
        let (simple_bg_translations, complex_bg_result) = tokio::join!(
            // Process simple backgrounds (always use OCR/translation)
            async {
                if !simple_regions.is_empty() {
                    self.process_simple_backgrounds(&img, source_image_hash, &simple_regions, ocr_model_override, cache_enabled, target_language)
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
                        self.process_complex_banana(&img, &complex_regions, banana_model_override, target_language)
                            .await
                            .map(|bananas| (bananas, Vec::new()))
                            .context("Failed to process complex backgrounds with banana")
                    } else {
                        // Use OCR/translation (all regions in single call)
                        self.process_simple_backgrounds(&img, source_image_hash, &complex_regions, ocr_model_override, cache_enabled, target_language)
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

    /// Process simple backgrounds with OCR/translation API (ALL regions in single call)
    ///
    /// # Arguments:
    /// * `img` - Source image
    /// * `source_image_hash` - xxHash3 of the source image bytes (for cache keys)
    /// * `regions` - Regions to process
    /// * `model_override` - Optional model name to override the default from config
    /// * `cache_enabled` - Whether to use translation cache
    /// * `target_language` - Optional target language for translation (defaults to "English")
    ///
    /// # Returns:
    /// Vector of (region_id, OCRTranslation)
    async fn process_simple_backgrounds(
        &self,
        img: &DynamicImage,
        source_image_hash: u64,
        regions: &[&CategorizedRegion],
        model_override: Option<&str>,
        cache_enabled: bool,
        target_language: Option<&str>,
    ) -> Result<Vec<(usize, OCRTranslation)>> {
        let mut results = Vec::new();

        debug!("Processing {} regions in SINGLE API call", regions.len());

        // Check cache FIRST (before cropping/encoding) - MAJOR OPTIMIZATION
        // Generate cache keys from source image hash + bbox (no cropping needed!)
        let mut image_bytes_batch = Vec::new();
        let mut region_data: Vec<(usize, [i32; 4])> = Vec::new();

        for region in regions.iter() {
            // Generate cache key from source image hash + bbox (ultra-fast)
            let cache_key = TranslationCache::generate_key_from_source(
                source_image_hash,
                &region.bbox,
            );

            // Check cache BEFORE expensive cropping/encoding (only if cache enabled)
            if cache_enabled {
                if let Some(cached_translation) = self.cache.get(cache_key) {
                    debug!("Cache HIT for region {} (skipped crop/encode)", region.region_id);
                    results.push((region.region_id, cached_translation));
                    continue;
                }
            }

            debug!("Cache {} for region {} (will crop/encode)",
                if cache_enabled { "MISS" } else { "DISABLED" },
                region.region_id);
            region_data.push((region.region_id, region.bbox));
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

            debug!("Sending {} regions to API in single call", image_bytes_batch.len());

            // Call API for ALL uncached regions in SINGLE call
            let translations = self
                .api_client
                .ocr_translate_batch(image_bytes_batch, model_override, target_language)
                .await
                .context("OCR/translation API call failed")?;

            // Store in cache and results
            for ((region_id, bbox), translation) in region_data
                .into_iter()
                .zip(translations.into_iter())
            {
                // Use source_image_hash for cache key consistency
                if cache_enabled {
                    let cache_key = TranslationCache::generate_key_from_source(source_image_hash, &bbox);
                    self.cache.put(cache_key, &translation);
                }
                results.push((region_id, translation));
            }
        }

        Ok(results)
    }

    /// Process complex backgrounds with banana mode (1 image per call)
    ///
    /// # Arguments:
    /// * `img` - Source image
    /// * `regions` - Complex background regions
    /// * `model_override` - Optional model name to override the default from config
    /// * `target_language` - Optional target language for translation (defaults to "English")
    ///
    /// # Returns:
    /// Vector of BananaResult
    async fn process_complex_banana(
        &self,
        img: &DynamicImage,
        regions: &[&CategorizedRegion],
        model_override: Option<&str>,
        target_language: Option<&str>,
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
            let model_override = model_override.map(|s| s.to_string());
            let target_language = target_language.map(|s| s.to_string());

            // Spawn concurrent tasks for each banana call
            let task = tokio::spawn(async move {
                api_client.banana_translate(region_id, png_bytes, model_override.as_deref(), target_language.as_deref()).await
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

    /// Execute Phase 2 on multiple pages as a batch (fewer API calls)
    ///
    /// This combines all simple regions from all pages into fewer API calls,
    /// dramatically reducing API rate limit issues.
    ///
    /// # Arguments
    /// * `pages` - Vector of (ImageData, Phase1Output) pairs
    /// * `ocr_model_override` - Optional OCR/translation model name override
    /// * `banana_model_override` - Optional banana mode image model name override
    /// * `banana_mode` - Whether to use banana mode for complex backgrounds
    /// * `cache_enabled` - Whether to use translation cache
    /// * `custom_api_keys` - Optional custom API keys to use instead of config keys
    /// * `target_language` - Optional target language for translation (defaults to "English")
    /// * `reuse_factor` - Number of parallel requests per API key (default: 4, range: 1-8)
    ///
    /// # Returns
    /// Vector of Phase2Output for each page
    pub async fn execute_batch(
        &self,
        pages: &[(ImageData, Phase1Output)],
        ocr_model_override: Option<&str>,
        banana_model_override: Option<&str>,
        banana_mode: bool,
        cache_enabled: bool,
        custom_api_keys: Option<&[String]>,
        target_language: Option<&str>,
        reuse_factor: usize,
    ) -> Result<Vec<Phase2Output>> {
        if pages.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Phase 2 BATCH: Processing {} pages together", pages.len());

        // Use custom API keys if provided, otherwise use default client
        let api_client: Arc<ApiClient> = if let Some(custom_keys) = custom_api_keys {
            if !custom_keys.is_empty() {
                debug!("Using {} custom API keys from request", custom_keys.len());
                self.api_client.with_custom_keys(custom_keys.to_vec())
            } else {
                debug!("Custom API keys array is empty, falling back to config keys");
                Arc::clone(&self.api_client)
            }
        } else {
            Arc::clone(&self.api_client)
        };

        // Verify we have API keys
        if api_client.total_keys().await == 0 {
            anyhow::bail!("No API keys available. Please provide API keys in request or configure GEMINI_API_KEYS in .env");
        }

        // Collect all regions from all pages with their metadata
        // (page_index, region_id, source_image_hash, bbox, background_type, cropped_bytes)
        let mut all_simple_regions: Vec<(usize, usize, u64, [i32; 4], Vec<u8>)> = Vec::new();
        let mut all_complex_regions: Vec<(usize, usize, [i32; 4], Vec<u8>)> = Vec::new();
        let mut cached_results: Vec<(usize, usize, OCRTranslation)> = Vec::new();

        // Pre-process all pages
        for (image_data, phase1_output) in pages {
            // Load image
            let img_owned;
            let img: &DynamicImage = if let Some(ref decoded) = image_data.decoded_image {
                decoded.as_ref()
            } else {
                img_owned = image::load_from_memory(&image_data.image_bytes)
                    .context("Failed to load image")?;
                &img_owned
            };

            let source_image_hash = xxh3_64(&image_data.image_bytes);
            let page_index = phase1_output.page_index;

            for region in &phase1_output.regions {
                let is_simple = region.background_type == BackgroundType::Simple;

                // Check cache first (for simple regions or complex without banana)
                if is_simple || !banana_mode {
                    if cache_enabled {
                        let cache_key = TranslationCache::generate_key_from_source(
                            source_image_hash,
                            &region.bbox,
                        );
                        if let Some(cached_translation) = self.cache.get(cache_key) {
                            cached_results.push((page_index, region.region_id, cached_translation));
                            continue;
                        }
                    }
                }

                // Crop the region
                let [x1, y1, x2, y2] = region.bbox;
                let width = (x2 - x1).max(1) as u32;
                let height = (y2 - y1).max(1) as u32;
                let cropped = img.crop_imm(x1 as u32, y1 as u32, width, height);

                let mut png_bytes = Vec::new();
                cropped
                    .write_to(
                        &mut std::io::Cursor::new(&mut png_bytes),
                        image::ImageFormat::Png,
                    )
                    .context("Failed to encode cropped image")?;

                if is_simple || !banana_mode {
                    all_simple_regions.push((
                        page_index,
                        region.region_id,
                        source_image_hash,
                        region.bbox,
                        png_bytes,
                    ));
                } else {
                    all_complex_regions.push((
                        page_index,
                        region.region_id,
                        region.bbox,
                        png_bytes,
                    ));
                }
            }
        }

        debug!(
            "Batch collected: {} simple regions, {} complex regions, {} cache hits",
            all_simple_regions.len(),
            all_complex_regions.len(),
            cached_results.len()
        );

        // Process all simple regions SPLIT ACROSS API KEYS with REUSE_FACTOR for parallelism
        let simple_translations = if !all_simple_regions.is_empty() {
            // Get number of available API keys
            let num_keys = api_client.total_keys().await.max(1);

            debug!(
                "🔑 Phase 2: {} regions, {} keys, reuse_factor={}",
                all_simple_regions.len(),
                num_keys,
                reuse_factor
            );

            // Step 1: Split regions into chunks for each API key
            let chunk_size = (all_simple_regions.len() + num_keys - 1) / num_keys; // Round up
            let key_chunks: Vec<Vec<(usize, usize, u64, [i32; 4], Vec<u8>)>> = all_simple_regions
                .chunks(chunk_size)
                .map(|chunk| chunk.to_vec())
                .collect();

            // Step 2: Further split each key's chunk by reuse_factor
            let mut all_sub_chunks: Vec<(usize, Vec<(usize, usize, u64, [i32; 4], Vec<u8>)>)> = Vec::new();

            for (key_idx, key_chunk) in key_chunks.into_iter().enumerate() {
                if key_chunk.is_empty() {
                    continue;
                }

                // Calculate sub-chunk size for this key
                let sub_chunk_size = (key_chunk.len() + reuse_factor - 1) / reuse_factor; // Round up
                let sub_chunks: Vec<Vec<(usize, usize, u64, [i32; 4], Vec<u8>)>> = key_chunk
                    .chunks(sub_chunk_size)
                    .map(|sub| sub.to_vec())
                    .collect();

                debug!(
                    "  Key {}: {} regions → {} sub-chunks of ~{} regions each",
                    key_idx,
                    key_chunk.len(),
                    sub_chunks.len(),
                    sub_chunk_size
                );

                // Store with key index for pinning
                for sub_chunk in sub_chunks {
                    all_sub_chunks.push((key_idx, sub_chunk));
                }
            }

            debug!(
                "✓ Created {} total API calls ({} keys × {} reuse_factor)",
                all_sub_chunks.len(),
                num_keys,
                reuse_factor
            );

            // Step 3: Process all sub-chunks in parallel with pinned keys
            let target_language_str = target_language.map(|s| s.to_string());
            let tasks: Vec<_> = all_sub_chunks
                .into_iter()
                .map(|(key_idx, sub_chunk)| {
                    let api_client = Arc::clone(&api_client);
                    let model_override = ocr_model_override.map(|s| s.to_string());
                    let target_language = target_language_str.clone();

                    tokio::spawn(async move {
                        let image_bytes: Vec<Vec<u8>> = sub_chunk
                            .iter()
                            .map(|(_, _, _, _, bytes)| bytes.clone())
                            .collect();

                        // Use pinned key for this sub-chunk
                        let translations = api_client
                            .ocr_translate_batch_with_key(image_bytes, model_override.as_deref(), target_language.as_deref(), key_idx)
                            .await?;

                        Ok::<_, anyhow::Error>((sub_chunk, translations))
                    })
                })
                .collect();

            // Wait for all parallel API calls
            let results = try_join_all(tasks)
                .await
                .context("One or more API tasks panicked")?
                .into_iter()
                .collect::<Result<Vec<_>>>()
                .context("One or more API calls failed")?;

            // Flatten results and cache
            let mut all_translations = Vec::new();
            for (chunk, translations) in results {
                for ((page_index, region_id, source_hash, bbox, _), translation) in
                    chunk.iter().zip(translations.iter())
                {
                    if cache_enabled {
                        let cache_key = TranslationCache::generate_key_from_source(*source_hash, bbox);
                        self.cache.put(cache_key, translation);
                    }
                    all_translations.push((*page_index, *region_id, translation.clone()));
                }
            }

            all_translations
        } else {
            Vec::new()
        };

        // Process complex regions with banana mode (concurrent per-region calls)
        let banana_results = if !all_complex_regions.is_empty() && banana_mode {
            let target_language_str = target_language.map(|s| s.to_string());
            let tasks: Vec<_> = all_complex_regions
                .into_iter()
                .map(|(page_index, region_id, _bbox, png_bytes)| {
                    let api_client = Arc::clone(&api_client);
                    let model_override = banana_model_override.map(|s| s.to_string());
                    let target_language = target_language_str.clone();

                    tokio::spawn(async move {
                        let result = api_client
                            .banana_translate(region_id, png_bytes, model_override.as_deref(), target_language.as_deref())
                            .await?;
                        Ok::<_, anyhow::Error>((page_index, result))
                    })
                })
                .collect();

            let results = try_join_all(tasks)
                .await
                .context("One or more banana tasks panicked")?
                .into_iter()
                .collect::<Result<Vec<_>>>()
                .context("One or more banana API calls failed")?;

            results
        } else {
            Vec::new()
        };

        // Assemble Phase2Output for each page
        let mut outputs: Vec<Phase2Output> = pages
            .iter()
            .map(|(_, phase1_output)| Phase2Output {
                page_index: phase1_output.page_index,
                simple_bg_translations: Vec::new(),
                complex_bg_bananas: Vec::new(),
                complex_bg_translations: Vec::new(),
            })
            .collect();

        // Create a map from page_index to output index
        let page_to_idx: std::collections::HashMap<usize, usize> = pages
            .iter()
            .enumerate()
            .map(|(i, (_, p))| (p.page_index, i))
            .collect();

        // Distribute cached results
        for (page_index, region_id, translation) in cached_results {
            if let Some(&idx) = page_to_idx.get(&page_index) {
                outputs[idx].simple_bg_translations.push((region_id, translation));
            }
        }

        // Distribute simple translations
        for (page_index, region_id, translation) in simple_translations {
            if let Some(&idx) = page_to_idx.get(&page_index) {
                outputs[idx].simple_bg_translations.push((region_id, translation));
            }
        }

        // Distribute banana results
        for (page_index, banana_result) in banana_results {
            if let Some(&idx) = page_to_idx.get(&page_index) {
                outputs[idx].complex_bg_bananas.push(banana_result);
            }
        }

        debug!("Phase 2 BATCH: Completed processing {} pages", pages.len());

        Ok(outputs)
    }
}
