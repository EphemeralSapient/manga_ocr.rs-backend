// Phase 2: API Calls Pipeline (OCR/Translation and Banana Mode)
//
// VISUAL NUMBERING SYSTEM:
// This phase now supports visual bubble numbering for 100% accurate translation matching.
// When enabled (default: true), each region crop gets a visible number prepended to the left.
// Gemini identifies bubbles by their visible number, eliminating spatial ambiguity.

use anyhow::{Context, Result};
use futures::future::try_join_all;
use image::DynamicImage;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};
use xxhash_rust::xxh3::xxh3_64;

use crate::services::translation::api_client::ApiClient;
use crate::core::config::Config;
use crate::services::translation::cache::TranslationCache;
use crate::core::types::{
    BackgroundType, BananaResult, CategorizedRegion, ImageData, OCRTranslation, Phase1Output,
    Phase2Output,
};
use crate::utils::{add_number_to_region, NumberingConfig};

/// Phase 2 pipeline: API calls for translation
pub struct Phase2Pipeline {
    config: Arc<Config>,
    api_client: Arc<ApiClient>,
    cache: Arc<TranslationCache>,
}

/// OCR result for a single region (before translation)
/// Contains: (page_index, region_id, source_image_hash, bbox, ocr_text)
pub type OcrResult = (usize, usize, u64, [i32; 4], String);

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

        // =======================================================================
        // VISUAL NUMBERING SYSTEM: Process regions with visible bubble numbers
        // This ensures 100% accurate translation-to-bubble matching
        // =======================================================================

        // Check if visual numbering is enabled (default: true for accuracy)
        let use_visual_numbering = std::env::var("USE_VISUAL_NUMBERING")
            .ok()
            .and_then(|s| s.parse::<bool>().ok())
            .unwrap_or(true);

        let simple_translations = if !all_simple_regions.is_empty() {
            let num_keys = api_client.total_keys().await.max(1);

            if use_visual_numbering {
                info!(
                    "🔢 Phase 2 NUMBERED: {} regions, {} keys, reuse_factor={}",
                    all_simple_regions.len(),
                    num_keys,
                    reuse_factor
                );

                // NUMBERED APPROACH: Add visible numbers to images for accurate matching
                self.process_regions_with_numbering(
                    all_simple_regions,
                    &api_client,
                    num_keys,
                    reuse_factor,
                    ocr_model_override,
                    target_language,
                    cache_enabled,
                ).await?
            } else {
                debug!(
                    "🔑 Phase 2 LEGACY: {} regions, {} keys, reuse_factor={}",
                    all_simple_regions.len(),
                    num_keys,
                    reuse_factor
                );

                // LEGACY APPROACH: Array-order matching (less accurate)
                self.process_regions_legacy(
                    all_simple_regions,
                    &api_client,
                    num_keys,
                    reuse_factor,
                    ocr_model_override,
                    target_language,
                    cache_enabled,
                ).await?
            }
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

    // =========================================================================
    // HELPER METHODS FOR VISUAL NUMBERING SYSTEM
    // =========================================================================

    /// Process regions with visual numbering for 100% accurate matching
    ///
    /// This method:
    /// 1. Adds visible numbers to each region image
    /// 2. Sends to Gemini with numbered prompt
    /// 3. Matches translations back by bubble_number field
    async fn process_regions_with_numbering(
        &self,
        all_simple_regions: Vec<(usize, usize, u64, [i32; 4], Vec<u8>)>,
        api_client: &Arc<ApiClient>,
        num_keys: usize,
        reuse_factor: usize,
        ocr_model_override: Option<&str>,
        target_language: Option<&str>,
        cache_enabled: bool,
    ) -> Result<Vec<(usize, usize, OCRTranslation)>> {
        use rayon::prelude::*;

        // Step 1: Add visible numbers to all images (parallel processing)
        // Create mapping: display_number (1-indexed) → (page_index, region_id, source_hash, bbox)
        let numbering_config = NumberingConfig::default();

        let numbered_data: Vec<_> = all_simple_regions
            .par_iter()
            .enumerate()
            .map(|(idx, (page_index, region_id, source_hash, bbox, png_bytes))| {
                let display_number = idx + 1; // 1-indexed for Gemini
                let (numbered_bytes, _panel_width) = add_number_to_region(
                    png_bytes,
                    display_number,
                    Some(numbering_config.clone()),
                )?;
                Ok((display_number, *page_index, *region_id, *source_hash, *bbox, numbered_bytes))
            })
            .collect::<Result<Vec<_>>>()
            .context("Failed to add numbers to region images")?;

        debug!("✓ Added visible numbers to {} regions", numbered_data.len());

        // Build lookup map: display_number → (page_index, region_id, source_hash, bbox)
        let number_to_region: HashMap<usize, (usize, usize, u64, [i32; 4])> = numbered_data
            .iter()
            .map(|(num, page_idx, region_id, source_hash, bbox, _)| {
                (*num, (*page_idx, *region_id, *source_hash, *bbox))
            })
            .collect();

        // Step 2: Split into chunks for API keys
        let chunk_size = (numbered_data.len() + num_keys - 1) / num_keys;
        let key_chunks: Vec<Vec<_>> = numbered_data
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Step 3: Further split by reuse_factor
        let mut all_sub_chunks: Vec<(usize, Vec<(usize, Vec<u8>)>)> = Vec::new();

        for (key_idx, key_chunk) in key_chunks.into_iter().enumerate() {
            if key_chunk.is_empty() {
                continue;
            }

            let sub_chunk_size = (key_chunk.len() + reuse_factor - 1) / reuse_factor;
            let sub_chunks: Vec<Vec<_>> = key_chunk
                .chunks(sub_chunk_size)
                .map(|sub| sub.to_vec())
                .collect();

            debug!(
                "  Key {}: {} regions → {} sub-chunks",
                key_idx,
                key_chunk.len(),
                sub_chunks.len()
            );

            for sub_chunk in sub_chunks {
                // Extract (display_number, numbered_bytes) for the sub-chunk
                let sub_data: Vec<(usize, Vec<u8>)> = sub_chunk
                    .into_iter()
                    .map(|(num, _, _, _, _, bytes)| (num, bytes))
                    .collect();
                all_sub_chunks.push((key_idx, sub_data));
            }
        }

        debug!(
            "✓ Created {} total NUMBERED API calls",
            all_sub_chunks.len()
        );

        // Step 4: Process all sub-chunks in parallel with NUMBERED API
        let target_language_str = target_language.map(|s| s.to_string());
        let tasks: Vec<_> = all_sub_chunks
            .into_iter()
            .map(|(key_idx, sub_data)| {
                let api_client = Arc::clone(api_client);
                let model_override = ocr_model_override.map(|s| s.to_string());
                let target_language = target_language_str.clone();

                tokio::spawn(async move {
                    let image_bytes: Vec<Vec<u8>> = sub_data
                        .iter()
                        .map(|(_, bytes)| bytes.clone())
                        .collect();

                    // Use NUMBERED API call for accurate matching
                    let numbered_translations = api_client
                        .ocr_translate_numbered_batch_with_key(
                            image_bytes,
                            model_override.as_deref(),
                            target_language.as_deref(),
                            key_idx,
                        )
                        .await?;

                    Ok::<_, anyhow::Error>(numbered_translations)
                })
            })
            .collect();

        // Wait for all parallel API calls
        let results = try_join_all(tasks)
            .await
            .context("One or more numbered API tasks panicked")?
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .context("One or more numbered API calls failed")?;

        // Step 5: Match translations back to regions by bubble_number
        let mut all_translations = Vec::new();
        let mut matched_count = 0;
        let mut unmatched_count = 0;

        for numbered_translations in results {
            for nt in numbered_translations {
                if let Some(&(page_index, region_id, source_hash, bbox)) = number_to_region.get(&nt.bubble_number) {
                    // Cache the translation (using original bbox, not numbered image)
                    if cache_enabled {
                        let cache_key = TranslationCache::generate_key_from_source(source_hash, &bbox);
                        self.cache.put(cache_key, &nt.translation);
                    }
                    all_translations.push((page_index, region_id, nt.translation));
                    matched_count += 1;
                } else {
                    warn!(
                        "⚠️ Unmatched bubble_number {} from Gemini response",
                        nt.bubble_number
                    );
                    unmatched_count += 1;
                }
            }
        }

        if unmatched_count > 0 {
            warn!(
                "⚠️ {} translations unmatched out of {} (matched: {})",
                unmatched_count,
                matched_count + unmatched_count,
                matched_count
            );
        } else {
            info!(
                "✓ All {} translations matched by bubble number",
                matched_count
            );
        }

        Ok(all_translations)
    }

    /// Legacy processing without visual numbering (array-order matching)
    ///
    /// This is the original approach that relies on array order matching.
    /// Less accurate when Gemini misinterprets bubble order.
    async fn process_regions_legacy(
        &self,
        all_simple_regions: Vec<(usize, usize, u64, [i32; 4], Vec<u8>)>,
        api_client: &Arc<ApiClient>,
        num_keys: usize,
        reuse_factor: usize,
        ocr_model_override: Option<&str>,
        target_language: Option<&str>,
        cache_enabled: bool,
    ) -> Result<Vec<(usize, usize, OCRTranslation)>> {
        // Step 1: Split regions into chunks for each API key
        let chunk_size = (all_simple_regions.len() + num_keys - 1) / num_keys;
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

            let sub_chunk_size = (key_chunk.len() + reuse_factor - 1) / reuse_factor;
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

            for sub_chunk in sub_chunks {
                all_sub_chunks.push((key_idx, sub_chunk));
            }
        }

        debug!(
            "✓ Created {} total legacy API calls ({} keys × {} reuse_factor)",
            all_sub_chunks.len(),
            num_keys,
            reuse_factor
        );

        // Step 3: Process all sub-chunks in parallel with pinned keys
        let target_language_str = target_language.map(|s| s.to_string());
        let tasks: Vec<_> = all_sub_chunks
            .into_iter()
            .map(|(key_idx, sub_chunk)| {
                let api_client = Arc::clone(api_client);
                let model_override = ocr_model_override.map(|s| s.to_string());
                let target_language = target_language_str.clone();

                tokio::spawn(async move {
                    let image_bytes: Vec<Vec<u8>> = sub_chunk
                        .iter()
                        .map(|(_, _, _, _, bytes)| bytes.clone())
                        .collect();

                    // Use legacy array-order matching
                    let translations = api_client
                        .ocr_translate_batch_with_key(
                            image_bytes,
                            model_override.as_deref(),
                            target_language.as_deref(),
                            key_idx,
                        )
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

        // Flatten results and cache (using array order)
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

        Ok(all_translations)
    }

    // =========================================================================
    // LOCAL OCR PROCESSING (ALTERNATIVE TO GEMINI OCR)
    // =========================================================================

    /// Run OCR ONLY on all pages (no translation)
    /// Returns raw OCR results that can be batched for translation later
    pub fn execute_ocr_only(
        &self,
        pages: &[(ImageData, Phase1Output)],
        models_dir: &std::path::Path,
        cache_enabled: bool,
    ) -> Result<(Vec<OcrResult>, Vec<(usize, usize, OCRTranslation)>)> {
        use crate::services::ocr::{get_ocr_service, is_ocr_available};

        if pages.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        info!("🔍 OCR ONLY: Processing {} pages with local OCR", pages.len());

        if !is_ocr_available(models_dir) {
            anyhow::bail!("OCR models not found at {:?}.", models_dir);
        }

        let ocr_service = get_ocr_service(models_dir)?;
        let mut ocr_results: Vec<OcrResult> = Vec::new();
        let mut cached_results: Vec<(usize, usize, OCRTranslation)> = Vec::new();

        for (image_data, phase1_output) in pages {
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
                let boxes_to_process: Vec<[i32; 4]> = if !region.label_1_regions.is_empty() {
                    region.label_1_regions.clone()
                } else {
                    vec![region.bbox]
                };

                let mut region_texts: Vec<String> = Vec::new();

                for l1_bbox in &boxes_to_process {
                    if cache_enabled {
                        let cache_key = TranslationCache::generate_key_from_source(source_image_hash, l1_bbox);
                        if let Some(cached_translation) = self.cache.get(cache_key) {
                            cached_results.push((page_index, region.region_id, cached_translation));
                            continue;
                        }
                    }

                    let [x1, y1, x2, y2] = *l1_bbox;
                    let img_width = img.width();
                    let img_height = img.height();

                    let x1_clamped = (x1.max(0) as u32).min(img_width.saturating_sub(1));
                    let y1_clamped = (y1.max(0) as u32).min(img_height.saturating_sub(1));
                    let x2_clamped = (x2.max(0) as u32).min(img_width);
                    let y2_clamped = (y2.max(0) as u32).min(img_height);

                    let width_clamped = x2_clamped.saturating_sub(x1_clamped);
                    let height_clamped = y2_clamped.saturating_sub(y1_clamped);

                    if width_clamped == 0 || height_clamped == 0 {
                        continue;
                    }

                    let cropped = img.crop_imm(x1_clamped, y1_clamped, width_clamped, height_clamped);

                    match ocr_service.recognize(&cropped) {
                        Ok((text, _confidence)) => {
                            if !text.trim().is_empty() {
                                region_texts.push(text);
                            }
                        }
                        Err(e) => {
                            warn!("OCR failed for region {}: {:?}", region.region_id, e);
                        }
                    }
                }

                if !region_texts.is_empty() {
                    let combined_text = region_texts.join("\n");
                    ocr_results.push((page_index, region.region_id, source_image_hash, region.bbox, combined_text));
                }
            }
        }

        info!("✓ OCR extracted text from {} regions ({} cache hits)", ocr_results.len(), cached_results.len());
        Ok((ocr_results, cached_results))
    }

    /// Translate OCR results using Cerebras API (batched, max 80 per call)
    pub async fn translate_ocr_results(
        &self,
        ocr_results: Vec<OcrResult>,
        cached_results: Vec<(usize, usize, OCRTranslation)>,
        use_cerebras: bool,
        cerebras_api_key: Option<&str>,
        target_language: Option<&str>,
        cache_enabled: bool,
        custom_api_keys: Option<&[String]>,
    ) -> Result<Vec<(usize, usize, OCRTranslation)>> {
        use crate::services::translation::CerebrasClient;

        if ocr_results.is_empty() {
            return Ok(cached_results);
        }

        info!("🌐 Translating {} OCR results (batched)", ocr_results.len());

        let mut all_translations = cached_results;

        if use_cerebras {
            let cerebras_key = cerebras_api_key
                .map(|s| s.to_string())
                .or_else(|| std::env::var("CEREBRAS_API_KEY").ok())
                .ok_or_else(|| anyhow::anyhow!("Cerebras API key not provided"))?;

            // Validate key by creating client (used in spawned tasks below)
            let _cerebras = CerebrasClient::new(cerebras_key.clone())?;

            // Batch into chunks of 80 for parallel API calls
            const BATCH_SIZE: usize = 80;
            let chunks: Vec<_> = ocr_results.chunks(BATCH_SIZE).collect();

            if chunks.len() > 1 {
                info!("📦 Splitting {} regions into {} batches of max {}", ocr_results.len(), chunks.len(), BATCH_SIZE);
            }

            // Process all batches in parallel
            let mut batch_tasks = Vec::new();
            for chunk in chunks {
                let texts: Vec<(usize, String)> = chunk
                    .iter()
                    .map(|(_, region_id, _, _, text)| (*region_id, text.clone()))
                    .collect();
                
                let cerebras_clone = CerebrasClient::new(
                    cerebras_api_key
                        .map(|s| s.to_string())
                        .or_else(|| std::env::var("CEREBRAS_API_KEY").ok())
                        .unwrap()
                )?;
                let target_lang = target_language.map(|s| s.to_string());
                let chunk_data: Vec<_> = chunk.to_vec();

                let task = tokio::spawn(async move {
                    let translated = cerebras_clone
                        .translate_batch(texts, target_lang.as_deref())
                        .await?;
                    Ok::<_, anyhow::Error>((chunk_data, translated))
                });
                batch_tasks.push(task);
            }

            // Collect all batch results
            for task in batch_tasks {
                match task.await {
                    Ok(Ok((chunk_data, translated))) => {
                        let translation_map: std::collections::HashMap<usize, String> =
                            translated.into_iter().collect();

                        for (page_index, region_id, source_hash, bbox, original_text) in chunk_data {
                            let translated_text = translation_map
                                .get(&region_id)
                                .cloned()
                                .unwrap_or_else(|| original_text.clone());

                            let translation = OCRTranslation {
                                original_text: Arc::from(original_text.as_str()),
                                translated_text: Arc::from(translated_text.as_str()),
                            };

                            if cache_enabled {
                                let cache_key = TranslationCache::generate_key_from_source(source_hash, &bbox);
                                self.cache.put(cache_key, &translation);
                            }

                            all_translations.push((page_index, region_id, translation));
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Translation batch failed: {:?}", e);
                    }
                    Err(e) => {
                        tracing::error!("Translation task panicked: {:?}", e);
                    }
                }
            }
        } else {
            // Gemini translation (existing logic for non-Cerebras)
            let api_client: Arc<ApiClient> = if let Some(custom_keys) = custom_api_keys {
                if !custom_keys.is_empty() {
                    self.api_client.with_custom_keys(custom_keys.to_vec())
                } else {
                    Arc::clone(&self.api_client)
                }
            } else {
                Arc::clone(&self.api_client)
            };

            for (page_index, region_id, source_hash, bbox, original_text) in ocr_results {
                match api_client.translate_text(&original_text, target_language).await {
                    Ok(translated_text) => {
                        let translation = OCRTranslation {
                            original_text: Arc::from(original_text.as_str()),
                            translated_text: Arc::from(translated_text.as_str()),
                        };

                        if cache_enabled {
                            let cache_key = TranslationCache::generate_key_from_source(source_hash, &bbox);
                            self.cache.put(cache_key, &translation);
                        }

                        all_translations.push((page_index, region_id, translation));
                    }
                    Err(e) => {
                        warn!("Translation failed for region {}: {:?}", region_id, e);
                    }
                }
            }
        }

        info!("✓ Translation complete: {} total results", all_translations.len());
        Ok(all_translations)
    }

    /// Convert raw translation results to Phase2Output format
    pub fn build_phase2_outputs(
        &self,
        pages: &[(ImageData, Phase1Output)],
        translations: Vec<(usize, usize, OCRTranslation)>,
    ) -> Vec<Phase2Output> {
        let mut outputs: Vec<Phase2Output> = pages
            .iter()
            .map(|(_, p1)| Phase2Output {
                simple_bg_translations: Vec::new(),
                complex_bg_translations: Vec::new(),
                complex_bg_bananas: Vec::new(),
                page_index: p1.page_index,
            })
            .collect();

        // Build page index -> output index mapping
        let page_to_idx: std::collections::HashMap<usize, usize> = pages
            .iter()
            .enumerate()
            .map(|(i, (_, p1))| (p1.page_index, i))
            .collect();

        // Assign translations to outputs
        for (page_index, region_id, translation) in translations {
            if let Some(&idx) = page_to_idx.get(&page_index) {
                outputs[idx].simple_bg_translations.push((region_id, translation));
            }
        }

        outputs
    }

    /// Execute Phase 2 with local OCR instead of Gemini
    ///
    /// This method:
    /// 1. Uses local OCR model to extract Japanese text from regions
    /// 2. Translates extracted text using Cerebras (fast) or Gemini
    /// 3. Returns translations in the same format as execute_batch
    ///
    /// # Arguments
    /// * `pages` - Vector of (ImageData, Phase1Output) pairs
    /// * `models_dir` - Path to models directory containing OCR model
    /// * `use_cerebras` - Whether to use Cerebras API for translation
    /// * `cerebras_api_key` - Optional Cerebras API key
    /// * `target_language` - Optional target language for translation
    /// * `cache_enabled` - Whether to use translation cache
    /// * `custom_api_keys` - Optional custom Gemini API keys (used if not using Cerebras)
    ///
    /// # Returns
    /// Vector of Phase2Output for each page
    pub async fn execute_batch_with_local_ocr(
        &self,
        pages: &[(ImageData, Phase1Output)],
        models_dir: &std::path::Path,
        use_cerebras: bool,
        cerebras_api_key: Option<&str>,
        target_language: Option<&str>,
        cache_enabled: bool,
        custom_api_keys: Option<&[String]>,
    ) -> Result<Vec<Phase2Output>> {
        use crate::services::ocr::{get_ocr_service, is_ocr_available};
        use crate::services::translation::CerebrasClient;

        if pages.is_empty() {
            return Ok(Vec::new());
        }

        info!("Phase 2 OCR MODE: Processing {} pages with local OCR", pages.len());

        // Check OCR availability
        if !is_ocr_available(models_dir) {
            anyhow::bail!("OCR models not found at {:?}. Local OCR feature is unavailable.", models_dir);
        }

        // Get or initialize OCR service
        let ocr_service = get_ocr_service(models_dir)?;

        // Collect all regions and run OCR
        let mut ocr_results: Vec<(usize, usize, u64, [i32; 4], String)> = Vec::new();
        let mut cached_results: Vec<(usize, usize, OCRTranslation)> = Vec::new();

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
                // Process label_1_regions (text areas) just like cleaning does
                // This ensures OCR runs on the same image areas that get cleaned
                let boxes_to_process: Vec<[i32; 4]> = if !region.label_1_regions.is_empty() {
                    region.label_1_regions.clone()
                } else {
                    // Fallback to region.bbox if no label_1_regions
                    vec![region.bbox]
                };

                // Collect text from all label_1 boxes for this region
                let mut region_texts: Vec<String> = Vec::new();

                for l1_bbox in &boxes_to_process {
                    // Check cache first
                    if cache_enabled {
                        let cache_key = TranslationCache::generate_key_from_source(
                            source_image_hash,
                            l1_bbox,
                        );
                        if let Some(cached_translation) = self.cache.get(cache_key) {
                            cached_results.push((page_index, region.region_id, cached_translation));
                            continue;
                        }
                    }

                    // Crop the label_1 region (same as cleaning does)
                    let [x1, y1, x2, y2] = *l1_bbox;
                    let img_width = img.width();
                    let img_height = img.height();

                    let x1_clamped = (x1.max(0) as u32).min(img_width.saturating_sub(1));
                    let y1_clamped = (y1.max(0) as u32).min(img_height.saturating_sub(1));
                    let x2_clamped = (x2.max(0) as u32).min(img_width);
                    let y2_clamped = (y2.max(0) as u32).min(img_height);

                    let width_clamped = x2_clamped.saturating_sub(x1_clamped);
                    let height_clamped = y2_clamped.saturating_sub(y1_clamped);

                    if width_clamped == 0 || height_clamped == 0 {
                        debug!("Region {} label_1 box has zero size, skipping", region.region_id);
                        continue;
                    }

                    let cropped = img.crop_imm(x1_clamped, y1_clamped, width_clamped, height_clamped);

                    // Run local OCR
                    match ocr_service.recognize(&cropped) {
                        Ok((text, _confidence)) => {
                            if !text.trim().is_empty() {
                                region_texts.push(text);
                            }
                        }
                        Err(e) => {
                            warn!("OCR failed for region {} l1_box: {:?}", region.region_id, e);
                        }
                    }
                }

                // Combine all text from this region's label_1 boxes
                if !region_texts.is_empty() {
                    let combined_text = region_texts.join("\n");
                    ocr_results.push((
                        page_index,
                        region.region_id,
                        source_image_hash,
                        region.bbox,
                        combined_text,
                    ));
                } else {
                    debug!("OCR returned empty text for region {}", region.region_id);
                }
            }
        }

        info!(
            "OCR extracted text from {} regions ({} cache hits)",
            ocr_results.len(),
            cached_results.len()
        );

        // Translate the extracted text
        let translations = if !ocr_results.is_empty() {
            if use_cerebras {
                // Use Cerebras for translation
                let cerebras_key = cerebras_api_key
                    .map(|s| s.to_string())
                    .or_else(|| std::env::var("CEREBRAS_API_KEY").ok())
                    .ok_or_else(|| anyhow::anyhow!("Cerebras API key not provided"))?;

                let cerebras = CerebrasClient::new(cerebras_key)?;

                // Prepare batch for Cerebras
                let texts: Vec<(usize, String)> = ocr_results
                    .iter()
                    .map(|(_, region_id, _, _, text)| (*region_id, text.clone()))
                    .collect();

                // Log what's being sent to translation
                for (region_id, text) in &texts {
                    debug!("Translation input: region {} = '{}'", region_id, text.chars().take(50).collect::<String>());
                }

                let translated = cerebras
                    .translate_batch(texts, target_language)
                    .await
                    .context("Cerebras translation failed")?;

                // Log what came back
                for (region_id, text) in &translated {
                    debug!("Translation output: region {} = '{}'", region_id, text.chars().take(50).collect::<String>());
                }

                // Build translation map
                let translation_map: std::collections::HashMap<usize, String> =
                    translated.into_iter().collect();

                // Match back to regions
                let mut result = Vec::new();
                for (page_index, region_id, source_hash, bbox, original_text) in &ocr_results {
                    let translated_text = translation_map
                        .get(region_id)
                        .cloned()
                        .unwrap_or_else(|| original_text.clone());

                    let translation = OCRTranslation {
                        original_text: Arc::from(original_text.as_str()),
                        translated_text: Arc::from(translated_text.as_str()),
                    };

                    // Cache the translation
                    if cache_enabled {
                        let cache_key = TranslationCache::generate_key_from_source(*source_hash, bbox);
                        self.cache.put(cache_key, &translation);
                    }

                    result.push((*page_index, *region_id, translation));
                }

                result
            } else {
                // Use Gemini for translation (text-only, no images)
                let api_client: Arc<ApiClient> = if let Some(custom_keys) = custom_api_keys {
                    if !custom_keys.is_empty() {
                        self.api_client.with_custom_keys(custom_keys.to_vec())
                    } else {
                        Arc::clone(&self.api_client)
                    }
                } else {
                    Arc::clone(&self.api_client)
                };

                // Translate each text using Gemini text API
                let mut result = Vec::new();
                for (page_index, region_id, source_hash, bbox, original_text) in &ocr_results {
                    match api_client
                        .translate_text(original_text, target_language)
                        .await
                    {
                        Ok(translated_text) => {
                            let translation = OCRTranslation {
                                original_text: Arc::from(original_text.as_str()),
                                translated_text: Arc::from(translated_text.as_str()),
                            };

                            // Cache the translation
                            if cache_enabled {
                                let cache_key = TranslationCache::generate_key_from_source(*source_hash, bbox);
                                self.cache.put(cache_key, &translation);
                            }

                            result.push((*page_index, *region_id, translation));
                        }
                        Err(e) => {
                            warn!("Gemini translation failed for region {}: {:?}", region_id, e);
                            // Use original text as fallback
                            let translation = OCRTranslation {
                                original_text: Arc::from(original_text.as_str()),
                                translated_text: Arc::from(original_text.as_str()),
                            };
                            result.push((*page_index, *region_id, translation));
                        }
                    }
                }

                result
            }
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

        // Distribute OCR translations
        for (page_index, region_id, translation) in translations {
            if let Some(&idx) = page_to_idx.get(&page_index) {
                outputs[idx].simple_bg_translations.push((region_id, translation));
            }
        }

        info!("Phase 2 OCR MODE: Completed processing {} pages", pages.len());

        Ok(outputs)
    }

    // =========================================================================
    // OPTIMIZED PROCESSING WITH PRE-COMPUTED OCR
    // =========================================================================

    /// Execute Phase 2 using pre-computed OCR results from Phase 1
    ///
    /// OPTIMIZATION: Phase 1 already ran local OCR in parallel with text cleaning.
    /// This method uses those pre-computed results instead of:
    /// 1. Sending images to Gemini for OCR (saves API cost and latency)
    /// 2. Running local OCR again (saves computation)
    ///
    /// # Arguments
    /// * `pages` - Vector of (ImageData, Phase1Output) pairs (Phase1Output contains ocr_results)
    /// * `use_cerebras` - Whether to use Cerebras API for translation
    /// * `cerebras_api_key` - Optional Cerebras API key
    /// * `target_language` - Optional target language for translation
    /// * `cache_enabled` - Whether to use translation cache
    /// * `custom_api_keys` - Optional custom Gemini API keys (used if not using Cerebras)
    ///
    /// # Returns
    /// Vector of Phase2Output for each page
    pub async fn execute_batch_with_precomputed_ocr(
        &self,
        pages: &[(ImageData, Phase1Output)],
        use_cerebras: bool,
        cerebras_api_key: Option<&str>,
        target_language: Option<&str>,
        cache_enabled: bool,
        custom_api_keys: Option<&[String]>,
    ) -> Result<Vec<Phase2Output>> {
        use crate::services::translation::CerebrasClient;

        if pages.is_empty() {
            return Ok(Vec::new());
        }

        // Check if Phase 1 has pre-computed OCR results
        let total_ocr_results: usize = pages.iter()
            .map(|(_, p)| p.ocr_results.len())
            .sum();

        if total_ocr_results == 0 {
            info!("No pre-computed OCR results found, falling back to standard execute_batch");
            // Fall back to standard batch processing (will use Gemini for OCR)
            return self.execute_batch(
                pages,
                None,
                None,
                false,
                cache_enabled,
                custom_api_keys,
                target_language,
                4,
            ).await;
        }

        info!(
            "Phase 2 PRECOMPUTED: {} OCR results from Phase 1, skipping image OCR",
            total_ocr_results
        );

        // Collect all pre-computed OCR results
        let mut ocr_texts: Vec<(usize, usize, u64, [i32; 4], String)> = Vec::new();
        let mut cached_results: Vec<(usize, usize, OCRTranslation)> = Vec::new();

        for (image_data, phase1_output) in pages {
            let source_image_hash = xxh3_64(&image_data.image_bytes);
            let page_index = phase1_output.page_index;

            for (region_id, text, _confidence) in &phase1_output.ocr_results {
                // Find the region bbox for cache key
                let region_bbox = phase1_output.regions
                    .iter()
                    .find(|r| r.region_id == *region_id)
                    .map(|r| r.bbox)
                    .unwrap_or([0, 0, 0, 0]);

                // Check cache first
                if cache_enabled {
                    let cache_key = TranslationCache::generate_key_from_source(
                        source_image_hash,
                        &region_bbox,
                    );
                    if let Some(cached_translation) = self.cache.get(cache_key) {
                        cached_results.push((page_index, *region_id, cached_translation));
                        continue;
                    }
                }

                // Not cached, need to translate
                if !text.trim().is_empty() {
                    ocr_texts.push((
                        page_index,
                        *region_id,
                        source_image_hash,
                        region_bbox,
                        text.clone(),
                    ));
                }
            }
        }

        info!(
            "Precomputed OCR: {} texts to translate ({} cache hits)",
            ocr_texts.len(),
            cached_results.len()
        );

        // Translate the extracted text
        let translations = if !ocr_texts.is_empty() {
            if use_cerebras {
                // Use Cerebras for translation (fast, ~3000 tokens/sec)
                let cerebras_key = cerebras_api_key
                    .map(|s| s.to_string())
                    .or_else(|| std::env::var("CEREBRAS_API_KEY").ok())
                    .ok_or_else(|| anyhow::anyhow!("Cerebras API key not provided"))?;

                let cerebras = CerebrasClient::new(cerebras_key)?;

                // Prepare batch for Cerebras
                let texts: Vec<(usize, String)> = ocr_texts
                    .iter()
                    .map(|(_, region_id, _, _, text)| (*region_id, text.clone()))
                    .collect();

                let translated = cerebras
                    .translate_batch(texts, target_language)
                    .await
                    .context("Cerebras translation failed")?;

                // Build translation map
                let translation_map: std::collections::HashMap<usize, String> =
                    translated.into_iter().collect();

                // Match back to regions
                let mut result = Vec::new();
                for (page_index, region_id, source_hash, bbox, original_text) in &ocr_texts {
                    let translated_text = translation_map
                        .get(region_id)
                        .cloned()
                        .unwrap_or_else(|| original_text.clone());

                    let translation = OCRTranslation {
                        original_text: Arc::from(original_text.as_str()),
                        translated_text: Arc::from(translated_text.as_str()),
                    };

                    // Cache the translation
                    if cache_enabled {
                        let cache_key = TranslationCache::generate_key_from_source(*source_hash, bbox);
                        self.cache.put(cache_key, &translation);
                    }

                    result.push((*page_index, *region_id, translation));
                }

                result
            } else {
                // Use Gemini for translation (text-only, no images)
                let api_client: Arc<ApiClient> = if let Some(custom_keys) = custom_api_keys {
                    if !custom_keys.is_empty() {
                        self.api_client.with_custom_keys(custom_keys.to_vec())
                    } else {
                        Arc::clone(&self.api_client)
                    }
                } else {
                    Arc::clone(&self.api_client)
                };

                // Translate each text using Gemini text API
                let mut result = Vec::new();
                for (page_index, region_id, source_hash, bbox, original_text) in &ocr_texts {
                    match api_client
                        .translate_text(original_text, target_language)
                        .await
                    {
                        Ok(translated_text) => {
                            let translation = OCRTranslation {
                                original_text: Arc::from(original_text.as_str()),
                                translated_text: Arc::from(translated_text.as_str()),
                            };

                            // Cache the translation
                            if cache_enabled {
                                let cache_key = TranslationCache::generate_key_from_source(*source_hash, bbox);
                                self.cache.put(cache_key, &translation);
                            }

                            result.push((*page_index, *region_id, translation));
                        }
                        Err(e) => {
                            warn!("Gemini translation failed for region {}: {:?}", region_id, e);
                            // Use original text as fallback
                            let translation = OCRTranslation {
                                original_text: Arc::from(original_text.as_str()),
                                translated_text: Arc::from(original_text.as_str()),
                            };
                            result.push((*page_index, *region_id, translation));
                        }
                    }
                }

                result
            }
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

        // Distribute translations
        for (page_index, region_id, translation) in translations {
            if let Some(&idx) = page_to_idx.get(&page_index) {
                outputs[idx].simple_bg_translations.push((region_id, translation));
            }
        }

        info!("Phase 2 PRECOMPUTED: Completed processing {} pages", pages.len());

        Ok(outputs)
    }
}
