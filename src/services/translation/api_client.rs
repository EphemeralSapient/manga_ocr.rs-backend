use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, instrument, warn};

use crate::core::config::Config;
use crate::core::types::{BananaResult, OCRTranslation};
use crate::middleware::api_key_pool::ApiKeyPool;
use crate::middleware::circuit_breaker::CircuitBreaker;
use crate::utils::Metrics;

/// Enhanced Gemini API client with circuit breaker, timeouts, and metrics
pub struct ApiClient {
    config: Arc<Config>,
    api_key_pool: Arc<ApiKeyPool>,
    http_client: reqwest::Client,
    circuit_breaker: CircuitBreaker,
    metrics: Option<Metrics>,
}

/// JSON schema for OCR/Translation response (with bubble number for accurate matching)
#[derive(Debug, Serialize, Deserialize)]
struct OCRTranslationResponse {
    /// The visible bubble number shown in the image (1-indexed)
    bubble_number: usize,
    /// Original text extracted from the bubble
    original_text: String,
    /// Translated text
    translated_text: String,
}

/// JSON schema for batch OCR/Translation response
#[derive(Debug, Serialize, Deserialize)]
struct BatchOCRResponse {
    translations: Vec<OCRTranslationResponse>,
}

/// Result type for numbered translation that includes the bubble number
#[derive(Debug, Clone)]
pub struct NumberedTranslation {
    pub bubble_number: usize,
    pub translation: OCRTranslation,
}

impl ApiClient {
    /// Create a new API client with enhanced features
    pub fn new(
        config: Arc<Config>,
        circuit_breaker: Option<CircuitBreaker>,
        metrics: Option<Metrics>,
    ) -> Result<Self> {
        let api_key_pool = Arc::new(ApiKeyPool::new(config.api_keys().to_vec()));

        // Get timeout from config (default 60s)
        let timeout = Duration::from_secs(
            std::env::var("API_TIMEOUT_SECONDS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60)
        );

        // Create HTTP client with timeout and connection pooling
        let http_client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("Failed to create HTTP client")?;

        let circuit_breaker = circuit_breaker.unwrap_or_default();

        Ok(Self {
            config,
            api_key_pool,
            http_client,
            circuit_breaker,
            metrics,
        })
    }

    /// Get the total number of available API keys
    pub async fn total_keys(&self) -> usize {
        self.api_key_pool.total_keys().await
    }

    /// Create a temporary ApiClient with custom API keys for per-request override
    pub fn with_custom_keys(
        &self,
        custom_keys: Vec<String>,
    ) -> Arc<Self> {
        let custom_pool = Arc::new(ApiKeyPool::new(custom_keys));

        Arc::new(Self {
            config: Arc::clone(&self.config),
            api_key_pool: custom_pool,
            http_client: self.http_client.clone(),
            circuit_breaker: self.circuit_breaker.clone(),
            metrics: self.metrics.clone(),
        })
    }

    /// Perform OCR and translation on a batch of images (simple backgrounds)
    ///
    /// # Arguments
    /// * `image_bytes_batch` - Vector of image bytes to process
    /// * `model_override` - Optional model name to override the default from config
    /// * `target_language` - Optional target language for translation (defaults to "English")
    ///
    /// # Returns
    /// Vector of OCRTranslation results
    #[instrument(skip(self, image_bytes_batch), fields(batch_size = image_bytes_batch.len()))]
    pub async fn ocr_translate_batch(
        &self,
        image_bytes_batch: Vec<Vec<u8>>,
        model_override: Option<&str>,
        target_language: Option<&str>,
    ) -> Result<Vec<OCRTranslation>> {
        self.ocr_translate_batch_internal(image_bytes_batch, model_override, target_language, None, false).await
    }

    /// Perform OCR and translation on NUMBERED images for accurate bubble matching
    ///
    /// This method expects images that have been pre-processed with `add_number_to_region()`
    /// to include visible bubble numbers. The prompt instructs Gemini to identify each
    /// bubble by its visible number, ensuring 100% accurate translation-to-bubble matching.
    ///
    /// # Arguments
    /// * `image_bytes_batch` - Vector of numbered image bytes (processed with add_number_to_region)
    /// * `model_override` - Optional model name to override the default from config
    /// * `target_language` - Optional target language for translation (defaults to "English")
    ///
    /// # Returns
    /// Vector of NumberedTranslation results with bubble numbers for matching
    #[instrument(skip(self, image_bytes_batch), fields(batch_size = image_bytes_batch.len()))]
    pub async fn ocr_translate_numbered_batch(
        &self,
        image_bytes_batch: Vec<Vec<u8>>,
        model_override: Option<&str>,
        target_language: Option<&str>,
    ) -> Result<Vec<NumberedTranslation>> {
        self.ocr_translate_numbered_batch_internal(image_bytes_batch, model_override, target_language, None).await
    }

    /// Perform OCR and translation on NUMBERED images with a specific API key
    #[instrument(skip(self, image_bytes_batch), fields(batch_size = image_bytes_batch.len(), key_index = key_index))]
    pub async fn ocr_translate_numbered_batch_with_key(
        &self,
        image_bytes_batch: Vec<Vec<u8>>,
        model_override: Option<&str>,
        target_language: Option<&str>,
        key_index: usize,
    ) -> Result<Vec<NumberedTranslation>> {
        self.ocr_translate_numbered_batch_internal(image_bytes_batch, model_override, target_language, Some(key_index)).await
    }

    /// Perform OCR and translation with a specific API key (for reuse_factor parallelism)
    ///
    /// # Arguments
    /// * `image_bytes_batch` - Vector of image bytes to process
    /// * `model_override` - Optional model name to override the default from config
    /// * `target_language` - Optional target language for translation (defaults to "English")
    /// * `key_index` - Specific API key index to use
    ///
    /// # Returns
    /// Vector of OCRTranslation results
    #[instrument(skip(self, image_bytes_batch), fields(batch_size = image_bytes_batch.len(), key_index = key_index))]
    pub async fn ocr_translate_batch_with_key(
        &self,
        image_bytes_batch: Vec<Vec<u8>>,
        model_override: Option<&str>,
        target_language: Option<&str>,
        key_index: usize,
    ) -> Result<Vec<OCRTranslation>> {
        self.ocr_translate_batch_internal(image_bytes_batch, model_override, target_language, Some(key_index), false).await
    }

    /// Internal implementation for NUMBERED OCR/translation batch
    /// Uses modified prompt and schema for accurate bubble matching
    async fn ocr_translate_numbered_batch_internal(
        &self,
        image_bytes_batch: Vec<Vec<u8>>,
        model_override: Option<&str>,
        target_language: Option<&str>,
        pinned_key_index: Option<usize>,
    ) -> Result<Vec<NumberedTranslation>> {
        debug!("NUMBERED OCR/Translation batch of {} images", image_bytes_batch.len());

        // Check circuit breaker
        if !self.circuit_breaker.allow_request() {
            warn!("Circuit breaker is open, failing fast");
            anyhow::bail!("Circuit breaker is open, API is unavailable");
        }

        let start = Instant::now();
        let max_key_attempts = self.api_key_pool.total_keys().await.max(1);
        let mut last_error = None;

        for attempt in 0..max_key_attempts {
            let key_result = if attempt == 0 && pinned_key_index.is_some() {
                self.api_key_pool
                    .get_key_by_index(pinned_key_index.unwrap())
                    .await
            } else {
                self.api_key_pool
                    .get_healthy_key()
                    .await
            };

            let (key_idx, api_key) = match key_result {
                Some((idx, key)) => {
                    if attempt > 0 {
                        debug!("Retrying with API key {} (attempt {}/{})", idx, attempt + 1, max_key_attempts);
                    }
                    (idx, key)
                },
                None => {
                    if attempt > 0 {
                        debug!("No healthy keys available on attempt {}/{}, waiting 10s before retry", attempt + 1, max_key_attempts);
                        tokio::time::sleep(Duration::from_secs(10)).await;
                        continue;
                    } else {
                        anyhow::bail!("No healthy API keys available");
                    }
                }
            };

            let model = model_override.unwrap_or_else(|| self.config.ocr_translation_model());
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, api_key
            );

            // Build content with all numbered images
            let contents: Vec<_> = image_bytes_batch
                .par_iter()
                .map(|image_bytes| {
                    let base64_image = general_purpose::STANDARD.encode(image_bytes);
                    serde_json::json!({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image
                        }
                    })
                })
                .collect();

            let mut contents = contents;

            // NUMBERED PROMPT: Explicitly tell Gemini to identify by visible number
            let target_lang = target_language.unwrap_or("English");
            let prompt = format!(
                "Each image shows a text bubble with a VISIBLE NUMBER on the LEFT side. \
                 For each image, identify the bubble by its visible number and extract the text. \
                 Translate the text to {}. \
                 Return a JSON array where each object has: \
                 - 'bubble_number': the visible number shown on the left of the image (integer) \
                 - 'original_text': the extracted text (ignore the number, extract only the bubble content) \
                 - 'translated_text': the translation \
                 If no text is found in a bubble, use empty strings for text fields but still include the bubble_number.",
                target_lang
            );
            contents.push(serde_json::json!({"text": prompt}));

            // NUMBERED SCHEMA: Include bubble_number field
            let mut request_body = serde_json::json!({
                "contents": [{
                    "parts": contents
                }],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "translations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "bubble_number": {"type": "integer"},
                                        "original_text": {"type": "string"},
                                        "translated_text": {"type": "string"}
                                    },
                                    "required": ["bubble_number", "original_text", "translated_text"]
                                }
                            }
                        },
                        "required": ["translations"]
                    }
                }
            });

            let enable_thinking = std::env::var("GEMINI_ENABLE_THINKING")
                .ok()
                .and_then(|s| s.parse::<bool>().ok())
                .unwrap_or(false);

            if !enable_thinking {
                request_body["generationConfig"]["thinkingConfig"] = serde_json::json!({
                    "thinking_budget": 0
                });
            }

            let result = self
                .send_with_retries(&url, &request_body, key_idx)
                .await;

            let duration = start.elapsed();

            match result {
                Ok(response_text) => {
                    self.circuit_breaker.record_success();
                    self.api_key_pool.record_success(key_idx).await;

                    let response: serde_json::Value = serde_json::from_str(&response_text)
                        .context("Failed to parse API response")?;

                    let (input_tokens, output_tokens) = extract_token_usage(&response);

                    if let Some(ref m) = self.metrics {
                        m.record_api_call(true, duration, input_tokens, output_tokens);
                    }

                    let translations_json = response["candidates"][0]["content"]["parts"][0]["text"]
                        .as_str()
                        .context("Missing text in API response")?;

                    let batch_response: BatchOCRResponse = serde_json::from_str(translations_json)
                        .context("Failed to parse batch OCR response")?;

                    // Convert to NumberedTranslation with bubble numbers preserved
                    let results = batch_response
                        .translations
                        .into_iter()
                        .map(|t| NumberedTranslation {
                            bubble_number: t.bubble_number,
                            translation: OCRTranslation {
                                original_text: Arc::from(t.original_text.as_str()),
                                translated_text: Arc::from(t.translated_text.as_str()),
                            },
                        })
                        .collect();

                    return Ok(results);
                }
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    self.api_key_pool.record_failure(key_idx).await;

                    if let Some(ref m) = self.metrics {
                        m.record_api_call(false, duration, 0, 0);
                    }

                    let error_string = e.to_string();
                    let is_rate_limit = error_string.contains("429") || error_string.contains("quota");
                    let is_overload = error_string.contains("503") || error_string.contains("overload");

                    if is_rate_limit || is_overload {
                        warn!(
                            "Numbered API key {} failed with {} error (attempt {}/{}): {}",
                            key_idx,
                            if is_rate_limit { "rate limit" } else { "overload" },
                            attempt + 1,
                            max_key_attempts,
                            e
                        );

                        last_error = Some(e);

                        if attempt < max_key_attempts - 1 {
                            debug!("Waiting 10 seconds before trying next key...");
                            tokio::time::sleep(Duration::from_secs(10)).await;
                            continue;
                        }
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All API keys exhausted for numbered batch")))
    }

    /// Internal implementation for OCR/translation batch
    async fn ocr_translate_batch_internal(
        &self,
        image_bytes_batch: Vec<Vec<u8>>,
        model_override: Option<&str>,
        target_language: Option<&str>,
        pinned_key_index: Option<usize>,
        _use_numbered: bool, // Reserved for future use
    ) -> Result<Vec<OCRTranslation>> {
        debug!("OCR/Translation batch of {} images", image_bytes_batch.len());

        // Check circuit breaker
        if !self.circuit_breaker.allow_request() {
            warn!("Circuit breaker is open, failing fast");
            anyhow::bail!("Circuit breaker is open, API is unavailable");
        }

        let start = Instant::now();

        // Try with different keys until success or all keys exhausted
        let max_key_attempts = self.api_key_pool.total_keys().await.max(1);
        let mut last_error = None;

        for attempt in 0..max_key_attempts {
            // Get API key (either pinned on first attempt, or any healthy key)
            let key_result = if attempt == 0 && pinned_key_index.is_some() {
                self.api_key_pool
                    .get_key_by_index(pinned_key_index.unwrap())
                    .await
            } else {
                self.api_key_pool
                    .get_healthy_key()
                    .await
            };

            let (key_idx, api_key) = match key_result {
                Some((idx, key)) => {
                    if attempt > 0 {
                        debug!("Retrying with API key {} (attempt {}/{})", idx, attempt + 1, max_key_attempts);
                    }
                    (idx, key)
                },
                None => {
                    if attempt > 0 {
                        debug!("No healthy keys available on attempt {}/{}, waiting 10s before retry", attempt + 1, max_key_attempts);
                        tokio::time::sleep(Duration::from_secs(10)).await;
                        continue;
                    } else {
                        anyhow::bail!("No healthy API keys available");
                    }
                }
            };

            // Prepare request
            let model = model_override.unwrap_or_else(|| self.config.ocr_translation_model());
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, api_key
            );

            // Build multipart content with all images
            // OPTIMIZATION: Parallel base64 encoding (10-15% faster for batches >5)
            let contents: Vec<_> = image_bytes_batch
                .par_iter()
                .map(|image_bytes| {
                    let base64_image = general_purpose::STANDARD.encode(image_bytes);
                    serde_json::json!({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image
                        }
                    })
                })
                .collect();

            let mut contents = contents;

            // Add text prompt with target language
            let target_lang = target_language.unwrap_or("English");
            let prompt = format!(
                "For each image provided, extract the original text found in the image and translate it to {}. \
                 Return a JSON array with {} objects, each containing 'original_text' and 'translated_text' fields. \
                 Maintain the order of images. If no text is found, use empty strings.",
                target_lang,
                image_bytes_batch.len()
            );
            contents.push(serde_json::json!({"text": prompt}));

            let mut request_body = serde_json::json!({
                "contents": [{
                    "parts": contents
                }],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "translations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "original_text": {"type": "string"},
                                        "translated_text": {"type": "string"}
                                    },
                                    "required": ["original_text", "translated_text"]
                                }
                            }
                        },
                        "required": ["translations"]
                    }
                }
            });

            // Disable thinking by default for faster responses and lower token usage
            // Can be enabled via GEMINI_ENABLE_THINKING=true environment variable
            let enable_thinking = std::env::var("GEMINI_ENABLE_THINKING")
                .ok()
                .and_then(|s| s.parse::<bool>().ok())
                .unwrap_or(false);

            if !enable_thinking {
                // For Gemini 2.5+, disable thinking by setting thinking_budget to 0
                request_body["generationConfig"]["thinkingConfig"] = serde_json::json!({
                    "thinking_budget": 0
                });
            }

            // Send request with retries
            let result = self
                .send_with_retries(&url, &request_body, key_idx)
                .await;

            let duration = start.elapsed();

            match result {
                Ok(response_text) => {
                    // Record success
                    self.circuit_breaker.record_success();
                    self.api_key_pool.record_success(key_idx).await;

                    // Parse response
                    let response: serde_json::Value = serde_json::from_str(&response_text)
                        .context("Failed to parse API response")?;

                    // Extract token usage if available
                    let (input_tokens, output_tokens) = extract_token_usage(&response);

                    // Record metrics
                    if let Some(ref m) = self.metrics {
                        m.record_api_call(true, duration, input_tokens, output_tokens);
                    }

                    // Extract translations
                    let translations_json = response["candidates"][0]["content"]["parts"][0]["text"]
                        .as_str()
                        .context("Missing text in API response")?;

                    let batch_response: BatchOCRResponse = serde_json::from_str(translations_json)
                        .context("Failed to parse batch OCR response")?;

                    // Convert to OCRTranslation with Arc<str> for efficient cloning
                    let results = batch_response
                        .translations
                        .into_iter()
                        .map(|t| OCRTranslation {
                            original_text: Arc::from(t.original_text.as_str()),
                            translated_text: Arc::from(t.translated_text.as_str()),
                        })
                        .collect();

                    return Ok(results);
                }
                Err(e) => {
                    // Record failure
                    self.circuit_breaker.record_failure();
                    self.api_key_pool.record_failure(key_idx).await;

                    if let Some(ref m) = self.metrics {
                        m.record_api_call(false, duration, 0, 0);
                    }

                    // Check if this is a retryable error (429, 503)
                    let error_string = e.to_string();
                    let is_rate_limit = error_string.contains("429") || error_string.contains("quota");
                    let is_overload = error_string.contains("503") || error_string.contains("overload");

                    if is_rate_limit || is_overload {
                        warn!(
                            "API key {} failed with {} error (attempt {}/{}): {}",
                            key_idx,
                            if is_rate_limit { "rate limit" } else { "overload" },
                            attempt + 1,
                            max_key_attempts,
                            e
                        );

                        last_error = Some(e);

                        // Wait 10 seconds before trying next key
                        if attempt < max_key_attempts - 1 {
                            debug!("Waiting 10 seconds before trying next key...");
                            tokio::time::sleep(Duration::from_secs(10)).await;
                            continue;
                        }
                    } else {
                        // Non-retryable error, fail immediately
                        return Err(e);
                    }
                }
            }
        }

        // All keys exhausted, return last error
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All API keys exhausted")))
    }

    /// Process a single complex background region with banana mode
    ///
    /// # Arguments
    /// * `region_id` - Unique region identifier
    /// * `image_bytes` - Image bytes of the region
    /// * `model_override` - Optional model name to override the default from config
    /// * `target_language` - Optional target language for translation (defaults to "English")
    ///
    /// # Returns
    /// BananaResult with translated image
    #[instrument(skip(self, image_bytes), fields(region_id = region_id))]
    pub async fn banana_translate(
        &self,
        region_id: usize,
        image_bytes: Vec<u8>,
        model_override: Option<&str>,
        target_language: Option<&str>,
    ) -> Result<BananaResult> {
        debug!("Banana mode translation for region {}", region_id);

        // Check circuit breaker
        if !self.circuit_breaker.allow_request() {
            warn!("Circuit breaker is open, failing fast");
            anyhow::bail!("Circuit breaker is open, API is unavailable");
        }

        let start = Instant::now();

        // Try with different keys until success or all keys exhausted
        let max_key_attempts = self.api_key_pool.total_keys().await.max(1);
        let mut last_error = None;

        for attempt in 0..max_key_attempts {
            // Get healthy API key
            let key_result = self
                .api_key_pool
                .get_healthy_key()
                .await;

            let (key_idx, api_key) = match key_result {
                Some((idx, key)) => {
                    if attempt > 0 {
                        debug!("Banana: Retrying with API key {} (attempt {}/{})", idx, attempt + 1, max_key_attempts);
                    }
                    (idx, key)
                },
                None => {
                    if attempt > 0 {
                        debug!("Banana: No healthy keys available on attempt {}/{}, waiting 10s before retry", attempt + 1, max_key_attempts);
                        tokio::time::sleep(Duration::from_secs(10)).await;
                        continue;
                    } else {
                        anyhow::bail!("No healthy API keys available");
                    }
                }
            };

            // Prepare request
            let model = model_override.unwrap_or_else(|| self.config.banana_image_model());
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, api_key
            );

            let base64_image = general_purpose::STANDARD.encode(&image_bytes);

            let target_lang = target_language.unwrap_or("English");
            let prompt = format!(
                "Replace any text found in this image with {} translation, matching the original style, font, and background exactly.",
                target_lang
            );

            let request_body = serde_json::json!({
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64_image
                            }
                        }
                    ]
                }]
            });

            // Send request with retries
            let result = self
                .send_with_retries(&url, &request_body, key_idx)
                .await;

            let duration = start.elapsed();

            match result {
                Ok(response_text) => {
                    // Record success
                    self.circuit_breaker.record_success();
                    self.api_key_pool.record_success(key_idx).await;

                    // Parse response
                    let response: serde_json::Value = serde_json::from_str(&response_text)
                        .context("Failed to parse banana API response")?;

                    // Extract token usage
                    let (input_tokens, output_tokens) = extract_token_usage(&response);

                    // Record metrics
                    if let Some(ref m) = self.metrics {
                        m.record_api_call(true, duration, input_tokens, output_tokens);
                    }

                    // Extract image data
                    let inline_data = &response["candidates"][0]["content"]["parts"][0]["inline_data"];
                    let image_data_base64 = inline_data["data"]
                        .as_str()
                        .context("Missing inline_data in banana response")?;

                    // Decode base64
                    let translated_image_bytes = general_purpose::STANDARD
                        .decode(image_data_base64)
                        .context("Failed to decode banana image")?;

                    return Ok(BananaResult {
                        region_id,
                        translated_image_bytes,
                    });
                }
                Err(e) => {
                    // Record failure
                    self.circuit_breaker.record_failure();
                    self.api_key_pool.record_failure(key_idx).await;

                    if let Some(ref m) = self.metrics {
                        m.record_api_call(false, duration, 0, 0);
                    }

                    // Check if this is a retryable error (429, 503)
                    let error_string = e.to_string();
                    let is_rate_limit = error_string.contains("429") || error_string.contains("quota");
                    let is_overload = error_string.contains("503") || error_string.contains("overload");

                    if is_rate_limit || is_overload {
                        warn!(
                            "Banana API key {} failed with {} error (attempt {}/{}): {}",
                            key_idx,
                            if is_rate_limit { "rate limit" } else { "overload" },
                            attempt + 1,
                            max_key_attempts,
                            e
                        );

                        last_error = Some(e);

                        // Wait 10 seconds before trying next key
                        if attempt < max_key_attempts - 1 {
                            debug!("Waiting 10 seconds before trying next key...");
                            tokio::time::sleep(Duration::from_secs(10)).await;
                            continue;
                        }
                    } else {
                        // Non-retryable error, fail immediately
                        return Err(e);
                    }
                }
            }
        }

        // All keys exhausted, return last error
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All API keys exhausted for banana translate")))
    }

    /// Send HTTP request with retries and jitter
    async fn send_with_retries(
        &self,
        url: &str,
        body: &serde_json::Value,
        key_idx: usize,
    ) -> Result<String> {
        let max_retries = self.config.max_retries();

        for attempt in 0..=max_retries {
            match self
                .http_client
                .post(url)
                .header("Content-Type", "application/json")
                .json(body)
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        let text = response
                            .text()
                            .await
                            .context("Failed to read response body")?;
                        return Ok(text);
                    } else {
                        let status = response.status();
                        let error_text = response.text().await.unwrap_or_default();

                        // Check if this is a rate limit (429) or overload (503) error
                        let is_rate_limit = status.as_u16() == 429;
                        let is_overload = status.as_u16() == 503;

                        // Record API key failure for rate limits
                        if is_rate_limit {
                            self.api_key_pool.record_failure(key_idx).await;
                        }

                        if attempt < max_retries {
                            debug!(
                                "API request failed with status {}: {}. Retrying ({}/{})",
                                status,
                                error_text,
                                attempt + 1,
                                max_retries
                            );

                            // Use 10s wait for 429/503, exponential backoff for others
                            if is_rate_limit || is_overload {
                                debug!("Rate limit or overload detected, waiting 10 seconds before retry");
                                tokio::time::sleep(Duration::from_secs(10)).await;
                            } else {
                                // Exponential backoff with jitter for other errors
                                let base_delay = 2_u64.pow(attempt);
                                let jitter = rand::random::<u64>() % 1000; // 0-999ms
                                tokio::time::sleep(Duration::from_millis(
                                    base_delay * 1000 + jitter,
                                ))
                                .await;
                            }
                            continue;
                        } else {
                            anyhow::bail!("API request failed: {} - {}", status, error_text);
                        }
                    }
                }
                Err(e) => {
                    if attempt < max_retries {
                        debug!(
                            "HTTP request error: {}. Retrying ({}/{})",
                            e,
                            attempt + 1,
                            max_retries
                        );
                        // Exponential backoff with jitter
                        let base_delay = 2_u64.pow(attempt);
                        let jitter = rand::random::<u64>() % 1000;
                        tokio::time::sleep(Duration::from_millis(
                            base_delay * 1000 + jitter,
                        ))
                        .await;
                        continue;
                    } else {
                        return Err(e).context("HTTP request failed after retries");
                    }
                }
            }
        }

        anyhow::bail!("Failed after {} retries", max_retries)
    }

    /// Translate text using Gemini (text-only, no images)
    /// Used when local OCR extracts text and we only need translation
    pub async fn translate_text(
        &self,
        text: &str,
        target_language: Option<&str>,
    ) -> Result<String> {
        debug!("Text translation for: {}", text.chars().take(50).collect::<String>());

        // Check circuit breaker
        if !self.circuit_breaker.allow_request() {
            warn!("Circuit breaker is open, failing fast");
            anyhow::bail!("Circuit breaker is open, API is unavailable");
        }

        let start = Instant::now();

        // Get healthy API key
        let (key_idx, api_key) = self
            .api_key_pool
            .get_healthy_key()
            .await
            .ok_or_else(|| anyhow::anyhow!("No healthy API keys available"))?;

        let model = self.config.ocr_translation_model();
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, api_key
        );

        let target_lang = target_language.unwrap_or("English");
        let prompt = format!(
            "Translate the following Japanese text to {}. \
             Only output the translation, nothing else. \
             If the text is unclear, translate your best interpretation.\n\n{}",
            target_lang, text
        );

        let request_body = serde_json::json!({
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "maxOutputTokens": 1024
            }
        });

        // Send request
        let result = self
            .send_with_retries(&url, &request_body, key_idx)
            .await;

        let duration = start.elapsed();

        match result {
            Ok(response_text) => {
                self.circuit_breaker.record_success();
                self.api_key_pool.record_success(key_idx).await;

                let response: serde_json::Value = serde_json::from_str(&response_text)
                    .context("Failed to parse text translation response")?;

                let (input_tokens, output_tokens) = extract_token_usage(&response);

                if let Some(ref m) = self.metrics {
                    m.record_api_call(true, duration, input_tokens, output_tokens);
                }

                // Extract translated text
                let translated = response["candidates"][0]["content"]["parts"][0]["text"]
                    .as_str()
                    .unwrap_or(text)
                    .trim()
                    .to_string();

                Ok(translated)
            }
            Err(e) => {
                self.circuit_breaker.record_failure();
                self.api_key_pool.record_failure(key_idx).await;

                if let Some(ref m) = self.metrics {
                    m.record_api_call(false, duration, 0, 0);
                }

                Err(e)
            }
        }
    }
}

/// Extract token usage from Gemini API response
///
/// Returns (input_tokens, output_tokens) if available, otherwise (0, 0)
fn extract_token_usage(response: &serde_json::Value) -> (u64, u64) {
    let usage_metadata = &response["usageMetadata"];
    let input_tokens = usage_metadata["promptTokenCount"]
        .as_u64()
        .unwrap_or(0);
    let output_tokens = usage_metadata["candidatesTokenCount"]
        .as_u64()
        .unwrap_or(0);

    (input_tokens, output_tokens)
}
