use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine};
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

/// JSON schema for OCR/Translation response
#[derive(Debug, Serialize, Deserialize)]
struct OCRTranslationResponse {
    original_text: String,
    translated_text: String,
}

/// JSON schema for batch OCR/Translation response
#[derive(Debug, Serialize, Deserialize)]
struct BatchOCRResponse {
    translations: Vec<OCRTranslationResponse>,
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

    /// Perform OCR and translation on a batch of images (simple backgrounds)
    ///
    /// # Arguments
    /// * `image_bytes_batch` - Vector of image bytes to process
    ///
    /// # Returns
    /// Vector of OCRTranslation results
    #[instrument(skip(self, image_bytes_batch), fields(batch_size = image_bytes_batch.len()))]
    pub async fn ocr_translate_batch(
        &self,
        image_bytes_batch: Vec<Vec<u8>>,
    ) -> Result<Vec<OCRTranslation>> {
        debug!("OCR/Translation batch of {} images", image_bytes_batch.len());

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
            .context("No healthy API keys available")?;

        // Prepare request
        let model = self.config.ocr_translation_model();
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, api_key
        );

        // Build multipart content with all images
        let mut contents = Vec::new();
        for image_bytes in &image_bytes_batch {
            let base64_image = general_purpose::STANDARD.encode(image_bytes);
            contents.push(serde_json::json!({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64_image
                }
            }));
        }

        // Add text prompt
        let prompt = format!(
            "For each image provided, extract the original text found in the image and translate it to English. \
             Return a JSON array with {} objects, each containing 'original_text' and 'translated_text' fields. \
             Maintain the order of images. If no text is found, use empty strings.",
            image_bytes_batch.len()
        );
        contents.push(serde_json::json!({"text": prompt}));

        let request_body = serde_json::json!({
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

                // Convert to OCRTranslation
                let results = batch_response
                    .translations
                    .into_iter()
                    .map(|t| OCRTranslation {
                        original_text: t.original_text,
                        translated_text: t.translated_text,
                    })
                    .collect();

                Ok(results)
            }
            Err(e) => {
                // Record failure
                self.circuit_breaker.record_failure();
                self.api_key_pool.record_failure(key_idx).await;

                if let Some(ref m) = self.metrics {
                    m.record_api_call(false, duration, 0, 0);
                }

                Err(e)
            }
        }
    }

    /// Process a single complex background region with banana mode
    ///
    /// # Arguments
    /// * `region_id` - Unique region identifier
    /// * `image_bytes` - Image bytes of the region
    ///
    /// # Returns
    /// BananaResult with translated image
    #[instrument(skip(self, image_bytes), fields(region_id = region_id))]
    pub async fn banana_translate(
        &self,
        region_id: String,
        image_bytes: Vec<u8>,
    ) -> Result<BananaResult> {
        debug!("Banana mode translation for region {}", region_id);

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
            .context("No healthy API keys available")?;

        // Prepare request
        let model = self.config.banana_image_model();
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, api_key
        );

        let base64_image = general_purpose::STANDARD.encode(&image_bytes);

        let request_body = serde_json::json!({
            "contents": [{
                "parts": [
                    {"text": "Replace any text found in this image with English translation, matching the original style, font, and background exactly."},
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

                Ok(BananaResult {
                    region_id,
                    translated_image_bytes,
                })
            }
            Err(e) => {
                // Record failure
                self.circuit_breaker.record_failure();
                self.api_key_pool.record_failure(key_idx).await;

                if let Some(ref m) = self.metrics {
                    m.record_api_call(false, duration, 0, 0);
                }

                Err(e)
            }
        }
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

                        // Record API key failure for specific error codes
                        if status.is_client_error() && status.as_u16() == 429 {
                            // Rate limit error
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
                            // Exponential backoff with jitter
                            let base_delay = 2_u64.pow(attempt);
                            let jitter = rand::random::<u64>() % 1000; // 0-999ms
                            tokio::time::sleep(Duration::from_millis(
                                base_delay * 1000 + jitter,
                            ))
                            .await;
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
