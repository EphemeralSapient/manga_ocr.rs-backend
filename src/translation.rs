use crate::config::Config;
use crate::middleware::api_key_pool::ApiKeyPool;
use crate::schema::{self, GenerateContentResponse};
use crate::services::traits::Translator;
use crate::types::TextTranslation;
use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose, Engine};
use serde::Deserialize;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{debug, info, trace, warn};

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct InlineDataResponse {
    mime_type: String,
    data: String,
}

pub struct TranslationService {
    client: reqwest::Client,
    config: Arc<Config>,
    api_key_pool: Arc<ApiKeyPool>,
}

impl TranslationService {
    pub fn new(config: Arc<Config>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap();

        // Initialize API key pool with health tracking
        let api_keys: Vec<String> = config.api_keys().iter().map(|s| s.to_string()).collect();
        let api_key_pool = Arc::new(ApiKeyPool::new(api_keys));

        info!("✓ API key pool initialized with {} keys", config.api_keys().len());

        Self {
            client,
            config,
            api_key_pool,
        }
    }



    pub async fn translate_bubble(&self, bubble_bytes: &[u8], model: Option<&str>) -> Result<TextTranslation> {
        let translation_model = model.unwrap_or(self.config.translation_model());
        debug!("📝 [TRANSLATION] Starting translation with model: {}", translation_model);
        let translation_start = std::time::Instant::now();

        // Decode image dimensions for coordinate denormalization
        let bubble_img = image::load_from_memory(bubble_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to decode bubble image: {}", e))?;
        let bubble_width = bubble_img.width();
        let bubble_height = bubble_img.height();
        let aspect_ratio = bubble_width as f32 / bubble_height as f32;
        let bubble_shape = if aspect_ratio > 1.3 {
            "WIDE/HORIZONTAL"
        } else if aspect_ratio < 0.7 {
            "TALL/VERTICAL"
        } else {
            "SQUARE/BALANCED"
        };
        debug!("Bubble dimensions: {}x{} (aspect: {:.2}, shape: {})",
            bubble_width, bubble_height, aspect_ratio, bubble_shape);

        let prompt = format!(r#"Analyze this manga/comic speech bubble image and extract:

BUBBLE DIMENSIONS: {}x{} pixels (aspect ratio: {:.2}, shape: {})
IMPORTANT: Optimize line breaks for this bubble shape!
- TALL/VERTICAL bubbles: Use 2-3 SHORT lines (break at natural phrases/commas)
- WIDE/HORIZONTAL bubbles: Use 1-2 LONG lines (fewer breaks)
- SQUARE bubbles: Balance line lengths evenly

LANGUAGE DETECTION: This text may be in Japanese, Chinese (Simplified/Traditional), Korean, or other languages.
Read ALL visible text carefully - do not miss any characters, even small or faint ones.

1. ORIGINAL TEXT (original_text):
   - Transcribe EVERY character you see in the image EXACTLY as written
   - Include ALL text: main dialogue, small notes, sound effects, everything visible
   - Maintain the original character order and spacing
   - Double-check you haven't missed any characters, especially small furigana or side text

2. ENGLISH TRANSLATION (english_translation):
   - Translate the COMPLETE original text to natural, fluent English
   - CRITICAL: Ensure EVERY word and phrase from the original is translated - DO NOT omit anything
   - LINE BREAKS: Optimize for the bubble shape specified above (TALL/VERTICAL/WIDE)
     * Break at natural phrase boundaries (commas, conjunctions, pauses)
     * For TALL bubbles: prefer 2-3 short lines
     * For WIDE bubbles: prefer 1-2 long lines
     * Use \n separators between lines
   - Use natural English phrasing while keeping the FULL meaning and context
   - For manga/comic dialogue: keep translations concise but COMPLETE to fit visual space
   - Consider context: informal speech, formal language, character personality, emotional tone
   - If there are multiple speakers or text elements, translate ALL of them in sequence
   - VERIFY: Count the original phrases and ensure all are present in translation

3. TEXT COLOR (font_color):
   - RGB format 'R,G,B' (e.g. '0,0,0' for black, '255,255,255' for white)
   - Choose a color that provides good contrast against the background
   - Most manga uses black text on white backgrounds

4. BACKGROUND ANALYSIS (redraw_bg_required):
   - Set to FALSE (default) if background is:
     * Single solid color (white, black, gray, or any uniform color)
     * Very simple uniform pattern
     * No gradients, shadows, or texture variations
   - Set to TRUE ONLY if background has:
     * Multiple different colors or gradients
     * Detailed artwork, illustrations, or photographs
     * Shadows, lighting effects, or complex textures
     * Visual elements or objects behind the text
   - PREFER FALSE for simple backgrounds to save processing cost

FINAL REQUIREMENTS - VERIFY BEFORE SUBMITTING:
✓ original_text: ALL characters transcribed exactly as shown (no missing text!)
✓ english_translation: COMPLETE translation with ALL phrases (count and verify!)
✓ Line breaks optimized for bubble shape with \n separators
✓ Text color chosen for good contrast (usually '0,0,0' for black)
✓ Background analysis accurate (prefer FALSE for simple backgrounds)

CRITICAL: Read the image carefully and ensure NO text is missed in transcription or translation.
"#, bubble_width, bubble_height, aspect_ratio, bubble_shape);

        let base64_image = general_purpose::STANDARD.encode(bubble_bytes);
        trace!("Bubble image encoded to base64: {} bytes → {} bytes",
            bubble_bytes.len(), base64_image.len());

        let max_retries = self.config.max_retries();
        for attempt in 1..=max_retries {
            // Get a healthy key from the pool
            let (key_index, api_key) = match self.api_key_pool.get_healthy_key().await {
                Some(key_pair) => key_pair,
                None => {
                    warn!("No healthy API keys available (attempt {}/{})", attempt, max_retries);
                    if attempt < max_retries {
                        sleep(Duration::from_secs(2)).await;
                        continue;
                    } else {
                        return Err(anyhow::anyhow!("All API keys are unhealthy"));
                    }
                }
            };

            debug!("Translation attempt {}/{} (using API key index {})",
                attempt, max_retries, key_index);

            let request = schema::build_translation_request(
                prompt.to_string(),
                base64_image.clone(),
                schema::create_translation_schema(),
            );

            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                translation_model, api_key
            );

            debug!("Sending request to Gemini API (model: {}, key: ***REDACTED***)", translation_model);

            match self.client.post(&url).json(&request).send().await {
                Ok(response) => {
                    trace!("Received API response with status: {}", response.status());
                    let response_text = response.text().await?;
                    trace!("Response body length: {} bytes", response_text.len());

                    match serde_json::from_str::<GenerateContentResponse>(&response_text) {
                        Ok(gen_response) => {
                            if let Some(candidates) = gen_response.candidates {
                                if let Some(first_candidate) = candidates.first() {
                                    if let Some(text_part) = first_candidate.content.parts.first() {
                                        if let Some(text) = &text_part.text {
                                            trace!("Parsing translation JSON: {}", text);
                                            match serde_json::from_str::<TextTranslation>(text) {
                                                Ok(translation) => {
                                                    if Self::validate_translation(&translation) {
                                                        // Record success for this API key
                                                        self.api_key_pool.record_success(key_index).await;

                                                        let total_time = translation_start.elapsed();
                                                        debug!("✅ [TRANSLATION] Success in {:.2}ms (key {}): '{}' → '{}'",
                                                            total_time.as_secs_f64() * 1000.0,
                                                            key_index,
                                                            translation.original_text.chars().take(30).collect::<String>(),
                                                            translation.english_translation.chars().take(30).collect::<String>());
                                                        debug!("  Font: {}, Color: {}, Redraw: {}",
                                                            translation.font_family,
                                                            translation.font_color,
                                                            translation.redraw_bg_required);
                                                        return Ok(translation);
                                                    }
                                                    warn!("Translation validation failed (key {})", key_index);
                                                    // Record failure for validation failure
                                                    self.api_key_pool.record_failure(key_index).await;
                                                }
                                                Err(e) => {
                                                    warn!("Failed to parse translation (key {}): {}", key_index, e);
                                                    self.api_key_pool.record_failure(key_index).await;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse response (key {}): {}", key_index, e);
                            self.api_key_pool.record_failure(key_index).await;
                        }
                    }
                }
                Err(e) => {
                    warn!("Request failed (attempt {}, key {}): {}", attempt, key_index, e);
                    self.api_key_pool.record_failure(key_index).await;
                }
            }

            if attempt < max_retries {
                sleep(Duration::from_secs(1)).await;
            }
        }

        Err(anyhow::anyhow!("All translation attempts failed"))
    }

    fn validate_translation(translation: &TextTranslation) -> bool {
        // Check for empty translation
        if translation.english_translation.trim().is_empty() {
            warn!("⚠️ Translation validation failed: empty translation");
            return false;
        }

        // Note: Removed overly simplistic keyword check that caused false positives
        // Valid translations like "I cannot go" or "Sorry for the wait" should not be rejected.
        // If Gemini fails, it will typically return empty string or malformed JSON.

        // Font family will use default 'arial' via serde default
        // No need to validate font_family anymore

        true
    }

    pub async fn remove_text_from_image(&self, image_bytes: &[u8], model: Option<&str>) -> Result<Vec<u8>> {
        let image_gen_model = model.unwrap_or(self.config.image_gen_model());
        info!("🎨 [IMAGE_GEN] Starting AI text removal with model: {}", image_gen_model);
        let gen_start = std::time::Instant::now();

        // Get a healthy key from the pool
        let (key_index, api_key) = self.api_key_pool.get_healthy_key().await
            .ok_or_else(|| anyhow::anyhow!("No healthy API keys available for image generation"))?;

        debug!("Using API key index {} for image generation", key_index);

        let cleaning_prompt = "Cleanly remove only any language text or characters";
        let base64_image = general_purpose::STANDARD.encode(image_bytes);
        debug!("Image encoded for AI generation: {} bytes", base64_image.len());

        let request = schema::build_image_gen_request(
            cleaning_prompt.to_string(),
            base64_image,
        );

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            image_gen_model, api_key
        );

        let response = match self.client.post(&url).json(&request).send().await {
            Ok(resp) => resp,
            Err(e) => {
                self.api_key_pool.record_failure(key_index).await;
                return Err(e.into());
            }
        };
        trace!("Received image generation response with status: {}", response.status());

        let response_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                self.api_key_pool.record_failure(key_index).await;
                return Err(e.into());
            }
        };
        trace!("Image generation response body length: {} bytes", response_text.len());

        let gen_response: GenerateContentResponse = match serde_json::from_str(&response_text) {
            Ok(resp) => resp,
            Err(e) => {
                self.api_key_pool.record_failure(key_index).await;
                return Err(e).context("Failed to parse image generation response");
            }
        };

        if let Some(candidates) = gen_response.candidates {
            for candidate in candidates {
                for part in candidate.content.parts {
                    if let Some(inline_data) = part.inline_data {
                        match general_purpose::STANDARD.decode(&inline_data.data) {
                            Ok(image_bytes) => {
                                // Record success for this API key
                                self.api_key_pool.record_success(key_index).await;

                                let total_time = gen_start.elapsed();
                                info!("✅ [IMAGE_GEN] Completed in {:.2}ms (key {}): generated {} bytes",
                                    total_time.as_secs_f64() * 1000.0, key_index, image_bytes.len());
                                return Ok(image_bytes);
                            }
                            Err(e) => {
                                self.api_key_pool.record_failure(key_index).await;
                                return Err(e.into());
                            }
                        }
                    }
                }
            }
        }

        self.api_key_pool.record_failure(key_index).await;
        Err(anyhow::anyhow!("Failed to get cleaned image"))
    }
}

// Trait implementation for Translator
#[async_trait]
impl Translator for TranslationService {
    async fn translate_bubble(
        &self,
        bubble_bytes: &[u8],
        model: Option<&str>,
    ) -> Result<TextTranslation> {
        self.translate_bubble(bubble_bytes, model).await
    }

    async fn remove_text_from_image(
        &self,
        image_bytes: &[u8],
        model: Option<&str>,
    ) -> Result<Vec<u8>> {
        self.remove_text_from_image(image_bytes, model).await
    }
}
