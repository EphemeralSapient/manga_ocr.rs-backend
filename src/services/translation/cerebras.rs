// Cerebras API Client for fast translation
// Uses Cerebras Cloud SDK compatible API with Structured Outputs
// ~3000 tokens/sec throughput - batches all text into single call

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, error};

/// Retry configuration
const MAX_RETRIES: u32 = 3;
const INITIAL_RETRY_DELAY_MS: u64 = 1000;
const MAX_RETRY_DELAY_MS: u64 = 10000;

/// Cerebras API endpoint
const CEREBRAS_API_URL: &str = "https://api.cerebras.ai/v1/chat/completions";

/// Default model - gpt-oss-120b for best quality
const DEFAULT_MODEL: &str = "gpt-oss-120b";

/// Cerebras API client
pub struct CerebrasClient {
    api_key: String,
    http_client: reqwest::Client,
    model: String,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct JsonSchema {
    name: String,
    strict: bool,
    schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    json_schema: JsonSchema,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    max_completion_tokens: u32,
    temperature: f32,
    top_p: f32,
    reasoning_effort: String,
    response_format: ResponseFormat,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Structured output schema for translations
#[derive(Debug, Deserialize)]
struct TranslationsResponse {
    translations: Vec<TranslationItem>,
}

#[derive(Debug, Deserialize)]
struct TranslationItem {
    id: usize,
    text: String,
}

impl CerebrasClient {
    /// Create a new Cerebras client
    pub fn new(api_key: String) -> Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("Cerebras API key is required");
        }

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .pool_max_idle_per_host(5)
            .pool_idle_timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            api_key,
            http_client,
            model: DEFAULT_MODEL.to_string(),
        })
    }

    /// Create client from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("CEREBRAS_API_KEY")
            .context("CEREBRAS_API_KEY environment variable not set")?;
        Self::new(api_key)
    }

    /// Build the JSON schema for structured output
    fn build_translation_schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "The region ID from the input"
                            },
                            "text": {
                                "type": "string",
                                "description": "The translated text"
                            }
                        },
                        "required": ["id", "text"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["translations"],
            "additionalProperties": false
        })
    }

    /// Translate Japanese text to target language using structured outputs
    /// Batches all input lines into a single API call for efficiency
    ///
    /// # Arguments
    /// * `texts` - Vector of (region_id, original_text) pairs
    /// * `target_language` - Target language (default: "English")
    ///
    /// # Returns
    /// Vector of (region_id, translated_text) pairs
    pub async fn translate_batch(
        &self,
        texts: Vec<(usize, String)>,
        target_language: Option<&str>,
    ) -> Result<Vec<(usize, String)>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let target_lang = target_language.unwrap_or("English");

        // Build the input as JSON array for clarity
        let input_items: Vec<serde_json::Value> = texts
            .iter()
            .map(|(id, text)| json!({"id": id, "text": text}))
            .collect();

        let input_json = serde_json::to_string(&input_items)
            .context("Failed to serialize input")?;

        info!(
            "Cerebras: Translating {} regions to {} (structured output)",
            texts.len(),
            target_lang
        );

        let system_prompt = format!(
            "You are a Japanese to {} translator. \
             Translate each text naturally while fixing any OCR errors. \
             Do not add extra words or explanations. \
             Return translations in the exact JSON format specified.",
            target_lang
        );

        let user_prompt = format!(
            "Translate each Japanese text to {}. Input:\n{}",
            target_lang, input_json
        );

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            stream: false,
            max_completion_tokens: 65536,
            temperature: 1.0,
            top_p: 1.0,
            reasoning_effort: "medium".to_string(),
            response_format: ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: JsonSchema {
                    name: "translation_response".to_string(),
                    strict: true,
                    schema: Self::build_translation_schema(),
                },
            },
        };

        let start = Instant::now();

        // Retry loop with exponential backoff
        let mut last_error: Option<anyhow::Error> = None;
        let mut retry_delay_ms = INITIAL_RETRY_DELAY_MS;
        
        let response_data = 'retry: {
            for attempt in 0..=MAX_RETRIES {
                if attempt > 0 {
                    warn!(
                        "Cerebras: Retry attempt {} after {}ms delay",
                        attempt, retry_delay_ms
                    );
                    tokio::time::sleep(Duration::from_millis(retry_delay_ms)).await;
                    // Exponential backoff with cap
                    retry_delay_ms = (retry_delay_ms * 2).min(MAX_RETRY_DELAY_MS);
                }

                let send_result = self
                    .http_client
                    .post(CEREBRAS_API_URL)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .await;

                let response = match send_result {
                    Ok(resp) => resp,
                    Err(e) => {
                        let is_retryable = e.is_timeout() || e.is_connect();
                        error!(
                            "Cerebras: Request failed (attempt {}): {} (retryable: {})",
                            attempt + 1, e, is_retryable
                        );
                        if is_retryable && attempt < MAX_RETRIES {
                            last_error = Some(e.into());
                            continue;
                        }
                        return Err(e).context("Failed to send request to Cerebras API");
                    }
                };

                let status = response.status();
                
                // Check for retryable HTTP status codes (5xx, 429)
                if status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    let error_text = response.text().await.unwrap_or_default();
                    error!(
                        "Cerebras: Server error {} (attempt {}): {}",
                        status, attempt + 1, error_text
                    );
                    if attempt < MAX_RETRIES {
                        last_error = Some(anyhow::anyhow!("Cerebras API error: {} - {}", status, error_text));
                        continue;
                    }
                    anyhow::bail!("Cerebras API error after {} retries: {} - {}", MAX_RETRIES + 1, status, error_text);
                }

                if !status.is_success() {
                    let error_text = response.text().await.unwrap_or_default();
                    anyhow::bail!("Cerebras API error: {} - {}", status, error_text);
                }

                match response.json::<ChatCompletionResponse>().await {
                    Ok(data) => break 'retry data,
                    Err(e) => {
                        error!("Cerebras: Failed to parse response (attempt {}): {}", attempt + 1, e);
                        if attempt < MAX_RETRIES {
                            last_error = Some(e.into());
                            continue;
                        }
                        return Err(e).context("Failed to parse Cerebras response");
                    }
                }
            }
            
            // All retries exhausted
            return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Unknown error after retries")));
        };

        let elapsed = start.elapsed();

        if let Some(usage) = &response_data.usage {
            let tokens_per_sec = usage.completion_tokens as f64 / elapsed.as_secs_f64();
            info!(
                "Cerebras: {} tokens in {:.2}s ({:.0} tok/s)",
                usage.total_tokens,
                elapsed.as_secs_f64(),
                tokens_per_sec
            );
        }

        // Parse the structured JSON response
        let content = response_data
            .choices
            .first()
            .map(|c| c.message.content.as_str())
            .unwrap_or("{}");

        let translations_response: TranslationsResponse = serde_json::from_str(content)
            .context("Failed to parse structured translation response")?;

        // Convert to output format
        let mut translations: Vec<(usize, String)> = translations_response
            .translations
            .into_iter()
            .map(|item| (item.id, item.text))
            .collect();

        // Fill in any missing translations with original text
        let translated_ids: std::collections::HashSet<usize> =
            translations.iter().map(|(id, _)| *id).collect();

        for (id, original) in &texts {
            if !translated_ids.contains(id) {
                warn!("Cerebras: Missing translation for region {}, using original", id);
                translations.push((*id, original.clone()));
            }
        }

        debug!("Cerebras: Successfully translated {} regions", translations.len());
        Ok(translations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_structured_response() {
        let json_content = r#"{"translations":[{"id":0,"text":"Hello world"},{"id":1,"text":"How are you?"},{"id":2,"text":"Good morning"}]}"#;

        let response: TranslationsResponse = serde_json::from_str(json_content).unwrap();

        assert_eq!(response.translations.len(), 3);
        assert_eq!(response.translations[0].id, 0);
        assert_eq!(response.translations[0].text, "Hello world");
        assert_eq!(response.translations[1].id, 1);
        assert_eq!(response.translations[1].text, "How are you?");
        assert_eq!(response.translations[2].id, 2);
        assert_eq!(response.translations[2].text, "Good morning");
    }

    #[test]
    fn test_schema_generation() {
        let schema = CerebrasClient::build_translation_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema.get("required").is_some());
    }
}
