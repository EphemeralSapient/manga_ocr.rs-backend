// Cerebras API Client for fast translation
// Uses Cerebras Cloud SDK compatible API (OpenAI-compatible endpoint)
// ~3000 tokens/sec throughput - batches all text into single call

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

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
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_completion_tokens: u32,
    temperature: f32,
    top_p: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
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
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl CerebrasClient {
    /// Create a new Cerebras client
    pub fn new(api_key: String) -> Result<Self> {
        if api_key.is_empty() {
            anyhow::bail!("Cerebras API key is required");
        }

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120)) // 2 minute timeout
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

    /// Translate Japanese text to target language
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

        // Build the combined text with region markers
        let combined_text: String = texts
            .iter()
            .map(|(id, text)| format!("[{}] {}", id, text))
            .collect::<Vec<_>>()
            .join("\n");

        let region_ids: Vec<usize> = texts.iter().map(|(id, _)| *id).collect();

        info!(
            "Cerebras: Translating {} regions ({} chars) to {}",
            texts.len(),
            combined_text.len(),
            target_lang
        );

        let system_prompt = format!(
            "Translate Japanese to {} while rectifying any OCR mistakes and keeping it natural. \
             Do not add extra wording or actions. \
             Input has multiple lines prefixed with [ID]. \
             Output each translation on its own line prefixed with the same [ID]. \
             Keep the [ID] prefix exactly as given. \
             If a line is unclear, translate your best interpretation.",
            target_lang
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
                    content: combined_text,
                },
            ],
            max_completion_tokens: 65536,
            temperature: 0.7,
            top_p: 1.0,
            reasoning_effort: Some("medium".to_string()),
        };

        let start = Instant::now();

        let response = self
            .http_client
            .post(CEREBRAS_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Cerebras API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Cerebras API error: {} - {}", status, error_text);
        }

        let response_data: ChatCompletionResponse = response
            .json()
            .await
            .context("Failed to parse Cerebras response")?;

        let elapsed = start.elapsed();

        if let Some(usage) = &response_data.usage {
            let tokens_per_sec = usage.completion_tokens as f64 / elapsed.as_secs_f64();
            info!(
                "Cerebras: {} tokens in {:.1}s ({:.0} tok/s)",
                usage.total_tokens,
                elapsed.as_secs_f64(),
                tokens_per_sec
            );
        }

        // Parse the response - extract translations by ID
        let content = response_data
            .choices
            .first()
            .map(|c| c.message.content.as_str())
            .unwrap_or("");

        let mut translations: Vec<(usize, String)> = Vec::new();

        // Parse lines looking for [ID] prefixes
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Try to extract [ID] prefix
            if let Some(rest) = line.strip_prefix('[') {
                if let Some(end_bracket) = rest.find(']') {
                    if let Ok(id) = rest[..end_bracket].parse::<usize>() {
                        let translation = rest[end_bracket + 1..].trim().to_string();
                        translations.push((id, translation));
                    }
                }
            }
        }

        // If parsing failed, try to match by order
        if translations.is_empty() && !content.is_empty() {
            warn!("Cerebras: Failed to parse [ID] format, falling back to order matching");
            let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
            for (i, line) in lines.iter().enumerate() {
                if i < region_ids.len() {
                    translations.push((region_ids[i], line.trim().to_string()));
                }
            }
        }

        // Fill in missing translations
        let translated_ids: std::collections::HashSet<usize> =
            translations.iter().map(|(id, _)| *id).collect();

        for id in &region_ids {
            if !translated_ids.contains(id) {
                warn!("Cerebras: Missing translation for region {}", id);
                // Use original text as fallback
                if let Some((_, original)) = texts.iter().find(|(rid, _)| rid == id) {
                    translations.push((*id, original.clone()));
                }
            }
        }

        debug!("Cerebras: Translated {} regions", translations.len());
        Ok(translations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_response() {
        let content = "[0] Hello world\n[1] How are you?\n[2] Good morning";
        let lines: Vec<(usize, String)> = content
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix('[') {
                    if let Some(end_bracket) = rest.find(']') {
                        if let Ok(id) = rest[..end_bracket].parse::<usize>() {
                            let translation = rest[end_bracket + 1..].trim().to_string();
                            return Some((id, translation));
                        }
                    }
                }
                None
            })
            .collect();

        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], (0, "Hello world".to_string()));
        assert_eq!(lines[1], (1, "How are you?".to_string()));
        assert_eq!(lines[2], (2, "Good morning".to_string()));
    }
}
