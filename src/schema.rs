/// Gemini API Schema Definitions
///
/// This module contains all request/response structures for the Gemini Vision API.
/// Schemas are used for structured JSON output from the Gemini API.

use serde::{Deserialize, Serialize};

// ============================================================================
// GEMINI API REQUEST STRUCTURES
// ============================================================================

#[derive(Debug, Serialize)]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize)]
pub struct Content {
    pub parts: Vec<ContentPart>,
}

#[derive(Debug, Serialize)]
pub struct ContentPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<InlineData>,
}

#[derive(Debug, Serialize)]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_modalities: Option<Vec<String>>,
}

// ============================================================================
// GEMINI API RESPONSE STRUCTURES
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    pub candidates: Option<Vec<Candidate>>,
    #[allow(dead_code)]
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: ContentResponse,
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContentResponse {
    pub parts: Vec<ContentPartResponse>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentPartResponse {
    pub text: Option<String>,
    pub inline_data: Option<InlineDataResponse>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineDataResponse {
    #[allow(dead_code)]
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[allow(dead_code)]
    pub prompt_token_count: Option<u32>,
    #[allow(dead_code)]
    pub candidates_token_count: Option<u32>,
    #[allow(dead_code)]
    pub total_token_count: Option<u32>,
}

// ============================================================================
// TRANSLATION SCHEMA FACTORY
// ============================================================================

/// Creates the JSON schema for manga translation responses
///
/// This schema enforces structured output from Gemini API with:
/// - original_text: Transcription of source text
/// - english_translation: Translated English text
/// - font_color: RGB color for text rendering
/// - redraw_bg_required: Whether background needs AI redrawing
pub fn create_translation_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "original_text": {
                "type": "string",
                "description": "Original text on the image - transcribe EVERY character exactly as shown"
            },
            "english_translation": {
                "type": "string",
                "description": "English translation preserving line breaks with \\n characters"
            },
            "font_color": {
                "type": "string",
                "description": "Text color in RGB format like '0,0,0' for black or '255,255,255' for white. Choose color with good contrast against the background."
            },
            "redraw_bg_required": {
                "type": "boolean",
                "description": "True if background is complex and requires AI redrawing. False if background is simple (solid color or basic pattern)."
            }
        },
        "required": ["original_text", "english_translation", "font_color", "redraw_bg_required"]
    })
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Build a complete Gemini API request for translation
pub fn build_translation_request(
    prompt: String,
    image_base64: String,
    schema: serde_json::Value,
) -> GenerateContentRequest {
    GenerateContentRequest {
        contents: vec![Content {
            parts: vec![
                ContentPart {
                    text: Some(prompt),
                    inline_data: None,
                },
                ContentPart {
                    text: None,
                    inline_data: Some(InlineData {
                        mime_type: "image/png".to_string(),
                        data: image_base64,
                    }),
                },
            ],
        }],
        generation_config: Some(GenerationConfig {
            response_mime_type: Some("application/json".to_string()),
            response_schema: Some(schema),
            response_modalities: None,
        }),
    }
}

/// Build a Gemini API request for image generation (background redrawing)
pub fn build_image_gen_request(
    prompt: String,
    image_base64: String,
) -> GenerateContentRequest {
    GenerateContentRequest {
        contents: vec![Content {
            parts: vec![
                ContentPart {
                    text: Some(prompt),
                    inline_data: None,
                },
                ContentPart {
                    text: None,
                    inline_data: Some(InlineData {
                        mime_type: "image/png".to_string(),
                        data: image_base64,
                    }),
                },
            ],
        }],
        generation_config: Some(GenerationConfig {
            response_mime_type: None,
            response_schema: None,
            response_modalities: Some(vec!["TEXT".to_string(), "IMAGE".to_string()]),
        }),
    }
}
