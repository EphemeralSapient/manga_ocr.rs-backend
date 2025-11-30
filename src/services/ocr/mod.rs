// OCR Service Module - Local CJK OCR for Japanese text recognition
// CPU-only ONNX inference for Japanese manga text recognition

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use once_cell::sync::OnceCell;
use ort::{session::Session, value::Value};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Global singleton OCR instance for warm-up and reuse
static OCR_INSTANCE: OnceCell<Arc<OcrService>> = OnceCell::new();

/// Embedded OCR model bytes (only for local builds, not CI)
#[cfg(ocr_model_embedded)]
const OCR_MODEL_BYTES: &[u8] = include_bytes!("../../../models/ocr/ocr.onnx");

#[cfg(ocr_model_embedded)]
const OCR_VOCAB_BYTES: &[u8] = include_bytes!("../../../models/ocr/cjk_vocab.txt");

/// Placeholder for CI builds (model loaded from disk)
#[cfg(not(ocr_model_embedded))]
const OCR_MODEL_BYTES: &[u8] = &[];

#[cfg(not(ocr_model_embedded))]
const OCR_VOCAB_BYTES: &[u8] = &[];

/// OCR model input dimensions
const TARGET_HEIGHT: u32 = 60;
const MIN_WIDTH: u32 = 10;

/// OCR Service for local Japanese text recognition
pub struct OcrService {
    session: Mutex<Session>,
    vocab: HashMap<usize, String>,
    blank_index: usize,
}

impl OcrService {
    /// Initialize OCR service with CPU-only ONNX runtime
    /// First tries embedded model (local builds), then falls back to disk (CI builds)
    pub fn new(models_dir: &Path) -> Result<Self> {
        let model_path = models_dir.join("ocr").join("ocr.onnx");
        let vocab_path = models_dir.join("ocr").join("cjk_vocab.txt");

        // Try embedded model first (for local builds)
        let (session, vocab) = if OCR_MODEL_BYTES.len() > 10_000 && OCR_VOCAB_BYTES.len() > 1_000 {
            info!("Loading OCR model from embedded bytes ({:.1} MB)", OCR_MODEL_BYTES.len() as f64 / 1_048_576.0);

            let session = Session::builder()?
                .with_intra_threads(4)?
                .with_inter_threads(2)?
                .commit_from_memory(OCR_MODEL_BYTES)
                .context("Failed to load embedded OCR model")?;

            let vocab = Self::load_vocabulary_from_bytes(OCR_VOCAB_BYTES)?;

            (session, vocab)
        } else {
            // Fall back to disk loading (CI builds or embedded not available)
            if !model_path.exists() {
                anyhow::bail!(
                    "OCR model not found at: {}. Local OCR feature is unavailable.",
                    model_path.display()
                );
            }

            if !vocab_path.exists() {
                anyhow::bail!(
                    "OCR vocabulary not found at: {}. Local OCR feature is unavailable.",
                    vocab_path.display()
                );
            }

            info!("Loading OCR model from disk: {}", model_path.display());

            let session = Session::builder()?
                .with_intra_threads(4)?
                .with_inter_threads(2)?
                .commit_from_file(&model_path)
                .context("Failed to load OCR ONNX model from disk")?;

            let vocab = Self::load_vocabulary(&vocab_path)?;

            (session, vocab)
        };

        let blank_index = vocab.len().saturating_sub(1);

        info!(
            "OCR service initialized: vocab_size={}, blank_index={}",
            vocab.len(),
            blank_index
        );

        Ok(Self {
            session: Mutex::new(session),
            vocab,
            blank_index,
        })
    }

    /// Load vocabulary from file (format: index\tchar)
    fn load_vocabulary(vocab_path: &Path) -> Result<HashMap<usize, String>> {
        let content = std::fs::read_to_string(vocab_path)
            .context("Failed to read vocabulary file")?;

        let mut vocab = HashMap::new();

        // Build default ASCII mapping (indices 0-96)
        for i in 0..97 {
            if (32..127).contains(&i) {
                vocab.insert(i, (i as u8 as char).to_string());
            } else if i == 0 {
                vocab.insert(i, " ".to_string());
            }
        }

        // Parse vocab file
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some((idx_str, char_str)) = line.split_once('\t') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    let ch = if char_str == "<space>" {
                        " ".to_string()
                    } else if char_str == "<blank>" {
                        continue; // Skip blank token
                    } else {
                        char_str.to_string()
                    };
                    vocab.insert(idx, ch);
                }
            }
        }

        debug!("Loaded {} vocabulary entries from file", vocab.len());
        Ok(vocab)
    }

    /// Load vocabulary from bytes (for embedded vocab)
    fn load_vocabulary_from_bytes(vocab_bytes: &[u8]) -> Result<HashMap<usize, String>> {
        let content = std::str::from_utf8(vocab_bytes)
            .context("Failed to parse vocabulary bytes as UTF-8")?;

        let mut vocab = HashMap::new();

        // Build default ASCII mapping (indices 0-96)
        for i in 0..97 {
            if (32..127).contains(&i) {
                vocab.insert(i, (i as u8 as char).to_string());
            } else if i == 0 {
                vocab.insert(i, " ".to_string());
            }
        }

        // Parse vocab content
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some((idx_str, char_str)) = line.split_once('\t') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    let ch = if char_str == "<space>" {
                        " ".to_string()
                    } else if char_str == "<blank>" {
                        continue; // Skip blank token
                    } else {
                        char_str.to_string()
                    };
                    vocab.insert(idx, ch);
                }
            }
        }

        debug!("Loaded {} vocabulary entries from embedded bytes", vocab.len());
        Ok(vocab)
    }

    /// Preprocess image region for OCR
    /// - Convert to RGB
    /// - Resize to target height (60px) maintaining aspect ratio
    /// - Normalize to [0, 1] float32
    /// - Return tensor in [1, 3, H, W] format
    fn preprocess_image(&self, image: &DynamicImage) -> Result<(Array4<f32>, i32)> {
        let (w, h) = image.dimensions();

        // Resize to target height maintaining aspect ratio
        let scale = TARGET_HEIGHT as f32 / h as f32;
        let new_w = ((w as f32 * scale) as u32).max(MIN_WIDTH);

        let resized = image.resize_exact(
            new_w,
            TARGET_HEIGHT,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to RGB
        let rgb = resized.to_rgb8();

        // Create tensor [1, 3, H, W]
        let mut tensor = Array4::<f32>::zeros((1, 3, TARGET_HEIGHT as usize, new_w as usize));

        for y in 0..TARGET_HEIGHT as usize {
            for x in 0..new_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                tensor[[0, 0, y, x]] = pixel[0] as f32 / 255.0; // R
                tensor[[0, 1, y, x]] = pixel[1] as f32 / 255.0; // G
                tensor[[0, 2, y, x]] = pixel[2] as f32 / 255.0; // B
            }
        }

        // seq_length = width / 4 (LSTM stride)
        let seq_length = (new_w / 4) as i32;

        Ok((tensor, seq_length))
    }

    /// CTC greedy decode
    /// - Collapse repeated characters
    /// - Remove blank tokens
    fn ctc_decode(&self, logits: &[f32], seq_len: usize, vocab_size: usize) -> (String, f32) {
        let mut decoded = Vec::new();
        let mut confidences = Vec::new();
        let mut prev_idx: Option<usize> = None;

        for t in 0..seq_len {
            // Find best class at this timestep
            let offset = t * vocab_size;
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;

            for i in 0..vocab_size {
                let val = logits[offset + i];
                if val > best_val {
                    best_val = val;
                    best_idx = i;
                }
            }

            // Collapse repeats and remove blanks
            if best_idx != self.blank_index && Some(best_idx) != prev_idx {
                decoded.push(best_idx);
                // Convert log prob to confidence
                let confidence = best_val.exp().min(1.0);
                confidences.push(confidence);
            }

            prev_idx = Some(best_idx);
        }

        // Convert indices to characters
        let text: String = decoded
            .iter()
            .map(|&idx| self.vocab.get(&idx).map(|s| s.as_str()).unwrap_or("?"))
            .collect();

        let avg_confidence = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        };

        (text, avg_confidence)
    }

    /// Run OCR on a single image region
    /// Returns (recognized_text, confidence)
    pub fn recognize(&self, image: &DynamicImage) -> Result<(String, f32)> {
        let (tensor, seq_length) = self.preprocess_image(image)?;

        // Prepare inputs - use into_raw_vec_and_offset for owned data
        let data_shape: Vec<usize> = tensor.shape().to_vec();
        let (data_flat, _offset) = tensor.into_raw_vec_and_offset();

        // Create ONNX Runtime values - convert shape to required format
        let shape_arr: [usize; 4] = [
            data_shape[0],
            data_shape[1],
            data_shape[2],
            data_shape[3],
        ];
        let data_value = Value::from_array((shape_arr, data_flat))?;
        let seq_lengths_value = Value::from_array(([1usize], vec![seq_length]))?;

        // Run inference - extract data while session is held, then release
        let (dims, logits) = {
            let mut session = self.session.lock();
            let outputs = session.run(ort::inputs![
                "data" => data_value,
                "seq_lengths" => seq_lengths_value
            ])?;

            // Extract logits - first try named output, then first available
            let (shape, logits_data) = if let Some(output) = outputs.get("output") {
                output.try_extract_tensor::<f32>()?
            } else {
                // Try getting first output
                let first_key = outputs.keys().next()
                    .context("No outputs from OCR model")?;
                outputs[first_key].try_extract_tensor::<f32>()?
            };

            // Copy data before releasing session
            let dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            let logits: Vec<f32> = logits_data.to_vec();
            (dims, logits)
        };

        // Determine actual sequence length and vocab size from output shape
        let (actual_seq_len, vocab_size) = if dims.len() == 3 {
            if dims[1] == 1 {
                // [S, 1, V]
                (dims[0], dims[2])
            } else {
                // [1, S, V]
                (dims[1], dims[2])
            }
        } else if dims.len() == 2 {
            // [S, V]
            (dims[0], dims[1])
        } else {
            anyhow::bail!("Unexpected logits shape: {:?}", dims);
        };

        // CTC decode
        let (text, confidence) = self.ctc_decode(&logits, actual_seq_len, vocab_size);

        debug!("OCR result: '{}' (confidence: {:.2})", text, confidence);
        Ok((text, confidence))
    }

    /// Run OCR on multiple image regions in parallel
    pub fn recognize_batch(&self, images: &[DynamicImage]) -> Vec<Result<(String, f32)>> {
        // Process sequentially since we have a single session
        // Could be parallelized with session pool if needed
        images
            .iter()
            .map(|img| self.recognize(img))
            .collect()
    }
}

/// Get or initialize the global OCR service singleton
pub fn get_ocr_service(models_dir: &Path) -> Result<Arc<OcrService>> {
    OCR_INSTANCE
        .get_or_try_init(|| {
            info!("Initializing global OCR service (warm-up)");
            OcrService::new(models_dir).map(Arc::new)
        })
        .map(Arc::clone)
}

/// Check if OCR service is available (embedded or on disk)
pub fn is_ocr_available(models_dir: &Path) -> bool {
    // Check embedded models first (for local builds)
    if OCR_MODEL_BYTES.len() > 10_000 && OCR_VOCAB_BYTES.len() > 1_000 {
        return true;
    }

    // Fall back to disk check
    let model_path = models_dir.join("ocr").join("ocr.onnx");
    let vocab_path = models_dir.join("ocr").join("cjk_vocab.txt");
    model_path.exists() && vocab_path.exists()
}

/// Warm up the OCR service by loading models
pub fn warmup_ocr_service(models_dir: &Path) -> Result<()> {
    if !is_ocr_available(models_dir) {
        warn!("OCR models not available (neither embedded nor on disk), skipping warm-up");
        return Ok(());
    }

    let _ = get_ocr_service(models_dir)?;
    info!("OCR service warmed up successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_loading_from_bytes() {
        let content = b"0\t<space>\n32\t \n33\t!\n97\ta\n12345\t\xe6\xbc\xa2";
        let vocab = OcrService::load_vocabulary_from_bytes(content).unwrap();
        assert!(vocab.contains_key(&32));
        assert!(vocab.contains_key(&12345));
    }
}
