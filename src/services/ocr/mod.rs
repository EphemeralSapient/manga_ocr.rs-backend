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

/// Bounding box for detected text line
#[derive(Debug, Clone)]
struct TextLine {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// Detect if image contains vertical text using projection profile analysis
/// More conservative detection - requires strong evidence of vertical text
fn is_vertical_text(image: &DynamicImage) -> bool {
    let gray = image.to_luma8();
    let (w, h) = gray.dimensions();

    if w == 0 || h == 0 {
        return false;
    }

    // Very wide images are definitely horizontal
    if w > h * 2 {
        return false;
    }

    // Compute horizontal and vertical projections
    let mut h_proj = vec![0u64; h as usize];
    let mut v_proj = vec![0u64; w as usize];

    for y in 0..h {
        for x in 0..w {
            let pixel = gray.get_pixel(x, y).0[0];
            let val = (255 - pixel) as u64;
            h_proj[y as usize] += val;
            v_proj[x as usize] += val;
        }
    }

    let h_mean: f64 = h_proj.iter().sum::<u64>() as f64 / h_proj.len().max(1) as f64;
    let v_mean: f64 = v_proj.iter().sum::<u64>() as f64 / v_proj.len().max(1) as f64;

    let h_thresh = h_mean * 0.15;
    let v_thresh = v_mean * 0.15;

    let mut h_transitions = 0;
    let mut v_transitions = 0;

    for i in 1..h_proj.len() {
        if (h_proj[i-1] as f64 > h_thresh) != (h_proj[i] as f64 > h_thresh) {
            h_transitions += 1;
        }
    }

    for i in 1..v_proj.len() {
        if (v_proj[i-1] as f64 > v_thresh) != (v_proj[i] as f64 > v_thresh) {
            v_transitions += 1;
        }
    }

    let aspect = h as f64 / w as f64;

    // Tall aspect ratio with more horizontal transitions = likely vertical text
    if aspect > 1.3 && h_transitions > v_transitions {
        return true;
    }

    // Very tall and narrow is likely vertical
    if aspect > 2.0 {
        return true;
    }

    // Moderate aspect with significantly more horizontal transitions
    if aspect > 1.0 && h_transitions > v_transitions * 2 {
        return true;
    }

    false
}

/// Detect vertical text columns using projection profile
/// Returns columns from RIGHT to LEFT (Japanese reading order)
fn detect_vertical_columns(image: &DynamicImage) -> Vec<TextLine> {
    let gray = image.to_luma8();
    let (w, h) = gray.dimensions();

    if w < 10 || h < 10 {
        return vec![TextLine { x: 0, y: 0, width: w, height: h }];
    }

    // Binarize with threshold
    let mut binary = vec![vec![0u8; w as usize]; h as usize];
    let mut total: u64 = 0;
    for y in 0..h {
        for x in 0..w {
            total += gray.get_pixel(x, y).0[0] as u64;
        }
    }
    let mean = (total / (w as u64 * h as u64)) as u8;
    let threshold = mean.saturating_sub(20); // Slightly below mean for dark text

    for y in 0..h {
        for x in 0..w {
            let pixel = gray.get_pixel(x, y).0[0];
            binary[y as usize][x as usize] = if pixel < threshold { 255 } else { 0 };
        }
    }

    // Compute vertical projection (sum per column)
    let mut v_proj = vec![0u64; w as usize];
    for x in 0..w as usize {
        for y in 0..h as usize {
            v_proj[x] += binary[y][x] as u64;
        }
    }

    // Find column boundaries
    let v_mean: f64 = v_proj.iter().sum::<u64>() as f64 / v_proj.len().max(1) as f64;
    let v_thresh = v_mean * 0.2; // Higher threshold to avoid noise

    let mut columns = Vec::new();
    let mut in_col = false;
    let mut col_start = 0;

    // Minimum column width based on image size (at least 15px or 5% of width)
    let min_col_width = (w as usize / 20).max(15);

    for x in 0..w as usize {
        if v_proj[x] as f64 > v_thresh {
            if !in_col {
                col_start = x;
                in_col = true;
            }
        } else if in_col {
            let col_width = x - col_start;
            if col_width >= min_col_width {
                // Find vertical extent for this column
                let (y1, y2) = find_vertical_extent(&binary, col_start, x, h as usize);
                columns.push(TextLine {
                    x: col_start.saturating_sub(2) as u32,
                    y: y1 as u32,
                    width: (col_width + 4).min(w as usize - col_start) as u32,
                    height: (y2 - y1) as u32,
                });
            }
            in_col = false;
        }
    }

    // Handle last column
    if in_col {
        let col_width = w as usize - col_start;
        if col_width >= min_col_width {
            let (y1, y2) = find_vertical_extent(&binary, col_start, w as usize, h as usize);
            columns.push(TextLine {
                x: col_start.saturating_sub(2) as u32,
                y: y1 as u32,
                width: (col_width + 4).min(w as usize - col_start) as u32,
                height: (y2 - y1) as u32,
            });
        }
    }

    // Reverse for right-to-left reading order (Japanese)
    columns.reverse();

    if columns.is_empty() {
        vec![TextLine { x: 0, y: 0, width: w, height: h }]
    } else {
        columns
    }
}

/// Find vertical extent of text in a column range
fn find_vertical_extent(binary: &[Vec<u8>], x1: usize, x2: usize, h: usize) -> (usize, usize) {
    let mut h_proj = vec![0u64; h];
    for y in 0..h {
        for x in x1..x2 {
            h_proj[y] += binary[y][x] as u64;
        }
    }

    let h_mean: f64 = h_proj.iter().sum::<u64>() as f64 / h_proj.len().max(1) as f64;
    let h_thresh = h_mean * 0.05;

    let mut y1 = 0;
    let mut y2 = h;

    for y in 0..h {
        if h_proj[y] as f64 > h_thresh {
            y1 = y.saturating_sub(2);
            break;
        }
    }

    for y in (0..h).rev() {
        if h_proj[y] as f64 > h_thresh {
            y2 = (y + 3).min(h);
            break;
        }
    }

    (y1, y2)
}

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
        let (session, vocab, blank_index) = if OCR_MODEL_BYTES.len() > 10_000 && OCR_VOCAB_BYTES.len() > 1_000 {
            info!("Loading OCR model from embedded bytes ({:.1} MB)", OCR_MODEL_BYTES.len() as f64 / 1_048_576.0);

            let session = Session::builder()?
                .with_intra_threads(1)?
                .with_inter_threads(1)?
                .commit_from_memory(OCR_MODEL_BYTES)
                .context("Failed to load embedded OCR model")?;

            let (vocab, blank_idx) = Self::load_vocabulary_from_bytes(OCR_VOCAB_BYTES)?;

            (session, vocab, blank_idx)
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
                .with_intra_threads(1)?
                .with_inter_threads(1)?
                .commit_from_file(&model_path)
                .context("Failed to load OCR ONNX model from disk")?;

            let (vocab, blank_idx) = Self::load_vocabulary(&vocab_path)?;

            (session, vocab, blank_idx)
        };

        // Verify vocab has CJK characters
        let has_hiragana = vocab.get(&272).map(|s| s.as_str()) == Some("あ");
        let has_kanji = vocab.contains_key(&500);

        info!(
            "OCR service initialized: vocab_size={}, blank_index={}, has_hiragana={}, has_kanji={}",
            vocab.len(),
            blank_index,
            has_hiragana,
            has_kanji
        );

        Ok(Self {
            session: Mutex::new(session),
            vocab,
            blank_index,
        })
    }

    /// Load vocabulary from file (format: index\tchar)
    /// Returns (vocab_map, blank_index)
    fn load_vocabulary(vocab_path: &Path) -> Result<(HashMap<usize, String>, usize)> {
        let content = std::fs::read_to_string(vocab_path)
            .context("Failed to read vocabulary file")?;

        let mut vocab = HashMap::new();
        let mut blank_index = 0usize;

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
                    if char_str == "<blank>" {
                        blank_index = idx; // Store blank index
                        continue; // Don't add to vocab map
                    }
                    let ch = if char_str == "<space>" {
                        " ".to_string()
                    } else {
                        char_str.to_string()
                    };
                    vocab.insert(idx, ch);
                }
            }
        }

        debug!("Loaded {} vocabulary entries from file, blank_index={}", vocab.len(), blank_index);
        Ok((vocab, blank_index))
    }

    /// Load vocabulary from bytes (for embedded vocab)
    /// Returns (vocab_map, blank_index)
    fn load_vocabulary_from_bytes(vocab_bytes: &[u8]) -> Result<(HashMap<usize, String>, usize)> {
        let content = std::str::from_utf8(vocab_bytes)
            .context("Failed to parse vocabulary bytes as UTF-8")?;

        let mut vocab = HashMap::new();
        let mut blank_index = 0usize;

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
                    if char_str == "<blank>" {
                        blank_index = idx; // Store blank index
                        continue; // Don't add to vocab map
                    }
                    let ch = if char_str == "<space>" {
                        " ".to_string()
                    } else {
                        char_str.to_string()
                    };
                    vocab.insert(idx, ch);
                }
            }
        }

        debug!("Loaded {} vocabulary entries from embedded bytes, blank_index={}", vocab.len(), blank_index);
        Ok((vocab, blank_index))
    }

    /// Preprocess image region for OCR
    /// - Detect if vertical text and rotate 90° CCW if needed
    /// - Convert to RGB
    /// - Resize to target height (60px) maintaining aspect ratio
    /// - Normalize to [0, 1] float32
    /// - Return tensor in [1, 3, H, W] format
    fn preprocess_image(&self, image: &DynamicImage) -> Result<(Array4<f32>, i32, bool)> {
        // Detect vertical text and rotate if needed
        let is_vertical = is_vertical_text(image);

        let processed = if is_vertical {
            // Rotate 90° counter-clockwise to convert vertical text to horizontal
            // Note: rotate270() = 90° CCW, rotate90() = 90° CW
            image.rotate270()
        } else {
            image.clone()
        };

        let (w, h) = processed.dimensions();

        // Resize to target height maintaining aspect ratio
        let scale = TARGET_HEIGHT as f32 / h as f32;
        let new_w = ((w as f32 * scale) as u32).max(MIN_WIDTH);

        let resized = processed.resize_exact(
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

        Ok((tensor, seq_length, is_vertical))
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

        // Convert indices to characters with debug logging for missing mappings
        let text: String = decoded
            .iter()
            .map(|&idx| {
                match self.vocab.get(&idx) {
                    Some(s) => s.as_str().to_string(),
                    None => {
                        debug!("CTC decode: index {} not in vocab (vocab size: {})", idx, self.vocab.len());
                        "?".to_string()
                    }
                }
            })
            .collect();

        let avg_confidence = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        };

        // Log first few decoded indices for debugging
        if !decoded.is_empty() {
            let first_5: Vec<_> = decoded.iter().take(5).collect();
            debug!("CTC decode: first indices {:?}, blank_idx={}", first_5, self.blank_index);
        }

        (text, avg_confidence)
    }

    /// Preprocess a single column image for OCR (already extracted, just resize and normalize)
    fn preprocess_column(&self, column_image: &DynamicImage) -> Result<(Array4<f32>, i32)> {
        // Already rotated if needed, just resize to target height
        let (w, h) = column_image.dimensions();

        // Resize to target height maintaining aspect ratio
        let scale = TARGET_HEIGHT as f32 / h.max(1) as f32;
        let new_w = ((w as f32 * scale) as u32).max(MIN_WIDTH);

        let resized = column_image.resize_exact(
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

    /// Run inference on a single preprocessed tensor
    fn run_inference(&self, tensor: Array4<f32>, seq_length: i32) -> Result<(String, f32)> {
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

            // Extract logits - try known output names, then first available
            let (shape, logits_data) = if let Some(output) = outputs.get("logsoftmax") {
                output.try_extract_tensor::<f32>()?
            } else if let Some(output) = outputs.get("output") {
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
        Ok((text, confidence))
    }

    /// Check if text result looks like valid CJK output (not garbage)
    fn is_quality_result(text: &str, confidence: f32) -> bool {
        if text.is_empty() {
            return false;
        }

        // Count CJK vs ASCII characters
        let mut cjk_count = 0;
        let mut ascii_count = 0;
        let mut total_chars = 0;

        for c in text.chars() {
            total_chars += 1;
            if c.is_ascii_alphanumeric() || c.is_ascii_punctuation() {
                ascii_count += 1;
            } else if c as u32 > 0x3000 {
                // CJK ranges start around 0x3000
                cjk_count += 1;
            }
        }

        // If has ANY CJK characters, much more lenient threshold
        if cjk_count > 0 {
            // CJK text with confidence >= 0.2 is acceptable
            return confidence >= 0.2;
        }

        // Pure ASCII results require higher confidence
        // Single ASCII char is almost always garbage
        if total_chars <= 2 && ascii_count == total_chars {
            return false;
        }

        // Multi-char ASCII needs decent confidence
        if ascii_count == total_chars && confidence < 0.5 {
            return false;
        }

        // Mixed or short text - be lenient
        confidence >= 0.2
    }

    /// Run OCR on a single image region
    /// Returns (recognized_text, confidence)
    /// For manga: ALWAYS tries vertical column detection first since manga text is typically vertical.
    /// Falls back to full-image processing if column detection yields nothing.
    pub fn recognize(&self, image: &DynamicImage) -> Result<(String, f32)> {
        // STRATEGY: For manga, always try column-based vertical processing first.
        // Japanese manga text is almost always vertical, even in near-square bubbles.

        // Step 1: Try column detection (regardless of is_vertical_text heuristic)
        let columns = detect_vertical_columns(image);

        if columns.len() > 1 {
            // Multiple columns detected - process each separately
            debug!("OCR: Detected {} columns, processing individually", columns.len());

            let mut results = Vec::new();
            let mut total_confidence = 0.0f32;

            for (i, col) in columns.iter().enumerate() {
                // Extract column region
                let col_img = image.crop_imm(
                    col.x.min(image.width() - 1),
                    col.y.min(image.height() - 1),
                    col.width.min(image.width() - col.x),
                    col.height.min(image.height() - col.y),
                );

                // Rotate column 90° CCW for horizontal recognition
                let rotated = col_img.rotate270();

                // Preprocess and run inference
                let (tensor, seq_length) = self.preprocess_column(&rotated)?;
                let (text, confidence) = self.run_inference(tensor, seq_length)?;

                // Only include quality results
                if Self::is_quality_result(&text, confidence) {
                    debug!("OCR column {}: '{}' (conf: {:.2})", i + 1, text, confidence);
                    results.push(text);
                    total_confidence += confidence;
                } else if !text.is_empty() {
                    debug!("OCR column {}: FILTERED '{}' (conf: {:.2})", i + 1, text, confidence);
                }
            }

            if !results.is_empty() {
                let combined_text = results.join("\n");
                let avg_confidence = total_confidence / results.len() as f32;
                debug!("OCR combined result: '{}' (avg confidence: {:.2})", combined_text, avg_confidence);
                return Ok((combined_text, avg_confidence));
            }
            // All columns filtered - fall through to single-image approaches
            debug!("OCR: All {} columns filtered, trying full-image approaches", columns.len());
        }

        // Step 2: Single column or no columns - try rotated (vertical) first for manga
        // Rotate 90° CCW to convert potential vertical text to horizontal
        let rotated = image.rotate270();
        let (tensor_rot, seq_len_rot) = self.preprocess_column(&rotated)?;
        let (text_rot, conf_rot) = self.run_inference(tensor_rot, seq_len_rot)?;

        // Also try original orientation (horizontal text)
        let (tensor_orig, seq_len_orig) = self.preprocess_column(image)?;
        let (text_orig, conf_orig) = self.run_inference(tensor_orig, seq_len_orig)?;

        // Evaluate quality of both approaches
        let rot_quality = Self::is_quality_result(&text_rot, conf_rot);
        let orig_quality = Self::is_quality_result(&text_orig, conf_orig);

        // Count CJK characters to prioritize CJK results
        let rot_cjk = text_rot.chars().filter(|&c| c as u32 > 0x3000).count();
        let orig_cjk = text_orig.chars().filter(|&c| c as u32 > 0x3000).count();

        // Decision logic: prefer result with more CJK characters, then higher confidence
        if rot_quality && orig_quality {
            // Both valid - prefer more CJK content, then higher confidence
            if rot_cjk > orig_cjk || (rot_cjk == orig_cjk && conf_rot >= conf_orig) {
                debug!("OCR: Using rotated result '{}' (conf: {:.2}, cjk: {})", text_rot, conf_rot, rot_cjk);
                return Ok((text_rot, conf_rot));
            } else {
                debug!("OCR: Using original result '{}' (conf: {:.2}, cjk: {})", text_orig, conf_orig, orig_cjk);
                return Ok((text_orig, conf_orig));
            }
        } else if rot_quality {
            debug!("OCR: Using rotated result '{}' (conf: {:.2})", text_rot, conf_rot);
            return Ok((text_rot, conf_rot));
        } else if orig_quality {
            debug!("OCR: Using original result '{}' (conf: {:.2})", text_orig, conf_orig);
            return Ok((text_orig, conf_orig));
        }

        // Neither passed quality check - return the better one anyway if it has CJK content
        // This is a fallback for difficult cases
        if rot_cjk > 0 || orig_cjk > 0 {
            if rot_cjk >= orig_cjk && !text_rot.is_empty() {
                debug!("OCR: Fallback to rotated '{}' (conf: {:.2}, cjk: {})", text_rot, conf_rot, rot_cjk);
                return Ok((text_rot, conf_rot));
            } else if !text_orig.is_empty() {
                debug!("OCR: Fallback to original '{}' (conf: {:.2}, cjk: {})", text_orig, conf_orig, orig_cjk);
                return Ok((text_orig, conf_orig));
            }
        }

        // Both are garbage - return empty
        debug!("OCR: Both orientations produced garbage, returning empty");
        Ok((String::new(), 0.0))
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
