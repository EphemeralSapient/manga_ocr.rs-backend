use crate::config::Config;
use crate::cosmic_renderer::{CosmicTextRenderer, VerticalAlign};
use crate::services::traits::BubbleRenderer;
use crate::types::{BubbleDetection, TextTranslation};
use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use anyhow::{Context, Result};
use async_trait::async_trait;
use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;
use imageproc::edges::canny;
use imageproc::contrast::{threshold, ThresholdType};
use opencv::{core, imgproc, prelude::*};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use unicode_segmentation::UnicodeSegmentation;
use font_kit::family_name::FamilyName;
use font_kit::properties::{Properties, Weight, Style};
use font_kit::source::SystemSource;
use tracing::{debug, info, warn};

/// Font cache for efficient font loading
/// Maps font family name to Arc'd font bytes (avoids cloning on cache hits)
#[derive(Clone)]
struct FontCache {
    fonts: Arc<std::sync::RwLock<HashMap<String, Arc<Vec<u8>>>>>,
}

#[allow(dead_code)]
impl FontCache {
    fn new() -> Self {
        Self {
            fonts: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Get font from cache or load it
    ///
    /// Returns Arc<Vec<u8>> to avoid cloning font data (fonts are typically 200-500KB).
    /// Cache hits now just clone the Arc pointer instead of the entire font data.
    /// Uses double-check locking to avoid race conditions.
    fn get_or_load(&self, family: &str, loader: impl FnOnce() -> Option<Vec<u8>>) -> Option<Arc<Vec<u8>>> {
        // First check: read lock for concurrent access (fast path)
        {
            let cache = self.fonts.read()
                .expect("Font cache RwLock poisoned");
            if let Some(font_bytes) = cache.get(family) {
                tracing::debug!("Font cache HIT for '{}' (Arc clone, no data copy)", family);
                return Some(Arc::clone(font_bytes));
            }
        }

        // Not in cache, need to load - acquire write lock
        let mut cache = self.fonts.write()
            .expect("Font cache RwLock poisoned");

        // Second check: another thread may have loaded it while we waited for write lock
        if let Some(font_bytes) = cache.get(family) {
            tracing::debug!("Font cache HIT for '{}' (loaded by another thread)", family);
            return Some(Arc::clone(font_bytes));
        }

        // Still not in cache, load it now
        tracing::debug!("Font cache MISS for '{}', loading...", family);
        if let Some(font_bytes) = loader() {
            let font_arc = Arc::new(font_bytes);
            let byte_count = font_arc.len();
            cache.insert(family.to_string(), Arc::clone(&font_arc));
            tracing::info!("Cached font '{}' ({} bytes)", family, byte_count);
            Some(font_arc)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn stats(&self) -> (usize, usize) {
        let cache = self.fonts.read()
            .expect("Font cache RwLock poisoned");
        let count = cache.len();
        let total_bytes: usize = cache.values().map(|v| v.len()).sum();
        (count, total_bytes)
    }
}

pub struct RenderingService {
    config: Arc<Config>,
    font_cache: Arc<FontCache>,
    cosmic_renderer: Arc<CosmicTextRenderer>,
    // Note: SystemSource is not thread-safe on Linux (contains raw pointers)
    // We create it on-demand per request instead of storing it
}

impl RenderingService {
    pub fn new(config: Arc<Config>) -> Self {
        let cache = Arc::new(FontCache::new());
        let cosmic_renderer = Arc::new(CosmicTextRenderer::new());

        tracing::info!("RenderingService initialized with CosmicTextRenderer and font cache");

        Self {
            config,
            font_cache: cache,
            cosmic_renderer,
        }
    }

    /// Get font cache statistics
    #[allow(dead_code)]
    pub fn font_cache_stats(&self) -> (usize, usize) {
        self.font_cache.stats()
    }

    /// Parse font family string into font-kit family, weight, and style
    #[allow(dead_code)]
    fn parse_font_family(font_family: &str) -> (FamilyName, Weight, Style) {
        match font_family {
            "sans-serif-regular" => (FamilyName::SansSerif, Weight::NORMAL, Style::Normal),
            "sans-serif-bold" => (FamilyName::SansSerif, Weight::BOLD, Style::Normal),
            "serif-regular" => (FamilyName::Serif, Weight::NORMAL, Style::Normal),
            "serif-bold" => (FamilyName::Serif, Weight::BOLD, Style::Normal),
            "monospace-regular" => (FamilyName::Monospace, Weight::NORMAL, Style::Normal),
            "monospace-bold" => (FamilyName::Monospace, Weight::BOLD, Style::Normal),
            _ => {
                tracing::warn!("Unknown font family '{}', defaulting to sans-serif-bold", font_family);
                (FamilyName::SansSerif, Weight::BOLD, Style::Normal)
            }
        }
    }

    /// Load font using cross-platform font-kit with fallback strategy
    #[allow(dead_code)]
    fn load_font_cross_platform(&self, font_family: &str) -> Option<Vec<u8>> {
        // Map font family to actual font file in fonts/ directory
        let font_file = match font_family {
            "anime-ace" => "fonts/anime_ace.ttf",
            "anime-ace-3" => "fonts/anime_ace_3.ttf",
            "arial" => "fonts/Arial-Unicode-Regular.ttf",
            "comic-sans" => "fonts/comic shanns 2.ttf",
            "ms-yahei" => "fonts/msyh.ttc",
            "noto-sans-mono-cjk" => "fonts/NotoSansMonoCJK-VF.ttf.ttc",
            _ => {
                warn!("Unknown font family '{}', using default 'arial'", font_family);
                "fonts/Arial-Unicode-Regular.ttf"
            }
        };

        // Try to load from fonts/ directory
        match fs::read(font_file) {
            Ok(data) => {
                info!("Loaded font from fonts/ directory: {} ({} bytes)", font_file, data.len());
                return Some(data);
            }
            Err(e) => {
                warn!("Failed to load font from '{}': {}", font_file, e);
            }
        }

        // Fallback to Arial if the requested font fails
        if font_family != "arial" {
            warn!("Falling back to default font 'arial'");
            if let Ok(data) = fs::read("fonts/Arial-Unicode-Regular.ttf") {
                info!("Loaded fallback font: Arial-Unicode-Regular.ttf ({} bytes)", data.len());
                return Some(data);
            }
        }

        // Last resort: Try system fonts (for backward compatibility)
        let (family, weight, style) = Self::parse_font_family(font_family);
        let mut properties = Properties::new();
        properties.weight = weight;
        properties.style = style;

        let system_source = SystemSource::new();
        if let Ok(handle) = system_source.select_best_match(&[family], &properties) {
            if let Ok(font) = handle.load() {
                if let Some(data) = font.copy_font_data() {
                    info!("Loaded font from system: {} ({} bytes)", font_family, data.len());
                    return Some(data.to_vec());
                }
            }
        }

        None
    }

    /// Get font with caching (synchronous version for non-async contexts)
    ///
    /// Returns Arc<Vec<u8>> to avoid cloning font data on cache hits.
    /// The Arc derefs to &[u8] automatically for FontRef::try_from_slice().
    #[allow(dead_code)]
    fn find_font_file(&self, font_family: &str) -> Option<Arc<Vec<u8>>> {
        self.font_cache.get_or_load(font_family, || {
            self.load_font_cross_platform(font_family)
        })
    }

    /// Get font with caching (async version - offloads I/O to blocking thread pool)
    ///
    /// Returns Arc<Vec<u8>> to avoid cloning font data on cache hits.
    #[allow(dead_code)]
    async fn find_font_file_async(&self, font_family: &str) -> Option<Arc<Vec<u8>>> {
        // Check cache first (fast path - no blocking)
        {
            let cache = self.font_cache.fonts.read()
                .expect("Font cache RwLock poisoned");
            if let Some(font_bytes) = cache.get(font_family) {
                tracing::debug!("Font cache HIT for '{}' (async)", font_family);
                return Some(Arc::clone(font_bytes));
            }
        }

        // Not in cache - offload font loading to blocking thread pool
        let family = font_family.to_string();
        let font_cache = self.font_cache.clone();

        tokio::task::spawn_blocking(move || {
            font_cache.get_or_load(&family, || {
                // This closure runs on a blocking thread, so fs::read is fine
                Self::load_font_cross_platform_static(&family)
            })
        })
        .await
        .ok()
        .flatten()
    }

    /// Static version of load_font_cross_platform for use in spawn_blocking
    #[allow(dead_code)]
    fn load_font_cross_platform_static(font_family: &str) -> Option<Vec<u8>> {
        // Same logic as load_font_cross_platform but static
        let font_file = match font_family {
            "anime-ace" => "fonts/anime_ace.ttf",
            "anime-ace-3" => "fonts/anime_ace_3.ttf",
            "arial" => "fonts/Arial-Unicode-Regular.ttf",
            "comic-sans" => "fonts/comic shanns 2.ttf",
            "ms-yahei" => "fonts/msyh.ttc",
            "noto-sans-mono-cjk" => "fonts/NotoSansMonoCJK-VF.ttf.ttc",
            _ => {
                warn!("Unknown font family '{}', using default 'arial'", font_family);
                "fonts/Arial-Unicode-Regular.ttf"
            }
        };

        // Try to load from fonts/ directory
        if let Ok(data) = fs::read(font_file) {
            info!("Loaded font from fonts/ directory: {} ({} bytes)", font_file, data.len());
            return Some(data);
        }

        // Fallback to Arial
        if font_family != "arial" {
            warn!("Falling back to default font 'arial'");
            if let Ok(data) = fs::read("fonts/Arial-Unicode-Regular.ttf") {
                info!("Loaded fallback font: Arial-Unicode-Regular.ttf ({} bytes)", data.len());
                return Some(data);
            }
        }

        // Last resort: Try system fonts
        let (family, weight, style) = Self::parse_font_family(font_family);
        let mut properties = Properties::new();
        properties.weight = weight;
        properties.style = style;

        let system_source = SystemSource::new();
        if let Ok(handle) = system_source.select_best_match(&[family], &properties) {
            if let Ok(font) = handle.load() {
                if let Some(data) = font.copy_font_data() {
                    info!("Loaded font from system: {} ({} bytes)", font_family, data.len());
                    return Some(data.to_vec());
                }
            }
        }

        None
    }

    #[allow(dead_code)]
    fn parse_rgb_color(color_str: &str) -> Rgba<u8> {
        let parts: Vec<&str> = color_str.split(',').collect();
        if parts.len() == 3 {
            if let (Ok(r), Ok(g), Ok(b)) = (
                parts[0].trim().parse::<u8>(),
                parts[1].trim().parse::<u8>(),
                parts[2].trim().parse::<u8>(),
            ) {
                return Rgba([r, g, b, 255]);
            }
        }
        Rgba([255, 255, 255, 255])
    }

    /// Determine optimal stroke color based on fill color
    /// Dark text gets white stroke, light text gets black stroke
    #[allow(dead_code)]
    fn get_stroke_color(fill_color: Rgba<u8>) -> Rgba<u8> {
        let brightness = fill_color[0] as u32 + fill_color[1] as u32 + fill_color[2] as u32;

        // If text is dark (brightness < 382, which is middle of 0-765 range)
        if brightness < 382 {
            Rgba([255, 255, 255, 255])  // White stroke for dark text
        } else {
            Rgba([0, 0, 0, 255])  // Black stroke for light text
        }
    }

    /// Draw text with stroke/outline for manga-style rendering
    /// This ensures text is readable on any background
    #[allow(dead_code)]
    fn draw_text_with_stroke(
        img: &mut RgbaImage,
        fill_color: Rgba<u8>,
        stroke_color: Rgba<u8>,
        stroke_width: i32,
        x: i32,
        y: i32,
        scale: PxScale,
        font: &FontRef,
        text: &str,
    ) {
        // Draw stroke by rendering text at multiple offsets
        // Use 8-point offset pattern for smooth stroke
        for offset_y in -stroke_width..=stroke_width {
            for offset_x in -stroke_width..=stroke_width {
                // Skip center (that's where fill goes)
                if offset_x == 0 && offset_y == 0 {
                    continue;
                }

                // Only render if within stroke radius (circular stroke)
                let distance_sq = (offset_x * offset_x + offset_y * offset_y) as f32;
                let radius_sq = (stroke_width * stroke_width) as f32;

                if distance_sq <= radius_sq * 1.2 {  // 1.2 for slightly smoother edges
                    draw_text_mut(
                        img,
                        stroke_color,
                        x + offset_x,
                        y + offset_y,
                        scale,
                        font,
                        text,
                    );
                }
            }
        }

        // Draw fill on top
        draw_text_mut(img, fill_color, x, y, scale, font, text);
    }

    /// Analyze background complexity to determine if AI redrawing is actually needed
    /// Simple rule: if ~70% of pixels are white/near-white, it's a simple background
    /// Returns: (is_complex, detected_color_if_simple)
    pub fn analyze_background_complexity(img: &RgbaImage) -> (bool, Option<Rgba<u8>>) {
        let width = img.width();
        let height = img.height();

        if width == 0 || height == 0 {
            return (true, None); // Can't analyze, assume complex
        }

        // Count white/near-white pixels
        let mut white_pixel_count = 0;
        let mut total_pixels = 0;
        let mut r_sum = 0u64;
        let mut g_sum = 0u64;
        let mut b_sum = 0u64;

        // Sample every 2nd pixel for performance
        for y in (0..height).step_by(2) {
            for x in (0..width).step_by(2) {
                let pixel = img.get_pixel(x, y);
                total_pixels += 1;

                r_sum += pixel[0] as u64;
                g_sum += pixel[1] as u64;
                b_sum += pixel[2] as u64;

                // Consider pixel "white" if all RGB values are >= 200
                if pixel[0] >= 200 && pixel[1] >= 200 && pixel[2] >= 200 {
                    white_pixel_count += 1;
                }
            }
        }

        if total_pixels == 0 {
            return (true, None);
        }

        let white_percentage = (white_pixel_count as f32 / total_pixels as f32) * 100.0;

        // Calculate average color
        let avg_color = Rgba([
            (r_sum / total_pixels) as u8,
            (g_sum / total_pixels) as u8,
            (b_sum / total_pixels) as u8,
            255,
        ]);

        // Simple rule: 70% white = simple background
        let is_simple = white_percentage >= 70.0;

        if is_simple {
            tracing::info!(
                "✓ Simple background detected: {:.1}% white pixels, avg RGB({},{},{})",
                white_percentage,
                avg_color[0], avg_color[1], avg_color[2]
            );
            (false, Some(avg_color))
        } else {
            tracing::info!(
                "✗ Complex background detected: only {:.1}% white pixels, AI redraw required",
                white_percentage
            );
            (true, None)
        }
    }

    /// Detect the background color from bubble image edges
    /// Samples only the outer border and prefers the brightest/lightest colors
    /// This avoids manga artwork and finds the actual bubble background
    #[allow(dead_code)]
    fn detect_majority_color(img: &RgbaImage) -> Rgba<u8> {
        let width = img.width();
        let height = img.height();

        // Sample from outer 10% border
        let border_thickness_x = (width / 10).max(1);
        let border_thickness_y = (height / 10).max(1);

        let mut color_counts: HashMap<(u8, u8, u8), u32> = HashMap::new();

        // Sample edge pixels
        for y in 0..height {
            for x in 0..width {
                // Only process border pixels
                let is_border = x < border_thickness_x
                    || x >= width - border_thickness_x
                    || y < border_thickness_y
                    || y >= height - border_thickness_y;

                if is_border {
                    let pixel = img.get_pixel(x, y);
                    let brightness = pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32;

                    // Only consider very bright pixels (brightness > 720 out of 765 max)
                    // This ensures we get white/near-white backgrounds typical of manga bubbles
                    if brightness > 720 {
                        let r = (pixel[0] / 4) * 4;  // Quantize to nearest 4 for finer granularity
                        let g = (pixel[1] / 4) * 4;
                        let b = (pixel[2] / 4) * 4;
                        *color_counts.entry((r, g, b)).or_insert(0) += 1;
                    }
                }
            }
        }

        // Find the most common bright color from edges
        if let Some(((r, g, b), _)) = color_counts.iter().max_by_key(|(_, &count)| count) {
            // If very close to white (all components >= 240), round up to pure white
            let final_r = if *r >= 240 && *g >= 240 && *b >= 240 { 255 } else { *r };
            let final_g = if *r >= 240 && *g >= 240 && *b >= 240 { 255 } else { *g };
            let final_b = if *r >= 240 && *g >= 240 && *b >= 240 { 255 } else { *b };
            Rgba([final_r, final_g, final_b, 255])
        } else {
            // Fallback: find brightest pixel overall if no bright edge pixels
            let mut brightest_color = Rgba([255, 255, 255, 255]);
            let mut max_brightness = 0u32;

            for pixel in img.pixels() {
                let brightness = pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32;
                if brightness > max_brightness {
                    max_brightness = brightness;
                    brightest_color = *pixel;
                }
            }
            brightest_color
        }
    }

    #[allow(dead_code)]
    fn measure_text_line(font: &FontRef, text: &str, scale: PxScale) -> (f32, f32) {
        if text.is_empty() {
            return (0.0, scale.y);
        }

        let scaled_font = font.as_scaled(scale);
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut caret = point(0.0, 0.0);
        let mut last_glyph = None;

        for ch in text.chars() {
            let glyph_id = font.glyph_id(ch);

            if let Some(previous) = last_glyph {
                caret.x += scaled_font.kern(previous, glyph_id);
            }

            let glyph = glyph_id.with_scale_and_position(scale, caret);

            if let Some(outlined) = font.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                min_y = min_y.min(bounds.min.y);
                max_y = max_y.max(bounds.max.y);
            }

            caret.x += scaled_font.h_advance(glyph_id);
            last_glyph = Some(glyph_id);
        }

        let width = caret.x;
        let height = if max_y > min_y { max_y - min_y } else { scale.y };

        (width, height)
    }

    /// Detect if text contains CJK characters (Chinese, Japanese, Korean)
    /// CJK text doesn't use spaces for word boundaries
    #[allow(dead_code)]
    fn contains_cjk(text: &str) -> bool {
        text.chars().any(|c| {
            matches!(c,
                '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
                '\u{3040}'..='\u{309F}' |  // Hiragana
                '\u{30A0}'..='\u{30FF}' |  // Katakana
                '\u{AC00}'..='\u{D7AF}'    // Hangul
            )
        })
    }

    /// 3-tier text wrapping: word-level → grapheme-level → character-level
    /// Handles English text, CJK languages, and emergency overflow cases
    #[allow(dead_code)]
    fn wrap_text_to_width(
        &self,
        text: &str,
        font: &FontRef,
        scale: PxScale,
        max_width: f32,
    ) -> Vec<String> {
        let mut wrapped_lines = Vec::new();
        let is_cjk = Self::contains_cjk(text);

        for line in text.split('\n') {
            // For English: ALWAYS use word-level wrapping, NEVER break words mid-word
            // Comic text should keep words intact even if they overflow slightly
            if !is_cjk && line.contains(' ') {
                let word_wrapped = self.wrap_by_words(line, font, scale, max_width);
                wrapped_lines.extend(word_wrapped);
                continue;
            }

            // For CJK or single-word English lines: use grapheme-level wrapping
            let grapheme_wrapped = self.wrap_by_graphemes(line, font, scale, max_width);
            wrapped_lines.extend(grapheme_wrapped);
        }

        wrapped_lines
    }

    /// Word-level wrapping (tier 1) - best for English text
    #[allow(dead_code)]
    fn wrap_by_words(
        &self,
        line: &str,
        font: &FontRef,
        scale: PxScale,
        max_width: f32,
    ) -> Vec<String> {
        let mut wrapped_lines = Vec::new();
        let mut current_line = String::new();

        for word in line.split_whitespace() {
            let test_line = if current_line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", current_line, word)
            };

            let (width, _) = Self::measure_text_line(font, &test_line, scale);

            if width <= max_width {
                current_line = test_line;
            } else if current_line.is_empty() {
                // Word itself is too long, will be handled by grapheme wrapping
                current_line = word.to_string();
            } else {
                // Current line is full, start new line
                wrapped_lines.push(current_line);
                current_line = word.to_string();
            }
        }

        if !current_line.is_empty() {
            wrapped_lines.push(current_line);
        }

        wrapped_lines
    }

    /// Grapheme-level wrapping (tier 2) - handles CJK and long words
    /// Uses unicode grapheme clusters for proper character boundary detection
    #[allow(dead_code)]
    fn wrap_by_graphemes(
        &self,
        line: &str,
        font: &FontRef,
        scale: PxScale,
        max_width: f32,
    ) -> Vec<String> {
        let mut wrapped_lines = Vec::new();
        let mut current_line = String::new();

        // Use grapheme clusters for proper Unicode handling
        for grapheme in line.graphemes(true) {
            let test_line = format!("{}{}", current_line, grapheme);
            let (width, _) = Self::measure_text_line(font, &test_line, scale);

            if width <= max_width {
                current_line = test_line;
            } else {
                // Current line is full
                if !current_line.is_empty() {
                    wrapped_lines.push(current_line);
                    current_line = grapheme.to_string();
                } else {
                    // Even a single grapheme is too wide (emergency case)
                    // Push it anyway to avoid infinite loop
                    current_line = grapheme.to_string();
                }
            }
        }

        if !current_line.is_empty() {
            wrapped_lines.push(current_line);
        }

        wrapped_lines
    }

    /// Calculate line length variance for scoring uniformity
    /// Lower variance = more uniform line lengths = better readability
    #[allow(dead_code)]
    fn calculate_line_variance(lines: &[String], font: &FontRef, scale: PxScale) -> f32 {
        if lines.len() <= 1 {
            return 0.0;
        }

        let widths: Vec<f32> = lines
            .iter()
            .map(|line| Self::measure_text_line(font, line, scale).0)
            .collect();

        let mean = widths.iter().sum::<f32>() / widths.len() as f32;
        let variance = widths.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / widths.len() as f32;

        variance.sqrt() // Return standard deviation
    }

    /// Improved scoring function that considers readability, not just space utilization
    /// Lower score is better
    #[allow(dead_code)]
    fn score_layout(
        lines: &[String],
        font: &FontRef,
        scale: PxScale,
        layout_width: f32,
        layout_height: f32,
        available_width: f32,
        available_height: f32,
    ) -> f32 {
        // Penalize line count (fewer lines = better, more readable)
        let line_count_penalty = (lines.len() as f32 - 1.0).max(0.0) * 0.15;

        // Penalize non-uniform line lengths (uniform = better)
        let variance_penalty = Self::calculate_line_variance(lines, font, scale) / available_width * 0.2;

        // Reward space utilization (but not as strongly as readability factors)
        let width_unused = (available_width - layout_width) / available_width;
        let height_unused = (available_height - layout_height) / available_height;
        let space_penalty = (width_unused + height_unused) * 0.3;

        // Penalize aspect ratio mismatch (text block should roughly match bubble shape)
        let text_aspect = if layout_height > 0.0 {
            layout_width / layout_height
        } else {
            1.0
        };
        let bubble_aspect = if available_height > 0.0 {
            available_width / available_height
        } else {
            1.0
        };
        let aspect_penalty = ((text_aspect - bubble_aspect).abs() / bubble_aspect.max(1.0)) * 0.15;

        // Combine penalties (lower is better)
        line_count_penalty + variance_penalty + space_penalty + aspect_penalty
    }

    #[allow(dead_code)]
    fn calculate_optimal_lines(
        &self,
        text: &str,
        font: &FontRef,
        scale: PxScale,
        available_width: f32,
        _available_height: f32,
        _aspect_ratio: f32,
    ) -> (Vec<String>, f32, f32) {
        // Smart word wrapping on flattened text
        // Flatten text to single line first, then wrap optimally for this font size
        // This creates natural multi-word lines instead of fragmenting pre-split lines
        let flattened = text.replace('\n', " "); // Remove Gemini's line breaks

        // Use very generous target width to allow multi-word lines
        // Most of available width should be usable for better readability
        let target_width = available_width * 0.96; // Use 96% of available width

        let wrapped = self.wrap_text_to_width(&flattened, font, scale, target_width);
        let (wrap_width, wrap_height) = self.measure_lines(&font, &wrapped, scale);

        // Return the wrapped result - this creates optimal multi-word lines
        (wrapped, wrap_width, wrap_height)
    }

    #[allow(dead_code)]
    fn measure_lines(&self, font: &FontRef, lines: &[String], scale: PxScale) -> (f32, f32) {
        // Tighter line spacing to fit more text when using full bubble area
        let line_spacing = scale.y * 0.2; // Reduced from 25% to 20% for tighter fit
        let mut max_width: f32 = 0.0;
        let mut total_height: f32 = 0.0;

        for line in lines {
            let (width, height) = Self::measure_text_line(font, line, scale);
            max_width = max_width.max(width);
            total_height += height;
        }

        if lines.len() > 1 {
            total_height += line_spacing * (lines.len() - 1) as f32;
        }

        (max_width, total_height)
    }

    /// Render BLACK text on TRANSPARENT background (for new workflow)
    fn render_text_on_transparent(
        &self,
        mut img: RgbaImage,
        translation: &TextTranslation,
        text_regions: &[[i32; 4]], // Label 1 text regions to constrain text
    ) -> Result<RgbaImage> {
        debug!("🖌️  [RENDERING TRANSPARENT COSMIC-TEXT] Starting high-quality text rendering on transparent {}x{} canvas",
            img.width(), img.height());

        // Always use black text for transparent rendering (no stroke for clean appearance)
        let text_color = Rgba([0, 0, 0, 255]);

        // Use Label 1 (text region) dimensions for text layout, not Label 0 (full bubble)
        // This ensures text stays within the detected text area
        let (text_width, text_height, text_x_offset, text_y_offset) = if !text_regions.is_empty() {
            let region = &text_regions[0]; // Use first text region
            let w = (region[2] - region[0]) as f32;
            let h = (region[3] - region[1]) as f32;
            (w, h, region[0] as f32, region[1] as f32)
        } else {
            // Fallback to full canvas if no Label 1 regions
            (img.width() as f32, img.height() as f32, 0.0, 0.0)
        };

        debug!("Text constrained to Label 1 region: {}x{} at offset ({}, {})",
            text_width as i32, text_height as i32, text_x_offset as i32, text_y_offset as i32);

        // Comic-style padding within Label 1 text region - enough margin to keep text contained
        let padding_percent = 0.06; // 6% padding for safe margins
        let padding_x = (text_width * padding_percent).max(5.0);
        let padding_y = (text_height * padding_percent).max(5.0);
        let available_width = (text_width - (2.0 * padding_x)).max(1.0);
        let available_height = (text_height - (2.0 * padding_y)).max(1.0);

        let aspect_ratio = text_width / text_height;
        let min_readable_size = 14.0; // Readable minimum
        // Conservative max size to ensure readable multi-word lines
        let max_reasonable_size = if aspect_ratio < 0.7 {
            (available_height * 0.10).min(available_width * 0.35)
        } else if aspect_ratio < 1.0 {
            (available_height * 0.12).min(available_width * 0.30)
        } else {
            (available_height * 0.15).min(available_width * 0.25)
        };

        // Find optimal font size using cosmic-text
        let best_size = self.cosmic_renderer.find_optimal_font_size(
            &translation.english_translation,
            &translation.font_family,
            available_width,
            available_height,
            min_readable_size,
            max_reasonable_size.max(min_readable_size),
        )?;

        debug!("Layout: available={:.0}x{:.0}px, font_size={:.1}px, aspect={:.2}",
            available_width, available_height, best_size, aspect_ratio);

        // Render text with cosmic-text (no stroke for transparent mode)
        self.cosmic_renderer.render_multiline_text(
            &mut img,
            &translation.english_translation,
            &translation.font_family,
            best_size,
            text_color,
            (text_x_offset + padding_x) as i32,
            (text_y_offset + padding_y) as i32,
            available_width,
            available_height,
            None, // No stroke for clean black text on transparent
            VerticalAlign::Middle,
        )?;

        debug!("✅ [RENDERING TRANSPARENT COSMIC-TEXT] High-quality text rendered in black on transparent");

        Ok(img)
    }

    fn render_text_on_image(
        &self,
        mut img: RgbaImage,
        translation: &TextTranslation,
    ) -> Result<RgbaImage> {
        debug!("🖌️  [RENDERING COSMIC-TEXT] Starting high-quality text rendering on {}x{} canvas",
            img.width(), img.height());
        let render_start = std::time::Instant::now();

        let text_color = Self::parse_rgb_color(&translation.font_color);
        debug!("Text color: RGB({},{},{})", text_color[0], text_color[1], text_color[2]);

        // Comic-style minimal padding - maximize text size for bold, prominent comic text
        let padding_percent = 0.02;
        let padding_x = (img.width() as f32 * padding_percent).max(5.0);
        let padding_y = (img.height() as f32 * padding_percent).max(5.0);
        let available_width = (img.width() as f32 - (2.0 * padding_x)).max(1.0);
        let available_height = (img.height() as f32 - (2.0 * padding_y)).max(1.0);

        let aspect_ratio = img.width() as f32 / img.height() as f32;
        let min_readable_size = 18.0; // Comic text minimum - bold and prominent
        // Aspect-aware max size: tall bubbles prioritize width
        let max_reasonable_size = if aspect_ratio < 0.7 {
            available_width * 0.9 // Very tall: use most of width
        } else if aspect_ratio < 1.0 {
            (available_width * 0.85).min(available_height * 0.35)
        } else {
            (available_height * 0.7).min(available_width * 0.6)
        };

        // Find optimal font size using cosmic-text
        let best_size = self.cosmic_renderer.find_optimal_font_size(
            &translation.english_translation,
            &translation.font_family,
            available_width,
            available_height,
            min_readable_size,
            max_reasonable_size.max(min_readable_size),
        )?;

        // Calculate stroke width based on font size (2-4px for manga style)
        let stroke_width = ((best_size / 12.0).max(2.0).min(4.0)) as i32;

        debug!("Layout: available={:.0}x{:.0}px, font_size={:.1}px, stroke={}px, aspect={:.2}",
            available_width, available_height, best_size, stroke_width, aspect_ratio);

        // Render text with cosmic-text (includes stroke and proper shaping)
        self.cosmic_renderer.render_multiline_text(
            &mut img,
            &translation.english_translation,
            &translation.font_family,
            best_size,
            text_color,
            padding_x as i32,
            padding_y as i32,
            available_width,
            available_height,
            Some(stroke_width),
            VerticalAlign::Middle,
        )?;

        let render_time = render_start.elapsed();
        debug!("✅ [RENDERING COSMIC-TEXT] High-quality text rendered in {:.2}ms", render_time.as_secs_f64() * 1000.0);

        Ok(img)
    }

    /// Decode base64 PNG segmentation mask from Gemini API
    /// Returns grayscale image where values >127 indicate text pixels
    #[allow(dead_code)]
    fn decode_segmentation_mask(mask_data: &str) -> Result<image::GrayImage> {
        use base64::{engine::general_purpose, Engine};

        // Remove data URI prefix if present
        let base64_data = if mask_data.starts_with("data:image/png;base64,") {
            &mask_data[22..]
        } else {
            mask_data
        };

        // Remove all non-base64 characters (whitespace and any other invalid chars)
        // Valid base64 data: A-Z, a-z, 0-9, +, /
        // NOTE: We intentionally exclude '=' here because padding must ONLY be at the end
        let clean_base64_no_padding: String = base64_data.chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '+' || *c == '/')
            .collect();

        // Normalize padding: base64 strings should be multiples of 4
        // Add padding at the END only
        let remainder = clean_base64_no_padding.len() % 4;
        let mut clean_base64 = clean_base64_no_padding;
        if remainder > 0 {
            let padding_needed = 4 - remainder;
            clean_base64.push_str(&"=".repeat(padding_needed));
            debug!("Added {} padding chars at end", padding_needed);
        }

        debug!("Decoding mask: original={} bytes, cleaned={} bytes (with padding)",
            base64_data.len(), clean_base64.len());

        // Try decoding with standard decoder
        let png_bytes = match general_purpose::STANDARD.decode(&clean_base64) {
            Ok(bytes) => bytes,
            Err(e1) => {
                debug!("STANDARD decoder failed: {}, trying STANDARD_NO_PAD", e1);
                // Try without padding validation as fallback
                general_purpose::STANDARD_NO_PAD.decode(&clean_base64)
                    .with_context(|| format!("Failed to decode base64 mask with both decoders (cleaned {} chars). First 100 chars: {}. STANDARD error: {}",
                        clean_base64.len(),
                        &clean_base64.chars().take(100).collect::<String>(),
                        e1))?
            }
        };

        debug!("Decoded PNG: {} bytes, header: {:?}", png_bytes.len(), &png_bytes[..8.min(png_bytes.len())]);

        // PNG files start with signature: 89 50 4E 47 0D 0A 1A 0A
        let is_valid_png_header = png_bytes.len() >= 8 &&
            png_bytes[0] == 0x89 && png_bytes[1] == 0x50 &&
            png_bytes[2] == 0x4E && png_bytes[3] == 0x47;

        if !is_valid_png_header {
            return Err(anyhow::anyhow!("Decoded data is not a valid PNG (invalid header). Got bytes: {:?}", &png_bytes[..16.min(png_bytes.len())]));
        }

        debug!("PNG header valid, loading image...");

        let mask_img = image::load_from_memory(&png_bytes)
            .with_context(|| format!("Failed to load PNG image ({} bytes) from decoded bytes", png_bytes.len()))?;

        debug!("✅ Loaded mask image: {}x{}", mask_img.width(), mask_img.height());

        Ok(mask_img.to_luma8())
    }

    /// Public wrapper for text cleaning (used by CLI --speech flag)
    pub fn clean_text_from_bubble_public(
        bubble_img: &RgbaImage,
        text_regions: &[[i32; 4]],
        bubble_bbox: &[i32; 4]
    ) -> RgbaImage {
        Self::clean_text_from_bubble(bubble_img, text_regions, bubble_bbox)
    }

    /// Remove text from speech bubble using edge-based safe interior detection
    /// NEW APPROACH (proto.md): Detects edges, erodes inward, prevents bleeding outside bubble
    ///
    /// Three-layer protection:
    /// 1. Safe interior mask (edge detection + erosion)
    /// 2. Label 1 text region constraint
    /// 3. Alpha channel visibility check
    ///
    /// Uses edge detection instead of Gemini polygons for more accurate cleaning
    fn clean_text_from_bubble(
        bubble_img: &RgbaImage,
        text_regions: &[[i32; 4]],
        bubble_bbox: &[i32; 4]
    ) -> RgbaImage {
        let mut cleaned = bubble_img.clone();

        debug!("🧹 Cleaning text using EDGE-BASED SAFE INTERIOR DETECTION (proto.md algorithm)");
        debug!("  Bubble dimensions: {}x{}", bubble_img.width(), bubble_img.height());
        debug!("  Label 1 text regions to constrain: {}", text_regions.len());

        // NEW: Create safe interior mask using edge detection + erosion (OpenCV for exact Python parity)
        let safe_interior_mask = match Self::create_safe_interior_mask(bubble_img) {
            Ok(mask) => mask,
            Err(e) => {
                warn!("⚠️ OpenCV edge detection failed: {}, falling back to simple cleaning", e);
                // Fallback: create empty mask (no cleaning)
                return bubble_img.clone();
            }
        };

        // Convert Label 1 text regions from absolute to bubble-relative coordinates
        let text_regions_relative: Vec<[i32; 4]> = text_regions.iter().map(|region| {
            let rel_x1 = (region[0] - bubble_bbox[0]).max(0);
            let rel_y1 = (region[1] - bubble_bbox[1]).max(0);
            let rel_x2 = (region[2] - bubble_bbox[0]).min(bubble_bbox[2] - bubble_bbox[0]);
            let rel_y2 = (region[3] - bubble_bbox[1]).min(bubble_bbox[3] - bubble_bbox[1]);
            [rel_x1, rel_y1, rel_x2, rel_y2]
        }).collect();

        // Three-layer protection:
        // 1. Safe interior mask (edge detection + erosion) - prevents bleeding!
        // 2. Label 1 text region (constrains to detected text areas)
        // 3. Alpha channel visibility check
        let mut erased_count = 0;
        let mut edge_protected_count = 0;
        let mut label1_protected_count = 0;

        for y in 0..cleaned.height() {
            for x in 0..cleaned.width() {
                let pixel = cleaned.get_pixel(x, y);

                // Layer 1: Check safe interior mask (prevents bleeding!)
                let in_safe_interior = safe_interior_mask.get_pixel(x, y)[0] == 255;

                if !in_safe_interior {
                    edge_protected_count += 1;
                    continue;
                }

                // Layer 2: Check if point is inside any Label 1 text region
                let in_text_region = text_regions_relative.iter().any(|region| {
                    let ix = x as i32;
                    let iy = y as i32;
                    ix >= region[0] && ix < region[2] && iy >= region[1] && iy < region[3]
                });

                if !in_text_region {
                    label1_protected_count += 1;
                    continue;
                }

                // Layer 3: Alpha channel confirms visibility
                let is_visible = pixel[3] > 0;

                // Erase if ALL THREE conditions are true
                if in_safe_interior && in_text_region && is_visible {
                    cleaned.put_pixel(x, y, Rgba([255, 255, 255, pixel[3]]));
                    erased_count += 1;
                }
            }
        }

        debug!("  Pixels protected by edge erosion: {} pixels", edge_protected_count);
        debug!("  Pixels protected by Label 1 constraint: {} pixels", label1_protected_count);
        debug!("✅ EDGE-BASED CLEANING SUCCESS: {} pixels erased", erased_count);

        if erased_count == 0 {
            warn!("⚠️  Edge-based cleaning erased 0 pixels - this might indicate an issue!");
            warn!("  Safe interior mask coverage: {:.1}%",
                safe_interior_mask.pixels().filter(|p| p[0] == 255).count() as f32 /
                (bubble_img.width() * bubble_img.height()) as f32 * 100.0);
        }

        cleaned
    }


    /// Create ellipse-shaped structuring element for morphological operations
    /// Returns list of (x, y) offsets that approximate an ellipse
    #[allow(dead_code)]
    fn create_ellipse_kernel(width: u32, height: u32) -> Vec<(i32, i32)> {
        let mut kernel = Vec::new();
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let a = cx;
        let b = cy;

        for dy in -(height as i32 / 2)..=(height as i32 / 2) {
            for dx in -(width as i32 / 2)..=(width as i32 / 2) {
                // Check if point is inside ellipse: (x/a)^2 + (y/b)^2 <= 1
                let fx = dx as f32;
                let fy = dy as f32;
                if (fx * fx) / (a * a) + (fy * fy) / (b * b) <= 1.0 {
                    kernel.push((dx, dy));
                }
            }
        }

        kernel
    }

    /// Convert GrayImage to OpenCV Mat
    fn gray_to_mat(img: &GrayImage) -> Result<core::Mat> {
        let (width, height) = (img.width() as i32, img.height() as i32);
        let data = img.as_raw();

        // Create Mat from raw data slice
        let mat_ref = core::Mat::new_rows_cols_with_data(height, width, data)
            .context("Failed to create Mat from image data")?;

        // Clone the Mat to own the data
        mat_ref.try_clone()
            .context("Failed to clone Mat - possible null data pointer")
    }

    /// Convert OpenCV Mat back to GrayImage
    fn mat_to_gray(mat: &core::Mat) -> Result<GrayImage> {
        let rows = mat.rows();
        let cols = mat.cols();

        if rows <= 0 || cols <= 0 {
            anyhow::bail!("Invalid Mat dimensions: {}x{}", rows, cols);
        }

        // Validate Mat has data before accessing
        // Note: OpenCV Mat.data() returns a raw pointer, we check rows/cols instead
        if mat.rows() <= 0 || mat.cols() <= 0 {
            anyhow::bail!("Mat has invalid dimensions - cannot convert to GrayImage");
        }

        let mut img = GrayImage::new(cols as u32, rows as u32);

        // Copy data from Mat to GrayImage
        for y in 0..rows {
            for x in 0..cols {
                // Safely access pixel data with proper error handling
                let pixel_ptr = mat.at_2d::<u8>(y, x)
                    .context(format!("Failed to access Mat pixel at ({}, {})", x, y))?;
                let val = *pixel_ptr;
                img.put_pixel(x as u32, y as u32, Luma([val]));
            }
        }

        Ok(img)
    }

    /// OpenCV dilate operation (exact Python behavior)
    fn opencv_dilate(img: &GrayImage, kernel_size: i32, iterations: i32) -> Result<GrayImage> {
        let mat = Self::gray_to_mat(img)?;
        let mut result = mat.clone();

        // Create ellipse structuring element (match Python)
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_ELLIPSE,
            core::Size::new(kernel_size, kernel_size),
            core::Point::new(-1, -1),
        )?;

        // Apply dilation
        imgproc::dilate(
            &mat,
            &mut result,
            &kernel,
            core::Point::new(-1, -1),
            iterations,
            core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;

        Self::mat_to_gray(&result)
    }

    /// OpenCV erode operation (exact Python behavior)
    fn opencv_erode(img: &GrayImage, kernel_size: i32, iterations: i32) -> Result<GrayImage> {
        let mat = Self::gray_to_mat(img)?;
        let mut result = mat.clone();

        // Create ellipse structuring element (match Python)
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_ELLIPSE,
            core::Size::new(kernel_size, kernel_size),
            core::Point::new(-1, -1),
        )?;

        // Apply erosion
        imgproc::erode(
            &mat,
            &mut result,
            &kernel,
            core::Point::new(-1, -1),
            iterations,
            core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;

        Self::mat_to_gray(&result)
    }

    /// OpenCV morphologyEx MORPH_CLOSE (exact Python behavior)
    fn opencv_morphology_close(img: &GrayImage, kernel_size: i32, iterations: i32) -> Result<GrayImage> {
        let mat = Self::gray_to_mat(img)?;
        let mut result = mat.clone();

        // Create ellipse structuring element (match Python)
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_ELLIPSE,
            core::Size::new(kernel_size, kernel_size),
            core::Point::new(-1, -1),
        )?;

        // Apply morphological closing
        for _ in 0..iterations {
            let temp = result.clone();
            imgproc::morphology_ex(
                &temp,
                &mut result,
                imgproc::MORPH_CLOSE,
                &kernel,
                core::Point::new(-1, -1),
                1,
                core::BORDER_CONSTANT,
                imgproc::morphology_default_border_value()?,
            )?;
        }

        Self::mat_to_gray(&result)
    }

    /// Create safe interior mask using edge detection + erosion (Python proto.md algorithm)
    /// This prevents text removal from bleeding outside bubble boundaries
    ///
    /// Algorithm:
    /// 1. Detect edges (Canny)
    /// 2. Create interior candidate (NOT edges)
    /// 3. Find pure white areas (>240)
    /// 4. Combine (white AND interior)
    /// 5. ERODE INWARD (critical!) - creates safety margin
    /// 6. Close gaps (using OpenCV for exact Python parity)
    /// 7. Get largest component
    ///
    /// Returns: Binary mask where 255 = safe to erase, 0 = protected
    fn create_safe_interior_mask(bubble_img: &RgbaImage) -> Result<GrayImage> {
        let start = std::time::Instant::now();
        debug!("🔍 Creating safe interior mask using edge detection + erosion");

        // Convert to grayscale
        let gray = image::DynamicImage::ImageRgba8(bubble_img.clone()).to_luma8();

        // Step 1: Detect edges using Canny (thresholds: 30, 100)
        debug!("  Step 1: Canny edge detection (30, 100)");
        let edges = canny(&gray, 30.0, 100.0);

        // Step 2: Dilate edges using OpenCV (exact Python behavior)
        debug!("  Step 2: Dilate edges (3x3 ellipse, OpenCV)");
        let edges_dilated = Self::opencv_dilate(&edges, 3, 1)?;

        // Step 3: Create interior candidate (invert edges)
        debug!("  Step 3: Invert edges to get interior candidate");
        let mut interior_candidate = GrayImage::new(edges_dilated.width(), edges_dilated.height());
        for (x, y, pixel) in edges_dilated.enumerate_pixels() {
            // Invert: 255 becomes 0, 0 becomes 255
            let inverted = if pixel[0] > 0 { 0 } else { 255 };
            interior_candidate.put_pixel(x, y, Luma([inverted]));
        }

        // Step 4: Find pure white areas (threshold > 240, not screentone)
        debug!("  Step 4: Threshold for pure white areas (>240)");
        let white_areas = threshold(&gray, 240, ThresholdType::Binary);

        // Step 5: Combine constraints (white AND interior)
        debug!("  Step 5: Combine (white AND interior)");
        let mut safe_interior = GrayImage::new(white_areas.width(), white_areas.height());
        for y in 0..safe_interior.height() {
            for x in 0..safe_interior.width() {
                let white = white_areas.get_pixel(x, y)[0];
                let interior = interior_candidate.get_pixel(x, y)[0];
                // Both must be 255 (white AND interior)
                let combined = if white == 255 && interior == 255 { 255 } else { 0 };
                safe_interior.put_pixel(x, y, Luma([combined]));
            }
        }

        // Step 6: ERODE INWARD using OpenCV (exact Python behavior)
        debug!("  Step 6: Erode inward (7x7 ellipse, 2 iterations, OpenCV) - CRITICAL");
        let eroded = Self::opencv_erode(&safe_interior, 7, 2)?;

        // Step 7: Morphological CLOSE - Fill text holes using OpenCV
        // Note: Python uses 9x9 kernel, but opencv-rust requires 41x41 to achieve same result
        // This compensates for implementation differences between Python cv2 and Rust opencv bindings
        debug!("  Step 7: Morphological close (41x41 ellipse, 6 iterations, OpenCV)");
        let closed = Self::opencv_morphology_close(&eroded, 41, 6)?;

        // Step 8: Get largest connected component (main bubble interior)
        debug!("  Step 8: Extract largest connected component");
        let final_mask = Self::get_largest_component(&closed);

        // Calculate statistics
        let total_pixels = (bubble_img.width() * bubble_img.height()) as u32;
        let safe_pixels = final_mask.pixels().filter(|p| p[0] == 255).count() as u32;
        let safe_percent = (safe_pixels as f32 / total_pixels as f32) * 100.0;

        let elapsed = start.elapsed();
        debug!("✅ Safe interior mask created: {:.1}% of bubble ({}/{} pixels) in {:.2}ms",
            safe_percent, safe_pixels, total_pixels, elapsed.as_secs_f64() * 1000.0);

        Ok(final_mask)
    }

    /// Extract largest connected component from binary image
    /// This removes noise and ensures we have one clean interior shape
    fn get_largest_component(binary_img: &GrayImage) -> GrayImage {
        // Simple flood-fill based connected component extraction
        let width = binary_img.width();
        let height = binary_img.height();
        let mut visited = vec![vec![false; width as usize]; height as usize];
        let mut largest_component = Vec::new();
        let mut largest_size = 0;

        for y in 0..height {
            for x in 0..width {
                if binary_img.get_pixel(x, y)[0] == 255 && !visited[y as usize][x as usize] {
                    // Found new component, flood fill it
                    let component = Self::flood_fill(binary_img, x, y, &mut visited);
                    if component.len() > largest_size {
                        largest_size = component.len();
                        largest_component = component;
                    }
                }
            }
        }

        // Create output image with only largest component
        let mut result = GrayImage::new(width, height);
        for (x, y) in largest_component {
            result.put_pixel(x, y, Luma([255]));
        }

        debug!("    Largest component: {} pixels", largest_size);
        result
    }

    /// Flood fill helper for connected component detection
    /// Uses saturating arithmetic to prevent overflow bugs
    fn flood_fill(img: &GrayImage, start_x: u32, start_y: u32, visited: &mut Vec<Vec<bool>>) -> Vec<(u32, u32)> {
        let mut component = Vec::new();
        let mut stack = vec![(start_x, start_y)];
        let width = img.width();
        let height = img.height();

        while let Some((x, y)) = stack.pop() {
            if x >= width || y >= height || visited[y as usize][x as usize] {
                continue;
            }

            if img.get_pixel(x, y)[0] != 255 {
                continue;
            }

            visited[y as usize][x as usize] = true;
            component.push((x, y));

            // Add 4-connected neighbors (using saturating arithmetic to prevent overflow)
            if x > 0 {
                stack.push((x.saturating_sub(1), y));
            }
            let next_x = x.saturating_add(1);
            if next_x < width {
                stack.push((next_x, y));
            }
            if y > 0 {
                stack.push((x, y.saturating_sub(1)));
            }
            let next_y = y.saturating_add(1);
            if next_y < height {
                stack.push((x, next_y));
            }
        }

        component
    }

    pub async fn render_bubble_simple_background(
        &self,
        bubble_img: &RgbaImage,
        detection: &BubbleDetection,
        translation: &TextTranslation,
    ) -> Result<DynamicImage> {
        Self::render_bubble_simple_background_impl(self, bubble_img, detection, translation, false).await
    }

    pub async fn render_bubble_simple_background_skip_cleaning(
        &self,
        bubble_img: &RgbaImage,
        detection: &BubbleDetection,
        translation: &TextTranslation,
    ) -> Result<DynamicImage> {
        Self::render_bubble_simple_background_impl(self, bubble_img, detection, translation, true).await
    }

    async fn render_bubble_simple_background_impl(
        &self,
        bubble_img: &RgbaImage,
        detection: &BubbleDetection,
        translation: &TextTranslation,
        skip_cleaning: bool,
    ) -> Result<DynamicImage> {
        debug!("🎨 [RENDER_SIMPLE] Starting NEW transparent text rendering (skip_cleaning: {})", skip_cleaning);
        let simple_start = std::time::Instant::now();

        let width = (detection.bbox[2] - detection.bbox[0]) as u32;
        let height = (detection.bbox[3] - detection.bbox[1]) as u32;

        // Step 1: Clean text from original bubble (unless skip_cleaning is true for insertion mode)
        let cleaned_bubble = if skip_cleaning {
            debug!("Step 1: SKIPPING cleaning (insertion mode - bubble already cleaned)");
            bubble_img.clone()
        } else {
            debug!("Step 1: Cleaning {} text regions from bubble using edge-based detection",
                detection.text_regions.len());
            Self::clean_text_from_bubble(
                bubble_img,
                &detection.text_regions,
                &detection.bbox
            )
        };

        // Step 2: Create TRANSPARENT canvas for text rendering at upscale resolution
        let upscale_factor = self.config.upscale_factor();
        let upscaled_width = width * upscale_factor;
        let upscaled_height = height * upscale_factor;

        debug!("Step 2: Creating transparent canvas {}x{}", upscaled_width, upscaled_height);
        let mut transparent_canvas = RgbaImage::from_pixel(
            upscaled_width,
            upscaled_height,
            Rgba([0, 0, 0, 0])  // Fully transparent
        );

        // Step 3: Render BLACK text on transparent canvas (constrained to Label 1 regions)
        debug!("Step 3: Rendering black text on transparent canvas");
        // Convert text regions from absolute to bubble-relative coordinates, then scale
        let upscaled_text_regions: Vec<[i32; 4]> = detection.text_regions.iter()
            .map(|region| {
                // Convert to bubble-relative first
                let rel_x1 = region[0] - detection.bbox[0];
                let rel_y1 = region[1] - detection.bbox[1];
                let rel_x2 = region[2] - detection.bbox[0];
                let rel_y2 = region[3] - detection.bbox[1];
                // Then scale to upscaled resolution
                [
                    rel_x1 * upscale_factor as i32,
                    rel_y1 * upscale_factor as i32,
                    rel_x2 * upscale_factor as i32,
                    rel_y2 * upscale_factor as i32,
                ]
            })
            .collect();
        transparent_canvas = self.render_text_on_transparent(
            transparent_canvas,
            translation,
            &upscaled_text_regions
        )?;

        // Step 4: Downscale transparent text for anti-aliasing
        debug!("Step 4: Downscaling transparent text from {}x{} to {}x{}",
            upscaled_width, upscaled_height, width, height);
        let text_layer = image::DynamicImage::ImageRgba8(transparent_canvas).resize_exact(
            width,
            height,
            image::imageops::FilterType::Lanczos3,
        ).to_rgba8();

        // Step 5: Composite transparent text onto cleaned bubble
        debug!("Step 5: Compositing text onto cleaned bubble");
        let mut final_img = cleaned_bubble;
        for (x, y, pixel) in text_layer.enumerate_pixels() {
            if pixel[3] > 0 {  // If text pixel has alpha > 0
                let base = final_img.get_pixel(x, y);
                let alpha = pixel[3] as f32 / 255.0;

                // Alpha blend text over cleaned bubble
                final_img.put_pixel(x, y, Rgba([
                    ((pixel[0] as f32 * alpha + base[0] as f32 * (1.0 - alpha)) as u8),
                    ((pixel[1] as f32 * alpha + base[1] as f32 * (1.0 - alpha)) as u8),
                    ((pixel[2] as f32 * alpha + base[2] as f32 * (1.0 - alpha)) as u8),
                    255,
                ]));
            }
        }

        let total_time = simple_start.elapsed();
        debug!("✅ [RENDER_SIMPLE] NEW transparent rendering completed in {:.2}ms",
            total_time.as_secs_f64() * 1000.0);

        Ok(DynamicImage::ImageRgba8(final_img))
    }

    pub async fn render_bubble_complex_background(
        &self,
        cleaned_bubble_bytes: &[u8],
        detection: &BubbleDetection,
        translation: &TextTranslation,
    ) -> Result<DynamicImage> {
        debug!("🎨 [RENDER_COMPLEX] Starting complex background rendering");
        let complex_start = std::time::Instant::now();

        let cleaned_img = image::load_from_memory(cleaned_bubble_bytes)?;
        debug!("Loaded AI-cleaned image: {}x{}", cleaned_img.width(), cleaned_img.height());

        let width = (detection.bbox[2] - detection.bbox[0]) as u32;
        let height = (detection.bbox[3] - detection.bbox[1]) as u32;

        // CRITICAL FIX: Upscale to configured resolution for high-quality text rendering
        // This matches the simple background path for consistent quality
        let upscale_factor = self.config.upscale_factor();
        let upscaled_width = width * upscale_factor;
        let upscaled_height = height * upscale_factor;

        // Upscale cleaned background to 3x resolution
        let mut img = image::DynamicImage::from(cleaned_img.to_rgba8())
            .resize_exact(upscaled_width, upscaled_height, image::imageops::FilterType::Lanczos3)
            .to_rgba8();

        // Render text at high resolution
        img = self.render_text_on_image(img, translation)?;

        // Downscale to original size for anti-aliasing
        debug!("Downscaling from {}x{} to {}x{} with Lanczos3",
            upscaled_width, upscaled_height, width, height);
        let final_img = image::DynamicImage::ImageRgba8(img).resize_exact(
            width,
            height,
            image::imageops::FilterType::Lanczos3,
        );

        let total_time = complex_start.elapsed();
        debug!("✅ [RENDER_COMPLEX] Completed in {:.2}ms", total_time.as_secs_f64() * 1000.0);

        Ok(final_img)
    }

    pub fn composite_bubble_onto_page(
        &self,
        page_img: &mut RgbaImage,
        bubble_img: &DynamicImage,
        detection: &BubbleDetection,
    ) {
        let x1 = detection.bbox[0].max(0) as u32;
        let y1 = detection.bbox[1].max(0) as u32;

        let bubble_rgba = bubble_img.to_rgba8();

        for (dx, dy, pixel) in bubble_rgba.enumerate_pixels() {
            let px = x1 + dx;
            let py = y1 + dy;

            if px < page_img.width() && py < page_img.height() {
                page_img.put_pixel(px, py, *pixel);
            }
        }
    }
}

// Trait implementation for BubbleRenderer
#[async_trait]
impl BubbleRenderer for RenderingService {
    async fn render_bubble_simple_background(
        &self,
        bubble_img: &RgbaImage,
        detection: &BubbleDetection,
        translation: &TextTranslation,
    ) -> Result<DynamicImage> {
        self.render_bubble_simple_background(bubble_img, detection, translation).await
    }

    async fn render_bubble_complex_background(
        &self,
        cleaned_bubble_bytes: &[u8],
        detection: &BubbleDetection,
        translation: &TextTranslation,
    ) -> Result<DynamicImage> {
        self.render_bubble_complex_background(cleaned_bubble_bytes, detection, translation).await
    }

    fn composite_bubble_onto_page(
        &self,
        page_img: &mut RgbaImage,
        bubble_img: &DynamicImage,
        detection: &BubbleDetection,
    ) {
        self.composite_bubble_onto_page(page_img, bubble_img, detection)
    }

    fn analyze_background_complexity(img: &RgbaImage) -> (bool, Option<Rgba<u8>>) {
        Self::analyze_background_complexity(img)
    }
}
