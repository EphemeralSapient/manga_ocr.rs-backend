use anyhow::Result;
use cosmic_text::{
    Attrs, Buffer, Color as CosmicColor, Family, FontSystem, Metrics, Shaping, SwashCache,
    Weight as CosmicWeight, Wrap,
};
use image::{Rgba, RgbaImage};
use std::sync::Arc;
use parking_lot::Mutex;  // Use parking_lot for better performance with sync mutex
use tracing::{debug, info};

/// Detect if text contains CJK (Chinese, Japanese, Korean) characters
/// CJK characters are visually denser and need larger font sizes for readability
pub fn is_cjk_text(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
            '\u{3040}'..='\u{309F}' |  // Hiragana
            '\u{30A0}'..='\u{30FF}' |  // Katakana
            '\u{AC00}'..='\u{D7AF}'    // Hangul
        )
    })
}

/// High-quality text renderer using cosmic-text
/// Provides advanced text shaping, proper multi-line layout, and better font rendering
pub struct CosmicTextRenderer {
    font_system: Arc<Mutex<FontSystem>>,
    swash_cache: Arc<Mutex<SwashCache>>,
}

impl CosmicTextRenderer {
    pub fn new() -> Self {
        info!("Initializing CosmicTextRenderer (fonts/ directory only, no system scan)");

        // Create font system with ONLY custom fonts (skip system font scanning)
        let font_system = Self::create_font_system_from_directory();

        let swash_cache = SwashCache::new();

        info!("✓ Renderer initialized");

        Self {
            font_system: Arc::new(Mutex::new(font_system)),
            swash_cache: Arc::new(Mutex::new(swash_cache)),
        }
    }

    /// Create FontSystem with only custom fonts from fonts/ directory
    /// This skips system font scanning for faster initialization
    fn create_font_system_from_directory() -> FontSystem {
        use cosmic_text::fontdb;

        // Create empty font database (no system fonts)
        let mut db = fontdb::Database::new();

        // Load only custom fonts from fonts/ directory
        let custom_fonts = vec![
            ("fonts/anime_ace.ttf", "Anime Ace"),
            ("fonts/anime_ace_3.ttf", "Anime Ace 3"),
            ("fonts/arial-unicode.ttf", "Arial Unicode"),
            ("fonts/comic-sans.ttf", "Comic Sans"),
            ("fonts/msyh.ttc", "Microsoft YaHei"),
            ("fonts/noto-sans-mono-cjk.ttc", "Noto Sans Mono CJK"),
        ];

        for (path, name) in custom_fonts {
            if let Ok(font_data) = std::fs::read(path) {
                db.load_font_data(font_data);
                debug!("✓ Font: {} from {}", name, path);
            } else {
                info!("⚠ Font not found: {} ({})", name, path);
            }
        }

        // Create FontSystem with custom database
        FontSystem::new_with_locale_and_db("en-US".to_string(), db)
    }

    /// Load a Google Font into the font system
    /// The font_data should be the raw font file bytes (TTF, OTF, or WOFF2)
    pub fn load_google_font(&self, font_data: Vec<u8>, family_name: &str) -> Result<()> {
        let mut font_system = self.font_system.lock();
        font_system.db_mut().load_font_data(font_data);
        info!("✓ Loaded Google Font: {}", family_name);
        Ok(())
    }

    /// Map font family string to cosmic-text Family
    /// For Google Fonts, the family name is used directly
    fn parse_font_family(font_family: &str) -> Family<'static> {
        match font_family {
            "anime-ace" => Family::Name("Anime Ace"),
            "anime-ace-3" => Family::Name("Anime Ace 3"),
            "arial" => Family::Name("Arial Unicode"),
            "comic-sans" => Family::Name("Comic Sans"),
            "ms-yahei" => Family::Name("Microsoft YaHei"),
            "noto-sans-mono-cjk" => Family::Name("Noto Sans Mono CJK"),
            "sans-serif-regular" | "sans-serif-bold" => Family::SansSerif,
            "serif-regular" | "serif-bold" => Family::Serif,
            "monospace-regular" | "monospace-bold" => Family::Monospace,
            // For any other font (like Google Fonts), use the name directly
            _ => {
                // If it looks like a font name (contains spaces or capitals), use it as-is
                if font_family.chars().any(|c| c.is_uppercase() || c == ' ') {
                    Family::Name(Box::leak(font_family.to_string().into_boxed_str()))
                } else {
                    debug!("Unknown font family '{}', defaulting to SansSerif", font_family);
                    Family::SansSerif
                }
            }
        }
    }

    /// Determine if font should be bold
    fn is_bold_font(font_family: &str) -> CosmicWeight {
        if font_family.contains("bold") {
            CosmicWeight::BOLD
        } else {
            CosmicWeight::NORMAL
        }
    }


    /// Determine optimal stroke color based on fill color
    fn get_stroke_color(fill_color: Rgba<u8>) -> Rgba<u8> {
        let brightness = fill_color[0] as u32 + fill_color[1] as u32 + fill_color[2] as u32;

        if brightness < 382 {
            Rgba([255, 255, 255, 255]) // White stroke for dark text
        } else {
            Rgba([0, 0, 0, 255]) // Black stroke for light text
        }
    }

    /// Measure text dimensions with cosmic-text using ACTUAL LAYOUT POSITIONS
    /// Returns (width, height) in pixels based on real glyph layout, not heuristics
    ///
    /// Uses LayoutRun's line_top + line_height for accurate vertical measurement,
    /// and line_w for horizontal measurement (no arbitrary padding).
    pub fn measure_text(
        &self,
        text: &str,
        font_family: &str,
        font_size: f32,
        max_width: Option<f32>,
    ) -> Result<(f32, f32)> {
        // Early return for empty text
        if text.trim().is_empty() {
            return Ok((0.0, 0.0));
        }

        let mut font_system = self.font_system.lock();

        let family = Self::parse_font_family(font_family);
        let weight = Self::is_bold_font(font_family);

        // Use font_size * 1.2 as default line height (cosmic-text standard)
        let line_height = font_size * 1.2;
        let metrics = Metrics::new(font_size, line_height);

        let mut buffer = Buffer::new(&mut font_system, metrics);

        if let Some(width) = max_width {
            buffer.set_size(&mut font_system, Some(width), None);
        }

        // Word wrap only - never break words mid-character
        buffer.set_wrap(&mut font_system, Wrap::Word);

        let attrs = Attrs::new().family(family).weight(weight);
        buffer.set_text(&mut font_system, text, &attrs, Shaping::Advanced);
        buffer.shape_until_scroll(&mut font_system, false);

        // Calculate dimensions from ACTUAL layout positions
        let mut max_width_seen = 0.0f32;
        let mut max_bottom = 0.0f32;

        for run in buffer.layout_runs() {
            // Width: use line_w directly (actual rendered width)
            max_width_seen = max_width_seen.max(run.line_w);

            // Height: track the bottom of the last line
            // line_top is Y offset to top of line, line_height is the line's height
            let line_bottom = run.line_top + run.line_height;
            max_bottom = max_bottom.max(line_bottom);
        }

        // If no runs, return zero dimensions
        if max_bottom == 0.0 {
            return Ok((0.0, 0.0));
        }

        Ok((max_width_seen, max_bottom))
    }

    /// Render text with advanced shaping and optional stroke
    ///
    /// # Arguments
    /// * `region_bounds` - Optional (min_x, min_y, max_x, max_y) to constrain rendering within specific region
    pub fn render_text(
        &self,
        img: &mut RgbaImage,
        text: &str,
        font_family: &str,
        font_size: f32,
        color: Rgba<u8>,
        x: i32,
        y: i32,
        max_width: Option<f32>,
        stroke_width: Option<i32>,
        _region_bounds: Option<(i32, i32, i32, i32)>,
    ) -> Result<()> {
        let stroke_color = stroke_width.map(|_| Self::get_stroke_color(color));

        // Render stroke first (if requested)
        if let (Some(width), Some(s_color)) = (stroke_width, stroke_color) {
            for offset_y in -width..=width {
                for offset_x in -width..=width {
                    if offset_x == 0 && offset_y == 0 {
                        continue;
                    }

                    let distance_sq = (offset_x * offset_x + offset_y * offset_y) as f32;
                    let radius_sq = (width * width) as f32;

                    if distance_sq <= radius_sq * 1.2 {
                        self.render_text_internal(
                            img,
                            text,
                            font_family,
                            font_size,
                            s_color,
                            x + offset_x,
                            y + offset_y,
                            max_width,
                            _region_bounds, // Apply region bounds to stroke as well
                        )?;
                    }
                }
            }
        }

        // Render fill text
        self.render_text_internal(img, text, font_family, font_size, color, x, y, max_width, _region_bounds)?;

        Ok(())
    }

    /// Internal text rendering implementation
    /// Uses consistent line_height = font_size * 1.2 to match measure_text
    ///
    /// # Arguments
    /// * `region_bounds` - Optional (min_x, min_y, max_x, max_y) to constrain rendering within specific region
    fn render_text_internal(
        &self,
        img: &mut RgbaImage,
        text: &str,
        font_family: &str,
        font_size: f32,
        color: Rgba<u8>,
        x: i32,
        y: i32,
        max_width: Option<f32>,
        _region_bounds: Option<(i32, i32, i32, i32)>,
    ) -> Result<()> {
        let family = Self::parse_font_family(font_family);
        let weight = Self::is_bold_font(font_family);

        let mut font_system = self.font_system.lock();

        // Use consistent line_height = font_size * 1.2 (matches measure_text)
        let line_height = font_size * 1.2;
        let metrics = Metrics::new(font_size, line_height);
        let mut buffer = Buffer::new(&mut font_system, metrics);

        if let Some(width) = max_width {
            buffer.set_size(&mut font_system, Some(width), None);
        }

        // Word wrap only - never break words mid-character
        buffer.set_wrap(&mut font_system, Wrap::Word);

        let attrs = Attrs::new().family(family).weight(weight);
        buffer.set_text(&mut font_system, text, &attrs, Shaping::Advanced);
        buffer.shape_until_scroll(&mut font_system, false);

        let cosmic_color = CosmicColor::rgba(color[0], color[1], color[2], color[3]);

        let mut swash_cache = self.swash_cache.lock();

        buffer.draw(&mut font_system, &mut swash_cache, cosmic_color, |px_x, px_y, _w, _h, pixel_color| {
            let img_x = x + px_x;
            let img_y = y + px_y;

            // Only check bounds to prevent crashes, don't clip text rendering
            // Text can now overflow freely without being clipped
            if img_x >= 0 && img_x < img.width() as i32
                && img_y >= 0 && img_y < img.height() as i32 {
                let existing = img.get_pixel(img_x as u32, img_y as u32);

                // Alpha blend
                let alpha = pixel_color.a() as f32 / 255.0;
                let inv_alpha = 1.0 - alpha;

                let blended = Rgba([
                    ((pixel_color.r() as f32 * alpha) + (existing[0] as f32 * inv_alpha)) as u8,
                    ((pixel_color.g() as f32 * alpha) + (existing[1] as f32 * inv_alpha)) as u8,
                    ((pixel_color.b() as f32 * alpha) + (existing[2] as f32 * inv_alpha)) as u8,
                    existing[3].max(pixel_color.a()),
                ]);

                img.put_pixel(img_x as u32, img_y as u32, blended);
            }
        });

        Ok(())
    }

    /// Render multi-line text with automatic layout and vertical alignment
    /// Uses consistent line_height = font_size * 1.2 to match measure_text
    pub fn render_multiline_text(
        &self,
        img: &mut RgbaImage,
        text: &str,
        font_family: &str,
        font_size: f32,
        color: Rgba<u8>,
        x: i32,
        y: i32,
        max_width: f32,
        max_height: f32,
        stroke_width: Option<i32>,
        vertical_align: VerticalAlign,
    ) -> Result<()> {
        debug!("Rendering multi-line text: size={:.1}px, max_width={:.0}px, max_height={:.0}px",
               font_size, max_width, max_height);

        // Calculate y_offset for vertical alignment
        let y_offset = {
            let mut font_system = self.font_system.lock();

            let family = Self::parse_font_family(font_family);
            let weight = Self::is_bold_font(font_family);

            // Use consistent line_height = font_size * 1.2
            let line_height = font_size * 1.2;
            let metrics = Metrics::new(font_size, line_height);
            let mut buffer = Buffer::new(&mut font_system, metrics);

            buffer.set_size(&mut font_system, Some(max_width), Some(max_height));
            buffer.set_wrap(&mut font_system, Wrap::Word);

            let attrs = Attrs::new().family(family).weight(weight);
            buffer.set_text(&mut font_system, text, &attrs, Shaping::Advanced);
            buffer.shape_until_scroll(&mut font_system, false);

            // Calculate actual text height from layout runs
            let mut max_bottom = 0.0f32;
            for run in buffer.layout_runs() {
                max_bottom = max_bottom.max(run.line_top + run.line_height);
            }

            match vertical_align {
                VerticalAlign::Top => 0,
                VerticalAlign::Middle => ((max_height - max_bottom) / 2.0).max(0.0) as i32,
                VerticalAlign::Bottom => (max_height - max_bottom).max(0.0) as i32,
            }
        };

        // Render with stroke and fill
        if let Some(width) = stroke_width {
            let stroke_color = Self::get_stroke_color(color);
            for offset_y in -width..=width {
                for offset_x in -width..=width {
                    if offset_x == 0 && offset_y == 0 {
                        continue;
                    }

                    let distance_sq = (offset_x * offset_x + offset_y * offset_y) as f32;
                    let radius_sq = (width * width) as f32;

                    if distance_sq <= radius_sq * 1.2 {
                        self.render_text_internal(
                            img,
                            text,
                            font_family,
                            font_size,
                            stroke_color,
                            x + offset_x,
                            y + y_offset + offset_y,
                            Some(max_width),
                            None, // No region bounds for stroke
                        )?;
                    }
                }
            }
        }

        // Render fill
        self.render_text_internal(img, text, font_family, font_size, color, x, y + y_offset, Some(max_width), None)?;

        Ok(())
    }

    /// Find optimal font size to fit text in given dimensions
    ///
    /// Uses BINARY SEARCH + LOCAL REFINEMENT for O(log n) performance.
    ///
    /// The challenge: font_size → dimensions isn't perfectly monotonic due to line breaks.
    /// A slightly larger font might cause fewer line breaks (word fits on line),
    /// resulting in smaller total height. This creates local discontinuities.
    ///
    /// Solution:
    /// 1. Binary search to find approximate fit (fast, O(log n))
    /// 2. Local neighborhood search (±3px) to find true maximum (handles discontinuities)
    pub fn find_optimal_font_size(
        &self,
        text: &str,
        font_family: &str,
        max_width: f32,
        max_height: f32,
        min_size: f32,
        max_size: f32,
        stroke_width: Option<f32>,
    ) -> Result<f32> {
        // Apply stroke margin
        let stroke_extra = stroke_width.unwrap_or(0.0) * 2.0;
        let effective_width = (max_width - stroke_extra).max(10.0);
        let effective_height = (max_height - stroke_extra).max(10.0);

        // Absolute minimum readable size
        let absolute_min = 6.0f32;
        let search_min = min_size.max(absolute_min);
        let search_max = max_size.min(effective_height); // Font can't be taller than box

        // Helper: check if size fits
        let fits = |size: f32| -> Result<bool> {
            let (w, h) = self.measure_text(text, font_family, size, Some(effective_width))?;
            Ok(w <= effective_width && h <= effective_height)
        };

        // PHASE 1: Binary search to find approximate boundary
        // Find largest size where text fits (within 1px precision)
        let mut lo = search_min;
        let mut hi = search_max;
        let mut best_fit = search_min;

        // Quick check: does max size fit?
        if fits(search_max)? {
            best_fit = search_max;
        } else if !fits(search_min)? {
            // Even min doesn't fit - try going smaller
            let mut emergency_size = search_min;
            while emergency_size >= absolute_min {
                if fits(emergency_size)? {
                    best_fit = emergency_size;
                    break;
                }
                emergency_size -= 1.0;
            }
            if emergency_size < absolute_min {
                tracing::warn!(
                    "Text doesn't fit even at {:.1}px: '{:.30}...'",
                    absolute_min, text
                );
                return Ok(absolute_min);
            }
        } else {
            // Binary search: find the largest fitting size
            while hi - lo > 1.0 {
                let mid = (lo + hi) / 2.0;
                if fits(mid)? {
                    lo = mid;
                    best_fit = mid;
                } else {
                    hi = mid;
                }
            }
        }

        // PHASE 2: Local refinement to handle line-break discontinuities
        // Check neighborhood of ±3px with 0.5px steps
        let mut final_best = best_fit;
        for offset in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let test_size = (best_fit + offset).clamp(search_min, search_max);
            if test_size > final_best && fits(test_size)? {
                final_best = test_size;
            }
        }

        tracing::debug!(
            "Font size: {:.1}px for {}x{} (text: '{:.20}...')",
            final_best, effective_width as i32, effective_height as i32, text
        );

        Ok(final_best)
    }
}

/// Vertical alignment options for text rendering
#[derive(Debug, Clone, Copy)]
pub enum VerticalAlign {
    Top,
    Middle,
    Bottom,
}

impl Default for CosmicTextRenderer {
    fn default() -> Self {
        Self::new()
    }
}
