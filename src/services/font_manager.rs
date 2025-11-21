// Font Manager - Downloads and caches Google Fonts

use anyhow::{Context, Result};
use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Font Manager - handles downloading and caching Google Fonts
pub struct FontManager {
    cache_dir: PathBuf,
    font_cache: Arc<Mutex<LruCache<String, Vec<u8>>>>,
    client: reqwest::Client,
}

impl FontManager {
    /// Create a new FontManager
    pub fn new(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().join("fonts");
        std::fs::create_dir_all(&cache_dir)
            .context("Failed to create font cache directory")?;

        let font_cache = Arc::new(Mutex::new(
            LruCache::new(NonZeroUsize::new(20).unwrap())
        ));

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        info!("Font Manager initialized (cache: {})", cache_dir.display());

        Ok(Self {
            cache_dir,
            font_cache,
            client,
        })
    }

    /// Get or download a Google Font
    /// Returns the font data as bytes
    pub async fn get_google_font(&self, family: &str) -> Result<Vec<u8>> {
        let cache_key = format!("google_{}", family);

        // Check in-memory cache first
        {
            let mut cache = self.font_cache.lock();
            if let Some(font_data) = cache.get(&cache_key) {
                debug!("Google Font '{}' found in memory cache", family);
                return Ok(font_data.clone());
            }
        }

        // Check disk cache
        let font_path = self.cache_dir.join(format!("{}.ttf", sanitize_filename(family)));
        if font_path.exists() {
            debug!("Google Font '{}' found in disk cache", family);
            let font_data = tokio::fs::read(&font_path)
                .await
                .context("Failed to read cached font")?;

            // Store in memory cache
            self.font_cache.lock().put(cache_key, font_data.clone());

            return Ok(font_data);
        }

        // Download from Google Fonts
        info!("Downloading Google Font '{}'...", family);
        let font_data = self.download_google_font(family).await?;

        // Save to disk cache
        tokio::fs::write(&font_path, &font_data)
            .await
            .context("Failed to cache font to disk")?;

        // Store in memory cache
        self.font_cache.lock().put(cache_key, font_data.clone());

        info!("Google Font '{}' downloaded and cached", family);
        Ok(font_data)
    }

    /// Download a Google Font from Google Fonts API
    async fn download_google_font(&self, family: &str) -> Result<Vec<u8>> {
        // Step 1: Get CSS from Google Fonts API
        let css_url = format!(
            "https://fonts.googleapis.com/css2?family={}&display=swap",
            urlencoding::encode(family)
        );

        let css_response = self.client
            .get(&css_url)
            .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .send()
            .await
            .context("Failed to fetch Google Fonts CSS")?;

        if !css_response.status().is_success() {
            anyhow::bail!("Google Fonts API returned status: {}", css_response.status());
        }

        let css_text = css_response.text().await?;

        // Step 2: Extract font URL from CSS
        // CSS format: src: url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxK.woff2) format('woff2');
        let font_url = extract_font_url(&css_text)
            .ok_or_else(|| anyhow::anyhow!("Could not find font URL in CSS for '{}'", family))?;

        debug!("Font URL: {}", font_url);

        // Step 3: Download the actual font file
        let font_response = self.client
            .get(&font_url)
            .send()
            .await
            .context("Failed to download font file")?;

        if !font_response.status().is_success() {
            anyhow::bail!("Font download failed with status: {}", font_response.status());
        }

        let font_data = font_response.bytes().await?.to_vec();

        // Convert woff2 to ttf if needed
        // For now, we'll try to use the font directly
        // cosmic-text should handle both TTF and WOFF2

        Ok(font_data)
    }

    /// Clear the font cache
    pub fn clear_cache(&self) -> Result<()> {
        self.font_cache.lock().clear();
        info!("Font memory cache cleared");
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.font_cache.lock();
        (cache.len(), cache.cap().get())
    }
}

/// Extract font URL from Google Fonts CSS
fn extract_font_url(css: &str) -> Option<String> {
    // Look for the first woff2 or ttf URL
    for line in css.lines() {
        if line.contains("src:") && line.contains("url(") {
            if let Some(start) = line.find("url(") {
                if let Some(end) = line[start..].find(')') {
                    let url = &line[start + 4..start + end];
                    let url = url.trim_matches(|c| c == '\'' || c == '"' || c == ' ');

                    // Prefer woff2 for better compression, but accept ttf
                    if url.contains(".woff2") || url.contains(".ttf") {
                        return Some(url.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Sanitize filename to prevent path traversal
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            ' ' => '_',
            _ => '-',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("Roboto"), "Roboto");
        assert_eq!(sanitize_filename("Open Sans"), "Open_Sans");
        assert_eq!(sanitize_filename("Noto Sans JP"), "Noto_Sans_JP");
        assert_eq!(sanitize_filename("Font@#$%Name"), "Font----Name");
    }

    #[test]
    fn test_extract_font_url() {
        let css = r#"
            @font-face {
              font-family: 'Roboto';
              src: url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxK.woff2) format('woff2');
            }
        "#;

        let url = extract_font_url(css);
        assert!(url.is_some());
        assert!(url.unwrap().contains("KFOmCnqEu92Fr1Mu4mxK.woff2"));
    }
}
