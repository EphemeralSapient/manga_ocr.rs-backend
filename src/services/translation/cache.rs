use anyhow::{Context, Result};
use lru::LruCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use xxhash_rust::xxh3::xxh3_64;

use crate::core::types::OCRTranslation;
use crate::utils::Metrics;

/// Cache entry for translation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    original_text: String,
    translated_text: String,
}

/// Enhanced translation cache with async I/O, LRU eviction, and debounced persistence.
///
/// Improvements over the old cache:
/// - Uses xxHash3 (10-100x faster than SHA1)
/// - Async file I/O with tokio::fs (non-blocking)
/// - LRU eviction to limit memory usage
/// - Debounced persistence (saves periodically, not on every put)
/// - Integrated metrics tracking
#[derive(Clone)]
pub struct TranslationCache {
    inner: Arc<CacheInner>,
}

struct CacheInner {
    // LRU cache with parking_lot RwLock for better performance
    cache: RwLock<LruCache<String, CacheEntry>>,
    cache_file: PathBuf,

    // Debounced persistence
    dirty: Arc<RwLock<bool>>,
    save_notify: Arc<Notify>,

    // Metrics integration
    metrics: Option<Metrics>,
}

impl TranslationCache {
    /// Create new translation cache with configurable LRU size and persistence interval.
    ///
    /// # Arguments
    /// * `cache_dir` - Directory to store cache file
    /// * `max_entries` - Maximum number of entries before LRU eviction (default: 10000)
    /// * `save_interval` - How often to save to disk (default: 30 seconds, 0 = immediate)
    /// * `metrics` - Optional metrics collector
    pub async fn new(
        cache_dir: &str,
        max_entries: Option<usize>,
        save_interval: Option<Duration>,
        metrics: Option<Metrics>,
    ) -> Result<Self> {
        // Create cache directory if it doesn't exist
        let cache_path = Path::new(cache_dir);
        if !cache_path.exists() {
            tokio::fs::create_dir_all(cache_path)
                .await
                .context("Failed to create cache directory")?;
        }

        let cache_file = cache_path.join("translations.json");

        // Load existing cache from file (async)
        let cache_data = if cache_file.exists() {
            let data = tokio::fs::read_to_string(&cache_file)
                .await
                .context("Failed to read cache file")?;
            serde_json::from_str::<std::collections::HashMap<String, CacheEntry>>(&data)
                .unwrap_or_default()
        } else {
            std::collections::HashMap::new()
        };

        // Convert to LRU cache
        let max = NonZeroUsize::new(max_entries.unwrap_or(10000))
            .expect("max_entries must be > 0");
        let mut lru = LruCache::new(max);
        for (k, v) in cache_data {
            lru.put(k, v);
        }

        // Update metrics with initial cache size
        if let Some(ref m) = metrics {
            m.update_cache_size(lru.len());
        }

        let inner = Arc::new(CacheInner {
            cache: RwLock::new(lru),
            cache_file: cache_file.clone(),
            dirty: Arc::new(RwLock::new(false)),
            save_notify: Arc::new(Notify::new()),
            metrics,
        });

        let cache = Self { inner };

        // Start background persistence task if save_interval > 0
        if let Some(interval) = save_interval {
            if interval.as_secs() > 0 {
                cache.start_persistence_task(interval);
            }
        }

        Ok(cache)
    }

    /// Generate cache key from image bytes and bbox using xxHash3.
    ///
    /// xxHash3 is 10-100x faster than SHA1 and sufficient for cache key generation.
    ///
    /// # Arguments
    /// * `image_bytes` - Image bytes of the region
    /// * `bbox` - Bounding box of the region
    ///
    /// # Returns
    /// xxHash3 hash as hex string
    pub fn generate_key(image_bytes: &[u8], bbox: &[i32; 4]) -> String {
        // Create a single buffer to hash both image bytes and bbox
        let mut hash_input = Vec::with_capacity(image_bytes.len() + 16);
        hash_input.extend_from_slice(image_bytes);
        hash_input.extend_from_slice(&bbox[0].to_le_bytes());
        hash_input.extend_from_slice(&bbox[1].to_le_bytes());
        hash_input.extend_from_slice(&bbox[2].to_le_bytes());
        hash_input.extend_from_slice(&bbox[3].to_le_bytes());

        // xxHash3 is much faster than SHA1
        let hash = xxh3_64(&hash_input);
        format!("{:016x}", hash)
    }

    /// Get translation from cache
    pub fn get(&self, key: &str) -> Option<OCRTranslation> {
        let mut cache = self.inner.cache.write();
        let entry = cache.get(key)?;

        // Record cache hit
        if let Some(ref m) = self.inner.metrics {
            m.record_cache_hit();
        }

        Some(OCRTranslation {
            original_text: entry.original_text.clone(),
            translated_text: entry.translated_text.clone(),
        })
    }

    /// Store translation in cache (non-blocking with debounced persistence)
    pub fn put(&self, key: String, translation: &OCRTranslation) {
        let entry = CacheEntry {
            original_text: translation.original_text.clone(),
            translated_text: translation.translated_text.clone(),
        };

        {
            let mut cache = self.inner.cache.write();
            cache.put(key, entry);

            // Update metrics
            if let Some(ref m) = self.inner.metrics {
                m.update_cache_size(cache.len());
            }
        }

        // Mark as dirty (needs saving)
        *self.inner.dirty.write() = true;
        self.inner.save_notify.notify_one();
    }

    /// Record cache miss (for metrics)
    pub fn record_miss(&self) {
        if let Some(ref m) = self.inner.metrics {
            m.record_cache_miss();
        }
    }

    /// Manually trigger cache save (async, non-blocking)
    pub async fn save(&self) -> Result<()> {
        let cache_data = {
            let cache = self.inner.cache.read();
            let mut map = std::collections::HashMap::new();
            for (k, v) in cache.iter() {
                map.insert(k.clone(), v.clone());
            }
            map
        };

        let json = serde_json::to_string_pretty(&cache_data)
            .context("Failed to serialize cache")?;

        tokio::fs::write(&self.inner.cache_file, json)
            .await
            .context("Failed to write cache file")?;

        *self.inner.dirty.write() = false;

        Ok(())
    }

    /// Start background task for periodic persistence
    fn start_persistence_task(&self, interval: Duration) {
        let inner = Arc::clone(&self.inner);

        tokio::spawn(async move {
            let mut last_save = Instant::now();

            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;

                let is_dirty = *inner.dirty.read();
                let should_save = is_dirty && last_save.elapsed() >= interval;

                if should_save {
                    // Save cache to disk
                    let cache_data = {
                        let cache = inner.cache.read();
                        let mut map = std::collections::HashMap::new();
                        for (k, v) in cache.iter() {
                            map.insert(k.clone(), v.clone());
                        }
                        map
                    };

                    if let Ok(json) = serde_json::to_string_pretty(&cache_data) {
                        let _ = tokio::fs::write(&inner.cache_file, json).await;
                        *inner.dirty.write() = false;
                        last_save = Instant::now();
                    }
                }
            }
        });
    }

    /// Get cache statistics
    pub async fn stats(&self) -> (usize, f64) {
        let entries = self.inner.cache.read().len();

        // Calculate file size in MB (async)
        let size_bytes = if self.inner.cache_file.exists() {
            tokio::fs::metadata(&self.inner.cache_file)
                .await
                .ok()
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };

        let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

        (entries, size_mb)
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut cache = self.inner.cache.write();
        cache.clear();
        *self.inner.dirty.write() = true;

        if let Some(ref m) = self.inner.metrics {
            m.update_cache_size(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_put_get() {
        let cache = TranslationCache::new(
            ".cache_test",
            Some(100),
            Some(Duration::from_secs(5)),
            None,
        )
        .await
        .unwrap();

        let translation = OCRTranslation {
            original_text: "こんにちは".to_string(),
            translated_text: "Hello".to_string(),
        };

        let key = TranslationCache::generate_key(b"test_image_bytes", &[0, 0, 100, 100]);
        cache.put(key.clone(), &translation);

        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().translated_text, "Hello");

        // Cleanup
        let _ = tokio::fs::remove_dir_all(".cache_test").await;
    }

    #[test]
    fn test_xxhash_generation() {
        let key1 = TranslationCache::generate_key(b"test", &[0, 0, 100, 100]);
        let key2 = TranslationCache::generate_key(b"test", &[0, 0, 100, 100]);
        let key3 = TranslationCache::generate_key(b"test", &[0, 0, 100, 101]);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}
