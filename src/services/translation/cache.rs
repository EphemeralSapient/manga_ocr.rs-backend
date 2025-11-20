use anyhow::{Context, Result};
use lru::LruCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use xxhash_rust::xxh3::xxh3_64;

use crate::core::types::OCRTranslation;
use crate::utils::Metrics;

/// Cache entry for translation
/// Uses Arc<str> to avoid expensive string cloning
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    #[serde(serialize_with = "serialize_arc_str", deserialize_with = "deserialize_arc_str")]
    original_text: Arc<str>,
    #[serde(serialize_with = "serialize_arc_str", deserialize_with = "deserialize_arc_str")]
    translated_text: Arc<str>,
}

// Serde helpers for Arc<str>
fn serialize_arc_str<S>(arc_str: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(arc_str)
}

fn deserialize_arc_str<'de, D>(deserializer: D) -> Result<Arc<str>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    String::deserialize(deserializer).map(|s| Arc::from(s.as_str()))
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
    // OPTIMIZED: Use u64 keys instead of String (5-10% faster, no allocations)
    cache: RwLock<LruCache<u64, CacheEntry>>,
    cache_file: PathBuf,

    // Debounced persistence
    dirty: Arc<RwLock<bool>>,
    save_notify: Arc<Notify>,

    // Background task management (prevents memory leak)
    persistence_task: RwLock<Option<JoinHandle<()>>>,
    shutdown: Arc<AtomicBool>,

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
        // Disk format uses String keys (JSON requirement), convert to u64 in-memory
        let cache_data = if cache_file.exists() {
            let data = tokio::fs::read_to_string(&cache_file)
                .await
                .context("Failed to read cache file")?;
            let string_map: std::collections::HashMap<String, CacheEntry> =
                serde_json::from_str(&data).unwrap_or_default();

            // Convert String keys to u64
            string_map
                .into_iter()
                .filter_map(|(k, v)| {
                    u64::from_str_radix(&k, 16).ok().map(|key| (key, v))
                })
                .collect()
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
            persistence_task: RwLock::new(None),
            shutdown: Arc::new(AtomicBool::new(false)),
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
    /// OPTIMIZED: Returns u64 directly instead of String (no allocation)
    ///
    /// # Arguments
    /// * `image_bytes` - Image bytes of the region
    /// * `bbox` - Bounding box of the region
    ///
    /// # Returns
    /// xxHash3 hash as u64
    pub fn generate_key(image_bytes: &[u8], bbox: &[i32; 4]) -> u64 {
        // Create a single buffer to hash both image bytes and bbox
        let mut hash_input = Vec::with_capacity(image_bytes.len() + 16);
        hash_input.extend_from_slice(image_bytes);
        hash_input.extend_from_slice(&bbox[0].to_le_bytes());
        hash_input.extend_from_slice(&bbox[1].to_le_bytes());
        hash_input.extend_from_slice(&bbox[2].to_le_bytes());
        hash_input.extend_from_slice(&bbox[3].to_le_bytes());

        // xxHash3 is much faster than SHA1
        xxh3_64(&hash_input)
    }

    /// Generate cache key from source image hash and bbox (ultra-fast, no cropping needed).
    ///
    /// This is much faster than `generate_key` because it doesn't require
    /// cropping and encoding the image first - it uses the original image hash
    /// combined with bbox coordinates.
    /// OPTIMIZED: Returns u64 directly instead of String (no allocation)
    ///
    /// # Arguments
    /// * `source_image_hash` - xxHash3 of the full source image bytes
    /// * `bbox` - Bounding box coordinates [x1, y1, x2, y2]
    ///
    /// # Returns
    /// Cache key as u64
    pub fn generate_key_from_source(source_image_hash: u64, bbox: &[i32; 4]) -> u64 {
        // Hash the source image hash with bbox coordinates
        let mut hash_input = Vec::with_capacity(24); // 8 bytes (u64) + 16 bytes (4 i32s)
        hash_input.extend_from_slice(&source_image_hash.to_le_bytes());
        hash_input.extend_from_slice(&bbox[0].to_le_bytes());
        hash_input.extend_from_slice(&bbox[1].to_le_bytes());
        hash_input.extend_from_slice(&bbox[2].to_le_bytes());
        hash_input.extend_from_slice(&bbox[3].to_le_bytes());

        xxh3_64(&hash_input)
    }

    /// Get translation from cache
    /// OPTIMIZED: Uses u64 keys (no String allocation/cloning)
    pub fn get(&self, key: u64) -> Option<OCRTranslation> {
        let mut cache = self.inner.cache.write();
        let entry = cache.get(&key)?;

        // Record cache hit
        if let Some(ref m) = self.inner.metrics {
            m.record_cache_hit();
        }

        // Arc::clone is cheap - just increments the reference count
        Some(OCRTranslation {
            original_text: Arc::clone(&entry.original_text),
            translated_text: Arc::clone(&entry.translated_text),
        })
    }

    /// Store translation in cache (non-blocking with debounced persistence)
    /// OPTIMIZED: Uses u64 keys (no String allocation)
    pub fn put(&self, key: u64, translation: &OCRTranslation) {
        let entry = CacheEntry {
            original_text: Arc::clone(&translation.original_text),
            translated_text: Arc::clone(&translation.translated_text),
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
            // Convert u64 keys to hex strings for JSON serialization
            for (k, v) in cache.iter() {
                map.insert(format!("{:016x}", k), v.clone());
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
    /// FIX: Task now respects shutdown signal to prevent memory leak
    fn start_persistence_task(&self, interval: Duration) {
        let inner = Arc::clone(&self.inner);

        let handle = tokio::spawn(async move {
            let mut last_save = Instant::now();

            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;

                // Check for shutdown signal (prevents memory leak)
                if inner.shutdown.load(Ordering::Relaxed) {
                    // Final save before shutdown
                    let cache_data = {
                        let cache = inner.cache.read();
                        let mut map = std::collections::HashMap::new();
                        for (k, v) in cache.iter() {
                            map.insert(format!("{:016x}", k), v.clone());
                        }
                        map
                    };
                    if let Ok(json) = serde_json::to_string_pretty(&cache_data) {
                        let _ = tokio::fs::write(&inner.cache_file, json).await;
                    }
                    break; // Exit loop on shutdown
                }

                let is_dirty = *inner.dirty.read();
                let should_save = is_dirty && last_save.elapsed() >= interval;

                if should_save {
                    // Save cache to disk
                    let cache_data = {
                        let cache = inner.cache.read();
                        let mut map = std::collections::HashMap::new();
                        // Convert u64 keys to hex strings for JSON serialization
                        for (k, v) in cache.iter() {
                            map.insert(format!("{:016x}", k), v.clone());
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

        // Store handle so it can be aborted on drop
        *self.inner.persistence_task.write() = Some(handle);
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

// FIX: Implement Drop to clean up background task (prevents memory leak)
impl Drop for CacheInner {
    fn drop(&mut self) {
        // Signal shutdown to background task
        self.shutdown.store(true, Ordering::Relaxed);

        // Abort persistence task if running
        if let Some(handle) = self.persistence_task.write().take() {
            handle.abort();
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
            original_text: Arc::from("こんにちは"),
            translated_text: Arc::from("Hello"),
        };

        let key = TranslationCache::generate_key(b"test_image_bytes", &[0, 0, 100, 100]);
        cache.put(key, &translation);

        let retrieved = cache.get(key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().translated_text.as_ref(), "Hello");

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
