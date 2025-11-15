use crate::services::traits::CacheStore;
use anyhow::Result;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use tracing::{info, debug, trace};

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    checksum: String,
    translation: String,
}

/// Thread-safe translation cache with internal synchronization
///
/// Uses RwLock for concurrent read access (high-contention scenario)
/// and atomic bool for dirty flag to avoid locking overhead.
///
/// **Thread Safety**: This cache is internally thread-safe and can be
/// shared across threads via Arc<Cache> without external Mutex wrapping.
pub struct Cache {
    cache_dir: PathBuf,
    cache_data: Arc<RwLock<HashMap<String, String>>>,
    dirty: AtomicBool,
}

impl Cache {
    pub fn new(cache_dir: &str) -> Result<Self> {
        debug!("💾 [CACHE] Initializing cache in directory: {}", cache_dir);
        let cache_dir = PathBuf::from(cache_dir);
        fs::create_dir_all(&cache_dir)?;

        let cache_file = cache_dir.join("translations.json.gz");
        let cache_data = if cache_file.exists() {
            debug!("Loading existing cache from: {:?}", cache_file);
            Self::load_cache(&cache_file)?
        } else {
            debug!("No existing cache found, starting fresh");
            HashMap::new()
        };

        info!("✓ Cache initialized with {} entries", cache_data.len());

        Ok(Self {
            cache_dir,
            cache_data: Arc::new(RwLock::new(cache_data)),
            dirty: AtomicBool::new(false),
        })
    }

    fn load_cache(cache_file: &PathBuf) -> Result<HashMap<String, String>> {
        let file = fs::File::open(cache_file)?;
        let file_size = file.metadata()?.len();
        trace!("Reading cache file: {} bytes (compressed)", file_size);

        let mut decoder = GzDecoder::new(file);
        let mut json_str = String::new();
        decoder.read_to_string(&mut json_str)?;
        trace!("Decompressed cache: {} bytes", json_str.len());

        let entries: Vec<CacheEntry> = serde_json::from_str(&json_str)?;
        let mut map = HashMap::new();
        for entry in entries {
            map.insert(entry.checksum, entry.translation);
        }

        debug!("✓ Loaded {} cache entries", map.len());
        Ok(map)
    }

    pub fn save(&self) -> Result<()> {
        if !self.dirty.load(Ordering::Acquire) {
            trace!("Cache not dirty, skipping save");
            return Ok(());
        }

        let cache_data = self.cache_data.read().unwrap();
        debug!("💾 [CACHE] Saving {} entries to disk...", cache_data.len());
        let save_start = std::time::Instant::now();

        let cache_file = self.cache_dir.join("translations.json.gz");
        let entries: Vec<CacheEntry> = cache_data
            .iter()
            .map(|(k, v)| CacheEntry {
                checksum: k.clone(),
                translation: v.clone(),
            })
            .collect();

        let json_str = serde_json::to_string(&entries)?;
        trace!("Serialized cache: {} bytes (uncompressed)", json_str.len());

        let file = fs::File::create(&cache_file)?;
        let mut encoder = GzEncoder::new(file, Compression::best());
        encoder.write_all(json_str.as_bytes())?;
        encoder.finish()?;

        let file_size = fs::metadata(&cache_file)?.len();
        let save_time = save_start.elapsed();
        info!("✅ [CACHE] Saved in {:.2}ms: {} entries, {:.2} KB on disk",
            save_time.as_secs_f64() * 1000.0,
            cache_data.len(),
            file_size as f64 / 1024.0);

        Ok(())
    }

    pub fn compute_checksum(image_bytes: &[u8]) -> String {
        let mut hasher = Sha1::new();
        hasher.update(image_bytes);
        let hash = format!("{:x}", hasher.finalize());
        trace!("Computed SHA1: {} (from {} bytes)", &hash[..16], image_bytes.len());
        hash
    }

    pub fn get(&self, checksum: &str) -> Option<String> {
        let cache_data = self.cache_data.read().unwrap();
        let result = cache_data.get(checksum).cloned();
        if result.is_some() {
            debug!("✓ [CACHE] HIT for checksum: {}...", &checksum[..16]);
        } else {
            debug!("✗ [CACHE] MISS for checksum: {}...", &checksum[..16]);
        }
        result
    }

    pub fn insert(&self, checksum: String, translation: String) {
        debug!("💾 [CACHE] INSERT checksum: {}... ({} bytes)",
            &checksum[..16], translation.len());
        let mut cache_data = self.cache_data.write().unwrap();
        cache_data.insert(checksum, translation);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn stats(&self) -> (usize, f64) {
        let cache_data = self.cache_data.read().unwrap();
        let count = cache_data.len();
        let cache_file = self.cache_dir.join("translations.json.gz");
        let size_mb = if cache_file.exists() {
            cache_file.metadata().map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0)
        } else {
            0.0
        };
        (count, size_mb)
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        let _ = self.save();
    }
}

// Trait implementation for CacheStore
// Note: Methods no longer require &mut self for insert
impl CacheStore for Cache {
    fn get(&self, checksum: &str) -> Option<String> {
        self.get(checksum)
    }

    fn insert(&self, checksum: String, translation: String) {
        // Delegate to the interior mutability version
        Cache::insert(self, checksum, translation)
    }

    fn save(&self) -> Result<()> {
        self.save()
    }

    fn stats(&self) -> (usize, f64) {
        self.stats()
    }

    fn compute_checksum(image_bytes: &[u8]) -> String {
        Self::compute_checksum(image_bytes)
    }
}
