// Smart API key pool with health tracking and automatic failover
//
// Tracks health of each API key and routes requests to healthy keys

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn, info};

/// Health status of an API key
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyHealth {
    Healthy,
    Degraded,
    Unhealthy,
}

/// API key with health tracking
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub key: String,
    pub index: usize,
    consecutive_failures: u32,
    consecutive_successes: u32,
    #[allow(dead_code)]
    last_used: Option<Instant>,
    last_failure: Option<Instant>,
    total_requests: u64,
    total_failures: u64,
}

impl ApiKey {
    fn new(key: String, index: usize) -> Self {
        Self {
            key,
            index,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_used: None,
            last_failure: None,
            total_requests: 0,
            total_failures: 0,
        }
    }

    fn health(&self) -> KeyHealth {
        // Consider unhealthy if 3+ consecutive failures
        if self.consecutive_failures >= 3 {
            return KeyHealth::Unhealthy;
        }

        // Consider degraded if recent failure within last minute
        if let Some(last_failure) = self.last_failure {
            if last_failure.elapsed() < Duration::from_secs(60) && self.consecutive_failures > 0 {
                return KeyHealth::Degraded;
            }
        }

        KeyHealth::Healthy
    }

    fn failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_failures as f64 / self.total_requests as f64
    }

    #[allow(dead_code)]
    fn record_success(&mut self) {
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;
        self.last_used = Some(Instant::now());
        self.total_requests += 1;

        // Reset unhealthy status after sustained success
        if self.consecutive_successes >= 5 {
            if self.last_failure.is_some() {
                info!(
                    "API key {} recovered ({} consecutive successes)",
                    self.index, self.consecutive_successes
                );
            }
            self.last_failure = None;
        }
    }

    #[allow(dead_code)]
    fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
        self.last_failure = Some(Instant::now());
        self.last_used = Some(Instant::now());
        self.total_requests += 1;
        self.total_failures += 1;

        if self.consecutive_failures >= 3 {
            warn!(
                "API key {} marked unhealthy ({} consecutive failures, {:.1}% overall failure rate)",
                self.index,
                self.consecutive_failures,
                self.failure_rate() * 100.0
            );
        }
    }

    fn should_recover(&self) -> bool {
        // Allow recovery attempt if unhealthy for more than 5 minutes
        if let Some(last_failure) = self.last_failure {
            last_failure.elapsed() > Duration::from_secs(300)
        } else {
            true
        }
    }
}

/// API key pool with intelligent routing
pub struct ApiKeyPool {
    keys: Arc<RwLock<Vec<ApiKey>>>,
    round_robin_index: AtomicUsize,
}

impl ApiKeyPool {
    pub fn new(keys: Vec<String>) -> Self {
        let api_keys = keys
            .into_iter()
            .enumerate()
            .map(|(i, key)| ApiKey::new(key, i))
            .collect();

        Self {
            keys: Arc::new(RwLock::new(api_keys)),
            round_robin_index: AtomicUsize::new(0),
        }
    }

    /// Get a healthy API key
    pub async fn get_healthy_key(&self) -> Option<(usize, String)> {
        let keys = self.keys.read().await;

        // First, try to get a healthy key
        let healthy_keys: Vec<_> = keys
            .iter()
            .filter(|k| k.health() == KeyHealth::Healthy)
            .collect();

        if !healthy_keys.is_empty() {
            let index = self.round_robin_index.fetch_add(1, Ordering::Relaxed);
            let key = &healthy_keys[index % healthy_keys.len()];
            debug!("Using healthy API key {} (total: {} healthy)", key.index, healthy_keys.len());
            return Some((key.index, key.key.clone()));
        }

        // If no healthy keys, try degraded
        let degraded_keys: Vec<_> = keys
            .iter()
            .filter(|k| k.health() == KeyHealth::Degraded)
            .collect();

        if !degraded_keys.is_empty() {
            let index = self.round_robin_index.fetch_add(1, Ordering::Relaxed);
            let key = &degraded_keys[index % degraded_keys.len()];
            warn!("No healthy keys, using degraded API key {}", key.index);
            return Some((key.index, key.key.clone()));
        }

        // Last resort: try unhealthy keys that should recover
        let recoverable_keys: Vec<_> = keys
            .iter()
            .filter(|k| k.health() == KeyHealth::Unhealthy && k.should_recover())
            .collect();

        if !recoverable_keys.is_empty() {
            let index = self.round_robin_index.fetch_add(1, Ordering::Relaxed);
            let key = &recoverable_keys[index % recoverable_keys.len()];
            warn!("All keys unhealthy, attempting recovery with key {}", key.index);
            return Some((key.index, key.key.clone()));
        }

        warn!("No API keys available (all unhealthy)");
        None
    }

    /// Record successful use of a key
    #[allow(dead_code)]
    pub async fn record_success(&self, key_index: usize) {
        let mut keys = self.keys.write().await;
        if let Some(key) = keys.get_mut(key_index) {
            key.record_success();
        }
    }

    /// Record failed use of a key
    #[allow(dead_code)]
    pub async fn record_failure(&self, key_index: usize) {
        let mut keys = self.keys.write().await;
        if let Some(key) = keys.get_mut(key_index) {
            key.record_failure();
        }
    }

    /// Get statistics for all keys
    #[allow(dead_code)]
    pub async fn stats(&self) -> Vec<(usize, KeyHealth, u64, u64, f64)> {
        let keys = self.keys.read().await;
        keys.iter()
            .map(|k| {
                (
                    k.index,
                    k.health(),
                    k.total_requests,
                    k.total_failures,
                    k.failure_rate(),
                )
            })
            .collect()
    }

    /// Get count of healthy keys
    #[allow(dead_code)]
    pub async fn healthy_count(&self) -> usize {
        let keys = self.keys.read().await;
        keys.iter().filter(|k| k.health() == KeyHealth::Healthy).count()
    }

    /// Reset all keys to healthy state
    #[allow(dead_code)]
    pub async fn reset_all(&self) {
        let mut keys = self.keys.write().await;
        for key in keys.iter_mut() {
            key.consecutive_failures = 0;
            key.consecutive_successes = 0;
            key.last_failure = None;
        }
        info!("All API keys reset to healthy state");
    }

    /// Get total number of keys
    #[allow(dead_code)]
    pub async fn total_keys(&self) -> usize {
        let keys = self.keys.read().await;
        keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_api_key_pool_round_robin() {
        let keys = vec!["key1".to_string(), "key2".to_string(), "key3".to_string()];
        let pool = ApiKeyPool::new(keys);

        // Get keys in round-robin fashion
        let (idx1, _) = pool.get_healthy_key().await.unwrap();
        pool.record_success(idx1).await;

        let (idx2, _) = pool.get_healthy_key().await.unwrap();
        pool.record_success(idx2).await;

        let (idx3, _) = pool.get_healthy_key().await.unwrap();
        pool.record_success(idx3).await;

        // Should cycle back to first
        let (idx4, _) = pool.get_healthy_key().await.unwrap();
        assert_eq!(idx4, idx1);
    }

    #[tokio::test]
    async fn test_api_key_health_tracking() {
        let keys = vec!["key1".to_string()];
        let pool = ApiKeyPool::new(keys);

        let (idx, _) = pool.get_healthy_key().await.unwrap();

        // Record 3 failures
        for _ in 0..3 {
            pool.record_failure(idx).await;
        }

        // Check stats
        let stats = pool.stats().await;
        assert_eq!(stats[0].1, KeyHealth::Unhealthy);
        assert_eq!(stats[0].3, 3); // 3 failures
    }

    #[tokio::test]
    async fn test_api_key_recovery() {
        let keys = vec!["key1".to_string()];
        let pool = ApiKeyPool::new(keys);

        let (idx, _) = pool.get_healthy_key().await.unwrap();

        // Make it unhealthy
        for _ in 0..3 {
            pool.record_failure(idx).await;
        }

        // Record successes
        for _ in 0..5 {
            pool.record_success(idx).await;
        }

        // Should be healthy again
        let stats = pool.stats().await;
        assert_eq!(stats[0].1, KeyHealth::Healthy);
    }

    #[tokio::test]
    async fn test_no_healthy_keys_fallback() {
        let keys = vec!["key1".to_string(), "key2".to_string()];
        let pool = ApiKeyPool::new(keys);

        // Make all keys unhealthy
        for i in 0..2 {
            for _ in 0..3 {
                pool.record_failure(i).await;
            }
        }

        assert_eq!(pool.healthy_count().await, 0);

        // Should still return a key (for recovery attempt)
        let result = pool.get_healthy_key().await;
        assert!(result.is_some());
    }
}
