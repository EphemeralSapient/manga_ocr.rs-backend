use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Global metrics collector for the application.
///
/// Tracks API usage, cache performance, phase durations, and more.
/// Thread-safe and can be shared across the application.
#[derive(Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    // API Metrics
    api_calls_total: AtomicUsize,
    api_calls_success: AtomicUsize,
    api_calls_failed: AtomicUsize,
    api_tokens_input: AtomicU64,
    api_tokens_output: AtomicU64,
    api_latency_ms: RwLock<Vec<u64>>,

    // Cache Metrics
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    cache_size: AtomicUsize,

    // Phase Metrics
    phase1_duration_ms: RwLock<Vec<u64>>,
    phase2_duration_ms: RwLock<Vec<u64>>,
    phase3_duration_ms: RwLock<Vec<u64>>,
    phase4_duration_ms: RwLock<Vec<u64>>,

    // Batch Metrics
    batches_processed: AtomicUsize,
    images_processed: AtomicUsize,

    // Per-endpoint request counters
    endpoint_counters: DashMap<String, AtomicUsize>,

    // Circuit breaker state tracking
    circuit_breaker_trips: AtomicUsize,

    // Start time for uptime calculation
    start_time: Instant,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                api_calls_total: AtomicUsize::new(0),
                api_calls_success: AtomicUsize::new(0),
                api_calls_failed: AtomicUsize::new(0),
                api_tokens_input: AtomicU64::new(0),
                api_tokens_output: AtomicU64::new(0),
                api_latency_ms: RwLock::new(Vec::new()),
                cache_hits: AtomicUsize::new(0),
                cache_misses: AtomicUsize::new(0),
                cache_size: AtomicUsize::new(0),
                phase1_duration_ms: RwLock::new(Vec::new()),
                phase2_duration_ms: RwLock::new(Vec::new()),
                phase3_duration_ms: RwLock::new(Vec::new()),
                phase4_duration_ms: RwLock::new(Vec::new()),
                batches_processed: AtomicUsize::new(0),
                images_processed: AtomicUsize::new(0),
                endpoint_counters: DashMap::new(),
                circuit_breaker_trips: AtomicUsize::new(0),
                start_time: Instant::now(),
            }),
        }
    }

    // API Metrics
    pub fn record_api_call(&self, success: bool, duration: Duration, input_tokens: u64, output_tokens: u64) {
        self.inner.api_calls_total.fetch_add(1, Ordering::Relaxed);
        if success {
            self.inner.api_calls_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.inner.api_calls_failed.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.api_tokens_input.fetch_add(input_tokens, Ordering::Relaxed);
        self.inner.api_tokens_output.fetch_add(output_tokens, Ordering::Relaxed);
        self.inner.api_latency_ms.write().push(duration.as_millis() as u64);
    }

    // Cache Metrics
    pub fn record_cache_hit(&self) {
        self.inner.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self) {
        self.inner.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_cache_size(&self, size: usize) {
        self.inner.cache_size.store(size, Ordering::Relaxed);
    }

    // Phase Metrics
    pub fn record_phase1_duration(&self, duration: Duration) {
        self.inner.phase1_duration_ms.write().push(duration.as_millis() as u64);
    }

    pub fn record_phase2_duration(&self, duration: Duration) {
        self.inner.phase2_duration_ms.write().push(duration.as_millis() as u64);
    }

    pub fn record_phase3_duration(&self, duration: Duration) {
        self.inner.phase3_duration_ms.write().push(duration.as_millis() as u64);
    }

    pub fn record_phase4_duration(&self, duration: Duration) {
        self.inner.phase4_duration_ms.write().push(duration.as_millis() as u64);
    }

    // Batch Metrics
    pub fn record_batch_processed(&self, num_images: usize) {
        self.inner.batches_processed.fetch_add(1, Ordering::Relaxed);
        self.inner.images_processed.fetch_add(num_images, Ordering::Relaxed);
    }

    // Endpoint Metrics
    pub fn record_endpoint_request(&self, endpoint: &str) {
        self.inner.endpoint_counters
            .entry(endpoint.to_string())
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    // Circuit Breaker Metrics
    pub fn record_circuit_breaker_trip(&self) {
        self.inner.circuit_breaker_trips.fetch_add(1, Ordering::Relaxed);
    }

    // Get snapshot for reporting
    pub fn snapshot(&self) -> MetricsSnapshot {
        let api_latency = self.inner.api_latency_ms.read();
        let api_latency_avg = if !api_latency.is_empty() {
            api_latency.iter().sum::<u64>() / api_latency.len() as u64
        } else {
            0
        };
        let api_latency_p50 = percentile(&api_latency, 0.5);
        let api_latency_p95 = percentile(&api_latency, 0.95);
        let api_latency_p99 = percentile(&api_latency, 0.99);
        drop(api_latency);

        let phase1_durations = self.inner.phase1_duration_ms.read();
        let phase1_avg = avg(&phase1_durations);
        drop(phase1_durations);

        let phase2_durations = self.inner.phase2_duration_ms.read();
        let phase2_avg = avg(&phase2_durations);
        drop(phase2_durations);

        let phase3_durations = self.inner.phase3_duration_ms.read();
        let phase3_avg = avg(&phase3_durations);
        drop(phase3_durations);

        let phase4_durations = self.inner.phase4_duration_ms.read();
        let phase4_avg = avg(&phase4_durations);
        drop(phase4_durations);

        let cache_hits = self.inner.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.inner.cache_misses.load(Ordering::Relaxed);
        let cache_total = cache_hits + cache_misses;
        let cache_hit_rate = if cache_total > 0 {
            cache_hits as f64 / cache_total as f64
        } else {
            0.0
        };

        MetricsSnapshot {
            api_calls_total: self.inner.api_calls_total.load(Ordering::Relaxed),
            api_calls_success: self.inner.api_calls_success.load(Ordering::Relaxed),
            api_calls_failed: self.inner.api_calls_failed.load(Ordering::Relaxed),
            api_tokens_input: self.inner.api_tokens_input.load(Ordering::Relaxed),
            api_tokens_output: self.inner.api_tokens_output.load(Ordering::Relaxed),
            api_latency_avg_ms: api_latency_avg,
            api_latency_p50_ms: api_latency_p50,
            api_latency_p95_ms: api_latency_p95,
            api_latency_p99_ms: api_latency_p99,
            cache_hits,
            cache_misses,
            cache_hit_rate,
            cache_size: self.inner.cache_size.load(Ordering::Relaxed),
            phase1_avg_ms: phase1_avg,
            phase2_avg_ms: phase2_avg,
            phase3_avg_ms: phase3_avg,
            phase4_avg_ms: phase4_avg,
            batches_processed: self.inner.batches_processed.load(Ordering::Relaxed),
            images_processed: self.inner.images_processed.load(Ordering::Relaxed),
            circuit_breaker_trips: self.inner.circuit_breaker_trips.load(Ordering::Relaxed),
            uptime_seconds: self.inner.start_time.elapsed().as_secs(),
        }
    }

    /// Generate Prometheus-format metrics
    pub fn to_prometheus(&self) -> String {
        let snapshot = self.snapshot();
        format!(
            r#"# HELP api_calls_total Total number of API calls made
# TYPE api_calls_total counter
api_calls_total {{}} {}

# HELP api_calls_success Number of successful API calls
# TYPE api_calls_success counter
api_calls_success {{}} {}

# HELP api_calls_failed Number of failed API calls
# TYPE api_calls_failed counter
api_calls_failed {{}} {}

# HELP api_tokens_input_total Total input tokens consumed
# TYPE api_tokens_input_total counter
api_tokens_input_total {{}} {}

# HELP api_tokens_output_total Total output tokens generated
# TYPE api_tokens_output_total counter
api_tokens_output_total {{}} {}

# HELP api_latency_avg_ms Average API latency in milliseconds
# TYPE api_latency_avg_ms gauge
api_latency_avg_ms {{}} {}

# HELP cache_hit_rate Cache hit rate (0.0 to 1.0)
# TYPE cache_hit_rate gauge
cache_hit_rate {{}} {}

# HELP cache_size Current cache size
# TYPE cache_size gauge
cache_size {{}} {}

# HELP phase_avg_duration_ms Average phase duration in milliseconds
# TYPE phase_avg_duration_ms gauge
phase_avg_duration_ms {{phase="1"}} {}
phase_avg_duration_ms {{phase="2"}} {}
phase_avg_duration_ms {{phase="3"}} {}
phase_avg_duration_ms {{phase="4"}} {}

# HELP batches_processed_total Total number of batches processed
# TYPE batches_processed_total counter
batches_processed_total {{}} {}

# HELP images_processed_total Total number of images processed
# TYPE images_processed_total counter
images_processed_total {{}} {}

# HELP circuit_breaker_trips_total Total circuit breaker trips
# TYPE circuit_breaker_trips_total counter
circuit_breaker_trips_total {{}} {}

# HELP uptime_seconds Application uptime in seconds
# TYPE uptime_seconds counter
uptime_seconds {{}} {}
"#,
            snapshot.api_calls_total,
            snapshot.api_calls_success,
            snapshot.api_calls_failed,
            snapshot.api_tokens_input,
            snapshot.api_tokens_output,
            snapshot.api_latency_avg_ms,
            snapshot.cache_hit_rate,
            snapshot.cache_size,
            snapshot.phase1_avg_ms,
            snapshot.phase2_avg_ms,
            snapshot.phase3_avg_ms,
            snapshot.phase4_avg_ms,
            snapshot.batches_processed,
            snapshot.images_processed,
            snapshot.circuit_breaker_trips,
            snapshot.uptime_seconds,
        )
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub api_calls_total: usize,
    pub api_calls_success: usize,
    pub api_calls_failed: usize,
    pub api_tokens_input: u64,
    pub api_tokens_output: u64,
    pub api_latency_avg_ms: u64,
    pub api_latency_p50_ms: u64,
    pub api_latency_p95_ms: u64,
    pub api_latency_p99_ms: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_hit_rate: f64,
    pub cache_size: usize,
    pub phase1_avg_ms: u64,
    pub phase2_avg_ms: u64,
    pub phase3_avg_ms: u64,
    pub phase4_avg_ms: u64,
    pub batches_processed: usize,
    pub images_processed: usize,
    pub circuit_breaker_trips: usize,
    pub uptime_seconds: u64,
}

fn percentile(values: &[u64], p: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((values.len() as f64 - 1.0) * p) as usize;
    sorted[idx]
}

fn avg(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.iter().sum::<u64>() / values.len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = Metrics::new();

        metrics.record_api_call(true, Duration::from_millis(100), 500, 200);
        metrics.record_api_call(false, Duration::from_millis(50), 0, 0);
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        metrics.record_batch_processed(10);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.api_calls_total, 2);
        assert_eq!(snapshot.api_calls_success, 1);
        assert_eq!(snapshot.api_calls_failed, 1);
        assert_eq!(snapshot.api_tokens_input, 500);
        assert_eq!(snapshot.api_tokens_output, 200);
        assert_eq!(snapshot.cache_hits, 1);
        assert_eq!(snapshot.cache_misses, 1);
        assert_eq!(snapshot.cache_hit_rate, 0.5);
        assert_eq!(snapshot.batches_processed, 1);
        assert_eq!(snapshot.images_processed, 10);
    }

    #[test]
    fn test_prometheus_format() {
        let metrics = Metrics::new();
        metrics.record_api_call(true, Duration::from_millis(100), 500, 200);

        let prometheus = metrics.to_prometheus();
        assert!(prometheus.contains("api_calls_total {} 1"));
        assert!(prometheus.contains("api_tokens_input_total {} 500"));
    }
}
