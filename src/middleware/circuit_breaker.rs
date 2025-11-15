// Circuit breaker pattern for failing fast on unhealthy services
//
// Prevents cascading failures by short-circuiting calls to failing services

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn, info};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests fail fast
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// How long to wait before attempting recovery
    pub timeout: Duration,
    /// Number of successful calls in half-open state to close circuit
    pub success_threshold: u32,
    /// Sliding window size for failure tracking
    pub window_size: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout: Duration::from_secs(60),
            success_threshold: 2,
            window_size: 10,
        }
    }
}

/// Circuit breaker error
#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum CircuitBreakerError<E> {
    #[error("Circuit breaker is open (service unavailable)")]
    CircuitOpen,

    #[error("Operation failed: {0}")]
    OperationFailed(E),
}

#[allow(dead_code)]
struct CircuitBreakerInner {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<Instant>,
    recent_results: Vec<bool>, // true = success, false = failure
}

/// Circuit breaker implementation
#[allow(dead_code)]
pub struct CircuitBreaker {
    inner: Arc<RwLock<CircuitBreakerInner>>,
    config: CircuitBreakerConfig,
}

#[allow(dead_code)]
impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CircuitBreakerInner {
                state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
                recent_results: Vec::new(),
            })),
            config,
        }
    }

    /// Get current circuit state
    pub async fn state(&self) -> CircuitState {
        self.inner.read().await.state
    }

    /// Get failure statistics
    pub async fn stats(&self) -> (u32, u32, CircuitState) {
        let inner = self.inner.read().await;
        (inner.failure_count, inner.success_count, inner.state)
    }

    /// Reset circuit breaker to closed state
    pub async fn reset(&self) {
        let mut inner = self.inner.write().await;
        inner.state = CircuitState::Closed;
        inner.failure_count = 0;
        inner.success_count = 0;
        inner.last_failure_time = None;
        inner.recent_results.clear();
        info!("Circuit breaker reset to Closed state");
    }

    /// Execute an operation through the circuit breaker
    pub async fn call<F, T, E, Fut>(&self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        // Check if we should attempt the call
        {
            let mut inner = self.inner.write().await;

            match inner.state {
                CircuitState::Open => {
                    // Check if timeout has elapsed
                    if let Some(last_failure) = inner.last_failure_time {
                        if last_failure.elapsed() >= self.config.timeout {
                            debug!("Circuit breaker transitioning from Open to HalfOpen");
                            inner.state = CircuitState::HalfOpen;
                            inner.success_count = 0;
                        } else {
                            return Err(CircuitBreakerError::CircuitOpen);
                        }
                    } else {
                        return Err(CircuitBreakerError::CircuitOpen);
                    }
                }
                CircuitState::HalfOpen => {
                    // Allow limited requests through
                }
                CircuitState::Closed => {
                    // Normal operation
                }
            }
        }

        // Execute the operation
        let result = operation().await;

        // Update circuit breaker state based on result
        let mut inner = self.inner.write().await;

        match &result {
            Ok(_) => {
                // Record success
                self.record_success(&mut inner);
            }
            Err(_) => {
                // Record failure
                self.record_failure(&mut inner);
            }
        }

        result.map_err(CircuitBreakerError::OperationFailed)
    }

    fn record_success(&self, inner: &mut CircuitBreakerInner) {
        inner.recent_results.push(true);
        if inner.recent_results.len() > self.config.window_size {
            inner.recent_results.remove(0);
        }

        match inner.state {
            CircuitState::HalfOpen => {
                inner.success_count += 1;
                if inner.success_count >= self.config.success_threshold {
                    info!(
                        "Circuit breaker transitioning from HalfOpen to Closed ({} consecutive successes)",
                        inner.success_count
                    );
                    inner.state = CircuitState::Closed;
                    inner.failure_count = 0;
                    inner.success_count = 0;
                    inner.last_failure_time = None;
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                if inner.failure_count > 0 {
                    debug!("Resetting failure count after success");
                    inner.failure_count = 0;
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but handle gracefully
            }
        }
    }

    fn record_failure(&self, inner: &mut CircuitBreakerInner) {
        inner.recent_results.push(false);
        if inner.recent_results.len() > self.config.window_size {
            inner.recent_results.remove(0);
        }

        inner.last_failure_time = Some(Instant::now());

        match inner.state {
            CircuitState::Closed => {
                inner.failure_count += 1;
                if inner.failure_count >= self.config.failure_threshold {
                    warn!(
                        "Circuit breaker opening after {} failures",
                        inner.failure_count
                    );
                    inner.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                warn!("Circuit breaker transitioning from HalfOpen back to Open (failure detected)");
                inner.state = CircuitState::Open;
                inner.success_count = 0;
            }
            CircuitState::Open => {
                // Already open, just record the failure
            }
        }
    }

    /// Get failure rate from recent results
    pub async fn failure_rate(&self) -> f64 {
        let inner = self.inner.read().await;
        if inner.recent_results.is_empty() {
            return 0.0;
        }

        let failures = inner.recent_results.iter().filter(|&&x| !x).count();
        failures as f64 / inner.recent_results.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            timeout: Duration::from_secs(60),
            success_threshold: 2,
            window_size: 10,
        };
        let cb = CircuitBreaker::new(config);

        // Record 3 failures
        for _ in 0..3 {
            let result = cb.call(|| async { Err::<(), _>("error") }).await;
            assert!(result.is_err());
        }

        // Circuit should be open now
        assert_eq!(cb.state().await, CircuitState::Open);

        // Next call should fail fast
        let result = cb.call(|| async { Ok::<(), String>(()) }).await;
        assert!(matches!(result, Err(CircuitBreakerError::CircuitOpen)));
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_millis(100),
            success_threshold: 2,
            window_size: 10,
        };
        let cb = CircuitBreaker::new(config);

        // Open the circuit
        for _ in 0..2 {
            let _ = cb.call(|| async { Err::<(), _>("error") }).await;
        }
        assert_eq!(cb.state().await, CircuitState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Next call should transition to half-open
        let _ = cb.call(|| async { Ok::<(), String>(()) }).await;
        assert_eq!(cb.state().await, CircuitState::HalfOpen);

        // One more success should close the circuit
        let _ = cb.call(|| async { Ok::<(), String>(()) }).await;
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_failure_rate() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        // 3 successes, 2 failures
        for _ in 0..3 {
            let _ = cb.call(|| async { Ok::<(), String>(()) }).await;
        }
        for _ in 0..2 {
            let _ = cb.call(|| async { Err::<(), _>("error") }).await;
        }

        let failure_rate = cb.failure_rate().await;
        assert!((failure_rate - 0.4).abs() < 0.01); // 2/5 = 0.4
    }
}
