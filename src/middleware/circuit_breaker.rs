use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests are allowed
    Closed,
    /// Circuit is open, requests are blocked (failing fast)
    Open,
    /// Circuit is half-open, allowing test requests to check if service recovered
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit
    pub failure_threshold: usize,
    /// How long to wait before attempting recovery (half-open state)
    pub timeout: Duration,
    /// Number of consecutive successes in half-open state to close circuit
    pub success_threshold: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout: Duration::from_secs(60),
            success_threshold: 3,
        }
    }
}

/// Circuit breaker for protecting against cascading failures
///
/// Implements the circuit breaker pattern to prevent continuous attempts
/// when a service (like an API) is down, allowing it time to recover.
///
/// States:
/// - Closed: Normal operation, requests pass through
/// - Open: Service is down, fail fast without making requests
/// - Half-Open: Testing if service has recovered
#[derive(Clone)]
pub struct CircuitBreaker {
    inner: Arc<RwLock<CircuitBreakerInner>>,
    config: CircuitBreakerConfig,
}

struct CircuitBreakerInner {
    state: CircuitState,
    consecutive_failures: usize,
    consecutive_successes: usize,
    last_failure_time: Option<Instant>,
    total_failures: usize,
    total_successes: usize,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default configuration
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }

    /// Create a new circuit breaker with custom configuration
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CircuitBreakerInner {
                state: CircuitState::Closed,
                consecutive_failures: 0,
                consecutive_successes: 0,
                last_failure_time: None,
                total_failures: 0,
                total_successes: 0,
            })),
            config,
        }
    }

    /// Check if a request should be allowed
    ///
    /// Returns true if the request can proceed, false if it should fail fast
    pub fn allow_request(&self) -> bool {
        let mut inner = self.inner.write();

        match inner.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout has elapsed
                if let Some(last_failure) = inner.last_failure_time {
                    if last_failure.elapsed() >= self.config.timeout {
                        // Transition to half-open to test if service recovered
                        inner.state = CircuitState::HalfOpen;
                        inner.consecutive_successes = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                // Allow one test request at a time in half-open state
                true
            }
        }
    }

    /// Record a successful request
    pub fn record_success(&self) {
        let mut inner = self.inner.write();
        inner.total_successes += 1;
        inner.consecutive_failures = 0;

        match inner.state {
            CircuitState::Closed => {
                // No state change needed
            }
            CircuitState::HalfOpen => {
                inner.consecutive_successes += 1;
                if inner.consecutive_successes >= self.config.success_threshold {
                    // Service has recovered, close the circuit
                    inner.state = CircuitState::Closed;
                    inner.consecutive_successes = 0;
                }
            }
            CircuitState::Open => {
                // This shouldn't happen, but if it does, transition to half-open
                inner.state = CircuitState::HalfOpen;
                inner.consecutive_successes = 1;
            }
        }
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        let mut inner = self.inner.write();
        inner.total_failures += 1;
        inner.consecutive_successes = 0;
        inner.last_failure_time = Some(Instant::now());

        match inner.state {
            CircuitState::Closed => {
                inner.consecutive_failures += 1;
                if inner.consecutive_failures >= self.config.failure_threshold {
                    // Too many failures, open the circuit
                    inner.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                // Test failed, reopen the circuit
                inner.state = CircuitState::Open;
                inner.consecutive_failures = 1;
            }
            CircuitState::Open => {
                inner.consecutive_failures += 1;
            }
        }
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        self.inner.read().state
    }

    /// Get statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        let inner = self.inner.read();
        CircuitBreakerStats {
            state: inner.state,
            consecutive_failures: inner.consecutive_failures,
            consecutive_successes: inner.consecutive_successes,
            total_failures: inner.total_failures,
            total_successes: inner.total_successes,
        }
    }

    /// Reset the circuit breaker to closed state
    pub fn reset(&self) {
        let mut inner = self.inner.write();
        inner.state = CircuitState::Closed;
        inner.consecutive_failures = 0;
        inner.consecutive_successes = 0;
        inner.last_failure_time = None;
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: CircuitState,
    pub consecutive_failures: usize,
    pub consecutive_successes: usize,
    pub total_failures: usize,
    pub total_successes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed_to_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            timeout: Duration::from_secs(1),
            success_threshold: 2,
        };
        let breaker = CircuitBreaker::with_config(config);

        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.allow_request());

        // Record failures to open the circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Should not allow requests when open
        assert!(!breaker.allow_request());
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_millis(100),
            success_threshold: 2,
        };
        let breaker = CircuitBreaker::with_config(config);

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));

        // Should transition to half-open
        assert!(breaker.allow_request());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Successful requests should close the circuit
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_millis(100),
            success_threshold: 2,
        };
        let breaker = CircuitBreaker::with_config(config);

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));
        assert!(breaker.allow_request());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Failed test should reopen the circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }
}
