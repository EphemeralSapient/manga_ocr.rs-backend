// Retry logic with exponential backoff
//
// Provides configurable retry policies for transient failures

use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Retry configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Whether to add jitter to prevent thundering herd
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

#[allow(dead_code)]
impl RetryConfig {
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            ..Default::default()
        }
    }

    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 1.5,
            jitter: true,
        }
    }

    pub fn conservative() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 3.0,
            jitter: true,
        }
    }
}

/// Retry policy trait for determining if an error is retryable
#[allow(dead_code)]
pub trait RetryPolicy<E> {
    fn should_retry(&self, error: &E, attempt: u32) -> bool;
}

/// Default retry policy that retries all errors
#[allow(dead_code)]
pub struct AlwaysRetry;

impl<E> RetryPolicy<E> for AlwaysRetry {
    fn should_retry(&self, _error: &E, _attempt: u32) -> bool {
        true
    }
}

/// Retry errors based on HTTP status codes
#[allow(dead_code)]
pub struct HttpRetryPolicy;

impl RetryPolicy<reqwest::Error> for HttpRetryPolicy {
    fn should_retry(&self, error: &reqwest::Error, _attempt: u32) -> bool {
        if let Some(status) = error.status() {
            // Retry on server errors (5xx) and rate limiting (429)
            status.is_server_error() || status.as_u16() == 429
        } else {
            // Retry on network errors (connection, timeout)
            error.is_timeout() || error.is_connect()
        }
    }
}

/// Execute an async operation with retry logic
#[allow(dead_code)]
pub async fn with_retry<F, T, E, Fut, P>(
    mut operation: F,
    config: &RetryConfig,
    policy: &P,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    P: RetryPolicy<E>,
    E: std::fmt::Display,
{
    let mut delay = config.initial_delay;

    for attempt in 1..=config.max_attempts {
        match operation().await {
            Ok(result) => {
                if attempt > 1 {
                    debug!("Operation succeeded on attempt {}/{}", attempt, config.max_attempts);
                }
                return Ok(result);
            }
            Err(e) => {
                if attempt >= config.max_attempts {
                    warn!(
                        "Operation failed after {} attempts: {}",
                        config.max_attempts, e
                    );
                    return Err(e);
                }

                if !policy.should_retry(&e, attempt) {
                    debug!("Error not retryable: {}", e);
                    return Err(e);
                }

                warn!(
                    "Operation failed (attempt {}/{}), retrying after {:?}: {}",
                    attempt, config.max_attempts, delay, e
                );

                // Add jitter if enabled (±20% random variance)
                let actual_delay = if config.jitter {
                    let jitter_factor = 0.8 + (rand::random::<f64>() * 0.4); // 0.8 to 1.2
                    Duration::from_secs_f64(delay.as_secs_f64() * jitter_factor)
                } else {
                    delay
                };

                sleep(actual_delay).await;

                // Exponential backoff
                delay = Duration::from_secs_f64(
                    (delay.as_secs_f64() * config.backoff_multiplier).min(config.max_delay.as_secs_f64())
                );
            }
        }
    }

    unreachable!()
}

/// Retry decorator for wrapping operations
#[allow(dead_code)]
pub struct RetryDecorator<F, P> {
    operation: F,
    config: RetryConfig,
    policy: P,
}

#[allow(dead_code)]
impl<F, P> RetryDecorator<F, P> {
    pub fn new(operation: F, config: RetryConfig, policy: P) -> Self {
        Self {
            operation,
            config,
            policy,
        }
    }
}

#[allow(dead_code)]
impl<F, T, E, Fut, P> RetryDecorator<F, P>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    P: RetryPolicy<E>,
    E: std::fmt::Display,
{
    pub async fn execute(&mut self) -> Result<T, E> {
        with_retry(&mut self.operation, &self.config, &self.policy).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retry_success_first_attempt() {
        let config = RetryConfig::new(3);
        let mut attempts = 0;

        let result = with_retry(
            || async {
                attempts += 1;
                Ok::<_, String>(42)
            },
            &config,
            &AlwaysRetry,
        )
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts, 1);
    }

    #[tokio::test]
    async fn test_retry_success_second_attempt() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            jitter: false,
        };
        let mut attempts = 0;

        let result = with_retry(
            || async {
                attempts += 1;
                if attempts < 2 {
                    Err("transient error".to_string())
                } else {
                    Ok(42)
                }
            },
            &config,
            &AlwaysRetry,
        )
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts, 2);
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            jitter: false,
        };
        let mut attempts = 0;

        let result = with_retry(
            || async {
                attempts += 1;
                Err::<i32, _>("persistent error".to_string())
            },
            &config,
            &AlwaysRetry,
        )
        .await;

        assert!(result.is_err());
        assert_eq!(attempts, 3);
    }
}
