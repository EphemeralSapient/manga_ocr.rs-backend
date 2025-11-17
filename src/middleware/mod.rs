// Middleware for resilient service calls
//
// Provides API key health tracking, circuit breaking, and rate limiting

pub mod api_key_pool;
pub mod circuit_breaker;
// pub mod rate_limiter;  // TODO: Implement next

// Re-export commonly used types
pub use api_key_pool::ApiKeyPool;
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
