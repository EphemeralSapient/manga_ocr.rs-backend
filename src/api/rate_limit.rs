// Rate limiting middleware for API requests

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Maximum requests per window
    pub max_requests: u32,
    /// Time window duration
    pub window: Duration,
    /// Whether to enable rate limiting
    pub enabled: bool,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window: Duration::from_secs(60),
            enabled: false, // Disabled by default
        }
    }
}

impl RateLimiterConfig {
    /// Create from main Config
    pub fn from_config(config: &crate::config::Config) -> Self {
        Self {
            enabled: config.rate_limit.enabled,
            max_requests: config.rate_limit.max_requests,
            window: Duration::from_secs(config.rate_limit.window_seconds),
        }
    }
}

/// Rate limiter state
#[derive(Debug)]
struct RateLimitEntry {
    count: u32,
    window_start: Instant,
}

/// Rate limiter for IP-based rate limiting
pub struct RateLimiter {
    config: RateLimiterConfig,
    state: Arc<RwLock<HashMap<IpAddr, RateLimitEntry>>>,
}

impl RateLimiter {
    pub fn new(config: RateLimiterConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if request is allowed
    pub async fn check(&self, ip: IpAddr) -> bool {
        if !self.config.enabled {
            return true;
        }

        let mut state = self.state.write().await;
        let now = Instant::now();

        let entry = state.entry(ip).or_insert(RateLimitEntry {
            count: 0,
            window_start: now,
        });

        // Check if window has expired
        if now.duration_since(entry.window_start) > self.config.window {
            debug!("Rate limit window expired for {}, resetting", ip);
            entry.count = 0;
            entry.window_start = now;
        }

        // Check if limit exceeded
        if entry.count >= self.config.max_requests {
            warn!("Rate limit exceeded for {} ({}/{})", ip, entry.count, self.config.max_requests);
            return false;
        }

        entry.count += 1;
        debug!("Rate limit check passed for {} ({}/{})", ip, entry.count, self.config.max_requests);
        true
    }

    /// Get current rate limit status for an IP
    pub async fn status(&self, ip: IpAddr) -> (u32, u32) {
        let state = self.state.read().await;
        if let Some(entry) = state.get(&ip) {
            let now = Instant::now();
            if now.duration_since(entry.window_start) <= self.config.window {
                return (entry.count, self.config.max_requests);
            }
        }
        (0, self.config.max_requests)
    }

    /// Clean up expired entries
    #[allow(dead_code)]
    pub async fn cleanup(&self) {
        let mut state = self.state.write().await;
        let now = Instant::now();
        state.retain(|_, entry| {
            now.duration_since(entry.window_start) <= self.config.window
        });
        debug!("Rate limiter cleanup: {} active entries", state.len());
    }
}

/// Extract IP address from request
fn extract_ip(req: &Request) -> Option<IpAddr> {
    // Try X-Forwarded-For header first (for proxies)
    if let Some(forwarded) = req.headers().get("x-forwarded-for") {
        if let Ok(forwarded_str) = forwarded.to_str() {
            if let Some(first_ip) = forwarded_str.split(',').next() {
                if let Ok(ip) = first_ip.trim().parse() {
                    return Some(ip);
                }
            }
        }
    }

    // Fall back to X-Real-IP
    if let Some(real_ip) = req.headers().get("x-real-ip") {
        if let Ok(ip_str) = real_ip.to_str() {
            if let Ok(ip) = ip_str.parse() {
                return Some(ip);
            }
        }
    }

    // TODO: Fall back to connection remote address
    // This requires access to ConnectInfo which needs to be added to the extension

    None
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(limiter): State<Arc<RateLimiter>>,
    req: Request,
    next: Next,
) -> Response {
    let ip = extract_ip(&req).unwrap_or_else(|| "127.0.0.1".parse().unwrap());

    if !limiter.check(ip).await {
        let (current, max) = limiter.status(ip).await;
        return (
            StatusCode::TOO_MANY_REQUESTS,
            format!("Rate limit exceeded: {}/{} requests in window", current, max),
        )
            .into_response();
    }

    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter() {
        let config = RateLimiterConfig {
            max_requests: 3,
            window: Duration::from_secs(60),
            enabled: true,
        };
        let limiter = RateLimiter::new(config);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();

        // First 3 requests should succeed
        assert!(limiter.check(ip).await);
        assert!(limiter.check(ip).await);
        assert!(limiter.check(ip).await);

        // 4th request should fail
        assert!(!limiter.check(ip).await);

        // Status should show limit reached
        let (current, max) = limiter.status(ip).await;
        assert_eq!(current, 3);
        assert_eq!(max, 3);
    }

    #[tokio::test]
    async fn test_rate_limiter_disabled() {
        let config = RateLimiterConfig {
            max_requests: 1,
            window: Duration::from_secs(60),
            enabled: false,
        };
        let limiter = RateLimiter::new(config);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();

        // Should allow unlimited requests when disabled
        for _ in 0..10 {
            assert!(limiter.check(ip).await);
        }
    }
}
