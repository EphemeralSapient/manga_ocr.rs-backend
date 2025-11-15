// API modules for different versions

pub mod errors;
pub mod handlers;
pub mod v1;
pub mod v2;
pub mod rate_limit;

// Re-export commonly used types
pub use errors::ApiError;
