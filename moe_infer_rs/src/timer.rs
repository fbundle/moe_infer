use std::time::{SystemTime, UNIX_EPOCH};

/// Returns wall-clock time in milliseconds (same behavior as C gettimeofday).
#[inline]
pub fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}
