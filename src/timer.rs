use std::time::Instant;

/// Returns a monotonic instant for elapsed-time measurements.
/// Unlike SystemTime, this is not affected by NTP adjustments.
#[inline]
pub fn now() -> Instant {
    Instant::now()
}
