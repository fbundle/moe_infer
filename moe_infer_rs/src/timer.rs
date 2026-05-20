use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Returns wall-clock time in milliseconds (same behavior as C gettimeofday).
#[inline]
pub fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

/// Simple relative timer using Instant.
#[derive(Debug)]
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Timer { start: Instant::now() }
    }

    /// Elapsed in milliseconds.
    #[inline]
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

impl Default for Timer {
    fn default() -> Self { Self::new() }
}

/// Timing accumulator for per-phase measurements (per-inference-engine).
#[derive(Debug, Default, Clone)]
pub struct TimingAccum {
    pub deferred_wait: f64,
    pub deferred_cpu: f64,
    pub input_norm: f64,
    pub cmd1_submit: f64,
    pub cmd1_wait: f64,
    pub cpu_attn: f64,
    pub cmd2_encode: f64,
    pub cmd2_wait: f64,
    pub routing_cpu: f64,
    pub expert_io: f64,
    pub cmd3_encode: f64,
    pub total: f64,
    pub count: usize,
}

impl TimingAccum {
    pub fn reset(&mut self) {
        *self = TimingAccum::default();
    }

    pub fn print(&self) {
        if self.count == 0 { return; }
        let n = self.count as f64;
        eprintln!("\n[timing] Per-layer breakdown (avg of {} layers, ms):", self.count);
        eprintln!("  deferred_wait:  {:6.3}", self.deferred_wait / n);
        eprintln!("  deferred_cpu:   {:6.3}", self.deferred_cpu / n);
        eprintln!("  input_norm:     {:6.3}", self.input_norm / n);
        eprintln!("  cmd1_submit:    {:6.3}", self.cmd1_submit / n);
        eprintln!("  cmd1_wait:      {:6.3}", self.cmd1_wait / n);
        eprintln!("  cpu_attn:       {:6.3}", self.cpu_attn / n);
        eprintln!("  cmd2_encode:    {:6.3}", self.cmd2_encode / n);
        eprintln!("  cmd2_wait:      {:6.3}", self.cmd2_wait / n);
        eprintln!("  routing_cpu:    {:6.3}", self.routing_cpu / n);
        eprintln!("  expert_io:      {:6.3}", self.expert_io / n);
        eprintln!("  cmd3_encode:    {:6.3}", self.cmd3_encode / n);
        eprintln!("  total_layer:    {:6.3}", self.total / n);
    }
}
