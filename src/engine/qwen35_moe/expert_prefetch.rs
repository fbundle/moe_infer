//! Speculative expert prefetch — shared infrastructure used by FusedExp6.
//!
//! Design
//! ------
//! After op2 of layer L is encoded (but before its command buffer commits and
//! the GPU starts running it), the main thread spawns rayon tasks that pread
//! experts L+1 will probably need into a dedicated pool of `Buffer`s. The
//! prediction is simply "L+1 routes to the same expert indices as L," which
//! has high empirical hit rate in MoE because the residual-stream input to
//! adjacent layers is highly correlated.
//!
//! The prefetch runs in parallel with the GPU's op1[L+1] + op2[L] command
//! buffer. By the time `route_experts(L+1)` runs on the CPU, hits in the
//! prefetch pool are zero-latency.
//!
//! The pool is intentionally TINY (one slot per top-k = 8 slots, ~13 MB on
//! Qwen3.6) so it doesn't fight the OS page cache for the mmap'd expert
//! file the way the existing LRU does.

use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use metal::{Buffer, Device};

use crate::engine::metal_context::metal_buf_shared;
use crate::model::ExpertFileType;

/// One pre-allocated GPU buffer + its current contents key.
///
/// `key` encodes BOTH the layer and the expert index:
///   key < 0  : slot empty / busy being filled (don't read)
///   key >= 0 : `(layer << 32) | expert_idx`, contents are valid
///
/// We do not need a separate "ready" flag — the writer flips `key` from -1
/// to the valid value *after* the pread completes (with a Release store), so
/// any reader that sees the valid key (with an Acquire load) is guaranteed
/// to see the fully-written buffer.
pub struct PrefetchSlot {
    pub buffer: Buffer,
    pub key: Arc<AtomicI64>,
}

pub struct PrefetchPool {
    slots: Vec<PrefetchSlot>,
    expert_size: usize,
}

impl PrefetchPool {
    pub fn new(device: &Device, num_slots: usize, expert_size: usize) -> Self {
        let slots = (0..num_slots)
            .map(|_| PrefetchSlot {
                buffer: metal_buf_shared(device, expert_size),
                key: Arc::new(AtomicI64::new(-1)),
            })
            .collect();
        eprintln!(
            "[prefetch] {} slots × {:.1} MB = {:.1} MB pool",
            num_slots,
            expert_size as f64 / (1024.0 * 1024.0),
            (num_slots * expert_size) as f64 / (1024.0 * 1024.0),
        );
        PrefetchPool { slots, expert_size }
    }

    /// Look up whether `(layer, expert_idx)` is in the pool with valid data.
    /// Returns a clone of the slot's buffer if found.
    pub fn lookup(&self, layer: usize, expert_idx: usize) -> Option<Buffer> {
        let target = ((layer as i64) << 32) | (expert_idx as i64);
        for slot in &self.slots {
            if slot.key.load(Ordering::Acquire) == target {
                return Some(slot.buffer.clone());
            }
        }
        None
    }

    /// Stop accepting future hits on these slots — call before reassigning
    /// to a new layer to ensure stale (layer, idx) entries no longer match.
    /// (Safe to call while previous prefetches are still in flight; their
    /// completed writes will land on the *new* key set by `schedule`.)
    pub fn invalidate_all(&self) {
        for slot in &self.slots {
            slot.key.store(-1, Ordering::Release);
        }
    }

    /// Spawn rayon tasks to fill the pool with `indices` for `layer`.
    /// Mutates slot keys to -1 (busy) before scheduling; each background
    /// task flips its slot's key to the valid value when its pread is done.
    ///
    /// `model` is held by an Arc so each background task can borrow the
    /// expert file without lifetime gymnastics. The fd lives as long as
    /// the Model does.
    pub fn schedule(
        &self,
        model: Arc<crate::model::Model>,
        layer: usize,
        indices: &[usize],
    ) {
        // Mark all slots busy first so concurrent lookups don't read stale entries.
        self.invalidate_all();

        let n = indices.len().min(self.slots.len());
        for (slot_i, &expert_idx) in indices.iter().take(n).enumerate() {
            let key_handle = self.slots[slot_i].key.clone();
            // Buffer is Clone (it's an Objective-C object handle); the
            // underlying GPU-shared memory is safe to write from a worker
            // thread because we hold the only writer (no other task is
            // touching this slot until the next schedule() call).
            let buf = self.slots[slot_i].buffer.clone();
            let model_for_bg = model.clone();
            let expert_size = self.expert_size;
            let layer_for_bg = layer;
            rayon::spawn(move || {
                if layer_for_bg >= model_for_bg.expert_files.len() {
                    return;
                }
                let ef: &ExpertFileType = &model_for_bg.expert_files[layer_for_bg];
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        buf.contents() as *mut u8,
                        expert_size,
                    )
                };
                if let Err(e) = ef.read_expert(expert_idx, dst) {
                    eprintln!("[prefetch] L{} E{} pread failed: {:?}", layer_for_bg, expert_idx, e);
                    return;
                }
                let key = ((layer_for_bg as i64) << 32) | (expert_idx as i64);
                key_handle.store(key, Ordering::Release);
            });
        }
    }
}

// Buffer is a Metal handle (Objective-C object). The metal crate already
// implements Send for it; we just need this for the Arc holding the pool to
// be Send + Sync across worker threads.
unsafe impl Send for PrefetchPool {}
unsafe impl Sync for PrefetchPool {}
