//! Gemma 4 MetalContext — pipeline state + buffer allocation.
//!
//! Status: skeleton. The full struct will end up similar in shape to
//! `engine/qwen35_moe/metal_context.rs` but with different pipelines
//! (sliding-attention, GELU, logit softcap, dual RoPE) and without the
//! DeltaNet state buffers.
//!
//! Reuses for now:
//!   - Shared matvec kernels from qwen35_moe/shaders.metal
//!   - `metal_buf_shared` allocator
//!   - `MAX_K` constant
//!   - Expert LRU cache (`ExpertCache`)

#![allow(dead_code)]

use metal::*;

use crate::error::MoEError;
use crate::engine::metal_context::{ExpertCache, ExpertBuffer};

/// Holds per-engine Metal device + queue + pipeline states + persistent buffers
/// for one Gemma 4 model. Equivalent to qwen35_moe::MetalContext but for the
/// Gemma 4 kernel surface.
///
/// TODO: enumerate pipeline state fields once the kernel set in
/// shaders.metal is finalised. For now this is intentionally empty so the
/// engine.rs skeleton can compile.
pub struct Gemma4MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub library: Library,

    // ── Pipelines (TODO: fill in once kernels exist) ──────────────────
    pub gelu_fused: Option<ComputePipelineState>,
    pub logit_softcap: Option<ComputePipelineState>,

    // ── Persistent buffers (TODO: design per-layer KV cache for sliding
    // window vs full attention; the two need different shapes) ───────
    // pub kv_full: Vec<Buffer>,    // full-attn layers: [max_seq, kv_dim]
    // pub kv_sliding: Vec<Buffer>, // sliding layers:    [sliding_window, kv_dim]
    //   (sliding only needs to keep `sliding_window` positions in a ring buffer)

    // Expert MoE infrastructure — reusable from qwen35_moe.
    pub expert_buffer: ExpertBuffer,
    pub expert_cache: Option<ExpertCache>,
}

impl Gemma4MetalContext {
    pub fn new<C: super::constants::Gemma4ModelConfig>(
        _wf: &crate::model::weights::WeightFile,
        _num_active_experts: usize,
        _label: &str,
        _expert_cache_count: usize,
    ) -> Result<Self, MoEError> {
        Err(MoEError::Config("Gemma4MetalContext::new is unimplemented — engine port in progress".into()))
    }
}

// Re-export shared helpers so the rest of gemma4_moe doesn't import from qwen35_moe directly.
pub use crate::engine::metal_context::WeightBuffer;
