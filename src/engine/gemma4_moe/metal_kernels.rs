//! Gemma 4 specific Metal kernel dispatch helpers.
//!
//! Status: skeleton. Wrappers around the kernels in `shaders.metal`. Each
//! one is documented but not yet wired (kernel implementations are also
//! TODO). The signatures here are committed to and form the contract that
//! the kernels and the engine glue code must match.

#![allow(dead_code)]

use std::ffi::c_void;
use metal::*;

use crate::engine::gemma4_metal_context::Gemma4MetalContext;

unsafe fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let val: *const u32 = &value;
    encoder.set_bytes(index, 4, val as *const c_void);
}

unsafe fn set_f32(encoder: &ComputeCommandEncoderRef, index: u64, value: f32) {
    let val: *const f32 = &value;
    encoder.set_bytes(index, 4, val as *const c_void);
}

/// Sliding-window causal SDPA (single-token query path).
///
/// Equivalent to `attn_sdpa_fused` from qwen35_moe, but K/V positions are
/// restricted to the last `sliding_window` tokens before the query position.
///
/// Status: TODO. Kernel skeleton in shaders.metal is `#if 0`'d; this
/// dispatcher panics. Wire up once the kernel exists.
pub fn encode_attn_sdpa_sliding(
    _ctx: &Gemma4MetalContext,
    _encoder: &ComputeCommandEncoderRef,
    _q: &BufferRef, _q_offset: u64,
    _k_cache: &BufferRef,
    _v_cache: &BufferRef,
    _out: &BufferRef, _o_offset: u64,
    _seq_len: u32,
    _sliding_window: u32,
    _num_q_heads: u32,
    _head_dim: u32,
) {
    unimplemented!("encode_attn_sdpa_sliding — Gemma 4 sliding-window kernel not yet implemented");
}

/// GELU activation with multiplicative gating (Gemma 4's FFN nonlinearity).
/// Computes `out[i] = gelu_pytorch_tanh(gate[i]) * up[i]`.
pub fn encode_gelu_fused(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    gate: &BufferRef, gate_offset: u64,
    up: &BufferRef, up_offset: u64,
    out: &BufferRef, out_offset: u64,
    dim: u32,
) {
    let pipeline = ctx.gelu_fused.as_ref().expect("gelu_fused kernel not loaded");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(gate), gate_offset);
    encoder.set_buffer(1, Some(up), up_offset);
    encoder.set_buffer(2, Some(out), out_offset);
    unsafe { set_u32(encoder, 3, dim); }
    encoder.dispatch_thread_groups(
        MTLSize::new(((dim + 255) / 256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Final-logit softcap: `logits[i] := cap * tanh(logits[i] / cap)`.
/// Applied after lm_head, before sampling. Gemma 4 uses cap = 30.0.
pub fn encode_logit_softcap(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    logits: &BufferRef, logits_offset: u64,
    vocab_size: u32,
    cap: f32,
) {
    let pipeline = ctx.logit_softcap.as_ref().expect("logit_softcap kernel not loaded");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(logits), logits_offset);
    unsafe {
        set_u32(encoder, 1, vocab_size);
        set_f32(encoder, 2, cap);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(((vocab_size + 255) / 256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

// TODO: dispatchers for sliding-attention RoPE variants
// (q_head_norm_rope_gemma_sliding, q_head_norm_rope_gemma_full,
//  k_head_norm_rope_gemma_sliding, k_head_norm_rope_gemma_full).
// Probably reuse the qwen35_moe q_head_norm_rope kernel with different
// parameters (rotary_dim, rope_theta) — depends on whether the partial
// rotary case can be expressed with the existing kernel by passing
// rotary_dim = head_dim/4. If so, no new kernel needed, just new
// dispatcher.
