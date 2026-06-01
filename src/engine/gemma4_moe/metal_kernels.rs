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
/// Online-softmax over positions [max(0, seq_len-sliding_window), seq_len).
/// Uses qwen35's compile-time HEAD_DIM=256 — matches Gemma 4 sliding layers.
/// kv_dim and heads_per_kv are passed as runtime constants because Gemma 4
/// uses 8 kv heads (Qwen3.6 uses 2).
pub fn encode_attn_sdpa_sliding_causal(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    q: &BufferRef, q_offset: u64,
    k_cache: &BufferRef,
    v_cache: &BufferRef,
    out: &BufferRef, o_offset: u64,
    seq_len: u32,
    sliding_window: u32,
    num_q_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    heads_per_kv: u32,
) {
    let pipeline = &ctx.attn_sdpa_sliding_causal;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(q), q_offset);
    encoder.set_buffer(1, Some(k_cache), 0);
    encoder.set_buffer(2, Some(v_cache), 0);
    encoder.set_buffer(3, Some(out), o_offset);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    unsafe {
        set_u32(encoder, 4, seq_len);
        set_u32(encoder, 5, sliding_window);
        set_f32(encoder, 6, scale);
        set_u32(encoder, 7, kv_dim);
        set_u32(encoder, 8, heads_per_kv);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_q_heads as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// RMSNorm without learnable scale — for Gemma 4's v_norm on full layers.
/// Per-head: dispatches num_heads threadgroups; each TG normalises its
/// `head_dim` elements of `x` into `out`.
pub fn encode_rms_norm_no_scale(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, out_offset: u64,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) {
    let pipeline = &ctx.rms_norm_no_scale;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(x), x_offset);
    encoder.set_buffer(1, Some(out), out_offset);
    unsafe {
        set_u32(encoder, 2, head_dim);
        set_f32(encoder, 3, eps);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
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
    let pipeline = &ctx.gelu_fused;
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
    let pipeline = &ctx.logit_softcap;
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
