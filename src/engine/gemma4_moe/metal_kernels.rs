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

// ---------------------------------------------------------------------------
// Shared-kernel dispatchers — Qwen3.6 kernels reachable via Gemma4MetalContext.
// These mirror qwen35_moe::metal_kernels::encode_* but take our context type
// because qwen's dispatchers expect a different MetalContext struct.
// ---------------------------------------------------------------------------

use crate::constants::{ROWS_PER_TG, TG_SIZE, RMS_NORM_EPS};

/// Encode BF16 matvec: out[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim].
pub fn encode_matvec_bf16(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_bf16: &BufferRef, w_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.matvec_bf16);
    encoder.set_buffer(0, Some(w_bf16), w_offset);
    encoder.set_buffer(1, Some(x), x_offset);
    encoder.set_buffer(2, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 3, out_dim);
        set_u32(encoder, 4, in_dim);
    }
    let num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
}

/// Encode fused RMSNorm with per-channel BF16 weight.
/// `out[i] = (x[i] * inv_rms) * bf16_to_f32(weight[i])`.
pub fn encode_rms_norm_fused_bf16(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    x: &BufferRef, x_offset: u64,
    weight: &BufferRef, w_offset: u64,
    out: &BufferRef, o_offset: u64,
    dim: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.rms_norm_fused_bf16);
    encoder.set_buffer(0, Some(x), x_offset);
    encoder.set_buffer(1, Some(weight), w_offset);
    encoder.set_buffer(2, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 3, dim);
        set_f32(encoder, 4, RMS_NORM_EPS);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Encode element-wise residual add: `out[i] = a[i] + b[i]`.
pub fn encode_residual_add(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    a: &BufferRef, a_offset: u64,
    b: &BufferRef, b_offset: u64,
    out: &BufferRef, o_offset: u64,
    dim: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.residual_add);
    encoder.set_buffer(0, Some(a), a_offset);
    encoder.set_buffer(1, Some(b), b_offset);
    encoder.set_buffer(2, Some(out), o_offset);
    unsafe { set_u32(encoder, 3, dim); }
    encoder.dispatch_thread_groups(
        MTLSize::new(((dim + 255) / 256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Q head-norm + RoPE — Gemma 4 variant without query-gate split.
/// `rotary_dim` is the number of head dims that get rotated: head_dim
/// for sliding layers (full rotary), head_dim/4 for full layers
/// (partial_rotary_factor=0.25).
pub fn encode_q_head_norm_rope_no_gate(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    q_proj: &BufferRef, q_offset: u64,
    q_norm_w: &BufferRef, w_offset: u64,
    q_out: &BufferRef, o_offset: u64,
    num_q_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    rope_theta: f32,
    pos: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.q_head_norm_rope_no_gate);
    encoder.set_buffer(0, Some(q_proj), q_offset);
    encoder.set_buffer(1, Some(q_norm_w), w_offset);
    encoder.set_buffer(2, Some(q_out), o_offset);
    unsafe {
        set_u32(encoder, 3, head_dim);
        set_u32(encoder, 4, rotary_dim);
        set_f32(encoder, 5, rope_theta);
        set_u32(encoder, 6, pos);
        set_f32(encoder, 7, RMS_NORM_EPS);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_q_heads as u64, 1, 1),
        MTLSize::new(head_dim as u64, 1, 1),
    );
}

/// K head-norm + RoPE — qwen35's k_head_norm_rope is byte-compatible.
/// In-place over `k_buf`.
pub fn encode_k_head_norm_rope(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    k_buf: &BufferRef, k_offset: u64,
    k_norm_w: &BufferRef, w_offset: u64,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    rope_theta: f32,
    pos: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.k_head_norm_rope);
    encoder.set_buffer(0, Some(k_buf), k_offset);
    encoder.set_buffer(1, Some(k_norm_w), w_offset);
    unsafe {
        set_u32(encoder, 2, head_dim);
        set_u32(encoder, 3, rotary_dim);
        set_f32(encoder, 4, rope_theta);
        set_u32(encoder, 5, pos);
        set_f32(encoder, 6, RMS_NORM_EPS);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_kv_heads as u64, 1, 1),
        MTLSize::new(head_dim as u64, 1, 1),
    );
}

/// KV-cache append — copies k and v at `pos` into the layer's caches.
pub fn encode_kv_cache_append(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    k: &BufferRef,
    v: &BufferRef,
    k_cache: &BufferRef,
    v_cache: &BufferRef,
    pos: u32,
    kv_dim: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.kv_cache_append);
    encoder.set_buffer(0, Some(k), 0);
    encoder.set_buffer(1, Some(v), 0);
    encoder.set_buffer(2, Some(k_cache), 0);
    encoder.set_buffer(3, Some(v_cache), 0);
    unsafe {
        set_u32(encoder, 4, pos);
        set_u32(encoder, 5, kv_dim);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(((kv_dim + 255) / 256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Per-layer scalar multiply (Gemma 4's `layer_scalar`).
pub fn encode_mul_scalar_bf16(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    buf: &BufferRef, buf_offset: u64,
    scalar_bf16: &BufferRef, s_offset: u64,
    dim: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.mul_scalar_bf16);
    encoder.set_buffer(0, Some(buf), buf_offset);
    encoder.set_buffer(1, Some(scalar_bf16), s_offset);
    unsafe { set_u32(encoder, 2, dim); }
    encoder.dispatch_thread_groups(
        MTLSize::new(((dim + 255) / 256) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Router RMSNorm — weight is a single bf16 scalar `router.scale`; the
/// effective per-channel scale is `router.scale * sqrt(hidden)^-1`.
pub fn encode_rms_norm_router(
    ctx: &Gemma4MetalContext,
    encoder: &ComputeCommandEncoderRef,
    x: &BufferRef, x_offset: u64,
    scale_bf16: &BufferRef, s_offset: u64,
    out: &BufferRef, o_offset: u64,
    dim: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.rms_norm_router);
    encoder.set_buffer(0, Some(x), x_offset);
    encoder.set_buffer(1, Some(scale_bf16), s_offset);
    encoder.set_buffer(2, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 3, dim);
        set_f32(encoder, 4, RMS_NORM_EPS);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}
