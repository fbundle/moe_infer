/// GPU kernel dispatch wrappers.
///
/// Port of metal_dequant_matvec, metal_swiglu, encode_matvec_v3, etc. from main.m.
use metal::*;
use std::ffi::c_void;

use crate::constants::{ROWS_PER_TG, TG_SIZE};
use crate::engine::metal_context::MetalContext;

/// Helper: set a u32 constant buffer value at the given index.
unsafe fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let val: *const u32 = &value;
    encoder.set_bytes(index, 4, val as *const c_void);
}

/// Helper: set an f32 constant buffer value at the given index.
unsafe fn set_f32(encoder: &ComputeCommandEncoderRef, index: u64, value: f32) {
    let val: *const f32 = &value;
    encoder.set_bytes(index, 4, val as *const c_void);
}

// ---------------------------------------------------------------------------
// Dequant matvec (v3 / fast / naive)
// ---------------------------------------------------------------------------

/// Encode a dequant matvec with buffer offsets (called within an existing encoder).
pub fn encode_matvec_offset(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_packed: &BufferRef, w_offset: u64,
    scales: &BufferRef, s_offset: u64,
    biases: &BufferRef, b_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    use_fast: i32,
) {
    // Variant picker for INT4 matvecs. Default: `v4_nr4` (llama.cpp-inspired
    // NR0=4 row sharing) — best decode throughput on Apple Silicon across
    // all model sizes we've measured (Qwen3.5-4B and Gemma-4-12B). Override
    // via env var `MATVEC_V3_VARIANT`:
    //   v4_nr4      — DEFAULT. 4 rows per SIMD group, no threadgroup cache.
    //   v6          — v4_nr4 + pre-multiplied lx; ALU-saving variant for
    //                 hardware where shifts are expensive.
    //   tiled       — v3 with 4096-elem threadgroup x-cache (in_dim > 4096)
    //   large       — v3 with 8192-elem x-cache
    //   sbcache     — v3 with x + scale/bias in threadgroup memory
    //   splitk      — 2-pass split-K reduction
    //   fast        — v1 fallback (small TG, no tiling)
    let env_variant = std::env::var("MATVEC_V3_VARIANT").ok();
    let need_large_int4 = use_fast >= 3 && in_dim > 4096;
    let mut variant = if let Some(v) = env_variant.clone() {
        v
    } else if use_fast >= 3 && ctx.matvec_v4_nr4.is_some() {
        // Default for any INT4 matvec when the v4_nr4 pipeline is loaded.
        "v4_nr4".to_string()
    } else if need_large_int4 {
        "tiled".to_string()
    } else {
        String::new()
    };
    // v7 requires in_dim % 512 == 0; transparently fall back when not.
    if variant == "v7" && (in_dim % 512 != 0 || ctx.matvec_v7.is_none()) {
        variant = "v4_nr4".to_string();
    }

    // v4_nr4: usable at ANY in_dim (no threadgroup cache, so not bound by
    // 4096-tile limit). Dispatch: 4 rows/SIMD-group × 2 SIMD-groups/TG = 8
    // rows per TG, 64 threads per TG.
    if use_fast >= 3 && variant == "v4_nr4" && ctx.matvec_v4_nr4.is_some() {
        let pipeline = ctx.matvec_v4_nr4.as_ref().unwrap();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(w_packed), w_offset);
        encoder.set_buffer(1, Some(scales), s_offset);
        encoder.set_buffer(2, Some(biases), b_offset);
        encoder.set_buffer(3, Some(x), x_offset);
        encoder.set_buffer(4, Some(out), o_offset);
        unsafe {
            set_u32(encoder, 5, out_dim);
            set_u32(encoder, 6, in_dim);
            set_u32(encoder, 7, group_size);
        }
        const ROWS_PER_TG_V4: u32 = 8; // NR0=4 × NSG=2
        let num_tgs = (out_dim + ROWS_PER_TG_V4 - 1) / ROWS_PER_TG_V4;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(64, 1, 1), // NSG=2 × 32 lanes
        );
        return;
    }

    // v7: MLX-style qmv_fast (values_per_thread=16, pre-mult x cache).
    // REQUIRES in_dim % 512 == 0. Same dispatch shape as v4_nr4: 8 rows/TG,
    // 64 threads/TG. When alignment fails, transparently fall back to
    // v4_nr4 below.
    if use_fast >= 3 && variant == "v7"
        && ctx.matvec_v7.is_some()
        && in_dim % 512 == 0
    {
        let pipeline = ctx.matvec_v7.as_ref().unwrap();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(w_packed), w_offset);
        encoder.set_buffer(1, Some(scales), s_offset);
        encoder.set_buffer(2, Some(biases), b_offset);
        encoder.set_buffer(3, Some(x), x_offset);
        encoder.set_buffer(4, Some(out), o_offset);
        unsafe {
            set_u32(encoder, 5, out_dim);
            set_u32(encoder, 6, in_dim);
            set_u32(encoder, 7, group_size);
        }
        const ROWS_PER_TG_V7: u32 = 8; // NR0=4 × NSG=2
        let num_tgs = (out_dim + ROWS_PER_TG_V7 - 1) / ROWS_PER_TG_V7;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        return;
    }
    // v6: v4_nr4 + pre-multiplied lx + NSG=4. Dispatch: 4 rows × 4 SGs = 16
    // rows per TG, 128 threads per TG.
    if use_fast >= 3 && variant == "v6" && ctx.matvec_v6.is_some() {
        let pipeline = ctx.matvec_v6.as_ref().unwrap();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(w_packed), w_offset);
        encoder.set_buffer(1, Some(scales), s_offset);
        encoder.set_buffer(2, Some(biases), b_offset);
        encoder.set_buffer(3, Some(x), x_offset);
        encoder.set_buffer(4, Some(out), o_offset);
        unsafe {
            set_u32(encoder, 5, out_dim);
            set_u32(encoder, 6, in_dim);
            set_u32(encoder, 7, group_size);
        }
        const ROWS_PER_TG_V6: u32 = 8;  // NR0=4 × NSG=2
        let num_tgs = (out_dim + ROWS_PER_TG_V6 - 1) / ROWS_PER_TG_V6;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(64, 1, 1),  // NSG=2 × 32 lanes
        );
        return;
    }

    // Handle split-K specially — it's a 2-pass dispatch with a partials buffer.
    if need_large_int4 && variant == "splitk"
        && ctx.matvec_v3_splitk_pass1.is_some()
        && ctx.matvec_v3_splitk_pass2.is_some()
        && ctx.buf_splitk_partials.is_some()
    {
        const K_SPLIT: u32 = 4;
        let p1 = ctx.matvec_v3_splitk_pass1.as_ref().unwrap();
        let p2 = ctx.matvec_v3_splitk_pass2.as_ref().unwrap();
        let partials = ctx.buf_splitk_partials.as_ref().unwrap();

        // Pass 1: 1D linearised grid = num_row_tiles * K_SPLIT.
        let num_row_tiles = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
        encoder.set_compute_pipeline_state(p1);
        encoder.set_buffer(0, Some(w_packed), w_offset);
        encoder.set_buffer(1, Some(scales), s_offset);
        encoder.set_buffer(2, Some(biases), b_offset);
        encoder.set_buffer(3, Some(x), x_offset);
        encoder.set_buffer(4, Some(partials), 0);
        unsafe {
            set_u32(encoder, 5, out_dim);
            set_u32(encoder, 6, in_dim);
            set_u32(encoder, 7, group_size);
            set_u32(encoder, 8, num_row_tiles);
        }
        encoder.dispatch_thread_groups(
            MTLSize::new((num_row_tiles as u64) * (K_SPLIT as u64), 1, 1),
            MTLSize::new(TG_SIZE as u64, 1, 1),
        );

        // Pass 2: reduce K_SPLIT partials per output row.
        encoder.set_compute_pipeline_state(p2);
        encoder.set_buffer(0, Some(partials), 0);
        encoder.set_buffer(1, Some(out), o_offset);
        unsafe { set_u32(encoder, 2, out_dim); }
        let tgs = (out_dim + 255) / 256;
        encoder.dispatch_thread_groups(
            MTLSize::new(tgs as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        return;
    }

    // Single-dispatch variants.
    let pipeline = if need_large_int4 {
        match variant.as_str() {
            "large"   if ctx.matvec_v3_tiled_large.is_some()
                      => ctx.matvec_v3_tiled_large.as_ref().unwrap(),
            "sbcache" if ctx.matvec_v3_tiled_sbcache.is_some()
                      => ctx.matvec_v3_tiled_sbcache.as_ref().unwrap(),
            "fast"    => &ctx.matvec_fast,
            // "tiled" (default) and any unknown name → 4096-tile variant.
            _ if ctx.matvec_v3_tiled.is_some()
                      => ctx.matvec_v3_tiled.as_ref().unwrap(),
            _         => &ctx.matvec_fast,
        }
    } else if use_fast >= 3 {
        &ctx.matvec_v3
    } else if use_fast >= 1 {
        &ctx.matvec_fast
    } else {
        &ctx.matvec_naive
    };

    // Dispatch shape: v3-style (ROWS_PER_TG tile) for all variants except
    // explicit "fast" fallback which uses 64-thread-per-row.
    let use_fast = if need_large_int4 {
        if variant == "fast" { 1 } else { 3 }
    } else {
        use_fast
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w_packed), w_offset);
    encoder.set_buffer(1, Some(scales), s_offset);
    encoder.set_buffer(2, Some(biases), b_offset);
    encoder.set_buffer(3, Some(x), x_offset);
    encoder.set_buffer(4, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 5, out_dim);
        set_u32(encoder, 6, in_dim);
        set_u32(encoder, 7, group_size);
    }

    if use_fast >= 3 {
        let num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(TG_SIZE as u64, 1, 1),
        );
    } else {
        let tg_size: u64 = if use_fast >= 1 { 64 } else { 256 };
        encoder.dispatch_thread_groups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(tg_size, 1, 1),
        );
    }
}

/// Encode FP8_E4M3 dequant matvec with buffer offsets.
pub fn encode_matvec_fp8_e4m3_offset(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_u8: &BufferRef, w_offset: u64,
    scales: &BufferRef, s_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
) {
    let pipeline = ctx.matvec_fp8_e4m3.as_ref().expect("matvec_fp8_e4m3 kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w_u8), w_offset);
    encoder.set_buffer(1, Some(scales), s_offset);
    encoder.set_buffer(2, Some(x), x_offset);
    encoder.set_buffer(3, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 4, out_dim);
        set_u32(encoder, 5, in_dim);
        set_u32(encoder, 6, group_size);
    }
    let num_tgs = (out_dim as u64 + ROWS_PER_TG as u64 - 1) / ROWS_PER_TG as u64;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
}

/// Encode a BF16 matvec with buffer offsets (for BQ4: attention, routers, lm_head).
pub fn encode_matvec_bf16_offset(
    ctx: &MetalContext,
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

/// Encode INT8 per-channel symmetric matvec (lm_head).
pub fn encode_matvec_int8_offset(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_i8: &BufferRef, w_offset: u64,
    scales: &BufferRef, s_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.matvec_int8);
    encoder.set_buffer(0, Some(w_i8), w_offset);
    encoder.set_buffer(1, Some(scales), s_offset);
    encoder.set_buffer(2, Some(x), x_offset);
    encoder.set_buffer(3, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 4, out_dim);
        set_u32(encoder, 5, in_dim);
    }
    let num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
}

// ---------------------------------------------------------------------------
// Fused gate_proj + up_proj + SwiGLU
// ---------------------------------------------------------------------------

/// Encode the fused gate+up+SwiGLU dispatch. Replaces a separate gate matvec,
/// up matvec, and elementwise swiglu with a single kernel that reads x once
/// into shared memory and computes both projections + activation per row.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_gate_up_swiglu(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    gate_w: &BufferRef, gate_w_off: u64,
    gate_s: &BufferRef, gate_s_off: u64,
    gate_b: &BufferRef, gate_b_off: u64,
    up_w:   &BufferRef, up_w_off:   u64,
    up_s:   &BufferRef, up_s_off:   u64,
    up_b:   &BufferRef, up_b_off:   u64,
    x:      &BufferRef, x_off:      u64,
    out:    &BufferRef, out_off:    u64,
    out_dim: u32,
    in_dim:  u32,
    group_size: u32,
) -> bool {
    let pipeline = match ctx.fused_gate_up_swiglu_v3.as_ref() {
        Some(p) => p,
        None => return false,
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(gate_w), gate_w_off);
    encoder.set_buffer(1, Some(gate_s), gate_s_off);
    encoder.set_buffer(2, Some(gate_b), gate_b_off);
    encoder.set_buffer(3, Some(up_w),   up_w_off);
    encoder.set_buffer(4, Some(up_s),   up_s_off);
    encoder.set_buffer(5, Some(up_b),   up_b_off);
    encoder.set_buffer(6, Some(x),      x_off);
    encoder.set_buffer(7, Some(out),    out_off);
    unsafe {
        set_u32(encoder, 8,  out_dim);
        set_u32(encoder, 9,  in_dim);
        set_u32(encoder, 10, group_size);
    }
    let num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
    true
}

/// Encode fused gate + up + GELU(tanh-approx) for Gemma's MLP. Same buffer
/// layout as `encode_fused_gate_up_swiglu`; the only difference is the
/// activation. Returns false when the pipeline isn't in the loaded bundle.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_gate_up_geglu_tanh(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    gate_w: &BufferRef, gate_w_off: u64,
    gate_s: &BufferRef, gate_s_off: u64,
    gate_b: &BufferRef, gate_b_off: u64,
    up_w:   &BufferRef, up_w_off:   u64,
    up_s:   &BufferRef, up_s_off:   u64,
    up_b:   &BufferRef, up_b_off:   u64,
    x:      &BufferRef, x_off:      u64,
    out:    &BufferRef, out_off:    u64,
    out_dim: u32,
    in_dim:  u32,
    group_size: u32,
) -> bool {
    let pipeline = match ctx.fused_gate_up_geglu_tanh_v3.as_ref() {
        Some(p) => p,
        None => return false,
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(gate_w), gate_w_off);
    encoder.set_buffer(1, Some(gate_s), gate_s_off);
    encoder.set_buffer(2, Some(gate_b), gate_b_off);
    encoder.set_buffer(3, Some(up_w),   up_w_off);
    encoder.set_buffer(4, Some(up_s),   up_s_off);
    encoder.set_buffer(5, Some(up_b),   up_b_off);
    encoder.set_buffer(6, Some(x),      x_off);
    encoder.set_buffer(7, Some(out),    out_off);
    unsafe {
        set_u32(encoder, 8,  out_dim);
        set_u32(encoder, 9,  in_dim);
        set_u32(encoder, 10, group_size);
    }
    let num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
    true
}


// ---------------------------------------------------------------------------
// SwiGLU
// ---------------------------------------------------------------------------

/// Encode SwiGLU into an existing encoder.
pub fn encode_swiglu(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    gate: &BufferRef, gate_offset: u64,
    up: &BufferRef, up_offset: u64,
    out: &BufferRef, out_offset: u64,
    dim: u32,
) {
    if let Some(ref swiglu_vec4) = ctx.swiglu_vec4 {
        if dim % 4 == 0 {
            encoder.set_compute_pipeline_state(swiglu_vec4);
            encoder.set_buffer(0, Some(gate), gate_offset);
            encoder.set_buffer(1, Some(up), up_offset);
            encoder.set_buffer(2, Some(out), out_offset);
            unsafe { set_u32(encoder, 3, dim); }
            let vec_dim = dim / 4;
            let num_tgs = (vec_dim + 255) / 256;
            encoder.dispatch_thread_groups(
                MTLSize::new(num_tgs as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            return;
        }
    }

    encoder.set_compute_pipeline_state(&ctx.swiglu);
    encoder.set_buffer(0, Some(gate), gate_offset);
    encoder.set_buffer(1, Some(up), up_offset);
    encoder.set_buffer(2, Some(out), out_offset);
    unsafe { set_u32(encoder, 3, dim); }
    let num_tgs = (dim + 255) / 256;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Encode FP4_E2M1 dequant matvec with buffer offsets.
pub fn encode_matvec_fp4_e2m1_offset(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_packed: &BufferRef, w_offset: u64,
    scales: &BufferRef, s_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
) {
    let pipeline = ctx.matvec_fp4_e2m1.as_ref().expect("matvec_fp4_e2m1 kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w_packed), w_offset);
    encoder.set_buffer(1, Some(scales), s_offset);
    encoder.set_buffer(2, Some(x), x_offset);
    encoder.set_buffer(3, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 4, out_dim);
        set_u32(encoder, 5, in_dim);
        set_u32(encoder, 6, group_size);
    }
    let num_tgs = (out_dim as u64 + ROWS_PER_TG as u64 - 1) / ROWS_PER_TG as u64;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
}

// ---------------------------------------------------------------------------
// RMS Normalization
// ---------------------------------------------------------------------------

// ─── Linear attention GPU kernels ─────────────────────────────────────────

/// Encode gated delta net step — SSM recurrence (one threadgroup per v-head).
pub fn encode_gated_delta_net_step(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    state: &BufferRef,
    q: &BufferRef, q_offset: u64,
    k: &BufferRef, k_offset: u64,
    v: &BufferRef, v_offset: u64,
    g_decay: &BufferRef,
    beta_gate: &BufferRef,
    output: &BufferRef,
    num_v_heads: u32,
    k_heads_per_v: u32,
    _key_dim: u32,   // kernel hardcodes key_dim=128
    value_dim: u32,
) {
    debug_assert!(_key_dim == 128, "gated_delta_net_step kernel hardcodes key_dim=128");
    debug_assert!(value_dim == 128, "gated_delta_net_step kernel hardcodes value_dim=128");
    let pipeline = ctx.gated_delta_net_step.as_ref().expect("gated_delta_net_step kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(state), 0);
    encoder.set_buffer(1, Some(q), q_offset);
    encoder.set_buffer(2, Some(k), k_offset);
    encoder.set_buffer(3, Some(v), v_offset);
    encoder.set_buffer(4, Some(g_decay), 0);
    encoder.set_buffer(5, Some(beta_gate), 0);
    encoder.set_buffer(6, Some(output), 0);
    unsafe { set_u32(encoder, 7, k_heads_per_v); }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_v_heads as u64, 1, 1),
        MTLSize::new(value_dim as u64, 1, 1),
    );
}

/// Encode compute_decay_beta — computes g_decay and beta_gate from alpha, beta, A_log, dt_bias.
pub fn encode_compute_decay_beta(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    alpha: &BufferRef,
    beta: &BufferRef,
    a_log: &BufferRef, a_log_offset: u64,
    dt_bias: &BufferRef, dt_bias_offset: u64,
    g_decay: &BufferRef,
    beta_gate: &BufferRef,
    num_v_heads: u32,
) {
    let pipeline = ctx.compute_decay_beta.as_ref().expect("compute_decay_beta kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(alpha), 0);
    encoder.set_buffer(1, Some(beta), 0);
    encoder.set_buffer(2, Some(a_log), a_log_offset);
    encoder.set_buffer(3, Some(dt_bias), dt_bias_offset);
    encoder.set_buffer(4, Some(g_decay), 0);
    encoder.set_buffer(5, Some(beta_gate), 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(num_v_heads as u64, 1, 1),
    );
}

/// Encode RMS norm for q/k (per-head, bare norm with scale).
pub fn encode_rms_norm_qk(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    q: &BufferRef, q_offset: u64,
    k: &BufferRef, k_offset: u64,
    num_heads: u32,
    key_dim: u32,
    inv_scale: f32,
) {
    let pipeline = ctx.rms_norm_qk.as_ref().expect("rms_norm_qk kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(q), q_offset);
    encoder.set_buffer(1, Some(k), k_offset);
    unsafe {
        set_u32(encoder, 2, key_dim);
        set_f32(encoder, 3, inv_scale);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(key_dim as u64, 1, 1),
    );
}

/// Encode gated RMS norm (z-gated output normalization, per v-head).
pub fn encode_gated_rms_norm(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    values: &BufferRef,
    z: &BufferRef,
    weight: &BufferRef, weight_offset: u64,  // bf16 u16 weight, value_dim elements, shared across heads
    output: &BufferRef,
    num_v_heads: u32,
    value_dim: u32,
) {
    let pipeline = ctx.gated_rms_norm.as_ref().expect("gated_rms_norm kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(values), 0);
    encoder.set_buffer(1, Some(z), 0);
    encoder.set_buffer(2, Some(weight), weight_offset);
    encoder.set_buffer(3, Some(output), 0);
    unsafe {
        set_u32(encoder, 4, value_dim);
        set_f32(encoder, 5, 1e-6);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_v_heads as u64, 1, 1),
        MTLSize::new(value_dim as u64, 1, 1),
    );
}

/// Encode depthwise conv1d step (with SiLU activation and state update).
pub fn encode_conv1d_step(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    conv_state: &BufferRef,   // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    input: &BufferRef,        // [conv_dim]
    weights: &BufferRef, weights_offset: u64,  // bf16 u16 weight, [conv_dim * 4]
    output: &BufferRef,       // [conv_dim]
    conv_dim: u32,
) {
    let pipeline = ctx.conv1d_step.as_ref().expect("conv1d_step kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(conv_state), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(weights), weights_offset);
    encoder.set_buffer(3, Some(output), 0);
    unsafe { set_u32(encoder, 4, conv_dim); }
    let num_tgs = (conv_dim + 255) / 256;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

// ---------------------------------------------------------------------------
// Batched matvec variants (`_n`) for batched prefill.
//
// Input x is [N, in_dim] row-major, output is [N, out_dim] row-major.
// Internally launches ceil(out_dim / ROWS_PER_TG) * N threadgroups,
// linearized as tgid = row_tile + n * num_row_tiles.
// ---------------------------------------------------------------------------

pub fn encode_matvec_bf16_n(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_bf16: &BufferRef, w_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
    n: u32,
) {
    // Use GEMM-tiled variant when available and N >= 2.
    // For N=1, fall back to per-token kernel — GEMM tile would waste 3/4 of the work.
    if n >= 2 && ctx.matvec_bf16_gemm_n.is_some() {
        let ncols_per_tg: u32 = 4;
        let pipeline = ctx.matvec_bf16_gemm_n.as_ref().unwrap();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(w_bf16), w_offset);
        encoder.set_buffer(1, Some(x), x_offset);
        encoder.set_buffer(2, Some(out), o_offset);
        let num_row_tiles = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
        let num_n_tiles   = (n + ncols_per_tg - 1) / ncols_per_tg;
        unsafe {
            set_u32(encoder, 3, out_dim);
            set_u32(encoder, 4, in_dim);
            set_u32(encoder, 5, n);
            set_u32(encoder, 6, num_row_tiles);
        }
        encoder.dispatch_thread_groups(
            MTLSize::new((num_row_tiles as u64) * (num_n_tiles as u64), 1, 1),
            MTLSize::new(TG_SIZE as u64, 1, 1),
        );
        return;
    }
    let pipeline = ctx.matvec_bf16_n.as_ref().expect("matvec_bf16_n kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w_bf16), w_offset);
    encoder.set_buffer(1, Some(x), x_offset);
    encoder.set_buffer(2, Some(out), o_offset);
    let num_row_tiles = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
    unsafe {
        set_u32(encoder, 3, out_dim);
        set_u32(encoder, 4, in_dim);
        set_u32(encoder, 5, num_row_tiles);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new((num_row_tiles as u64) * (n as u64), 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
}

pub fn encode_matvec_int8_n(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_i8: &BufferRef, w_offset: u64,
    scales: &BufferRef, s_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
    n: u32,
) {
    let pipeline = ctx.matvec_int8_n.as_ref().expect("matvec_int8_n kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w_i8), w_offset);
    encoder.set_buffer(1, Some(scales), s_offset);
    encoder.set_buffer(2, Some(x), x_offset);
    encoder.set_buffer(3, Some(out), o_offset);
    let num_row_tiles = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
    unsafe {
        set_u32(encoder, 4, out_dim);
        set_u32(encoder, 5, in_dim);
        set_u32(encoder, 6, num_row_tiles);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new((num_row_tiles as u64) * (n as u64), 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
}

/// Encode causal batched SDPA for prefill.
///
/// N new tokens with Q [N, num_q_heads, head_dim] vs K/V cache where the
/// new tokens' K/V are at positions [past_pos .. past_pos+N). Causal mask
/// is implicit: token i only attends positions 0..(past_pos + i).
pub fn encode_attn_sdpa_causal_n(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    q: &BufferRef, q_offset: u64,
    k_cache: &BufferRef,
    v_cache: &BufferRef,
    out: &BufferRef, o_offset: u64,
    past_pos: u32,
    num_q_heads: u32,
    head_dim: u32,
    n: u32,
) {
    let pipeline = ctx.attn_sdpa_causal_n.as_ref().expect("attn_sdpa_causal_n kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(q), q_offset);
    encoder.set_buffer(1, Some(k_cache), 0);
    encoder.set_buffer(2, Some(v_cache), 0);
    encoder.set_buffer(3, Some(out), o_offset);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    unsafe {
        set_u32(encoder, 4, past_pos);
        set_u32(encoder, 5, num_q_heads);
        set_f32(encoder, 6, scale);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new((num_q_heads as u64) * (n as u64), 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Append N tokens' K/V into the KV cache at positions [past_pos .. past_pos+N).
pub fn encode_kv_cache_append_n(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    k_in: &BufferRef, k_offset: u64,
    v_in: &BufferRef, v_offset: u64,
    k_cache: &BufferRef,
    v_cache: &BufferRef,
    past_pos: u32,
    kv_dim: u32,
    n: u32,
) {
    let pipeline = ctx.kv_cache_append_n.as_ref().expect("kv_cache_append_n kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(k_in), k_offset);
    encoder.set_buffer(1, Some(v_in), v_offset);
    encoder.set_buffer(2, Some(k_cache), 0);
    encoder.set_buffer(3, Some(v_cache), 0);
    let tgs_per_row = (kv_dim + 255) / 256;
    unsafe {
        set_u32(encoder, 4, past_pos);
        set_u32(encoder, 5, kv_dim);
        set_u32(encoder, 6, tgs_per_row);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new((tgs_per_row as u64) * (n as u64), 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Encode a GPU-side buffer copy: src[src_offset..src_offset + count*4]
/// → dst[dst_offset..dst_offset + count*4]. Used inside a compute encoder
/// to copy between buffers while preserving encoder-order serialization.
pub fn encode_buffer_copy_f32(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    src: &BufferRef, src_offset: u64,
    dst: &BufferRef, dst_offset: u64,
    count: u32,
) {
    let pipeline = ctx.buffer_copy_f32.as_ref().expect("buffer_copy_f32 missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(src), src_offset);
    encoder.set_buffer(1, Some(dst), dst_offset);
    unsafe { set_u32(encoder, 2, count); }
    let tgs = (count + 255) / 256;
    encoder.dispatch_thread_groups(
        MTLSize::new(tgs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Encode v7_n batched matvec: N x vectors share weight reads. Caller
/// must ensure in_dim % 512 == 0; returns false otherwise (caller falls
/// back to per-token v7 dispatches).
///
/// Picks `v7_n8` if n >= 5 (better weight reuse) else `v7_n4`. For n > 8,
/// the caller dispatches multiple times with x/out advanced by 8.
#[allow(clippy::too_many_arguments)]
pub fn encode_dequant_matvec_4bit_v7_n(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_packed: &BufferRef, w_offset: u64,
    scales: &BufferRef, s_offset: u64,
    biases: &BufferRef, b_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    n_tokens: u32,
) -> bool {
    if in_dim % 512 != 0 || n_tokens == 0 {
        return false;
    }
    let pipeline = if n_tokens >= 5 {
        match ctx.matvec_v7_n8.as_ref() { Some(p) => p, None => return false }
    } else {
        match ctx.matvec_v7_n4.as_ref() { Some(p) => p, None => return false }
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w_packed), w_offset);
    encoder.set_buffer(1, Some(scales), s_offset);
    encoder.set_buffer(2, Some(biases), b_offset);
    encoder.set_buffer(3, Some(x), x_offset);
    encoder.set_buffer(4, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 5, out_dim);
        set_u32(encoder, 6, in_dim);
        set_u32(encoder, 7, group_size);
        set_u32(encoder, 8, n_tokens);
    }
    const ROWS_PER_TG_V7N: u32 = 8;
    let num_tgs = (out_dim + ROWS_PER_TG_V7N - 1) / ROWS_PER_TG_V7N;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(64, 1, 1),
    );
    true
}

pub fn encode_dequant_matvec_4bit_n(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_packed: &BufferRef, w_offset: u64,
    scales: &BufferRef, s_offset: u64,
    biases: &BufferRef, b_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    n: u32,
) {
    let pipeline = ctx.dequant_matvec_4bit_n.as_ref().expect("dequant_matvec_4bit_n kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w_packed), w_offset);
    encoder.set_buffer(1, Some(scales), s_offset);
    encoder.set_buffer(2, Some(biases), b_offset);
    encoder.set_buffer(3, Some(x), x_offset);
    encoder.set_buffer(4, Some(out), o_offset);
    let num_row_tiles = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
    unsafe {
        set_u32(encoder, 5, out_dim);
        set_u32(encoder, 6, in_dim);
        set_u32(encoder, 7, group_size);
        set_u32(encoder, 8, num_row_tiles);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new((num_row_tiles as u64) * (n as u64), 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
}
