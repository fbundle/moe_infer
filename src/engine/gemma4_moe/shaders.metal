// Gemma 4 26B-A4B specific Metal kernels.
//
// The matvec family (`matvec_bf16`, `matvec_int8`, `dequant_matvec_4bit_*`,
// and the `_n` batched variants) is shared with Qwen3.6 — see
// `engine/qwen35_moe/shaders.metal`. This file only defines the kernels
// that Gemma 4 needs and Qwen does not.
//
// Status:
//   - All kernels in this file are SKELETONS. They compile (or, where
//     marked with #if 0, are commented out) but have not been validated
//     against a reference implementation. Each carries a TODO with the
//     verification step it needs.

// NOTE: This file is intended to be CONCATENATED after
// engine/qwen35_moe/shaders.metal at compile time, so we don't repeat
// #include <metal_stdlib>, `using namespace metal`, or `bf16_to_f32` —
// they're inherited from the qwen35 source.

// ============================================================================
// Sliding-window causal attention (Gemma 4 sliding-attention layers).
//
// Standard SDPA but the K/V index range is restricted to a window of
// `sliding_window` positions ending at the query position. Equivalent to
// causal mask AND distance-mask (i.e., position j is attended iff
// max(0, pos - sliding_window + 1) <= j <= pos).
//
// Grid: one threadgroup per query head (single-token; for prefill batched
// version use `attn_sdpa_sliding_causal_n`, TBD).
//
// TODO: implement. Skeleton omitted until we have a reference output to
// diff against. Probable approach: clone `attn_sdpa_fused` from
// qwen35_moe/shaders.metal and replace the unbounded `pos < seq_len` loop
// with `max(0, seq_len - sliding_window) <= pos < seq_len`.
// ============================================================================

#if 0
kernel void attn_sdpa_sliding_causal(
    device const float* Q          [[buffer(0)]],   // [num_heads, HEAD_DIM]
    device const float* K_cache    [[buffer(1)]],
    device const float* V_cache    [[buffer(2)]],
    device float*       output     [[buffer(3)]],
    constant uint&      seq_len    [[buffer(4)]],
    constant uint&      sliding_window [[buffer(5)]],
    constant float&     scale      [[buffer(6)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // TODO: clone attn_sdpa_fused, restrict pos loop to
    //   uint start = (seq_len > sliding_window) ? (seq_len - sliding_window) : 0;
    //   for (uint pos = start + simd_group; pos < seq_len; pos += BN) { ... }
}
#endif

// ============================================================================
// Q/K head norm + RoPE — Gemma 4 variants.
//
// Gemma 4 sliding layers: RoPE applied to the full head_dim with theta=10k.
// Gemma 4 full layers:    RoPE applied to first 25% of each head's dims
//                          (partial_rotary_factor=0.25), with theta=1M.
//
// Both use Q_norm and K_norm (per-head RMSNorm) before RoPE. So we need
// FOUR variants (or two with a flag):
//
//   q_head_norm_rope_gemma_sliding   — theta=10k, full rotary
//   q_head_norm_rope_gemma_full      — theta=1M, partial rotary (first 25%)
//   k_head_norm_rope_gemma_sliding   — same as above for K
//   k_head_norm_rope_gemma_full      — same as above for K
//
// TODO: implement. Probable approach: copy qwen35 `q_head_norm_rope`
// kernel; parameterize partial_rotary_dim (full = head_dim, sliding = 0
// meaning all rotated, full-attn = head_dim/4). Gemma does NOT have a
// query-gate split (Qwen's q_proj produces 2*q_dim for Q+Q_gate; Gemma's
// produces q_dim).
// ============================================================================

// ============================================================================
// GELU activation (gelu_pytorch_tanh approximation, used in Gemma 4 FFN).
//
// gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Used inside each expert's FFN: `gelu(gate(x)) * up(x)` (not SwiGLU; the
// gating is multiplicative-only, the activation is GELU instead of SiLU).
// ============================================================================

kernel void gelu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float g = gate[tid];
    // gelu_pytorch_tanh
    float g3 = g * g * g;
    float inner = 0.7978845608f * (g + 0.044715f * g3);  // sqrt(2/pi) ≈ 0.7978845608
    float gelu_g = 0.5f * g * (1.0f + tanh(inner));
    out[tid] = gelu_g * up[tid];
}

// ============================================================================
// Final logit softcap.  logits := softcap * tanh(logits / softcap)
// Applied AFTER lm_head, BEFORE softmax/sampling.
//
// For Gemma 4: softcap = 30.0.
// ============================================================================

kernel void logit_softcap(
    device float*  logits     [[buffer(0)]],
    constant uint& vocab_size [[buffer(1)]],
    constant float& cap       [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= vocab_size) return;
    float x = logits[tid];
    logits[tid] = cap * tanh(x / cap);
}
