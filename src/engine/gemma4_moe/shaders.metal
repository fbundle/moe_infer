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
// Sliding-window causal SDPA — for Gemma 4 sliding-attention layers.
//
// Online softmax over a restricted position range:
//   start = max(0, seq_len - sliding_window)
//   only positions [start, seq_len) are attended
//
// Reuses qwen35's compile-time HEAD_DIM=256 (matches Gemma 4 sliding
// layers). For Gemma 4's full-attn layers (head_dim=512) write a
// separate kernel — single-shader can't handle both because V (the
// per-lane accumulator size) must be compile-time.
//
// Grid: one threadgroup per query head. TG: 256 threads = 8 SIMD groups
// of 32. Same online-softmax algorithm as `attn_sdpa_fused`; only the
// position-loop bound differs.
// ============================================================================
//
// Gemma 4 sliding layers use KV_DIM_GEMMA_SLIDING = num_kv_heads(8) *
// head_dim(256) = 2048. Qwen's KV_DIM is 2*256=512. So the kernel can't
// reuse qwen35's KV_DIM #define — it takes kv_dim as a runtime constant.
// Same for HEADS_PER_KV (Gemma 4 sliding: 16/8=2; Qwen: 16/2=8).

kernel void attn_sdpa_sliding_causal(
    device const float* Q              [[buffer(0)]],   // [num_q_heads, HEAD_DIM]
    device const float* K_cache        [[buffer(1)]],   // [max_seq, kv_dim]
    device const float* V_cache        [[buffer(2)]],
    device float*       output         [[buffer(3)]],   // [num_q_heads, HEAD_DIM]
    constant uint&      seq_len        [[buffer(4)]],
    constant uint&      sliding_window [[buffer(5)]],
    constant float&     scale          [[buffer(6)]],
    constant uint&      kv_dim         [[buffer(7)]],   // num_kv_heads * HEAD_DIM
    constant uint&      heads_per_kv   [[buffer(8)]],   // num_q_heads / num_kv_heads
    uint tgid       [[threadgroup_position_in_grid]],   // = query head index
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint BD = 32;
    constexpr uint BN = 8;
    constexpr uint V  = HEAD_DIM / BD;   // 256/32 = 8 — same as Qwen sliding

    uint h    = tgid;
    uint kv_h = h / heads_per_kv;

    device const float* qh = Q + h * HEAD_DIM;
    device const float* k_base = K_cache + kv_h * HEAD_DIM;
    device const float* v_base = V_cache + kv_h * HEAD_DIM;

    float q_vals[V];
    float o_vals[V] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    constexpr float log2_e = 1.442695041f;
    float q_scale = scale * log2_e;
    uint elem_base = simd_lane * V;
    for (uint j = 0; j < V; j++) {
        q_vals[j] = q_scale * qh[elem_base + j];
    }

    float max_score = -1e30f;
    float sum_exp   = 0.0f;

    // Sliding-window restriction: only attend the last `sliding_window` positions.
    uint start_pos = (seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;

    for (uint pos = start_pos + simd_group; pos < seq_len; pos += BN) {
        device const float* kp = k_base + pos * kv_dim + elem_base;
        device const float* vp = v_base + pos * kv_dim + elem_base;

        float score = 0.0f;
        for (uint j = 0; j < V; j++) {
            score += q_vals[j] * kp[j];
        }
        score = simd_sum(score);

        float new_max   = max(max_score, score);
        float factor    = fast::exp2(max_score - new_max);
        float exp_score = fast::exp2(score - new_max);

        max_score = new_max;
        sum_exp   = sum_exp * factor + exp_score;

        for (uint j = 0; j < V; j++) {
            o_vals[j] = o_vals[j] * factor + exp_score * vp[j];
        }
    }

    // Merge across SIMD groups (identical to attn_sdpa_fused's tail).
    threadgroup float sg_max[BD];
    threadgroup float sg_sum[BN];
    threadgroup float sg_partial[BN * HEAD_DIM];

    sg_max[simd_lane] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0) {
        sg_max[simd_group] = max_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max  = sg_max[simd_lane];
    float global_max = simd_max(local_max);

    float group_max     = simd_broadcast_first(max_score);
    float group_sum     = simd_broadcast_first(sum_exp);
    float rescale       = fast::exp2(group_max - global_max);
    float rescaled_sum  = group_sum * rescale;

    for (uint j = 0; j < V; j++) {
        o_vals[j] *= rescale;
    }
    for (uint j = 0; j < V; j++) {
        sg_partial[simd_group * HEAD_DIM + elem_base + j] = o_vals[j];
    }
    if (simd_lane == 0) {
        sg_sum[simd_group] = rescaled_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < V; j++) {
        float sum = 0.0f;
        for (uint g = 0; g < BN; g++) {
            sum += sg_partial[g * HEAD_DIM + elem_base + j];
        }
        o_vals[j] = sum;
    }

    float local_sum  = (simd_lane < BN) ? sg_sum[simd_lane] : 0.0f;
    float global_sum = simd_sum(local_sum);

    for (uint j = 0; j < V; j++) {
        o_vals[j] = (global_sum == 0.0f) ? 0.0f : (o_vals[j] / global_sum);
    }

    device float* out_ptr = output + h * HEAD_DIM + elem_base;
    for (uint j = 0; j < V; j++) {
        out_ptr[j] = o_vals[j];
    }
}

// ============================================================================
// Full causal SDPA for head_dim=512 — Gemma 4 full-attention layers.
//
// Same online-softmax structure as attn_sdpa_sliding_causal, but:
//   - HEAD_DIM = 512 (Gemma 4's global_head_dim), overriding the qwen35
//     #define = 256 via a local constexpr.
//   - V = HEAD_DIM/BD = 512/32 = 16 per-lane accumulator (vs 8 sliding).
//   - No sliding-window restriction: attends [0, seq_len) (full causal).
//   - kv_dim and heads_per_kv come from runtime constants because the
//     5 full layers in 26B-A4B use num_global_key_value_heads=2 (not 8).
//
// Threadgroup memory: sg_partial[BN * 512] = 8*512 = 4096 floats = 16 KB.
// Within Metal's 32 KB TG-memory budget.
// ============================================================================

kernel void attn_sdpa_causal_h512(
    device const float* Q              [[buffer(0)]],   // [num_q_heads, 512]
    device const float* K_cache        [[buffer(1)]],   // [max_seq, kv_dim]
    device const float* V_cache        [[buffer(2)]],
    device float*       output         [[buffer(3)]],   // [num_q_heads, 512]
    constant uint&      seq_len        [[buffer(4)]],
    constant float&     scale          [[buffer(5)]],
    constant uint&      kv_dim         [[buffer(6)]],   // num_kv_heads * 512
    constant uint&      heads_per_kv   [[buffer(7)]],   // num_q_heads / num_kv_heads
    uint tgid       [[threadgroup_position_in_grid]],   // = query head index
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint H  = 512;          // local override of HEAD_DIM
    constexpr uint BD = 32;
    constexpr uint BN = 8;
    constexpr uint V  = H / BD;       // 16

    uint h    = tgid;
    uint kv_h = h / heads_per_kv;

    device const float* qh = Q + h * H;
    device const float* k_base = K_cache + kv_h * H;
    device const float* v_base = V_cache + kv_h * H;

    float q_vals[V];
    float o_vals[V];
    for (uint j = 0; j < V; j++) { o_vals[j] = 0.0f; }

    constexpr float log2_e = 1.442695041f;
    float q_scale = scale * log2_e;
    uint elem_base = simd_lane * V;
    for (uint j = 0; j < V; j++) {
        q_vals[j] = q_scale * qh[elem_base + j];
    }

    float max_score = -1e30f;
    float sum_exp   = 0.0f;

    // Full causal: attend positions [0, seq_len).
    for (uint pos = simd_group; pos < seq_len; pos += BN) {
        device const float* kp = k_base + pos * kv_dim + elem_base;
        device const float* vp = v_base + pos * kv_dim + elem_base;

        float score = 0.0f;
        for (uint j = 0; j < V; j++) {
            score += q_vals[j] * kp[j];
        }
        score = simd_sum(score);

        float new_max   = max(max_score, score);
        float factor    = fast::exp2(max_score - new_max);
        float exp_score = fast::exp2(score - new_max);

        max_score = new_max;
        sum_exp   = sum_exp * factor + exp_score;

        for (uint j = 0; j < V; j++) {
            o_vals[j] = o_vals[j] * factor + exp_score * vp[j];
        }
    }

    threadgroup float sg_max[BD];
    threadgroup float sg_sum[BN];
    threadgroup float sg_partial[BN * H];

    sg_max[simd_lane] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0) {
        sg_max[simd_group] = max_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max  = sg_max[simd_lane];
    float global_max = simd_max(local_max);

    float group_max     = simd_broadcast_first(max_score);
    float group_sum     = simd_broadcast_first(sum_exp);
    float rescale       = fast::exp2(group_max - global_max);
    float rescaled_sum  = group_sum * rescale;

    for (uint j = 0; j < V; j++) {
        o_vals[j] *= rescale;
    }
    for (uint j = 0; j < V; j++) {
        sg_partial[simd_group * H + elem_base + j] = o_vals[j];
    }
    if (simd_lane == 0) {
        sg_sum[simd_group] = rescaled_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < V; j++) {
        float sum = 0.0f;
        for (uint g = 0; g < BN; g++) {
            sum += sg_partial[g * H + elem_base + j];
        }
        o_vals[j] = sum;
    }

    float local_sum  = (simd_lane < BN) ? sg_sum[simd_lane] : 0.0f;
    float global_sum = simd_sum(local_sum);

    for (uint j = 0; j < V; j++) {
        o_vals[j] = (global_sum == 0.0f) ? 0.0f : (o_vals[j] / global_sum);
    }

    device float* out_ptr = output + h * H + elem_base;
    for (uint j = 0; j < V; j++) {
        out_ptr[j] = o_vals[j];
    }
}

// ============================================================================
// RMSNorm without learnable weight — for Gemma 4's v_norm on full-attention
// layers. Returns rms_norm(x) without multiplying by a weight vector.
// Operates per-head: dispatches one TG per (token, head); each TG normalizes
// its head_dim elements.
// ============================================================================

// ============================================================================
// RMSNorm with per-channel BF16 weight — SAFE variant (no simd_sum with
// masked lanes). The qwen35 rms_norm_fused_bf16 uses simd_sum after a
// conditional, which is undefined when not all lanes participate. This
// version uses loop-based reduction throughout.
// ============================================================================

// Plain copy test — verify GPU sees CPU-written buffer.
kernel void copy_test(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      dim [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < dim) dst[tid] = src[tid];
}

// Read input_layernorm.weight (bf16) and dump first 8 to floats.
kernel void weight_dump(
    device const uint16_t* w [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < 8) dst[tid] = bf16_to_f32(w[tid]);
}

// Hardcoded constant write — bypass everything.
kernel void const_write(
    device float* dst [[buffer(0)]],
    constant uint& dim [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < dim) dst[tid] = 7.5f;
}

kernel void rms_norm_safe(
    device const float*    x       [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],
    device float*          out     [[buffer(2)]],
    constant uint&         dim     [[buffer(3)]],
    constant float&        eps     [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]]
) {
    threadgroup float ss[256];
    float local = 0.0f;
    for (uint i = lid; i < dim; i += 256) {
        float v = x[i];
        local += v * v;
    }
    ss[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) ss[lid] += ss[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(ss[0] / float(dim) + eps);
    for (uint i = lid; i < dim; i += 256) {
        float w = bf16_to_f32(weight[i]);
        out[i] = x[i] * inv_rms * w;
    }
}

kernel void rms_norm_no_scale(
    device const float* x         [[buffer(0)]],
    device float*       out       [[buffer(1)]],
    constant uint&      dim       [[buffer(2)]],
    constant float&     eps       [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]]
) {
    threadgroup float ss[256];
    device const float* xh = x + tgid * dim;
    device float* oh = out + tgid * dim;

    float local = 0.0f;
    for (uint i = lid; i < dim; i += 256) {
        float v = xh[i];
        local += v * v;
    }
    ss[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) ss[lid] += ss[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(ss[0] / float(dim) + eps);

    for (uint i = lid; i < dim; i += 256) {
        oh[i] = xh[i] * inv_rms;
    }
}

// ============================================================================
// Q head norm + RoPE — Gemma 4 (NO query-gate split).
//
// Qwen3.6's q_head_norm_rope expects q_proj output to be [num_q, 2*head_dim]
// because Qwen fuses Q and Q-gate into a single linear. Gemma 4 has no
// query-gate; q_proj output is just [num_q, head_dim]. So we write a
// variant that:
//   - reads from [num_q * head_dim] (not 2x)
//   - writes a single q_out (no q_gate_out)
//   - rotary_dim is a runtime parameter (full head_dim for sliding;
//     head_dim/4 for full layers — Gemma's partial_rotary_factor=0.25)
//   - rope_theta is runtime (10k sliding / 1M full)
//   - eps comes from runtime (Gemma uses 1e-6 like Qwen)
//
// For K, qwen35's k_head_norm_rope works as-is — K already has no gate.
// ============================================================================

kernel void q_head_norm_rope_no_gate(
    device const float*    q_proj       [[buffer(0)]],   // [num_q * head_dim]
    device const uint16_t* q_norm_w     [[buffer(1)]],   // [head_dim] bf16, shared across heads
    device float*          q_out        [[buffer(2)]],   // [num_q * head_dim]
    constant uint&         head_dim     [[buffer(3)]],
    constant uint&         rotary_dim   [[buffer(4)]],
    constant float&        rope_theta   [[buffer(5)]],
    constant uint&         pos          [[buffer(6)]],
    constant float&        eps          [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    uint base = head * head_dim;
    float q_val = q_proj[base + tid];

    // RMS norm reduction over head_dim threads.
    threadgroup float partial[512];
    partial[tid] = q_val * q_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < head_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(head_dim) + eps);

    q_val *= inv_rms * bf16_to_f32(q_norm_w[tid]);

    // RoPE on the first `rotary_dim` dims. For Gemma 4 sliding: rotary_dim
    // = head_dim. For full layers: rotary_dim = head_dim / 4 (partial 0.25).
    if (tid < rotary_dim) {
        uint rot_half = rotary_dim / 2;
        float theta;
        float pair_val;
        if (tid < rot_half) {
            theta = float(pos) * pow(rope_theta, -2.0f * float(tid) / float(rotary_dim));
            pair_val = q_proj[base + tid + rot_half]
                       * inv_rms * bf16_to_f32(q_norm_w[tid + rot_half]);
        } else {
            uint pair = tid - rot_half;
            theta = float(pos) * pow(rope_theta, -2.0f * float(pair) / float(rotary_dim));
            pair_val = q_proj[base + pair]
                       * inv_rms * bf16_to_f32(q_norm_w[pair]);
        }
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        if (tid < rot_half) {
            q_out[base + tid] = q_val * cos_t - pair_val * sin_t;
        } else {
            q_out[base + tid] = pair_val * sin_t + q_val * cos_t;
        }
    } else {
        q_out[base + tid] = q_val;
    }
}

// ============================================================================
// BF16 matvec for in_dim > 4096 — needed for Gemma 4 full-layer o_proj
// where in_dim = num_heads * global_head_dim = 16 * 512 = 8192.
// qwen35's matvec_bf16 has x_shared[4096] which silently overflows for
// larger in_dim. Same kernel structure but with x_shared[8192].
// Uses ~32 KB threadgroup memory (within Apple Silicon's 32 KB budget).
// ============================================================================
kernel void matvec_bf16_h8192(
    device const uint16_t* W_bf16 [[buffer(0)]],
    device const float*    x      [[buffer(1)]],
    device float*          out    [[buffer(2)]],
    constant uint&         out_dim [[buffer(3)]],
    constant uint&         in_dim  [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    threadgroup float x_shared[8192];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;
    device const uint16_t* w_row = W_bf16 + row * in_dim;
    float acc = 0.0f;
    for (uint col = simd_lane; col < in_dim; col += 32) {
        acc += bf16_to_f32(w_row[col]) * x_shared[col];
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Q/K head norm + RoPE — Gemma 4 FULL-attention ProportionalRoPE variant.
//
// Differs from q_head_norm_rope_no_gate in two ways:
//
//   1. The rotated dims aren't the first `rotary_dim` — they are the FIRST
//      half of the rotated_dims taken from the LEFT half of the head, AND
//      the first half of the rotated_dims taken from the RIGHT half of the
//      head (where left = head[0..head_dim/2], right = head[head_dim/2..]).
//
//   2. The rotation angle exponents are scaled by the FULL head_dim (not
//      rotary_dim) → "proportional" RoPE.
//
// Matches `ProportionalRoPE` in vendor/mlx-vlm/mlx_vlm/models/gemma4/
// rope_utils.py. Used for Gemma 4 26B-A4B full-attention layers with
// head_dim=512 and partial_rotary_factor=0.25 → rotary_dim=128.
//
// Dispatch: num_q_heads × head_dim threads (same as no_gate variant).
// Constraint: head_dim ≤ 512 (partial[512] threadgroup buffer).
// ============================================================================

kernel void q_head_norm_rope_proportional(
    device const float*    q_proj      [[buffer(0)]],
    device const uint16_t* q_norm_w    [[buffer(1)]],
    device float*          q_out       [[buffer(2)]],
    constant uint&         head_dim    [[buffer(3)]],
    constant uint&         rotary_dim  [[buffer(4)]],
    constant float&        rope_theta  [[buffer(5)]],
    constant uint&         pos         [[buffer(6)]],
    constant float&        eps         [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    uint base = head * head_dim;
    float q_val = q_proj[base + tid];

    // RMS norm reduction over head_dim threads.
    threadgroup float partial[512];
    partial[tid] = q_val * q_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < head_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(head_dim) + eps);

    // Normalised value (RMSNorm × q_norm.weight)
    q_val *= inv_rms * bf16_to_f32(q_norm_w[tid]);

    // ProportionalRoPE: rotate (tid, tid + head_dim/2) for tid < rotary_dim/2
    // AND (tid - head_dim/2, tid) for tid in [head_dim/2, head_dim/2 + rotary_dim/2).
    // Angle index i ∈ [0, rotary_dim/2) where i = tid (left) or tid - head_dim/2 (right).
    // freq exponent uses head_dim (not rotary_dim) — that's the "proportional" part.
    uint rot_half = rotary_dim / 2;
    uint hd_half  = head_dim / 2;
    bool is_left  = tid < rot_half;
    bool is_right = (tid >= hd_half) && (tid < hd_half + rot_half);
    if (is_left || is_right) {
        uint i = is_left ? tid : (tid - hd_half);
        float theta = float(pos) * pow(rope_theta, -2.0f * float(i) / float(head_dim));
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        // Pair value: for left lane, pair is normed right[tid+hd_half]; for
        // right lane, pair is normed left[tid-hd_half]. Both read q_proj and
        // apply RMSNorm+weight to match.
        uint pair_idx = is_left ? (tid + hd_half) : (tid - hd_half);
        float pair_val = q_proj[base + pair_idx]
                         * inv_rms * bf16_to_f32(q_norm_w[pair_idx]);
        if (is_left) {
            q_out[base + tid] = q_val * cos_t - pair_val * sin_t;
        } else {
            // q_out[right] = right*cos + left*sin
            q_out[base + tid] = pair_val * sin_t + q_val * cos_t;
        }
    } else {
        q_out[base + tid] = q_val;
    }
}

// ============================================================================
// Outer (post-residual) multiply by per-layer `layer_scalar`.
// Multiplies a [HIDDEN_DIM] buffer in-place by a single bf16 scalar.
// ============================================================================

kernel void mul_scalar_bf16(
    device float*          buf    [[buffer(0)]],
    device const uint16_t* scalar [[buffer(1)]],   // bf16 (1 element)
    constant uint&         dim    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float s = bf16_to_f32(scalar[0]);
    buf[tid] = buf[tid] * s;
}

// ============================================================================
// Router RMS norm — RMSNorm with per-channel BF16 weight (router.scale) and
// an extra global multiplier sqrt(hidden_size)^-1.
//
// Confirmed against safetensors: router.scale shape = [hidden_size], BF16.
//
// MLX-VLM `Gemma4SparseMoeBlock.Router.__call__`:
//     effective_weight = self.scale * (self.hidden_size ** -0.5)
//     hidden_normed = mx.fast.rms_norm(hidden, weight=effective_weight, eps=eps)
//     logits = self.proj(hidden_normed)
// We fold the sqrt(hidden)^-1 factor inline rather than precomputing a
// scaled weight buffer (host-side scratch would cost an extra dispatch).
// ============================================================================

kernel void rms_norm_router(
    device const float*    x         [[buffer(0)]],
    device const uint16_t* weight    [[buffer(1)]],   // [dim] bf16 — router.scale
    device float*          out       [[buffer(2)]],
    constant uint&         dim       [[buffer(3)]],
    constant float&        eps       [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float ss[256];
    float local = 0.0f;
    for (uint i = lid; i < dim; i += 256) {
        float v = x[i];
        local += v * v;
    }
    ss[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) ss[lid] += ss[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(ss[0] / float(dim) + eps);
    float global_scale = rsqrt(float(dim));   // sqrt(hidden)^-1 factor
    for (uint i = lid; i < dim; i += 256) {
        out[i] = x[i] * inv_rms * bf16_to_f32(weight[i]) * global_scale;
    }
}

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
    // gelu_pytorch_tanh; use metal::precise::tanh to avoid fast-math
    // NaN behaviour for moderately large inner arguments.
    float g3 = g * g * g;
    float inner = 0.7978845608f * (g + 0.044715f * g3);
    // Clamp inner to safe range; tanh saturates anyway.
    inner = clamp(inner, -20.0f, 20.0f);
    float gelu_g = 0.5f * g * (1.0f + metal::precise::tanh(inner));
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
