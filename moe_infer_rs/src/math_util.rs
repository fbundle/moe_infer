#![allow(dead_code)]
/// Shared types, constants, and CPU helpers used across engine modules.
use crate::constants::RMS_NORM_EPS;
use crate::model::weights::WeightFile;

// ─── bf16 / f32 conversion ───────────────────────────────────────────────

/// Convert bf16 (uint16) to f32.
pub fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

/// CPU reference: 4-bit dequantized matrix-vector multiply.
pub fn dequant_matvec_4bit(
    w_packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) {
    let num_groups = in_dim / group_size;
    let packed_per_group = group_size / 8;
    let packed_cols = in_dim / 8;

    for row in 0..out_dim {
        let mut acc = 0.0f32;
        let w_row = &w_packed[row * packed_cols..];
        let s_row = &scales[row * num_groups..];
        let b_row = &biases[row * num_groups..];

        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let bias = bf16_to_f32(b_row[g]);

            let base_packed = g * packed_per_group;
            let base_x = g * group_size;

            for p in 0..packed_per_group {
                let packed = w_row[base_packed + p];
                let x_base = base_x + p * 8;

                for n in 0..8 {
                    let nibble = (packed >> (n * 4)) & 0xF;
                    let w_val = (nibble as f32) * scale + bias;
                    acc += w_val * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

/// CPU reference: RMS normalization.
pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt().recip();
    for i in 0..dim {
        out[i] = x[i] * rms * weight[i];
    }
}

// ─── CPU helper functions ────────────────────────────────────────────────

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

pub fn softmax(x: &mut [f32]) {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

pub fn topk(scores: &[f32], k: usize, indices: &mut [usize], values: &mut [f32]) {
    for (i, &score) in scores.iter().enumerate() {
        if i < k {
            let mut pos = i;
            while pos > 0 && values[(pos - 1) / 2] > score {
                values[pos] = values[(pos - 1) / 2];
                indices[pos] = indices[(pos - 1) / 2];
                pos = (pos - 1) / 2;
            }
            values[pos] = score;
            indices[pos] = i;
        } else if score > values[0] {
            values[0] = score;
            indices[0] = i;
            let mut pos = 0;
            loop {
                let left = 2 * pos + 1;
                let right = 2 * pos + 2;
                let mut smallest = pos;
                if left < k && values[left] < values[smallest] { smallest = left; }
                if right < k && values[right] < values[smallest] { smallest = right; }
                if smallest == pos { break; }
                values.swap(pos, smallest);
                indices.swap(pos, smallest);
                pos = smallest;
            }
        }
    }
}

pub fn normalize_weights(weights: &mut [f32]) {
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in weights.iter_mut() { *w *= inv; }
    }
}

pub fn rms_norm_bare(x: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms;
    }
}

pub fn rms_norm_gated(
    x: &[f32], z: &[f32], w_bf16: &[u16],
    out: &mut [f32], dim: usize, eps: f32,
) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        let w = bf16_to_f32(w_bf16[i]);
        let silu_z = z[i] / (1.0 + (-z[i]).exp());
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}

pub fn conv1d_step(
    conv_state: &[f32],
    new_input: &[f32],
    weight_bf16: &[u16],
    out: &mut [f32],
    channels: usize,
    kernel_size: usize,
) {
    for c in 0..channels {
        let mut acc = 0.0f32;
        for k in 0..kernel_size - 1 {
            let w = bf16_to_f32(weight_bf16[c * kernel_size + k]);
            acc += conv_state[k * channels + c] * w;
        }
        let w = bf16_to_f32(weight_bf16[c * kernel_size + (kernel_size - 1)]);
        acc += new_input[c] * w;
        out[c] = acc;
    }
    silu(&mut out[..channels]);
}

// ─── RoPE ─────────────────────────────────────────────────────────────────

pub fn apply_rope(
    q: &mut [f32], k: &mut [f32], pos: usize,
    num_q_heads: usize, num_kv_heads: usize,
    head_dim: usize, rotary_dim: usize, rope_theta: f64,
) {
    let pos_f = pos as f32;
    let half = rotary_dim / 2;
    for h in 0..num_q_heads {
        let qh = &mut q[h * head_dim..];
        for i in 0..half {
            let theta = pos_f as f64 * rope_theta.powf(-2.0 * (i as f64) / rotary_dim as f64);
            let cos = theta.cos() as f32;
            let sin = theta.sin() as f32;
            let (q0, q1) = (qh[i], qh[i + half]);
            qh[i] = q0 * cos - q1 * sin;
            qh[i + half] = q0 * sin + q1 * cos;
        }
    }
    for h in 0..num_kv_heads {
        let kh = &mut k[h * head_dim..];
        for i in 0..half {
            let theta = pos_f as f64 * rope_theta.powf(-2.0 * (i as f64) / rotary_dim as f64);
            let cos = theta.cos() as f32;
            let sin = theta.sin() as f32;
            let (k0, k1) = (kh[i], kh[i + half]);
            kh[i] = k0 * cos - k1 * sin;
            kh[i + half] = k0 * sin + k1 * cos;
        }
    }
}

// ─── Token embedding lookup ────────────────────────────────────────────────

pub fn embed_lookup(wf: &WeightFile, token_id: usize, out: &mut [f32], hidden_dim: usize) {
    let (Some(w), Some(s), Some(b)) = (
        wf.get_tensor_u32("model.embed_tokens.weight"),
        wf.get_tensor_u16("model.embed_tokens.scales"),
        wf.get_tensor_u16("model.embed_tokens.biases"),
    ) else {
        out.fill(0.0);
        return;
    };
    let w_info = wf.get_tensor_info("model.embed_tokens.weight").unwrap();
    let packed_cols = w_info.shape[1];
    let s_info = wf.get_tensor_info("model.embed_tokens.scales").unwrap();
    let num_groups = s_info.shape[1];
    let group_size = hidden_dim / num_groups;
    let packed_per_group = group_size / 8;
    let w_row = &w[token_id * packed_cols..];
    let s_row = &s[token_id * num_groups..];
    let b_row = &b[token_id * num_groups..];
    for g in 0..num_groups {
        let scale = bf16_to_f32(s_row[g]);
        let bias = bf16_to_f32(b_row[g]);
        let base = g * group_size;
        for p in 0..packed_per_group {
            let packed = w_row[g * packed_per_group + p];
            for n in 0..8 {
                let nibble = (packed >> (n * 4)) & 0xF;
                out[base + p * 8 + n] = (nibble as f32) * scale + bias;
            }
        }
    }
}

// ─── Final RMS norm ────────────────────────────────────────────────────────

pub fn final_norm(wf: &WeightFile, hidden: &mut [f32], hidden_dim: usize) {
    let Some(fnw_u16) = wf.get_tensor_u16("model.norm.weight") else { return };
    let fnw_f32: Vec<f32> = fnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
    let sum_sq: f32 = hidden[..hidden_dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / hidden_dim as f32 + RMS_NORM_EPS).sqrt();
    for i in 0..hidden_dim {
        hidden[i] *= inv_rms * fnw_f32[i];
    }
}

// ─── LM head (moved to lm_head.rs) ──────────────────────────────────────────

// ─── SwiGLU ─────────────────────────────────────────────────────────────────

/// SwiGLU activation: out[i] = gate[i] * silu(up[i]).
pub fn swiglu_fused(gate: &[f32], up: &[f32], out: &mut [f32], dim: usize) {
    for i in 0..dim {
        let silu_g = gate[i] / (1.0 + (-gate[i]).exp());
        out[i] = silu_g * up[i];
    }
}

// ─── Sigmoid gate (in-place) ─────────────────────────────────────────────────

/// Element-wise sigmoid gating: out[i] *= sigmoid(gate[i]).
pub fn sigmoid_gate_inplace(out: &mut [f32], gate: &[f32], dim: usize) {
    for i in 0..dim {
        let g = 1.0 / (1.0 + (-gate[i]).exp());
        out[i] *= g;
    }
}

// ─── Residual add ─────────────────────────────────────────────────────────────

pub fn residual_add(src: &[f32], dst: &[f32], out: &mut [f32], dim: usize) {
    for i in 0..dim {
        out[i] = src[i] + dst[i];
    }
}

// ─── RMS norm with bf16 weights ───────────────────────────────────────────────

/// RMS norm where the weight tensor is in bf16 (avoids an intermediate f32 conversion).
pub fn rms_norm_bf16(x: &[f32], weight_bf16: &[u16], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms * bf16_to_f32(weight_bf16[i]);
    }
}

/// RMS norm (in-place) with bf16 weights.
pub fn rms_norm_bf16_inplace(x: &mut [f32], weight_bf16: &[u16], dim: usize, eps: f32) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        x[i] = x[i] * inv_rms * bf16_to_f32(weight_bf16[i]);
    }
}

// ─── Convenience: dequant matvec with tensor lookup ──────────────────────────

/// Look up a weight/scales/biases tensor triple by prefix, then run dequant matvec.
/// Returns false if any tensor is missing.
pub fn matvec_lookup(
    wf: &WeightFile, prefix: &str, x: &[f32], out: &mut [f32],
    out_dim: usize, in_dim: usize, group_size: usize,
) -> bool {
    let (Some(w), Some(s), Some(b)) = (
        wf.get_tensor_u32(&format!("{}.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.biases", prefix)),
    ) else {
        return false;
    };
    dequant_matvec_4bit(w, s, b, x, out, out_dim, in_dim, group_size);
    true
}

// ─── Full attention ──────────────────────────────────────────────────────────

/// Batched Q*K^T scores for all query heads at current position.
/// Q: [num_q_heads * head_dim], K_cache: [max_seq * kv_dim]
/// scores: [num_q_heads * max_seq]
pub fn attention_scores_batched(
    q: &[f32], k_cache: &[f32], scores: &mut [f32],
    num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
    kv_dim: usize, seq_len: usize, max_seq: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = num_q_heads / num_kv_heads;
    for qh in 0..num_q_heads {
        let kh = qh / n_rep;
        let q_head = &q[qh * head_dim..(qh + 1) * head_dim];
        let k_head_base = kh * head_dim;
        let scores_row = &mut scores[qh * max_seq..];
        for t in 0..seq_len {
            let k_t = &k_cache[t * kv_dim + k_head_base..t * kv_dim + k_head_base + head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim { dot += q_head[d] * k_t[d]; }
            scores_row[t] = dot * scale;
        }
    }
}

/// Softmax over the sequence dimension for each head independently.
/// scores: [num_q_heads * max_seq], softmax applied over first seq_len entries per head.
pub fn attention_softmax_batched(scores: &mut [f32], num_q_heads: usize, seq_len: usize, max_seq: usize) {
    for h in 0..num_q_heads {
        let s = &mut scores[h * max_seq..h * max_seq + seq_len];
        let max_val = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for v in s.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in s.iter_mut() { *v *= inv; }
        }
    }
}

/// Weighted V sum from attention scores for all query heads.
/// scores: [num_q_heads * max_seq], V_cache: [max_seq * kv_dim]
/// out: [num_q_heads * head_dim]
pub fn attention_values_batched(
    scores: &[f32], v_cache: &[f32], out: &mut [f32],
    num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
    kv_dim: usize, seq_len: usize, max_seq: usize,
) {
    let n_rep = num_q_heads / num_kv_heads;
    out[..num_q_heads * head_dim].fill(0.0);
    for qh in 0..num_q_heads {
        let kh = qh / n_rep;
        let scores_row = &scores[qh * max_seq..qh * max_seq + seq_len];
        let v_head_base = kh * head_dim;
        let out_row = &mut out[qh * head_dim..(qh + 1) * head_dim];
        for t in 0..seq_len {
            let s = scores_row[t];
            let v_t = &v_cache[t * kv_dim + v_head_base..t * kv_dim + v_head_base + head_dim];
            for d in 0..head_dim { out_row[d] += s * v_t[d]; }
        }
    }
}

// ─── KV cache append (CPU-side) ──────────────────────────────────────────────

pub fn kv_cache_append(k: &[f32], v: &[f32], k_cache: &mut [f32], v_cache: &mut [f32],
                        pos: usize, kv_dim: usize) {
    let dst = pos * kv_dim;
    k_cache[dst..dst + kv_dim].copy_from_slice(&k[..kv_dim]);
    v_cache[dst..dst + kv_dim].copy_from_slice(&v[..kv_dim]);
}

// ─── Gated DeltaNet helpers ──────────────────────────────────────────────────

/// Compute per-v-head decay and beta gate from alpha/beta projections.
/// alpha: [num_v_heads], beta: [num_v_heads]
/// a_log: [num_v_heads] from model weights (log of A), dt_bias: [num_v_heads] bf16
/// g_decay: [num_v_heads] output, beta_gate_out: [num_v_heads] output
/// Matches GPU kernel `compute_decay_beta`.
pub fn compute_decay_beta(
    alpha: &[f32], beta: &[f32],
    a_log: &[f32], dt_bias: &[u16],
    g_decay: &mut [f32], beta_gate_out: &mut [f32],
    num_v_heads: usize,
) {
    for h in 0..num_v_heads {
        let a_val = alpha[h];
        let dt_b = bf16_to_f32(dt_bias[h]);
        let a = a_log[h].exp();
        let softplus = (1.0 + (a_val + dt_b).exp()).ln();
        g_decay[h] = (-a * softplus).exp();
        beta_gate_out[h] = 1.0 / (1.0 + (-beta[h]).exp());
    }
}

/// Gated DeltaNet SSM recurrence step (single token, all heads).
/// Matches GPU kernel `gated_delta_net_step`.
///
/// state: [num_v_heads * value_dim * key_dim] persistent, in/out
/// q: [num_k_heads * key_dim], k: [num_k_heads * key_dim]
/// v: [num_v_heads * value_dim]
/// g_decay, beta_gate: [num_v_heads]
/// output: [num_v_heads * value_dim]
pub fn gated_delta_net_step(
    state: &mut [f32],
    q: &[f32], k: &[f32], v: &[f32],
    g_decay: &[f32], beta_gate: &[f32],
    output: &mut [f32],
    num_v_heads: usize, k_heads_per_v: usize,
    key_dim: usize, value_dim: usize,
) {
    for h in 0..num_v_heads {
        let kh = h / k_heads_per_v;
        let g = g_decay[h];
        let beta = beta_gate[h];
        let state_base = h * value_dim * key_dim;
        let k_base = kh * key_dim;
        let v_base = h * value_dim;

        for vi in 0..value_dim {
            // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
            let row_start = state_base + vi * key_dim;
            let mut kv_mem = 0.0f32;
            for ki in 0..key_dim {
                let s = state[row_start + ki] * g;
                state[row_start + ki] = s;
                kv_mem += s * k[k_base + ki];
            }

            // Step 3+4: Delta update: S[vi][ki] += k[ki] * delta
            let delta = (v[v_base + vi] - kv_mem) * beta;
            for ki in 0..key_dim {
                state[row_start + ki] += k[k_base + ki] * delta;
            }

            // Step 5: Output = dot(S[vi][:], q[:])
            let mut out_val = 0.0f32;
            for ki in 0..key_dim {
                out_val += state[row_start + ki] * q[k_base + ki];
            }
            output[v_base + vi] = out_val;
        }
    }
}

// ─── Gated RMS norm (linear attention output) ────────────────────────────────

/// Gated RMS norm for delta net output. Per v-head: normalize values,
/// gate with z via SiLU, scale with weight.
/// Matches GPU kernel `gated_rms_norm`.
pub fn gated_rms_norm(
    values: &[f32], z: &[f32], weight_bf16: &[u16],
    output: &mut [f32],
    num_heads: usize, value_dim: usize, eps: f32,
) {
    for h in 0..num_heads {
        let base = h * value_dim;

        // RMS norm: compute sum of squares
        let sum_sq: f32 = values[base..base + value_dim].iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / value_dim as f32 + eps).sqrt();

        for d in 0..value_dim {
            let normed = values[base + d] * inv_rms;
            let zval = z[base + d];
            let gate = zval / (1.0 + (-zval).exp()); // SiLU
            let w = bf16_to_f32(weight_bf16[d]);
            output[base + d] = normed * gate * w;
        }
    }
}

// ─── Q/K head norm + RoPE ────────────────────────────────────────────────────

/// Apply per-head RMS norm and RoPE to Q projection.
/// q_proj: [num_q_heads * 2 * head_dim] (interleaved Q and Q-gate)
/// q_norm_w: [head_dim] bf16, shared across heads
/// q_out: [num_q_heads * head_dim], q_gate_out: [num_q_heads * head_dim]
/// Matches GPU kernel `q_head_norm_rope`.
pub fn q_head_norm_rope(
    q_proj: &[f32], q_norm_w: &[u16],
    q_out: &mut [f32], q_gate_out: &mut [f32],
    num_q_heads: usize, head_dim: usize,
    rotary_dim: usize, rope_theta: f32, pos: usize, eps: f32,
) {
    let rot_half = rotary_dim / 2;
    for h in 0..num_q_heads {
        let src_base = h * 2 * head_dim;
        let out_base = h * head_dim;

        // Split Q and Q-gate
        for d in 0..head_dim {
            q_gate_out[out_base + d] = q_proj[src_base + head_dim + d];
        }

        // RMS norm per head
        let sum_sq: f32 = (0..head_dim).map(|d| {
            let v = q_proj[src_base + d]; v * v
        }).sum();
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();

        // Norm + RoPE
        for d in 0..head_dim {
            let q_val = q_proj[src_base + d] * inv_rms * bf16_to_f32(q_norm_w[d]);
            if d < rotary_dim {
                let (cos_t, sin_t, pair_val) = if d < rot_half {
                    let theta = (pos as f32) * rope_theta.powf(-2.0 * (d as f32) / (rotary_dim as f32));
                    let pair = q_proj[src_base + d + rot_half] * inv_rms * bf16_to_f32(q_norm_w[d + rot_half]);
                    (theta.cos(), theta.sin(), pair)
                } else {
                    let pair = d - rot_half;
                    let theta = (pos as f32) * rope_theta.powf(-2.0 * (pair as f32) / (rotary_dim as f32));
                    let pair_val = q_proj[src_base + pair] * inv_rms * bf16_to_f32(q_norm_w[pair]);
                    (theta.cos(), theta.sin(), pair_val)
                };
                if d < rot_half {
                    q_out[out_base + d] = q_val * cos_t - pair_val * sin_t;
                } else {
                    q_out[out_base + d] = pair_val * sin_t + q_val * cos_t;
                }
            } else {
                q_out[out_base + d] = q_val;
            }
        }
    }
}

/// Apply per-head RMS norm and RoPE to K projection (in-place on k_buf).
/// k_buf: [num_kv_heads * head_dim] in/out
/// k_norm_w: [head_dim] bf16, shared across heads
/// Matches GPU kernel `k_head_norm_rope`.
pub fn k_head_norm_rope(
    k_buf: &mut [f32], k_norm_w: &[u16],
    num_kv_heads: usize, head_dim: usize,
    rotary_dim: usize, rope_theta: f32, pos: usize, eps: f32,
) {
    let rot_half = rotary_dim / 2;
    for h in 0..num_kv_heads {
        let base = h * head_dim;

        let sum_sq: f32 = (0..head_dim).map(|d| {
            let v = k_buf[base + d]; v * v
        }).sum();
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();

        // Need to save original values for RoPE pair computation
        let orig: Vec<f32> = (0..head_dim).map(|d| k_buf[base + d]).collect();

        for d in 0..head_dim {
            let k_val = orig[d] * inv_rms * bf16_to_f32(k_norm_w[d]);
            if d < rotary_dim {
                let (cos_t, sin_t, pair_val) = if d < rot_half {
                    let theta = (pos as f32) * rope_theta.powf(-2.0 * (d as f32) / (rotary_dim as f32));
                    let pair = orig[d + rot_half] * inv_rms * bf16_to_f32(k_norm_w[d + rot_half]);
                    (theta.cos(), theta.sin(), pair)
                } else {
                    let pair = d - rot_half;
                    let theta = (pos as f32) * rope_theta.powf(-2.0 * (pair as f32) / (rotary_dim as f32));
                    let pair_val = orig[pair] * inv_rms * bf16_to_f32(k_norm_w[pair]);
                    (theta.cos(), theta.sin(), pair_val)
                };
                if d < rot_half {
                    k_buf[base + d] = k_val * cos_t - pair_val * sin_t;
                } else {
                    k_buf[base + d] = pair_val * sin_t + k_val * cos_t;
                }
            } else {
                k_buf[base + d] = k_val;
            }
        }
    }
}

// ─── MoE combine + residual ──────────────────────────────────────────────────

/// Weighted combine of expert outputs + shared expert + residual.
/// Matches GPU kernel `moe_combine_residual`.
pub fn moe_combine_residual(
    h_mid: &[f32], shared_out: &[f32], hidden_out: &mut [f32],
    expert_outs: &[&[f32]], expert_weights: &[f32],
    shared_gate_score: f32, dim: usize, k: usize,
) {
    let shared_gate = sigmoid(shared_gate_score);
    for i in 0..dim {
        let mut moe_sum = 0.0f32;
        for ki in 0..k {
            moe_sum += expert_weights[ki] * expert_outs[ki][i];
        }
        hidden_out[i] = h_mid[i] + moe_sum + shared_gate * shared_out[i];
    }
}
