/// CPU-only engine: no GPU resources, all dequant matvecs on CPU.

use crate::cache::Cache;
use crate::constants::{CONV_KERNEL_SIZE, FULL_ATTN_INTERVAL, GROUP_SIZE, MAX_SEQ, RMS_NORM_EPS};
use crate::engine::Engine;
use crate::error::MoEError;
use crate::model::Model;
use crate::engine::SignalCheckFn;
use crate::math::{
    apply_rope, bf16_to_f32, conv1d_step, dequant_matvec_4bit,
    normalize_weights, rms_norm, rms_norm_bare, rms_norm_gated, sigmoid,
    softmax, topk,
};

// ─── Execution context ─────────────────────────────────────────────────────

struct ExecCtx<'a> {
    model: &'a Model,
    cache: &'a mut Cache,
}

impl<'a> ExecCtx<'a> {
    // ── Embedding ──────────────────────────────────────────────────────────

    fn embed(&self, token_id: usize, out: &mut [f32]) {
        let hd = self.model.config.hidden_dim;
        let (Some(w), Some(s), Some(b)) = (
            self.model.wf.get_tensor_u32("model.embed_tokens.weight"),
            self.model.wf.get_tensor_u16("model.embed_tokens.scales"),
            self.model.wf.get_tensor_u16("model.embed_tokens.biases"),
        ) else { out.fill(0.0); return };
        let w_info = self.model.wf.get_tensor_info("model.embed_tokens.weight").unwrap();
        let packed_cols = w_info.shape[1];
        let s_info = self.model.wf.get_tensor_info("model.embed_tokens.scales").unwrap();
        let num_groups = s_info.shape[1];
        let group_size = hd / num_groups;
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

    // ── Input RMS norm ─────────────────────────────────────────────────────

    fn input_norm(&self, layer_idx: usize, hidden: &[f32], out: &mut [f32]) {
        let hd = self.model.config.hidden_dim;
        let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        if let Some(nw_u16) = self.model.wf.get_tensor_u16(&norm_name) {
            let nw: Vec<f32> = nw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
            rms_norm(hidden, &nw, out, hd, RMS_NORM_EPS);
        } else {
            out.copy_from_slice(hidden);
        }
    }

    // ── Post-attention RMS norm ────────────────────────────────────────────

    fn post_norm(&self, layer_idx: usize, hidden: &[f32], out: &mut [f32]) {
        let hd = self.model.config.hidden_dim;
        let name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
        if let Some(pnw_u16) = self.model.wf.get_tensor_u16(&name) {
            let pnw: Vec<f32> = pnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
            rms_norm(hidden, &pnw, out, hd, RMS_NORM_EPS);
        } else {
            out.copy_from_slice(hidden);
        }
    }

    // ── Full (self) attention ──────────────────────────────────────────────

    /// CPU-only full attention: QKV dequant, Q/K norms, RoPE, KV cache,
    /// attention scores + softmax + weighted sum, sigmoid gate, o_proj, residual.
    fn full_attention(
        &mut self, layer_idx: usize,
        hidden: &mut [f32], residual: &[f32], pos: usize, normed: &[f32],
    ) {
        let hd = self.model.config.hidden_dim;
        let num_q = self.model.config.num_attn_heads;
        let num_kv = self.model.config.num_kv_heads;
        let head_dim = self.model.config.head_dim;
        let q_dim = num_q * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv * head_dim;

        // QKV projections
        let mut q_proj = vec![0.0f32; q_proj_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let prefix = format!("model.layers.{}.self_attn", layer_idx);
        if let (Some(qw), Some(qs), Some(qb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.q_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.q_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.q_proj.biases", prefix)),
        ) { dequant_matvec_4bit(qw, qs, qb, &normed, &mut q_proj, q_proj_dim, hd, GROUP_SIZE); }
        if let (Some(kw), Some(ks), Some(kb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.k_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.k_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.k_proj.biases", prefix)),
        ) { dequant_matvec_4bit(kw, ks, kb, &normed, &mut k, kv_dim, hd, GROUP_SIZE); }
        if let (Some(vw), Some(vs), Some(vb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.v_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.v_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.v_proj.biases", prefix)),
        ) { dequant_matvec_4bit(vw, vs, vb, &normed, &mut v, kv_dim, hd, GROUP_SIZE); }

        // Split Q / Q-gate
        let mut q = vec![0.0f32; q_dim];
        let mut q_gate = vec![0.0f32; q_dim];
        for h in 0..num_q {
            let src = &q_proj[h * 2 * head_dim..];
            q[h * head_dim..][..head_dim].copy_from_slice(&src[..head_dim]);
            q_gate[h * head_dim..][..head_dim].copy_from_slice(&src[head_dim..2 * head_dim]);
        }

        // Q/K per-head norms
        let qn_name = format!("{}.q_norm.weight", prefix);
        let kn_name = format!("{}.k_norm.weight", prefix);
        if let Some(qnw) = self.model.wf.get_tensor_u16(&qn_name) {
            for h in 0..num_q {
                let qh = &mut q[h * head_dim..(h + 1) * head_dim];
                let ssq: f32 = qh.iter().map(|&x| x * x).sum();
                let inv = 1.0 / (ssq / head_dim as f32 + RMS_NORM_EPS).sqrt();
                for i in 0..head_dim.min(qnw.len()) { qh[i] *= inv * bf16_to_f32(qnw[i]); }
            }
        }
        if let Some(knw) = self.model.wf.get_tensor_u16(&kn_name) {
            for h in 0..num_kv {
                let kh = &mut k[h * head_dim..(h + 1) * head_dim];
                let ssq: f32 = kh.iter().map(|&x| x * x).sum();
                let inv = 1.0 / (ssq / head_dim as f32 + RMS_NORM_EPS).sqrt();
                for i in 0..head_dim.min(knw.len()) { kh[i] *= inv * bf16_to_f32(knw[i]); }
            }
        }

        // RoPE
        apply_rope(&mut q, &mut k, pos, num_q, num_kv, head_dim,
            self.model.config.rotary_dim, self.model.config.rope_theta);

        // Append K, V to cache
        let kv_cache = self.cache.kv[layer_idx].as_mut().unwrap();
        let cache_pos = kv_cache.len;
        assert!(cache_pos < MAX_SEQ, "sequence length {} exceeds MAX_SEQ ({})", cache_pos, MAX_SEQ);
        kv_cache.k_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim].copy_from_slice(&k);
        kv_cache.v_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim].copy_from_slice(&v);
        kv_cache.len += 1;
        let seq_len = kv_cache.len;

        // Attention scores + softmax + weighted sum (CPU)
        let heads_per_kv = num_q / num_kv;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; q_dim];
        for h in 0..num_q {
            let kv_h = h / heads_per_kv;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0.0f32; seq_len];
            for p in 0..seq_len {
                let kp = &kv_cache.k_cache[p * kv_dim + kv_h * head_dim..];
                scores[p] = qh.iter().zip(kp.iter().take(head_dim))
                    .map(|(&a, &b)| a * b).sum::<f32>() * scale;
            }
            let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = scores.iter().map(|&s| (s - max_val).exp()).sum();
            let inv_sum = 1.0 / sum;
            let oh = &mut attn_out[h * head_dim..(h + 1) * head_dim];
            for p in 0..seq_len {
                let weight = (scores[p] - max_val).exp() * inv_sum;
                let vp = &kv_cache.v_cache[p * kv_dim + kv_h * head_dim..];
                for d in 0..head_dim { oh[d] += weight * vp[d]; }
            }
        }

        // Sigmoid gate
        for i in 0..q_dim { attn_out[i] /= 1.0 + (-q_gate[i]).exp(); }

        // o_proj
        let o_name = format!("{}.o_proj", prefix);
        let mut o_out = vec![0.0f32; hd];
        if let (Some(ow), Some(os), Some(ob)) = (
            self.model.wf.get_tensor_u32(&format!("{}.weight", o_name)),
            self.model.wf.get_tensor_u16(&format!("{}.scales", o_name)),
            self.model.wf.get_tensor_u16(&format!("{}.biases", o_name)),
        ) { dequant_matvec_4bit(ow, os, ob, &attn_out, &mut o_out, hd, q_dim, GROUP_SIZE); }

        // Residual add
        for i in 0..hd { hidden[i] = residual[i] + o_out[i]; }
    }

    // ── Linear attention (GatedDeltaNet) ────────────────────────────────────

    fn linear_attention(
        &mut self, layer_idx: usize,
        hidden: &mut [f32], normed: &[f32], residual: &[f32],
    ) {
        let hd = self.model.config.hidden_dim;
        let n_k = self.model.config.linear_num_k_heads;
        let n_v = self.model.config.linear_num_v_heads;
        let total_key = self.model.config.linear_total_key;
        let total_value = self.model.config.linear_total_value;
        let qkv_dim = self.model.config.linear_conv_dim;
        let key_dim = total_key / n_k;
        let value_dim = total_value / n_v;
        let inv_scale = 1.0 / (key_dim as f32).sqrt();
        let k_heads_per_v = n_v / n_k;

        let prefix = format!("model.layers.{}.linear_attn", layer_idx);

        let mut qkv = vec![0.0f32; qkv_dim];
        let mut z = vec![0.0f32; total_value];
        let mut beta = vec![0.0f32; n_v];
        let mut alpha = vec![0.0f32; n_v];

        if let (Some(qw), Some(qs), Some(qb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.in_proj_qkv.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_qkv.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_qkv.biases", prefix)),
        ) { dequant_matvec_4bit(qw, qs, qb, normed, &mut qkv, qkv_dim, hd, GROUP_SIZE); }
        if let (Some(zw), Some(zs), Some(zb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.in_proj_z.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_z.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_z.biases", prefix)),
        ) { dequant_matvec_4bit(zw, zs, zb, normed, &mut z, total_value, hd, GROUP_SIZE); }
        if let (Some(bw), Some(bs), Some(bb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.in_proj_b.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_b.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_b.biases", prefix)),
        ) { dequant_matvec_4bit(bw, bs, bb, normed, &mut beta, n_v, hd, GROUP_SIZE); }
        if let (Some(aw), Some(ass), Some(ab)) = (
            self.model.wf.get_tensor_u32(&format!("{}.in_proj_a.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_a.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_a.biases", prefix)),
        ) { dequant_matvec_4bit(aw, ass, ab, normed, &mut alpha, n_v, hd, GROUP_SIZE); }

        // Conv1d step
        let mut conv_out = vec![0.0f32; qkv_dim];
        let state = self.cache.lin[layer_idx].as_mut().unwrap();
        if let Some(conv_w) = self.model.wf.get_tensor_u16(&format!("{}.conv1d.weight", prefix)) {
            conv1d_step(&state.conv_state, &qkv, conv_w, &mut conv_out, qkv_dim, CONV_KERNEL_SIZE);
        } else {
            conv_out.copy_from_slice(&qkv);
        }
        let shift = (CONV_KERNEL_SIZE - 2) * qkv_dim;
        state.conv_state.copy_within(qkv_dim.., 0);
        state.conv_state[shift..shift + qkv_dim].copy_from_slice(&qkv);

        let lq = &conv_out[..total_key];
        let lk = &conv_out[total_key..2 * total_key];
        let lv = &conv_out[2 * total_key..];

        // Q/K norms (CPU)
        let mut q_normed = vec![0.0f32; total_key];
        let mut k_normed = vec![0.0f32; total_key];
        for h in 0..n_k {
            let qh = &lq[h * key_dim..(h + 1) * key_dim];
            rms_norm_bare(qh, &mut q_normed[h * key_dim..(h + 1) * key_dim], key_dim, 1e-6);
            for d in &mut q_normed[h * key_dim..(h + 1) * key_dim] { *d *= inv_scale * inv_scale; }
        }
        for h in 0..n_k {
            let kh = &lk[h * key_dim..(h + 1) * key_dim];
            rms_norm_bare(kh, &mut k_normed[h * key_dim..(h + 1) * key_dim], key_dim, 1e-6);
            for d in &mut k_normed[h * key_dim..(h + 1) * key_dim] { *d *= inv_scale; }
        }

        let a_log = self.model.wf.get_tensor_f32(&format!("{}.A_log", prefix));
        let dt_bias = self.model.wf.get_tensor_u16(&format!("{}.dt_bias", prefix));

        let mut out_vals = vec![0.0f32; total_value];
        for vh in 0..n_v {
            let kh = vh / k_heads_per_v;
            let a_val = a_log.map_or(1.0, |al| al[vh]);
            let dt_b = dt_bias.map_or(0.0, |db| bf16_to_f32(db[vh]));
            let sp = (1.0 + (alpha[vh] + dt_b).exp()).ln();
            let g_decay = (-a_val.exp() * sp).exp();
            let beta_gate = sigmoid(beta[vh]);
            let so = vh * value_dim * key_dim;
            let ssm = &mut state.ssm_state[so..so + value_dim * key_dim];
            let v_h = &lv[vh * value_dim..(vh + 1) * value_dim];
            let k_h = &k_normed[kh * key_dim..(kh + 1) * key_dim];
            for vi in 0..value_dim {
                for ki in 0..key_dim { ssm[vi * key_dim + ki] *= g_decay; }
            }
            for vi in 0..value_dim {
                let kv_mem: f32 = (0..key_dim).map(|ki| ssm[vi * key_dim + ki] * k_h[ki]).sum();
                let delta = (v_h[vi] - kv_mem) * beta_gate;
                for ki in 0..key_dim { ssm[vi * key_dim + ki] += k_h[ki] * delta; }
            }
            let q_h = &q_normed[kh * key_dim..(kh + 1) * key_dim];
            let oh = &mut out_vals[vh * value_dim..(vh + 1) * value_dim];
            for vi in 0..value_dim {
                oh[vi] = (0..key_dim).map(|ki| ssm[vi * key_dim + ki] * q_h[ki]).sum();
            }
        }

        // RMSNormGated
        let mut gated_out = vec![0.0f32; total_value];
        if let Some(gnw) = self.model.wf.get_tensor_u16(&format!("{}.norm.weight", prefix)) {
            for vh in 0..n_v {
                let oh = &out_vals[vh * value_dim..(vh + 1) * value_dim];
                let zh = &z[vh * value_dim..(vh + 1) * value_dim];
                rms_norm_gated(oh, zh, gnw, &mut gated_out[vh * value_dim..(vh + 1) * value_dim], value_dim, RMS_NORM_EPS);
            }
        } else {
            gated_out.copy_from_slice(&out_vals);
        }

        // Output projection
        let mut attn_out = vec![0.0f32; hd];
        if let (Some(ow), Some(os), Some(ob)) = (
            self.model.wf.get_tensor_u32(&format!("{}.out_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.out_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.out_proj.biases", prefix)),
        ) { dequant_matvec_4bit(ow, os, ob, &gated_out, &mut attn_out, hd, total_value, GROUP_SIZE); }

        // Residual add
        for i in 0..hd { hidden[i] = residual[i] + attn_out[i]; }
    }

    // ── MoE layer ──────────────────────────────────────────────────────────

    fn moe_layer(&mut self, layer_idx: usize, hidden: &mut [f32], h_post: &[f32]) {
        let hd = self.model.config.hidden_dim;
        let n_experts = self.model.config.num_experts;
        let moe_inter = self.model.config.moe_intermediate;
        let shared_inter = self.model.config.shared_intermediate;
        let k = self.model.config.num_experts_per_tok;

        // h_mid = input hidden (residual)
        let h_mid = hidden.to_vec();

        // Router gate + shared expert projections
        let prefix = format!("model.layers.{}.mlp", layer_idx);
        let mut gate_scores = vec![0.0f32; n_experts];
        let mut shared_gate = vec![0.0f32; shared_inter];
        let mut shared_up = vec![0.0f32; shared_inter];
        let mut shared_gate_score = 0.0f32;

        if let (Some(gw), Some(gs), Some(gb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.gate.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.gate.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.gate.biases", prefix)),
        ) { dequant_matvec_4bit(gw, gs, gb, &h_post, &mut gate_scores, n_experts, hd, GROUP_SIZE); }
        if let (Some(sgw), Some(sgs), Some(sgb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.shared_expert.gate_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.biases", prefix)),
        ) { dequant_matvec_4bit(sgw, sgs, sgb, &h_post, &mut shared_gate, shared_inter, hd, GROUP_SIZE); }
        if let (Some(suw), Some(sus), Some(sub)) = (
            self.model.wf.get_tensor_u32(&format!("{}.shared_expert.up_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.biases", prefix)),
        ) { dequant_matvec_4bit(suw, sus, sub, &h_post, &mut shared_up, shared_inter, hd, GROUP_SIZE); }
        if let (Some(segw), Some(segs), Some(segb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.shared_expert_gate.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert_gate.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert_gate.biases", prefix)),
        ) {
            let mut tmp = [0.0f32];
            dequant_matvec_4bit(segw, segs, segb, &h_post, &mut tmp, 1, hd, GROUP_SIZE);
            shared_gate_score = tmp[0];
        }

        // Routing: softmax + topk
        softmax(&mut gate_scores);
        let mut expert_indices = vec![0usize; k];
        let mut expert_weights = vec![0.0f32; k];
        topk(&gate_scores, k, &mut expert_indices, &mut expert_weights);
        normalize_weights(&mut expert_weights);

        // Expert computation (CPU: read from fd, dequant, SwiGLU, combine)
        let mut moe_out = vec![0.0f32; hd];
        let expert_size = self.model.config.expert_size_4bit;
        let mut expert_data = vec![0u8; expert_size];
        let mut gate_tmp = vec![0.0f32; moe_inter];
        let mut up_tmp = vec![0.0f32; moe_inter];
        let mut act_tmp = vec![0.0f32; moe_inter];
        let mut eout = vec![0.0f32; hd];

        {
            let layout = &self.model.config.expert_layout_4bit;
            let expert_file = &self.model.expert_files[layer_idx];

            for (&eidx, &ew) in expert_indices.iter().zip(expert_weights.iter()) {
                if expert_file.read_expert(eidx, &mut expert_data).is_err() { continue; }

                let gw = &expert_data[layout.gate_w_off..];
                let gs = &expert_data[layout.gate_s_off..];
                let gb = &expert_data[layout.gate_b_off..];
                let uw = &expert_data[layout.up_w_off..];
                let us = &expert_data[layout.up_s_off..];
                let ub = &expert_data[layout.up_b_off..];
                let dw = &expert_data[layout.down_w_off..];
                let ds = &expert_data[layout.down_s_off..];
                let db = &expert_data[layout.down_b_off..];

                dequant_matvec_4bit(
                    unsafe { std::slice::from_raw_parts(gw.as_ptr() as *const u32, layout.gate_w_size / 4) },
                    unsafe { std::slice::from_raw_parts(gs.as_ptr() as *const u16, layout.gate_s_size / 2) },
                    unsafe { std::slice::from_raw_parts(gb.as_ptr() as *const u16, layout.gate_b_size / 2) },
                    &h_post, &mut gate_tmp, moe_inter, hd, GROUP_SIZE);
                dequant_matvec_4bit(
                    unsafe { std::slice::from_raw_parts(uw.as_ptr() as *const u32, layout.up_w_size / 4) },
                    unsafe { std::slice::from_raw_parts(us.as_ptr() as *const u16, layout.up_s_size / 2) },
                    unsafe { std::slice::from_raw_parts(ub.as_ptr() as *const u16, layout.up_b_size / 2) },
                    &h_post, &mut up_tmp, moe_inter, hd, GROUP_SIZE);

                for i in 0..moe_inter {
                    let g = gate_tmp[i];
                    act_tmp[i] = (g / (1.0 + (-g).exp())) * up_tmp[i];
                }

                dequant_matvec_4bit(
                    unsafe { std::slice::from_raw_parts(dw.as_ptr() as *const u32, layout.down_w_size / 4) },
                    unsafe { std::slice::from_raw_parts(ds.as_ptr() as *const u16, layout.down_s_size / 2) },
                    unsafe { std::slice::from_raw_parts(db.as_ptr() as *const u16, layout.down_b_size / 2) },
                    &act_tmp, &mut eout, hd, moe_inter, GROUP_SIZE);

                for d in 0..hd { moe_out[d] += eout[d] * ew; }
            }
        }

        // Shared expert SwiGLU + down_proj
        let mut shared_act = vec![0.0f32; shared_inter];
        for i in 0..shared_inter {
            let g = shared_gate[i];
            shared_act[i] = (g / (1.0 + (-g).exp())) * shared_up[i];
        }
        let mut shared_out = vec![0.0f32; hd];
        if let (Some(sdw), Some(sds), Some(sdb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.shared_expert.down_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.biases", prefix)),
        ) { dequant_matvec_4bit(sdw, sds, sdb, &shared_act, &mut shared_out, hd, shared_inter, GROUP_SIZE); }

        let shared_weight = sigmoid(shared_gate_score);

        // Final combine: hidden = h_mid + moe_out + shared_weight * shared_out
        for i in 0..hd {
            hidden[i] = h_mid[i] + moe_out[i] + shared_weight * shared_out[i];
        }
    }

    // ── Final norm + LM head ───────────────────────────────────────────────

    fn final_norm_and_lm_head(&self, hidden: &mut [f32], logits: &mut [f32]) {
        let hd = self.model.config.hidden_dim;
        let vs = self.model.config.vocab_size;

        // Final RMS norm
        if let Some(fnw_u16) = self.model.wf.get_tensor_u16("model.norm.weight") {
            let fnw: Vec<f32> = fnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
            let ssq: f32 = hidden[..hd].iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (ssq / hd as f32 + RMS_NORM_EPS).sqrt();
            for i in 0..hd { hidden[i] *= inv_rms * fnw[i]; }
        }

        // LM head (vocab projection)
        if let (Some(w), Some(s), Some(b)) = (
            self.model.wf.get_tensor_u32("lm_head.weight"),
            self.model.wf.get_tensor_u16("lm_head.scales"),
            self.model.wf.get_tensor_u16("lm_head.biases"),
        ) {
            dequant_matvec_4bit(w, s, b, hidden, logits, vs, hd, GROUP_SIZE);
        }
    }
}

// ─── EngineCPU ─────────────────────────────────────────────────────────────

pub struct EngineCPU<'a> {
    pub model: &'a Model,
}

impl<'a> Engine for EngineCPU<'a> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        let hd = self.model.config.hidden_dim;
        let vs = self.model.config.vocab_size;
        let num_layers = self.model.config.num_layers;
        let n = input_ids.len();
        if n == 0 { return Ok(vec![]); }

        let mut logits = vec![0.0f32; n * vs];
        let mut exec = ExecCtx { model: self.model, cache };
        let mut hidden = vec![0.0f32; hd];

        for (ti, &id) in input_ids.iter().enumerate() {
            exec.embed(id as usize, &mut hidden);

            for layer in 0..num_layers {
                if layer % 4 == 0 && check_signal() {
                    return Err(MoEError::Metal("interrupted".into()));
                }
                let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;
                let residual = hidden.to_vec();
                let mut normed = vec![0.0f32; hd];
                exec.input_norm(layer, &hidden, &mut normed);

                let cache_pos = exec.cache.pos;
                if is_full {
                    exec.full_attention(layer, &mut hidden, &residual, cache_pos, &normed);
                } else {
                    exec.linear_attention(layer, &mut hidden, &normed, &residual);
                }

                let mut h_post = vec![0.0f32; hd];
                exec.post_norm(layer, &hidden, &mut h_post);
                exec.moe_layer(layer, &mut hidden, &h_post);
            }

            exec.cache.pos += 1;
            exec.final_norm_and_lm_head(&mut hidden, &mut logits[ti * vs..(ti + 1) * vs]);
        }

        Ok(logits)
    }
}
