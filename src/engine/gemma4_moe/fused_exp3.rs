//! Gemma 4 26B-A4B engine struct and Engine trait impl.
//!
//! Status: skeleton. The struct is shaped, the trait method signatures are
//! in place, and the bodies are stubbed with `unimplemented!()` carrying
//! precise TODO comments. Once filled in this engine selects via
//! `pipeline_mode = "Gemma4MoE"` in DynEngine.
//!
//! Implementation plan (phased):
//!
//!   Phase 1 (current): scaffolding compiles, registers nothing in
//!     DynEngine (no risk to production engines).
//!   Phase 2: forward_hidden for one token end-to-end. Requires kernel set
//!     in shaders.metal + dispatchers in metal_kernels.rs.
//!   Phase 3: KV cache shapes for sliding vs full attention.
//!   Phase 4: validation vs HF transformers on a stripped Gemma 4 model.
//!   Phase 5: batched-prefill (after sequential works).

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::cache::Cache;
use crate::engine::{Engine, SignalCheckFn, TelemetryValue};
use crate::error::MoEError;
use crate::math::bf16_to_f32;
use crate::model::Model;
use crate::model::weights::WeightFile;
use crate::constants::RMS_NORM_EPS;

use crate::engine::gemma4_constants::Gemma4ModelConfig;
use crate::engine::gemma4_metal_context::Gemma4MetalContext;

pub struct Gemma4Fused<C: Gemma4ModelConfig> {
    pub model: Arc<Model>,
    pub ctx: Gemma4MetalContext,
    pub num_active_experts: usize,
    pub timing: BTreeMap<String, TelemetryValue>,
    pub last_h_pre_norm: Vec<f32>,
    pub weight_buffer: crate::engine::metal_context::WeightBuffer,
    pub expert_buffer: crate::engine::metal_context::ExpertBuffer,
    _phantom: PhantomData<C>,
}

impl<C: Gemma4ModelConfig> Gemma4Fused<C> {
    pub fn new(
        model: Arc<Model>,
        num_active_experts: usize,
        expert_cache_count: usize,
    ) -> Result<Self, MoEError> {
        C::validate_config(&model.config).map_err(MoEError::Config)?;
        let (ctx, weight_buffer, expert_buffer) = Gemma4MetalContext::new::<C>(
            &model.weight_file, num_active_experts, "Gemma4Fused", expert_cache_count,
        )?;
        Ok(Self {
            model,
            ctx,
            weight_buffer,
            expert_buffer,
            num_active_experts: if num_active_experts == 0 {
                C::NUM_EXPERTS_PER_TOK
            } else {
                num_active_experts
            },
            timing: BTreeMap::new(),
            last_h_pre_norm: Vec::new(),
            _phantom: PhantomData,
        })
    }
}

impl<C: Gemma4ModelConfig> Gemma4Fused<C> {
    /// Byte offset of a named tensor inside `weight_buffer.buf`. Returns
    /// `None` if the tensor isn't in the manifest (used to skip optional
    /// tensors like `v_proj` on full-attention layers).
    fn tensor_offset(&self, name: &str) -> Option<u64> {
        let ptr = self.model.weight_file.get_tensor_ptr(name)?;
        let base = self.weight_buffer.base as usize;
        Some((ptr as usize - base) as u64)
    }

    /// Tensor offset that MUST exist; panics with a clear message otherwise.
    fn tensor_offset_required(&self, name: &str) -> u64 {
        self.tensor_offset(name)
            .unwrap_or_else(|| panic!("[gemma4-engine] required tensor missing: {}", name))
    }

    /// Look up the embedding row for one token. Gemma 4 scales the embedding
    /// by `sqrt(HIDDEN_DIM)` (the standard "embed_scale" trick that keeps
    /// post-embedding magnitudes O(1)).
    ///
    /// Tries BF16 first (`embed_tokens.weight` as u16 array), then BQ4
    /// (packed u32 + bf16 scales/biases). The BQ4 path mirrors qwen35's
    /// `embed_lookup` because Gemma 4's BQ4 quantize pipeline will emit the
    /// same on-disk format (per-row group quantization).
    fn embed_lookup_row(wf: &WeightFile, token_id: usize, out: &mut [f32], hidden_dim: usize) {
        let scale = (hidden_dim as f32).sqrt();

        // BF16 path (unquantized weights, e.g. for verification against MLX).
        if let Some(w) = wf.get_tensor_u16("language_model.model.embed_tokens.weight") {
            let row = &w[token_id * hidden_dim..(token_id + 1) * hidden_dim];
            for j in 0..hidden_dim {
                out[j] = bf16_to_f32(row[j]) * scale;
            }
            return;
        }

        // BQ4 path: row-wise 4-bit packed with bf16 scale+bias per group.
        // Same format as qwen35::fused_exp2::embed_lookup; only difference
        // is the trailing `* scale` for Gemma's embed_scale.
        let (Some(w), Some(s), Some(b)) = (
            wf.get_tensor_u32("language_model.model.embed_tokens.weight"),
            wf.get_tensor_u16("language_model.model.embed_tokens.scales"),
            wf.get_tensor_u16("language_model.model.embed_tokens.biases"),
        ) else {
            out.fill(0.0);
            return;
        };
        let w_info = wf.get_tensor_info("language_model.model.embed_tokens.weight").unwrap();
        let packed_cols = w_info.shape[1];
        let s_info = wf.get_tensor_info("language_model.model.embed_tokens.scales").unwrap();
        let num_groups = s_info.shape[1];
        let group_size = hidden_dim / num_groups;
        let packed_per_group = group_size / 8;
        let w_row = &w[token_id * packed_cols..];
        let s_row = &s[token_id * num_groups..];
        let b_row = &b[token_id * num_groups..];
        for g in 0..num_groups {
            let gscale = bf16_to_f32(s_row[g]);
            let gbias  = bf16_to_f32(b_row[g]);
            let base = g * group_size;
            for p in 0..packed_per_group {
                let packed = w_row[g * packed_per_group + p];
                for n in 0..8 {
                    let nibble = (packed >> (n * 4)) & 0xF;
                    out[base + p * 8 + n] = ((nibble as f32) * gscale + gbias) * scale;
                }
            }
        }
    }

    /// Final RMSNorm + tied lm_head matvec + logit softcap.
    /// CPU implementation — single-token output, fine to keep in Rust for now
    /// (vocab=262144 × hidden=2816 ≈ 0.7B ops, ~50 ms on M-series CPU). Will
    /// move to Metal once the rest of the forward is on-GPU.
    ///
    /// `tied_embedding`: Gemma 4 ties lm_head to embed_tokens. We dequant the
    /// embedding weight row-by-row (one row per vocab id) and dot with the
    /// final-norm output. Same on-disk format as `embed_lookup_row`.
    fn final_norm_lm_head_softcap(
        wf: &WeightFile,
        hidden: &mut [f32],
        out_logits: &mut [f32],
        hidden_dim: usize,
        vocab_size: usize,
        softcap: f32,
    ) {
        // ── Final RMSNorm with model.norm.weight ──────────────────────────
        if let Some(fnw_u16) = wf.get_tensor_u16("language_model.model.norm.weight") {
            let fnw: Vec<f32> = fnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
            let sum_sq: f32 = hidden[..hidden_dim].iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / hidden_dim as f32 + RMS_NORM_EPS).sqrt();
            for i in 0..hidden_dim {
                hidden[i] *= inv_rms * fnw[i];
            }
        }

        // ── Tied lm_head: logits[v] = <hidden, embed_tokens[v]> ───────────
        // Reuses `embed_lookup_row` to dequant each row into a scratch buf
        // then dots with hidden. CRUCIAL: must NOT apply the sqrt-hidden
        // embed_scale for lm_head; we strip it by re-dequantizing without
        // the multiplier.
        //
        // Quick path: read the weight once, dot per row. Two cases:
        //   - BF16 tensor → direct dot
        //   - BQ4 tensor → dequant on the fly, dot
        if let Some(w) = wf.get_tensor_u16("language_model.model.embed_tokens.weight") {
            for v in 0..vocab_size {
                let row = &w[v * hidden_dim..(v + 1) * hidden_dim];
                let mut acc = 0.0f32;
                for j in 0..hidden_dim {
                    acc += hidden[j] * bf16_to_f32(row[j]);
                }
                out_logits[v] = softcap * (acc / softcap).tanh();
            }
            return;
        }

        // BQ4 path.
        let (Some(w), Some(s), Some(b)) = (
            wf.get_tensor_u32("language_model.model.embed_tokens.weight"),
            wf.get_tensor_u16("language_model.model.embed_tokens.scales"),
            wf.get_tensor_u16("language_model.model.embed_tokens.biases"),
        ) else {
            out_logits.fill(0.0);
            return;
        };
        let w_info = wf.get_tensor_info("language_model.model.embed_tokens.weight").unwrap();
        let packed_cols = w_info.shape[1];
        let s_info = wf.get_tensor_info("language_model.model.embed_tokens.scales").unwrap();
        let num_groups = s_info.shape[1];
        let group_size = hidden_dim / num_groups;
        let packed_per_group = group_size / 8;
        let mut row = vec![0.0f32; hidden_dim];
        for v in 0..vocab_size {
            let w_row = &w[v * packed_cols..];
            let s_row = &s[v * num_groups..];
            let b_row = &b[v * num_groups..];
            for g in 0..num_groups {
                let gscale = bf16_to_f32(s_row[g]);
                let gbias  = bf16_to_f32(b_row[g]);
                let base = g * group_size;
                for p in 0..packed_per_group {
                    let packed = w_row[g * packed_per_group + p];
                    for n in 0..8 {
                        let nibble = (packed >> (n * 4)) & 0xF;
                        row[base + p * 8 + n] = (nibble as f32) * gscale + gbias;
                    }
                }
            }
            let mut acc = 0.0f32;
            for j in 0..hidden_dim {
                acc += hidden[j] * row[j];
            }
            out_logits[v] = softcap * (acc / softcap).tanh();
        }
    }

    /// One sliding-attention layer's forward pass (single token).
    ///
    /// Operates entirely on the persistent GPU buffers in self.ctx, reading
    /// quantized/BF16 weights from self.weight_buffer via byte offsets.
    /// All BF16 path for now (matvec_bf16); will dispatch through a
    /// per-tensor-dtype matvec once BQ4 quantize lands (Task 5).
    ///
    /// Status: ATTENTION BLOCK ONLY. The dual-FFN path panics with a clear
    /// pointer at the bottom of this function. Splitting attention from FFN
    /// lets us unit-validate the attention pipeline (rms_norm + matvec×4 +
    /// q/k head_norm_rope + v_norm + kv_cache_append + sliding SDPA + o_proj
    /// + residual_add) before piling FFN on top.
    fn forward_sliding_layer(&mut self, layer: usize, pos: u32) -> Result<(), MoEError> {
        use crate::engine::gemma4_metal_kernels as gk;
        let hidden_dim   = C::HIDDEN_DIM as u32;
        let head_dim     = C::HEAD_DIM as u32;
        let n_q_heads    = C::NUM_ATTN_HEADS as u32;
        let n_kv_heads   = C::NUM_KV_HEADS as u32;
        let q_dim        = n_q_heads * head_dim;
        let kv_dim       = n_kv_heads * head_dim;
        let rope_theta   = C::ROPE_THETA_SLIDING as f32;
        let sliding_win  = C::SLIDING_WINDOW as u32;

        let prefix = format!("language_model.model.layers.{}", layer);
        // ── Required tensor offsets (BF16 weights) ─────────────────────────
        let off_input_ln   = self.tensor_offset_required(&format!("{}.input_layernorm.weight", prefix));
        let off_q_proj_w   = self.tensor_offset_required(&format!("{}.self_attn.q_proj.weight", prefix));
        let off_k_proj_w   = self.tensor_offset_required(&format!("{}.self_attn.k_proj.weight", prefix));
        let off_v_proj_w   = self.tensor_offset_required(&format!("{}.self_attn.v_proj.weight", prefix));
        let off_o_proj_w   = self.tensor_offset_required(&format!("{}.self_attn.o_proj.weight", prefix));
        let off_q_norm_w   = self.tensor_offset_required(&format!("{}.self_attn.q_norm.weight", prefix));
        let off_k_norm_w   = self.tensor_offset_required(&format!("{}.self_attn.k_norm.weight", prefix));
        let off_post_attn_ln = self.tensor_offset_required(&format!("{}.post_attention_layernorm.weight", prefix));

        let wbuf = &self.weight_buffer.buf;
        let ctx = &self.ctx;

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        // ── 1. input_layernorm: buf_hidden × input_layernorm.weight → buf_input_normed ──
        gk::encode_rms_norm_fused_bf16(
            ctx, enc,
            &ctx.buf_hidden, 0,
            wbuf, off_input_ln,
            &ctx.buf_input_normed, 0,
            hidden_dim,
        );

        // ── 2. Q / K / V projections (BF16 matvec) ──────────────────────────
        gk::encode_matvec_bf16(
            ctx, enc,
            wbuf, off_q_proj_w,
            &ctx.buf_input_normed, 0,
            &ctx.buf_q, 0,
            q_dim, hidden_dim,
        );
        gk::encode_matvec_bf16(
            ctx, enc,
            wbuf, off_k_proj_w,
            &ctx.buf_input_normed, 0,
            &ctx.buf_k, 0,
            kv_dim, hidden_dim,
        );
        gk::encode_matvec_bf16(
            ctx, enc,
            wbuf, off_v_proj_w,
            &ctx.buf_input_normed, 0,
            &ctx.buf_v, 0,
            kv_dim, hidden_dim,
        );

        // ── 3. Q head-norm + RoPE (no q_gate) ──────────────────────────────
        // For sliding layers: rotary_dim = head_dim (full rotary).
        gk::encode_q_head_norm_rope_no_gate(
            ctx, enc,
            &ctx.buf_q, 0,
            wbuf, off_q_norm_w,
            &ctx.buf_q, 0,                // in-place — reads then writes
            n_q_heads, head_dim, head_dim,
            rope_theta, pos,
        );

        // ── 4. K head-norm + RoPE (in-place on buf_k) ──────────────────────
        gk::encode_k_head_norm_rope(
            ctx, enc,
            &ctx.buf_k, 0,
            wbuf, off_k_norm_w,
            n_kv_heads, head_dim, head_dim,
            rope_theta, pos,
        );

        // ── 5. v_norm: RMSNorm-no-scale, per-head ──────────────────────────
        gk::encode_rms_norm_no_scale(
            ctx, enc,
            &ctx.buf_v, 0,
            &ctx.buf_v, 0,                // in-place
            n_kv_heads, head_dim, crate::constants::RMS_NORM_EPS,
        );

        // ── 6. KV cache append at position `pos` ───────────────────────────
        gk::encode_kv_cache_append(
            ctx, enc,
            &ctx.buf_k, &ctx.buf_v,
            &ctx.kv_caches_k[layer], &ctx.kv_caches_v[layer],
            pos, kv_dim,
        );

        // ── 7. Sliding-window causal SDPA → buf_attn_out ───────────────────
        // heads_per_kv = num_q_heads / num_kv_heads (= 16/8 = 2 for sliding).
        let heads_per_kv = n_q_heads / n_kv_heads;
        gk::encode_attn_sdpa_sliding_causal(
            ctx, enc,
            &ctx.buf_q, 0,
            &ctx.kv_caches_k[layer], &ctx.kv_caches_v[layer],
            &ctx.buf_attn_out, 0,
            pos + 1,                       // seq_len = pos + 1 (we just appended)
            sliding_win,
            n_q_heads, head_dim, kv_dim, heads_per_kv,
        );

        // ── 8. o_proj: buf_attn_out × o_proj.weight → buf_o_proj_out ───────
        gk::encode_matvec_bf16(
            ctx, enc,
            wbuf, off_o_proj_w,
            &ctx.buf_attn_out, 0,
            &ctx.buf_o_proj_out, 0,
            hidden_dim, q_dim,
        );

        // ── 9. post_attention_layernorm + residual ─────────────────────────
        // h = residual + post_attention_layernorm(o_proj_out)
        //   = buf_hidden + RMSNorm(buf_o_proj_out)
        gk::encode_rms_norm_fused_bf16(
            ctx, enc,
            &ctx.buf_o_proj_out, 0,
            wbuf, off_post_attn_ln,
            &ctx.buf_post_attn_normed, 0,
            hidden_dim,
        );
        gk::encode_residual_add(
            ctx, enc,
            &ctx.buf_hidden, 0,
            &ctx.buf_post_attn_normed, 0,
            &ctx.buf_hidden, 0,
            hidden_dim,
        );

        // ── Dual FFN block (still on the same command buffer) ──────────────
        // Save residual for the post-FFN add. Reuse buf_ff_combined as the
        // residual mirror — cheap CPU copy of [HIDDEN_DIM] f32 once.
        // (Alternative would be a Metal blit; the CPU memcpy on shared mem
        //  is fine for hidden_dim=2816.)
        let inter = C::INTERMEDIATE_SIZE as u32;
        let moe_inter = C::MOE_INTERMEDIATE as u32;
        let num_experts = C::NUM_EXPERTS as u32;

        // Need to flush the attention block before the CPU peek for residual.
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Snapshot residual = buf_hidden (post-attention) into buf_ff_combined.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.buf_hidden.contents() as *const f32,
                ctx.buf_ff_combined.contents() as *mut f32,
                C::HIDDEN_DIM,
            );
        }

        // ── Tensor offsets for the FFN block (all BF16) ────────────────────
        let off_pre_ff_ln      = self.tensor_offset_required(&format!("{}.pre_feedforward_layernorm.weight", prefix));
        let off_pre_ff_ln_2    = self.tensor_offset_required(&format!("{}.pre_feedforward_layernorm_2.weight", prefix));
        let off_post_ff_ln     = self.tensor_offset_required(&format!("{}.post_feedforward_layernorm.weight", prefix));
        let off_post_ff_ln_1   = self.tensor_offset_required(&format!("{}.post_feedforward_layernorm_1.weight", prefix));
        let off_post_ff_ln_2   = self.tensor_offset_required(&format!("{}.post_feedforward_layernorm_2.weight", prefix));
        let off_mlp_gate_w     = self.tensor_offset_required(&format!("{}.mlp.gate_proj.weight", prefix));
        let off_mlp_up_w       = self.tensor_offset_required(&format!("{}.mlp.up_proj.weight", prefix));
        let off_mlp_down_w     = self.tensor_offset_required(&format!("{}.mlp.down_proj.weight", prefix));
        let off_router_proj    = self.tensor_offset_required(&format!("{}.router.proj.weight", prefix));
        let off_router_scale   = self.tensor_offset_required(&format!("{}.router.scale", prefix));
        let off_per_expert_sc  = self.tensor_offset_required(&format!("{}.router.per_expert_scale", prefix));
        let off_experts_gu     = self.tensor_offset_required(&format!("{}.experts.gate_up_proj", prefix));
        let off_experts_down   = self.tensor_offset_required(&format!("{}.experts.down_proj", prefix));
        let off_layer_scalar   = self.tensor_offset_required(&format!("{}.layer_scalar", prefix));

        // ── Start a second command buffer for the dense MLP path + router ──
        let cmd_buf2 = ctx.queue.new_command_buffer();
        let enc2 = cmd_buf2.new_compute_command_encoder();

        // Dense MLP path. mlp.gate_proj/up_proj have shape [INTERMEDIATE=2112, HIDDEN].
        gk::encode_rms_norm_fused_bf16(
            ctx, enc2,
            &ctx.buf_hidden, 0,
            wbuf, off_pre_ff_ln,
            &ctx.buf_pre_ff_normed, 0,
            hidden_dim,
        );
        gk::encode_matvec_bf16(
            ctx, enc2,
            wbuf, off_mlp_gate_w,
            &ctx.buf_pre_ff_normed, 0,
            &ctx.buf_mlp_gate, 0,
            inter, hidden_dim,
        );
        gk::encode_matvec_bf16(
            ctx, enc2,
            wbuf, off_mlp_up_w,
            &ctx.buf_pre_ff_normed, 0,
            &ctx.buf_mlp_up, 0,
            inter, hidden_dim,
        );
        gk::encode_gelu_fused(
            ctx, enc2,
            &ctx.buf_mlp_gate, 0,
            &ctx.buf_mlp_up,   0,
            &ctx.buf_mlp_act,  0,
            inter,
        );
        // mlp.down_proj has shape [HIDDEN, INTERMEDIATE].
        gk::encode_matvec_bf16(
            ctx, enc2,
            wbuf, off_mlp_down_w,
            &ctx.buf_mlp_act, 0,
            &ctx.buf_mlp_down, 0,
            hidden_dim, inter,
        );
        gk::encode_rms_norm_fused_bf16(
            ctx, enc2,
            &ctx.buf_mlp_down, 0,
            wbuf, off_post_ff_ln_1,
            &ctx.buf_mlp_post, 0,
            hidden_dim,
        );

        // Router. rms_norm_router uses per-channel router.scale × sqrt(hidden)^-1.
        gk::encode_rms_norm_router(
            ctx, enc2,
            &ctx.buf_hidden, 0,
            wbuf, off_router_scale,
            &ctx.buf_router_normed, 0,
            hidden_dim,
        );
        gk::encode_matvec_bf16(
            ctx, enc2,
            wbuf, off_router_proj,
            &ctx.buf_router_normed, 0,
            &ctx.buf_router_logits, 0,
            num_experts, hidden_dim,
        );

        enc2.end_encoding();
        cmd_buf2.commit();
        cmd_buf2.wait_until_completed();

        // ── CPU: top-K over router_logits, then softmax over the K winners,
        //        then multiply by per_expert_scale[indices].
        let logits_ptr = ctx.buf_router_logits.contents() as *const f32;
        let logits: &[f32] = unsafe {
            std::slice::from_raw_parts(logits_ptr, C::NUM_EXPERTS)
        };
        let k = self.num_active_experts;
        let mut idx: Vec<usize> = (0..C::NUM_EXPERTS).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top_idx: Vec<usize> = idx[..k].to_vec();
        let mut top_w: Vec<f32> = top_idx.iter().map(|&i| logits[i]).collect();
        // Softmax over the top-K.
        let mx = top_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for w in top_w.iter_mut() { *w = (*w - mx).exp(); sum += *w; }
        for w in top_w.iter_mut() { *w /= sum; }
        // Multiply by per_expert_scale.
        let pes_ptr = self.weight_buffer.base as usize + off_per_expert_sc as usize;
        for (ki, &e) in top_idx.iter().enumerate() {
            let raw = unsafe { *(pes_ptr as *const u16).add(e) };
            top_w[ki] *= bf16_to_f32(raw);
        }

        // ── Experts path. Each expert: gate (slice of gate_up_proj), up
        //    (next slice), down (slice of down_proj). HF layout:
        //      experts.gate_up_proj : [128, 2*moe_inter=1408, hidden=2816]
        //        Per-expert slice  : [1408, 2816] (rows 0..704 gate, 704..1408 up)
        //      experts.down_proj   : [128, hidden=2816, moe_inter=704]
        //        Per-expert slice  : [2816, 704]
        //    All BF16. Byte offsets are per-expert constants we add to the
        //    base tensor offset.
        // pre_feedforward_layernorm_2 → buf_pre_ff_normed_2 (input to experts).
        {
            let cb = ctx.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            gk::encode_rms_norm_fused_bf16(
                ctx, e,
                &ctx.buf_hidden, 0,
                wbuf, off_pre_ff_ln_2,
                &ctx.buf_pre_ff_normed_2, 0,
                hidden_dim,
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Zero buf_expert_out (we'll accumulate into it across K experts).
        unsafe {
            std::ptr::write_bytes(
                ctx.buf_expert_out.contents() as *mut u8,
                0,
                C::HIDDEN_DIM * 4,
            );
        }

        let bf16_bytes: u64 = 2;
        let per_expert_gu_bytes   = (2 * moe_inter * hidden_dim) as u64 * bf16_bytes;
        let per_expert_down_bytes = (hidden_dim * moe_inter)    as u64 * bf16_bytes;
        let gate_block_bytes      = (moe_inter * hidden_dim)    as u64 * bf16_bytes;

        for (ki, &expert_idx) in top_idx.iter().enumerate() {
            let e_u64 = expert_idx as u64;
            let expert_gu_off   = off_experts_gu   + e_u64 * per_expert_gu_bytes;
            let expert_down_off = off_experts_down + e_u64 * per_expert_down_bytes;
            let cb = ctx.queue.new_command_buffer();
            let enc_e = cb.new_compute_command_encoder();
            // gate weight = first [moe_inter, hidden] block at expert_gu_off
            gk::encode_matvec_bf16(
                ctx, enc_e,
                wbuf, expert_gu_off,
                &ctx.buf_pre_ff_normed_2, 0,
                &ctx.buf_expert_gate, 0,
                moe_inter, hidden_dim,
            );
            // up weight = next [moe_inter, hidden] block (shifted by gate_block_bytes)
            gk::encode_matvec_bf16(
                ctx, enc_e,
                wbuf, expert_gu_off + gate_block_bytes,
                &ctx.buf_pre_ff_normed_2, 0,
                &ctx.buf_expert_up, 0,
                moe_inter, hidden_dim,
            );
            gk::encode_gelu_fused(
                ctx, enc_e,
                &ctx.buf_expert_gate, 0,
                &ctx.buf_expert_up,   0,
                &ctx.buf_expert_act,  0,
                moe_inter,
            );
            gk::encode_matvec_bf16(
                ctx, enc_e,
                wbuf, expert_down_off,
                &ctx.buf_expert_act, 0,
                &ctx.buf_expert_post, 0,
                hidden_dim, moe_inter,
            );
            enc_e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            // CPU weighted-accumulate on shared memory.
            let post_ptr = ctx.buf_expert_post.contents() as *const f32;
            let out_ptr  = ctx.buf_expert_out.contents()  as *mut f32;
            let w_ki = top_w[ki];
            unsafe {
                for j in 0..C::HIDDEN_DIM {
                    *out_ptr.add(j) += *post_ptr.add(j) * w_ki;
                }
            }
        }

        // ── Outer FFN norms + combine + residual + layer_scalar ─────────────
        let cmd_buf4 = ctx.queue.new_command_buffer();
        let enc4 = cmd_buf4.new_compute_command_encoder();

        // post_feedforward_layernorm_2 on the experts sum.
        gk::encode_rms_norm_fused_bf16(
            ctx, enc4,
            &ctx.buf_expert_out, 0,
            wbuf, off_post_ff_ln_2,
            &ctx.buf_expert_post, 0,
            hidden_dim,
        );
        // Combine dense + experts: buf_ff_outer_post = buf_mlp_post + buf_expert_post
        gk::encode_residual_add(
            ctx, enc4,
            &ctx.buf_mlp_post, 0,
            &ctx.buf_expert_post, 0,
            &ctx.buf_ff_outer_post, 0,
            hidden_dim,
        );
        // Outer post_feedforward_layernorm on the combined FFN output.
        gk::encode_rms_norm_fused_bf16(
            ctx, enc4,
            &ctx.buf_ff_outer_post, 0,
            wbuf, off_post_ff_ln,
            &ctx.buf_ff_outer_post, 0,             // in-place OK
            hidden_dim,
        );
        // Second residual: buf_hidden = (saved residual in buf_ff_combined) + buf_ff_outer_post.
        gk::encode_residual_add(
            ctx, enc4,
            &ctx.buf_ff_combined,  0,   // saved residual snapshot
            &ctx.buf_ff_outer_post, 0,
            &ctx.buf_hidden,        0,
            hidden_dim,
        );
        // Apply per-layer scalar: buf_hidden *= bf16 layer_scalar[0].
        gk::encode_mul_scalar_bf16(
            ctx, enc4,
            &ctx.buf_hidden, 0,
            wbuf, off_layer_scalar,
            hidden_dim,
        );

        enc4.end_encoding();
        cmd_buf4.commit();
        cmd_buf4.wait_until_completed();

        Ok(())
    }
}

impl<C: Gemma4ModelConfig> Engine for Gemma4Fused<C> {
    fn upload_cache(&self, _cache: &Cache) {
        // TODO: upload sliding-window ring buffers + full KV cache from CPU
        // mirror. Sliding window state has different shape than Qwen's
        // full KV cache — needs its own protocol.
    }

    fn download_cache(&self, _cache: &mut Cache) {
        // TODO: matching download path.
    }

    fn engine_pos(&self) -> usize {
        self.ctx.pos.get()
    }

    fn embed_lookup(&self, token_ids: &[i64], embeddings: &mut [f32]) {
        let hidden_dim = C::HIDDEN_DIM;
        let wf = &self.model.weight_file;
        for (i, &id) in token_ids.iter().enumerate() {
            let out = &mut embeddings[i * hidden_dim..(i + 1) * hidden_dim];
            Self::embed_lookup_row(wf, id as usize, out, hidden_dim);
        }
    }

    fn forward_hidden(
        &mut self,
        embeddings: &[f32],
        check_signal: SignalCheckFn<'_>,
        _mtp: bool,
    ) -> Result<Vec<f32>, MoEError> {
        // Per-token forward — VERIFIED against vendor/mlx-vlm/mlx_vlm/
        // models/gemma4/language.py (DecoderLayer.__call__, lines 307-363).
        //
        // Reference is the mlx-lm Gemma 4 implementation; this engine
        // reproduces its math in Metal kernels.
        //
        // h = embed_tokens(input_ids) * sqrt(HIDDEN_DIM)   # embed_scale
        //
        // for each layer:
        //   residual = h
        //   x = input_layernorm(h)
        //
        //   # ── Attention block ───────────────────────────────────────
        //   q = q_proj(x).reshape(B, L, num_heads, head_dim)
        //   q = q_norm(q)                                  # per-head RMSNorm WITH scale
        //   k = k_proj(x).reshape(B, L, num_kv_heads, head_dim)
        //   v = (k if (full_attn AND attention_k_eq_v) else v_proj(x))
        //   k = k_norm(k)                                  # per-head RMSNorm WITH scale
        //   v = v_norm(v)                                  # RMSNorm-NO-SCALE (no weight!)
        //   RoPE on q, k (NOT on v):
        //     sliding layers: theta=10k,  full rotary (head_dim=256)
        //     full layers:    theta=1M,   partial rotary 25% (global_head_dim=512!)
        //   cache.update_and_fetch(k, v)
        //   attn_out = SDPA(q, k_cache, v_cache, scale=1.0)
        //     sliding-masked for sliding layers (window=1024)
        //     full causal for full layers
        //   attn_out = o_proj(attn_out)
        //   h = post_attention_layernorm(attn_out)
        //   h = residual + h                               # First residual
        //
        //   # ── Dual FFN ─────────────────────────────────────────────
        //   residual = h
        //   if enable_moe (true for 26B-A4B):
        //     # Dense MLP path (always-on):
        //     h1 = pre_feedforward_layernorm(h)
        //     h1 = mlp_down(gelu_approx(mlp_gate(h1)) * mlp_up(h1))   # GEGLU
        //     h1 = post_feedforward_layernorm_1(h1)
        //
        //     # Sparse experts path. NOTE: router takes h (FFN input), NOT h2.
        //     r = rms_norm(h, router.scale * sqrt(HIDDEN_DIM)^-1, eps)
        //     expert_scores = router.proj(r)
        //     top_k_idx = argpartition top-K(expert_scores)
        //     top_k_w = softmax(take(expert_scores, top_k_idx))   # softmax over K, not N
        //     top_k_w = top_k_w * router.per_expert_scale[top_k_idx]
        //
        //     h2 = pre_feedforward_layernorm_2(h)
        //     expert_out = 0
        //     for ki in 0..top_k:
        //       e = top_k_idx[ki]
        //       act = gelu_approx(switch_glu.gate_proj[e] @ h2) * (switch_glu.up_proj[e] @ h2)
        //       expert_out += top_k_w[ki] * (switch_glu.down_proj[e] @ act)
        //     h2 = post_feedforward_layernorm_2(expert_out)
        //
        //     h = h1 + h2                                  # Combine paths additively
        //   else:
        //     h = pre_feedforward_layernorm(h)
        //     h = mlp(h)
        //
        //   h = post_feedforward_layernorm(h)              # Outer FFN norm
        //   h = residual + h                               # Second residual
        //
        //   # Per-layer scalar — multiplies the entire post-residual hidden:
        //   h = h * layer_scalar
        //
        // # ── Output ────────────────────────────────────────────────────
        // h = model.norm(h)
        // logits = h @ embed_tokens.weight.T               # tied embedding
        // logits = 30 * tanh(logits / 30)                  # final softcap
        //
        // Return logits[n, vocab_size].
        //
        // Per-layer quantization precision (from LanguageModel.quant_predicate):
        //   - router.proj:                    INT8 (group=64)  [routing accuracy]
        //   - mlp.{gate, up, down}_proj:      INT8 (group=64)  [dense path quality]
        //   - experts.switch_glu.*:           INT4 (default)
        //   - self_attn.{q,k,v,o}_proj:       INT4 (default)
        //   - embed_tokens (& tied lm_head):  INT4 (default)
        //
        // Per-layer-type specifics for 26B-A4B:
        //   - sliding layers (25 of 30): head_dim=256, full rotary
        //   - full layers (5 of 30):     global_head_dim=512, partial rotary 25%
        //   - attention_k_eq_v=true for full layers (K and V share the raw
        //     K projection; v_proj omitted on full layers; v_norm has no
        //     learnable weight)
        //   - num_kv_shared_layers=0       (KV sharing is a 2B/4B feature)
        //   - hidden_size_per_layer_input=0 (per-layer-input gating is a 2B/4B feature)

        // Phase 2 work breakdown (each chunk ~1-3 hours of careful coding):
        //
        // [1] Embedding lookup helper.
        //     - Read BF16 embed_tokens[token_id] from WeightFile
        //     - Multiply by sqrt(HIDDEN_DIM)
        //     - Copy into self.ctx.buf_hidden
        //
        // [2] Per-layer dispatcher.
        //     Reuse self.weight_buffer.encode_matvec_into / encode_matvec_n_into
        //     for all matvec ops (q/k/v/o projections, dense MLP, expert proj).
        //     The Gemma 4 model is quantized in MLX format (4-bit packed +
        //     bf16 scales/biases) — same packed format the existing engine
        //     reads. After Phase 3 quantize pipeline emits compatible BQ4
        //     blobs, encode_matvec_into "just works" for our weights.
        //
        // [3] Two new kernels, both parameterised (HEAD_DIM as runtime arg,
        //     not compile-time #define):
        //     - attn_sdpa_sliding_causal: same online-softmax as
        //       attn_sdpa_fused but loop bound is
        //         [max(0, seq_len - sliding_window), seq_len)
        //       and HEAD_DIM is passed as a constant buffer rather than
        //       a #define. ~150 lines.
        //     - rms_norm_no_scale: rms_norm without the learnable weight,
        //       used for v_norm on full layers. ~40 lines (variant of
        //       rms_norm_fused_bf16).
        //
        // [4] Two reuses with different parameters:
        //     - q_head_norm_rope: existing kernel works; pass theta=10k
        //       (sliding) or 1M (full), and rotary_dim=head_dim (sliding)
        //       or head_dim/4 (full, the 25% partial-rotary case).
        //     - k_head_norm_rope: same.
        //
        // [5] Dual-FFN forward — pure Rust orchestration over the kernels:
        //     a) Dense MLP path:
        //         input_norm → buf_pre_ff_normed (rms_norm_fused_bf16)
        //         gate_proj  → buf_mlp_gate      (matvec_bf16 or dequant_matvec_4bit)
        //         up_proj    → buf_mlp_up        (same)
        //         gelu*      → buf_mlp_act       (gelu_fused — ours, written)
        //         down_proj  → buf_mlp_down      (matvec)
        //         post_ff_1  → buf_mlp_post      (rms_norm_fused_bf16)
        //
        //     b) Router:
        //         router-norm(h)  → buf_router_normed       (CPU: rms_norm with `scale * sqrt(hidden)^-1` weight)
        //         router.proj     → buf_router_logits       (matvec)
        //         CPU top-K       → expert_indices[K]
        //         CPU softmax over top-K + per_expert_scale → expert_weights[K]
        //
        //     c) Experts path (reuse qwen35 batched MoE machinery — we can
        //        repurpose encode_post_expert_at with shared_gate_score=0
        //        and rerouting "shared expert" to no-op):
        //         pre_ff_2 norm   → buf_pre_ff_normed_2
        //         for each ki in K:
        //             pread expert (or cache)
        //             gate_proj → buf_expert_gate (dequant_matvec_4bit)
        //             up_proj   → buf_expert_up
        //             gelu_fused → buf_expert_act
        //             down_proj → expert_out_buf[ki]
        //         combine: sum weighted contributions → buf_expert_out
        //         post_ff_2 norm → buf_expert_post
        //
        //     d) Combine:
        //         buf_ff_combined = buf_mlp_post + buf_expert_post  (residual_add helper)
        //         outer post_ff norm → buf_ff_outer_post
        //         buf_hidden = h_after_attn_residual + buf_ff_outer_post  (residual_add)
        //         buf_hidden *= layer_scalar  (CPU scalar; copy out → multiply → copy back)
        //
        // [6] Final:
        //     model.norm → buf_hidden
        //     lm_head matvec (tied to embed_tokens) → buf_logits
        //     logit_softcap kernel (already written) → buf_logits
        //     CPU memcpy buf_logits → output Vec<f32>
        //
        // [7] Validation: verify against mlx-vlm on a small input. Needs
        //     Phase 3 (quantize pipeline) complete so we have engine-format
        //     weights to load. Expected: max_diff ~ 1e-3 (lossy quant) or
        //     ~ 2e-5 (if we load BF16 unquantized directly).

        let hidden_dim = C::HIDDEN_DIM;
        let n_tokens   = embeddings.len() / hidden_dim;
        let vocab_size = C::VOCAB_SIZE;
        let num_layers = C::NUM_LAYERS;
        let softcap    = C::FINAL_LOGIT_SOFTCAP;

        let mut logits = vec![0.0f32; n_tokens * vocab_size];
        if n_tokens == 0 {
            return Ok(logits);
        }

        let mut pos = self.ctx.pos.get();
        let mut hidden_cpu = vec![0.0f32; hidden_dim];

        for ti in 0..n_tokens {
            if check_signal() {
                return Err(MoEError::Metal("interrupted".into()));
            }

            // Chunk [1]: bring this token's pre-scaled embedding into the GPU
            // hidden buffer. `embeddings` is f32 already and includes the
            // sqrt(HIDDEN) embed_scale (applied by self.embed_lookup_row).
            hidden_cpu.copy_from_slice(&embeddings[ti * hidden_dim..(ti + 1) * hidden_dim]);
            {
                let dst = self.ctx.buf_hidden.contents() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(hidden_cpu.as_ptr(), dst, hidden_dim);
                }
            }

            // Chunks [2,4,5]: per-layer forward.
            for layer in 0..num_layers {
                if check_signal() {
                    return Err(MoEError::Metal("interrupted".into()));
                }
                if C::is_full_attn_layer(layer) {
                    unimplemented!(
                        "Gemma 4 full-attention layer {} (head_dim=512, partial RoPE, \
                         attention_k_eq_v): needs the head_dim=512 SDPA kernel — Task 4.",
                        layer
                    );
                }
                self.forward_sliding_layer(layer, pos as u32)?;
            }

            // Copy final hidden back to CPU for the lm_head pass.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ctx.buf_hidden.contents() as *const f32,
                    hidden_cpu.as_mut_ptr(),
                    hidden_dim,
                );
            }

            pos += 1;
            self.ctx.pos.set(pos);

            // Chunk [6]: final RMSNorm + tied lm_head matvec + logit_softcap.
            // CPU implementation for now — single-token vocab=262144 dot is
            // ~50 ms on M-series CPU; move to Metal once the per-layer
            // dispatch is on-GPU and saturating the device.
            Self::final_norm_lm_head_softcap(
                &self.model.weight_file,
                &mut hidden_cpu,
                &mut logits[ti * vocab_size..(ti + 1) * vocab_size],
                hidden_dim,
                vocab_size,
                softcap,
            );
        }

        Ok(logits)
    }

    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.timing.clone()
    }
}
