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

    /// Extract the layer index from the engine name prefix
    /// `"language_model.model.layers.N"` → N.
    fn layer_idx_from_prefix(prefix: &str) -> usize {
        prefix.rsplit('.').next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("[gemma4-engine] bad prefix: {}", prefix))
    }

    /// Looks up `prefix.weight` and dispatches the matvec kernel matching
    /// the tensor's dtype: matvec_bf16 for BF16, dequant_matvec_4bit_v3
    /// for INT4 (with scales+biases sibling tensors).
    fn encode_matvec_auto(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        prefix: &str,
        x: &metal::BufferRef, x_offset: u64,
        out: &metal::BufferRef, o_offset: u64,
        out_dim: u32, in_dim: u32,
    ) {
        use crate::engine::gemma4_metal_kernels as gk;
        let weight_name = format!("{}.weight", prefix);
        let info = self.model.weight_file.get_tensor_info(&weight_name)
            .unwrap_or_else(|| panic!("[gemma4-engine] missing tensor info: {}", weight_name));
        let dtype = info.dtype.as_str();
        let w_off = self.tensor_offset_required(&weight_name);
        let wbuf = &self.weight_buffer.buf;
        match dtype {
            "bf16" => {
                gk::encode_matvec_bf16(&self.ctx, encoder,
                    wbuf, w_off, x, x_offset, out, o_offset, out_dim, in_dim);
            }
            // "u32" is the engine's wire name for INT4 (per dtype.rs)
            "u32" | "int4" => {
                let s_off = self.tensor_offset_required(&format!("{}.scales", prefix));
                let b_off = self.tensor_offset_required(&format!("{}.biases", prefix));
                gk::encode_matvec_int4(&self.ctx, encoder,
                    wbuf, w_off, wbuf, s_off, wbuf, b_off,
                    x, x_offset, out, o_offset, out_dim, in_dim);
            }
            _ => panic!("[gemma4-engine] unsupported dtype '{}' for {}", dtype, weight_name),
        }
    }

    /// Per-expert matvec for tensors quantized as a single big 2D matrix
    /// (`experts.gate_up_proj`, `experts.down_proj`). Computes byte offset
    /// for expert e's row block within the merged tensor, then dispatches
    /// matvec on that slice. For INT4, scales and biases are sliced in
    /// parallel by the same row-block stride.
    ///
    /// `tensor_base_name` is e.g. "experts.gate_up_proj" — the engine looks
    /// up `{base}.weight` / `.scales` / `.biases` for INT4, or just `{base}`
    /// for the BF16 fallback.
    /// `expert_idx` selects which expert's slice to use.
    /// `row_offset` and `row_count` select a sub-block within the expert
    /// (e.g. rows 0..moe_inter for gate, rows moe_inter..2*moe_inter for up).
    #[allow(clippy::too_many_arguments)]
    fn encode_matvec_expert_slice(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        tensor_base: &str,
        expert_idx: usize,
        rows_per_expert: u32,
        row_offset: u32,
        row_count: u32,
        x: &metal::BufferRef, x_offset: u64,
        out: &metal::BufferRef, o_offset: u64,
        in_dim: u32,
    ) {
        use crate::engine::gemma4_metal_kernels as gk;
        // Try INT4 layout first (look for .weight + .scales + .biases triple).
        let weight_name = format!("{}.weight", tensor_base);
        let info = self.model.weight_file.get_tensor_info(&weight_name);
        let wbuf = &self.weight_buffer.buf;
        if let Some(_info) = info {
            // INT4 path. Per-row stride: packed weight = in_dim/8 u32 = in_dim/2 bytes.
            // The whole merged tensor is [num_experts * rows_per_expert, in_dim] INT4.
            const GS: u32 = 64;
            let groups_per_row = in_dim / GS;
            let w_off_base = self.tensor_offset_required(&weight_name);
            let s_off_base = self.tensor_offset_required(&format!("{}.scales", tensor_base));
            let b_off_base = self.tensor_offset_required(&format!("{}.biases", tensor_base));
            let expert_row0 = expert_idx as u64 * rows_per_expert as u64 + row_offset as u64;
            // Per-row sizes: weight = in_dim/2 bytes, scales/biases = groups_per_row*2 bytes
            let w_bytes_per_row = (in_dim / 2) as u64;
            let sb_bytes_per_row = (groups_per_row * 2) as u64;
            let w_off = w_off_base + expert_row0 * w_bytes_per_row;
            let s_off = s_off_base + expert_row0 * sb_bytes_per_row;
            let b_off = b_off_base + expert_row0 * sb_bytes_per_row;
            gk::encode_matvec_int4(&self.ctx, encoder,
                wbuf, w_off, wbuf, s_off, wbuf, b_off,
                x, x_offset, out, o_offset, row_count, in_dim);
            return;
        }
        // BF16 fallback: tensor stored under tensor_base (no .weight suffix in
        // the BF16 quantize path because we keep the HF name).
        let bf16_base_off = self.tensor_offset_required(tensor_base);
        let bf16_bytes: u64 = 2;
        let rows_offset_bytes = (expert_idx as u64 * rows_per_expert as u64
                                 + row_offset as u64)
                                * in_dim as u64 * bf16_bytes;
        gk::encode_matvec_bf16(&self.ctx, encoder,
            wbuf, bf16_base_off + rows_offset_bytes,
            x, x_offset, out, o_offset, row_count, in_dim);
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
        let info = wf.get_tensor_info("language_model.model.embed_tokens.weight");

        // BF16 path (unquantized weights, e.g. for verification against MLX).
        if info.map(|t| t.dtype == "bf16").unwrap_or(false) {
            let w = wf.get_tensor_u16("language_model.model.embed_tokens.weight").unwrap();
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
        let lm_info = wf.get_tensor_info("language_model.model.embed_tokens.weight");
        let is_bf16 = lm_info.map(|t| t.dtype == "bf16").unwrap_or(false);
        if is_bf16 {
            let w = wf.get_tensor_u16("language_model.model.embed_tokens.weight").unwrap();
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
        // BF16-only tensor offsets (norms). Matmul weights are looked up by
        // encode_matvec_auto which dispatches BF16 or INT4 per tensor dtype.
        let off_input_ln   = self.tensor_offset_required(&format!("{}.input_layernorm.weight", prefix));
        let off_q_norm_w   = self.tensor_offset_required(&format!("{}.self_attn.q_norm.weight", prefix));
        let off_k_norm_w   = self.tensor_offset_required(&format!("{}.self_attn.k_norm.weight", prefix));
        let off_post_attn_ln = self.tensor_offset_required(&format!("{}.post_attention_layernorm.weight", prefix));

        let wbuf = &self.weight_buffer.buf;
        let ctx = &self.ctx;

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        // 1. input_layernorm
        gk::encode_rms_norm_fused_bf16(
            ctx, enc, &ctx.buf_hidden, 0, wbuf, off_input_ln, &ctx.buf_input_normed, 0, hidden_dim);

        // 2. Q / K / V projections (BF16 or INT4, auto-dispatched).
        self.encode_matvec_auto(enc, &format!("{}.self_attn.q_proj", prefix),
            &ctx.buf_input_normed, 0, &ctx.buf_q, 0, q_dim, hidden_dim);
        self.encode_matvec_auto(enc, &format!("{}.self_attn.k_proj", prefix),
            &ctx.buf_input_normed, 0, &ctx.buf_k, 0, kv_dim, hidden_dim);
        self.encode_matvec_auto(enc, &format!("{}.self_attn.v_proj", prefix),
            &ctx.buf_input_normed, 0, &ctx.buf_v, 0, kv_dim, hidden_dim);

        // 3. Q head-norm + RoPE (no q_gate); sliding: full rotary.
        gk::encode_q_head_norm_rope_no_gate(
            ctx, enc, &ctx.buf_q, 0, wbuf, off_q_norm_w, &ctx.buf_q, 0,
            n_q_heads, head_dim, head_dim, rope_theta, pos);

        // 4. K head-norm + RoPE
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

        // 8. o_proj (auto-dispatched BF16 or INT4)
        self.encode_matvec_auto(enc, &format!("{}.self_attn.o_proj", prefix),
            &ctx.buf_attn_out, 0, &ctx.buf_o_proj_out, 0, hidden_dim, q_dim);

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

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // ── Dual FFN block (shared with forward_full_layer) ─────────────────
        self.forward_dual_ffn(&prefix)
    }

    /// One full-attention layer's forward pass (Gemma 4: every 6th layer in
    /// 26B-A4B). Differences from forward_sliding_layer's attention block:
    ///
    ///   - head_dim = global_head_dim = 512 (vs 256 sliding)
    ///   - num_kv_heads = NUM_KV_HEADS_FULL = 2 (vs 8 sliding)
    ///   - NO v_proj weight (attention_k_eq_v=True): V = raw K, BEFORE k_norm
    ///   - v_norm = rms_norm_no_scale (no learnable weight, same as sliding's)
    ///   - RoPE: theta=1M, partial rotary (first 25% = head_dim/4 = 128 dims)
    ///   - SDPA: full causal (no sliding-window restriction), head_dim=512
    ///     kernel (`attn_sdpa_causal_h512`)
    ///
    /// The dual-FFN block is identical to sliding layers, so we reuse the
    /// same orchestration code at the end. To avoid duplication we factor
    /// the FFN block into `forward_dual_ffn` and call it from both layer
    /// types.
    fn forward_full_layer(&mut self, layer: usize, pos: u32) -> Result<(), MoEError> {
        use crate::engine::gemma4_metal_kernels as gk;
        let hidden_dim   = C::HIDDEN_DIM as u32;
        let head_dim_full: u32 = (C::HEAD_DIM * 2) as u32;     // global_head_dim = 512
        let n_q_heads    = C::NUM_ATTN_HEADS as u32;
        let n_kv_heads_f = C::NUM_KV_HEADS_FULL as u32;
        let q_dim_full   = n_q_heads * head_dim_full;          // 16*512 = 8192
        let kv_dim_full  = n_kv_heads_f * head_dim_full;       // 2*512 = 1024
        let rope_theta   = C::ROPE_THETA_FULL as f32;
        let rotary_dim   = (head_dim_full as f32 * C::PARTIAL_ROTARY_FRACTION_FULL) as u32;
        // For partial_rotary_factor=0.25, head_dim=512 → rotary_dim=128.
        debug_assert!(rotary_dim > 0 && rotary_dim <= head_dim_full,
                      "rotary_dim out of range: {} (head_dim={})", rotary_dim, head_dim_full);

        let prefix = format!("language_model.model.layers.{}", layer);
        let off_input_ln     = self.tensor_offset_required(&format!("{}.input_layernorm.weight", prefix));
        let off_q_norm_w     = self.tensor_offset_required(&format!("{}.self_attn.q_norm.weight", prefix));
        let off_k_norm_w     = self.tensor_offset_required(&format!("{}.self_attn.k_norm.weight", prefix));
        let off_post_attn_ln = self.tensor_offset_required(&format!("{}.post_attention_layernorm.weight", prefix));
        // No v_proj on full layers (attention_k_eq_v).

        let wbuf = &self.weight_buffer.buf;
        let ctx = &self.ctx;

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        gk::encode_rms_norm_fused_bf16(
            ctx, enc, &ctx.buf_hidden, 0, wbuf, off_input_ln,
            &ctx.buf_input_normed, 0, hidden_dim);
        self.encode_matvec_auto(enc, &format!("{}.self_attn.q_proj", prefix),
            &ctx.buf_input_normed, 0, &ctx.buf_q, 0, q_dim_full, hidden_dim);
        self.encode_matvec_auto(enc, &format!("{}.self_attn.k_proj", prefix),
            &ctx.buf_input_normed, 0, &ctx.buf_k, 0, kv_dim_full, hidden_dim);

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // 4. V = raw K (before k_norm/RoPE). Copy on shared memory now,
        //    before k_norm runs in-place on buf_k.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.buf_k.contents() as *const f32,
                ctx.buf_v.contents() as *mut f32,
                kv_dim_full as usize,
            );
        }

        // 5. Q head-norm + RoPE (partial rotary, theta=1M).
        //    q_head_norm_rope_no_gate already supports head_dim up to 512
        //    via its partial[512] threadgroup buffer.
        let cmd_buf_qk = ctx.queue.new_command_buffer();
        let enc_qk = cmd_buf_qk.new_compute_command_encoder();
        gk::encode_q_head_norm_rope_no_gate(
            ctx, enc_qk,
            &ctx.buf_q, 0,
            wbuf, off_q_norm_w,
            &ctx.buf_q, 0,
            n_q_heads, head_dim_full, rotary_dim,
            rope_theta, pos,
        );
        // 6. K head-norm + RoPE — use q_head_norm_rope_no_gate for K too;
        //    structurally identical (no gate split), and supports head_dim
        //    up to 512 (qwen35's k_head_norm_rope only supports up to 256).
        gk::encode_q_head_norm_rope_no_gate(
            ctx, enc_qk,
            &ctx.buf_k, 0,
            wbuf, off_k_norm_w,
            &ctx.buf_k, 0,
            n_kv_heads_f, head_dim_full, rotary_dim,
            rope_theta, pos,
        );
        // 7. v_norm (RMSNormNoScale, per-head)
        gk::encode_rms_norm_no_scale(
            ctx, enc_qk,
            &ctx.buf_v, 0,
            &ctx.buf_v, 0,
            n_kv_heads_f, head_dim_full, crate::constants::RMS_NORM_EPS,
        );
        // 8. KV cache append (kv_dim = kv_dim_full = 1024)
        gk::encode_kv_cache_append(
            ctx, enc_qk,
            &ctx.buf_k, &ctx.buf_v,
            &ctx.kv_caches_k[layer], &ctx.kv_caches_v[layer],
            pos, kv_dim_full,
        );
        // 9. Full-causal SDPA (head_dim=512 kernel)
        let heads_per_kv_f = n_q_heads / n_kv_heads_f;
        gk::encode_attn_sdpa_causal_h512(
            ctx, enc_qk,
            &ctx.buf_q, 0,
            &ctx.kv_caches_k[layer], &ctx.kv_caches_v[layer],
            &ctx.buf_attn_out, 0,
            pos + 1,
            n_q_heads, kv_dim_full, heads_per_kv_f,
        );
        // 10. o_proj: [hidden, q_dim_full]
        self.encode_matvec_auto(enc_qk, &format!("{}.self_attn.o_proj", prefix),
            &ctx.buf_attn_out, 0, &ctx.buf_o_proj_out, 0, hidden_dim, q_dim_full);
        // 11. post_attention_layernorm + first residual
        gk::encode_rms_norm_fused_bf16(
            ctx, enc_qk,
            &ctx.buf_o_proj_out, 0,
            wbuf, off_post_attn_ln,
            &ctx.buf_post_attn_normed, 0,
            hidden_dim,
        );
        gk::encode_residual_add(
            ctx, enc_qk,
            &ctx.buf_hidden, 0,
            &ctx.buf_post_attn_normed, 0,
            &ctx.buf_hidden, 0,
            hidden_dim,
        );

        enc_qk.end_encoding();
        cmd_buf_qk.commit();
        cmd_buf_qk.wait_until_completed();

        // ── Dual-FFN block (identical to sliding layers) ────────────────────
        self.forward_dual_ffn(&prefix)
    }

    /// Dual-FFN block — extracted from forward_sliding_layer so full layers
    /// can reuse it. Operates on buf_hidden in-place: reads the post-attention
    /// hidden state, writes back the final layer output (after layer_scalar).
    ///
    /// Order:
    ///   residual_snapshot = buf_hidden  (via CPU memcpy → buf_ff_combined)
    ///   Dense MLP path → buf_mlp_post
    ///   Router (norm + matvec + CPU top-K/softmax/per_expert_scale)
    ///   Experts (K iterations, weighted CPU accumulate) → buf_expert_out
    ///   post_feedforward_layernorm_2 → buf_expert_post
    ///   combine: buf_mlp_post + buf_expert_post → buf_ff_outer_post
    ///   outer post_feedforward_layernorm → buf_ff_outer_post (in-place)
    ///   second residual: residual_snapshot + buf_ff_outer_post → buf_hidden
    ///   layer_scalar: buf_hidden *= bf16 scalar
    fn forward_dual_ffn(&mut self, prefix: &str) -> Result<(), MoEError> {
        use crate::engine::gemma4_metal_kernels as gk;
        let hidden_dim  = C::HIDDEN_DIM as u32;
        let inter       = C::INTERMEDIATE_SIZE as u32;
        let moe_inter   = C::MOE_INTERMEDIATE as u32;
        let num_experts = C::NUM_EXPERTS as u32;

        let off_pre_ff_ln      = self.tensor_offset_required(&format!("{}.pre_feedforward_layernorm.weight", prefix));
        let off_pre_ff_ln_2    = self.tensor_offset_required(&format!("{}.pre_feedforward_layernorm_2.weight", prefix));
        let off_post_ff_ln     = self.tensor_offset_required(&format!("{}.post_feedforward_layernorm.weight", prefix));
        let off_post_ff_ln_1   = self.tensor_offset_required(&format!("{}.post_feedforward_layernorm_1.weight", prefix));
        let off_post_ff_ln_2   = self.tensor_offset_required(&format!("{}.post_feedforward_layernorm_2.weight", prefix));
        // mlp.{gate,up,down}_proj and router.proj are looked up by
        // encode_matvec_auto — no upfront offset probe.
        let off_router_scale   = self.tensor_offset_required(&format!("{}.router.scale", prefix));
        let off_per_expert_sc  = self.tensor_offset_required(&format!("{}.router.per_expert_scale", prefix));
        // experts.{gate_up,down}_proj lookups happen inside
        // encode_matvec_expert_slice (the name structure differs between
        // BF16 — single tensor at the base name — and INT4 — three
        // sibling tensors at .weight/.scales/.biases).
        let off_layer_scalar   = self.tensor_offset_required(&format!("{}.layer_scalar", prefix));

        let wbuf = &self.weight_buffer.buf;
        let ctx = &self.ctx;

        // Snapshot residual.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.buf_hidden.contents() as *const f32,
                ctx.buf_ff_combined.contents() as *mut f32,
                C::HIDDEN_DIM,
            );
        }

        // Dense MLP path + Router.
        let cmd_buf2 = ctx.queue.new_command_buffer();
        let enc2 = cmd_buf2.new_compute_command_encoder();
        gk::encode_rms_norm_fused_bf16(ctx, enc2, &ctx.buf_hidden, 0, wbuf, off_pre_ff_ln, &ctx.buf_pre_ff_normed, 0, hidden_dim);
        self.encode_matvec_auto(enc2, &format!("{}.mlp.gate_proj", prefix),
            &ctx.buf_pre_ff_normed, 0, &ctx.buf_mlp_gate, 0, inter, hidden_dim);
        self.encode_matvec_auto(enc2, &format!("{}.mlp.up_proj", prefix),
            &ctx.buf_pre_ff_normed, 0, &ctx.buf_mlp_up, 0, inter, hidden_dim);
        gk::encode_gelu_fused(ctx, enc2, &ctx.buf_mlp_gate, 0, &ctx.buf_mlp_up, 0, &ctx.buf_mlp_act, 0, inter);
        self.encode_matvec_auto(enc2, &format!("{}.mlp.down_proj", prefix),
            &ctx.buf_mlp_act, 0, &ctx.buf_mlp_down, 0, hidden_dim, inter);
        gk::encode_rms_norm_fused_bf16(ctx, enc2, &ctx.buf_mlp_down, 0, wbuf, off_post_ff_ln_1, &ctx.buf_mlp_post, 0, hidden_dim);
        gk::encode_rms_norm_router(ctx, enc2, &ctx.buf_hidden, 0, wbuf, off_router_scale, &ctx.buf_router_normed, 0, hidden_dim);
        self.encode_matvec_auto(enc2, &format!("{}.router.proj", prefix),
            &ctx.buf_router_normed, 0, &ctx.buf_router_logits, 0, num_experts, hidden_dim);
        enc2.end_encoding();
        cmd_buf2.commit();
        cmd_buf2.wait_until_completed();

        // CPU: top-K + softmax + per_expert_scale.
        let logits_ptr = ctx.buf_router_logits.contents() as *const f32;
        let logits: &[f32] = unsafe { std::slice::from_raw_parts(logits_ptr, C::NUM_EXPERTS) };
        let k = self.num_active_experts;
        let mut idx: Vec<usize> = (0..C::NUM_EXPERTS).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top_idx: Vec<usize> = idx[..k].to_vec();
        let mut top_w: Vec<f32> = top_idx.iter().map(|&i| logits[i]).collect();
        let mx = top_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for w in top_w.iter_mut() { *w = (*w - mx).exp(); sum += *w; }
        for w in top_w.iter_mut() { *w /= sum; }
        let pes_ptr = self.weight_buffer.base as usize + off_per_expert_sc as usize;
        for (ki, &e) in top_idx.iter().enumerate() {
            let raw = unsafe { *(pes_ptr as *const u16).add(e) };
            top_w[ki] *= bf16_to_f32(raw);
        }

        // pre_feedforward_layernorm_2 → buf_pre_ff_normed_2.
        {
            let cb = ctx.queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            gk::encode_rms_norm_fused_bf16(ctx, e, &ctx.buf_hidden, 0, wbuf, off_pre_ff_ln_2, &ctx.buf_pre_ff_normed_2, 0, hidden_dim);
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Zero expert_out for accumulation.
        unsafe {
            std::ptr::write_bytes(ctx.buf_expert_out.contents() as *mut u8, 0, C::HIDDEN_DIM * 4);
        }

        // Experts loop (qwen35-style pread). For each top-K expert:
        //   1. pread the expert's INT4 blob from packed_experts/layer_XX.bin
        //      into expert_buffer.expert_data[ki] (~3.3 MB per expert).
        //   2. Dispatch matvecs reading from that small Metal buffer at the
        //      per-expert layout offsets (GATE_W_OFF, GATE_S_OFF, etc.).
        let expert_file = &self.model.expert_files[Self::layer_idx_from_prefix(prefix)];
        let expert_size = expert_file.expert_size();
        let eb = &self.expert_buffer;
        for (ki, &eidx) in top_idx.iter().enumerate() {
            // CPU pread (parallel-safe; rayon not used here for K=8).
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    eb.expert_data[ki].contents() as *mut u8, expert_size)
            };
            expert_file.read_expert(eidx, dst)
                .map_err(|e| MoEError::Io(std::io::Error::new(std::io::ErrorKind::Other,
                    format!("expert pread layer={} idx={}: {:?}", Self::layer_idx_from_prefix(prefix), eidx, e))))?;

            // GPU dispatch: per-expert matvecs from expert_data[ki]
            let cb = ctx.queue.new_command_buffer();
            let enc_e = cb.new_compute_command_encoder();
            // gate
            gk::encode_matvec_int4(ctx, enc_e,
                &eb.expert_data[ki], C::GATE_W_OFF as u64,
                &eb.expert_data[ki], C::GATE_S_OFF as u64,
                &eb.expert_data[ki], C::GATE_B_OFF as u64,
                &ctx.buf_pre_ff_normed_2, 0, &ctx.buf_expert_gate, 0,
                moe_inter, hidden_dim);
            // up
            gk::encode_matvec_int4(ctx, enc_e,
                &eb.expert_data[ki], C::UP_W_OFF as u64,
                &eb.expert_data[ki], C::UP_S_OFF as u64,
                &eb.expert_data[ki], C::UP_B_OFF as u64,
                &ctx.buf_pre_ff_normed_2, 0, &ctx.buf_expert_up, 0,
                moe_inter, hidden_dim);
            gk::encode_gelu_fused(ctx, enc_e,
                &ctx.buf_expert_gate, 0, &ctx.buf_expert_up, 0, &ctx.buf_expert_act, 0, moe_inter);
            // down
            gk::encode_matvec_int4(ctx, enc_e,
                &eb.expert_data[ki], C::DOWN_W_OFF as u64,
                &eb.expert_data[ki], C::DOWN_S_OFF as u64,
                &eb.expert_data[ki], C::DOWN_B_OFF as u64,
                &ctx.buf_expert_act, 0, &ctx.buf_expert_post, 0,
                hidden_dim, moe_inter);
            enc_e.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // CPU weighted-accumulate (shared memory).
            let post_ptr = ctx.buf_expert_post.contents() as *const f32;
            let out_ptr  = ctx.buf_expert_out.contents()  as *mut f32;
            let w_ki = top_w[ki];
            unsafe {
                for j in 0..C::HIDDEN_DIM { *out_ptr.add(j) += *post_ptr.add(j) * w_ki; }
            }
        }

        // Outer FFN.
        let cmd_buf4 = ctx.queue.new_command_buffer();
        let enc4 = cmd_buf4.new_compute_command_encoder();
        gk::encode_rms_norm_fused_bf16(ctx, enc4, &ctx.buf_expert_out,     0, wbuf, off_post_ff_ln_2, &ctx.buf_expert_post, 0, hidden_dim);
        gk::encode_residual_add(       ctx, enc4, &ctx.buf_mlp_post,       0, &ctx.buf_expert_post,   0, &ctx.buf_ff_outer_post, 0, hidden_dim);
        gk::encode_rms_norm_fused_bf16(ctx, enc4, &ctx.buf_ff_outer_post,  0, wbuf, off_post_ff_ln,   &ctx.buf_ff_outer_post, 0, hidden_dim);
        gk::encode_residual_add(       ctx, enc4, &ctx.buf_ff_combined,    0, &ctx.buf_ff_outer_post, 0, &ctx.buf_hidden, 0, hidden_dim);
        gk::encode_mul_scalar_bf16(    ctx, enc4, &ctx.buf_hidden,         0, wbuf, off_layer_scalar, hidden_dim);
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
                    self.forward_full_layer(layer, pos as u32)?;
                } else {
                    self.forward_sliding_layer(layer, pos as u32)?;
                }
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
