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

            // Chunk [1]: bring this token's pre-scaled embedding in.
            // `embeddings` was already produced by self.embed_lookup() so
            // the sqrt(HIDDEN) scaling is already baked in.
            hidden_cpu.copy_from_slice(&embeddings[ti * hidden_dim..(ti + 1) * hidden_dim]);

            // Chunks [2,4,5]: per-layer forward. Sliding layers use the new
            // attn_sdpa_sliding_causal + reused q/k_head_norm_rope with
            // theta=10k. Full layers (head_dim=512, partial RoPE) need a
            // separate kernel — see Task 4.
            for layer in 0..num_layers {
                if check_signal() {
                    return Err(MoEError::Metal("interrupted".into()));
                }
                if C::is_full_attn_layer(layer) {
                    unimplemented!(
                        "Gemma 4 full-attention layer {} (every 6th, head_dim=512, \
                         partial RoPE, attention_k_eq_v): needs a separate SDPA \
                         kernel — see Task 4 (head_dim=512 SDPA) in Phase 2 plan.",
                        layer
                    );
                }
                let _ = (layer, pos, &self.weight_buffer, &self.expert_buffer);
                unimplemented!(
                    "Gemma 4 sliding-attention layer {} (head_dim=256, window=1024): \
                     per-layer dispatcher not yet wired. Sliding-SDPA + \
                     rms_norm_no_scale kernels are in place; remaining work is \
                     pure orchestration (matvec dispatches + dual-FFN) in \
                     fused_exp3.rs — see Task 3 in Phase 2 plan.",
                    layer
                );
                #[allow(unreachable_code)]
                {
                    // Unreachable today; sketches the call structure so the
                    // dispatcher can be added incrementally without changing
                    // the surrounding control flow.
                    pos = pos;
                }
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
