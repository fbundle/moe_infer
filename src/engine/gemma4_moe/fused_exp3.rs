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
use crate::model::Model;

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
        // TODO: read from context once added.
        0
    }

    fn embed_lookup(&self, _token_ids: &[i64], _embeddings: &mut [f32]) {
        // TODO: dequantize a row of model.embed_tokens.weight.
        // Note: Gemma 4 scales embeddings by sqrt(hidden_dim) before use
        // (or applies an "embedding scale" — TODO: verify against HF model).
        unimplemented!("Gemma4Fused::embed_lookup");
    }

    fn forward_hidden(
        &mut self,
        _embeddings: &[f32],
        _check_signal: SignalCheckFn<'_>,
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

        unimplemented!("Gemma4Fused::forward_hidden — Phase 2 work");
    }

    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.timing.clone()
    }
}
