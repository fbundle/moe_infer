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
    _phantom: PhantomData<C>,
}

impl<C: Gemma4ModelConfig> Gemma4Fused<C> {
    pub fn new(
        model: Arc<Model>,
        num_active_experts: usize,
        expert_cache_count: usize,
    ) -> Result<Self, MoEError> {
        C::validate_config(&model.config).map_err(MoEError::Config)?;
        let ctx = Gemma4MetalContext::new::<C>(
            &model.weight_file, num_active_experts, "Gemma4Fused", expert_cache_count,
        )?;
        Ok(Self {
            model,
            ctx,
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
        // Per-token forward outline (TODO: implement). Based on actual
        // tensor names from unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit:
        //
        // For each token:
        //   for each layer:
        //     # ── Attention block ─────────────────────────────────────
        //     x = input_layernorm(hidden)
        //     q = q_proj(x);  q = q_norm(q)
        //     k = k_proj(x);  k = k_norm(k)
        //     v = v_proj(x)
        //     RoPE on q,k:
        //       sliding layers (5 of 6): theta=10k, rotary_dim=head_dim
        //       full layers (1 of 6):    theta=1M,  rotary_dim=head_dim/4
        //     KV write (sliding ring vs full append)
        //     attn_out = SDPA(q, k_cache, v_cache):
        //       sliding-masked for sliding layers (window=1024)
        //       full causal for full layers
        //     attn_out = o_proj(attn_out)
        //     hidden = hidden + post_attention_layernorm(attn_out)
        //
        //     # ── Dual FFN ────────────────────────────────────────────
        //     # Dense MLP path (always on; intermediate_size=2112):
        //     y_dense = pre_feedforward_layernorm(hidden)
        //     dense_out = mlp_down(gelu(mlp_gate(y_dense)) * mlp_up(y_dense))
        //     dense_out = post_feedforward_layernorm(dense_out)
        //
        //     # Sparse experts path (top-8 of 128; moe_intermediate=704):
        //     y_expert = pre_feedforward_layernorm_2(hidden)
        //     router_logits = router_proj(y_expert)
        //     router_logits = router_logits * router.scale + router.per_expert_scale
        //     router_logits = post_feedforward_layernorm_1(router_logits)  # ← TODO: verify
        //     idx, weights = top_k(softmax(router_logits), k=8)
        //     expert_out = 0
        //     for ki in 0..8:
        //       e = idx[ki]
        //       gate = SwitchGLU.gate_proj[e] @ y_expert
        //       up   = SwitchGLU.up_proj[e]   @ y_expert
        //       act  = gelu(gate) * up
        //       expert_out += weights[ki] * (SwitchGLU.down_proj[e] @ act)
        //     expert_out = post_feedforward_layernorm_2(expert_out)
        //
        //     # Combine paths and apply per-layer scalar:
        //     hidden = hidden + layer_scalar * (dense_out + expert_out)
        //     # (the exact gating/scaling formula needs verification against
        //     # mlx-lm's Gemma4DecoderLayer.forward — TODO.)
        //
        //   # ── Output ──────────────────────────────────────────────────
        //   x = model.norm(hidden)
        //   logits = x @ embed_tokens.weight.T   # tied — no separate lm_head
        //   logits = 30 * tanh(logits / 30)      # final softcap
        //
        // Return logits[n, vocab_size].
        //
        // CRITICAL TODOs before this can be implemented:
        //   1. Verify the dual-FFN combination formula against mlx-lm's
        //      Gemma4DecoderLayer source. The naming `_1` / `_2` suggests
        //      multiple stages but the actual graph topology needs reading.
        //   2. Verify `layer_scalar` semantics — multiplicative on the
        //      combined FFN output, or per-path, or something else.
        //   3. Verify whether `pre_feedforward_layernorm_2` is on the
        //      experts input or on something else (the router?).
        //   4. Verify whether Q is scaled by 1/sqrt(head_dim) or by the
        //      query_pre_attn_scalar (the config has both).

        unimplemented!("Gemma4Fused::forward_hidden — Phase 2 work");
    }

    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.timing.clone()
    }
}
