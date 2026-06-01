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

use super::constants::Gemma4ModelConfig;
use super::metal_context::Gemma4MetalContext;

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
        // Per-token forward outline (TODO: implement):
        //
        // For each token:
        //   for each layer:
        //     input_layernorm  → buf_input
        //     q/k/v projections (matvec_bf16)
        //     q_norm, k_norm
        //     RoPE (sliding-variant for non-full, full-variant for full):
        //       sliding: theta=10k, rotary_dim = head_dim
        //       full:    theta=1M,  rotary_dim = head_dim / 4
        //     KV write (sliding ring vs full append)
        //     SDPA (sliding-masked vs full causal)
        //     o_proj
        //     residual + post_attention_layernorm
        //     MoE: router → top-8 → 128 experts → no shared expert
        //     pre_feedforward_layernorm before MoE? TODO: verify order
        //     gelu-gated FFN per expert
        //     down_proj
        //     combine + residual + post_feedforward_layernorm
        //
        // final_norm
        // lm_head — tied: dot against embed_tokens.weight^T
        // logit softcap: 30 * tanh(logits / 30)
        //
        // Return logits[n, vocab_size].

        unimplemented!("Gemma4Fused::forward_hidden — Phase 2 work");
    }

    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.timing.clone()
    }
}
