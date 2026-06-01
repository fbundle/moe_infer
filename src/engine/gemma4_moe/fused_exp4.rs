//! Gemma4 FusedExp4 — performance variant on top of FusedExp3.
//!
//! FusedExp3 is a faithful port of mlx-vlm's `DecoderLayer.__call__`.
//! FusedExp4 adds three optimisations found in the mlx-vlm reference but
//! that require non-trivial engine work to capture. None of these is
//! exotic — they're all just engine plumbing that mlx-vlm gets "for free"
//! from MLX's stack and we have to implement explicitly.
//!
//! Status: scaffold only. Compiles, not registered in DynEngine. Each
//! optimisation has a TODO with the concrete change it requires.
//!
//! ── Optimisation 1: RotatingKVCache for sliding-attention layers ───────
//!
//! Source: `vendor/mlx-vlm/mlx_vlm/models/cache.py::RotatingKVCache`
//! Used by Gemma 4: see `language.py::Gemma4TextModel.make_cache` —
//!   sliding-attention layers use `RotatingKVCache(max_size=sliding_window)`,
//!   not the unbounded KVCache.
//!
//! Why it matters: 25 of 30 layers in 26B-A4B are sliding-attention with
//! window=1024. Without a ring buffer, each of those layers' KV cache
//! grows unboundedly with context. With a ring buffer they cap at
//! 1024 × kv_dim per layer, regardless of context length.
//!
//! Concrete savings on Gemma 4 26B-A4B at 100K tokens:
//!   - Without ring buffer: 25 layers × 100K × 2 × 2048 × 2 bytes = ~20 GB
//!   - With ring buffer:    25 layers × 1024 × 2 × 2048 × 2 bytes = ~210 MB
//!   - Difference: ~95× less KV memory across the sliding-attn layers.
//!
//! FusedExp3 (this branch's baseline) is expected to use unbounded KV for
//! ALL layers because that mirrors the simplest reading of attention. The
//! KV cache memory is then the limiting factor for context length.
//! FusedExp4's ring buffer for sliding layers lifts that limit.
//!
//! Implementation sketch:
//!   - Per-sliding-layer Buffer of shape [sliding_window, kv_dim]
//!   - A `write_index` cursor that wraps modulo sliding_window
//!   - sliding-attention kernel reads positions
//!       [(write_index - min(seq_len, sliding_window)) mod sliding_window
//!         ..  write_index]  modulo sliding_window
//!   - Mask needs to encode "valid_count" so positions beyond what's been
//!     written so far are ignored on early tokens
//!
//! Caveats:
//!   - Token positions for RoPE come from the absolute sequence position,
//!     not the ring-buffer slot. So K cached BEFORE RoPE (in mlx-vlm
//!     `RotatingKVCache` stores raw projections) and RoPE is applied at
//!     query-time. Inverts our current ordering — needs careful kernel
//!     refactor.
//!
//! ── Optimisation 2: K-equals-V on full-attention layers ────────────────
//!
//! Source: `language.py::Attention.__init__` line ~156:
//!   self.use_k_eq_v = (
//!       getattr(config, "attention_k_eq_v", False) and not self.is_sliding
//!   )
//!
//! For Gemma 4 26B-A4B, `attention_k_eq_v=True`, so the 5 full-attention
//! layers re-use the raw K projection (BEFORE k_norm) as the V tensor.
//! Concretely, `v_proj` is NOT a separate weight matrix for those layers;
//! Values are just `K_pre_norm`, then run through `v_norm` (a RMSNorm-no-
//! scale — no learnable weight).
//!
//! Implications:
//!   - Skip v_proj matvec on full layers: saves
//!       hidden_dim × kv_dim × bytes_per_param  per full layer per token.
//!       For 26B-A4B: 2816 × (8 × 512) × 0.5 (INT4) ≈ 5.8 MB per full
//!       layer per token of weight bandwidth saved.
//!       Across 5 full layers: ~29 MB/token saved.
//!   - v_norm has no learnable weight: skip the weight load entirely.
//!   - For full layers num_global_key_value_heads is used (different from
//!     num_key_value_heads for sliding); head_dim becomes global_head_dim
//!     (512 for 26B-A4B vs 256 for sliding).
//!
//! FusedExp3 will likely treat full and sliding layers uniformly (always
//! run v_proj). FusedExp4 special-cases the full layers' attention to
//! skip v_proj and bind v_norm with no weight.
//!
//! ── Optimisation 3: Per-layer-type mask deduplication ──────────────────
//!
//! Source: `language.py::Gemma4TextModel._make_masks` (lines 448-468).
//! Builds one mask object per layer TYPE (full or sliding), not per layer.
//! For 30 layers split 25/5, that's 2 mask objects vs 30.
//!
//! Modest saving — mask construction is cheap — but matters at long
//! context where the sliding-attention mask has a non-trivial structure
//! (causal + window) and we'd otherwise rebuild it 25× per forward.
//!
//! Implementation: in `forward_hidden`, build `full_mask` and
//! `sliding_mask` once per call, pass the appropriate one to each layer.
//!
//! ── Optimisation 4 (future, not in mlx-vlm directly): batched experts ──
//!
//! Carry the unique-expert pool design from FusedExp3-equivalent (qwen35
//! batched prefill) across to Gemma 4. Each token's top-8 experts come
//! from a pool of 128; with batching we pread each unique expert once per
//! layer rather than per token. For prefill workloads on Gemma 4 this
//! should give the same ~2× speedup we measured on Qwen3.6.

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

/// FusedExp4: FusedExp3 + ring-buffered sliding KV cache + k_eq_v on full
/// layers + per-layer-type mask deduplication.
///
/// Currently stubbed. Will share most of FusedExp3's surface; the
/// optimisations are localised to:
///   - sliding-attention kernel + KV-cache allocation
///   - attention dispatcher (skips v_proj on full layers)
///   - `forward_hidden` mask construction
pub struct Gemma4FusedExp4<C: Gemma4ModelConfig> {
    pub model: Arc<Model>,
    pub ctx: Gemma4MetalContext,
    pub weight_buffer: crate::engine::metal_context::WeightBuffer,
    pub expert_buffer: crate::engine::metal_context::ExpertBuffer,
    pub num_active_experts: usize,
    pub timing: BTreeMap<String, TelemetryValue>,
    pub last_h_pre_norm: Vec<f32>,
    _phantom: PhantomData<C>,
}

impl<C: Gemma4ModelConfig> Gemma4FusedExp4<C> {
    pub fn new(
        model: Arc<Model>,
        num_active_experts: usize,
        expert_cache_count: usize,
    ) -> Result<Self, MoEError> {
        C::validate_config(&model.config).map_err(MoEError::Config)?;
        let (ctx, weight_buffer, expert_buffer) = Gemma4MetalContext::new::<C>(
            &model.weight_file, num_active_experts, "Gemma4FusedExp4", expert_cache_count,
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

impl<C: Gemma4ModelConfig> Engine for Gemma4FusedExp4<C> {
    fn upload_cache(&self, _cache: &Cache) {
        // TODO: same as fused_exp3 plus uploading the sliding-layer ring
        // buffers' (data, write_index, valid_count) triples.
    }

    fn download_cache(&self, _cache: &mut Cache) {
        // TODO: matching download.
    }

    fn engine_pos(&self) -> usize { 0 }

    fn embed_lookup(&self, _token_ids: &[i64], _embeddings: &mut [f32]) {
        unimplemented!("Gemma4FusedExp4::embed_lookup");
    }

    fn forward_hidden(
        &mut self,
        _embeddings: &[f32],
        _check_signal: SignalCheckFn<'_>,
        _mtp: bool,
    ) -> Result<Vec<f32>, MoEError> {
        // Same overall flow as Gemma4Fused (fused_exp3), with these
        // localised changes:
        //
        //   1. Build per-layer-type masks ONCE per call:
        //        full_mask     = causal mask up to seq_len
        //        sliding_mask  = causal + window mask up to seq_len
        //
        //   2. Sliding-attn layers use ring-buffered KV:
        //        write_index = (pos + n_new) mod sliding_window
        //        valid_count = min(pos + n_new, sliding_window)
        //        SDPA kernel reads kv[valid_count slots ending at write_index]
        //
        //   3. Full-attn layers with attention_k_eq_v skip v_proj:
        //        v_raw = k_proj(x)              # before k_norm
        //        k     = k_norm(k_raw)
        //        v     = v_norm(v_raw)          # RMSNorm-no-scale; no weights to load
        //      And the engine never allocates `v_proj_*` weight references
        //      for full layers (the model file won't contain them either).
        //
        // Otherwise identical to fused_exp3.
        unimplemented!("Gemma4FusedExp4::forward_hidden — Phase 2+ work");
    }

    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.timing.clone()
    }
}
