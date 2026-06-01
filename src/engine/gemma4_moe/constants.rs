//! Compile-time model-dimension trait for Gemma 4 MoE inference.
//!
//! Gemma 4 26B-A4B differs from Qwen3.6-35B-A3B in several important ways
//! that prevent reuse of the Qwen ModelConfig trait:
//!
//!   - No linear-attention / DeltaNet layers. Every non-full layer is a
//!     standard multi-head attention with a sliding-window mask
//!     (window=1024).
//!   - 5:1 sliding:full layer pattern (every 6th layer is full attention).
//!   - Two RoPE configurations — sliding layers use theta=10k, full layers
//!     use theta=1M with a 25% partial-rotary factor.
//!   - **Dual FFN per layer** (per MLX-LM source): a dense MLP at
//!     INTERMEDIATE_SIZE=2112 running on every token, PLUS a sparse
//!     experts path (128 experts, top-8) at MOE_INTERMEDIATE=704. Both
//!     paths use their own pre/post FFN norms (hence the _1 / _2 suffix
//!     variants in the tensor names). This is unlike Qwen3.6's shared-
//!     expert design where the "shared" part is a small 512-dim expert;
//!     in Gemma 4 the always-on path is the full dense MLP and the routed
//!     experts are extras on top.
//!   - GELU activation (gelu_pytorch_tanh) instead of SwiGLU.
//!   - Tied input/output word embeddings (no separate lm_head matrix).
//!   - Final logit softcapping: y = 30 * tanh(x / 30).
//!   - Vision + audio multimodal tokens (text-only inference ignores).
//!
//! So Gemma 4 gets its own trait `Gemma4ModelConfig` and its own engine
//! struct hierarchy parallel to FusedExp2/3, sharing only the underlying
//! Metal context, matvec kernels, and quant-pipeline infrastructure.

#![allow(dead_code)]

#[allow(non_snake_case)]
pub trait Gemma4ModelConfig: 'static {
    // Core transformer dims
    const HIDDEN_DIM: usize;
    const NUM_LAYERS: usize;
    const NUM_ATTN_HEADS: usize;
    const NUM_KV_HEADS: usize;
    const HEAD_DIM: usize;
    const VOCAB_SIZE: usize;
    const INTERMEDIATE_SIZE: usize;       // dense FFN intermediate (for non-MoE layers if any)

    // MoE (sparse experts path; runs on top of the always-on dense MLP)
    const NUM_EXPERTS: usize;
    const NUM_EXPERTS_PER_TOK: usize;
    const MOE_INTERMEDIATE: usize;        // per-expert intermediate size
    // NOTE: Gemma 4 has no small "shared expert" like Qwen3.6. Instead it
    // has the full dense MLP (sized INTERMEDIATE_SIZE) running on every
    // token in parallel with the sparse experts. See `INTERMEDIATE_SIZE`.

    // Attention type pattern
    /// Length of one period of the attention pattern. For 26B-A4B this is 6
    /// (5 sliding + 1 full repeating).
    const ATTN_PATTERN_PERIOD: usize;
    /// Within one period, which index (0-based) is the full-attention layer?
    /// For 26B-A4B this is 5 (layers 5, 11, 17, 23, 29 are full).
    const FULL_ATTN_INDEX_IN_PATTERN: usize;
    /// Sliding-window length in tokens.
    const SLIDING_WINDOW: usize;

    // RoPE — two configurations
    const ROPE_THETA_SLIDING: f64;
    const ROPE_THETA_FULL: f64;
    /// Fraction of head_dim that gets RoPE applied in full-attention layers
    /// (e.g., 0.25 → only first 25% of each head's dims are rotated).
    /// Sliding layers apply RoPE to the full head_dim.
    const PARTIAL_ROTARY_FRACTION_FULL: f32;

    // Gemma 4 specifics
    /// Soft-cap applied to final logits: logits = cap * tanh(logits / cap).
    /// `None` (encoded as 0.0) means no softcap.
    const FINAL_LOGIT_SOFTCAP: f32;
    /// Soft-cap applied to attention logits before softmax (Gemma 2 had this;
    /// Gemma 4 may or may not — check per-version).
    const ATTN_LOGIT_SOFTCAP: f32;
    /// Whether input embeddings are tied to the output projection
    /// (one matrix used for both).
    const TIED_WORD_EMBEDDINGS: bool;
    /// `Q_pre_attn_scalar`: scale applied to Q before attention. Gemma uses
    /// 1/sqrt(query_pre_attn_scalar) instead of the default 1/sqrt(head_dim).
    const QUERY_PRE_ATTN_SCALAR: f32;

    // KV dim (derived)
    const KV_DIM: usize;

    // Expert 4-bit packed layout (matches BQ4 scheme)
    const EXPERT_SIZE_4BIT: usize;
    const GATE_W_OFF: usize;
    const GATE_S_OFF: usize;
    const GATE_B_OFF: usize;
    const UP_W_OFF: usize;
    const UP_S_OFF: usize;
    const UP_B_OFF: usize;
    const DOWN_W_OFF: usize;
    const DOWN_S_OFF: usize;
    const DOWN_B_OFF: usize;
    const GATE_W_SIZE: usize;
    const GATE_S_SIZE: usize;
    const GATE_B_SIZE: usize;
    const UP_W_SIZE: usize;
    const UP_S_SIZE: usize;
    const UP_B_SIZE: usize;
    const DOWN_W_SIZE: usize;
    const DOWN_S_SIZE: usize;
    const DOWN_B_SIZE: usize;

    const EXPECTED_ARCHITECTURE: &'static str;

    /// Returns true if `layer_idx` is a full-attention layer; false for
    /// sliding-window. Default impl uses the period + index pattern.
    fn is_full_attn_layer(layer_idx: usize) -> bool {
        (layer_idx % Self::ATTN_PATTERN_PERIOD) == Self::FULL_ATTN_INDEX_IN_PATTERN
    }

    fn validate_config(c: &crate::model::config::ModelConfig) -> Result<(), String> {
        let mut errs = Vec::new();
        let get = |k| c.get_usize(k).unwrap_or(0);

        let archs: Vec<&str> = c.resolve("architectures")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        if !archs.iter().any(|a| *a == Self::EXPECTED_ARCHITECTURE) {
            errs.push(format!("architecture mismatch: found={:?}, expected=\"{}\"",
                archs, Self::EXPECTED_ARCHITECTURE));
        }

        if get("hidden_size") != Self::HIDDEN_DIM {
            errs.push(format!("hidden_size: config={}, const={}", get("hidden_size"), Self::HIDDEN_DIM));
        }
        if get("num_hidden_layers") != Self::NUM_LAYERS {
            errs.push(format!("num_hidden_layers: config={}, const={}", get("num_hidden_layers"), Self::NUM_LAYERS));
        }
        if get("num_attention_heads") != Self::NUM_ATTN_HEADS {
            errs.push(format!("num_attention_heads: config={}, const={}", get("num_attention_heads"), Self::NUM_ATTN_HEADS));
        }
        if get("num_key_value_heads") != Self::NUM_KV_HEADS {
            errs.push(format!("num_key_value_heads: config={}, const={}", get("num_key_value_heads"), Self::NUM_KV_HEADS));
        }
        if get("head_dim") != Self::HEAD_DIM {
            errs.push(format!("head_dim: config={}, const={}", get("head_dim"), Self::HEAD_DIM));
        }
        if get("vocab_size") != Self::VOCAB_SIZE {
            errs.push(format!("vocab_size: config={}, const={}", get("vocab_size"), Self::VOCAB_SIZE));
        }
        // MoE config lives under different keys for Gemma 4; tolerate both
        // "num_local_experts" and "num_experts".
        let cfg_num_experts = c.get_usize("num_local_experts")
            .unwrap_or_else(|| get("num_experts"));
        if cfg_num_experts != Self::NUM_EXPERTS {
            errs.push(format!("num_experts: config={}, const={}", cfg_num_experts, Self::NUM_EXPERTS));
        }
        if get("num_experts_per_tok") != Self::NUM_EXPERTS_PER_TOK {
            errs.push(format!("num_experts_per_tok: config={}, const={}", get("num_experts_per_tok"), Self::NUM_EXPERTS_PER_TOK));
        }
        if get("moe_intermediate_size") != Self::MOE_INTERMEDIATE {
            errs.push(format!("moe_intermediate_size: config={}, const={}", get("moe_intermediate_size"), Self::MOE_INTERMEDIATE));
        }
        if get("sliding_window") != Self::SLIDING_WINDOW {
            errs.push(format!("sliding_window: config={}, const={}", get("sliding_window"), Self::SLIDING_WINDOW));
        }

        if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
    }
}

// ─── Expert layout helper (shared shape with Qwen's BQ4 scheme) ───────

pub struct ExpertLayout {
    pub gate_w_off: usize,
    pub gate_s_off: usize,
    pub gate_b_off: usize,
    pub up_w_off: usize,
    pub up_s_off: usize,
    pub up_b_off: usize,
    pub down_w_off: usize,
    pub down_s_off: usize,
    pub down_b_off: usize,
    pub gate_w_size: usize,
    pub gate_s_size: usize,
    pub gate_b_size: usize,
    pub up_w_size: usize,
    pub up_s_size: usize,
    pub up_b_size: usize,
    pub down_w_size: usize,
    pub down_s_size: usize,
    pub down_b_size: usize,
    pub expert_size_4bit: usize,
}

pub const fn expert_layout(hd: usize, mi: usize, gs: usize) -> ExpertLayout {
    let gate_w = mi * hd / 2;
    let gate_sb = mi * (hd / gs) * 2;
    let up_w = mi * hd / 2;
    let up_sb = mi * (hd / gs) * 2;
    let down_w = hd * mi / 2;
    let down_sb = hd * (mi / gs) * 2;
    ExpertLayout {
        gate_w_off: 0,
        gate_s_off: gate_w,
        gate_b_off: gate_w + gate_sb,
        up_w_off: gate_w + 2 * gate_sb,
        up_s_off: gate_w + 2 * gate_sb + up_w,
        up_b_off: gate_w + 2 * gate_sb + up_w + up_sb,
        down_w_off: gate_w + 2 * gate_sb + up_w + 2 * up_sb,
        down_s_off: gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w,
        down_b_off: gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + down_sb,
        gate_w_size: gate_w, gate_s_size: gate_sb, gate_b_size: gate_sb,
        up_w_size: up_w, up_s_size: up_sb, up_b_size: up_sb,
        down_w_size: down_w, down_s_size: down_sb, down_b_size: down_sb,
        expert_size_4bit: gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + 2 * down_sb,
    }
}

// Gemma 4 26B-A4B dims: hidden=2816, moe_intermediate=704
const L26B: ExpertLayout = expert_layout(2816, 704, 64);

// ─── Marker types ─────────────────────────────────────────────────────

/// Gemma 4 26B-A4B: 30 layers, 128 experts, top-8 routing, 4B active.
pub struct Gemma4_26B_A4B;

impl Gemma4ModelConfig for Gemma4_26B_A4B {
    const HIDDEN_DIM: usize = 2816;
    const NUM_LAYERS: usize = 30;
    const NUM_ATTN_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 256;
    const VOCAB_SIZE: usize = 262144;
    const INTERMEDIATE_SIZE: usize = 2112;

    const NUM_EXPERTS: usize = 128;
    const NUM_EXPERTS_PER_TOK: usize = 8;
    const MOE_INTERMEDIATE: usize = 704;

    const ATTN_PATTERN_PERIOD: usize = 6;
    const FULL_ATTN_INDEX_IN_PATTERN: usize = 5;
    const SLIDING_WINDOW: usize = 1024;

    const ROPE_THETA_SLIDING: f64 = 10_000.0;
    const ROPE_THETA_FULL: f64 = 1_000_000.0;
    const PARTIAL_ROTARY_FRACTION_FULL: f32 = 0.25;

    const FINAL_LOGIT_SOFTCAP: f32 = 30.0;
    const ATTN_LOGIT_SOFTCAP: f32 = 0.0;    // Gemma 4 does not soft-cap attention logits.
    const TIED_WORD_EMBEDDINGS: bool = true;
    const QUERY_PRE_ATTN_SCALAR: f32 = 256.0;  // = HEAD_DIM; defaults to 1/sqrt(head_dim).

    const KV_DIM: usize = Self::NUM_KV_HEADS * Self::HEAD_DIM;  // 8 * 256 = 2048

    const EXPERT_SIZE_4BIT: usize = L26B.expert_size_4bit;
    const GATE_W_OFF: usize = L26B.gate_w_off;
    const GATE_S_OFF: usize = L26B.gate_s_off;
    const GATE_B_OFF: usize = L26B.gate_b_off;
    const UP_W_OFF: usize = L26B.up_w_off;
    const UP_S_OFF: usize = L26B.up_s_off;
    const UP_B_OFF: usize = L26B.up_b_off;
    const DOWN_W_OFF: usize = L26B.down_w_off;
    const DOWN_S_OFF: usize = L26B.down_s_off;
    const DOWN_B_OFF: usize = L26B.down_b_off;
    const GATE_W_SIZE: usize = L26B.gate_w_size;
    const GATE_S_SIZE: usize = L26B.gate_s_size;
    const GATE_B_SIZE: usize = L26B.gate_b_size;
    const UP_W_SIZE: usize = L26B.up_w_size;
    const UP_S_SIZE: usize = L26B.up_s_size;
    const UP_B_SIZE: usize = L26B.up_b_size;
    const DOWN_W_SIZE: usize = L26B.down_w_size;
    const DOWN_S_SIZE: usize = L26B.down_s_size;
    const DOWN_B_SIZE: usize = L26B.down_b_size;

    const EXPECTED_ARCHITECTURE: &'static str = "Gemma4ForConditionalGeneration";
}

// Small test variant — useful for verify_nway-style validation once we
// have a stripped Gemma 4. Dimensions match a hypothetical 4-layer
// 4-expert stripping of the full model (mirrors the qwen35_moe pattern).
pub struct Gemma4Stripped;

impl Gemma4ModelConfig for Gemma4Stripped {
    const HIDDEN_DIM: usize = 2816;
    const NUM_LAYERS: usize = 6;               // one full attention period
    const NUM_ATTN_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 256;
    const VOCAB_SIZE: usize = 262144;
    const INTERMEDIATE_SIZE: usize = 2112;

    const NUM_EXPERTS: usize = 4;
    const NUM_EXPERTS_PER_TOK: usize = 4;
    const MOE_INTERMEDIATE: usize = 704;

    const ATTN_PATTERN_PERIOD: usize = 6;
    const FULL_ATTN_INDEX_IN_PATTERN: usize = 5;
    const SLIDING_WINDOW: usize = 1024;

    const ROPE_THETA_SLIDING: f64 = 10_000.0;
    const ROPE_THETA_FULL: f64 = 1_000_000.0;
    const PARTIAL_ROTARY_FRACTION_FULL: f32 = 0.25;

    const FINAL_LOGIT_SOFTCAP: f32 = 30.0;
    const ATTN_LOGIT_SOFTCAP: f32 = 0.0;
    const TIED_WORD_EMBEDDINGS: bool = true;
    const QUERY_PRE_ATTN_SCALAR: f32 = 256.0;

    const KV_DIM: usize = Self::NUM_KV_HEADS * Self::HEAD_DIM;

    const EXPERT_SIZE_4BIT: usize = L26B.expert_size_4bit;
    const GATE_W_OFF: usize = L26B.gate_w_off;
    const GATE_S_OFF: usize = L26B.gate_s_off;
    const GATE_B_OFF: usize = L26B.gate_b_off;
    const UP_W_OFF: usize = L26B.up_w_off;
    const UP_S_OFF: usize = L26B.up_s_off;
    const UP_B_OFF: usize = L26B.up_b_off;
    const DOWN_W_OFF: usize = L26B.down_w_off;
    const DOWN_S_OFF: usize = L26B.down_s_off;
    const DOWN_B_OFF: usize = L26B.down_b_off;
    const GATE_W_SIZE: usize = L26B.gate_w_size;
    const GATE_S_SIZE: usize = L26B.gate_s_size;
    const GATE_B_SIZE: usize = L26B.gate_b_size;
    const UP_W_SIZE: usize = L26B.up_w_size;
    const UP_S_SIZE: usize = L26B.up_s_size;
    const UP_B_SIZE: usize = L26B.up_b_size;
    const DOWN_W_SIZE: usize = L26B.down_w_size;
    const DOWN_S_SIZE: usize = L26B.down_s_size;
    const DOWN_B_SIZE: usize = L26B.down_b_size;

    const EXPECTED_ARCHITECTURE: &'static str = "Gemma4ForConditionalGeneration_Stripped";
}
