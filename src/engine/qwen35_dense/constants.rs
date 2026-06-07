/// Compile-time model dimensions for the dense Qwen3.5 family (e.g. Qwen3.5-4B).
///
/// Same DeltaNet+GatedAttn backbone as Qwen3.5-MoE but with a dense MLP at every
/// layer instead of a 256-expert MoE block. No expert pool, no routing. MTP is
/// not modeled in this engine (skipped for v1 — the MTP layer's weights live in
/// the quantized output but the forward pass ignores them).

#[allow(non_snake_case)]
pub trait ModelConfig: 'static {
    const HIDDEN_DIM: usize;
    const NUM_LAYERS: usize;
    const NUM_ATTN_HEADS: usize;
    const NUM_KV_HEADS: usize;
    const HEAD_DIM: usize;
    const VOCAB_SIZE: usize;

    /// Dense MLP intermediate dim (gate/up/down inner dim). Replaces MoE consts.
    const DENSE_INTERMEDIATE: usize;

    const LINEAR_NUM_V_HEADS: usize;
    const LINEAR_NUM_K_HEADS: usize;
    const LINEAR_KEY_DIM: usize;
    const LINEAR_VALUE_DIM: usize;
    const LINEAR_TOTAL_KEY: usize;
    const LINEAR_TOTAL_VALUE: usize;
    const LINEAR_CONV_DIM: usize;

    const ROPE_THETA: f64;
    const ROTARY_DIM: usize;

    const NUM_FULL_ATTN_LAYERS: usize;
    const NUM_LINEAR_LAYERS: usize;

    /// KV-cache projection dim = NUM_KV_HEADS * HEAD_DIM.
    const KV_DIM: usize;

    const EXPECTED_ARCHITECTURE: &'static str;

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

        let lnum_k = get("linear_num_key_heads");
        let lkey_dim = get("linear_key_head_dim");
        let lnum_v = get("linear_num_value_heads");
        let lval_dim = get("linear_value_head_dim");

        if get("hidden_size") != Self::HIDDEN_DIM { errs.push(format!("hidden_size: config={}, const={}", get("hidden_size"), Self::HIDDEN_DIM)); }
        if get("num_attention_heads") != Self::NUM_ATTN_HEADS { errs.push(format!("num_attention_heads: config={}, const={}", get("num_attention_heads"), Self::NUM_ATTN_HEADS)); }
        if get("num_key_value_heads") != Self::NUM_KV_HEADS { errs.push(format!("num_key_value_heads: config={}, const={}", get("num_key_value_heads"), Self::NUM_KV_HEADS)); }
        if get("head_dim") != Self::HEAD_DIM { errs.push(format!("head_dim: config={}, const={}", get("head_dim"), Self::HEAD_DIM)); }
        if get("vocab_size") != Self::VOCAB_SIZE { errs.push(format!("vocab_size: config={}, const={}", get("vocab_size"), Self::VOCAB_SIZE)); }
        if get("intermediate_size") != Self::DENSE_INTERMEDIATE { errs.push(format!("intermediate_size: config={}, const={}", get("intermediate_size"), Self::DENSE_INTERMEDIATE)); }
        if lnum_v != Self::LINEAR_NUM_V_HEADS { errs.push(format!("linear_num_value_heads: config={}, const={}", lnum_v, Self::LINEAR_NUM_V_HEADS)); }
        if lnum_k != Self::LINEAR_NUM_K_HEADS { errs.push(format!("linear_num_key_heads: config={}, const={}", lnum_k, Self::LINEAR_NUM_K_HEADS)); }
        if lnum_k * lkey_dim != Self::LINEAR_TOTAL_KEY { errs.push(format!("linear_total_key: config={}, const={}", lnum_k * lkey_dim, Self::LINEAR_TOTAL_KEY)); }
        if lnum_v * lval_dim != Self::LINEAR_TOTAL_VALUE { errs.push(format!("linear_total_value: config={}, const={}", lnum_v * lval_dim, Self::LINEAR_TOTAL_VALUE)); }

        // HF `num_hidden_layers` is the MAIN layer count (excludes MTP). Match directly.
        if get("num_hidden_layers") != Self::NUM_LAYERS {
            errs.push(format!("num_hidden_layers: config={}, const={}",
                get("num_hidden_layers"), Self::NUM_LAYERS));
        }

        if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
    }
}

// ─── Qwen3.5-4B marker ───────────────────────────────────────────────────────

/// Qwen3.5-4B: 32 main layers (24 linear-attn + 8 full-attn), hidden 2560,
/// intermediate 9216, 16 q-heads / 4 kv-heads (GQA 4:1), head_dim 256, vocab
/// 248320, tied word embeddings. 256K max position.
pub struct Qwen35Dense4B;

impl ModelConfig for Qwen35Dense4B {
    const HIDDEN_DIM: usize = 2560;
    const NUM_LAYERS: usize = 32;
    const NUM_ATTN_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 256;
    const VOCAB_SIZE: usize = 248320;
    const DENSE_INTERMEDIATE: usize = 9216;

    const LINEAR_NUM_V_HEADS: usize = 32;
    const LINEAR_NUM_K_HEADS: usize = 16;
    const LINEAR_KEY_DIM: usize = 128;
    const LINEAR_VALUE_DIM: usize = 128;
    const LINEAR_TOTAL_KEY: usize = 2048;     // 16 * 128
    const LINEAR_TOTAL_VALUE: usize = 4096;   // 32 * 128
    const LINEAR_CONV_DIM: usize = 8192;      // qkv stack for DeltaNet (matches MoE sibling)

    const ROPE_THETA: f64 = 10_000_000.0;
    const ROTARY_DIM: usize = 64;             // head_dim * partial_rotary_factor = 256 * 0.25

    const NUM_FULL_ATTN_LAYERS: usize = 8;    // every 4th: layers 3, 7, 11, 15, 19, 23, 27, 31
    const NUM_LINEAR_LAYERS: usize = 24;

    const KV_DIM: usize = 1024;               // 4 KV heads * 256 head_dim

    const EXPECTED_ARCHITECTURE: &'static str = "Qwen3_5ForConditionalGeneration";
}

/// Returns true if layer `i` (0-indexed) is full-attention. Matches the config
/// pattern: full at every `FULL_ATTN_INTERVAL`-th position (3, 7, 11, …).
#[inline]
pub fn is_full_attn_layer(i: usize) -> bool {
    (i + 1) % crate::constants::FULL_ATTN_INTERVAL == 0
}

/// Stripped Qwen3.5-Dense: 4 layers (3 linear + 1 full) — used for verifying
/// kernel-level numeric correctness without compounding error across 32 layers.
pub struct Qwen35DenseStripped;

impl ModelConfig for Qwen35DenseStripped {
    const HIDDEN_DIM: usize = 2560;
    const NUM_LAYERS: usize = 4;
    const NUM_ATTN_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 256;
    const VOCAB_SIZE: usize = 248320;
    const DENSE_INTERMEDIATE: usize = 9216;

    const LINEAR_NUM_V_HEADS: usize = 32;
    const LINEAR_NUM_K_HEADS: usize = 16;
    const LINEAR_KEY_DIM: usize = 128;
    const LINEAR_VALUE_DIM: usize = 128;
    const LINEAR_TOTAL_KEY: usize = 2048;
    const LINEAR_TOTAL_VALUE: usize = 4096;
    const LINEAR_CONV_DIM: usize = 8192;

    const ROPE_THETA: f64 = 10_000_000.0;
    const ROTARY_DIM: usize = 64;

    const NUM_FULL_ATTN_LAYERS: usize = 1;
    const NUM_LINEAR_LAYERS: usize = 3;

    const KV_DIM: usize = 1024;

    const EXPECTED_ARCHITECTURE: &'static str = "Qwen3_5ForConditionalGeneration_Stripped";
}
