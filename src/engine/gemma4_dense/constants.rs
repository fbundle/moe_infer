/// Compile-time model dimensions for Gemma 4 12B dense.
///
/// **Critical:** sliding-attn layers and full-attn layers have DIFFERENT
/// shape profiles. The 2× head_dim on full-attn layers is the source of all
/// the bugs in the v1 engine. Verified from the `google/gemma-4-12B`
/// safetensors header:
///
/// | tensor                              | sliding (layer 0)     | full (layer 5)        |
/// | ----------------------------------- | --------------------- | --------------------- |
/// | self_attn.q_proj.weight             | [4096, 3840]          | [8192, 3840]          |
/// | self_attn.k_proj.weight             | [2048, 3840]          | [512,  3840]          |
/// | self_attn.v_proj.weight             | [2048, 3840]          | (absent — K=V trick)  |
/// | self_attn.o_proj.weight             | [3840, 4096]          | [3840, 8192]          |
/// | self_attn.{q,k}_norm.weight         | [256]                 | [512]                 |
///
/// Derived:
///   sliding: 16 q-heads × 256 head_dim,  8 KV heads × 256 = 2048
///   full:    16 q-heads × 512 head_dim,  1 KV head  × 512 = 512
///
/// RoPE on sliding is full head_dim (256); on full it's partial 0.25 × 512 = 128.
/// SDPA on full uses GQA 16:1 (all 16 q-heads share the single KV head).

#[allow(non_snake_case)]
pub trait ModelConfig: 'static {
    // ── Architecture-wide ──
    const HIDDEN_DIM: usize;
    const NUM_LAYERS: usize;
    const NUM_ATTN_HEADS: usize;
    const VOCAB_SIZE: usize;
    const INTERMEDIATE: usize;

    // ── Sliding-attn dims (the "default" everywhere except `is_full_attn_layer`) ──
    const HEAD_DIM_SLIDING: usize;
    const NUM_KV_HEADS_SLIDING: usize;
    const ROTARY_DIM_SLIDING: usize;
    const ROPE_THETA_SLIDING: f64;

    // ── Full-attn dims (every Nth layer; see FULL_ATTN_INTERVAL) ──
    const HEAD_DIM_FULL: usize;
    const NUM_KV_HEADS_FULL: usize;
    const ROTARY_DIM_FULL: usize;
    const ROPE_THETA_FULL: f64;

    // ── Pattern ──
    const FULL_ATTN_INTERVAL: usize;
    const NUM_FULL_ATTN_LAYERS: usize;
    const NUM_SLIDING_LAYERS: usize;
    const SLIDING_WINDOW: usize;

    // ── Output ──
    const FINAL_LOGIT_SOFTCAP: f32;

    // ── Derived helpers ──
    /// Sliding Q dim = NUM_ATTN_HEADS * HEAD_DIM_SLIDING.
    const Q_DIM_SLIDING: usize;
    /// Sliding KV dim = NUM_KV_HEADS_SLIDING * HEAD_DIM_SLIDING.
    const KV_DIM_SLIDING: usize;
    /// Full Q dim = NUM_ATTN_HEADS * HEAD_DIM_FULL.
    const Q_DIM_FULL: usize;
    /// Full KV dim = NUM_KV_HEADS_FULL * HEAD_DIM_FULL.
    const KV_DIM_FULL: usize;

    const EXPECTED_ARCHITECTURE: &'static str;

    /// True for full-attn layers — every Nth layer (e.g., layer 5, 11, … on 12B).
    #[inline]
    fn is_full_attn_layer(i: usize) -> bool {
        (i + 1) % Self::FULL_ATTN_INTERVAL == 0
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

        macro_rules! check {
            ($key:expr, $const_val:expr) => {
                if get($key) != $const_val {
                    errs.push(format!("{}: config={}, const={}",
                        $key, get($key), $const_val));
                }
            };
        }
        check!("hidden_size", Self::HIDDEN_DIM);
        check!("num_hidden_layers", Self::NUM_LAYERS);
        check!("num_attention_heads", Self::NUM_ATTN_HEADS);
        check!("vocab_size", Self::VOCAB_SIZE);
        check!("intermediate_size", Self::INTERMEDIATE);
        check!("sliding_window", Self::SLIDING_WINDOW);
        // Note: config has `head_dim`=256 (sliding) — we don't validate full
        // dims since the HF config doesn't store them as a single flat key.

        if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
    }
}

// ─── Gemma 4 12B ────────────────────────────────────────────────────────────

pub struct Gemma4Dense12B;

impl ModelConfig for Gemma4Dense12B {
    const HIDDEN_DIM: usize = 3840;
    const NUM_LAYERS: usize = 48;
    const NUM_ATTN_HEADS: usize = 16;
    const VOCAB_SIZE: usize = 262144;
    const INTERMEDIATE: usize = 15360;

    const HEAD_DIM_SLIDING: usize = 256;
    const NUM_KV_HEADS_SLIDING: usize = 8;
    const ROTARY_DIM_SLIDING: usize = 256;   // full head_dim
    const ROPE_THETA_SLIDING: f64 = 10_000.0;

    const HEAD_DIM_FULL: usize = 512;
    const NUM_KV_HEADS_FULL: usize = 1;
    const ROTARY_DIM_FULL: usize = 128;      // 0.25 * 512
    const ROPE_THETA_FULL: f64 = 1_000_000.0;

    const FULL_ATTN_INTERVAL: usize = 6;
    const NUM_FULL_ATTN_LAYERS: usize = 8;
    const NUM_SLIDING_LAYERS: usize = 40;
    const SLIDING_WINDOW: usize = 1024;

    const FINAL_LOGIT_SOFTCAP: f32 = 30.0;

    const Q_DIM_SLIDING: usize = 16 * 256;     // 4096
    const KV_DIM_SLIDING: usize = 8 * 256;     // 2048
    const Q_DIM_FULL: usize = 16 * 512;        // 8192
    const KV_DIM_FULL: usize = 1 * 512;        // 512

    const EXPECTED_ARCHITECTURE: &'static str = "Gemma4UnifiedForConditionalGeneration";
}

// ─── Stripped variant for verification ──────────────────────────────────────

pub struct Gemma4Dense12BStripped;

impl ModelConfig for Gemma4Dense12BStripped {
    const HIDDEN_DIM: usize = 3840;
    const NUM_LAYERS: usize = 6;
    const NUM_ATTN_HEADS: usize = 16;
    const VOCAB_SIZE: usize = 262144;
    const INTERMEDIATE: usize = 15360;

    const HEAD_DIM_SLIDING: usize = 256;
    const NUM_KV_HEADS_SLIDING: usize = 8;
    const ROTARY_DIM_SLIDING: usize = 256;
    const ROPE_THETA_SLIDING: f64 = 10_000.0;

    const HEAD_DIM_FULL: usize = 512;
    const NUM_KV_HEADS_FULL: usize = 1;
    const ROTARY_DIM_FULL: usize = 128;
    const ROPE_THETA_FULL: f64 = 1_000_000.0;

    const FULL_ATTN_INTERVAL: usize = 6;
    const NUM_FULL_ATTN_LAYERS: usize = 1;
    const NUM_SLIDING_LAYERS: usize = 5;
    const SLIDING_WINDOW: usize = 1024;

    const FINAL_LOGIT_SOFTCAP: f32 = 30.0;

    const Q_DIM_SLIDING: usize = 16 * 256;
    const KV_DIM_SLIDING: usize = 8 * 256;
    const Q_DIM_FULL: usize = 16 * 512;
    const KV_DIM_FULL: usize = 1 * 512;

    const EXPECTED_ARCHITECTURE: &'static str = "Gemma4UnifiedForConditionalGeneration_Stripped";
}
