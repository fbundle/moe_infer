/// Compile-time model-dimenssion trait for Qwen3.6 MoE inference.
///
/// Implemented by marker types in this module so engine code can be generic
/// over `C: ModelConfig` instead of duplicating files per variant.

#[allow(non_snake_case)]
pub trait ModelConfig: 'static {
    const HIDDEN_DIM: usize;
    const NUM_LAYERS: usize;
    const NUM_ATTN_HEADS: usize;
    const NUM_KV_HEADS: usize;
    const HEAD_DIM: usize;
    const VOCAB_SIZE: usize;
    const NUM_EXPERTS: usize;
    const NUM_EXPERTS_PER_TOK: usize;
    const MOE_INTERMEDIATE: usize;
    const SHARED_INTERMEDIATE: usize;

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

    const KV_DIM: usize;

    // Expert 4-bit packed layout
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

    fn validate_config(hidden_dim: usize, num_layers: usize, num_experts: usize,
                       num_experts_per_tok: usize, moe_intermediate: usize,
                       shared_intermediate: usize, num_attn_heads: usize,
                       num_kv_heads: usize, head_dim: usize, vocab_size: usize,
                       linear_num_v_heads: usize, linear_num_k_heads: usize,
                       linear_total_key: usize, linear_total_value: usize,
                       architectures_str: &str,
    ) -> Result<(), String> {
        let mut errs = Vec::new();
        if !architectures_str.is_empty()
            && !architectures_str.split(',').any(|a| a.trim() == Self::EXPECTED_ARCHITECTURE)
        {
            errs.push(format!(
                "architecture mismatch: config={:?}, expected=\"{}\"",
                architectures_str, Self::EXPECTED_ARCHITECTURE));
        }
        if hidden_dim != Self::HIDDEN_DIM { errs.push(format!("hidden_dim: config={}, const={}", hidden_dim, Self::HIDDEN_DIM)); }
        if num_layers != Self::NUM_LAYERS { errs.push(format!("num_layers: config={}, const={}", num_layers, Self::NUM_LAYERS)); }
        if num_experts != Self::NUM_EXPERTS { errs.push(format!("num_experts: config={}, const={}", num_experts, Self::NUM_EXPERTS)); }
        if num_experts_per_tok != Self::NUM_EXPERTS_PER_TOK { errs.push(format!("num_experts_per_tok: config={}, const={}", num_experts_per_tok, Self::NUM_EXPERTS_PER_TOK)); }
        if moe_intermediate != Self::MOE_INTERMEDIATE { errs.push(format!("moe_intermediate: config={}, const={}", moe_intermediate, Self::MOE_INTERMEDIATE)); }
        if shared_intermediate != Self::SHARED_INTERMEDIATE { errs.push(format!("shared_intermediate: config={}, const={}", shared_intermediate, Self::SHARED_INTERMEDIATE)); }
        if num_attn_heads != Self::NUM_ATTN_HEADS { errs.push(format!("num_attn_heads: config={}, const={}", num_attn_heads, Self::NUM_ATTN_HEADS)); }
        if num_kv_heads != Self::NUM_KV_HEADS { errs.push(format!("num_kv_heads: config={}, const={}", num_kv_heads, Self::NUM_KV_HEADS)); }
        if head_dim != Self::HEAD_DIM { errs.push(format!("head_dim: config={}, const={}", head_dim, Self::HEAD_DIM)); }
        if vocab_size != Self::VOCAB_SIZE { errs.push(format!("vocab_size: config={}, const={}", vocab_size, Self::VOCAB_SIZE)); }
        if linear_num_v_heads != Self::LINEAR_NUM_V_HEADS { errs.push(format!("linear_num_v_heads: config={}, const={}", linear_num_v_heads, Self::LINEAR_NUM_V_HEADS)); }
        if linear_num_k_heads != Self::LINEAR_NUM_K_HEADS { errs.push(format!("linear_num_k_heads: config={}, const={}", linear_num_k_heads, Self::LINEAR_NUM_K_HEADS)); }
        if linear_total_key != Self::LINEAR_TOTAL_KEY { errs.push(format!("linear_total_key: config={}, const={}", linear_total_key, Self::LINEAR_TOTAL_KEY)); }
        if linear_total_value != Self::LINEAR_TOTAL_VALUE { errs.push(format!("linear_total_value: config={}, const={}", linear_total_value, Self::LINEAR_TOTAL_VALUE)); }
        if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
    }
}

// ─── Expert layout helper ─────────────────────────────────────────────

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

const L: ExpertLayout = expert_layout(2048, 512, 64);

// ─── Marker types ─────────────────────────────────────────────────────

/// Full model: 40 layers, 256 experts, 8 experts-per-tok.
pub struct FullModel;

impl ModelConfig for FullModel {
    const HIDDEN_DIM: usize = 2048;
    const NUM_LAYERS: usize = 40;
    const NUM_ATTN_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const VOCAB_SIZE: usize = 248320;
    const NUM_EXPERTS: usize = 256;
    const NUM_EXPERTS_PER_TOK: usize = 8;
    const MOE_INTERMEDIATE: usize = 512;
    const SHARED_INTERMEDIATE: usize = 512;
    const LINEAR_NUM_V_HEADS: usize = 32;
    const LINEAR_NUM_K_HEADS: usize = 16;
    const LINEAR_KEY_DIM: usize = 128;
    const LINEAR_VALUE_DIM: usize = 128;
    const LINEAR_TOTAL_KEY: usize = 2048;
    const LINEAR_TOTAL_VALUE: usize = 4096;
    const LINEAR_CONV_DIM: usize = 8192;
    const ROPE_THETA: f64 = 10_000_000.0;
    const ROTARY_DIM: usize = 64;
    const NUM_FULL_ATTN_LAYERS: usize = 10;
    const NUM_LINEAR_LAYERS: usize = 30;
    const KV_DIM: usize = 512;

    const EXPERT_SIZE_4BIT: usize = L.expert_size_4bit;
    const GATE_W_OFF: usize = L.gate_w_off;
    const GATE_S_OFF: usize = L.gate_s_off;
    const GATE_B_OFF: usize = L.gate_b_off;
    const UP_W_OFF: usize = L.up_w_off;
    const UP_S_OFF: usize = L.up_s_off;
    const UP_B_OFF: usize = L.up_b_off;
    const DOWN_W_OFF: usize = L.down_w_off;
    const DOWN_S_OFF: usize = L.down_s_off;
    const DOWN_B_OFF: usize = L.down_b_off;
    const GATE_W_SIZE: usize = L.gate_w_size;
    const GATE_S_SIZE: usize = L.gate_s_size;
    const GATE_B_SIZE: usize = L.gate_b_size;
    const UP_W_SIZE: usize = L.up_w_size;
    const UP_S_SIZE: usize = L.up_s_size;
    const UP_B_SIZE: usize = L.up_b_size;
    const DOWN_W_SIZE: usize = L.down_w_size;
    const DOWN_S_SIZE: usize = L.down_s_size;
    const DOWN_B_SIZE: usize = L.down_b_size;
    const EXPECTED_ARCHITECTURE: &'static str = "Qwen3_5MoeForConditionalGeneration";
}

/// Stripped model: 4 layers, 4 experts, 4 experts-per-tok (test model).
pub struct StrippedModel;

impl ModelConfig for StrippedModel {
    const HIDDEN_DIM: usize = 2048;
    const NUM_LAYERS: usize = 4;
    const NUM_ATTN_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const VOCAB_SIZE: usize = 248320;
    const NUM_EXPERTS: usize = 4;
    const NUM_EXPERTS_PER_TOK: usize = 4;
    const MOE_INTERMEDIATE: usize = 512;
    const SHARED_INTERMEDIATE: usize = 512;
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
    const KV_DIM: usize = 512;

    const EXPERT_SIZE_4BIT: usize = L.expert_size_4bit;
    const GATE_W_OFF: usize = L.gate_w_off;
    const GATE_S_OFF: usize = L.gate_s_off;
    const GATE_B_OFF: usize = L.gate_b_off;
    const UP_W_OFF: usize = L.up_w_off;
    const UP_S_OFF: usize = L.up_s_off;
    const UP_B_OFF: usize = L.up_b_off;
    const DOWN_W_OFF: usize = L.down_w_off;
    const DOWN_S_OFF: usize = L.down_s_off;
    const DOWN_B_OFF: usize = L.down_b_off;
    const GATE_W_SIZE: usize = L.gate_w_size;
    const GATE_S_SIZE: usize = L.gate_s_size;
    const GATE_B_SIZE: usize = L.gate_b_size;
    const UP_W_SIZE: usize = L.up_w_size;
    const UP_S_SIZE: usize = L.up_s_size;
    const UP_B_SIZE: usize = L.up_b_size;
    const DOWN_W_SIZE: usize = L.down_w_size;
    const DOWN_S_SIZE: usize = L.down_s_size;
    const DOWN_B_SIZE: usize = L.down_b_size;
    const EXPECTED_ARCHITECTURE: &'static str = "Qwen3_5MoeForConditionalGeneration_Stripped";
}
