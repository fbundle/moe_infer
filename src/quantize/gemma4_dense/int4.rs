//! Gemma 4 12B dense INT4 quantize pipeline.
//!
//! Mirrors `gemma4_moe/bq4.rs` (which we reuse `Gemma4Params` and `load_config`
//! from) but for the dense 12B variant:
//!   * no MoE (no experts, no router)
//!   * multimodal projections KEPT — for Gemma 4, audio is literally one matmul
//!     (`embed_audio.embedding_projection.weight [3840,640]`) and vision is
//!     just `patch_dense + pos_embedding + 2 LayerNorms`. No separate encoder.
//!   * `attention_k_eq_v=true` for the 8 full-attn layers (every 6th: layer
//!     5/11/17/23/29/35/41/47): those layers have no `v_proj.weight`. The
//!     name_map silently expands {L} for all 48 layers; missing-tensor lookups
//!     are no-ops.
//!
//! Quantization recipe (same shape as the rest of our INT4 schemes):
//!   * 2D `*.weight` → INT4 group=64
//!   * All norms / 1D biases / scalars / pos_embedding → BF16
//!   * `layer_scalar` (single fp scalar per layer) → BF16
//!
//! No norm-shift sanitize: Gemma 4 stores absolute norm weights in safetensors
//! (verified vs. mlx-vlm; see gemma4_moe/bq4.rs:295 comment for derivation).

use std::collections::HashMap;
use std::path::Path;

use crate::dtype::DType;
use crate::hf_util::HfRepo;
use crate::quantize::{QuantScheme, WeightClass, WeightKind};
use crate::qwen35_moe_common::{NameMap, load_name_mapping};
use crate::gemma4_bq4::{Gemma4Params, load_config};

const NAME_MAPPING_JSON: &str = include_str!("name_mapping.json");

/// Decide DType by engine-side tensor name.
///
/// All matrix weights → INT4; 1D / norm / scalar tensors → BF16; the
/// 3D vision pos_embedding stays BF16. The caller still has to validate that
/// the tensor is actually 2D before quantizing to INT4 (`classify` does this).
fn classify_dtype(engine_name: &str) -> DType {
    let bf16_suffixes: &[&str] = &[
        // Decoder norms (all the post-/pre-/input layernorms in Gemma 4).
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".pre_feedforward_layernorm.weight",
        ".post_feedforward_layernorm.weight",
        // Per-head Q/K norms.
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
        // Single scalar per layer.
        ".layer_scalar",
        // Final model norm.
        "language_model.model.norm.weight",
        // Vision encoder LayerNorms / norms (weight + bias).
        ".pos_norm.weight",
        ".pos_norm.bias",
        ".patch_ln1.weight",
        ".patch_ln1.bias",
        ".patch_ln2.weight",
        ".patch_ln2.bias",
        // Vision patch_dense BIAS (1D, ~3840 bytes — not worth INT4).
        ".patch_dense.bias",
        // 3D positional embedding (vision).
        ".pos_embedding",
    ];
    if bf16_suffixes.iter().any(|s| engine_name.ends_with(s)) {
        return DType::Bf16;
    }
    DType::Int4
}

// ─── Scheme ─────────────────────────────────────────────────────────────────

pub struct Gemma4DenseInt4Scheme {
    params: Gemma4Params,
    name_map: NameMap,
}

impl Gemma4DenseInt4Scheme {
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let params = load_config(model_path)?;
        let name_map = load_name_mapping(NAME_MAPPING_JSON, params.num_layers)?;
        eprintln!(
            "[gemma4-dense-int4] hidden={} layers={} heads={}/{} (sliding_kv={}) head_dim={} vocab={} sliding={}",
            params.hidden_dim, params.num_layers, params.num_attn_heads,
            params.num_kv_heads_full, params.num_kv_heads, params.head_dim,
            params.vocab_size, params.sliding_window,
        );
        eprintln!("[gemma4-dense-int4] name_map entries: {}", name_map.len());
        Ok(Self { params, name_map })
    }
}

impl QuantScheme for Gemma4DenseInt4Scheme {
    fn hidden_dim(&self) -> usize { self.params.hidden_dim }
    fn num_layers(&self) -> usize { self.params.num_layers }
    fn num_experts(&self) -> usize { 0 } // dense — no experts

    fn classify(&self, hf_name: &str, shape: &[usize]) -> WeightClass {
        // No skipping. Multimodal projections (audio + vision) are kept as
        // first-class weights — the engine will call into them when the
        // pipeline passes audio/image inputs.
        let engine_name = self.name_map.get(hf_name).cloned().unwrap_or_else(|| {
            eprintln!("[gemma4-dense-int4] WARN: unmapped tensor {} — passing through", hf_name);
            hf_name.to_string()
        });
        let mut quant = classify_dtype(&engine_name);
        // INT4 only makes sense for 2D matrices. Fall back to BF16 otherwise
        // (the dtype rule above doesn't see the shape, so this is the safety
        // net for any tensor that's 1D or 3D+ but doesn't match a BF16 suffix).
        if quant == DType::Int4 && shape.len() != 2 {
            quant = DType::Bf16;
        }
        WeightClass { name: engine_name, quant, kind: WeightKind::Normal }
    }

    fn sanitize(&self, _engine_name: &str, _values: &mut [f32], _shape: &mut Vec<usize>) {
        // Intentional no-op. Gemma 4 stores absolute norm weights in
        // safetensors — see the comment in gemma4_moe/bq4.rs sanitize()
        // for the diff-vs-mlx-vlm derivation. Don't add +1.
    }

    fn process_experts(
        &self,
        _repo: &HfRepo,
        _weight_map: &HashMap<String, String>,
        _classified: &[(String, WeightClass)],
        _output_dir: &Path,
    ) -> Result<usize, String> {
        Ok(0) // dense — no experts to pack
    }

    fn write_manifest_config(
        &self,
        cfg: &mut serde_json::Map<String, serde_json::Value>,
    ) {
        let p = &self.params;
        macro_rules! ins {
            ($k:expr, $v:expr) => { cfg.insert($k.into(), serde_json::Value::from($v)); };
        }
        ins!("architectures", serde_json::Value::Array(vec![
            serde_json::Value::String("Gemma4DenseForConditionalGeneration".into())
        ]));
        ins!("hidden_size", p.hidden_dim);
        ins!("num_hidden_layers", p.num_layers);
        ins!("num_attention_heads", p.num_attn_heads);
        ins!("num_key_value_heads", p.num_kv_heads);
        ins!("num_global_key_value_heads", p.num_kv_heads_full);
        ins!("head_dim", p.head_dim);
        ins!("intermediate_size", p.intermediate_size);
        ins!("vocab_size", p.vocab_size);
        ins!("sliding_window", p.sliding_window);
        ins!("final_logit_softcapping", p.final_logit_softcap);
        ins!("attention_k_eq_v", true);
        ins!("tie_word_embeddings", true);
        cfg.insert("layer_types".into(), serde_json::Value::Array(
            p.layer_types.iter().map(|s| serde_json::Value::String(s.clone())).collect()
        ));
        // Replicate Gemma 4's nested rope_parameters structure.
        let mut rope = serde_json::Map::new();
        let mut sliding = serde_json::Map::new();
        sliding.insert("rope_theta".into(),
                       serde_json::Value::from(p.rope_theta_sliding));
        sliding.insert("rope_type".into(),
                       serde_json::Value::String("default".into()));
        let mut full = serde_json::Map::new();
        full.insert("rope_theta".into(),
                    serde_json::Value::from(p.rope_theta_full));
        full.insert("partial_rotary_factor".into(),
                    serde_json::Value::from(p.partial_rotary_full));
        full.insert("rope_type".into(),
                    serde_json::Value::String("proportional".into()));
        rope.insert("sliding_attention".into(), serde_json::Value::Object(sliding));
        rope.insert("full_attention".into(), serde_json::Value::Object(full));
        cfg.insert("rope_parameters".into(), serde_json::Value::Object(rope));
    }
}
