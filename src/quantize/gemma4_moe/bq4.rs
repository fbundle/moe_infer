//! Gemma 4 26B-A4B BQ4 quantize pipeline.
//!
//! Quantizes matmul weights to INT4 (group=64), keeps norms/scalars at
//! BF16. Net: ~12 GB on-disk vs ~48 GB BF16 — fits in 16 GB Mac RAM.
//!
//! Why no INT4/INT8 yet:
//!   - forward_sliding_layer/forward_full_layer dispatch matvec_bf16
//!     for every projection (q/k/v/o, mlp gate/up/down, router proj,
//!     per-expert gate/up/down). The dispatchers don't yet handle
//!     mixed-dtype dispatch.
//!   - Until the engine can validate against MLX-VLM with BF16 weights,
//!     adding INT4/INT8 quant on top introduces extra noise. Get
//!     correctness first, optimize later.
//!
//! Expert tensors are written WHOLE (not split per-expert) because
//! `forward_dual_ffn` reads per-expert slices via byte-stride math
//! directly from the merged HF tensors:
//!     experts.gate_up_proj : [128, 2*moe_inter=1408, hidden=2816] BF16
//!     experts.down_proj    : [128, hidden=2816, moe_inter=704]    BF16
//! This is a big simplification vs Qwen3.6 where experts get split into
//! per-layer `packed_experts/layer_XX.bin` blobs.
//!
//! Future BQ4 path can switch to actual quantization by changing `bq4()`
//! to return the appropriate DType per tensor (matching MLX-LM's
//! quant_predicate: router/mlp INT8 group=64, experts/attention INT4).
//! Doing that requires teaching the engine's matvec dispatchers to pick
//! the right kernel per tensor's dtype, which is its own work item.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::dtype::DType;
use crate::hf_util::HfRepo;
use crate::quantize::{QuantScheme, WeightClass, WeightKind};
use crate::error::MoEError;

/// Embedded name map (HF→engine).
const NAME_MAPPING_JSON: &str = include_str!("name_mapping.json");

/// Minimal Gemma 4 config — fields we actually care about for the manifest.
pub struct Gemma4Params {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub moe_intermediate: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_attn_heads: usize,
    pub num_kv_heads: usize,
    pub num_kv_heads_full: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub sliding_window: usize,
    pub rope_theta_sliding: f64,
    pub rope_theta_full: f64,
    pub partial_rotary_full: f64,
    pub final_logit_softcap: f64,
    pub layer_types: Vec<String>,
    pub hf_config_raw: serde_json::Value,
}

pub fn load_config(model_path: &Path) -> Result<Gemma4Params, String> {
    let json_str = fs::read_to_string(model_path.join("config.json"))
        .map_err(|e| e.to_string())?;
    let root: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| e.to_string())?;
    let tc = root.get("text_config").unwrap_or(&root);

    let get_u = |k: &str, d: usize| -> usize {
        tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(d)
    };
    let get_f = |k: &str, d: f64| -> f64 {
        tc.get(k).and_then(|v| v.as_f64()).unwrap_or(d)
    };

    // RoPE config is nested under rope_parameters.{sliding,full}_attention.
    let rope_root = tc.get("rope_parameters");
    let rope_sliding_theta = rope_root
        .and_then(|r| r.get("sliding_attention"))
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000.0);
    let rope_full_theta = rope_root
        .and_then(|r| r.get("full_attention"))
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1_000_000.0);
    let partial_rotary_full = rope_root
        .and_then(|r| r.get("full_attention"))
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.25);

    // layer_types: list of "sliding_attention" / "full_attention" per layer.
    let layer_types: Vec<String> = tc.get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Ok(Gemma4Params {
        hidden_dim:       get_u("hidden_size", 0),
        num_layers:       get_u("num_hidden_layers", 0),
        num_experts:      get_u("num_experts", 0),
        top_k_experts:    get_u("top_k_experts", get_u("num_experts_per_tok", 0)),
        moe_intermediate: get_u("moe_intermediate_size", 0),
        intermediate_size:get_u("intermediate_size", 0),
        vocab_size:       get_u("vocab_size", 0),
        num_attn_heads:   get_u("num_attention_heads", 0),
        num_kv_heads:     get_u("num_key_value_heads", 0),
        num_kv_heads_full:get_u("num_global_key_value_heads", get_u("num_key_value_heads", 0)),
        head_dim:         get_u("head_dim", 0),
        global_head_dim:  get_u("global_head_dim", get_u("head_dim", 0)),
        sliding_window:   get_u("sliding_window", 0),
        rope_theta_sliding: rope_sliding_theta,
        rope_theta_full:    rope_full_theta,
        partial_rotary_full,
        final_logit_softcap: get_f("final_logit_softcapping", 0.0),
        layer_types,
        hf_config_raw: root,
    })
}

// ─── Name map ─────────────────────────────────────────────────────────────────
pub(crate) type NameMap = HashMap<String, String>;

pub(crate) fn load_name_mapping(json_str: &str, num_layers: usize) -> Result<NameMap, String> {
    let mapping: HashMap<String, String> =
        serde_json::from_str(json_str).map_err(|e| e.to_string())?;
    let mut flat = HashMap::new();
    for (hf_pat, engine_pat) in &mapping {
        if hf_pat.starts_with('_') { continue; }   // skip _comment_* keys
        if hf_pat.contains("{L}") {
            for l in 0..num_layers {
                flat.insert(
                    hf_pat.replace("{L}", &l.to_string()),
                    engine_pat.replace("{L}", &l.to_string()),
                );
            }
        } else {
            flat.insert(hf_pat.clone(), engine_pat.clone());
        }
    }
    Ok(flat)
}

// ─── Vision-tower detection ─────────────────────────────────────────────────
fn is_vision_tensor(hf_name: &str) -> bool {
    hf_name.starts_with("model.vision_tower.")
        || hf_name.starts_with("vision_tower.")
        || hf_name.starts_with("model.embed_vision.")
        || hf_name.starts_with("embed_vision.")
        || hf_name.starts_with("model.multimodal_projector.")
        || hf_name.starts_with("multimodal_projector.")
}

// ─── DType selection ─────────────────────────────────────────────────────────
//
// INT4 for all matmul weights (per MLX-LM's quant_predicate). BF16 for norms,
// per-channel/per-expert scalars, and the routing weights. Embed_tokens IS
// INT4 (the engine dequantizes one row at a time at lookup time).

fn classify_dtype(engine_name: &str, layer_types: &[String]) -> DType {
    // Tensors that MUST stay BF16 (norms, scalars, 1D weights).
    let bf16_suffixes: &[&str] = &[
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "pre_feedforward_layernorm_2.weight",
        "post_feedforward_layernorm.weight",
        "post_feedforward_layernorm_1.weight",
        "post_feedforward_layernorm_2.weight",
        "q_norm.weight",
        "k_norm.weight",
        "router.scale",
        "router.per_expert_scale",
        "layer_scalar",
        "model.norm.weight",
    ];
    if bf16_suffixes.iter().any(|s| engine_name.ends_with(s)) {
        return DType::Bf16;
    }
    // Full-layer o_proj has in_dim = 16*512 = 8192 which exceeds the INT4
    // dequant kernel's threadgroup x_shared[4096] budget. Keep those at
    // BF16. Sliding-layer o_proj has in_dim=4096 — fits.
    if engine_name.ends_with(".self_attn.o_proj.weight") {
        if let Some(li) = extract_layer_idx(engine_name) {
            if layer_types.get(li).map(|s| s == "full_attention").unwrap_or(false) {
                return DType::Bf16;
            }
        }
    }
    // Everything else is INT4: q/k/v_proj, sliding-layer o_proj,
    // mlp.{gate,up,down}, router.proj, experts.{gate_up,down}_proj,
    // embed_tokens.
    DType::Int4
}

/// Identifies the ZeroCenteredRMSNorm tensors in Gemma 4. These apply
/// `x * (1 + w)`; our kernel does `x * w`, so we shift the stored weight
/// by +1 at quantize time.
fn is_zero_centered_norm(engine_name: &str) -> bool {
    // Every norm in Gemma 4's decoder is zero-centered, INCLUDING q_norm
    // and k_norm (per the MLX-VLM Gemma 4 source: every RMSNorm in the
    // text model is Gemma4RMSNorm = zero-centered).
    let zc_suffixes: &[&str] = &[
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".pre_feedforward_layernorm.weight",
        ".pre_feedforward_layernorm_2.weight",
        ".post_feedforward_layernorm.weight",
        ".post_feedforward_layernorm_1.weight",
        ".post_feedforward_layernorm_2.weight",
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
    ];
    if zc_suffixes.iter().any(|s| engine_name.ends_with(s)) {
        return true;
    }
    // Final model norm.
    engine_name == "language_model.model.norm.weight"
}

fn extract_layer_idx(name: &str) -> Option<usize> {
    let i = name.find("layers.")? + 7;
    let rest = &name[i..];
    let end = rest.find('.').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ─── Scheme ─────────────────────────────────────────────────────────────────
pub struct Gemma4Bf16Scheme {
    params: Gemma4Params,
    name_map: NameMap,
}

impl Gemma4Bf16Scheme {
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let params = load_config(model_path)?;
        let name_map = load_name_mapping(NAME_MAPPING_JSON, params.num_layers)?;
        eprintln!("[gemma4-bf16] hidden={}, layers={}, experts={} top_k={} vocab={}",
            params.hidden_dim, params.num_layers, params.num_experts,
            params.top_k_experts, params.vocab_size);
        eprintln!("[gemma4-bf16] name_map entries: {}", name_map.len());
        Ok(Self { params, name_map })
    }
}

impl QuantScheme for Gemma4Bf16Scheme {
    fn hidden_dim(&self) -> usize { self.params.hidden_dim }
    fn num_layers(&self) -> usize { self.params.num_layers }
    fn num_experts(&self) -> usize { self.params.num_experts }

    fn classify(&self, hf_name: &str, _shape: &[usize]) -> WeightClass {
        if is_vision_tensor(hf_name) {
            return WeightClass {
                name: hf_name.to_string(),
                quant: DType::Bf16,
                kind: WeightKind::Skip,
            };
        }
        let engine_name = self.name_map.get(hf_name)
            .cloned()
            .unwrap_or_else(|| {
                eprintln!("[gemma4-bq4] WARN: unmapped tensor {} — passing through unchanged", hf_name);
                hf_name.to_string()
            });
        let quant = classify_dtype(&engine_name, &self.params.layer_types);
        WeightClass {
            name: engine_name,
            quant,
            // Even expert tensors get WeightKind::Normal — we DO NOT split them
            // per-expert. forward_dual_ffn reads per-expert slices via byte
            // arithmetic directly off the merged tensor.
            kind: WeightKind::Normal,
        }
    }

    fn sanitize(&self, engine_name: &str, values: &mut [f32], shape: &mut Vec<usize>) {
        // Reshape 3D experts tensors to 2D so INT4 group-quant runs per-row.
        if engine_name.ends_with(".experts.gate_up_proj")
            || engine_name.ends_with(".experts.down_proj")
        {
            if shape.len() == 3 {
                let (e, r, c) = (shape[0], shape[1], shape[2]);
                *shape = vec![e * r, c];
            }
        }
        // Gemma 4 uses ZeroCenteredRMSNorm: applies `x * (1 + w)` rather than
        // `x * w`. Our kernel does the latter, so we bake `+1` into the
        // stored weight at quantize time (same trick as qwen35's
        // is_qwen36 shift).
        if is_zero_centered_norm(engine_name) {
            for v in values.iter_mut() { *v += 1.0; }
        }
    }

    fn process_experts(
        &self,
        _repo: &HfRepo,
        _weight_map: &HashMap<String, String>,
        _classified: &[(String, WeightClass)],
        _output_dir: &Path,
    ) -> Result<usize, String> {
        // No-op: experts are written inline by the shard pass because their
        // WeightClass::kind is Normal. The engine reads them via byte
        // strides — no per-layer split needed.
        Ok(0)
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
            serde_json::Value::String("Gemma4ForConditionalGeneration".into())
        ]));
        ins!("hidden_size", p.hidden_dim);
        ins!("num_hidden_layers", p.num_layers);
        ins!("num_attention_heads", p.num_attn_heads);
        ins!("num_key_value_heads", p.num_kv_heads);
        ins!("num_global_key_value_heads", p.num_kv_heads_full);
        ins!("head_dim", p.head_dim);
        ins!("global_head_dim", p.global_head_dim);
        ins!("vocab_size", p.vocab_size);
        ins!("num_experts", p.num_experts);
        ins!("top_k_experts", p.top_k_experts);
        ins!("moe_intermediate_size", p.moe_intermediate);
        ins!("intermediate_size", p.intermediate_size);
        ins!("sliding_window", p.sliding_window);
        ins!("final_logit_softcapping", p.final_logit_softcap);
        ins!("attention_k_eq_v", true);
        ins!("tie_word_embeddings", true);
        cfg.insert("layer_types".into(), serde_json::Value::Array(
            p.layer_types.iter().map(|s| serde_json::Value::String(s.clone())).collect()
        ));
        // Replicate rope_parameters in a flat-ish form for the engine config check.
        let mut rope = serde_json::Map::new();
        let mut sliding = serde_json::Map::new();
        sliding.insert("rope_theta".into(), serde_json::Value::from(p.rope_theta_sliding));
        sliding.insert("rope_type".into(), serde_json::Value::String("default".into()));
        let mut full = serde_json::Map::new();
        full.insert("rope_theta".into(), serde_json::Value::from(p.rope_theta_full));
        full.insert("partial_rotary_factor".into(), serde_json::Value::from(p.partial_rotary_full));
        full.insert("rope_type".into(), serde_json::Value::String("proportional".into()));
        rope.insert("sliding_attention".into(), serde_json::Value::Object(sliding));
        rope.insert("full_attention".into(), serde_json::Value::Object(full));
        cfg.insert("rope_parameters".into(), serde_json::Value::Object(rope));
    }
}

/// Entry point. Just shells through to `crate::quantize::run`.
pub fn quantize(hf_dir: &str, out_dir: &str) -> Result<(), MoEError> {
    let config_dir = if std::path::Path::new(hf_dir).is_dir() {
        std::path::PathBuf::from(hf_dir)
    } else {
        let repo = crate::hf_util::HfRepo::from_hf(hf_dir)
            .map_err(MoEError::Config)?;
        repo.ensure("config.json").map_err(MoEError::Config)?;
        repo.path().to_path_buf()
    };
    let scheme = Gemma4Bf16Scheme::new(&config_dir).map_err(MoEError::Config)?;
    crate::quantize::run(hf_dir, out_dir, &scheme).map_err(MoEError::Config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_map_loads() {
        let m = load_name_mapping(NAME_MAPPING_JSON, 30).unwrap();
        // Spot checks
        assert_eq!(m.get("model.language_model.embed_tokens.weight").unwrap(),
                   "language_model.model.embed_tokens.weight");
        assert_eq!(m.get("model.language_model.layers.0.self_attn.q_proj.weight").unwrap(),
                   "language_model.model.layers.0.self_attn.q_proj.weight");
        // Full layer 5 (also has full-attn pattern)
        assert_eq!(m.get("model.language_model.layers.5.experts.gate_up_proj").unwrap(),
                   "language_model.model.layers.5.experts.gate_up_proj");
    }

    #[test]
    fn test_classify_vision_skipped() {
        let s = Gemma4Params {
            hidden_dim: 2816, num_layers: 30, num_experts: 128, top_k_experts: 8,
            moe_intermediate: 704, intermediate_size: 2112, vocab_size: 262144,
            num_attn_heads: 16, num_kv_heads: 8, num_kv_heads_full: 2,
            head_dim: 256, global_head_dim: 512, sliding_window: 1024,
            rope_theta_sliding: 10000.0, rope_theta_full: 1_000_000.0,
            partial_rotary_full: 0.25, final_logit_softcap: 30.0,
            layer_types: vec![], hf_config_raw: serde_json::Value::Null,
        };
        let nm = load_name_mapping(NAME_MAPPING_JSON, 30).unwrap();
        let scheme = Gemma4Bf16Scheme { params: s, name_map: nm };
        let cls = scheme.classify("model.vision_tower.encoder.layers.0.input_layernorm.weight", &[2816]);
        assert!(matches!(cls.kind, WeightKind::Skip));
    }
}
