// Int4Scheme for the dense Qwen3.5 family (e.g. Qwen3.5-4B).
//
// All 2D weights → INT4 group=64. Norms stay BF16, A_log stays FP32,
// conv1d.weight gets the moveaxis(2→1) sanitize. No expert tensors,
// no shared expert, no router. Tied word embeddings — no separate
// lm_head tensor exists in the source.

use std::collections::HashMap;
use std::path::Path;

use crate::dtype::DType;
use crate::hf_util::HfRepo;
use crate::quantize::{QuantScheme, WeightClass, WeightKind};

pub use crate::qwen35_moe_common::{QwenVersion, ModelParams, load_config};

use crate::qwen35_moe_common::{
    NameMap, load_name_mapping,
    split_on_last_dot,
    is_vision_tensor,
    is_norm_key, moveaxis_2_to_1,
};

const NAME_MAPPING_JSON: &str = include_str!("name_mapping.json");

// ─── INT4 dtype selection ────────────────────────────────────────────────────

fn int4_quant(mlx_name: &str, shape: &[usize]) -> DType {
    let (_, kind) = split_on_last_dot(mlx_name);
    match kind {
        "A_log" => { debug_assert!(shape.len() <= 1); DType::Fp32 }
        "scales" | "biases" | "bias" | "dt_bias" => {
            debug_assert!(shape.len() <= 2); DType::Bf16
        }
        "weight" => {
            if shape.len() != 2 { DType::Bf16 } else { DType::Int4 }
        }
        _ => DType::Bf16,
    }
}

// ─── DenseInt4Scheme ─────────────────────────────────────────────────────────

pub struct DenseInt4Scheme {
    version: QwenVersion,
    params: ModelParams,
    name_map: NameMap,
    eff_num_layers: usize,
}

impl DenseInt4Scheme {
    pub fn new(model_path: &Path, version: QwenVersion) -> Result<Self, String> {
        let params = load_config(model_path)?;
        let num_layers = params.num_layers;
        let eff_num_layers = num_layers + params.mtp_num_layers;
        let name_map = load_name_mapping(NAME_MAPPING_JSON, num_layers)?;

        eprintln!("Model config (dense INT4):");
        eprintln!("  hidden_dim={}, vocab_size={}", params.hidden_dim, params.vocab_size);
        eprintln!("  num_layers={} (main={}, mtp={})",
            eff_num_layers, num_layers, params.mtp_num_layers);
        eprintln!("  num_heads={}, num_kv_heads={}, head_dim={}",
            params.num_attn_heads, params.num_kv_heads, params.head_dim);
        eprintln!("  Name mapping entries: {}", name_map.len());

        Ok(Self { version, params, name_map, eff_num_layers })
    }
}

impl QuantScheme for DenseInt4Scheme {
    fn hidden_dim(&self) -> usize { self.params.hidden_dim }
    fn num_layers(&self) -> usize { self.eff_num_layers }
    fn num_experts(&self) -> usize { 0 }

    fn classify(&self, hf_name: &str, shape: &[usize]) -> WeightClass {
        let mlx_name = self.name_map.get(hf_name)
            .cloned()
            .unwrap_or_else(|| hf_name.to_string());

        if is_vision_tensor(&mlx_name) {
            return WeightClass { name: mlx_name, quant: DType::Bf16, kind: WeightKind::Skip };
        }
        if mlx_name.starts_with("model.visual.") || hf_name.starts_with("model.visual.") {
            return WeightClass { name: mlx_name, quant: DType::Bf16, kind: WeightKind::Skip };
        }

        let q = int4_quant(&mlx_name, shape);
        WeightClass { name: mlx_name, quant: q, kind: WeightKind::Normal }
    }

    fn sanitize(&self, mlx_name: &str, values: &mut [f32], shape: &mut Vec<usize>) {
        if self.version.is_qwen36() && is_norm_key(mlx_name) {
            for v in &mut *values { *v += 1.0; }
        }
        if mlx_name.contains("conv1d.weight") && shape.len() == 3 {
            moveaxis_2_to_1(values, shape);
        }
    }

    fn process_experts(
        &self,
        _repo: &HfRepo,
        _weight_map: &HashMap<String, String>,
        _classified: &[(String, WeightClass)],
        _output_dir: &Path,
    ) -> Result<usize, String> {
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
        ins!("hidden_size", p.hidden_dim);
        ins!("num_hidden_layers", self.eff_num_layers);
        ins!("num_attention_heads", p.num_attn_heads);
        ins!("num_key_value_heads", p.num_kv_heads);
        ins!("head_dim", p.head_dim);
        ins!("vocab_size", p.vocab_size);
        ins!("rms_norm_eps", p.rms_norm_eps);
        ins!("intermediate_size",
            p.hf_config_raw
                .get("text_config")
                .and_then(|t| t.get("intermediate_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0));
        ins!("full_attention_interval", p.full_attn_interval);
        ins!("linear_num_value_heads", p.linear_num_value_heads);
        ins!("linear_num_key_heads", p.linear_num_key_heads);
        ins!("linear_key_head_dim", p.linear_key_head_dim);
        ins!("linear_value_head_dim", p.linear_value_head_dim);
        ins!("linear_conv_kernel_dim", p.linear_conv_kernel_dim);
        ins!("partial_rotary_factor", p.partial_rotary_factor);
        ins!("rope_theta", p.rope_theta);
        ins!("mtp_num_hidden_layers", p.mtp_num_layers);
        ins!("tie_word_embeddings", true);

        let num_main = p.num_layers;
        let num_total = self.eff_num_layers;
        let mut layer_types: Vec<String> = (0..num_main)
            .map(|i| {
                if (i + 1) % p.full_attn_interval == 0 {
                    "full_attention".to_string()
                } else {
                    "linear_attention".to_string()
                }
            })
            .collect();
        for _ in num_main..num_total {
            layer_types.push("full_attention".to_string());
        }
        cfg.insert("layer_types".into(), serde_json::Value::Array(
            layer_types.iter().map(|s| serde_json::Value::String(s.clone())).collect()
        ));
    }
}
