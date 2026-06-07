// BQ4 scheme for the dense Qwen3.5 family (e.g. Qwen3.5-4B).
//
// Mirrors the qwen35_moe BQ4 idea — keep accuracy-critical projections in
// BF16, INT4-quantize only the bulk weights. For dense the critical path is:
//   - self_attn.{q,k,v,o}_proj  (8 full-attn layers)
//   - linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}  (24 layers)
// These run on every token and feed the residual stream; INT4 noise in them
// gets amplified by residual cancellation in deep layers.
//
// INT4-quantized:
//   - embed_tokens (also acts as the tied lm_head)
//   - mlp.{gate_proj,up_proj,down_proj}  (the bulk — ~75% of weight bytes)
//
// 1D / 3D weights (norms, conv1d, A_log, dt_bias) stay BF16/FP32 as in INT4 scheme.

use std::collections::HashMap;
use std::path::Path;

use crate::dtype::DType;
use crate::hf_util::HfRepo;
use crate::quantize::{QuantScheme, WeightClass, WeightKind};

pub use crate::qwen35_moe_common::{QwenVersion, ModelParams, load_config};

use crate::qwen35_moe_common::{
    NameMap, load_name_mapping,
    split_on_last_dot, strip_layer_prefix,
    is_vision_tensor,
    is_norm_key, moveaxis_2_to_1,
};

const NAME_MAPPING_JSON: &str = include_str!("name_mapping.json");

// ─── BQ4 dtype routing ───────────────────────────────────────────────────────

/// Per-block dtype for 2D weights. Block = tensor path with the layer-prefix
/// stripped, e.g. `"self_attn.q_proj"` or `"mlp.gate_proj"`.
fn matrix_table(block: &str) -> DType {
    match block {
        // Attention projections — full-attn AND linear-attn — stay BF16.
        "self_attn.q_proj"
        | "self_attn.k_proj"
        | "self_attn.v_proj"
        | "self_attn.o_proj"
        | "linear_attn.in_proj_qkv"
        | "linear_attn.in_proj_z"
        | "linear_attn.in_proj_a"
        | "linear_attn.in_proj_b"
        | "linear_attn.out_proj"
        // MTP fusion projection — small but accuracy-sensitive (gates residual + token).
        | "eh_proj"
            => DType::Bf16,
        // Everything else (mlp.gate_proj/up_proj/down_proj, embed_tokens, etc.) → INT4.
        _ => DType::Int4,
    }
}

fn bq4(mlx_name: &str, shape: &[usize]) -> DType {
    let (prefix, kind) = split_on_last_dot(mlx_name);
    match kind {
        "A_log" => { debug_assert!(shape.len() <= 1); DType::Fp32 }
        "scales" | "biases" | "bias" | "dt_bias" => {
            debug_assert!(shape.len() <= 2); DType::Bf16
        }
        "weight" => {
            if shape.len() != 2 { DType::Bf16 } else { matrix_table(strip_layer_prefix(prefix)) }
        }
        _ => DType::Bf16,
    }
}

// ─── DenseBQ4Scheme ──────────────────────────────────────────────────────────

pub struct DenseBQ4Scheme {
    version: QwenVersion,
    params: ModelParams,
    name_map: NameMap,
    eff_num_layers: usize,
}

impl DenseBQ4Scheme {
    pub fn new(model_path: &Path, version: QwenVersion) -> Result<Self, String> {
        let params = load_config(model_path)?;
        let num_layers = params.num_layers;
        let eff_num_layers = num_layers + params.mtp_num_layers;
        let name_map = load_name_mapping(NAME_MAPPING_JSON, num_layers)?;

        eprintln!("Model config (dense BQ4):");
        eprintln!("  hidden_dim={}, vocab_size={}", params.hidden_dim, params.vocab_size);
        eprintln!("  num_layers={} (main={}, mtp={})",
            eff_num_layers, num_layers, params.mtp_num_layers);
        eprintln!("  num_heads={}, num_kv_heads={}, head_dim={}",
            params.num_attn_heads, params.num_kv_heads, params.head_dim);
        eprintln!("  Name mapping entries: {}", name_map.len());
        eprintln!("  Quant: self_attn + linear_attn projections → BF16, MLP + embed → INT4");

        Ok(Self { version, params, name_map, eff_num_layers })
    }
}

impl QuantScheme for DenseBQ4Scheme {
    fn hidden_dim(&self) -> usize { self.params.hidden_dim }
    fn num_layers(&self) -> usize { self.eff_num_layers }
    fn num_experts(&self) -> usize { 0 }

    fn classify(&self, hf_name: &str, shape: &[usize]) -> WeightClass {
        let mlx_name = self.name_map.get(hf_name)
            .cloned()
            .unwrap_or_else(|| hf_name.to_string());

        if is_vision_tensor(&mlx_name)
            || mlx_name.starts_with("model.visual.")
            || hf_name.starts_with("model.visual.")
        {
            return WeightClass { name: mlx_name, quant: DType::Bf16, kind: WeightKind::Skip };
        }

        let q = bq4(&mlx_name, shape);
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
                if (i + 1) % p.full_attn_interval == 0 { "full_attention".into() }
                else { "linear_attention".into() }
            })
            .collect();
        for _ in num_main..num_total {
            layer_types.push("full_attention".into());
        }
        cfg.insert("layer_types".into(), serde_json::Value::Array(
            layer_types.iter().map(|s| serde_json::Value::String(s.clone())).collect()
        ));
    }
}
