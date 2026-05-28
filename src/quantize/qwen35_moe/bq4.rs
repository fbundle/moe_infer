// Qwen35MoE QuantScheme: name mapping, dtype classification, expert processing.
//
// Supports both HF-style raw BF16 repos and MLX-style pre-quantized repos.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use crate::dtype::{DType, GROUP_SIZE, quant_f32_to_int4};
use crate::hf_util::HfRepo;
use crate::quantize::{QuantScheme, WeightClass, WeightKind};
use crate::safetensors::{bytes_to_f32, parse_safetensors, read_tensor_bytes};

/// Embedded at compile time — no external file needed.
const NAME_MAPPING_JSON: &str = include_str!("name_mapping.json");

// ─── Qwen version ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub enum QwenVersion { V35, V36 }

impl QwenVersion {
    pub fn is_qwen36(self) -> bool { self == QwenVersion::V36 }
}

// ─── Model config ────────────────────────────────────────────────────────────

pub struct ModelParams {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub moe_intermediate: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_intermediate: usize,
    pub mtp_num_layers: usize,
    pub full_attn_interval: usize,
    pub vocab_size: usize,
    pub num_attn_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub linear_num_value_heads: usize,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub hf_config_raw: serde_json::Value,
}

pub fn load_config(model_path: &Path) -> Result<ModelParams, String> {
    let json_str = fs::read_to_string(model_path.join("config.json"))
        .map_err(|e| e.to_string())?;
    let root: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| e.to_string())?;
    let tc = root.get("text_config").unwrap_or(&root);

    let get = |key: &str, default: usize| -> usize {
        tc.get(key).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(default)
    };
    let get_f = |key: &str, default: f64| -> f64 {
        tc.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    };
    let rope = tc.get("rope_parameters");

    Ok(ModelParams {
        hidden_dim: get("hidden_size", 0),
        num_layers: get("num_hidden_layers", 0),
        moe_intermediate: get("moe_intermediate_size", 0),
        num_experts: get("num_experts", 0),
        num_experts_per_tok: get("num_experts_per_tok", 0),
        shared_intermediate: get("shared_expert_intermediate_size", 0),
        mtp_num_layers: get("mtp_num_hidden_layers", 0),
        full_attn_interval: get("full_attention_interval", 4),
        vocab_size: get("vocab_size", 0),
        num_attn_heads: get("num_attention_heads", 0),
        num_kv_heads: get("num_key_value_heads", 0),
        head_dim: get("head_dim", 0),
        rms_norm_eps: get_f("rms_norm_eps", 1e-6),
        linear_num_value_heads: get("linear_num_value_heads", 32),
        linear_num_key_heads: get("linear_num_key_heads", 16),
        linear_key_head_dim: get("linear_key_head_dim", 128),
        linear_value_head_dim: get("linear_value_head_dim", 128),
        linear_conv_kernel_dim: get("linear_conv_kernel_dim", 4),
        partial_rotary_factor: get_f("partial_rotary_factor",
            rope.and_then(|r| r.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64()).unwrap_or(0.25)),
        rope_theta: get_f("rope_theta",
            rope.and_then(|r| r.get("rope_theta"))
                .and_then(|v| v.as_f64()).unwrap_or(10_000_000.0)),
        hf_config_raw: root,
    })
}

// ─── Name mapping ────────────────────────────────────────────────────────────

type NameMap = HashMap<String, String>;

fn load_name_mapping(json_str: &str, num_layers: usize) -> Result<NameMap, String> {
    let mapping: HashMap<String, String> =
        serde_json::from_str(json_str).map_err(|e| e.to_string())?;
    let mut flat = HashMap::new();
    for (hf_pat, mlx_pat) in &mapping {
        if hf_pat.contains("{L}") {
            for l in 0..num_layers {
                flat.insert(
                    hf_pat.replace("{L}", &l.to_string()),
                    mlx_pat.replace("{L}", &l.to_string()),
                );
            }
        } else if hf_pat.contains("{B}") {
            for b in 0..27 {
                flat.insert(
                    hf_pat.replace("{B}", &b.to_string()),
                    mlx_pat.replace("{B}", &b.to_string()),
                );
            }
        } else {
            flat.insert(hf_pat.clone(), mlx_pat.clone());
        }
    }
    Ok(flat)
}

// ─── Name parsing ────────────────────────────────────────────────────────────

fn split_on_last_dot(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(idx) => (&name[..idx], &name[idx + 1..]),
        None => (name, ""),
    }
}

fn strip_layer_prefix(name: &str) -> &str {
    if let Some(after) = name.strip_prefix("language_model.model.layers.") {
        return after.find('.').map_or(after, |d| &after[d + 1..]);
    }
    if let Some(after) = name.strip_prefix("language_model.") { return after; }
    if let Some(after) = name.strip_prefix("vision_tower.blocks.") {
        return after.find('.').map_or(after, |d| &after[d + 1..]);
    }
    if let Some(after) = name.strip_prefix("vision_tower.") { return after; }
    if let Some(after) = name.strip_prefix("mtp.layers.") {
        return after.find('.').map_or(after, |d| &after[d + 1..]);
    }
    if let Some(after) = name.strip_prefix("mtp.") { return after; }
    name
}

fn matrix_table(block: &str) -> DType {
    match block {
        "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" | "self_attn.o_proj"
        | "mlp.gate" | "attn.qkv" | "attn.proj" | "patch_embed.proj" | "pos_embed" => DType::Bf16,
        "lm_head" => DType::Int8,
        _ => DType::Int4,
    }
}

fn bq4(mlx_name: &str, shape: &[usize]) -> DType {
    let (_, kind) = split_on_last_dot(mlx_name);
    match kind {
        "A_log" => { debug_assert!(shape.len() <= 1); DType::Fp32 }
        "scales" | "biases" | "bias" | "dt_bias" => { debug_assert!(shape.len() <= 2); DType::Bf16 }
        "weight" => {
            if shape.len() != 2 { DType::Bf16 }
            else { matrix_table(strip_layer_prefix(mlx_name)) }
        }
        _ => DType::Bf16,
    }
}

// ─── Layer extraction ────────────────────────────────────────────────────────

fn extract_layer(name: &str) -> Option<usize> {
    let haystack = name;
    let mut start = 0;
    while let Some(pos) = haystack[start..].find("layers.") {
        let after = start + pos + 7;
        let digits_end = haystack[after..]
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(haystack.len() - after);
        if digits_end > 0 {
            if let Ok(n) = haystack[after..after + digits_end].parse::<usize>() {
                if after + digits_end < haystack.len()
                    && haystack.as_bytes()[after + digits_end] == b'.'
                {
                    return Some(n);
                }
            }
        }
        start = after;
    }
    None
}

// ─── Expert detection ────────────────────────────────────────────────────────

fn is_expert_tensor(mlx_name: &str) -> bool {
    mlx_name.contains(".switch_mlp.gate_proj.")
        || mlx_name.contains(".switch_mlp.gate_up_proj.")
        || mlx_name.contains(".switch_mlp.up_proj.")
        || mlx_name.contains(".switch_mlp.down_proj.")
}

fn is_vision_tensor(mlx_name: &str) -> bool {
    mlx_name.starts_with("vision_tower.")
}

// ─── Sanitization constants ──────────────────────────────────────────────────

const MLX_NORM_KEYS: &[&str] = &[
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
];
const MTP_NORM_KEYS: &[&str] = &[
    ".hnorm.weight", ".enorm.weight", ".shared_head.norm.weight",
    ".norm1.weight", ".norm2.weight",
];

fn is_norm_key(mlx_name: &str) -> bool {
    MLX_NORM_KEYS.iter().chain(MTP_NORM_KEYS).any(|k| mlx_name.ends_with(k))
}

fn moveaxis_2_to_1(vals: &mut [f32], shape: &mut Vec<usize>) {
    let (c, k, s) = (shape[0], shape[1], shape[2]);
    let orig = vals.to_vec();
    for ci in 0..c {
        for ki in 0..k {
            for si in 0..s {
                let old_idx = ci * (k * s) + ki * s + si;
                let new_idx = ci * (s * k) + si * k + ki;
                vals[new_idx] = orig[old_idx];
            }
        }
    }
    *shape = vec![c, s, k];
}

// ─── QuantScheme ─────────────────────────────────────────────────────────────

pub struct Qwen35MoEScheme {
    version: QwenVersion,
    params: ModelParams,
    name_map: NameMap,
    eff_num_layers: usize,
    eff_num_experts: usize,
}

impl Qwen35MoEScheme {
    pub fn new(
        model_path: &Path,
        version: QwenVersion,
        strip_layers: usize,
        strip_experts: usize,
    ) -> Result<Self, String> {
        let params = load_config(model_path)?;
        let num_layers = params.num_layers;
        let num_experts = params.num_experts;

        let (eff_num_layers, eff_num_experts) = if strip_layers > 0 {
            let se = if strip_experts > 0 { strip_experts } else { num_experts };
            (strip_layers, se)
        } else {
            (num_layers, num_experts)
        };

        let name_map = load_name_mapping(NAME_MAPPING_JSON, num_layers)?;

        eprintln!("Model config:");
        eprintln!("  hidden_dim={}, vocab_size={}", params.hidden_dim, params.vocab_size);
        eprintln!("  num_layers={} (main={}, mtp={})",
            num_layers, num_layers - params.mtp_num_layers, params.mtp_num_layers);
        eprintln!("  num_experts={}, experts_per_tok={}", num_experts, params.num_experts_per_tok);
        eprintln!("  moe_intermediate={}, shared_intermediate={}",
            params.moe_intermediate, params.shared_intermediate);
        eprintln!("  Name mapping entries: {}", name_map.len());

        Ok(Self { version, params, name_map, eff_num_layers, eff_num_experts })
    }
}

impl QuantScheme for Qwen35MoEScheme {
    fn hidden_dim(&self) -> usize { self.params.hidden_dim }
    fn num_layers(&self) -> usize { self.eff_num_layers }
    fn num_experts(&self) -> usize { self.eff_num_experts }

    fn classify(&self, hf_name: &str, shape: &[usize]) -> WeightClass {
        // Map HF → MLX name (or identity for MLX repos)
        let mlx_name = self.name_map.get(hf_name)
            .cloned()
            .unwrap_or_else(|| hf_name.to_string());

        // Vision → Skip
        if is_vision_tensor(&mlx_name) {
            return WeightClass {
                name: mlx_name,
                quant: DType::Bf16,
                kind: WeightKind::Skip,
            };
        }

        // Expert → Expert(layer_idx)
        if is_expert_tensor(&mlx_name) {
            if let Some(layer) = extract_layer(hf_name) {
                let q = bq4(&mlx_name, shape);
                return WeightClass {
                    name: mlx_name,
                    quant: q,
                    kind: WeightKind::Expert(layer),
                };
            }
        }

        // Normal
        let q = bq4(&mlx_name, shape);
        WeightClass {
            name: mlx_name,
            quant: q,
            kind: WeightKind::Normal,
        }
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
        repo: &HfRepo,
        weight_map: &HashMap<String, String>,
        _classified: &[(String, WeightClass)],
        output_dir: &Path,
    ) -> Result<usize, String> {
        let inter = self.params.moe_intermediate;
        let hidden = self.params.hidden_dim;
        let _num_experts = self.params.num_experts;
        let eff_num_experts = self.eff_num_experts;
        let gs = GROUP_SIZE;

        let experts_dir = output_dir.join("packed_experts");

        // Group expert tensors by layer
        let mut expert_by_layer: BTreeMap<usize, (String, String)> = BTreeMap::new();
        for (hf_name, cls) in _classified {
            if let WeightKind::Expert(layer) = cls.kind {
                let entry = expert_by_layer.entry(layer)
                    .or_insert_with(|| (String::new(), String::new()));
                if cls.name.contains("gate_up_proj") || cls.name.contains("gate_proj") {
                    entry.0 = hf_name.clone();
                } else if cls.name.contains("down_proj") {
                    entry.1 = hf_name.clone();
                }
            }
        }

        let mut expert_layers_done = 0usize;

        for (layer_idx, (gate_up_key, down_key)) in &expert_by_layer {
            if gate_up_key.is_empty() || down_key.is_empty() {
                eprintln!("  Layer {} SKIPPED (missing keys)", layer_idx);
                continue;
            }

            let gu_shard = weight_map.get(gate_up_key).ok_or("shard not found")?;
            let down_shard = weight_map.get(down_key).ok_or("shard not found")?;

            let gu_path = repo.ensure(gu_shard)?;
            let down_path = if gu_shard == down_shard {
                gu_path.clone()
            } else {
                repo.ensure(down_shard)?
            };

            let gu_header = parse_safetensors(&gu_path)?;
            let down_header = parse_safetensors(&down_path)?;

            let gu_raw = read_tensor_bytes(&gu_path, &gu_header, gate_up_key)?;
            let gu_f32 = bytes_to_f32(&gu_raw, &gu_header.tensors[gate_up_key].dtype);
            let down_raw = read_tensor_bytes(&down_path, &down_header, down_key)?;
            let down_f32 = bytes_to_f32(&down_raw, &down_header.tensors[down_key].dtype);

            if repo.is_hf() {
                repo.remove(gu_shard);
                if gu_shard != down_shard { repo.remove(down_shard); }
            }

            // Quantize and pack: fused gate_up [E, 2*I, H] + down [E, H, I]
            let gate_w_bytes = inter * (hidden / 8) * 4;
            let gate_s_bytes = inter * (hidden / gs) * 2;
            let gate_b_bytes = gate_s_bytes;
            let up_w_bytes = gate_w_bytes;
            let up_s_bytes = gate_s_bytes;
            let up_b_bytes = gate_b_bytes;
            let down_w_bytes = hidden * (inter / 8) * 4;
            let down_s_bytes = hidden * (inter / gs) * 2;
            let down_b_bytes = down_s_bytes;
            let expert_size = gate_w_bytes + gate_s_bytes + gate_b_bytes
                + up_w_bytes + up_s_bytes + up_b_bytes
                + down_w_bytes + down_s_bytes + down_b_bytes;

            let mut buf = vec![0u8; eff_num_experts * expert_size];

            for e in 0..eff_num_experts {
                let gu_base = e * (2 * inter * hidden);
                let gate_f32: Vec<f32> = gu_f32[gu_base..gu_base + inter * hidden].to_vec();
                let up_f32: Vec<f32> =
                    gu_f32[gu_base + inter * hidden..gu_base + 2 * inter * hidden].to_vec();
                let down_base = e * (hidden * inter);
                let down_f32_e: Vec<f32> = down_f32[down_base..down_base + hidden * inter].to_vec();

                let (gate_p, gate_s, gate_b) = quant_f32_to_int4(&gate_f32, inter, hidden);
                let (up_p, up_s, up_b) = quant_f32_to_int4(&up_f32, inter, hidden);
                let (down_p, down_s, down_b) = quant_f32_to_int4(&down_f32_e, hidden, inter);

                let base = e * expert_size;
                copy_u32_bytes(&gate_p, &mut buf[base..base + gate_w_bytes]);
                let mut pos = base + gate_w_bytes;
                copy_u16_bytes(&gate_s, &mut buf[pos..pos + gate_s_bytes]);
                pos += gate_s_bytes;
                copy_u16_bytes(&gate_b, &mut buf[pos..pos + gate_b_bytes]);
                pos += gate_b_bytes;
                copy_u32_bytes(&up_p, &mut buf[pos..pos + up_w_bytes]);
                pos += up_w_bytes;
                copy_u16_bytes(&up_s, &mut buf[pos..pos + up_s_bytes]);
                pos += up_s_bytes;
                copy_u16_bytes(&up_b, &mut buf[pos..pos + up_b_bytes]);
                pos += up_b_bytes;
                copy_u32_bytes(&down_p, &mut buf[pos..pos + down_w_bytes]);
                pos += down_w_bytes;
                copy_u16_bytes(&down_s, &mut buf[pos..pos + down_s_bytes]);
                pos += down_s_bytes;
                copy_u16_bytes(&down_b, &mut buf[pos..pos + down_b_bytes]);
            }

            let out_path = experts_dir.join(format!("layer_{:02}.bin", layer_idx));
            fs::write(&out_path, &buf).map_err(|e| e.to_string())?;
            eprintln!("  Layer {:02}: {:.1} MB → {}",
                layer_idx,
                buf.len() as f64 / 1e6,
                out_path.file_name().unwrap().to_string_lossy());
            expert_layers_done += 1;
        }

        Ok(expert_layers_done)
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
        ins!("num_experts", self.eff_num_experts);
        ins!("num_experts_per_tok", p.num_experts_per_tok);
        ins!("moe_intermediate_size", p.moe_intermediate);
        ins!("shared_expert_intermediate_size", p.shared_intermediate);
        ins!("full_attention_interval", p.full_attn_interval);
        ins!("linear_num_value_heads", p.linear_num_value_heads);
        ins!("linear_num_key_heads", p.linear_num_key_heads);
        ins!("linear_key_head_dim", p.linear_key_head_dim);
        ins!("linear_value_head_dim", p.linear_value_head_dim);
        ins!("linear_conv_kernel_dim", p.linear_conv_kernel_dim);
        ins!("partial_rotary_factor", p.partial_rotary_factor);
        ins!("rope_theta", p.rope_theta);
        ins!("mtp_num_hidden_layers", if self.eff_num_layers < p.num_layers { 0 } else { p.mtp_num_layers });

        let num_main = p.num_layers - p.mtp_num_layers;
        let layer_types: Vec<String> = (0..self.eff_num_layers.min(num_main))
            .map(|i| {
                if (i + 1) % p.full_attn_interval == 0 {
                    "full_attention".to_string()
                } else {
                    "linear_attention".to_string()
                }
            })
            .collect();
        cfg.insert("layer_types".into(), serde_json::Value::Array(
            layer_types.iter().map(|s| serde_json::Value::String(s.clone())).collect()
        ));
    }
}

fn copy_u32_bytes(src: &[u32], dst: &mut [u8]) {
    let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
    dst.copy_from_slice(&bytes);
}

fn copy_u16_bytes(src: &[u16], dst: &mut [u8]) {
    let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
    dst.copy_from_slice(&bytes);
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_on_last_dot() {
        assert_eq!(split_on_last_dot("a.b.c.weight"), ("a.b.c", "weight"));
        assert_eq!(split_on_last_dot("nodot"), ("nodot", ""));
    }

    #[test]
    fn test_strip_layer_prefix() {
        assert_eq!(
            strip_layer_prefix("language_model.model.layers.3.self_attn.q_proj"),
            "self_attn.q_proj");
        assert_eq!(strip_layer_prefix("language_model.lm_head"), "lm_head");
    }

    #[test]
    fn test_matrix_table() {
        assert_eq!(matrix_table("self_attn.q_proj"), DType::Bf16);
        assert_eq!(matrix_table("lm_head"), DType::Int8);
        assert_eq!(matrix_table("mlp.switch_mlp.gate_up_proj"), DType::Int4);
    }

    #[test]
    fn test_bq4() {
        assert_eq!(bq4("language_model.model.layers.0.self_attn.q_proj.weight", &[8192, 2048]), DType::Bf16);
        assert_eq!(bq4("language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight", &[256, 2048]), DType::Int4);
        assert_eq!(bq4("language_model.model.layers.0.input_layernorm.weight", &[2048]), DType::Bf16);
        assert_eq!(bq4("language_model.model.layers.0.linear_attn.A_log", &[128]), DType::Fp32);
        assert_eq!(bq4("language_model.model.embed_tokens.scales", &[32, 32]), DType::Bf16);
    }
}
