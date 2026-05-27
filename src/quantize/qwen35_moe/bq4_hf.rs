// Full A→Z quantization pipeline: read HF safetensors → classify → encode → write BQ4.
//
// Called from Python via a single `moe_infer.quantize()` function.
//
// Input can be either a local model directory or a HuggingFace repo ID
// (e.g. "Qwen/Qwen3.6-35B-A3B").  In HF mode, files are downloaded one at
// a time from the Hub and deleted after their tensors are processed.
// This keeps peak disk usage near the size of one shard + the output.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::hf_util::HfRepo;
use crate::quant::{Quant, GROUP_SIZE, quant_f32_to_int4, quant_f32_to_int8, f32_to_bf16_u16};
use crate::safetensors::{bytes_to_f32, parse_safetensors, read_tensor_bytes};

const ALIGN: u64 = 64;

/// Embedded at compile time — no external file needed.
const NAME_MAPPING_JSON: &str = include_str!("name_mapping.json");

/// Which Qwen generation this model belongs to.  Qwen3.6 needs a +1.0
/// norm-weight correction relative to Qwen3.5 convention.
#[derive(Clone, Copy, PartialEq)]
pub enum QwenVersion {
    V35,
    V36,
}

impl QwenVersion {
    pub fn is_qwen36(self) -> bool {
        self == QwenVersion::V36
    }
}

pub struct Bq4 {
    strip_layers: usize,
    strip_experts: usize,
    version: QwenVersion,
}

impl Bq4 {
    pub fn new(strip_layers: usize, strip_experts: usize, version: QwenVersion) -> Self {
        Bq4 { strip_layers, strip_experts, version }
    }
}


/// Check for Python Ctrl-C signal.  No-op when built without python-bindings.
fn check_interrupt() -> Result<(), String> {
    #[cfg(feature = "python-bindings")]
    pyo3::Python::with_gil(|py| py.check_signals())
        .map_err(|e| format!("interrupted: {}", e))?;
    Ok(())
}

// ─── Name parsing & classification ───────────────────────────────────────────

pub fn split_on_last_dot(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(idx) => (&name[..idx], &name[idx + 1..]),
        None => (name, ""),
    }
}

pub fn strip_layer_prefix(name: &str) -> &str {
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

pub fn matrix_table(block: &str) -> Quant {
    match block {
        "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" | "self_attn.o_proj"
        | "mlp.gate" | "attn.qkv" | "attn.proj" | "patch_embed.proj" | "pos_embed" => Quant::Bf16,
        "lm_head" => Quant::Int8,
        _ => Quant::Int4,
    }
}

pub fn bq4(mlx_name: &str, shape: &[usize]) -> Quant {
    let (prefix, kind) = split_on_last_dot(mlx_name);
    match kind {
        "A_log" => { debug_assert!(shape.len() <= 1); Quant::Fp32 }
        "scales" | "biases" | "bias" | "dt_bias" => { debug_assert!(shape.len() <= 2); Quant::Bf16 }
        "weight" => {
            if shape.len() != 2 { Quant::Bf16 }
            else { matrix_table(strip_layer_prefix(prefix)) }
        }
        _ => panic!("unknown kind: {:?} in {}", kind, mlx_name),
    }
}

pub fn classify_weight(mlx_name: &str, shape: &[usize]) -> String {
    bq4(mlx_name, shape).as_str().to_string()
}

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
        assert_eq!(strip_layer_prefix("language_model.model.layers.3.self_attn.q_proj"), "self_attn.q_proj");
        assert_eq!(strip_layer_prefix("language_model.lm_head"), "lm_head");
        assert_eq!(strip_layer_prefix("self_attn.q_proj"), "self_attn.q_proj");
    }

    #[test]
    fn test_matrix_table() {
        assert_eq!(matrix_table("self_attn.q_proj"), Quant::Bf16);
        assert_eq!(matrix_table("lm_head"), Quant::Int8);
        assert_eq!(matrix_table("mlp.switch_mlp.gate_up_proj"), Quant::Int4);
    }

    #[test]
    fn test_bq4() {
        assert_eq!(bq4("language_model.model.layers.0.self_attn.q_proj.weight", &[8192, 2048]), Quant::Bf16);
        assert_eq!(bq4("language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight", &[256, 2048]), Quant::Int4);
        assert_eq!(bq4("language_model.model.layers.0.input_layernorm.weight", &[2048]), Quant::Bf16);
        assert_eq!(bq4("language_model.model.layers.0.linear_attn.A_log", &[128]), Quant::Fp32);
        assert_eq!(bq4("language_model.model.embed_tokens.scales", &[32, 32]), Quant::Bf16);
    }

}

/// Extract layer index from tensor name.  Looks for `layers.{N}.` pattern.
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

// ─── Name mapping ────────────────────────────────────────────────────────────

type NameMap = HashMap<String, String>;

fn load_name_mapping(
    json_str: &str,
    num_layers: usize,
    num_vision_blocks: usize,
) -> Result<NameMap, String> {
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
            for b in 0..num_vision_blocks {
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

// ─── HF config ───────────────────────────────────────────────────────────────

struct ModelParams {
    hidden_dim: usize,
    num_layers: usize,
    moe_intermediate: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    shared_intermediate: usize,
    mtp_num_layers: usize,
    full_attn_interval: usize,
    vocab_size: usize,
    num_attn_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    linear_num_value_heads: usize,
    linear_num_key_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    partial_rotary_factor: f64,
    rope_theta: f64,
    hf_config_raw: serde_json::Value,
}

fn load_config(model_path: &Path) -> Result<ModelParams, String> {
    let json_str =
        fs::read_to_string(model_path.join("config.json")).map_err(|e| e.to_string())?;
    let root: serde_json::Value =
        serde_json::from_str(&json_str).map_err(|e| e.to_string())?;
    let tc = root.get("text_config").unwrap_or(&root);

    let get = |key: &str, default: usize| -> usize {
        tc.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
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
        partial_rotary_factor: get_f(
            "partial_rotary_factor",
            rope.and_then(|r| r.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.25),
        ),
        rope_theta: get_f(
            "rope_theta",
            rope.and_then(|r| r.get("rope_theta"))
                .and_then(|v| v.as_f64())
                .unwrap_or(10_000_000.0),
        ),
        hf_config_raw: root,
    })
}

// ─── Sanitization ────────────────────────────────────────────────────────────

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
    MLX_NORM_KEYS
        .iter()
        .chain(MTP_NORM_KEYS)
        .any(|k| mlx_name.ends_with(k))
}

/// np.moveaxis(arr, 2, 1) for 3D f32 arrays: [C, K, S] → [C, S, K]
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

// ─── Expert detection ────────────────────────────────────────────────────────

fn is_expert_tensor(mlx_name: &str) -> bool {
    mlx_name.contains(".switch_mlp.gate_proj.")
        || mlx_name.contains(".switch_mlp.gate_up_proj.")
        || mlx_name.contains(".switch_mlp.up_proj.")
        || mlx_name.contains(".switch_mlp.down_proj.")
}

/// Vision encoder tensors are extracted separately — see quant/extract_vision_encoder.py.
fn is_vision_tensor(mlx_name: &str) -> bool {
    mlx_name.starts_with("vision_tower.")
}

// ─── Main pipeline ───────────────────────────────────────────────────────────

impl Bq4 {
    pub fn quantize(&self, input: &str, output: &str) -> Result<(), String> {

    // ── 0. Local directory or HF repo? ────────────────────────────────
    let repo = if Path::new(input).is_dir() {
        HfRepo::from_local(Path::new(input).to_path_buf())
    } else {
        HfRepo::from_hf(input)?
    };

    // ── 1. Load config ──────────────────────────────────────────────────
    repo.ensure("config.json")?;
    let params = load_config(repo.path())?;
    let hd = params.hidden_dim;
    let num_layers = params.num_layers;
    let mi = params.moe_intermediate;
    let num_experts = params.num_experts;
    let num_experts_per_tok = params.num_experts_per_tok;
    let shared_inter = params.shared_intermediate;
    let mtp_layers = params.mtp_num_layers;
    let full_attn_interval = params.full_attn_interval;
    let vocab_size = params.vocab_size;
    let num_attn_heads = params.num_attn_heads;
    let num_kv_heads = params.num_kv_heads;
    let head_dim = params.head_dim;
    let num_main_layers = num_layers - mtp_layers;

    eprintln!("Model config:");
    eprintln!("  hidden_dim={}, vocab_size={}", hd, vocab_size);
    eprintln!(
        "  num_layers={} (main={}, mtp={})",
        num_layers, num_main_layers, mtp_layers
    );
    eprintln!(
        "  num_experts={}, experts_per_tok={}",
        num_experts, num_experts_per_tok
    );
    eprintln!("  moe_intermediate={}, shared_intermediate={}", mi, shared_inter);

    // ── 2. Load name mapping ────────────────────────────────────────────
    let name_map = load_name_mapping(NAME_MAPPING_JSON, num_layers, 27)?;
    eprintln!("  Name mapping entries: {}", name_map.len());

    // ── 3. Load weight map ──────────────────────────────────────────────
    let index_path = repo.ensure("model.safetensors.index.json")?;
    let weight_map = {
        let idx_str = fs::read_to_string(&index_path).map_err(|e| e.to_string())?;
        let idx: serde_json::Value =
            serde_json::from_str(&idx_str).map_err(|e| e.to_string())?;
        let mut wm = HashMap::new();
        if let Some(map) = idx["weight_map"].as_object() {
            for (k, v) in map {
                wm.insert(k.clone(), v.as_str().unwrap_or("").to_string());
            }
        }
        wm
    };
    eprintln!("  Total tensors: {}", weight_map.len());

    // ── 4. Classify tensors ─────────────────────────────────────────────
    #[derive(Clone)]
    struct Classified {
        hf_name: String,
        shard: String,
        mlx_name: String,
        is_expert: bool,
    }
    let mut classified: Vec<Classified> = Vec::new();
    let mut unmapped: Vec<String> = Vec::new();

    for (hf_name, shard) in weight_map.iter() {
        match name_map.get(hf_name) {
            Some(mlx_name) => {
                if is_vision_tensor(mlx_name) {
                    continue;
                }
                classified.push(Classified {
                    hf_name: hf_name.clone(),
                    shard: shard.clone(),
                    mlx_name: mlx_name.clone(),
                    is_expert: is_expert_tensor(mlx_name),
                });
            }
            None => unmapped.push(hf_name.clone()),
        }
    }

    if !unmapped.is_empty() {
        return Err(format!(
            "{} tensors not in name mapping:\n  {}",
            unmapped.len(),
            unmapped.iter().take(20).cloned().collect::<Vec<_>>().join("\n  ")
        ));
    }

    classified.sort_by(|a, b| a.hf_name.cmp(&b.hf_name));
    let non_expert_count = classified.iter().filter(|c| !c.is_expert).count();
    let expert_count = classified.iter().filter(|c| c.is_expert).count();
    eprintln!("  Non-expert: {}, Expert: {}", non_expert_count, expert_count);

    // ── 4b. Strip mode ──────────────────────────────────────────────────
    let (eff_num_layers, eff_num_experts) = if self.strip_layers > 0 {
        let sl = self.strip_layers;
        let se = if self.strip_experts > 0 { self.strip_experts } else { num_experts };
        eprintln!("  [strip] layers={}, experts={}", sl, se);

        classified.retain(|c| match extract_layer(&c.hf_name) {
            Some(n) => n < sl,
            None => true,
        });

        eprintln!("  [strip] non-expert: {}, expert: {}",
            classified.iter().filter(|c| !c.is_expert).count(),
            classified.iter().filter(|c| c.is_expert).count());
        (sl, se)
    } else {
        (num_layers, num_experts)
    };

    // ── 5. Group by shard ───────────────────────────────────────────────
    let shard_order: Vec<String> = {
        let mut set: HashSet<String> = HashSet::new();
        for c in &classified {
            set.insert(c.shard.clone());
        }
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    };

    // ── 6. Create output directory ──────────────────────────────────────
    let output_dir = Path::new(output);
    fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    let experts_dir = output_dir.join("packed_experts");
    fs::create_dir_all(&experts_dir).map_err(|e| e.to_string())?;

    // ── 7. Process shards one at a time ─────────────────────────────────
    eprintln!("\n============================================================");
    eprintln!("Quantizing non-expert weights (BQ4)...");
    eprintln!("============================================================");

    let mut manifest: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    manifest.insert("model".into(), serde_json::Value::String(input.to_owned()));

    // Build config
    let mut cfg = serde_json::Map::new();
    macro_rules! ins {
        ($k:expr, $v:expr) => {
            cfg.insert($k.into(), serde_json::Value::from($v));
        };
    }
    ins!("hidden_size", hd);
    ins!("num_hidden_layers", eff_num_layers);
    ins!("num_attention_heads", num_attn_heads);
    ins!("num_key_value_heads", num_kv_heads);
    ins!("head_dim", head_dim);
    ins!("vocab_size", vocab_size);
    ins!("rms_norm_eps", params.rms_norm_eps);
    ins!("num_experts", eff_num_experts);
    ins!("num_experts_per_tok", num_experts_per_tok);
    ins!("moe_intermediate_size", mi);
    ins!("shared_expert_intermediate_size", shared_inter);
    ins!("full_attention_interval", full_attn_interval);
    ins!("linear_num_value_heads", params.linear_num_value_heads);
    ins!("linear_num_key_heads", params.linear_num_key_heads);
    ins!("linear_key_head_dim", params.linear_key_head_dim);
    ins!("linear_value_head_dim", params.linear_value_head_dim);
    ins!("linear_conv_kernel_dim", params.linear_conv_kernel_dim);
    ins!("partial_rotary_factor", params.partial_rotary_factor);
    ins!("rope_theta", params.rope_theta);
    ins!("mtp_num_hidden_layers", if self.strip_layers > 0 { 0 } else { mtp_layers });

    let layer_types: Vec<String> = (0..eff_num_layers.min(num_main_layers))
        .map(|i| {
            if (i + 1) % full_attn_interval == 0 {
                "full_attention".to_string()
            } else {
                "linear_attention".to_string()
            }
        })
        .collect();
    cfg.insert("layer_types".into(), serde_json::Value::Array(
        layer_types.iter().map(|s| serde_json::Value::String(s.clone())).collect()
    ));

    manifest.insert("config".into(), serde_json::Value::Object(cfg));
    manifest.insert("num_tensors".into(), serde_json::Value::from(0));

    let bin_path = output_dir.join("model_weights.bin");
    let mut out_f =
        fs::File::create(&bin_path).map_err(|e| format!("cannot create {}: {}", bin_path.display(), e))?;
    let mut offset: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut tensor_count: u64 = 0;
    let mut tensors_map = serde_json::Map::new();
    let mut quant_summary: HashMap<String, usize> = HashMap::new();

    let t0 = std::time::Instant::now();
    let mut t_count: usize = 0;

    for shard_name in &shard_order {
        // Ensure shard is available (download if HF mode)
        let shard_path = repo.ensure(shard_name)?;

        // Parse header
        eprintln!("  Caching header: {}", shard_name);
        let header = parse_safetensors(&shard_path)?;

        // Process tensors belonging to this shard
        let shard_tensors: Vec<&Classified> = classified.iter().filter(|c| c.shard == *shard_name).collect();

        for c in &shard_tensors {
            t_count += 1;
            if t_count % 50 == 0 {
                check_interrupt()?;
            }

            let meta = header.tensors.get(&c.hf_name)
                .ok_or_else(|| format!("{} not found in {}", c.hf_name, shard_name))?;
            let shape = meta.shape.clone();
            let dtype = meta.dtype.clone();

            if c.is_expert {
                // Expert tensors are processed in Phase 8 below.
                continue;
            }

            // ── Non-expert processing ──────────────────────────────
            let mlx_name = c.mlx_name.clone();
            let q = bq4(&mlx_name, &shape);
            let q_str = q.as_str().to_string();
            *quant_summary.entry(q_str.clone()).or_insert(0) += 1;

            // Read source as F32
            let raw_data = read_tensor_bytes(&shard_path, &header, &c.hf_name)?;
            let mut f32_vals = bytes_to_f32(&raw_data, &dtype);
            let mut out_shape = shape.clone();

            // ── Sanitize (BF16 only) ──────────────────────────────
            if q == Quant::Bf16 {
                if self.version.is_qwen36() && is_norm_key(&mlx_name) {
                    for v in &mut f32_vals { *v += 1.0; }
                }
                if mlx_name.contains("conv1d.weight") && out_shape.len() == 3 {
                    moveaxis_2_to_1(&mut f32_vals, &mut out_shape);
                }
            }

            // ── Pad inner dim for INT4 ────────────────────────────
            let out_dim = out_shape[0];
            let in_dim = if out_shape.len() >= 2 { out_shape[1] } else { 0 };
            let (padded_in, f32_padded) = if q == Quant::Int4 {
                let pi = (in_dim + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE;
                if pi != in_dim {
                    let mut p = vec![0.0f32; out_dim * pi];
                    for r in 0..out_dim {
                        let src = r * in_dim;
                        let dst = r * pi;
                        p[dst..dst + in_dim].copy_from_slice(&f32_vals[src..src + in_dim]);
                    }
                    (pi, p)
                } else {
                    (in_dim, f32_vals)
                }
            } else {
                (in_dim, f32_vals)
            };

            // ── Align ────────────────────────────────────────────
            if offset % ALIGN != 0 {
                let pad = ALIGN - (offset % ALIGN);
                out_f.write_all(&vec![0u8; pad as usize]).map_err(|e| e.to_string())?;
                offset += pad;
            }

            // ── Encode ───────────────────────────────────────────
            let base = if mlx_name.ends_with(".weight") {
                mlx_name[..mlx_name.len() - 7].to_string()
            } else {
                mlx_name.clone()
            };

            match q {
                Quant::Int4 => {
                    let (packed, scales, biases) = quant_f32_to_int4(&f32_padded, out_dim, padded_in);
                    let num_groups = padded_in / GROUP_SIZE;

                    let packed_bytes: Vec<u8> =
                        packed.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let scales_bytes: Vec<u8> =
                        scales.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let biases_bytes: Vec<u8> =
                        biases.iter().flat_map(|v| v.to_le_bytes()).collect();

                    for (data, suffix, data_shape, data_dtype) in [
                        (packed_bytes, ".weight", vec![out_dim, padded_in / 8], "u32"),
                        (scales_bytes, ".scales", vec![out_dim, num_groups], "bf16"),
                        (biases_bytes, ".biases", vec![out_dim, num_groups], "bf16"),
                    ] {
                        let tname = format!("{}{}", base, suffix);
                        let dlen = data.len() as u64;
                        out_f.write_all(&data).map_err(|e| e.to_string())?;
                        let mut entry = serde_json::Map::new();
                        entry.insert("offset".into(), serde_json::Value::from(offset));
                        entry.insert("size".into(), serde_json::Value::from(dlen));
                        entry.insert("shape".into(), serde_json::Value::Array(
                            data_shape.iter().map(|&n| serde_json::Value::from(n as u64)).collect(),
                        ));
                        entry.insert("dtype".into(), serde_json::Value::String(data_dtype.into()));
                        tensors_map.insert(tname, serde_json::Value::Object(entry));
                        offset += dlen;
                        total_bytes += dlen;
                        tensor_count += 1;
                    }
                }
                Quant::Fp32 => {
                    let data: Vec<u8> = f32_padded.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let dlen = data.len() as u64;
                    out_f.write_all(&data).map_err(|e| e.to_string())?;
                    let mut entry = serde_json::Map::new();
                    entry.insert("offset".into(), serde_json::Value::from(offset));
                    entry.insert("size".into(), serde_json::Value::from(dlen));
                    entry.insert("shape".into(), serde_json::Value::Array(
                        out_shape.iter().map(|&n| serde_json::Value::from(n as u64)).collect(),
                    ));
                    entry.insert("dtype".into(), serde_json::Value::String("f32".into()));
                    tensors_map.insert(base, serde_json::Value::Object(entry));
                    offset += dlen;
                    total_bytes += dlen;
                    tensor_count += 1;
                }
                Quant::Int8 => {
                    let (packed, scales) = quant_f32_to_int8(&f32_padded, out_dim, in_dim);
                    let packed_bytes: Vec<u8> = unsafe {
                        std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len()).to_vec()
                    };
                    let scales_bytes: Vec<u8> =
                        scales.iter().flat_map(|v| v.to_le_bytes()).collect();

                    for (data, suffix, data_shape, data_dtype) in [
                        (packed_bytes, ".weight", vec![out_dim, in_dim], "u8"),
                        (scales_bytes, ".scales", vec![out_dim], "f32"),
                    ] {
                        let tname = format!("{}{}", base, suffix);
                        let dlen = data.len() as u64;
                        out_f.write_all(&data).map_err(|e| e.to_string())?;
                        let mut entry = serde_json::Map::new();
                        entry.insert("offset".into(), serde_json::Value::from(offset));
                        entry.insert("size".into(), serde_json::Value::from(dlen));
                        entry.insert("shape".into(), serde_json::Value::Array(
                            data_shape.iter().map(|&n| serde_json::Value::from(n as u64)).collect(),
                        ));
                        entry.insert("dtype".into(), serde_json::Value::String(data_dtype.into()));
                        tensors_map.insert(tname, serde_json::Value::Object(entry));
                        offset += dlen;
                        total_bytes += dlen;
                        tensor_count += 1;
                    }
                }
                Quant::Bf16 => {
                    let bf16 = f32_to_bf16_u16(&f32_padded);
                    let data: Vec<u8> = bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let dlen = data.len() as u64;
                    out_f.write_all(&data).map_err(|e| e.to_string())?;
                    let mut entry = serde_json::Map::new();
                    entry.insert("offset".into(), serde_json::Value::from(offset));
                    entry.insert("size".into(), serde_json::Value::from(dlen));
                    entry.insert("shape".into(), serde_json::Value::Array(
                        out_shape.iter().map(|&n| serde_json::Value::from(n as u64)).collect(),
                    ));
                    entry.insert("dtype".into(), serde_json::Value::String("bf16".into()));
                    tensors_map.insert(mlx_name.clone(), serde_json::Value::Object(entry));
                    offset += dlen;
                    total_bytes += dlen;
                    tensor_count += 1;
                }
            }
        }

        // Done with this shard — delete to save storage
        if repo.is_hf() {
            repo.remove(shard_name);
            eprintln!("  Deleted {}", shard_name);
        }
    }

    manifest.insert("num_tensors".into(), serde_json::Value::from(tensor_count));
    manifest.insert("tensors".into(), serde_json::Value::Object(tensors_map));

    let elapsed = t0.elapsed();
    eprintln!(
        "  {} tensors, {:.2} GB",
        tensor_count,
        total_bytes as f64 / 1e9
    );
    eprintln!(
        "  Written in {:.1}s ({:.1} GB/s)",
        elapsed.as_secs_f64(),
        total_bytes as f64 / elapsed.as_secs_f64() / 1e9
    );
    eprintln!("  By dtype: {:?}", quant_summary);

    // Write manifest
    let json_path = output_dir.join("model_weights.json");
    let json_str = serde_json::to_string_pretty(&serde_json::Value::Object(manifest))
        .map_err(|e| e.to_string())?;
    fs::write(&json_path, json_str).map_err(|e| e.to_string())?;
    eprintln!("  Manifest: {}", json_path.display());

    // ── 8. Process experts (HF mode: download shards again as needed) ────
    eprintln!("\n============================================================");
    eprintln!("Quantizing expert weights (int4)...");
    eprintln!("============================================================");

    let t1 = std::time::Instant::now();
    let mut expert_layers_done = 0usize;

    // Group expert tensors by layer
    let mut expert_by_layer: BTreeMap<usize, (String, String)> = BTreeMap::new();
    for c in &classified {
        if !c.is_expert { continue; }
        if let Some(layer) = extract_layer(&c.hf_name) {
            let entry = expert_by_layer.entry(layer).or_insert_with(|| (String::new(), String::new()));
            if c.mlx_name.contains("gate_up_proj") {
                entry.0 = c.hf_name.clone(); // gate_up key
            } else if c.mlx_name.contains("down_proj") {
                entry.1 = c.hf_name.clone(); // down key
            }
        }
    }

    for (layer_idx, (gate_up_key, down_key)) in &expert_by_layer {
        check_interrupt()?;
        if gate_up_key.is_empty() || down_key.is_empty() {
            eprintln!("  Layer {} SKIPPED (missing keys)", layer_idx);
            continue;
        }

        // Determine which shards we need
        let gu_shard = weight_map.get(gate_up_key).ok_or("shard not found")?;
        let down_shard = weight_map.get(down_key).ok_or("shard not found")?;

        // Download the two shards (or one if same)
        let gu_path = repo.ensure(gu_shard)?;
        let down_path = if gu_shard == down_shard {
            gu_path.clone()
        } else {
            repo.ensure(down_shard)?
        };

        let gu_header = parse_safetensors(&gu_path)?;
        let down_header = parse_safetensors(&down_path)?;

        // Read gate_up_proj (fused [E, 2*I, H] as BF16)
        let gu_raw = read_tensor_bytes(&gu_path, &gu_header, gate_up_key)?;
        let gu_f32 = bytes_to_f32(&gu_raw, &gu_header.tensors[gate_up_key].dtype);

        // Read down_proj ([E, H, I] as BF16)
        let down_raw = read_tensor_bytes(&down_path, &down_header, down_key)?;
        let down_f32 = bytes_to_f32(&down_raw, &down_header.tensors[down_key].dtype);

        // Delete shards after reading
        if repo.is_hf() {
            repo.remove(gu_shard);
            if gu_shard != down_shard {
                repo.remove(down_shard);
            }
        }

        // ── Quantize and pack ────────────────────────────────────
        let inter = mi;
        let hidden = hd;
        let gs = GROUP_SIZE;

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

            let bytes: Vec<u8> = gate_p.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[base..base + gate_w_bytes].copy_from_slice(&bytes);
            let mut pos = base + gate_w_bytes;
            let bytes: Vec<u8> = gate_s.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + gate_s_bytes].copy_from_slice(&bytes);
            pos += gate_s_bytes;
            let bytes: Vec<u8> = gate_b.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + gate_b_bytes].copy_from_slice(&bytes);
            pos += gate_b_bytes;
            let bytes: Vec<u8> = up_p.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + up_w_bytes].copy_from_slice(&bytes);
            pos += up_w_bytes;
            let bytes: Vec<u8> = up_s.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + up_s_bytes].copy_from_slice(&bytes);
            pos += up_s_bytes;
            let bytes: Vec<u8> = up_b.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + up_b_bytes].copy_from_slice(&bytes);
            pos += up_b_bytes;
            let bytes: Vec<u8> = down_p.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + down_w_bytes].copy_from_slice(&bytes);
            pos += down_w_bytes;
            let bytes: Vec<u8> = down_s.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + down_s_bytes].copy_from_slice(&bytes);
            pos += down_s_bytes;
            let bytes: Vec<u8> = down_b.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + down_b_bytes].copy_from_slice(&bytes);
        }

        let out_path = experts_dir.join(format!("layer_{:02}.bin", layer_idx));
        fs::write(&out_path, &buf).map_err(|e| e.to_string())?;
        eprintln!(
            "  Layer {:02}: {:.1} MB → {}",
            layer_idx,
            buf.len() as f64 / 1e6,
            out_path.file_name().unwrap().to_string_lossy()
        );
        expert_layers_done += 1;
    }

    let t2 = t1.elapsed();
    eprintln!(
        "\n  {} expert layers in {:.1}s",
        expert_layers_done,
        t2.as_secs_f64()
    );

    // ── 9. Write config.json ────────────────────────────────────────────
    let dst_config = output_dir.join("config.json");
    let config_str = serde_json::to_string_pretty(&params.hf_config_raw)
        .map_err(|e| e.to_string())?;
    fs::write(&dst_config, config_str).map_err(|e| e.to_string())?;

    // ── 10. Summary ─────────────────────────────────────────────────────
    let total_time = t0.elapsed();
    let bin_size = fs::metadata(&bin_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("\n============================================================");
    eprintln!("Done!");
    eprintln!("  model_weights.bin : {:.2} GB", bin_size as f64 / 1e9);
    eprintln!("  model_weights.json: {}", json_path.display());
    eprintln!("  packed_experts    : {} layers", expert_layers_done);
    eprintln!("  Total time        : {:.1}s", total_time.as_secs_f64());
    eprintln!("============================================================");

    Ok(())
}
}
