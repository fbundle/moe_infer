// Runtime model configuration loader -- parses model_config.json without serde.
// Mirrors moe_infer_mlx/core_src/model_config.h

use crate::constants::MAX_K;
use crate::types::*;
use std::fs;

// ---------------------------------------------------------------------------
// Minimal JSON helpers -- single-pass string scanning, no allocation beyond
// the sub-slices we inspect.
// ---------------------------------------------------------------------------

/// Find the position immediately after `"key":` (past colon and whitespace).
fn seek_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\"", key);
    let pos = text.find(&needle)?;
    let tail = &text[pos + needle.len()..];
    let colon = tail.find(':')?;
    Some(tail[colon + 1..].trim_start())
}

/// Parse an integer field, falling back to `default` when the key is missing
/// or the value cannot be parsed.
fn json_int(text: &str, key: &str, default: i32) -> i32 {
    let s = match seek_value(text, key) {
        Some(v) => v,
        None => return default,
    };
    // Stop at any character that cannot be part of an integer literal.
    let end = s
        .find(|c: char| matches!(c, ',' | '}' | ']' | '\n' | '\r' | ' ' | '\t'))
        .unwrap_or(s.len());
    if end == 0 {
        return default;
    }
    s[..end].parse::<i32>().unwrap_or(default)
}

/// Parse a floating-point field, falling back to `default`.
fn json_float(text: &str, key: &str, default: f32) -> f32 {
    let s = match seek_value(text, key) {
        Some(v) => v,
        None => return default,
    };
    let end = s
        .find(|c: char| matches!(c, ',' | '}' | ']' | '\n' | '\r' | ' ' | '\t'))
        .unwrap_or(s.len());
    if end == 0 {
        return default;
    }
    s[..end].trim().parse::<f32>().unwrap_or(default)
}

/// Parse a nested `{ ... }` object and extract its fields into an ExpertLayout.
/// Returns `None` if the key is missing or the value is not an object.
fn json_layout(text: &str, key: &str) -> Option<ExpertLayout> {
    let s = seek_value(text, key)?;
    if !s.starts_with('{') {
        return None;
    }

    // Track brace depth to find the matching closing '}'.
    let mut depth: i32 = 0;
    let mut end = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = i + 1; // include the closing brace
                    break;
                }
            }
            _ => {}
        }
    }
    if end == 0 {
        return None; // unbalanced braces
    }

    let obj = &s[..end];

    Some(ExpertLayout {
        gate_w_off: json_int(obj, "gate_w_off", 0),
        gate_s_off: json_int(obj, "gate_s_off", 0),
        gate_b_off: json_int(obj, "gate_b_off", 0),
        up_w_off: json_int(obj, "up_w_off", 0),
        up_s_off: json_int(obj, "up_s_off", 0),
        up_b_off: json_int(obj, "up_b_off", 0),
        down_w_off: json_int(obj, "down_w_off", 0),
        down_s_off: json_int(obj, "down_s_off", 0),
        down_b_off: json_int(obj, "down_b_off", 0),
        gate_w_size: json_int(obj, "gate_w_size", 0),
        gate_s_size: json_int(obj, "gate_s_size", 0),
        gate_b_size: json_int(obj, "gate_b_size", 0),
        up_w_size: json_int(obj, "up_w_size", 0),
        up_s_size: json_int(obj, "up_s_size", 0),
        up_b_size: json_int(obj, "up_b_size", 0),
        down_w_size: json_int(obj, "down_w_size", 0),
        down_s_size: json_int(obj, "down_s_size", 0),
        down_b_size: json_int(obj, "down_b_size", 0),
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load `model_config.json` from `model_path`, returning a fully-populated
/// `ModelConfig` (with defaults filled in for any missing keys).
pub fn load_config(model_path: &str) -> Result<ModelConfig, String> {
    let path = format!("{}/model_config.json", model_path);
    let text =
        fs::read_to_string(&path).map_err(|e| format!("ERROR: Cannot read {}: {}", path, e))?;

    let mut cfg = ModelConfig::default();

    cfg.hidden_dim = json_int(&text, "hidden_dim", 2048);
    cfg.num_layers = json_int(&text, "num_layers", 40);
    cfg.num_attn_heads = json_int(&text, "num_attn_heads", 16);
    cfg.num_kv_heads = json_int(&text, "num_kv_heads", 2);
    cfg.vocab_size = json_int(&text, "vocab_size", 248320);
    cfg.num_experts = json_int(&text, "num_experts", 256);
    cfg.num_experts_per_tok = json_int(&text, "num_experts_per_tok", 8);
    cfg.moe_intermediate = json_int(&text, "moe_intermediate", 512);
    cfg.shared_intermediate = json_int(&text, "shared_intermediate", 512);
    cfg.linear_num_v_heads = json_int(&text, "linear_num_v_heads", 32);
    cfg.linear_num_k_heads = json_int(&text, "linear_num_k_heads", 16);
    cfg.rotary_dim = json_int(&text, "rotary_dim", 64);
    cfg.linear_total_key = json_int(&text, "linear_total_key", 2048);
    cfg.linear_total_value = json_int(&text, "linear_total_value", 4096);
    cfg.linear_conv_dim = json_int(&text, "linear_conv_dim", 8192);
    cfg.num_full_attn_layers = json_int(&text, "num_full_attn_layers", 10);
    cfg.num_linear_layers = json_int(&text, "num_linear_layers", 30);
    cfg.expert_size_4bit = json_int(&text, "expert_size_4bit", 1769472);
    cfg.expert_size_2bit = json_int(&text, "expert_size_2bit", 983040);

    if let Some(layout) = json_layout(&text, "expert_layout_4bit") {
        cfg.layout_4bit = layout;
    }
    if let Some(layout) = json_layout(&text, "expert_layout_2bit") {
        cfg.layout_2bit = layout;
    }

    cfg.head_dim = json_int(&text, "head_dim", 256);
    cfg.group_size = json_int(&text, "group_size", 64);
    cfg.full_attn_interval = json_int(&text, "full_attn_interval", 4);
    cfg.conv_kernel_size = json_int(&text, "conv_kernel_size", 4);
    cfg.max_seq_len = json_int(&text, "max_seq_len", 1048576);
    cfg.gpu_kv_seq = json_int(&text, "gpu_kv_seq", 8192);
    cfg.max_k = json_int(&text, "max_k", 8);
    cfg.linear_key_dim = json_int(&text, "linear_key_dim", 128);
    cfg.linear_value_dim = json_int(&text, "linear_value_dim", 128);
    cfg.rms_norm_eps = json_float(&text, "rms_norm_eps", 1e-6);
    cfg.rope_theta = json_float(&text, "rope_theta", 10_000_000.0);

    // Validate against compile-time limits.
    if cfg.max_k > MAX_K as i32 {
        return Err(format!(
            "ERROR: model max_k={} exceeds compile-time MAX_K={}",
            cfg.max_k, MAX_K
        ));
    }

    // Print summary (matching the C version's printf output).
    println!(
        "[config] hidden_dim={}, num_layers={} ({} full + {} linear)",
        cfg.hidden_dim, cfg.num_layers, cfg.num_full_attn_layers, cfg.num_linear_layers
    );
    println!(
        "  experts={} (K={}), moe_inter={}, shared_inter={}",
        cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate, cfg.shared_intermediate
    );
    println!(
        "  attn_heads={}, kv_heads={}, vocab={}, head_dim={}",
        cfg.num_attn_heads, cfg.num_kv_heads, cfg.vocab_size, cfg.head_dim
    );
    println!(
        "  linear: v_heads={}, k_heads={}",
        cfg.linear_num_v_heads, cfg.linear_num_k_heads
    );
    println!(
        "  expert_size={} bytes (4-bit), {} bytes (2-bit)",
        cfg.expert_size_4bit, cfg.expert_size_2bit
    );

    Ok(cfg)
}
