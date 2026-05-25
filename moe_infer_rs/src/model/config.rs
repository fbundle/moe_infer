use std::path::Path;

const GROUP_SIZE: usize = 64;

/// Runtime model configuration — JSON-like key-value store.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    data: serde_json::Map<String, serde_json::Value>,
}

impl ModelConfig {
    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.data.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.data.get(key).and_then(|v| v.as_f64())
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.data.get(key).and_then(|v| v.as_str())
    }

    pub fn get_object(&self, key: &str) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.data.get(key).and_then(|v| v.as_object())
    }

    /// Convenience: access a nested key like "expert_layout_4bit.gate_w_off".
    pub fn get_nested_usize(&self, key: &str) -> Option<usize> {
        let (obj_key, field) = key.split_once('.')?;
        self.get_object(obj_key)?.get(field)?.as_u64().map(|v| v as usize)
    }

    pub fn usize_or(&self, key: &str, default: usize) -> usize {
        self.get_usize(key).unwrap_or(default)
    }

    pub fn from_map(data: serde_json::Map<String, serde_json::Value>) -> Self {
        ModelConfig { data }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &serde_json::Value)> {
        self.data.iter()
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

// ─── Expert layout helpers ───────────────────────────────────────────────

fn layout_to_json(hd: usize, mi: usize, gs: usize) -> serde_json::Map<String, serde_json::Value> {
    let gate_w = mi * hd / 2;
    let gate_sb = mi * (hd / gs) * 2;
    let up_w = gate_w;
    let up_sb = gate_sb;
    let down_w = hd * mi / 2;
    let down_sb = hd * (mi / gs) * 2;

    let gate_w_off = 0;
    let gate_s_off = gate_w;
    let gate_b_off = gate_w + gate_sb;
    let up_w_off = gate_w + 2 * gate_sb;
    let up_s_off = up_w_off + up_w;
    let up_b_off = up_s_off + up_sb;
    let down_w_off = up_b_off + up_sb;
    let down_s_off = down_w_off + down_w;
    let down_b_off = down_s_off + down_sb;

    serde_json::json!({
        "gate_w_off": gate_w_off,
        "gate_s_off": gate_s_off,
        "gate_b_off": gate_b_off,
        "up_w_off": up_w_off,
        "up_s_off": up_s_off,
        "up_b_off": up_b_off,
        "down_w_off": down_w_off,
        "down_s_off": down_s_off,
        "down_b_off": down_b_off,
        "gate_w_size": gate_w,
        "gate_s_size": gate_sb,
        "gate_b_size": gate_sb,
        "up_w_size": up_w,
        "up_s_size": up_sb,
        "up_b_size": up_sb,
        "down_w_size": down_w,
        "down_s_size": down_sb,
        "down_b_size": down_sb,
    }).as_object().unwrap().clone()
}

// ─── Derived field computation ───────────────────────────────────────────

fn add_derived_fields(root: &serde_json::Value, data: &mut serde_json::Map<String, serde_json::Value>) {
    let hd = data.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let mi = data.get("moe_intermediate_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let num_layers = data.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let num_kv_heads = data.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let head_dim = data.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let lnum_k = data.get("linear_num_key_heads").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let lnum_v = data.get("linear_num_value_heads").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let lkey_dim = data.get("linear_key_head_dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let lval_dim = data.get("linear_value_head_dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let fa_interval = data.get("full_attention_interval").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let rope_theta = root.get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(|x| x.as_f64())
        .unwrap_or(10000.0);
    let partial_rotary = root.get("rope_parameters")
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|x| x.as_f64())
        .unwrap_or(0.25) as f32;
    let rotary_dim = (head_dim as f32 * partial_rotary) as usize;

    let linear_total_key = lnum_k * lkey_dim;
    let linear_total_value = lnum_v * lval_dim;
    let linear_conv_dim = linear_total_key * 2 + linear_total_value;
    let num_full_attn_layers = num_layers / fa_interval;
    let num_linear_layers = num_layers - num_full_attn_layers;
    let kv_dim = num_kv_heads * head_dim;

    let layout_4bit = layout_to_json(hd, mi, GROUP_SIZE);
    let expert_size_4bit = layout_4bit.get("down_b_off").unwrap().as_u64().unwrap() as usize
        + layout_4bit.get("down_b_size").unwrap().as_u64().unwrap() as usize;
    let layout_2bit = layout_to_json(hd, mi, GROUP_SIZE);
    let expert_size_2bit = layout_2bit.get("down_b_off").unwrap().as_u64().unwrap() as usize
        + layout_2bit.get("down_b_size").unwrap().as_u64().unwrap() as usize;

    let extras = serde_json::json!({
        "linear_total_key": linear_total_key,
        "linear_total_value": linear_total_value,
        "linear_conv_dim": linear_conv_dim,
        "num_full_attn_layers": num_full_attn_layers,
        "num_linear_layers": num_linear_layers,
        "rotary_dim": rotary_dim,
        "rope_theta": rope_theta,
        "kv_dim": kv_dim,
        "expert_size_4bit": expert_size_4bit,
        "expert_size_2bit": expert_size_2bit,
        "expert_layout_4bit": layout_4bit,
        "expert_layout_2bit": layout_2bit,
        "group_size": GROUP_SIZE,
        "bits": 4,
    });
    if let Some(obj) = extras.as_object() {
        for (k, v) in obj {
            data.insert(k.clone(), v.clone());
        }
    }

    // Backward-compat aliases (normalized names → raw JSON names)
    let aliases: &[(&str, &str)] = &[
        ("hidden_dim", "hidden_size"),
        ("num_layers", "num_hidden_layers"),
        ("moe_intermediate", "moe_intermediate_size"),
        ("shared_intermediate", "shared_expert_intermediate_size"),
        ("num_attn_heads", "num_attention_heads"),
        ("num_kv_heads", "num_key_value_heads"),
        ("linear_num_v_heads", "linear_num_value_heads"),
        ("linear_num_k_heads", "linear_num_key_heads"),
        ("linear_key_dim", "linear_key_head_dim"),
        ("linear_value_dim", "linear_value_head_dim"),
    ];
    for (alias, raw) in aliases {
        if let Some(v) = data.get(*raw) {
            data.insert(alias.to_string(), v.clone());
        }
    }
}

// ─── Loader ──────────────────────────────────────────────────────────────

/// Load model configuration from an HF config.json file.
pub fn load_model_config(model_path: &Path) -> anyhow::Result<ModelConfig> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)?;
    let root: serde_json::Value = serde_json::from_str(&content)?;

    // Resolve text_config
    let tc = root.get("text_config").unwrap_or(&root);
    let mut data = tc.as_object().cloned().unwrap_or_default();

    // Add root-level keys that aren't in text_config (architectures, etc.)
    if let Some(root_obj) = root.as_object() {
        for (k, v) in root_obj {
            if k != "text_config" && !data.contains_key(k) {
                data.insert(k.clone(), v.clone());
            }
        }
    }

    add_derived_fields(&root, &mut data);
    data.insert("model_path".to_string(), model_path.to_string_lossy().to_string().into());

    Ok(ModelConfig { data })
}
