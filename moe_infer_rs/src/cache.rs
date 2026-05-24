use metal::Buffer;

use crate::constants::{CONV_KERNEL_SIZE, FULL_ATTN_INTERVAL, MAX_SEQ};
use crate::model::config::ModelConfig;

// ─── Cache (data only) ──────────────────────────────────────────────────────

pub struct Cache {
    pub pos: usize,
    pub kv: Vec<Option<FullAttnCache>>,
    pub lin: Vec<Option<LinearAttnState>>,
}

impl Cache {
    pub fn new(config: &ModelConfig) -> Self {
        let num_layers = config.get_usize("num_layers").unwrap();
        let num_kv_heads = config.get_usize("num_kv_heads").unwrap();
        let head_dim = config.get_usize("head_dim").unwrap();
        let kv_dim = num_kv_heads * head_dim;
        let linear_num_v_heads = config.get_usize("linear_num_v_heads").unwrap();
        let linear_total_key = config.get_usize("linear_total_key").unwrap();
        let linear_num_k_heads = config.get_usize("linear_num_k_heads").unwrap();
        let linear_total_value = config.get_usize("linear_total_value").unwrap();
        let linear_conv_dim = config.get_usize("linear_conv_dim").unwrap();

        let mut kv = Vec::with_capacity(num_layers);
        let mut lin = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            if (layer + 1) % FULL_ATTN_INTERVAL == 0 {
                kv.push(Some(FullAttnCache::new(MAX_SEQ, kv_dim)));
                lin.push(None);
            } else {
                kv.push(None);
                lin.push(Some(LinearAttnState::new(
                    linear_num_v_heads,
                    linear_total_key / linear_num_k_heads,
                    linear_total_value / linear_num_v_heads,
                    linear_conv_dim,
                )));
            }
        }
        Cache { pos: 0, kv, lin }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        for kv in self.kv.iter_mut().flatten() {
            kv.reset();
        }
        for s in self.lin.iter_mut().flatten() {
            s.conv_state.fill(0.0);
            s.ssm_state.fill(0.0);
        }
    }
}

// ─── Full attention cache ───────────────────────────────────────────────────

pub struct FullAttnCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub len: usize,
}

impl FullAttnCache {
    pub fn new(max_seq: usize, kv_dim: usize) -> Self {
        FullAttnCache {
            k_cache: vec![0.0f32; max_seq * kv_dim],
            v_cache: vec![0.0f32; max_seq * kv_dim],
            len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

// ─── Linear attention state ─────────────────────────────────────────────────

pub struct LinearAttnState {
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    pub ssm_state_gpu: Option<Buffer>,
}

impl LinearAttnState {
    pub fn new(num_v_heads: usize, key_dim: usize, value_dim: usize, qkv_dim: usize) -> Self {
        LinearAttnState {
            conv_state: vec![0.0f32; (CONV_KERNEL_SIZE - 1) * qkv_dim],
            ssm_state: vec![0.0f32; num_v_heads * value_dim * key_dim],
            ssm_state_gpu: None,
        }
    }
}
