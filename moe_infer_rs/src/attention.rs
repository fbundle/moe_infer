// KV Cache and Linear Attention state types.
// The forward-pass logic lives in layer_forward.rs.

use crate::types::*;

// ---- KV Cache ----

impl KVCache {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let kv_size = max_seq_len * num_kv_heads * head_dim;
        Self {
            k_cache: vec![0u16; kv_size],
            v_cache: vec![0u16; kv_size],
            len: 0,
        }
    }
}

// ---- Linear Attention State ----

impl LinearAttnState {
    pub fn new(conv_kernel_size: i32, linear_conv_dim: i32, linear_num_v_heads: i32, linear_value_dim: i32, linear_key_dim: i32) -> Self {
        let conv_len = (conv_kernel_size - 1) as usize * linear_conv_dim as usize;
        let ssm_len = linear_num_v_heads as usize * linear_value_dim as usize * linear_key_dim as usize;
        Self {
            conv_state: vec![0.0f32; conv_len],
            ssm_state: vec![0.0f32; ssm_len],
        }
    }
}
