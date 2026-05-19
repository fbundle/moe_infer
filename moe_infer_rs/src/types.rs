// Core model types — mirrors common.h + model_config.h
// ---- ExpertLayout ----
#[derive(Debug, Clone, Default)]
pub struct ExpertLayout {
    pub gate_w_off: i32,
    pub gate_s_off: i32,
    pub gate_b_off: i32,
    pub up_w_off: i32,
    pub up_s_off: i32,
    pub up_b_off: i32,
    pub down_w_off: i32,
    pub down_s_off: i32,
    pub down_b_off: i32,
    pub gate_w_size: i32,
    pub gate_s_size: i32,
    pub gate_b_size: i32,
    pub up_w_size: i32,
    pub up_s_size: i32,
    pub up_b_size: i32,
    pub down_w_size: i32,
    pub down_s_size: i32,
    pub down_b_size: i32,
}

// ---- ModelConfig ----
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_dim: i32,
    pub num_layers: i32,
    pub num_attn_heads: i32,
    pub num_kv_heads: i32,
    pub vocab_size: i32,
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub moe_intermediate: i32,
    pub shared_intermediate: i32,
    pub linear_num_v_heads: i32,
    pub linear_num_k_heads: i32,
    pub rotary_dim: i32,
    pub linear_total_key: i32,
    pub linear_total_value: i32,
    pub linear_conv_dim: i32,
    pub num_full_attn_layers: i32,
    pub num_linear_layers: i32,
    pub expert_size_4bit: i32,
    pub expert_size_2bit: i32,
    pub layout_4bit: ExpertLayout,
    pub layout_2bit: ExpertLayout,
    // Architectural constants from model_config.json
    pub head_dim: i32,
    pub group_size: i32,
    pub full_attn_interval: i32,
    pub conv_kernel_size: i32,
    pub max_seq_len: i32,
    pub gpu_kv_seq: i32,
    pub max_k: i32,
    pub linear_key_dim: i32,
    pub linear_value_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 2048,
            num_layers: 40,
            num_attn_heads: 16,
            num_kv_heads: 2,
            vocab_size: 248320,
            num_experts: 256,
            num_experts_per_tok: 8,
            moe_intermediate: 512,
            shared_intermediate: 512,
            linear_num_v_heads: 32,
            linear_num_k_heads: 16,
            rotary_dim: 64,
            linear_total_key: 2048,
            linear_total_value: 4096,
            linear_conv_dim: 8192,
            num_full_attn_layers: 10,
            num_linear_layers: 30,
            expert_size_4bit: 1769472,
            expert_size_2bit: 983040,
            layout_4bit: ExpertLayout::default(),
            layout_2bit: ExpertLayout::default(),
            head_dim: 256,
            group_size: 64,
            full_attn_interval: 4,
            conv_kernel_size: 4,
            max_seq_len: 1048576,
            gpu_kv_seq: 8192,
            max_k: 8,
            linear_key_dim: 128,
            linear_value_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
        }
    }
}

// ---- TensorInfo / TensorManifest ----
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub ndim: i32,
    pub shape: [i32; 4],
    pub dtype: String,
}

#[derive(Debug, Clone)]
pub struct TensorManifest {
    pub tensors: Vec<TensorInfo>,
}

// ---- WeightFile ----
#[derive(Debug)]
pub struct WeightFile {
    pub data: *mut u8,
    pub size: usize,
    pub manifest: TensorManifest,
}

// ---- KV Cache / Linear Attention State ----
#[derive(Debug)]
pub struct KVCache {
    pub k_cache: Vec<u16>, // bf16
    pub v_cache: Vec<u16>, // bf16
    pub len: i32,
}

#[derive(Debug)]
pub struct LinearAttnState {
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
}

// ---- Timing accum ----
#[derive(Debug, Default)]
pub struct LayerTimingAccum {
    pub deferred_wait: f64,
    pub deferred_cpu: f64,
    pub input_norm: f64,
    pub cmd1_submit: f64,
    pub cmd1_wait: f64,
    pub cpu_attn: f64,
    pub cmd2_encode: f64,
    pub cmd2_wait: f64,
    pub routing_cpu: f64,
    pub spec_route: f64,
    pub expert_io: f64,
    pub cmd3_encode: f64,
    pub total: f64,
    pub count: i32,
}
