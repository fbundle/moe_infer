//! Gemma 4 MetalContext — Metal device + compiled shaders + persistent buffers.

#![allow(dead_code)]

use metal::*;

use crate::error::MoEError;
use crate::engine::metal_context::ExpertBuffer;
pub use crate::engine::metal_context::{metal_buf_shared, WeightBuffer, MAX_K, ExpertCache};
use crate::model::weights::WeightFile;
use crate::engine::gemma4_constants::Gemma4ModelConfig;

const SHADER_SOURCE: &str = concat!(
    include_str!("../qwen35_moe/shaders.metal"),
    include_str!("shaders.metal"),
);

pub struct Gemma4MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub library: Library,
    pub pos: std::cell::Cell<usize>,

    // Shared pipelines from qwen35 shaders.
    pub matvec_bf16: ComputePipelineState,
    pub dequant_matvec_4bit_v3: ComputePipelineState,
    pub matvec_int8: ComputePipelineState,
    pub rms_norm_fused_bf16: ComputePipelineState,
    pub residual_add: ComputePipelineState,
    pub q_head_norm_rope: ComputePipelineState,
    pub k_head_norm_rope: ComputePipelineState,
    pub kv_cache_append: ComputePipelineState,
    pub attn_sdpa_fused: ComputePipelineState,
    pub swiglu_fused: ComputePipelineState,

    // Gemma4-specific pipelines.
    pub gelu_fused: ComputePipelineState,
    pub logit_softcap: ComputePipelineState,
    pub attn_sdpa_sliding_causal: ComputePipelineState,
    pub rms_norm_no_scale: ComputePipelineState,

    // Persistent per-token buffers (single-token forward).
    pub buf_hidden: Buffer,
    pub buf_input_normed: Buffer,
    pub buf_post_attn_normed: Buffer,
    pub buf_pre_ff_normed: Buffer,
    pub buf_pre_ff_normed_2: Buffer,
    pub buf_q: Buffer,
    pub buf_k: Buffer,
    pub buf_v: Buffer,
    pub buf_attn_out: Buffer,
    pub buf_o_proj_out: Buffer,
    pub kv_caches_k: Vec<Buffer>,
    pub kv_caches_v: Vec<Buffer>,

    pub buf_mlp_gate: Buffer,
    pub buf_mlp_up: Buffer,
    pub buf_mlp_act: Buffer,
    pub buf_mlp_down: Buffer,
    pub buf_mlp_post: Buffer,

    pub buf_router_normed: Buffer,
    pub buf_router_logits: Buffer,
    pub buf_expert_gate: Buffer,
    pub buf_expert_up: Buffer,
    pub buf_expert_act: Buffer,
    pub buf_expert_out: Buffer,
    pub buf_expert_post: Buffer,

    pub buf_ff_combined: Buffer,
    pub buf_ff_outer_post: Buffer,
    pub buf_logits: Buffer,
}

impl Gemma4MetalContext {
    pub fn new<C: Gemma4ModelConfig>(
        wf: &WeightFile,
        _num_active_experts: usize,
        label: &str,
        expert_cache_count: usize,
    ) -> Result<(Self, WeightBuffer, ExpertBuffer), MoEError> {
        let device = Device::system_default()
            .ok_or_else(|| MoEError::Metal("No Metal device found".into()))?;
        eprintln!("[gemma4-metal] Device: {} (unified={})",
            device.name(),
            if device.has_unified_memory() { "YES" } else { "NO" });

        let queue = device.new_command_queue();

        let compile_opts = CompileOptions::new();
        compile_opts.set_fast_math_enabled(true);
        compile_opts.set_language_version(MTLLanguageVersion::V3_1);

        eprintln!("[gemma4-metal] Compiling concat(qwen35, gemma4) shaders…");
        let library = device
            .new_library_with_source(SHADER_SOURCE, &compile_opts)
            .map_err(|e| MoEError::Shader(format!("Shader compilation failed: {:?}", e)))?;
        eprintln!("[gemma4-metal] Shader compilation OK");

        let make = |name: &str| -> Result<ComputePipelineState, MoEError> {
            let function = library.get_function(name, None)
                .map_err(|e| MoEError::Shader(format!("Shader function '{}' not found: {}", name, e)))?;
            device.new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MoEError::Shader(format!("Pipeline '{}' creation failed: {:?}", name, e)))
        };

        let matvec_bf16            = make("matvec_bf16")?;
        let dequant_matvec_4bit_v3 = make("dequant_matvec_4bit_v3")?;
        let matvec_int8            = make("matvec_int8")?;
        let rms_norm_fused_bf16    = make("rms_norm_fused_bf16")?;
        let residual_add           = make("residual_add")?;
        let q_head_norm_rope       = make("q_head_norm_rope")?;
        let k_head_norm_rope       = make("k_head_norm_rope")?;
        let kv_cache_append        = make("kv_cache_append")?;
        let attn_sdpa_fused        = make("attn_sdpa_fused")?;
        let swiglu_fused           = make("swiglu_fused")?;
        let gelu_fused             = make("gelu_fused")?;
        let logit_softcap          = make("logit_softcap")?;
        let attn_sdpa_sliding_causal = make("attn_sdpa_sliding_causal")?;
        let rms_norm_no_scale      = make("rms_norm_no_scale")?;

        let hidden = C::HIDDEN_DIM;
        let n_q_heads = C::NUM_ATTN_HEADS;
        let head_dim_sliding = C::HEAD_DIM;
        let global_head_dim = head_dim_sliding * 2;
        let q_dim_max = n_q_heads * global_head_dim;
        let kv_dim_max = C::NUM_KV_HEADS * global_head_dim;

        let num_experts = C::NUM_EXPERTS;
        let inter = C::INTERMEDIATE_SIZE;
        let moe_inter = C::MOE_INTERMEDIATE;
        let vocab = C::VOCAB_SIZE;
        let num_layers = C::NUM_LAYERS;
        let expert_size = C::EXPERT_SIZE_4BIT;

        let alloc = |elements: usize| metal_buf_shared(&device, elements * 4);

        let kv_caches_k: Vec<Buffer> = (0..num_layers).map(|_| alloc(crate::constants::MAX_SEQ * kv_dim_max)).collect();
        let kv_caches_v: Vec<Buffer> = (0..num_layers).map(|_| alloc(crate::constants::MAX_SEQ * kv_dim_max)).collect();

        let mut expert_buffer = ExpertBuffer::new(&device, expert_size, hidden, moe_inter, moe_inter);
        if expert_cache_count > 0 {
            expert_buffer.init_cache(&device, expert_cache_count, expert_size);
        }
        let weight_buffer = WeightBuffer::new(&device, wf);

        eprintln!(
            "[gemma4-engine] {} layers hidden={} experts={} mode={}",
            num_layers, hidden, num_experts, label
        );

        let ctx = Self {
            buf_hidden:          alloc(hidden),
            buf_input_normed:    alloc(hidden),
            buf_post_attn_normed:alloc(hidden),
            buf_pre_ff_normed:   alloc(hidden),
            buf_pre_ff_normed_2: alloc(hidden),
            buf_q:               alloc(q_dim_max),
            buf_k:               alloc(kv_dim_max),
            buf_v:               alloc(kv_dim_max),
            buf_attn_out:        alloc(q_dim_max),
            buf_o_proj_out:      alloc(hidden),
            kv_caches_k,
            kv_caches_v,
            buf_mlp_gate:        alloc(inter),
            buf_mlp_up:          alloc(inter),
            buf_mlp_act:         alloc(inter),
            buf_mlp_down:        alloc(hidden),
            buf_mlp_post:        alloc(hidden),
            buf_router_normed:   alloc(hidden),
            buf_router_logits:   alloc(num_experts),
            buf_expert_gate:     alloc(moe_inter),
            buf_expert_up:       alloc(moe_inter),
            buf_expert_act:      alloc(moe_inter),
            buf_expert_out:      alloc(hidden),
            buf_expert_post:     alloc(hidden),
            buf_ff_combined:     alloc(hidden),
            buf_ff_outer_post:   alloc(hidden),
            buf_logits:          alloc(vocab),

            device,
            queue,
            library,
            pos: std::cell::Cell::new(0),

            matvec_bf16,
            dequant_matvec_4bit_v3,
            matvec_int8,
            rms_norm_fused_bf16,
            residual_add,
            q_head_norm_rope,
            k_head_norm_rope,
            kv_cache_append,
            attn_sdpa_fused,
            swiglu_fused,
            gelu_fused,
            logit_softcap,
            attn_sdpa_sliding_causal,
            rms_norm_no_scale,
        };

        Ok((ctx, weight_buffer, expert_buffer))
    }
}
