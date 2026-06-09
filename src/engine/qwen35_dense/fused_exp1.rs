//! Dense Qwen3.5 engine — single-command-buffer fused forward.
//!
//! Per token: open ONE Metal command buffer, encode embedding-copy + 32 layers
//! (linear-attn or full-attn, both followed by dense MLP) + final norm + tied
//! lm_head, commit once, wait once, read logits. No per-layer CPU sync — that
//! cost only existed in the MoE engine because of expert routing.
//!
//! Buffer reuse: `buf_moe_hidden` is the cross-layer accumulator. Within a
//! layer:
//!     buf_moe_hidden  --input_layernorm-->  buf_qkv_x
//!     attn(buf_qkv_x) --o_proj-->           buf_out_proj
//!     residual_add(buf_out_proj, buf_moe_hidden) → buf_temp_residual
//!     buf_temp_residual --post_attn_norm--> buf_post_normed
//!     mlp(buf_post_normed) --down_proj-->    buf_mlp_down
//!     residual_add(buf_mlp_down, buf_temp_residual) → buf_moe_hidden
//! and the cycle repeats for the next layer.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

use metal::*;
use objc::rc::autoreleasepool;

use crate::cache::Cache;
use crate::constants::{FULL_ATTN_INTERVAL, RMS_NORM_EPS};
use crate::engine::metal_context::{metal_buf_shared, MetalContext, WeightBuffer};
use crate::engine::metal_kernels;
use crate::engine::qwen35_dense_constants::{is_full_attn_layer, ModelConfig};
use crate::engine::{Engine, EngineSnapshot, SignalCheckFn, TelemetryValue};
use crate::error::MoEError;
use crate::math::bf16_to_f32;
use crate::model::Model;

pub struct FusedDense<C: ModelConfig> {
    pub model: Arc<Model>,
    pub ctx: MetalContext,
    pub weight_buffer: WeightBuffer,

    /// Logits scratch [VOCAB_SIZE] f32 — destination of tied lm_head matvec.
    pub buf_logits: Buffer,

    /// Dense-MLP gate projection output [DENSE_INTERMEDIATE] f32.
    pub buf_mlp_gate: Buffer,
    /// Dense-MLP up projection output [DENSE_INTERMEDIATE] f32.
    pub buf_mlp_up: Buffer,
    /// SwiGLU(gate, up) output [DENSE_INTERMEDIATE] f32.
    pub buf_mlp_act: Buffer,
    /// Dense-MLP down projection output [HIDDEN_DIM] f32.
    pub buf_mlp_down: Buffer,

    pub timing: BTreeMap<String, TelemetryValue>,
    _phantom: PhantomData<C>,
}

impl<C: ModelConfig> FusedDense<C> {
    pub fn new(
        model: Arc<Model>,
        _num_active_experts: usize,
        _expert_cache_count: usize,
    ) -> Result<Self, MoEError> {
        C::validate_config(&model.config).map_err(MoEError::Config)?;

        // Dense Qwen3.5 has different GQA constants (4 kv-heads, 4 q-per-kv) than
        // the qwen35_moe shaders' compile-time #defines. Use the dense fork.
        const DENSE_SHADERS: &str = include_str!("shaders.metal");
        let mut ctx = MetalContext::init_with_shaders(DENSE_SHADERS)?;
        ctx.init_linear_attn_buffers(
            C::NUM_LINEAR_LAYERS,
            C::LINEAR_CONV_DIM,
            C::LINEAR_NUM_V_HEADS,
            C::LINEAR_TOTAL_VALUE,
            C::LINEAR_KEY_DIM,
            C::LINEAR_VALUE_DIM,
            C::HIDDEN_DIM,
            /* num_experts */ 0,
            /* shared_intermediate */ 0,
            C::NUM_FULL_ATTN_LAYERS,
            C::KV_DIM,
            C::NUM_ATTN_HEADS,
            C::HEAD_DIM,
            /* q_proj_dim */ C::NUM_ATTN_HEADS * 2 * C::HEAD_DIM,
        );

        let weight_buffer = WeightBuffer::new(&ctx.device, &model.weight_file);

        // Apple's Metal `newBufferWithLength:options:` does NOT zero-initialize
        // StorageModeShared buffers. The DeltaNet recurrence reads conv_state
        // and delta_state at pos=0; uninitialized garbage there silently
        // corrupts the SSM state for the rest of the sequence. Explicitly zero
        // both for every linear-attn layer.
        unsafe {
            for buf in &ctx.buf_conv_state {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, buf.length() as usize);
            }
            for buf in &ctx.buf_delta_state {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, buf.length() as usize);
            }
            // KV cache buffers — should not be read past `pos`, but zero
            // anyway for symmetry / future safety.
            for buf in &ctx.buf_kv_k {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, buf.length() as usize);
            }
            for buf in &ctx.buf_kv_v {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, buf.length() as usize);
            }
        }

        let buf_logits = metal_buf_shared(&ctx.device, C::VOCAB_SIZE * 4);
        let buf_mlp_gate = metal_buf_shared(&ctx.device, C::DENSE_INTERMEDIATE * 4);
        let buf_mlp_up = metal_buf_shared(&ctx.device, C::DENSE_INTERMEDIATE * 4);
        let buf_mlp_act = metal_buf_shared(&ctx.device, C::DENSE_INTERMEDIATE * 4);
        let buf_mlp_down = metal_buf_shared(&ctx.device, C::HIDDEN_DIM * 4);

        eprintln!(
            "[qwen35_dense] init: arch={} hidden={} layers={} kv_heads={} intermediate={}",
            C::EXPECTED_ARCHITECTURE,
            C::HIDDEN_DIM,
            C::NUM_LAYERS,
            C::NUM_KV_HEADS,
            C::DENSE_INTERMEDIATE,
        );

        Ok(FusedDense {
            model, ctx, weight_buffer, buf_logits,
            buf_mlp_gate, buf_mlp_up, buf_mlp_act, buf_mlp_down,
            timing: BTreeMap::new(),
            _phantom: PhantomData,
        })
    }

    fn embed_one(&self, token_id: usize, out: &mut [f32]) {
        let wf = &self.model.weight_file;
        let (Some(w), Some(s), Some(b)) = (
            wf.get_tensor_u32("language_model.model.embed_tokens.weight"),
            wf.get_tensor_u16("language_model.model.embed_tokens.scales"),
            wf.get_tensor_u16("language_model.model.embed_tokens.biases"),
        ) else {
            out.fill(0.0);
            return;
        };
        let w_info = wf.get_tensor_info("language_model.model.embed_tokens.weight").unwrap();
        let packed_cols = w_info.shape[1];
        let s_info = wf.get_tensor_info("language_model.model.embed_tokens.scales").unwrap();
        let num_groups = s_info.shape[1];
        let hidden_dim = C::HIDDEN_DIM;
        let group_size = hidden_dim / num_groups;
        let packed_per_group = group_size / 8;
        let w_row = &w[token_id * packed_cols..];
        let s_row = &s[token_id * num_groups..];
        let b_row = &b[token_id * num_groups..];
        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let bias = bf16_to_f32(b_row[g]);
            let base = g * group_size;
            for p in 0..packed_per_group {
                let packed = w_row[g * packed_per_group + p];
                for n in 0..8 {
                    let nibble = (packed >> (n * 4)) & 0xF;
                    out[base + p * 8 + n] = (nibble as f32) * scale + bias;
                }
            }
        }
    }

    /// Encode one full-attn layer's GPU dispatches into `enc`.
    /// Reads from buf_moe_hidden; writes attention output to buf_out_proj.
    fn encode_full_attn(&self, enc: &ComputeCommandEncoderRef, layer: usize, fa_idx: usize, pos: usize) {
        let c = &self.ctx;
        let gw = &self.weight_buffer;
        let wf = &self.model.weight_file;

        let hidden_dim = C::HIDDEN_DIM;
        let num_q = C::NUM_ATTN_HEADS;
        let num_kv = C::NUM_KV_HEADS;
        let head_dim = C::HEAD_DIM;
        let q_dim = num_q * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv * head_dim;
        let rotary_dim = C::ROTARY_DIM;
        let rope_theta = C::ROPE_THETA as f32;
        let seq_len = pos + 1;

        let prefix = format!("language_model.model.layers.{}.self_attn", layer);
        let buf_moe = c.buf_moe_hidden.as_ref().unwrap();
        let qkv_x = c.buf_qkv_x.as_ref().unwrap();
        let qbuf = c.buf_qkv_q.as_ref().unwrap();
        let kbuf = c.buf_qkv_k.as_ref().unwrap();
        let vbuf = c.buf_qkv_v.as_ref().unwrap();
        let q_out = c.buf_attn_q.as_ref().unwrap();
        let q_gate = c.buf_attn_q_gate.as_ref().unwrap();
        let attn_out = c.buf_attn_out.as_ref().unwrap();
        let kc = &c.buf_kv_k[fa_idx];
        let vc = &c.buf_kv_v[fa_idx];
        let o_proj = c.buf_out_proj.as_ref().unwrap();

        // ── input_layernorm ──
        let inw_ptr = wf.get_tensor_ptr(
            &format!("language_model.model.layers.{}.input_layernorm.weight", layer)).unwrap();
        let inw_off = (inw_ptr as usize - gw.base as usize) as u64;
        {
            let pipe = c.rms_norm_fused_bf16.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(buf_moe), 0);
            enc.set_buffer(1, Some(&gw.buf), inw_off);
            enc.set_buffer(2, Some(qkv_x), 0);
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }

        // ── q/k/v projections (q is double-wide: q+gate stacked) ──
        gw.encode_matvec_into(wf, c, enc, &format!("{}.q_proj", prefix), qkv_x, 0, qbuf, 0, q_proj_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.k_proj", prefix), qkv_x, 0, kbuf, 0, kv_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.v_proj", prefix), qkv_x, 0, vbuf, 0, kv_dim, hidden_dim);

        // ── q_head_norm_rope: q_norm + RoPE on q part, split off gate ──
        {
            let qn_ptr = wf.get_tensor_ptr(&format!("{}.q_norm.weight", prefix))
                .expect("q_norm.weight missing");
            let qn_off = (qn_ptr as usize - gw.base as usize) as u64;
            let pipe = c.q_head_norm_rope.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(qbuf), 0);
            enc.set_buffer(1, Some(&gw.buf), qn_off);
            enc.set_buffer(2, Some(q_out), 0);
            enc.set_buffer(3, Some(q_gate), 0);
            enc.set_bytes(4, 4, &(head_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &(rotary_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &rope_theta as *const f32 as *const c_void);
            enc.set_bytes(7, 4, &(pos as u32) as *const u32 as *const c_void);
            enc.set_bytes(8, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_q as u64, 1, 1),
                MTLSize::new(head_dim as u64, 1, 1));
        }

        // ── k_head_norm_rope: in-place on kbuf ──
        {
            let kn_ptr = wf.get_tensor_ptr(&format!("{}.k_norm.weight", prefix))
                .expect("k_norm.weight missing");
            let kn_off = (kn_ptr as usize - gw.base as usize) as u64;
            let pipe = c.k_head_norm_rope.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(kbuf), 0);
            enc.set_buffer(1, Some(&gw.buf), kn_off);
            enc.set_bytes(2, 4, &(head_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &(rotary_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &rope_theta as *const f32 as *const c_void);
            enc.set_bytes(5, 4, &(pos as u32) as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_kv as u64, 1, 1),
                MTLSize::new(head_dim as u64, 1, 1));
        }

        // ── KV-cache append ──
        {
            let pipe = c.kv_cache_append.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(kbuf), 0);
            enc.set_buffer(1, Some(vbuf), 0);
            enc.set_buffer(2, Some(kc), 0);
            enc.set_buffer(3, Some(vc), 0);
            enc.set_bytes(4, 4, &(pos as u32) as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &(kv_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((kv_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1));
        }

        // ── SDPA (fused online softmax) ──
        {
            let pipe = c.attn_sdpa_fused.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(q_out), 0);
            enc.set_buffer(1, Some(kc), 0);
            enc.set_buffer(2, Some(vc), 0);
            enc.set_buffer(3, Some(attn_out), 0);
            enc.set_bytes(4, 4, &(seq_len as u32) as *const u32 as *const c_void);
            let scale: f32 = 1.0 / (head_dim as f32).sqrt();
            enc.set_bytes(5, 4, &scale as *const f32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_q as u64, 1, 1),
                MTLSize::new(256, 1, 1));
        }

        // ── output gate (multiplicative sigmoid on attn output) ──
        {
            let pipe = c.sigmoid_gate.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(attn_out), 0);
            enc.set_buffer(1, Some(q_gate), 0);
            enc.set_bytes(2, 4, &(q_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((q_dim as u32 + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1));
        }

        // ── o_proj → buf_out_proj ──
        gw.encode_matvec_into(wf, c, enc, &format!("{}.o_proj", prefix),
            attn_out, 0, o_proj, 0, hidden_dim, q_dim);
    }

    /// Encode one linear-attn (DeltaNet) layer's GPU dispatches into `enc`.
    /// Reads from buf_moe_hidden; writes attention output to buf_out_proj.
    fn encode_linear_attn(&self, enc: &ComputeCommandEncoderRef, layer: usize, linear_idx: usize) {
        let c = &self.ctx;
        let gw = &self.weight_buffer;
        let wf = &self.model.weight_file;

        let hidden_dim = C::HIDDEN_DIM;
        let qkv_dim = C::LINEAR_CONV_DIM;
        let total_key = C::LINEAR_TOTAL_KEY;
        let total_value = C::LINEAR_TOTAL_VALUE;
        let num_k_heads = C::LINEAR_NUM_K_HEADS;
        let num_v_heads = C::LINEAR_NUM_V_HEADS;
        let key_dim = C::LINEAR_KEY_DIM;
        let value_dim = C::LINEAR_VALUE_DIM;
        let k_heads_per_v = num_v_heads / num_k_heads;
        let inv_scale = 1.0 / (key_dim as f32).sqrt();

        let prefix = format!("language_model.model.layers.{}.linear_attn", layer);
        let buf_moe = c.buf_moe_hidden.as_ref().unwrap();
        let qkv_x = c.buf_qkv_x.as_ref().unwrap();

        // ── input_layernorm ──
        let inw_ptr = wf.get_tensor_ptr(
            &format!("language_model.model.layers.{}.input_layernorm.weight", layer)).unwrap();
        let inw_off = (inw_ptr as usize - gw.base as usize) as u64;
        {
            let pipe = c.rms_norm_fused_bf16.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(buf_moe), 0);
            enc.set_buffer(1, Some(&gw.buf), inw_off);
            enc.set_buffer(2, Some(qkv_x), 0);
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }

        // ── projections: in_proj_qkv (qkv stack), in_proj_z (gate), in_proj_a/b (alpha/beta) ──
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_qkv", prefix), qkv_x, 0, &c.batch_out[0], 0, qkv_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_z", prefix), qkv_x, 0, &c.batch_out[1], 0, total_value, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_b", prefix), qkv_x, 0, &c.batch_out[2], 0, num_v_heads, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_a", prefix), qkv_x, 0, &c.batch_out[3], 0, num_v_heads, hidden_dim);

        // ── conv1d_step on the qkv stack ──
        if let Some(conv_w_ptr) = wf.get_tensor_ptr(&format!("{}.conv1d.weight", prefix)) {
            let conv_w_off = (conv_w_ptr as usize - gw.base as usize) as u64;
            metal_kernels::encode_conv1d_step(c, enc,
                &c.buf_conv_state[linear_idx],
                &c.batch_out[0],
                &gw.buf, conv_w_off,
                c.buf_conv_output.as_ref().unwrap(),
                qkv_dim as u32);
        }

        // Debug capture: conv1d output for layer 0 → slot N+5..+7 (qkv_dim=8192 needs 3.2 slots; copy 2560-element chunks)
        if std::env::var("QWEN35_DENSE_CAPTURE_LA0").is_ok() && layer == 0 {
            use crate::engine::metal_kernels::encode_buffer_copy_f32 as cp;
            // Just first 2560 of V (post-conv1d+SiLU) → slot N+6
            cp(c, enc, c.buf_conv_output.as_ref().unwrap(), (4096 * 4) as u64,
                &self.buf_logits, ((C::NUM_LAYERS + 6) * hidden_dim * 4) as u64,
                2560);
            // First 2560 of in_proj_qkv output V section (batch_out[0]) → slot N+5
            cp(c, enc, &c.batch_out[0], (4096 * 4) as u64,
                &self.buf_logits, ((C::NUM_LAYERS + 5) * hidden_dim * 4) as u64,
                2560);
            // buf_qkv_x (input to in_proj matvecs, after input_layernorm) → slot N+7
            cp(c, enc, qkv_x, 0,
                &self.buf_logits, ((C::NUM_LAYERS + 7) * hidden_dim * 4) as u64,
                hidden_dim as u32);
            // buf_moe_hidden (input to input_layernorm = should be the token embedding) → slot N+8
            cp(c, enc, buf_moe, 0,
                &self.buf_logits, ((C::NUM_LAYERS + 8) * hidden_dim * 4) as u64,
                hidden_dim as u32);
        }

        // ── rms_norm_qk: per-head norm on Q and K segments of conv_output ──
        metal_kernels::encode_rms_norm_qk(c, enc,
            c.buf_conv_output.as_ref().unwrap(), 0,
            c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,
            num_k_heads as u32, key_dim as u32, inv_scale);

        // ── compute_decay_beta(alpha, beta, A_log, dt_bias) → g_decay, beta_gate ──
        {
            let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
            let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
            let a_log_off = a_log_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
            let dt_bias_off = dt_bias_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
            metal_kernels::encode_compute_decay_beta(c, enc,
                &c.batch_out[3], &c.batch_out[2],
                if a_log_ptr.is_some() { &gw.buf } else { &c.batch_out[3] }, a_log_off,
                if dt_bias_ptr.is_some() { &gw.buf } else { &c.batch_out[2] }, dt_bias_off,
                c.buf_delta_g_decay.as_ref().unwrap(),
                c.buf_delta_beta.as_ref().unwrap(),
                num_v_heads as u32);
        }

        // ── gated_delta_net_step: SSM recurrence ──
        {
            let q_off = 0u64;
            let k_off = (total_key * 4) as u64;
            let v_off = (2 * total_key * 4) as u64;
            let conv_out = c.buf_conv_output.as_ref().unwrap();
            metal_kernels::encode_gated_delta_net_step(c, enc,
                &c.buf_delta_state[linear_idx],
                conv_out, q_off,
                conv_out, k_off,
                conv_out, v_off,
                c.buf_delta_g_decay.as_ref().unwrap(),
                c.buf_delta_beta.as_ref().unwrap(),
                c.buf_delta_output.as_ref().unwrap(),
                num_v_heads as u32, k_heads_per_v as u32, key_dim as u32, value_dim as u32);
        }

        // ── gated_rms_norm: z-gated output normalization ──
        {
            let gnw_ptr = wf.get_tensor_ptr(&format!("{}.norm.weight", prefix));
            if let Some(gnw_p) = gnw_ptr {
                let gnw_off = (gnw_p as usize - gw.base as usize) as u64;
                metal_kernels::encode_gated_rms_norm(c, enc,
                    c.buf_delta_output.as_ref().unwrap(),
                    &c.batch_out[1],
                    &gw.buf, gnw_off,
                    &c.batch_out[6],
                    num_v_heads as u32, value_dim as u32);
            }
        }

        // Capture intermediates between conv1d and out_proj — slots N+9, N+10, N+11
        if std::env::var("QWEN35_DENSE_CAPTURE_LA0").is_ok() && layer == 0 {
            use crate::engine::metal_kernels::encode_buffer_copy_f32 as cp;
            // First 2560 of buf_delta_output (after gated_delta_net_step) → slot N+9
            cp(c, enc, c.buf_delta_output.as_ref().unwrap(), 0,
                &self.buf_logits, ((C::NUM_LAYERS + 9) * hidden_dim * 4) as u64,
                2560);
            // First 2560 of batch_out[6] (after gated_rms_norm) → slot N+10
            cp(c, enc, &c.batch_out[6], 0,
                &self.buf_logits, ((C::NUM_LAYERS + 10) * hidden_dim * 4) as u64,
                2560);
            // First 2560 of conv_output Q segment (post-rms_norm_qk) → slot N+11
            cp(c, enc, c.buf_conv_output.as_ref().unwrap(), 0,
                &self.buf_logits, ((C::NUM_LAYERS + 11) * hidden_dim * 4) as u64,
                2048);
        }

        // ── out_proj → buf_out_proj ──
        let o_proj = c.buf_out_proj.as_ref().unwrap();
        gw.encode_matvec_into(wf, c, enc, &format!("{}.out_proj", prefix),
            &c.batch_out[6], 0, o_proj, 0, hidden_dim, total_value);

        if std::env::var("QWEN35_DENSE_CAPTURE_LA0").is_ok() && layer == 0 {
            use crate::engine::metal_kernels::encode_buffer_copy_f32 as cp;
            // Full buf_out_proj (linear-attn output, before residual_add) → slot N+12
            cp(c, enc, o_proj, 0,
                &self.buf_logits, ((C::NUM_LAYERS + 12) * hidden_dim * 4) as u64,
                hidden_dim as u32);
        }
    }

    /// Like `encode_post_attn_and_mlp` but with debug captures at the probe layer.
    fn encode_post_attn_and_mlp_traced(
        &self, enc: &ComputeCommandEncoderRef, layer: usize, capture_at: Option<usize>,
    ) {
        self.encode_post_attn_and_mlp_impl(enc, layer, capture_at == Some(layer));
    }

    fn encode_post_attn_and_mlp(&self, enc: &ComputeCommandEncoderRef, layer: usize) {
        self.encode_post_attn_and_mlp_impl(enc, layer, false);
    }

    /// Encode: residual_add(o_proj, buf_moe_hidden) → buf_temp_residual;
    /// post_attention_layernorm(buf_temp_residual) → buf_post_normed;
    /// dense MLP(buf_post_normed) → buf_mlp_down;
    /// residual_add(buf_mlp_down, buf_temp_residual) → buf_moe_hidden.
    fn encode_post_attn_and_mlp_impl(&self, enc: &ComputeCommandEncoderRef, layer: usize, trace: bool) {
        let c = &self.ctx;
        let gw = &self.weight_buffer;
        let wf = &self.model.weight_file;
        let hidden_dim = C::HIDDEN_DIM;
        let inter = C::DENSE_INTERMEDIATE;

        let o_proj = c.buf_out_proj.as_ref().unwrap();
        let buf_moe = c.buf_moe_hidden.as_ref().unwrap();
        let temp_res = c.buf_temp_residual.as_ref().unwrap();
        let post_normed = c.buf_post_normed.as_ref().unwrap();

        // ── residual_add(o_proj, buf_moe_hidden) → buf_temp_residual ──
        {
            let pipe = c.residual_add.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(o_proj), 0);
            enc.set_buffer(1, Some(buf_moe), 0);
            enc.set_buffer(2, Some(temp_res), 0);
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1));
        }

        // ── post_attention_layernorm(buf_temp_residual) → buf_post_normed ──
        let pnw_ptr = wf.get_tensor_ptr(
            &format!("language_model.model.layers.{}.post_attention_layernorm.weight", layer)).unwrap();
        let pnw_off = (pnw_ptr as usize - gw.base as usize) as u64;
        {
            let pipe = c.rms_norm_fused_bf16.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(temp_res), 0);
            enc.set_buffer(1, Some(&gw.buf), pnw_off);
            enc.set_buffer(2, Some(post_normed), 0);
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }

        // ── Dense MLP: gate_proj + up_proj + swiglu + down_proj ──
        let mlp_prefix = format!("language_model.model.layers.{}.mlp", layer);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.gate_proj", mlp_prefix),
            post_normed, 0, &self.buf_mlp_gate, 0, inter, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.up_proj", mlp_prefix),
            post_normed, 0, &self.buf_mlp_up, 0, inter, hidden_dim);
        metal_kernels::encode_swiglu(c, enc,
            &self.buf_mlp_gate, 0, &self.buf_mlp_up, 0,
            &self.buf_mlp_act, 0, inter as u32);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.down_proj", mlp_prefix),
            &self.buf_mlp_act, 0, &self.buf_mlp_down, 0, hidden_dim, inter);

        if trace {
            use crate::engine::metal_kernels::encode_buffer_copy_f32 as cp;
            // Slot N+2 = buf_post_normed (input to MLP, after post_attn_norm)
            cp(c, enc, post_normed, 0,
                &self.buf_logits, ((C::NUM_LAYERS + 2) * hidden_dim * 4) as u64,
                hidden_dim as u32);
            // Slot N+3 = buf_mlp_down (MLP output before final residual)
            cp(c, enc, &self.buf_mlp_down, 0,
                &self.buf_logits, ((C::NUM_LAYERS + 3) * hidden_dim * 4) as u64,
                hidden_dim as u32);
            // Slot N+4 = buf_temp_residual (post-attn residual; input to post_attn_norm)
            cp(c, enc, temp_res, 0,
                &self.buf_logits, ((C::NUM_LAYERS + 4) * hidden_dim * 4) as u64,
                hidden_dim as u32);
            // Slot N+13 = first 2560 of buf_mlp_gate (gate_proj output)
            cp(c, enc, &self.buf_mlp_gate, 0,
                &self.buf_logits, ((C::NUM_LAYERS + 13) * hidden_dim * 4) as u64,
                hidden_dim as u32);
            // Slot N+14 = first 2560 of buf_mlp_up
            cp(c, enc, &self.buf_mlp_up, 0,
                &self.buf_logits, ((C::NUM_LAYERS + 14) * hidden_dim * 4) as u64,
                hidden_dim as u32);
            // Slot N+15 = first 2560 of buf_mlp_act (after swiglu)
            cp(c, enc, &self.buf_mlp_act, 0,
                &self.buf_logits, ((C::NUM_LAYERS + 15) * hidden_dim * 4) as u64,
                hidden_dim as u32);
        }

        // ── residual_add(buf_mlp_down, buf_temp_residual) → buf_moe_hidden ──
        {
            let pipe = c.residual_add.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&self.buf_mlp_down), 0);
            enc.set_buffer(1, Some(temp_res), 0);
            enc.set_buffer(2, Some(buf_moe), 0);
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1));
        }
    }

    /// Encode final_norm(buf_moe_hidden) → buf_qkv_x; tied-lm_head matvec → buf_logits.
    fn encode_final_norm_and_lm_head(&self, enc: &ComputeCommandEncoderRef) {
        let c = &self.ctx;
        let gw = &self.weight_buffer;
        let wf = &self.model.weight_file;
        let hidden_dim = C::HIDDEN_DIM;

        let buf_moe = c.buf_moe_hidden.as_ref().unwrap();
        let qkv_x = c.buf_qkv_x.as_ref().unwrap();

        let fnw_ptr = wf.get_tensor_ptr("language_model.model.norm.weight").unwrap();
        let fnw_off = (fnw_ptr as usize - gw.base as usize) as u64;
        {
            let pipe = c.rms_norm_fused_bf16.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(buf_moe), 0);
            enc.set_buffer(1, Some(&gw.buf), fnw_off);
            enc.set_buffer(2, Some(qkv_x), 0);
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }

        // Tied lm_head: matvec against embed_tokens.weight (treated as [vocab, hidden]).
        gw.encode_matvec_into(wf, c, enc, "language_model.model.embed_tokens",
            qkv_x, 0, &self.buf_logits, 0, C::VOCAB_SIZE, hidden_dim);
    }
}

impl<C: ModelConfig> Engine for FusedDense<C> {
    fn upload_cache(&self, _cache: &Cache) {}
    fn download_cache(&self, _cache: &mut Cache) {}

    fn engine_pos(&self) -> usize { self.ctx.pos.get() }

    fn embed_lookup(&self, token_ids: &[i64], embeddings: &mut [f32]) {
        let hidden_dim = C::HIDDEN_DIM;
        for (i, &id) in token_ids.iter().enumerate() {
            self.embed_one(id as usize, &mut embeddings[i * hidden_dim..(i + 1) * hidden_dim]);
        }
    }

    fn forward_hidden(
        &mut self,
        embeddings: &[f32],
        _check_signal: SignalCheckFn<'_>,
        _mtp: bool,
    ) -> Result<Vec<f32>, MoEError> {
        let hidden_dim = C::HIDDEN_DIM;
        let n_tokens = embeddings.len() / hidden_dim;
        let vocab_size = C::VOCAB_SIZE;

        let mut logits = vec![0.0f32; n_tokens * vocab_size];
        if n_tokens == 0 { return Ok(logits); }

        let mut pos = self.ctx.pos.get();
        // Grow KV cache up-front to cover this whole forward (pos + n_tokens).
        // Doing it before the per-token loop means at most one grow per call,
        // and no in-flight command buffer can reference the old buffers
        // because we're between forward() invocations.
        self.ctx.ensure_max_seq(pos + n_tokens);
        for ti in 0..n_tokens {

            // ── Copy this token's embedding → buf_moe_hidden ──
            {
                let buf = self.ctx.buf_moe_hidden.as_ref().unwrap();
                let src = &embeddings[ti * hidden_dim..(ti + 1) * hidden_dim];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        buf.contents() as *mut f32,
                        hidden_dim);
                }
            }

            // Debug instrumentation, all gated on `QWEN35_DENSE_DEBUG` so non-debug
            // forwards pay zero cost.
            //   - QWEN35_DENSE_STOP_LAYER=N : after running N layers, copy buf_moe_hidden
            //     to the first HIDDEN_DIM slots of buf_logits and skip final_norm+lm_head.
            //   - QWEN35_DENSE_CAPTURE_ALL=1 : after EACH layer, copy buf_moe_hidden to
            //     buf_logits[layer * HIDDEN_DIM .. (layer+1) * HIDDEN_DIM]. Skip
            //     final_norm+lm_head. Python reads logits.reshape(-1)[:N_LAYERS*HIDDEN_DIM]
            //     .reshape(N_LAYERS, HIDDEN_DIM) for the per-layer trace.
            //   - QWEN35_DENSE_CAPTURE_MID=L : at layer L, additionally capture three
            //     intermediate states (post-input-norm, post-attn-residual, post-mlp)
            //     into buf_logits at offsets [N_LAYERS+0, N_LAYERS+1, N_LAYERS+2] × HIDDEN_DIM.
            let stop_at_layer: usize = std::env::var("QWEN35_DENSE_STOP_LAYER")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(C::NUM_LAYERS);
            let capture_all = std::env::var("QWEN35_DENSE_CAPTURE_ALL").is_ok();
            let capture_mid_at: Option<usize> = std::env::var("QWEN35_DENSE_CAPTURE_MID")
                .ok().and_then(|s| s.parse().ok());

            // ── ONE command buffer for the whole network ──
            autoreleasepool(|| {
                use crate::engine::metal_kernels::encode_buffer_copy_f32 as cp;
                let cb = self.ctx.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();

                let n_run = stop_at_layer.min(C::NUM_LAYERS);
                for layer in 0..n_run {
                    if is_full_attn_layer(layer) {
                        let fa_idx = layer / FULL_ATTN_INTERVAL;
                        self.encode_full_attn(&enc, layer, fa_idx, pos);
                    } else {
                        let linear_idx = layer - (layer + 1) / FULL_ATTN_INTERVAL;
                        self.encode_linear_attn(&enc, layer, linear_idx);
                    }

                    // Capture attention output (before residual) at the probe layer.
                    if capture_mid_at == Some(layer) {
                        let mid_off = ((C::NUM_LAYERS + 1) * hidden_dim * 4) as u64;
                        cp(&self.ctx, &enc,
                            self.ctx.buf_out_proj.as_ref().unwrap(), 0,
                            &self.buf_logits, mid_off,
                            hidden_dim as u32);
                    }

                    self.encode_post_attn_and_mlp_traced(&enc, layer, capture_mid_at);

                    // Capture post-MLP (= layer output = next layer's input).
                    if capture_all {
                        let off = (layer * hidden_dim * 4) as u64;
                        cp(&self.ctx, &enc,
                            self.ctx.buf_moe_hidden.as_ref().unwrap(), 0,
                            &self.buf_logits, off,
                            hidden_dim as u32);
                    }
                }

                if !capture_all && stop_at_layer >= C::NUM_LAYERS {
                    self.encode_final_norm_and_lm_head(&enc);
                } else if !capture_all {
                    cp(&self.ctx, &enc,
                        self.ctx.buf_moe_hidden.as_ref().unwrap(), 0,
                        &self.buf_logits, 0,
                        hidden_dim as u32);
                }

                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            });

            // ── Read logits for this token ──
            let logits_slice = &mut logits[ti * vocab_size..(ti + 1) * vocab_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buf_logits.contents() as *const f32,
                    logits_slice.as_mut_ptr(),
                    vocab_size);
            }

            pos += 1;
            self.ctx.pos.set(pos);
        }

        Ok(logits)
    }

    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> { self.timing.clone() }

    fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot { pos: self.ctx.pos.get(), ..EngineSnapshot::default() }
    }

    fn restore(&mut self, snap: &EngineSnapshot) { self.ctx.pos.set(snap.pos); }
}
