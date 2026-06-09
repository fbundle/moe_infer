//! Gemma 4 12B dense engine — Phase 4 (per-layer-type dims).
//!
//! Layers come in two shapes:
//!   * sliding:  head_dim=256, q_dim=4096, kv_dim=2048, 8 KV heads (GQA 2:1)
//!   * full:     head_dim=512, q_dim=8192, kv_dim= 512, 1 KV head  (GQA 16:1)
//!
//! Full-attn layers also use the K=V trick (no separate v_proj — K is reused
//! as V) and a partial RoPE on 0.25 × head_dim = 128 elements (vs sliding's
//! full 256). Per-token scratch buffers (`buf_q`, `buf_attn_out`, `buf_k`)
//! are sized to the LARGER of the two shapes so a single allocation handles
//! both layer types; KV cache is allocated per-layer based on layer type.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};
use objc::rc::autoreleasepool;

use crate::constants::RMS_NORM_EPS;
use crate::engine::metal_context::{metal_buf_shared, MetalContext, WeightBuffer};
use crate::engine::metal_kernels::encode_buffer_copy_f32;
use crate::engine::gemma4_dense_constants::ModelConfig;
use crate::engine::{Engine, EngineSnapshot, SignalCheckFn, TelemetryValue};
use crate::error::MoEError;
use crate::math::bf16_to_f32;
use crate::model::Model;

pub struct FusedGemma4Dense<C: ModelConfig> {
    pub model: Arc<Model>,
    pub ctx: MetalContext,
    pub weight_buffer: WeightBuffer,

    /// Per-layer KV cache K. Size depends on layer type:
    /// sliding → MAX_SEQ * KV_DIM_SLIDING, full → MAX_SEQ * KV_DIM_FULL.
    pub buf_kv_k: Vec<Buffer>,
    /// Same as buf_kv_k but for V. On full-attn layers this is the same
    /// allocation as buf_kv_k[layer] (K=V trick) — we *alias* by storing
    /// a clone of the K buffer here so the SDPA dispatch can just read
    /// from buf_kv_v[layer] without a special-case.
    pub buf_kv_v: Vec<Buffer>,

    // ── Per-token scratch — sized to MAX of sliding/full ──
    pub buf_hidden: Buffer,        // [HIDDEN_DIM]
    pub buf_normed: Buffer,        // [HIDDEN_DIM]
    pub buf_attn_out: Buffer,      // [Q_DIM_FULL] (= 8192, > Q_DIM_SLIDING=4096)
    pub buf_q: Buffer,             // [Q_DIM_FULL]
    pub buf_k: Buffer,             // [KV_DIM_SLIDING] (= 2048, > KV_DIM_FULL=512)
    pub buf_v: Buffer,             // [KV_DIM_SLIDING]
    pub buf_mlp_gate: Buffer,
    pub buf_mlp_up: Buffer,
    pub buf_mlp_act: Buffer,
    pub buf_mlp_down: Buffer,
    pub buf_post_resid: Buffer,
    pub buf_logits: Buffer,

    pub timing: BTreeMap<String, TelemetryValue>,
    _phantom: PhantomData<C>,
}

impl<C: ModelConfig> FusedGemma4Dense<C> {
    pub fn new(
        model: Arc<Model>,
        _num_active_experts: usize,
        _expert_cache_count: usize,
    ) -> Result<Self, MoEError> {
        C::validate_config(&model.config).map_err(MoEError::Config)?;

        const GEMMA4_DENSE_SHADERS: &str = include_str!("shaders.metal");
        let ctx = MetalContext::init_with_shaders(GEMMA4_DENSE_SHADERS)?;

        let weight_buffer = WeightBuffer::new(&ctx.device, &model.weight_file);

        // KV cache: one buffer per layer; size depends on layer type.
        let max_seq = crate::constants::MAX_SEQ;
        let mut buf_kv_k = Vec::with_capacity(C::NUM_LAYERS);
        let mut buf_kv_v = Vec::with_capacity(C::NUM_LAYERS);
        for layer in 0..C::NUM_LAYERS {
            let kv_dim = if C::is_full_attn_layer(layer) {
                C::KV_DIM_FULL
            } else {
                C::KV_DIM_SLIDING
            };
            let sz = max_seq * kv_dim * 4;
            let k_buf = metal_buf_shared(&ctx.device, sz);
            unsafe {
                std::ptr::write_bytes(k_buf.contents() as *mut u8, 0, k_buf.length() as usize);
            }
            // K cache and V cache are SEPARATE buffers — even on full-attn
            // (K=V trick). K stores `k_norm(k_proj) + RoPE`; V stores
            // `v_norm(k_proj)` (no RoPE, different normalisation). They are
            // derived from the same raw k_proj output but get different
            // downstream processing per transformers' attention forward.
            let v_buf = metal_buf_shared(&ctx.device, sz);
            unsafe {
                std::ptr::write_bytes(v_buf.contents() as *mut u8, 0, v_buf.length() as usize);
            }
            buf_kv_k.push(k_buf);
            buf_kv_v.push(v_buf);
        }

        // Scratch sized to the larger of the two layer types.
        let max_q_dim = C::Q_DIM_FULL.max(C::Q_DIM_SLIDING);
        let max_kv_dim = C::KV_DIM_SLIDING.max(C::KV_DIM_FULL);

        let buf_hidden     = metal_buf_shared(&ctx.device, C::HIDDEN_DIM * 4);
        let buf_normed     = metal_buf_shared(&ctx.device, C::HIDDEN_DIM * 4);
        let buf_attn_out   = metal_buf_shared(&ctx.device, max_q_dim * 4);
        let buf_q          = metal_buf_shared(&ctx.device, max_q_dim * 4);
        let buf_k          = metal_buf_shared(&ctx.device, max_kv_dim * 4);
        let buf_v          = metal_buf_shared(&ctx.device, max_kv_dim * 4);
        let buf_mlp_gate   = metal_buf_shared(&ctx.device, C::INTERMEDIATE * 4);
        let buf_mlp_up     = metal_buf_shared(&ctx.device, C::INTERMEDIATE * 4);
        let buf_mlp_act    = metal_buf_shared(&ctx.device, C::INTERMEDIATE * 4);
        let buf_mlp_down   = metal_buf_shared(&ctx.device, C::HIDDEN_DIM * 4);
        let buf_post_resid = metal_buf_shared(&ctx.device, C::HIDDEN_DIM * 4);
        let buf_logits     = metal_buf_shared(&ctx.device, C::VOCAB_SIZE * 4);

        eprintln!(
            "[gemma4_dense] init: arch={} hidden={} layers={} (full={}, sliding={}) \
             sliding[{}h×{}d kv×{}] full[{}h×{}d kv×{}] window={} ffn={} vocab={}",
            C::EXPECTED_ARCHITECTURE,
            C::HIDDEN_DIM, C::NUM_LAYERS, C::NUM_FULL_ATTN_LAYERS, C::NUM_SLIDING_LAYERS,
            C::NUM_ATTN_HEADS, C::HEAD_DIM_SLIDING, C::NUM_KV_HEADS_SLIDING,
            C::NUM_ATTN_HEADS, C::HEAD_DIM_FULL, C::NUM_KV_HEADS_FULL,
            C::SLIDING_WINDOW, C::INTERMEDIATE, C::VOCAB_SIZE,
        );

        Ok(FusedGemma4Dense {
            model, ctx, weight_buffer,
            buf_kv_k, buf_kv_v,
            buf_hidden, buf_normed, buf_attn_out,
            buf_q, buf_k, buf_v,
            buf_mlp_gate, buf_mlp_up, buf_mlp_act, buf_mlp_down,
            buf_post_resid,
            buf_logits,
            timing: BTreeMap::new(),
            _phantom: PhantomData,
        })
    }

    fn pipeline(&self, name: &str) -> metal::ComputePipelineState {
        let f = self.ctx.library.get_function(name, None)
            .unwrap_or_else(|e| panic!("kernel '{}' not found: {}", name, e));
        self.ctx.device.new_compute_pipeline_state_with_function(&f)
            .unwrap_or_else(|e| panic!("pipeline '{}' build failed: {:?}", name, e))
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

    fn encode_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        norm_name: &str,
        src: &Buffer, dst: &Buffer,
    ) {
        let wf = &self.model.weight_file;
        let gw = &self.weight_buffer;
        let nw_ptr = wf.get_tensor_ptr(norm_name)
            .unwrap_or_else(|| panic!("missing norm weight: {}", norm_name));
        let nw_off = (nw_ptr as usize - gw.base as usize) as u64;
        let hidden_dim = C::HIDDEN_DIM as u32;
        let pipe = self.ctx.rms_norm_fused_bf16.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(&gw.buf), nw_off);
        enc.set_buffer(2, Some(dst), 0);
        enc.set_bytes(3, 4, &hidden_dim as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    }

    fn encode_matvec(
        &self,
        enc: &ComputeCommandEncoderRef,
        tensor_prefix: &str,
        src: &Buffer, dst: &Buffer,
        out_dim: usize, in_dim: usize,
    ) {
        self.weight_buffer.encode_matvec_into(
            &self.model.weight_file, &self.ctx, enc,
            tensor_prefix, src, 0, dst, 0, out_dim, in_dim,
        );
    }

    fn encode_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize, pos: usize,
    ) {
        let is_full = C::is_full_attn_layer(layer);
        let prefix = format!("language_model.model.layers.{}.self_attn", layer);
        let norm_prefix = format!("language_model.model.layers.{}", layer);

        let (head_dim, q_dim, kv_dim, rotary_dim, rope_theta, num_kv) = if is_full {
            (C::HEAD_DIM_FULL, C::Q_DIM_FULL, C::KV_DIM_FULL,
             C::ROTARY_DIM_FULL, C::ROPE_THETA_FULL as f32, C::NUM_KV_HEADS_FULL)
        } else {
            (C::HEAD_DIM_SLIDING, C::Q_DIM_SLIDING, C::KV_DIM_SLIDING,
             C::ROTARY_DIM_SLIDING, C::ROPE_THETA_SLIDING as f32, C::NUM_KV_HEADS_SLIDING)
        };

        // 1. input_layernorm: buf_hidden → buf_normed (the attention input).
        self.encode_rms_norm(
            enc,
            &format!("{}.input_layernorm.weight", norm_prefix),
            &self.buf_hidden, &self.buf_normed,
        );

        // 2. Q / K projections. V projection happens ONLY on sliding (full
        //    uses K=V).
        self.encode_matvec(enc, &format!("{}.q_proj", prefix),
            &self.buf_normed, &self.buf_q, q_dim, C::HIDDEN_DIM);
        self.encode_matvec(enc, &format!("{}.k_proj", prefix),
            &self.buf_normed, &self.buf_k, kv_dim, C::HIDDEN_DIM);

        // 3. Set up V. Per transformers' Gemma4UnifiedTextAttention.forward:
        //      value_states = v_proj(x) if v_proj else key_states  # raw k_proj
        //      value_states = v_norm(value_states)                  # RMS, no weight
        //    So V is from raw k_proj output (full) or raw v_proj output (sliding),
        //    THEN per-head RMS-normalised with NO weight.
        if is_full {
            // Copy raw k_proj output (buf_k) into buf_v BEFORE k_norm/RoPE.
            self.encode_buf_copy(enc, &self.buf_k, &self.buf_v, kv_dim as u32);
        } else {
            // Sliding: v_proj produces raw V, then v_norm.
            self.encode_matvec(enc, &format!("{}.v_proj", prefix),
                &self.buf_normed, &self.buf_v, kv_dim, C::HIDDEN_DIM);
        }
        // Apply v_norm (per-head RMS norm, no weight).
        self.encode_v_rms_norm(enc, head_dim, num_kv);

        // 4. Q / K norm + RoPE. Different kernel variant for full-attn (h=512).
        self.encode_q_norm_rope(enc, &prefix, head_dim, rotary_dim, rope_theta, pos);
        self.encode_k_norm_rope(enc, &prefix, head_dim, rotary_dim, rope_theta, pos, num_kv);

        // 5. KV append at position `pos`. buf_v already holds v_norm'd V.
        self.encode_kv_append(enc, layer, pos, kv_dim, is_full);

        // 6. SDPA. Gemma 4 uses scaling=1.0 (not 1/sqrt(d) — see
        //    `self.scaling = 1.0` in Gemma4UnifiedTextAttention.__init__).
        //    The k_norm.weight (≈0.12 for sliding, similar for full) acts
        //    as the implicit attention scaling. Using 1/sqrt(d) here would
        //    flatten the softmax.
        let seq_len = (pos + 1) as u32;
        let scale = 1.0f32;
        if is_full {
            self.encode_sdpa_full(enc, layer, seq_len, scale);
        } else {
            let win_start = (pos + 1).saturating_sub(C::SLIDING_WINDOW) as u32;
            self.encode_sdpa_sliding(enc, layer, win_start, seq_len, scale);
        }

        // 7. o_proj — SDPA wrote into buf_q, o_proj produces buf_post_resid.
        self.encode_matvec(enc, &format!("{}.o_proj", prefix),
            &self.buf_q, &self.buf_post_resid,
            C::HIDDEN_DIM, q_dim);

        // 8. post_attention_layernorm in-place on buf_post_resid.
        self.encode_rms_norm(
            enc,
            &format!("{}.post_attention_layernorm.weight", norm_prefix),
            &self.buf_post_resid, &self.buf_normed,
        );
        self.encode_buf_copy(enc, &self.buf_normed, &self.buf_post_resid,
            C::HIDDEN_DIM as u32);

        // 9. Plain residual: buf_hidden += buf_post_resid (no scaling here —
        //    layer_scalar is applied ONCE at end-of-layer on the whole stream).
        self.encode_residual_add(enc);
    }

    fn encode_v_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        head_dim: usize, num_kv: usize,
    ) {
        let kernel_name = if head_dim == 512 {
            "gemma_v_rms_norm_h512"
        } else {
            "gemma_v_rms_norm_h256"
        };
        let pipe = self.pipeline(kernel_name);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_v), 0);
        let head_dim_u = head_dim as u32;
        enc.set_bytes(1, 4, &head_dim_u as *const u32 as *const c_void);
        enc.set_bytes(2, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_kv as u64, 1, 1),
            MTLSize::new(head_dim as u64, 1, 1));
    }

    fn encode_buf_copy(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer, dst: &Buffer,
        count: u32,
    ) {
        let pipe = self.pipeline("gemma_buf_copy");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((count as u64 + 255) / 256), 1, 1),
            MTLSize::new(256, 1, 1));
    }

    fn encode_residual_add(
        &self,
        enc: &ComputeCommandEncoderRef,
    ) {
        let pipe = self.pipeline("gemma_residual_add");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_hidden), 0);
        enc.set_buffer(1, Some(&self.buf_post_resid), 0);
        let count = C::HIDDEN_DIM as u32;
        enc.set_bytes(2, 4, &count as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((C::HIDDEN_DIM as u64 + 255) / 256), 1, 1),
            MTLSize::new(256, 1, 1));
    }

    fn encode_scale_inplace(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize,
    ) {
        // layer_scalar is stored as BF16; decode on CPU and pass as uniform.
        let scale_name = format!("language_model.model.layers.{}.layer_scalar", layer);
        let wf = &self.model.weight_file;
        let scale_f32: f32 = wf.get_tensor_u16(&scale_name)
            .and_then(|t| t.first().copied())
            .map(bf16_to_f32)
            .unwrap_or(1.0);
        let pipe = self.pipeline("gemma_scale_inplace_const");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_hidden), 0);
        enc.set_bytes(1, 4, &scale_f32 as *const f32 as *const c_void);
        let count = C::HIDDEN_DIM as u32;
        enc.set_bytes(2, 4, &count as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((C::HIDDEN_DIM as u64 + 255) / 256), 1, 1),
            MTLSize::new(256, 1, 1));
    }

    fn encode_q_norm_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        prefix: &str,
        head_dim: usize, rotary_dim: usize, rope_theta: f32, pos: usize,
    ) {
        let wf = &self.model.weight_file;
        let gw = &self.weight_buffer;
        let qn_ptr = wf.get_tensor_ptr(&format!("{}.q_norm.weight", prefix))
            .expect("q_norm.weight missing");
        let qn_off = (qn_ptr as usize - gw.base as usize) as u64;
        let kernel_name = if head_dim == 512 { "gemma_q_norm_rope_h512" } else { "gemma_q_norm_rope_safe" };
        let pipe = self.pipeline(kernel_name);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_q), 0);
        enc.set_buffer(1, Some(&gw.buf), qn_off);
        enc.set_buffer(2, Some(&self.buf_attn_out), 0);
        let head_dim_u = head_dim as u32;
        let rotary_dim_u = rotary_dim as u32;
        let pos_u = pos as u32;
        enc.set_bytes(3, 4, &head_dim_u as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &rotary_dim_u as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &rope_theta as *const f32 as *const c_void);
        enc.set_bytes(6, 4, &pos_u as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(C::NUM_ATTN_HEADS as u64, 1, 1),
            MTLSize::new(head_dim as u64, 1, 1));
    }

    fn encode_k_norm_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        prefix: &str,
        head_dim: usize, rotary_dim: usize, rope_theta: f32, pos: usize,
        num_kv: usize,
    ) {
        let wf = &self.model.weight_file;
        let gw = &self.weight_buffer;
        let kn_ptr = wf.get_tensor_ptr(&format!("{}.k_norm.weight", prefix))
            .expect("k_norm.weight missing");
        let kn_off = (kn_ptr as usize - gw.base as usize) as u64;
        let kernel_name = if head_dim == 512 { "gemma_k_norm_rope_h512" } else { "gemma_k_norm_rope_safe" };
        let pipe = self.pipeline(kernel_name);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_k), 0);
        enc.set_buffer(1, Some(&gw.buf), kn_off);
        let head_dim_u = head_dim as u32;
        let rotary_dim_u = rotary_dim as u32;
        let pos_u = pos as u32;
        enc.set_bytes(2, 4, &head_dim_u as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &rotary_dim_u as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &rope_theta as *const f32 as *const c_void);
        enc.set_bytes(5, 4, &pos_u as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_kv as u64, 1, 1),
            MTLSize::new(head_dim as u64, 1, 1));
    }

    fn encode_kv_append(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize, pos: usize,
        kv_dim: usize, _is_full: bool,
    ) {
        let pipe = self.ctx.kv_cache_append.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&self.buf_k), 0);
        enc.set_buffer(1, Some(&self.buf_v), 0);
        enc.set_buffer(2, Some(&self.buf_kv_k[layer]), 0);
        enc.set_buffer(3, Some(&self.buf_kv_v[layer]), 0);
        let pos_u = pos as u32;
        let kv_dim_u = kv_dim as u32;
        enc.set_bytes(4, 4, &pos_u as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &kv_dim_u as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((kv_dim as u64 + 255) / 256), 1, 1),
            MTLSize::new(256, 1, 1));
    }

    fn encode_sdpa_sliding(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize, seq_start: u32, seq_len: u32, scale: f32,
    ) {
        // Sliding uses the standard 256-head_dim kernel (attn_sdpa_sliding).
        let pipe = self.pipeline("attn_sdpa_sliding");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_attn_out), 0);
        enc.set_buffer(1, Some(&self.buf_kv_k[layer]), 0);
        enc.set_buffer(2, Some(&self.buf_kv_v[layer]), 0);
        enc.set_buffer(3, Some(&self.buf_q), 0);
        enc.set_bytes(4, 4, &seq_start as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &seq_len as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &scale as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(C::NUM_ATTN_HEADS as u64, 1, 1),
            MTLSize::new(256, 1, 1));
    }

    fn encode_sdpa_full(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize, seq_len: u32, scale: f32,
    ) {
        // Simple full-attn SDPA: one thread per output element. Bypasses any
        // SIMD-merge bug in the previous SDPA kernel. 16 TGs × 512 threads.
        let pipe = self.pipeline("attn_sdpa_full_h512_simple");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_attn_out), 0);
        enc.set_buffer(1, Some(&self.buf_kv_k[layer]), 0);
        enc.set_buffer(2, Some(&self.buf_kv_v[layer]), 0);
        enc.set_buffer(3, Some(&self.buf_q), 0);
        enc.set_bytes(4, 4, &seq_len as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &scale as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(C::NUM_ATTN_HEADS as u64, 1, 1),
            MTLSize::new(C::HEAD_DIM_FULL as u64, 1, 1));
    }

    fn encode_scaled_residual_add(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize,
    ) {
        let scale_name = format!("language_model.model.layers.{}.layer_scalar", layer);
        let wf = &self.model.weight_file;
        let scale_f32: f32 = wf.get_tensor_u16(&scale_name)
            .and_then(|t| t.first().copied())
            .map(bf16_to_f32)
            .unwrap_or(1.0);
        let pipe = self.pipeline("gemma_scaled_residual_add_const");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_hidden), 0);
        enc.set_buffer(1, Some(&self.buf_post_resid), 0);
        enc.set_bytes(2, 4, &scale_f32 as *const f32 as *const c_void);
        let count = C::HIDDEN_DIM as u32;
        enc.set_bytes(3, 4, &count as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((C::HIDDEN_DIM as u64 + 255) / 256), 1, 1),
            MTLSize::new(256, 1, 1));
    }

    fn encode_mlp(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: usize,
    ) {
        let prefix = format!("language_model.model.layers.{}", layer);
        self.encode_rms_norm(
            enc,
            &format!("{}.pre_feedforward_layernorm.weight", prefix),
            &self.buf_hidden, &self.buf_normed,
        );
        self.encode_matvec(enc, &format!("{}.mlp.gate_proj", prefix),
            &self.buf_normed, &self.buf_mlp_gate,
            C::INTERMEDIATE, C::HIDDEN_DIM);
        self.encode_matvec(enc, &format!("{}.mlp.up_proj", prefix),
            &self.buf_normed, &self.buf_mlp_up,
            C::INTERMEDIATE, C::HIDDEN_DIM);
        // Gemma uses GELU(tanh-approx)(gate) * up — NOT SiLU.
        let pipe = self.pipeline("gemma_geglu_tanh");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_mlp_gate), 0);
        enc.set_buffer(1, Some(&self.buf_mlp_up), 0);
        enc.set_buffer(2, Some(&self.buf_mlp_act), 0);
        let count_u = C::INTERMEDIATE as u32;
        enc.set_bytes(3, 4, &count_u as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((C::INTERMEDIATE as u64 + 255) / 256), 1, 1),
            MTLSize::new(256, 1, 1));
        self.encode_matvec(enc, &format!("{}.mlp.down_proj", prefix),
            &self.buf_mlp_act, &self.buf_mlp_down,
            C::HIDDEN_DIM, C::INTERMEDIATE);
        self.encode_rms_norm(
            enc,
            &format!("{}.post_feedforward_layernorm.weight", prefix),
            &self.buf_mlp_down, &self.buf_post_resid,
        );
        // Plain residual add (no scaling).
        self.encode_residual_add(enc);
        // End-of-layer: multiply the WHOLE residual stream by layer_scalar.
        self.encode_scale_inplace(enc, layer);
    }

    /// Debug hook: copy buf_hidden into buf_logits[slot * HIDDEN_DIM ..]
    /// so Python can read intermediate residual-stream snapshots.
    fn dump_hidden_into_logits(
        &self,
        enc: &ComputeCommandEncoderRef,
        slot: usize,
    ) {
        let pipe = self.pipeline("gemma_buf_copy");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_hidden), 0);
        let off = (slot * C::HIDDEN_DIM * 4) as u64;
        enc.set_buffer(1, Some(&self.buf_logits), off);
        let count = C::HIDDEN_DIM as u32;
        enc.set_bytes(2, 4, &count as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((C::HIDDEN_DIM as u64 + 255) / 256), 1, 1),
            MTLSize::new(256, 1, 1));
    }

    fn encode_final_norm_and_lm_head(
        &self,
        enc: &ComputeCommandEncoderRef,
    ) {
        self.encode_rms_norm(
            enc,
            "language_model.model.norm.weight",
            &self.buf_hidden, &self.buf_normed,
        );
        self.encode_matvec(enc, "language_model.model.embed_tokens",
            &self.buf_normed, &self.buf_logits,
            C::VOCAB_SIZE, C::HIDDEN_DIM);

        if std::env::var("GEMMA4_DENSE_NO_SOFTCAP").is_ok() {
            return;
        }
        let pipe = self.pipeline("logit_softcap_inplace");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&self.buf_logits), 0);
        let vocab = C::VOCAB_SIZE as u32;
        let cap = C::FINAL_LOGIT_SOFTCAP;
        enc.set_bytes(1, 4, &vocab as *const u32 as *const c_void);
        enc.set_bytes(2, 4, &cap as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((C::VOCAB_SIZE as u64 + 255) / 256), 1, 1),
            MTLSize::new(256, 1, 1));
    }
}

// ─── Engine trait ───────────────────────────────────────────────────────────

impl<C: ModelConfig> Engine for FusedGemma4Dense<C> {
    fn engine_pos(&self) -> usize { self.ctx.pos.get() }

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
        self.ctx.ensure_max_seq(pos + n_tokens);

        // Gemma family embed scaling: multiply by sqrt(hidden_dim). Without
        // this, the residual stream enters the first layer with norm ~3.1
        // instead of ~62, and every downstream norm + matvec downstream
        // produces values an order of magnitude too small.
        let embed_scale = (C::HIDDEN_DIM as f32).sqrt();

        for ti in 0..n_tokens {
            {
                let src = &embeddings[ti * hidden_dim..(ti + 1) * hidden_dim];
                unsafe {
                    let dst = self.buf_hidden.contents() as *mut f32;
                    for i in 0..hidden_dim {
                        *dst.add(i) = src[i] * embed_scale;
                    }
                }
            }

            let stop_at: usize = std::env::var("GEMMA4_DENSE_STOP_LAYER")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(C::NUM_LAYERS);
            // Per-layer hidden-state dump for debugging: copies the residual
            // stream into buf_logits[(k+1) * HIDDEN_DIM ..] after each layer
            // and the initial embed into buf_logits[0 .. HIDDEN_DIM]. Skips
            // the final_norm + lm_head + softcap so Python can read the raw
            // hidden states. Layout in buf_logits:
            //   [embed_post_scale, post_layer_0, post_layer_1, ..., post_layer_{stop_at-1}]
            // Total floats written: (stop_at + 1) * HIDDEN_DIM.
            let capture_hidden = std::env::var("GEMMA4_DENSE_CAPTURE_HIDDEN").is_ok();

            autoreleasepool(|| {
                let cb = self.ctx.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();

                if capture_hidden {
                    self.dump_hidden_into_logits(&enc, 0);
                }

                for layer in 0..stop_at.min(C::NUM_LAYERS) {
                    self.encode_attention(&enc, layer, pos);
                    self.encode_mlp(&enc, layer);
                    if capture_hidden {
                        self.dump_hidden_into_logits(&enc, layer + 1);
                    }
                }
                if !capture_hidden {
                    self.encode_final_norm_and_lm_head(&enc);
                }

                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            });

            let logits_slice = &mut logits[ti * vocab_size..(ti + 1) * vocab_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buf_logits.contents() as *const f32,
                    logits_slice.as_mut_ptr(),
                    vocab_size,
                );
            }

            pos += 1;
            self.ctx.pos.set(pos);
        }

        Ok(logits)
    }

    fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot { pos: self.ctx.pos.get(), ..EngineSnapshot::default() }
    }

    fn restore(&mut self, snap: &EngineSnapshot) {
        self.ctx.pos.set(snap.pos);
    }

    fn upload_cache(&self, _cache: &crate::cache::Cache) {}
    fn download_cache(&self, _cache: &mut crate::cache::Cache) {}

    fn embed_lookup(&self, token_ids: &[i64], embeddings: &mut [f32]) {
        let hidden_dim = C::HIDDEN_DIM;
        for (i, &id) in token_ids.iter().enumerate() {
            self.embed_one(
                id as usize,
                &mut embeddings[i * hidden_dim..(i + 1) * hidden_dim],
            );
        }
    }
}
