// GPU layer forward pass — Metal dispatch for MoE experts, attention, delta-net.
// Port of moe_infer_mlx/core_src/layer_forward.h GPU paths (CMD1/CMD2/CMD3).
//
// The CPU-only forward pass lives in cpu_forward.rs.

use crate::types::*;
use crate::weights::OwnedTensorHashTable;
use crate::metal::MetalCtx;
use crate::gpu_ops;
use crate::expert_io::ExpertLRUCache;
use crate::constants::MAX_K;
use crate::kernels::cpu_vec_madd;
use crate::cpu_forward::{
    step_input_norm, step_full_attention, step_linear_attention,
    step_moe_routing, step_cpu_expert, step_shared_expert, step_final_combine,
    CpuForwardScratch,
};
use std::fs::File;
use std::os::unix::fs::FileExt;

// ---- Layer weight cache ----

/// Cached weight pointers per layer — avoids repeated tensor lookups.
#[derive(Debug, Clone)]
pub struct LayerWeightCache {
    // Norms
    pub input_norm_w: Option<usize>,
    pub post_attn_norm_w: Option<usize>,

    // Full attention weights
    pub q_w: Option<usize>, pub q_s: Option<usize>, pub q_b: Option<usize>,
    pub k_w: Option<usize>, pub k_s: Option<usize>, pub k_b: Option<usize>,
    pub v_w: Option<usize>, pub v_s: Option<usize>, pub v_b: Option<usize>,
    pub o_w: Option<usize>, pub o_s: Option<usize>, pub o_b: Option<usize>,
    pub q_norm_w: Option<usize>, pub k_norm_w: Option<usize>,

    // Linear attention weights
    pub qkv_w: Option<usize>, pub qkv_s: Option<usize>, pub qkv_b: Option<usize>,
    pub z_w: Option<usize>,   pub z_s: Option<usize>,   pub z_b: Option<usize>,
    pub b_w: Option<usize>,   pub b_s: Option<usize>,   pub b_b: Option<usize>,
    pub a_w: Option<usize>,   pub a_s: Option<usize>,   pub a_b: Option<usize>,
    pub conv1d_w: Option<usize>,
    pub a_log: Option<usize>,
    pub dt_bias: Option<usize>,
    pub gated_norm_w: Option<usize>,
    pub out_proj_w: Option<usize>, pub out_proj_s: Option<usize>, pub out_proj_b: Option<usize>,

    // MoE weights
    pub gate_w: Option<usize>, pub gate_s: Option<usize>, pub gate_b: Option<usize>,
    pub sg_w: Option<usize>,  pub sg_s: Option<usize>,  pub sg_b: Option<usize>,
    pub su_w: Option<usize>,  pub su_s: Option<usize>,  pub su_b: Option<usize>,
    pub sd_w: Option<usize>,  pub sd_s: Option<usize>,  pub sd_b: Option<usize>,
    pub seg_w: Option<usize>, pub seg_s: Option<usize>, pub seg_b: Option<usize>,
}

impl LayerWeightCache {
    fn empty() -> Self {
        Self {
            input_norm_w: None, post_attn_norm_w: None,
            q_w: None, q_s: None, q_b: None,
            k_w: None, k_s: None, k_b: None,
            v_w: None, v_s: None, v_b: None,
            o_w: None, o_s: None, o_b: None,
            q_norm_w: None, k_norm_w: None,
            qkv_w: None, qkv_s: None, qkv_b: None,
            z_w: None, z_s: None, z_b: None,
            b_w: None, b_s: None, b_b: None,
            a_w: None, a_s: None, a_b: None,
            conv1d_w: None, a_log: None, dt_bias: None, gated_norm_w: None,
            out_proj_w: None, out_proj_s: None, out_proj_b: None,
            gate_w: None, gate_s: None, gate_b: None,
            sg_w: None, sg_s: None, sg_b: None,
            su_w: None, su_s: None, su_b: None,
            sd_w: None, sd_s: None, sd_b: None,
            seg_w: None, seg_s: None, seg_b: None,
        }
    }
}

/// Build per-layer weight pointer cache.
pub unsafe fn build_layer_cache(
    cfg: &ModelConfig,
    ht: &OwnedTensorHashTable,
    wf_data: *const u8,
) -> Vec<LayerWeightCache> {
    let mut cache = Vec::with_capacity(cfg.num_layers as usize);

    for i in 0..cfg.num_layers {
        let is_full = ((i + 1) % cfg.full_attn_interval) == 0;
        let mut lc = LayerWeightCache::empty();

        // Norms
        lc.input_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.input_layernorm.weight", i));
        lc.post_attn_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.post_attention_layernorm.weight", i));

        if is_full {
            lc.q_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.q_proj.weight", i));
            lc.q_s = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.q_proj.scales", i));
            lc.q_b = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.q_proj.biases", i));
            lc.k_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.k_proj.weight", i));
            lc.k_s = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.k_proj.scales", i));
            lc.k_b = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.k_proj.biases", i));
            lc.v_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.v_proj.weight", i));
            lc.v_s = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.v_proj.scales", i));
            lc.v_b = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.v_proj.biases", i));
            lc.o_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.o_proj.weight", i));
            lc.o_s = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.o_proj.scales", i));
            lc.o_b = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.o_proj.biases", i));
            lc.q_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.q_norm.weight", i));
            lc.k_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.k_norm.weight", i));
        } else {
            lc.qkv_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_qkv.weight", i));
            lc.qkv_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_qkv.scales", i));
            lc.qkv_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_qkv.biases", i));
            lc.z_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_z.weight", i));
            lc.z_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_z.scales", i));
            lc.z_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_z.biases", i));
            lc.b_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_b.weight", i));
            lc.b_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_b.scales", i));
            lc.b_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_b.biases", i));
            lc.a_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_a.weight", i));
            lc.a_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_a.scales", i));
            lc.a_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_a.biases", i));
            lc.conv1d_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.conv1d.weight", i));
            lc.a_log = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.A_log", i));
            lc.dt_bias = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.dt_bias", i));
            lc.gated_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.norm.weight", i));
            lc.out_proj_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.out_proj.weight", i));
            lc.out_proj_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.out_proj.scales", i));
            lc.out_proj_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.out_proj.biases", i));
        }

        // MoE weights (same for all layers)
        lc.gate_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.gate.weight", i));
        lc.gate_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.gate.scales", i));
        lc.gate_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.gate.biases", i));
        lc.sg_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.gate_proj.weight", i));
        lc.sg_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.gate_proj.scales", i));
        lc.sg_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.gate_proj.biases", i));
        lc.su_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.up_proj.weight", i));
        lc.su_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.up_proj.scales", i));
        lc.su_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.up_proj.biases", i));
        lc.sd_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.down_proj.weight", i));
        lc.sd_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.down_proj.scales", i));
        lc.sd_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.down_proj.biases", i));
        lc.seg_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert_gate.weight", i));
        lc.seg_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert_gate.scales", i));
        lc.seg_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert_gate.biases", i));

        cache.push(lc);
    }

    println!("[cache] Pre-computed weight pointers for {} layers", cfg.num_layers);
    cache
}

pub unsafe fn ht_offset(ht: &OwnedTensorHashTable, _wf_data: *const u8, name: &str) -> Option<usize> {
    ht.find(name).map(|t| t.offset as usize)
}

// ---- Active expert size ----

pub fn active_expert_size(cfg: &ModelConfig, use_2bit: bool) -> usize {
    if use_2bit { cfg.expert_size_2bit as usize } else { cfg.expert_size_4bit as usize }
}

// ============================================================================
// GPU-accelerated layer forward — CPU attention + GPU experts
// ============================================================================

/// Run one layer: CPU attention + GPU experts (when Metal available).
/// Falls back to CPU experts when Metal is unavailable.
///
/// # Safety
///
/// `wf_data` must point to a valid mmap'd weight file. `lc` offsets must
/// be valid within that mapping. Expert file descriptors must be open.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gpu_layer_forward(
    cfg: &ModelConfig,
    wf_data: *const u8,
    lc: &LayerWeightCache,
    hidden: &mut [f32],
    kv: Option<&mut KVCache>,
    la_state: Option<&mut LinearAttnState>,
    pos: i32,
    packed_fd: Option<&File>,
    use_2bit: bool,
    scratch: &mut CpuForwardScratch,
    metal_ctx: Option<&MetalCtx>,
    mut expert_cache: Option<&mut ExpertLRUCache>,
    layer_idx: usize,
) {
    let hd = cfg.hidden_dim as usize;
    let is_full = kv.is_some();
    let gpu_available = metal_ctx.is_some() && metal_ctx.unwrap().wf_buf.is_some();

    // Helper: offset → pointer (null fallback for required weights)
    let u32p = |o: Option<usize>| o.map_or(std::ptr::null(), |x| wf_data.add(x) as *const u32);
    let u16p = |o: Option<usize>| o.map_or(std::ptr::null(), |x| wf_data.add(x) as *const u16);
    let _f32p = |o: Option<usize>| o.map_or(std::ptr::null(), |x| wf_data.add(x) as *const f32);
    let u16o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const u16);
    let f32o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const f32);
    let u32o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const u32);

    // -- save residual --
    scratch.residual[..hd].copy_from_slice(hidden);

    // -- step 1: input norm --
    step_input_norm(hidden, u16o(lc.input_norm_w), &mut scratch.normed, hd, cfg.rms_norm_eps);

    // -- step 2: attention (CPU) --
    if is_full {
        step_full_attention(
            cfg, wf_data,
            u32p(lc.q_w), u16p(lc.q_s), u16p(lc.q_b),
            u32p(lc.k_w), u16p(lc.k_s), u16p(lc.k_b),
            u32p(lc.v_w), u16p(lc.v_s), u16p(lc.v_b),
            u32p(lc.o_w), u16p(lc.o_s), u16p(lc.o_b),
            u16o(lc.q_norm_w), u16o(lc.k_norm_w),
            &scratch.normed, kv.unwrap(), pos,
            &mut scratch.attn_proj,
            &mut scratch.q_proj_out, &mut scratch.q_buf, &mut scratch.q_gate,
            &mut scratch.k_buf, &mut scratch.v_buf, &mut scratch.attn_out,
        );
    } else {
        step_linear_attention(
            cfg, wf_data,
            u32p(lc.qkv_w), u16p(lc.qkv_s), u16p(lc.qkv_b),
            u32p(lc.z_w), u16p(lc.z_s), u16p(lc.z_b),
            u32p(lc.b_w), u16p(lc.b_s), u16p(lc.b_b),
            u32p(lc.a_w), u16p(lc.a_s), u16p(lc.a_b),
            u16o(lc.conv1d_w), f32o(lc.a_log), u16o(lc.dt_bias),
            u16o(lc.gated_norm_w),
            u32p(lc.out_proj_w), u16p(lc.out_proj_s), u16p(lc.out_proj_b),
            &scratch.normed, la_state.unwrap(),
            &mut scratch.attn_proj,
            &mut scratch.qkv_out, &mut scratch.z_out,
            &mut scratch.beta_out, &mut scratch.alpha_out,
            &mut scratch.conv_out, &mut scratch.out_values, &mut scratch.gated_out,
        );
    }

    // -- apply residual: hidden = residual + attn_proj --
    for i in 0..hd {
        hidden[i] = scratch.residual[i] + scratch.attn_proj[i];
    }

    // h_mid snapshot for final combine
    scratch.h_mid[..hd].copy_from_slice(hidden);

    // -- step 1b: post-attention norm --
    step_input_norm(hidden, u16o(lc.post_attn_norm_w), &mut scratch.h_post, hd, cfg.rms_norm_eps);

    // -- step 4: MoE routing (CPU) --
    scratch.gate_scores[..cfg.num_experts as usize].fill(0.0);
    let routing = step_moe_routing(
        cfg,
        u32p(lc.gate_w), u16p(lc.gate_s), u16p(lc.gate_b),
        u32o(lc.seg_w), u16o(lc.seg_s), u16o(lc.seg_b),
        &scratch.h_post,
        &mut scratch.gate_scores,
    );
    let actual_k = routing.expert_indices.len().min(MAX_K);

    // -- step 5-6: Expert compute --
    scratch.moe_out[..hd].fill(0.0);
    scratch.shared_out[..hd].fill(0.0);

    if gpu_available && actual_k > 0 {
        // ---- GPU EXPERT PATH ----
        let ctx = metal_ctx.unwrap();
        let wf_buf = ctx.wf_buf.as_ref().unwrap();
        let _wf_gpu_ptr = wf_buf.contents() as *const u8;
        let _layout = if use_2bit { &cfg.layout_2bit } else { &cfg.layout_4bit };
        let esz = active_expert_size(cfg, use_2bit);

        // Load expert data into GPU multi-expert buffers (cache-aware)
        if let Some(fd) = packed_fd {
            let mut cpu_buf = vec![0u8; esz];
            for k in 0..actual_k {
                let expert_idx = routing.expert_indices[k];
                // Try GPU cache first
                let cached = expert_cache.as_mut()
                    .and_then(|c| c.lookup(layer_idx as i32, expert_idx, cfg.num_experts));
                if let Some(cached_buf) = cached {
                    // Copy from cached GPU buffer to multi-expert buffer
                    let src = cached_buf.contents() as *const u8;
                    let dst = ctx.buf_multi_expert_data[k].contents() as *mut u8;
                    std::ptr::copy_nonoverlapping(src, dst, esz);
                } else {
                    // Cache miss — read from file and optionally insert into cache
                    let offset = (expert_idx as usize * esz) as u64;
                    if fd.read_exact_at(&mut cpu_buf, offset).is_ok() {
                        let dst = ctx.buf_multi_expert_data[k].contents() as *mut u8;
                        std::ptr::copy_nonoverlapping(cpu_buf.as_ptr(), dst, esz);
                        // Insert into cache for future use
                        if let Some(c) = expert_cache.as_mut() {
                            let cache_buf = c.insert(layer_idx as i32, expert_idx, cfg.num_experts);
                            let cache_dst = cache_buf.contents() as *mut u8;
                            std::ptr::copy_nonoverlapping(cpu_buf.as_ptr(), cache_dst, esz);
                        }
                    }
                }
            }
        }

        // Copy input hidden state to GPU multi-expert input buffer
        let input_ptr = ctx.buf_multi_expert_input.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(scratch.h_post.as_ptr(), input_ptr, hd);

        // Encode GPU expert forwards
        let cmd = ctx.queue.new_command_buffer();
        for k in 0..actual_k {
            gpu_ops::gpu_encode_expert_forward_slot(cfg, ctx, &cmd, k, use_2bit);
        }
        cmd.commit();
        cmd.wait_until_completed();

        // Read back expert outputs and accumulate
        for k in 0..actual_k {
            let out_ptr = ctx.buf_multi_expert_out[k].contents() as *const f32;
            let expert_out = std::slice::from_raw_parts(out_ptr, hd);
            cpu_vec_madd(&mut scratch.moe_out, expert_out, routing.expert_weights[k], hd);
        }

        // Shared expert on CPU (lightweight)
        scratch.shared_out[..hd].fill(0.0);
        step_shared_expert(
            cfg,
            u32p(lc.sg_w), u16p(lc.sg_s), u16p(lc.sg_b),
            u32p(lc.su_w), u16p(lc.su_s), u16p(lc.su_b),
            u32p(lc.sd_w), u16p(lc.sd_s), u16p(lc.sd_b),
            &scratch.h_post,
            routing.shared_gate_score,
            &mut scratch.shared_out,
        );
    } else {
        // ---- CPU EXPERT PATH ----
        let esz = active_expert_size(cfg, use_2bit);
        if let Some(fd) = packed_fd {
            let mut buf = vec![0u8; esz];
            for k in 0..actual_k {
                let offset = (routing.expert_indices[k] as usize * esz) as u64;
                if fd.read_exact_at(&mut buf, offset).is_ok() {
                    step_cpu_expert(cfg, &buf, &scratch.h_post, &mut scratch.expert_tmp, use_2bit);
                    cpu_vec_madd(&mut scratch.moe_out, &scratch.expert_tmp, routing.expert_weights[k], hd);
                }
            }
        }

        // Shared expert (CPU)
        step_shared_expert(
            cfg,
            u32p(lc.sg_w), u16p(lc.sg_s), u16p(lc.sg_b),
            u32p(lc.su_w), u16p(lc.su_s), u16p(lc.su_b),
            u32p(lc.sd_w), u16p(lc.sd_s), u16p(lc.sd_b),
            &scratch.h_post,
            routing.shared_gate_score,
            &mut scratch.shared_out,
        );
    }

    // -- step 7: final combine --
    step_final_combine(&scratch.h_mid, &scratch.moe_out, &scratch.shared_out, hidden, hd);
}
