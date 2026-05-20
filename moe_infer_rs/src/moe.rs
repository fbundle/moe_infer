/// Full MoE forward pass: K experts, weighted combination.
///
/// Port of run_moe_forward / run_moe_forward_fused from main.m:734-948.
use metal::*;
use std::os::fd::RawFd;

use crate::config::ModelConfig;
use crate::constants::MAX_K_FUSED;
use crate::error::MoEError;
use crate::expert;
use crate::kernels;
use crate::metal_context::{metal_buf_shared, MetalContext};
use crate::timer::now_ms;

/// Timing for a full MoE forward pass (K experts).
#[derive(Debug, Default, Clone)]
pub struct MoETiming {
    pub io_ms: f64,
    pub compute_ms: f64,
    pub combine_ms: f64,
    pub total_ms: f64,
    pub io_bytes: usize,
}

/// Run full MoE: K experts sequentially, then weighted combination.
pub fn run_moe_forward(
    ctx: &MetalContext,
    packed_fd: RawFd,
    expert_indices: &[usize],
    expert_weights: &[f32],
    x_buf: &Buffer,
    moe_out_buf: &Buffer,
    config: &ModelConfig,
    use_fast: i32,
) -> Result<MoETiming, MoEError> {
    let t0 = now_ms();
    let k = expert_indices.len();
    let hidden_dim = config.hidden_dim;
    let moe_inter = config.moe_intermediate;

    // Pre-allocate reusable buffers
    let expert_buf = metal_buf_shared(&ctx.device, config.expert_size_4bit);
    let gate_out = metal_buf_shared(&ctx.device, moe_inter * 4);
    let up_out = metal_buf_shared(&ctx.device, moe_inter * 4);
    let act_out = metal_buf_shared(&ctx.device, moe_inter * 4);
    let expert_out = metal_buf_shared(&ctx.device, hidden_dim * 4);

    // Stacked outputs for weighted combination
    let stacked = metal_buf_shared(&ctx.device, k * hidden_dim * 4);

    let mut io_ms = 0.0;
    let mut compute_ms = 0.0;
    let mut io_bytes = 0;

    // Run each expert
    for (ki, &expert_idx) in expert_indices.iter().enumerate() {
        let et = expert::run_expert_forward_fast(
            ctx, packed_fd, expert_idx,
            &expert_buf, x_buf,
            &gate_out, &up_out, &act_out,
            &expert_out, config, use_fast,
        )?;
        io_ms += et.io_ms;
        compute_ms += et.compute_ms;
        io_bytes += et.io_bytes;

        // Copy this expert's output into stacked buffer
        unsafe {
            let src = expert_out.contents() as *const f32;
            let dst = (stacked.contents() as *mut f32).add(ki * hidden_dim);
            std::ptr::copy_nonoverlapping(src, dst, hidden_dim);
        }
    }

    // Upload weights
    let t_combine = now_ms();
    let w_buf = metal_buf_shared(&ctx.device, k * 4);
    unsafe {
        let dst = w_buf.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(expert_weights.as_ptr(), dst, k);
    }

    // Run weighted sum kernel
    kernels::metal_weighted_sum(
        ctx, &stacked, &w_buf, moe_out_buf,
        k as u32, hidden_dim as u32,
    );
    let combine_ms = now_ms() - t_combine;

    Ok(MoETiming {
        io_ms,
        compute_ms,
        combine_ms,
        total_ms: now_ms() - t0,
        io_bytes,
    })
}

/// Pread task for parallel I/O (uses usize for pointer to be Send).
struct PreadTask {
    fd: RawFd,
    dst: usize,  // *mut u8 cast to usize for thread safety
    size: usize,
    offset: i64,
}

unsafe impl Send for PreadTask {}

/// Run fused MoE: parallel I/O + single command buffer for ALL K experts.
///
/// 1. Parallel pread: K threads load all K expert weights simultaneously
/// 2. Single command buffer with phased encoders:
///    Phase 1: gate_proj + up_proj for ALL K experts
///    Phase 2: SwiGLU for ALL K experts
///    Phase 3: down_proj for ALL K experts
///    Phase 4: blit copy + weighted_sum
/// 3. Single commit + wait
pub fn run_moe_forward_fused(
    ctx: &MetalContext,
    packed_fd: RawFd,
    expert_indices: &[usize],
    expert_weights: &[f32],
    x_buf: &Buffer,
    moe_out_buf: &Buffer,
    config: &ModelConfig,
) -> Result<MoETiming, MoEError> {
    let t0 = now_ms();
    let k = expert_indices.len().min(MAX_K_FUSED);
    let hidden_dim = config.hidden_dim;
    let moe_inter = config.moe_intermediate;
    let expert_size = config.expert_size_4bit;
    let layout = &config.expert_layout_4bit;

    // Pre-allocate all buffers upfront
    let expert_bufs: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, expert_size)).collect();
    let gate_outs: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, moe_inter * 4)).collect();
    let up_outs: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, moe_inter * 4)).collect();
    let act_outs: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, moe_inter * 4)).collect();
    let expert_outs: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, hidden_dim * 4)).collect();

    // Parallel I/O: load all K experts concurrently
    let t_io_start = now_ms();
    let tasks: Vec<PreadTask> = expert_indices[..k].iter().enumerate().map(|(ki, &ei)| {
        PreadTask {
            fd: packed_fd,
            dst: expert_bufs[ki].contents() as usize,
            size: expert_size,
            offset: (ei as i64) * (expert_size as i64),
        }
    }).collect();

    // Spawn K threads for parallel pread
    let handles: Vec<std::thread::JoinHandle<isize>> = tasks.into_iter().map(|task| {
        std::thread::spawn(move || {
            unsafe { libc::pread(task.fd, task.dst as *mut std::ffi::c_void, task.size, task.offset) }
        })
    }).collect();

    let mut io_bytes = 0;
    for (ki, handle) in handles.into_iter().enumerate() {
        let nread = handle.join().unwrap();
        if nread != expert_size as isize {
            return Err(MoEError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("fused pread expert {}: got {}, expected {}", expert_indices[ki], nread, expert_size),
            )));
        }
        io_bytes += expert_size;
    }
    let io_ms = now_ms() - t_io_start;

    let hidden = hidden_dim as u32;
    let inter = moe_inter as u32;
    let gs = config.group_size as u32;

    let t_compute_start = now_ms();
    let cmd_buf = ctx.queue.new_command_buffer();

    // Phase 1: gate_proj + up_proj for ALL experts
    {
        let encoder = cmd_buf.new_compute_command_encoder();
        for ki in 0..k {
            kernels::encode_matvec_v3(
                ctx, encoder,
                &expert_bufs[ki], layout.gate_w_off as u64,
                &expert_bufs[ki], layout.gate_s_off as u64,
                &expert_bufs[ki], layout.gate_b_off as u64,
                x_buf, 0, &gate_outs[ki], 0,
                inter, hidden, gs,
            );
            kernels::encode_matvec_v3(
                ctx, encoder,
                &expert_bufs[ki], layout.up_w_off as u64,
                &expert_bufs[ki], layout.up_s_off as u64,
                &expert_bufs[ki], layout.up_b_off as u64,
                x_buf, 0, &up_outs[ki], 0,
                inter, hidden, gs,
            );
        }
        encoder.end_encoding();
    }

    // Phase 2: SwiGLU for ALL experts
    {
        let encoder = cmd_buf.new_compute_command_encoder();
        for ki in 0..k {
            kernels::encode_swiglu(
                ctx, encoder,
                &gate_outs[ki], 0, &up_outs[ki], 0, &act_outs[ki], 0,
                inter,
            );
        }
        encoder.end_encoding();
    }

    // Phase 3: down_proj for ALL experts
    {
        let encoder = cmd_buf.new_compute_command_encoder();
        for ki in 0..k {
            kernels::encode_matvec_v3(
                ctx, encoder,
                &expert_bufs[ki], layout.down_w_off as u64,
                &expert_bufs[ki], layout.down_s_off as u64,
                &expert_bufs[ki], layout.down_b_off as u64,
                &act_outs[ki], 0, &expert_outs[ki], 0,
                hidden, inter, gs,
            );
        }
        encoder.end_encoding();
    }

    // Phase 4: blit copy + weighted sum
    {
        let stacked = metal_buf_shared(&ctx.device, k * hidden_dim * 4);
        let blit = cmd_buf.new_blit_command_encoder();
        for ki in 0..k {
            blit.copy_from_buffer(
                &expert_outs[ki], 0,
                &stacked, (ki * hidden_dim * 4) as u64,
                (hidden_dim * 4) as u64,
            );
        }
        blit.end_encoding();

        let w_buf = metal_buf_shared(&ctx.device, k * 4);
        unsafe {
            let dst = w_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(expert_weights.as_ptr(), dst, k);
        }

        let encoder = cmd_buf.new_compute_command_encoder();
        kernels::encode_weighted_sum(ctx, encoder, &stacked, &w_buf, moe_out_buf, k as u32, hidden);
        encoder.end_encoding();
    }

    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    let compute_ms = now_ms() - t_compute_start;

    Ok(MoETiming {
        io_ms,
        compute_ms,
        combine_ms: 0.0, // combined in compute
        total_ms: now_ms() - t0,
        io_bytes,
    })
}
