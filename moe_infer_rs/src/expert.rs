/// Expert forward pass: gate/up matvecs -> SwiGLU -> down matvec.
///
/// Single expert path (9 separate preads) and optimized path (1 pread + buffer offsets).
/// Port of run_expert_forward / run_expert_forward_fast from main.m:469-632.
use metal::*;
use std::os::fd::RawFd;

use crate::config::ModelConfig;
use crate::error::MoEError;
use crate::kernels;
use crate::metal_context::{metal_buf_pread, metal_buf_shared, MetalContext};
use crate::timer::now_ms;

/// Timing for a single expert forward pass.
#[derive(Debug, Default, Clone)]
pub struct ExpertTiming {
    pub io_ms: f64,
    pub compute_ms: f64,
    pub total_ms: f64,
    pub io_bytes: usize,
}

/// Run a single expert forward pass (original path: 9 separate preads).
/// Returns timing and writes result to out_buf.
pub fn run_expert_forward(
    ctx: &MetalContext,
    packed_fd: RawFd,
    expert_idx: usize,
    x_buf: &Buffer,
    out_buf: &Buffer,
    config: &ModelConfig,
    use_fast: i32,
) -> Result<ExpertTiming, MoEError> {
    let t0 = now_ms();
    let layout = &config.expert_layout_4bit;
    let expert_size = config.expert_size_4bit;
    let expert_offset = (expert_idx as i64) * (expert_size as i64);

    let t_io_start = now_ms();
    let gate_w = metal_buf_pread(&ctx.device, packed_fd, layout.gate_w_size, expert_offset + layout.gate_w_off as i64)?;
    let gate_s = metal_buf_pread(&ctx.device, packed_fd, layout.gate_s_size, expert_offset + layout.gate_s_off as i64)?;
    let gate_b = metal_buf_pread(&ctx.device, packed_fd, layout.gate_b_size, expert_offset + layout.gate_b_off as i64)?;
    let up_w = metal_buf_pread(&ctx.device, packed_fd, layout.up_w_size, expert_offset + layout.up_w_off as i64)?;
    let up_s = metal_buf_pread(&ctx.device, packed_fd, layout.up_s_size, expert_offset + layout.up_s_off as i64)?;
    let up_b = metal_buf_pread(&ctx.device, packed_fd, layout.up_b_size, expert_offset + layout.up_b_off as i64)?;
    let down_w = metal_buf_pread(&ctx.device, packed_fd, layout.down_w_size, expert_offset + layout.down_w_off as i64)?;
    let down_s = metal_buf_pread(&ctx.device, packed_fd, layout.down_s_size, expert_offset + layout.down_s_off as i64)?;
    let down_b = metal_buf_pread(&ctx.device, packed_fd, layout.down_b_size, expert_offset + layout.down_b_off as i64)?;
    let io_ms = now_ms() - t_io_start;

    let hidden = config.hidden_dim as u32;
    let inter = config.moe_intermediate as u32;
    let gs = config.group_size as u32;

    let gate_out = metal_buf_shared(&ctx.device, config.moe_intermediate * 4);
    let up_out = metal_buf_shared(&ctx.device, config.moe_intermediate * 4);
    let act_out = metal_buf_shared(&ctx.device, config.moe_intermediate * 4);

    let t_compute_start = now_ms();
    let cmd_buf = ctx.queue.new_command_buffer();

    // gate_proj: [4096] -> [1024]
    {
        let encoder = cmd_buf.new_compute_command_encoder();
        kernels::encode_matvec_offset(
            ctx, encoder, &gate_w, 0, &gate_s, 0, &gate_b, 0,
            x_buf, 0, &gate_out, 0, inter, hidden, gs, use_fast,
        );
        encoder.end_encoding();
    }

    // up_proj: [4096] -> [1024]
    {
        let encoder = cmd_buf.new_compute_command_encoder();
        kernels::encode_matvec_offset(
            ctx, encoder, &up_w, 0, &up_s, 0, &up_b, 0,
            x_buf, 0, &up_out, 0, inter, hidden, gs, use_fast,
        );
        encoder.end_encoding();
    }

    // SwiGLU
    {
        let encoder = cmd_buf.new_compute_command_encoder();
        kernels::encode_swiglu(ctx, encoder, &gate_out, 0, &up_out, 0, &act_out, 0, inter);
        encoder.end_encoding();
    }

    // down_proj: [1024] -> [4096]
    {
        let encoder = cmd_buf.new_compute_command_encoder();
        kernels::encode_matvec_offset(
            ctx, encoder, &down_w, 0, &down_s, 0, &down_b, 0,
            &act_out, 0, out_buf, 0, hidden, inter, gs, use_fast,
        );
        encoder.end_encoding();
    }

    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    let compute_ms = now_ms() - t_compute_start;

    Ok(ExpertTiming {
        io_ms,
        compute_ms,
        total_ms: now_ms() - t0,
        io_bytes: expert_size,
    })
}

/// Run a single expert forward pass (optimized: 1 pread, buffer offsets).
/// Uses a pre-allocated expert_buf to avoid per-expert allocation overhead.
pub fn run_expert_forward_fast(
    ctx: &MetalContext,
    packed_fd: RawFd,
    expert_idx: usize,
    expert_buf: &Buffer,
    x_buf: &Buffer,
    gate_out: &Buffer,
    up_out: &Buffer,
    act_out: &Buffer,
    out_buf: &Buffer,
    config: &ModelConfig,
    use_fast: i32,
) -> Result<ExpertTiming, MoEError> {
    let t0 = now_ms();
    let layout = &config.expert_layout_4bit;
    let expert_size = config.expert_size_4bit;
    let expert_offset = (expert_idx as i64) * (expert_size as i64);

    // Single pread for the entire expert
    let t_io_start = now_ms();
    let nread = unsafe {
        let ptr = expert_buf.contents() as *mut u8;
        let slice = std::slice::from_raw_parts_mut(ptr, expert_size);
        libc::pread(packed_fd, slice.as_mut_ptr() as *mut std::ffi::c_void, expert_size, expert_offset)
    };
    let io_ms = now_ms() - t_io_start;

    if nread != expert_size as isize {
        return Err(MoEError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("pread expert {}: got {}, expected {}", expert_idx, nread, expert_size),
        )));
    }

    let hidden = config.hidden_dim as u32;
    let inter = config.moe_intermediate as u32;
    let gs = config.group_size as u32;

    let t_compute_start = now_ms();
    let cmd_buf = ctx.queue.new_command_buffer();

    {
        let encoder = cmd_buf.new_compute_command_encoder();
        // gate_proj
        kernels::encode_matvec_offset(
            ctx, encoder,
            expert_buf, layout.gate_w_off as u64,
            expert_buf, layout.gate_s_off as u64,
            expert_buf, layout.gate_b_off as u64,
            x_buf, 0, gate_out, 0,
            inter, hidden, gs, use_fast,
        );
        encoder.end_encoding();
    }

    {
        let encoder = cmd_buf.new_compute_command_encoder();
        // up_proj
        kernels::encode_matvec_offset(
            ctx, encoder,
            expert_buf, layout.up_w_off as u64,
            expert_buf, layout.up_s_off as u64,
            expert_buf, layout.up_b_off as u64,
            x_buf, 0, up_out, 0,
            inter, hidden, gs, use_fast,
        );
        encoder.end_encoding();
    }

    {
        let encoder = cmd_buf.new_compute_command_encoder();
        // SwiGLU
        kernels::encode_swiglu(ctx, encoder, gate_out, 0, up_out, 0, act_out, 0, inter);
        encoder.end_encoding();
    }

    {
        let encoder = cmd_buf.new_compute_command_encoder();
        // down_proj
        kernels::encode_matvec_offset(
            ctx, encoder,
            expert_buf, layout.down_w_off as u64,
            expert_buf, layout.down_s_off as u64,
            expert_buf, layout.down_b_off as u64,
            act_out, 0, out_buf, 0,
            hidden, inter, gs, use_fast,
        );
        encoder.end_encoding();
    }

    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    let compute_ms = now_ms() - t_compute_start;

    Ok(ExpertTiming {
        io_ms,
        compute_ms,
        total_ms: now_ms() - t0,
        io_bytes: expert_size,
    })
}
