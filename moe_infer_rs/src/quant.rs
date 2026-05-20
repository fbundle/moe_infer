/// CPU reference implementations and bf16/f32 conversion helpers.
///
/// Port of bf16_to_f32, cpu_dequant_matvec_4bit, cpu_swiglu from main.m.

/// Convert bf16 (uint16) to f32.
pub fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

/// Convert f32 to bf16 (uint16).
pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// CPU reference: 4-bit dequantized matrix-vector multiply.
///
/// W_packed: [out_dim, in_dim/8] uint32
/// scales:   [out_dim, num_groups] uint16 (bf16)
/// biases:   [out_dim, num_groups] uint16 (bf16)
/// x:        [in_dim] float
/// out:      [out_dim] float
pub fn cpu_dequant_matvec_4bit(
    w_packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) {
    let num_groups = in_dim / group_size;
    let packed_per_group = group_size / 8;
    let packed_cols = in_dim / 8;

    for row in 0..out_dim {
        let mut acc = 0.0f32;
        let w_row = &w_packed[row * packed_cols..];
        let s_row = &scales[row * num_groups..];
        let b_row = &biases[row * num_groups..];

        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let bias = bf16_to_f32(b_row[g]);

            let base_packed = g * packed_per_group;
            let base_x = g * group_size;

            for p in 0..packed_per_group {
                let packed = w_row[base_packed + p];
                let x_base = base_x + p * 8;

                for n in 0..8 {
                    let nibble = (packed >> (n * 4)) & 0xF;
                    let w_val = (nibble as f32) * scale + bias;
                    acc += w_val * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

/// CPU reference: SwiGLU activation.
/// out[i] = silu(gate[i]) * up[i]
pub fn cpu_swiglu(gate: &[f32], up: &[f32], out: &mut [f32], dim: usize) {
    for i in 0..dim {
        let g = gate[i];
        let silu_g = g / (1.0f32 + (-g).exp());
        out[i] = silu_g * up[i];
    }
}

/// CPU reference: RMS normalization.
pub fn cpu_rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt().recip();
    for i in 0..dim {
        out[i] = x[i] * rms * weight[i];
    }
}

/// CPU reference: weighted sum of expert outputs.
/// out[i] = sum_k(weights[k] * expert_outs[k * dim + i])
pub fn cpu_weighted_sum(expert_outs: &[f32], weights: &[f32], out: &mut [f32], k: usize, dim: usize) {
    out.fill(0.0);
    for ki in 0..k {
        let wk = weights[ki];
        let ek = &expert_outs[ki * dim..];
        for d in 0..dim {
            out[d] += ek[d] * wk;
        }
    }
}

/// CPU reference: full expert forward pass.
/// gate_proj -> up_proj -> SwiGLU -> down_proj
pub fn cpu_expert_forward(
    w_packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    x: &[f32],
    out: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
    group_size: usize,
    expert_layout: &crate::config::ExpertLayout,
) {
    let hidden = hidden_dim;
    let inter = intermediate_dim;
    let gs = group_size;

    // gate_proj
    let mut gate_out = vec![0.0f32; inter];
    let gate_w = &w_packed[expert_layout.gate_w_off / 4..];
    let gate_s = &scales[expert_layout.gate_s_off / 2..];
    let gate_b = &biases[expert_layout.gate_b_off / 2..];
    cpu_dequant_matvec_4bit(gate_w, gate_s, gate_b, x, &mut gate_out, inter, hidden, gs);

    // up_proj
    let mut up_out = vec![0.0f32; inter];
    let up_w = &w_packed[expert_layout.up_w_off / 4..];
    let up_s = &scales[expert_layout.up_s_off / 2..];
    let up_b = &biases[expert_layout.up_b_off / 2..];
    cpu_dequant_matvec_4bit(up_w, up_s, up_b, x, &mut up_out, inter, hidden, gs);

    // SwiGLU
    let mut act_out = vec![0.0f32; inter];
    cpu_swiglu(&gate_out, &up_out, &mut act_out, inter);

    // down_proj
    let down_w = &w_packed[expert_layout.down_w_off / 4..];
    let down_s = &scales[expert_layout.down_s_off / 2..];
    let down_b = &biases[expert_layout.down_b_off / 2..];
    cpu_dequant_matvec_4bit(down_w, down_s, down_b, &act_out, out, hidden, inter, gs);
}
