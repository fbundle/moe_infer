#!/usr/bin/env python3
"""
Dump intermediate values from MLX-LM for a single token through the first linear attention layer.
Compare CPU reference implementation against MLX to validate the reference, then use it
to debug the Rust CPU path.
"""
import json
import struct
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, "/opt/homebrew/lib/python3.14/site-packages")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_next import (
    ModelArgs, Qwen3NextGatedDeltaNet,
)
from mlx_lm.models.gated_delta import compute_g


def bf16_to_f32(u16):
    return struct.unpack('!f', struct.pack('!I', int(u16) << 16))[0]


def load_tensor_from_safetensors(model_dir, tensor_name):
    """Load a single tensor from a safetensors file."""
    index_path = Path(model_dir) / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)

    if tensor_name not in idx["weight_map"]:
        return None, None
    shard = idx["weight_map"][tensor_name]
    shard_path = Path(model_dir) / shard

    with open(shard_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len

    if tensor_name not in header:
        return None, None

    meta = header[tensor_name]
    off = meta["data_offsets"]
    byte_len = off[1] - off[0]
    dtype = meta["dtype"]
    shape = meta["shape"]

    with open(shard_path, "rb") as f:
        f.seek(data_start + off[0])
        raw = f.read(byte_len)

    if dtype == "BF16":
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    elif dtype == "U32":
        arr = np.frombuffer(raw, dtype=np.uint32).reshape(shape)
    else:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    return arr, dtype


def dequant_4bit_weight(w_packed, scales, biases):
    """Dequantize 4-bit packed weight to float32.
    w_packed: [out_dim, in_dim/8] uint32
    scales/biases: [out_dim, in_dim/group_size] uint16 (bf16)
    Returns: [out_dim, in_dim] float32
    """
    out_dim, packed_cols = w_packed.shape
    group_size = 64
    num_groups = packed_cols * 8 // group_size
    in_dim = num_groups * group_size

    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    for row in range(out_dim):
        for g in range(num_groups):
            scale = bf16_to_f32(int(scales[row, g]))
            bias = bf16_to_f32(int(biases[row, g]))
            base_packed = g * (group_size // 8)
            base_x = g * group_size
            for p in range(group_size // 8):
                packed = int(w_packed[row, base_packed + p])
                x_base = base_x + p * 8
                for n in range(8):
                    nibble = (packed >> (n * 4)) & 0xF
                    result[row, x_base + n] = float(nibble) * scale + bias
    return result


def dequant_matvec_4bit(w_packed, scales, biases, x, group_size=64):
    """CPU reference: 4-bit dequant + matvec."""
    out_dim = w_packed.shape[0]
    num_groups = len(x) // group_size
    out = np.zeros(out_dim, dtype=np.float32)
    for row in range(out_dim):
        acc = 0.0
        packed_per_group = group_size // 8
        for g in range(num_groups):
            scale = bf16_to_f32(int(scales[row, g]))
            bias = bf16_to_f32(int(biases[row, g]))
            base_packed = g * packed_per_group
            base_x = g * group_size
            for p in range(packed_per_group):
                packed = int(w_packed[row, base_packed + p])
                x_base = base_x + p * 8
                for n in range(8):
                    nibble = (packed >> (n * 4)) & 0xF
                    w_val = float(nibble) * scale + bias
                    acc += w_val * x[x_base + n]
        out[row] = acc
    return out


def cpu_conv1d_step(conv_state, new_input, weight_bf16, kernel_size=4):
    """CPU reference conv1d step (matches Rust cpu_conv1d_step)."""
    channels = len(new_input)
    # weight_bf16 shape: [channels, kernel_size, 1] from safetensors
    # Flatten to [channels * kernel_size]
    if weight_bf16.ndim == 3:
        weight_flat = weight_bf16.reshape(-1)
    else:
        weight_flat = weight_bf16

    out = np.zeros(channels, dtype=np.float32)
    for c in range(channels):
        acc = 0.0
        for k in range(kernel_size - 1):
            w = bf16_to_f32(int(weight_flat[c * kernel_size + k]))
            acc += conv_state[k * channels + c] * w
        w = bf16_to_f32(int(weight_flat[c * kernel_size + (kernel_size - 1)]))
        acc += new_input[c] * w
        out[c] = acc
    # SiLU
    out = out / (1.0 + np.exp(-out))
    return out


def cpu_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rms_norm_cpu(x, weight_f32, eps=1e-6):
    """CPU RMS norm with weight."""
    sum_sq = np.sum(x * x)
    inv_rms = 1.0 / np.sqrt(sum_sq / len(x) + eps)
    return x * inv_rms * weight_f32


def rms_norm_bare_cpu(x, eps=1e-6):
    """CPU RMS norm without weight."""
    sum_sq = np.sum(x * x)
    inv_rms = 1.0 / np.sqrt(sum_sq / len(x) + eps)
    return x * inv_rms


def print_tensor(name, arr):
    """Print tensor stats."""
    flat = arr.flatten() if hasattr(arr, 'flatten') else np.array(arr)
    # Handle MX arrays
    if hasattr(flat, 'tolist'):
        flat = np.array(flat)
    rms = np.sqrt(np.mean(flat.astype(np.float64) ** 2))
    print(f"  {name}: RMS={rms:.6f}, first5={flat[:5]}")


def main():
    # Use environment variable for model dir
    model_dir = os.environ.get("MODEL_DIR",
        "/Volumes/Hippopotamus/vault/code/flash-moe/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit-stripped")

    with open(Path(model_dir) / "config.json") as f:
        config = json.load(f)
    tc = config.get("text_config", config)

    hidden_dim = tc["hidden_size"]
    num_k_heads = tc["linear_num_key_heads"]
    num_v_heads = tc["linear_num_value_heads"]
    key_dim = tc["linear_key_head_dim"]
    value_dim = tc["linear_value_head_dim"]
    total_key = num_k_heads * key_dim
    total_value = num_v_heads * value_dim
    qkv_dim = 2 * total_key + total_value  # = key_dim*2 + value_dim
    k_heads_per_v = num_v_heads // num_k_heads
    eps = tc["rms_norm_eps"]
    group_size = 64
    kernel_size = 4

    print(f"Model: {tc['model_type']}, hidden={hidden_dim}")
    print(f"Heads: K={num_k_heads}, V={num_v_heads}, k_dim={key_dim}, v_dim={value_dim}")
    print(f"total_key={total_key}, total_value={total_value}, qkv_dim={qkv_dim}")
    print(f"k_heads_per_v={k_heads_per_v}")

    # ─── Load all weights for layer 0 linear_attn ───
    prefix = "language_model.model.layers.0.linear_attn"
    weights = {}

    for name in [
        "in_proj_qkv.weight", "in_proj_qkv.scales", "in_proj_qkv.biases",
        "in_proj_z.weight", "in_proj_z.scales", "in_proj_z.biases",
        "in_proj_b.weight", "in_proj_b.scales", "in_proj_b.biases",
        "in_proj_a.weight", "in_proj_a.scales", "in_proj_a.biases",
        "conv1d.weight",
        "norm.weight",
        "out_proj.weight", "out_proj.scales", "out_proj.biases",
        "A_log", "dt_bias",
    ]:
        full_name = f"{prefix}.{name}"
        arr, dtype = load_tensor_from_safetensors(model_dir, full_name)
        if arr is not None:
            weights[name] = arr

    # Load input_layernorm
    norm_name = "language_model.model.layers.0.input_layernorm.weight"
    nw, _ = load_tensor_from_safetensors(model_dir, norm_name)
    nw_f32 = np.array([bf16_to_f32(int(v)) for v in nw.flatten()], dtype=np.float32)

    print(f"\nLoaded {len(weights)} weight tensors for linear_attn")

    # ─── Create known test input ───
    # Use the same RNG seed as Rust (or a known one)
    rng = np.random.RandomState(42)
    hidden = rng.randn(hidden_dim).astype(np.float32) * 0.02

    print(f"\nInput: RMS={np.sqrt(np.mean(hidden**2)):.6f}, first5={hidden[:5]}")

    # ─── CPU Reference Path ───
    print("\n=== CPU Reference Path ===")

    # Step 1: Input RMS norm
    normed = rms_norm_cpu(hidden, nw_f32, eps)
    residual = hidden.copy()
    print_tensor("normed", normed)

    # Step 2: Projections
    qkv = dequant_matvec_4bit(weights["in_proj_qkv.weight"], weights["in_proj_qkv.scales"], weights["in_proj_qkv.biases"], normed, group_size)
    z = dequant_matvec_4bit(weights["in_proj_z.weight"], weights["in_proj_z.scales"], weights["in_proj_z.biases"], normed, group_size)
    beta = dequant_matvec_4bit(weights["in_proj_b.weight"], weights["in_proj_b.scales"], weights["in_proj_b.biases"], normed, group_size)
    alpha = dequant_matvec_4bit(weights["in_proj_a.weight"], weights["in_proj_a.scales"], weights["in_proj_a.biases"], normed, group_size)
    print_tensor("qkv", qkv)
    print_tensor("z", z)
    print_tensor("beta", beta)
    print_tensor("alpha", alpha)

    # Step 3: Conv1d
    conv_state = np.zeros(3 * qkv_dim, dtype=np.float32)  # [kernel_size-1, channels] flattened
    conv_out = cpu_conv1d_step(conv_state, qkv, weights["conv1d.weight"], kernel_size)
    print_tensor("conv_out", conv_out)

    lin_q = conv_out[:total_key]
    lin_k = conv_out[total_key:2*total_key]
    lin_v = conv_out[2*total_key:]
    print_tensor("lin_q (after conv1d)", lin_q)
    print_tensor("lin_k (after conv1d)", lin_k)
    print_tensor("lin_v (after conv1d)", lin_v)

    # Step 4: Q/K RMS norm (bare)
    inv_scale = 1.0 / np.sqrt(key_dim)
    q_normed = np.zeros(total_key, dtype=np.float32)
    k_normed = np.zeros(total_key, dtype=np.float32)

    for h in range(num_k_heads):
        qh = lin_q[h * key_dim:(h + 1) * key_dim]
        qh_out = rms_norm_bare_cpu(qh, 1e-6)
        q_normed[h * key_dim:(h + 1) * key_dim] = qh_out * (inv_scale * inv_scale)

    for h in range(num_k_heads):
        kh = lin_k[h * key_dim:(h + 1) * key_dim]
        kh_out = rms_norm_bare_cpu(kh, 1e-6)
        k_normed[h * key_dim:(h + 1) * key_dim] = kh_out * inv_scale

    print_tensor("q_normed (scaled)", q_normed)
    print_tensor("k_normed (scaled)", k_normed)

    # Step 5: A_log / dt_bias
    a_log_raw = weights.get("A_log")
    dt_bias_raw = weights.get("dt_bias")
    a_log = a_log_raw.flatten().astype(np.float32) if a_log_raw is not None else None
    dt_bias = np.array([bf16_to_f32(int(v)) for v in dt_bias_raw.flatten()], dtype=np.float32) if dt_bias_raw is not None else None

    if a_log is not None:
        print(f"  A_log[:8]: {a_log[:8]}")
    if dt_bias is not None:
        print(f"  dt_bias[:8]: {dt_bias[:8]}")

    # Step 6: SSM state update (gated delta)
    ssm_state = np.zeros(num_v_heads * value_dim * key_dim, dtype=np.float32)
    out_values = np.zeros(total_value, dtype=np.float32)

    for vh in range(num_v_heads):
        kh = vh // k_heads_per_v
        a_val = 1.0 if a_log is None else float(a_log[vh])
        dt_b = 0.0 if dt_bias is None else float(dt_bias[vh])
        softplus_val = np.log(1.0 + np.exp(float(alpha[vh]) + dt_b))
        g_decay = np.exp(-np.exp(a_val) * softplus_val)
        beta_gate = cpu_sigmoid(float(beta[vh]))

        s_off = vh * value_dim * key_dim
        ssm = ssm_state[s_off:s_off + value_dim * key_dim].reshape(value_dim, key_dim)
        v_h = lin_v[vh * value_dim:(vh + 1) * value_dim]
        k_h = k_normed[kh * key_dim:(kh + 1) * key_dim]

        # Decay
        ssm *= g_decay
        # For first token, ssm starts at 0, so kv_mem = 0
        # kv_mem
        kv_mem = ssm @ k_h
        delta = (v_h - kv_mem) * beta_gate
        ssm += np.outer(delta, k_h)
        # Output
        q_h = q_normed[kh * key_dim:(kh + 1) * key_dim]
        out_values[vh * value_dim:(vh + 1) * value_dim] = ssm @ q_h

    print_tensor("out_values (SSM output)", out_values)

    # Step 7: Gated RMS norm
    gnw_raw = weights.get("norm.weight")
    gated_out = np.zeros(total_value, dtype=np.float32)

    if gnw_raw is not None:
        gnw_f32 = np.array([bf16_to_f32(int(v)) for v in gnw_raw.flatten()], dtype=np.float32)
        for vh in range(num_v_heads):
            oh = out_values[vh * value_dim:(vh + 1) * value_dim]
            zh = z[vh * value_dim:(vh + 1) * value_dim]
            gh = np.zeros(value_dim, dtype=np.float32)

            # cpu_rms_norm_gated: oh * inv_rms * w * silu(zh)
            sum_sq = np.sum(oh * oh)
            inv_rms = 1.0 / np.sqrt(sum_sq / value_dim + eps)
            for i in range(value_dim):
                w = gnw_f32[i]
                silu_z = zh[i] / (1.0 + np.exp(-zh[i]))
                gh[i] = oh[i] * inv_rms * w * silu_z

            gated_out[vh * value_dim:(vh + 1) * value_dim] = gh
    else:
        gated_out = out_values.copy()

    print_tensor("gated_out", gated_out)
    if gnw_raw is not None:
        print(f"  gnw_f32[:8]: {gnw_f32[:8]}")

    # Step 8: Output projection
    attn_out = dequant_matvec_4bit(
        weights["out_proj.weight"], weights["out_proj.scales"], weights["out_proj.biases"],
        gated_out, group_size
    )
    print_tensor("attn_out (out_proj)", attn_out)

    # Step 9: Residual
    hidden_out = residual + attn_out
    print_tensor("hidden_out (final)", hidden_out)

    # ─── MLX Reference Path ───
    print("\n=== MLX Reference Path ===")
    try:
        # Build MLX model args
        args = ModelArgs(
            model_type=tc["model_type"],
            hidden_size=hidden_dim,
            num_hidden_layers=tc["num_hidden_layers"],
            intermediate_size=tc.get("moe_intermediate_size", 512),
            num_attention_heads=tc["num_attention_heads"],
            linear_num_value_heads=num_v_heads,
            linear_num_key_heads=num_k_heads,
            linear_key_head_dim=key_dim,
            linear_value_head_dim=value_dim,
            linear_conv_kernel_dim=kernel_size,
            num_experts=tc["num_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            decoder_sparse_step=tc.get("decoder_sparse_step", 1),
            shared_expert_intermediate_size=tc["shared_expert_intermediate_size"],
            mlp_only_layers=tc.get("mlp_only_layers", []),
            moe_intermediate_size=tc["moe_intermediate_size"],
            rms_norm_eps=eps,
            vocab_size=tc["vocab_size"],
            num_key_value_heads=tc["num_key_value_heads"],
            rope_theta=10000000.0,
            partial_rotary_factor=0.25,
            max_position_embeddings=tc["max_position_embeddings"],
            head_dim=tc["head_dim"],
            norm_topk_prob=False,
            tie_word_embeddings=False,
            attention_bias=False,
            full_attention_interval=tc.get("full_attention_interval", 4),
        )
        model = Qwen3NextGatedDeltaNet(args)

        # Load weights into MLX model
        # MLX expects: in_proj_qkvz.weight, in_proj_ba.weight, conv1d.weight, norm.weight, out_proj.weight, A_log, dt_bias
        # But safetensors has: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
        # We need to merge in_proj_qkv + in_proj_z → in_proj_qkvz
        # and in_proj_b + in_proj_a → in_proj_ba

        # Dequantize the 4-bit weights
        print("  Dequantizing weights for MLX...")
        in_proj_qkv_f32 = dequant_4bit_weight(
            weights["in_proj_qkv.weight"], weights["in_proj_qkv.scales"], weights["in_proj_qkv.biases"]
        )
        in_proj_z_f32 = dequant_4bit_weight(
            weights["in_proj_z.weight"], weights["in_proj_z.scales"], weights["in_proj_z.biases"]
        )
        in_proj_b_f32 = dequant_4bit_weight(
            weights["in_proj_b.weight"], weights["in_proj_b.scales"], weights["in_proj_b.biases"]
        )
        in_proj_a_f32 = dequant_4bit_weight(
            weights["in_proj_a.weight"], weights["in_proj_a.scales"], weights["in_proj_a.biases"]
        )

        # Merge: in_proj_qkvz = [in_proj_qkv; in_proj_z] stacked vertically
        # in_proj_qkv: [8192, 2048] (total_key*2 + total_value = 2048*2 + 4096 = 8192)
        # in_proj_z: [4096, 2048] (total_value)
        # in_proj_qkvz: [12288, 2048] (8192 + 4096)
        in_proj_qkvz_f32 = np.vstack([in_proj_qkv_f32, in_proj_z_f32])
        # in_proj_ba: [64, 2048] (num_v_heads*2 = 32*2)
        in_proj_ba_f32 = np.vstack([in_proj_b_f32, in_proj_a_f32])

        # conv1d.weight: [8192, 4, 1] in safetensors, MLX expects [8192, 1, 4]
        conv1d_weight_bf16 = weights["conv1d.weight"]
        if conv1d_weight_bf16.ndim == 3:
            # Shape [8192, 4, 1] → transpose to [8192, 1, 4]
            conv1d_weight_bf16 = np.transpose(conv1d_weight_bf16, (0, 2, 1))
        conv1d_f32 = np.array([bf16_to_f32(int(v)) for v in conv1d_weight_bf16.flatten()],
                              dtype=np.float32).reshape(conv1d_weight_bf16.shape)

        # norm.weight: [128] bf16 → f32
        norm_w_f32 = np.array([bf16_to_f32(int(v)) for v in weights["norm.weight"].flatten()],
                              dtype=np.float32)

        # out_proj: dequantize
        out_proj_f32 = dequant_4bit_weight(
            weights["out_proj.weight"], weights["out_proj.scales"], weights["out_proj.biases"]
        )

        # Build MLX-compatible weight dict
        mlx_weights = {
            "in_proj_qkvz.weight": mx.array(in_proj_qkvz_f32).T,  # MLX Linear: [in_features, out_features]
            "in_proj_ba.weight": mx.array(in_proj_ba_f32).T,
            "conv1d.weight": mx.array(conv1d_f32),
            "norm.weight": mx.array(norm_w_f32),
            "out_proj.weight": mx.array(out_proj_f32).T,
            "A_log": mx.array(a_log) if a_log is not None else model.A_log,
            "dt_bias": mx.array(dt_bias) if dt_bias is not None else model.dt_bias,
        }
        model.update(mlx_weights)

        # Run MLX forward
        hidden_mx = mx.array(hidden.reshape(1, 1, -1))
        out_mx = model(hidden_mx, mask=None, cache=None)
        out_mx_np = np.array(out_mx).flatten()

        print_tensor("MLX output", out_mx_np)
        print_tensor("CPU output", hidden_out)

        max_diff = np.max(np.abs(out_mx_np - hidden_out))
        print(f"\n  CPU vs MLX max_diff: {max_diff:.6f}")

        # Also compare intermediates available from MLX
        # Compare normed
        normed_mx = mx.fast.rms_norm(hidden_mx, mx.array(1.0 + nw_f32).reshape(1, 1, -1), eps)
        normed_mx_np = np.array(normed_mx).flatten()
        print_tensor("MLX normed", normed_mx_np)
        print(f"  normed max_diff (CPU vs MLX): {np.max(np.abs(normed_mx_np - normed)):.8f}")

        # MLX qkvz/ba projections
        qkvz_mx_np = np.array(model.in_proj_qkvz(hidden_mx)).flatten()
        ba_mx_np = np.array(model.in_proj_ba(hidden_mx)).flatten()

        # CPU qkvz = concat(qkv, z), CPU ba = concat(beta, alpha)
        cpu_qkvz = np.concatenate([qkv, z])
        cpu_ba = np.concatenate([beta, alpha])

        print(f"  qkvz max_diff (CPU vs MLX): {np.max(np.abs(cpu_qkvz - qkvz_mx_np)):.8f}")
        print(f"  ba max_diff (CPU vs MLX): {np.max(np.abs(cpu_ba - ba_mx_np)):.8f}")

        # Compare q,k,v,z after fix_query_key_value_ordering
        q_mlx, k_mlx, v_mlx, z_mlx, b_mlx, a_mlx = model.fix_query_key_value_ordering(
            model.in_proj_qkvz(hidden_mx), model.in_proj_ba(hidden_mx)
        )
        print_tensor("MLX q (after ordering)", np.array(q_mlx).flatten())
        print_tensor("MLX k (after ordering)", np.array(k_mlx).flatten())
        print_tensor("MLX v (after ordering)", np.array(v_mlx).flatten())

        # CPU q,k,v from qkv projection (qkv = concat(all_q, all_k, all_v))
        # CPU order: [q0(128), ..., q15(128), k0(128), ..., k15(128), v0(128), ..., v31(128)]
        cpu_q = qkv[:total_key]
        cpu_k = qkv[total_key:2*total_key]
        cpu_v = qkv[2*total_key:]

        # But MLX order after fix_query_key_value_ordering interleaves by head group:
        # per head group: [q_h(128), k_h(128), v_{h*k}:v_{h*k+k-1}(k*128)]
        # Then reshape to separate q, k, v tensors
        nk = num_k_heads
        dn = key_dim
        nv_per_nk = k_heads_per_v
        dv = value_dim

        # Reorder CPU qkv to match MLX layout
        cpu_q_reordered = np.zeros(total_key, dtype=np.float32)
        cpu_k_reordered = np.zeros(total_key, dtype=np.float32)
        cpu_v_reordered = np.zeros(total_value, dtype=np.float32)

        for hk in range(nk):
            q_start, q_end = hk * dn, (hk + 1) * dn
            k_start, k_end = total_key + hk * dn, total_key + (hk + 1) * dn
            cpu_q_reordered[hk * dn:(hk+1) * dn] = cpu_q[hk*dn:(hk+1)*dn]
            cpu_k_reordered[hk * dn:(hk+1) * dn] = cpu_k[hk*dn:(hk+1)*dn]
            for vi in range(nv_per_nk):
                vh = hk * nv_per_nk + vi
                v_start = 2 * total_key + vh * dv
                v_end = 2 * total_key + (vh + 1) * dv
                cpu_v_reordered[vh * dv:(vh+1) * dv] = cpu_v[v_start-2*total_key:v_end-2*total_key]

        print(f"  q reorder check - MLX q[-128:] vs CPU q[-128:]:")
        print_tensor("MLX q[-128:]", np.array(q_mlx).flatten()[-128:])
        print_tensor("CPU q[-128:]", cpu_q_reordered[-128:])

        # Check if MLX in_proj_qkvz layout directly matches CPU's concat(qkv, z)
        # If they match, the ordering is the same.
        # Actually, let's check: MLX in_proj_qkvz outputs [2*key_dim + 2*value_dim]
        # = [2048*2 + 4096*2] = [4096 + 8192] = [12288]
        # But CPU qkv + z = 8192 + 4096 = 12288 too
        # So the total size matches
        # The question is whether the rows are in the same order
        if np.max(np.abs(cpu_qkvz - qkvz_mx_np)) > 1e-3:
            print("\n  *** QKVZ ordering differs! Comparing per-section... ***")
            # CPU layout: [q(2048), k(2048), v(4096), z(4096)]
            # MLX layout (interleaved): per head group: [q_h(128), k_h(128), v_hs(256=2*128), z_hs(256)]
            # Total per group: 128+128+256+256 = 768
            # 16 groups * 768 = 12288
            for hk in range(min(4, nk)):
                grp_start = hk * (dn + dn + nv_per_nk*dv + nv_per_nk*dv)
                print(f"  Head group {hk}: MLX q[{grp_start}:{grp_start+8}] = {qkvz_mx_np[grp_start:grp_start+8]}")
                # CPU q for this head: hk * dn
                cpu_q_h = cpu_q[hk*dn:hk*dn+8]
                print(f"  Head group {hk}: CPU q[{hk*dn}:{hk*dn+8}] = {cpu_q_h}")

    except Exception as e:
        print(f"  Error in MLX comparison: {e}")
        import traceback
        traceback.print_exc()

    # ─── Key findings ───
    print("\n=== Summary ===")
    print("Use these reference values to compare against Rust CPU debug output.")
    print("Save final hidden output for comparison:")
    print(f"  hidden_out RMS: {np.sqrt(np.mean(hidden_out**2)):.6f}")
    print(f"  hidden_out first 10: {hidden_out[:10]}")


if __name__ == "__main__":
    main()
