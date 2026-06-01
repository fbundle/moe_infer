#!/usr/bin/env python3
"""
repack_experts_2bit.py — Requantize 4-bit packed expert files to 2-bit format.

Reads packed_experts/layer_XX.bin files (512 experts x 7,077,888 bytes each)
and writes packed_experts_2bit/layer_XX.bin (512 experts x 3,932,160 bytes each).

4-bit format (per expert, 7,077,888 bytes):
  gate_proj: weights [1024, 512] u32 + scales [1024, 64] bf16 + biases [1024, 64] bf16
  up_proj:   weights [1024, 512] u32 + scales [1024, 64] bf16 + biases [1024, 64] bf16
  down_proj: weights [4096, 128] u32 + scales [4096, 16] bf16 + biases [4096, 16] bf16
  Total: 7,077,888 bytes

2-bit format (per expert, 3,932,160 bytes):
  gate_proj: weights [1024, 256] u32 + scales [1024, 64] bf16 + biases [1024, 64] bf16
  up_proj:   weights [1024, 256] u32 + scales [1024, 64] bf16 + biases [1024, 64] bf16
  down_proj: weights [4096, 64]  u32 + scales [4096, 16] bf16 + biases [4096, 16] bf16
  Total: 3,932,160 bytes  (44.5% reduction)

Requantization per group of 64 values:
  1. Dequantize: f[i] = uint4[i] * scale + bias  (range 0-15 mapped affinely)
  2. Compute optimal 2-bit params: S2 = (max(f) - min(f)) / 3, B2 = min(f)
  3. Quantize:   uint2[i] = clamp(round((f[i] - B2) / S2), 0, 3)
  4. Repack:     16 x 2-bit values per uint32 (vs 8 x 4-bit per uint32)

Scales/biases keep the same shape (group_size=64 preserved), just recomputed values.
Weight arrays halve in size (16 vals/u32 vs 8 vals/u32).

Usage:
    python repack_experts_2bit.py [--model PATH] [--layer N] [--verify]
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path


# ============================================================================
# Model config — loaded from model_weights.json or HuggingFace config.json
# ============================================================================

GROUP_SIZE = 64

def compute_expert_layout(hidden_dim, moe_intermediate, num_experts):
    """Compute 4-bit expert layout from model dimensions."""
    # gate_proj: [moe_intermediate, hidden] — 4-bit weights, scale, bias per group
    gate_w = moe_intermediate * hidden_dim // 8
    gate_sb = moe_intermediate * (hidden_dim // GROUP_SIZE) * 2  # uint16
    up_w = gate_w
    up_sb = gate_sb
    down_w = hidden_dim * moe_intermediate // 8
    down_sb = hidden_dim * (moe_intermediate // GROUP_SIZE) * 2

    layout = {
        'hidden_dim': hidden_dim,
        'moe_intermediate': moe_intermediate,
        'num_experts': num_experts,
        'group_size': GROUP_SIZE,
        'expert_size_4bit': 0,

        'gate_w_off': 0,
        'gate_w_size': gate_w,
        'gate_s_off': gate_w,
        'gate_s_size': gate_sb,
        'gate_b_off': gate_w + gate_sb,
        'gate_b_size': gate_sb,

        'up_w_off': gate_w + 2 * gate_sb,
        'up_w_size': up_w,
        'up_s_off': gate_w + 2 * gate_sb + up_w,
        'up_s_size': up_sb,
        'up_b_off': gate_w + 2 * gate_sb + up_w + up_sb,
        'up_b_size': up_sb,

        'down_w_off': gate_w + 2 * gate_sb + up_w + 2 * up_sb,
        'down_w_size': down_w,
        'down_s_off': gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w,
        'down_s_size': down_sb,
        'down_b_off': gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + down_sb,
        'down_b_size': down_sb,
    }
    layout['expert_size_4bit'] = layout['down_b_off'] + down_sb
    return layout


def load_model_config(model_path):
    """Load model dimensions from model_weights.json or config.json."""
    model_path = Path(model_path)
    mw_json = model_path / "model_weights.json"
    cfg_json = model_path / "config.json"

    if mw_json.exists():
        with open(mw_json) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        if cfg:
            return compute_expert_layout(
                cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_experts"]
            )

    if cfg_json.exists():
        with open(cfg_json) as f:
            cfg = json.load(f)
        return compute_expert_layout(
            cfg.get("hidden_size", 4096),
            cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 1024)),
            cfg.get("num_experts", 512),
        )

    # Fallback to 397B defaults
    print("WARNING: No config found, using Qwen3.5-397B defaults")
    return compute_expert_layout(4096, 1024, 512)


# Import json here (after it's used in load_model_config)
import json

# Build projection descriptors and 2-bit layout from model config
def build_layouts(layout):
    """Build PROJS_4BIT list and PROJS_2BIT_OFFSETS dict from layout."""
    h = layout['hidden_dim']
    mi = layout['moe_intermediate']

    projs_4bit = [
        ("gate", mi, h, layout['gate_w_off'], layout['gate_s_off'], layout['gate_b_off']),
        ("up",   mi, h, layout['up_w_off'],   layout['up_s_off'],   layout['up_b_off']),
        ("down", h,  mi, layout['down_w_off'], layout['down_s_off'], layout['down_b_off']),
    ]

    # 2-bit layout: same scale/bias sizes as 4-bit, weight arrays halved (16 vals/u32)
    gate_w2 = mi * h // 16
    up_w2 = gate_w2
    down_w2 = h * mi // 16
    gate_sb2 = layout['gate_s_size']  # scales/biases same size

    g_w_off_2 = 0
    g_s_off_2 = g_w_off_2 + gate_w2
    g_b_off_2 = g_s_off_2 + gate_sb2
    u_w_off_2 = g_b_off_2 + gate_sb2
    u_s_off_2 = u_w_off_2 + up_w2
    u_b_off_2 = u_s_off_2 + gate_sb2
    d_w_off_2 = u_b_off_2 + gate_sb2
    d_s_off_2 = d_w_off_2 + down_w2
    d_b_off_2 = d_s_off_2 + layout['down_s_size']
    expert_size_2bit = d_b_off_2 + layout['down_s_size']

    projs_2bit_offsets = {
        "gate": (g_w_off_2, g_s_off_2, g_b_off_2),
        "up":   (u_w_off_2, u_s_off_2, u_b_off_2),
        "down": (d_w_off_2, d_s_off_2, d_b_off_2),
    }

    layout['expert_size_2bit'] = expert_size_2bit
    return projs_4bit, projs_2bit_offsets


# ============================================================================
# bf16 <-> f32 conversion helpers
# ============================================================================

def bf16_to_f32(bf16_u16: np.ndarray) -> np.ndarray:
    """Convert array of uint16 (bf16 bit pattern) to float32 via view."""
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32: np.ndarray) -> np.ndarray:
    """Convert float32 array to uint16 (bf16 bit pattern). Truncates (no rounding)."""
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


# ============================================================================
# Unpack 4-bit: extract 8 x 4-bit values from each uint32
# ============================================================================

def unpack_4bit(packed: np.ndarray) -> np.ndarray:
    """
    Unpack 4-bit values from uint32 array.
    Input:  [..., N] uint32, each holding 8 x 4-bit values (LSB first)
    Output: [..., N*8] uint8, values in range [0, 15]
    """
    shape = packed.shape
    flat = packed.ravel()
    n = flat.size

    out = np.empty(n * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = ((flat >> (i * 4)) & 0xF).astype(np.uint8)

    return out.reshape(shape[:-1] + (shape[-1] * 8,))


# ============================================================================
# Unpack 2-bit: extract 16 x 2-bit values from each uint32
# ============================================================================

def unpack_2bit(packed: np.ndarray) -> np.ndarray:
    """
    Unpack 2-bit values from uint32 array.
    Input:  [..., N] uint32, each holding 16 x 2-bit values (LSB first)
    Output: [..., N*16] uint8, values in range [0, 3]
    """
    shape = packed.shape
    flat = packed.ravel()
    n = flat.size

    out = np.empty(n * 16, dtype=np.uint8)
    for i in range(16):
        out[i::16] = ((flat >> (i * 2)) & 0x3).astype(np.uint8)

    return out.reshape(shape[:-1] + (shape[-1] * 16,))


# ============================================================================
# Pack 2-bit: pack 16 x 2-bit values into each uint32
# ============================================================================

def pack_2bit(vals: np.ndarray) -> np.ndarray:
    """
    Pack 2-bit values into uint32 array.
    Input:  [..., M] uint8, values in range [0, 3], M must be multiple of 16
    Output: [..., M/16] uint32, each holding 16 x 2-bit values (LSB first)
    """
    shape = vals.shape
    assert shape[-1] % 16 == 0, f"Last dim {shape[-1]} not divisible by 16"
    n_packed = shape[-1] // 16

    flat = vals.reshape(-1, shape[-1])
    rows = flat.shape[0]

    out = np.zeros((rows, n_packed), dtype=np.uint32)
    for i in range(16):
        out |= flat[:, i::16].astype(np.uint32) << (i * 2)

    return out.reshape(shape[:-1] + (n_packed,))


# ============================================================================
# Requantize one projection: 4-bit -> dequant -> optimal 2-bit
# ============================================================================

def requantize_projection(
    packed_4bit: np.ndarray,   # [out_dim, packed_cols_4] uint32
    scales_bf16: np.ndarray,   # [out_dim, num_groups] uint16
    biases_bf16: np.ndarray,   # [out_dim, num_groups] uint16
    out_dim: int,
    in_dim: int,
) -> tuple:
    """
    Requantize a single projection from 4-bit to 2-bit.

    Returns: (packed_2bit, new_scales_bf16, new_biases_bf16, rmse)
      packed_2bit:     [out_dim, in_dim/16] uint32
      new_scales_bf16: [out_dim, num_groups] uint16 (bf16 bit pattern)
      new_biases_bf16: [out_dim, num_groups] uint16 (bf16 bit pattern)
      rmse:            float -- RMSE of dequantized 2-bit vs dequantized 4-bit
    """
    num_groups = in_dim // GROUP_SIZE

    # 1. Unpack 4-bit -> [out_dim, in_dim] uint8 in [0, 15]
    vals_4bit = unpack_4bit(packed_4bit)  # [out_dim, in_dim]
    assert vals_4bit.shape == (out_dim, in_dim)

    # 2. Dequantize to float32
    #    For row r, group g: f[r, g*64+i] = vals_4bit[r, g*64+i] * scale[r,g] + bias[r,g]
    scales_f32 = bf16_to_f32(scales_bf16)  # [out_dim, num_groups]
    biases_f32 = bf16_to_f32(biases_bf16)  # [out_dim, num_groups]

    # Reshape for broadcasting: [out_dim, num_groups, GROUP_SIZE]
    vals_grouped = vals_4bit.reshape(out_dim, num_groups, GROUP_SIZE).astype(np.float32)
    s = scales_f32[:, :, np.newaxis]  # [out_dim, num_groups, 1]
    b = biases_f32[:, :, np.newaxis]  # [out_dim, num_groups, 1]

    dequant = vals_grouped * s + b  # [out_dim, num_groups, GROUP_SIZE]

    # 3. Compute optimal 2-bit quantization per group
    #    S2 = (max - min) / 3,  B2 = min
    #    uint2 = clamp(round((val - B2) / S2), 0, 3)
    f_min = dequant.min(axis=2, keepdims=True)   # [out_dim, num_groups, 1]
    f_max = dequant.max(axis=2, keepdims=True)   # [out_dim, num_groups, 1]

    s2 = (f_max - f_min) / 3.0
    b2 = f_min

    # Handle degenerate groups where all values are identical (s2 == 0)
    degenerate = (s2 == 0.0)
    s2_safe = np.where(degenerate, 1.0, s2)

    vals_2bit_f = (dequant - b2) / s2_safe
    vals_2bit = np.clip(np.round(vals_2bit_f), 0, 3).astype(np.uint8)

    # 4. Compute reconstruction error (RMSE)
    recon = vals_2bit.astype(np.float32) * s2 + b2
    error = dequant - recon
    rmse = float(np.sqrt(np.mean(error ** 2)))

    # 5. Pack 2-bit values: [out_dim, num_groups, GROUP_SIZE] -> [out_dim, in_dim/16] uint32
    vals_2bit_flat = vals_2bit.reshape(out_dim, in_dim)
    packed_2bit = pack_2bit(vals_2bit_flat)

    # 6. Convert new scales/biases to bf16
    new_scales_bf16 = f32_to_bf16(s2.squeeze(axis=2).astype(np.float32))  # [out_dim, num_groups]
    new_biases_bf16 = f32_to_bf16(b2.squeeze(axis=2).astype(np.float32))  # [out_dim, num_groups]

    return packed_2bit, new_scales_bf16, new_biases_bf16, rmse


# ============================================================================
# Process one expert: read 4-bit blob, requantize all 3 projections
# ============================================================================

def requantize_expert(expert_blob: bytes, layout: dict, projs_4bit: list, projs_2bit_offsets: dict) -> tuple:
    """
    Requantize a single expert from 4-bit to 2-bit.
    """
    assert len(expert_blob) == layout['expert_size_4bit'], \
        f"Expected {layout['expert_size_4bit']} bytes, got {len(expert_blob)}"

    output = bytearray(layout['expert_size_2bit'])
    proj_rmses = {}

    for name, out_dim, in_dim, w_off, s_off, b_off in projs_4bit:
        packed_cols_4 = in_dim // 8   # uint32 columns in 4-bit format
        num_groups = in_dim // GROUP_SIZE

        # Read 4-bit components from blob
        w_end = w_off + out_dim * packed_cols_4 * 4
        s_end = s_off + out_dim * num_groups * 2
        b_end = b_off + out_dim * num_groups * 2

        packed_4bit = np.frombuffer(
            expert_blob[w_off:w_end], dtype=np.uint32
        ).reshape(out_dim, packed_cols_4)
        scales_bf16 = np.frombuffer(
            expert_blob[s_off:s_end], dtype=np.uint16
        ).reshape(out_dim, num_groups)
        biases_bf16 = np.frombuffer(
            expert_blob[b_off:b_end], dtype=np.uint16
        ).reshape(out_dim, num_groups)

        # Requantize
        packed_2bit, new_scales, new_biases, rmse = requantize_projection(
            packed_4bit, scales_bf16, biases_bf16, out_dim, in_dim
        )
        proj_rmses[name] = rmse

        # Write 2-bit data into output blob at the correct offsets
        w_off_2, s_off_2, b_off_2 = projs_2bit_offsets[name]

        w_data = packed_2bit.tobytes()
        s_data = new_scales.tobytes()
        b_data = new_biases.tobytes()

        output[w_off_2 : w_off_2 + len(w_data)] = w_data
        output[s_off_2 : s_off_2 + len(s_data)] = s_data
        output[b_off_2 : b_off_2 + len(b_data)] = b_data

    return bytes(output), proj_rmses


# ============================================================================
# Verify: dequantize both formats and compare
# ============================================================================

def verify_expert(expert_4bit: bytes, expert_2bit: bytes) -> dict:
    """
    Verify 2-bit requantization by dequantizing both and comparing.
    Returns dict of projection name -> max absolute error.
    """
    max_errors = {}

    for name, out_dim, in_dim, w_off_4, s_off_4, b_off_4 in PROJS_4BIT_GLOBAL:
        packed_cols_4 = in_dim // 8
        num_groups = in_dim // GROUP_SIZE

        # Dequantize 4-bit
        w4 = np.frombuffer(
            expert_4bit[w_off_4 : w_off_4 + out_dim * packed_cols_4 * 4],
            dtype=np.uint32).reshape(out_dim, packed_cols_4)
        s4 = np.frombuffer(
            expert_4bit[s_off_4 : s_off_4 + out_dim * num_groups * 2],
            dtype=np.uint16).reshape(out_dim, num_groups)
        b4 = np.frombuffer(
            expert_4bit[b_off_4 : b_off_4 + out_dim * num_groups * 2],
            dtype=np.uint16).reshape(out_dim, num_groups)

        vals4 = unpack_4bit(w4).reshape(out_dim, num_groups, GROUP_SIZE).astype(np.float32)
        sf4 = bf16_to_f32(s4)[:, :, np.newaxis]
        bf4 = bf16_to_f32(b4)[:, :, np.newaxis]
        deq4 = vals4 * sf4 + bf4

        # Dequantize 2-bit
        w_off_2, s_off_2, b_off_2 = projs_2bit_offsets[name]
        packed_cols_2 = in_dim // 16

        w2 = np.frombuffer(
            expert_2bit[w_off_2 : w_off_2 + out_dim * packed_cols_2 * 4],
            dtype=np.uint32).reshape(out_dim, packed_cols_2)
        s2 = np.frombuffer(
            expert_2bit[s_off_2 : s_off_2 + out_dim * num_groups * 2],
            dtype=np.uint16).reshape(out_dim, num_groups)
        b2 = np.frombuffer(
            expert_2bit[b_off_2 : b_off_2 + out_dim * num_groups * 2],
            dtype=np.uint16).reshape(out_dim, num_groups)

        vals2 = unpack_2bit(w2).reshape(out_dim, num_groups, GROUP_SIZE).astype(np.float32)
        sf2 = bf16_to_f32(s2)[:, :, np.newaxis]
        bf2 = bf16_to_f32(b2)[:, :, np.newaxis]
        deq2 = vals2 * sf2 + bf2

        max_errors[name] = float(np.max(np.abs(deq4 - deq2)))

    return max_errors


# ============================================================================
# Main
# ============================================================================

def run(model_path_str, output_dir_str=None, layer=None, verify=False, experts=None):
    model_path = Path(model_path_str)

    # Load model config and build layouts
    LAYOUT = load_model_config(str(model_path))
    projs_4bit, projs_2bit_offsets = build_layouts(LAYOUT)
    # Make available globally for functions that need them
    import builtins
    builtins.LAYOUT = LAYOUT
    builtins.PROJS_4BIT = projs_4bit
    builtins.PROJS_2BIT_OFFSETS = projs_2bit_offsets
    builtins.PROJS_4BIT_GLOBAL = projs_4bit  # for verify_expert

    input_dir = model_path / 'packed_experts'
    output_dir = Path(output_dir_str) if output_dir_str else model_path / 'packed_experts_2bit'

    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    num_experts = experts if experts else LAYOUT['num_experts']

    # Determine layers to process
    if layer is not None:
        layers = [layer]
    else:
        layers = []
        # Scan up to a generous max
        for i in range(200):
            if (input_dir / f'layer_{i:02d}.bin').exists():
                layers.append(i)
        if not layers:
            print(f"ERROR: No layer_XX.bin files found in {input_dir}", file=sys.stderr)
            sys.exit(1)

    esz4 = LAYOUT['expert_size_4bit']
    esz2 = LAYOUT['expert_size_2bit']

    print(f"Model:       {model_path}")
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Layers:      {layers}")
    print(f"Experts:     {num_experts}")
    print(f"4-bit size:  {esz4:,} bytes/expert  "
          f"({num_experts * esz4 / 1e9:.2f} GB/layer)")
    print(f"2-bit size:  {esz2:,} bytes/expert  "
          f"({num_experts * esz2 / 1e9:.2f} GB/layer)")
    print(f"Savings:     {1 - esz2 / esz4:.1%}")
    print()

    total_t0 = time.time()

    for layer_idx in layers:
        input_path = input_dir / f'layer_{layer_idx:02d}.bin'
        output_path = output_dir / f'layer_{layer_idx:02d}.bin'

        expected_size = num_experts * LAYOUT['expert_size_4bit']
        actual_size = input_path.stat().st_size
        if actual_size != expected_size:
            print(f"WARNING: layer_{layer_idx:02d}.bin is {actual_size:,} bytes, "
                  f"expected {expected_size:,} ({num_experts} x {LAYOUT['expert_size_4bit']:,})")
            num_experts_actual = actual_size // LAYOUT['expert_size_4bit']
            if actual_size % LAYOUT['expert_size_4bit'] != 0:
                print(f"ERROR: File size not a multiple of LAYOUT['expert_size_4bit'], skipping",
                      file=sys.stderr)
                continue
            print(f"  Adjusting to {num_experts_actual} experts based on file size")
        else:
            num_experts_actual = num_experts

        print(f"=== Layer {layer_idx:02d} ({num_experts_actual} experts, "
              f"{actual_size / 1e9:.2f} GB -> "
              f"{num_experts_actual * LAYOUT['expert_size_2bit'] / 1e9:.2f} GB) ===")

        layer_t0 = time.time()

        # Per-projection RMSE accumulators
        rmse_accum = {"gate": 0.0, "up": 0.0, "down": 0.0}
        max_error_accum = {"gate": 0.0, "up": 0.0, "down": 0.0}

        # Process experts one at a time to limit memory
        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            for eidx in range(num_experts_actual):
                fin.seek(eidx * LAYOUT['expert_size_4bit'])
                expert_4bit = fin.read(LAYOUT['expert_size_4bit'])
                if len(expert_4bit) != LAYOUT['expert_size_4bit']:
                    print(f"  ERROR: Short read for expert {eidx}: "
                          f"{len(expert_4bit)} bytes", file=sys.stderr)
                    break

                # Requantize
                expert_2bit, proj_rmses = requantize_expert(expert_4bit)
                assert len(expert_2bit) == LAYOUT['expert_size_2bit']

                # Optional verification (first 4 experts per layer)
                if verify and eidx < 4:
                    max_errs = verify_expert(expert_4bit, expert_2bit)
                    for p in ("gate", "up", "down"):
                        max_error_accum[p] = max(max_error_accum[p], max_errs[p])

                # Accumulate RMSE
                for p in ("gate", "up", "down"):
                    rmse_accum[p] += proj_rmses[p]

                # Write 2-bit expert (sequential, no seeking needed)
                fout.write(expert_2bit)

                # Progress every 32 experts
                if (eidx + 1) % 32 == 0 or eidx == num_experts_actual - 1:
                    elapsed = time.time() - layer_t0
                    rate = (eidx + 1) / elapsed
                    eta = (num_experts_actual - eidx - 1) / rate if rate > 0 else 0
                    print(f"  [{eidx+1:3d}/{num_experts_actual}] "
                          f"{elapsed:.1f}s elapsed, {rate:.1f} experts/s, "
                          f"ETA {eta:.0f}s")

        layer_elapsed = time.time() - layer_t0

        # Per-layer stats
        avg_rmse = {p: rmse_accum[p] / num_experts_actual for p in rmse_accum}
        print(f"\n  Layer {layer_idx:02d} done in {layer_elapsed:.1f}s "
              f"({num_experts_actual / layer_elapsed:.1f} experts/s)")
        print(f"  Avg RMSE:  gate={avg_rmse['gate']:.6f}  "
              f"up={avg_rmse['up']:.6f}  down={avg_rmse['down']:.6f}")
        if verify:
            print(f"  Max error: gate={max_error_accum['gate']:.6f}  "
                  f"up={max_error_accum['up']:.6f}  down={max_error_accum['down']:.6f}")

        out_size = output_path.stat().st_size
        print(f"  Output: {output_path} ({out_size / 1e9:.2f} GB)")
        print()

    total_elapsed = time.time() - total_t0
    print(f"Total time: {total_elapsed:.1f}s")
    print()
    print("2-bit expert layout offsets (for C/Metal code):")
    print(f"  #define LAYOUT['expert_size_2bit']  {LAYOUT['expert_size_2bit']}")
    print(f"  #define GATE_W_OFF_2  {GATE_W_OFF_2}")
    print(f"  #define GATE_S_OFF_2  {GATE_S_OFF_2}")
    print(f"  #define GATE_B_OFF_2  {GATE_B_OFF_2}")
    print(f"  #define UP_W_OFF_2    {UP_W_OFF_2}")
    print(f"  #define UP_S_OFF_2    {UP_S_OFF_2}")
    print(f"  #define UP_B_OFF_2    {UP_B_OFF_2}")
    print(f"  #define DOWN_W_OFF_2  {DOWN_W_OFF_2}")
    print(f"  #define DOWN_S_OFF_2  {DOWN_S_OFF_2}")
    print(f"  #define DOWN_B_OFF_2  {DOWN_B_OFF_2}")


def main():
    parser = argparse.ArgumentParser(
        description='Requantize 4-bit packed experts to 2-bit')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model directory (containing packed_experts/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: MODEL/packed_experts_2bit)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Process only this layer. Default: all layers.')
    parser.add_argument('--verify', action='store_true',
                        help='Verify by dequantizing and comparing to 4-bit')
    parser.add_argument('--experts', type=int, default=None,
                        help='Number of experts per layer (default: from config)')
    args = parser.parse_args()
    run(args.model, args.output, args.layer, args.verify, args.experts)

if __name__ == '__main__':
    main()
