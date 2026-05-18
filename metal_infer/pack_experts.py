#!/usr/bin/env python3
"""
pack_experts.py — Quantize BF16 expert weights to 4-bit and pack into per-layer binaries.

Reads raw BF16 expert tensors from HuggingFace safetensors, quantizes them using
MLX-style affine 4-bit quantization (group_size=64, uint32 packed, bf16 scales/biases),
and writes packed per-layer binary files (packed_experts/layer_XX.bin).

Uses vectorized numpy for fast quantization and direct file offset reads to keep
memory low (~1.5GB peak per layer instead of loading entire 5GB shard files).

Usage:
    python pack_experts.py --model ../hub/models--Qwen--Qwen3.5-35B-A3B
"""

import argparse
import json
import os
import struct
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

GROUP_SIZE = 64


def load_hf_config(model_path):
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    return cfg


def compute_expert_layout(hidden_dim, moe_intermediate):
    """Compute 4-bit expert component sizes and offsets.

    Weights are packed as uint32 (4 bytes each), each holding 8 4-bit values.
    Packed bytes = elements / 8 * 4 = elements / 2.
    """
    gate_w = moe_intermediate * hidden_dim // 2
    gate_sb = moe_intermediate * (hidden_dim // GROUP_SIZE) * 2
    up_w = gate_w
    up_sb = gate_sb
    down_w = hidden_dim * moe_intermediate // 2
    down_sb = hidden_dim * (moe_intermediate // GROUP_SIZE) * 2

    gate_w_off = 0
    gate_s_off = gate_w
    gate_b_off = gate_w + gate_sb
    up_w_off = gate_w + 2 * gate_sb
    up_s_off = up_w_off + up_w
    up_b_off = up_s_off + up_sb
    down_w_off = up_b_off + up_sb
    down_s_off = down_w_off + down_w
    down_b_off = down_s_off + down_sb
    expert_size = down_b_off + down_sb

    return {
        "expert_size": expert_size,
        "gate_w_size": gate_w, "gate_s_size": gate_sb, "gate_b_size": gate_sb,
        "up_w_size": up_w, "up_s_size": up_sb, "up_b_size": up_sb,
        "down_w_size": down_w, "down_s_size": down_sb, "down_b_size": down_sb,
        "gate_w_off": gate_w_off, "gate_s_off": gate_s_off, "gate_b_off": gate_b_off,
        "up_w_off": up_w_off, "up_s_off": up_s_off, "up_b_off": up_b_off,
        "down_w_off": down_w_off, "down_s_off": down_s_off, "down_b_off": down_b_off,
    }


def read_tensor(filepath, data_start, tensor_offsets, byte_len, shape, dtype_str):
    """Read a single tensor from a safetensors file using direct offset seek."""
    with open(filepath, 'rb') as f:
        f.seek(data_start + tensor_offsets[0])
        raw = f.read(byte_len)

    if dtype_str == 'BF16':
        arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (arr << 16).view(np.float32).reshape(shape)
    elif dtype_str == 'F32':
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def float32_to_bf16_bytes(arr_f32):
    """Convert float32 numpy array to bf16 bytes."""
    u16 = (arr_f32.ravel().view(np.uint32) >> 16).astype(np.uint16)
    return u16.tobytes()


def quantize_4bit_affine(weight_f32):
    """
    Vectorized 4-bit affine quantization: [out_dim, in_dim] -> packed uint32 + bf16 scales/biases.
    Processes in chunks along out_dim to limit memory.
    """
    out_dim, in_dim = weight_f32.shape
    num_groups = in_dim // GROUP_SIZE
    n_packed = in_dim // 8  # number of uint32 per row

    scales = np.zeros((out_dim, num_groups), dtype=np.float32)
    biases = np.zeros_like(scales)
    packed = np.zeros(out_dim * n_packed, dtype=np.uint32)

    # Process rows in chunks to avoid large intermediate arrays
    CHUNK = 64
    for r_start in range(0, out_dim, CHUNK):
        r_end = min(r_start + CHUNK, out_dim)
        chunk = weight_f32[r_start:r_end]

        # Reshape to [chunk, num_groups, GROUP_SIZE]
        groups = chunk.reshape(r_end - r_start, num_groups, GROUP_SIZE)

        gmin = groups.min(axis=2)
        gmax = groups.max(axis=2)

        s = (gmax - gmin) / 15.0
        s[s < 1e-12] = 1.0
        b = gmin

        scales[r_start:r_end] = s
        biases[r_start:r_end] = b

        q = np.clip(np.round((groups - b[:, :, np.newaxis]) / s[:, :, np.newaxis]),
                    0, 15).astype(np.uint32)

        # Pack 8 values per uint32
        q = q.reshape(r_end - r_start, n_packed, 8)
        for i in range(8):
            packed[r_start * n_packed:r_end * n_packed] |= (q[:, :, i].ravel().astype(np.uint64) << (4 * i)).astype(np.uint32)

    return packed, scales, biases


def build_expert(buf, base, packed, scales, biases,
                 w_off, s_off, b_off, w_size, s_size, b_size):
    """Write one projection's packed weights + scales + biases into the layer buffer."""
    struct.pack_into(f'<{len(packed)}I', buf, base + w_off, *packed)
    buf[base + s_off:base + s_off + s_size] = float32_to_bf16_bytes(scales)
    buf[base + b_off:base + b_off + b_size] = float32_to_bf16_bytes(biases)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize BF16 expert weights to 4-bit and pack into per-layer binaries")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <model>/packed_experts)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Layer range (e.g. '0-4' or '0,5,10')")
    args = parser.parse_args()

    model_path = Path(args.model)
    cfg = load_hf_config(str(model_path))

    hidden_dim = cfg["hidden_size"]
    moe_intermediate = cfg["moe_intermediate_size"]
    num_experts = cfg["num_experts"]
    num_layers = cfg["num_hidden_layers"]

    L = compute_expert_layout(hidden_dim, moe_intermediate)
    expert_sz = L["expert_size"]

    print(f"Model: {model_path}")
    print(f"  hidden={hidden_dim}, moe_inter={moe_intermediate}")
    print(f"  experts={num_experts}, layers={num_layers}")
    print(f"  expert_size={expert_sz:,} bytes (4-bit)")
    layer_mb = num_experts * expert_sz / 1e6
    total_gb = num_layers * layer_mb / 1e3
    print(f"  per-layer: {layer_mb:.0f} MB, total: {total_gb:.1f} GB")

    # Determine layers
    if args.layers is not None:
        layers = []
        for part in args.layers.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-', 1)
                layers.extend(range(int(a), int(b) + 1))
            else:
                layers.append(int(part))
        layers = sorted(set(layers))
    else:
        layers = list(range(num_layers))

    # Output directory
    output_dir = Path(args.output) if args.output else model_path / "packed_experts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load weight map
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        wm = json.load(f)["weight_map"]

    # Map each layer to its safetensors file paths
    layer_files = {}
    for layer in layers:
        gate_name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        down_name = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        layer_files[layer] = {
            "gate_up": {"name": gate_name, "file": wm[gate_name]},
            "down": {"name": down_name, "file": wm[down_name]},
        }

    # Pre-parse headers for all unique safetensors files (header only, not data)
    unique_files = set()
    for lf in layer_files.values():
        for v in lf.values():
            unique_files.add(v["file"])

    print(f"  safetensors files: {len(unique_files)}")
    print()

    file_info = {}
    for fname in unique_files:
        fpath = model_path / fname
        with open(fpath, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len))
            data_start = 8 + header_len
        file_info[fname] = {"header": header, "data_start": data_start, "path": str(fpath)}
    print(f"Parsed headers for {len(file_info)} safetensors files.")


    t_total = time.time()
    total_experts_processed = 0

    # Process each layer
    layer_iter = tqdm(layers, desc="Layers", unit="layer")
    for layer in layer_iter:
        t_layer = time.time()

        # Build output buffer
        layer_buf = bytearray(num_experts * expert_sz)

        # Read gate_up_proj tensor
        gu_info = layer_files[layer]["gate_up"]
        gu_file = file_info[gu_info["file"]]
        gu_meta = gu_file["header"][gu_info["name"]]
        gu_shape = gu_meta["shape"]  # [num_experts, moe_inter*2, hidden]
        gu_bytes = gu_meta["data_offsets"][1] - gu_meta["data_offsets"][0]

        gate_up_f32 = read_tensor(gu_file["path"], gu_file["data_start"],
                                  gu_meta["data_offsets"], gu_bytes,
                                  gu_shape, gu_meta["dtype"])
        gate_weight = gate_up_f32[:, :moe_intermediate, :].copy()
        up_weight = gate_up_f32[:, moe_intermediate:, :].copy()
        del gate_up_f32  # free ~1GB

        # Read down_proj tensor
        dn_info = layer_files[layer]["down"]
        dn_file = file_info[dn_info["file"]]
        dn_meta = dn_file["header"][dn_info["name"]]
        dn_shape = dn_meta["shape"]  # [num_experts, hidden, moe_inter]
        dn_bytes = dn_meta["data_offsets"][1] - dn_meta["data_offsets"][0]

        down_f32 = read_tensor(dn_file["path"], dn_file["data_start"],
                               dn_meta["data_offsets"], dn_bytes,
                               dn_shape, dn_meta["dtype"])

        # Quantize each expert
        for e in range(num_experts):
            base = e * expert_sz

            # gate_proj
            gp, gs, gb = quantize_4bit_affine(gate_weight[e].astype(np.float32))
            build_expert(layer_buf, base, gp, gs, gb,
                         L["gate_w_off"], L["gate_s_off"], L["gate_b_off"],
                         L["gate_w_size"], L["gate_s_size"], L["gate_b_size"])

            # up_proj
            up, us, ub = quantize_4bit_affine(up_weight[e].astype(np.float32))
            build_expert(layer_buf, base, up, us, ub,
                         L["up_w_off"], L["up_s_off"], L["up_b_off"],
                         L["up_w_size"], L["up_s_size"], L["up_b_size"])

            # down_proj
            dp, ds, db = quantize_4bit_affine(down_f32[e].astype(np.float32))
            build_expert(layer_buf, base, dp, ds, db,
                         L["down_w_off"], L["down_s_off"], L["down_b_off"],
                         L["down_w_size"], L["down_s_size"], L["down_b_size"])

            total_experts_processed += 1

        # Free the BF16 tensors
        del gate_weight, up_weight, down_f32

        # Write layer file
        out_path = output_dir / f"layer_{layer:02d}.bin"
        with open(out_path, 'wb') as f:
            f.write(layer_buf)

        elapsed = time.time() - t_layer
        total_experts_processed += num_experts
        layer_iter.set_postfix_str(f"L{layer:02d} {len(layer_buf)/1e6:.0f}MB {elapsed:.1f}s")

    elapsed = time.time() - t_total
    total_mb = len(layers) * num_experts * expert_sz / 1e6
    print(f"\nDone: {len(layers)} layers, {total_mb:.0f} MB total, {elapsed:.1f}s "
          f"({total_mb/elapsed:.0f} MB/s)")


if __name__ == "__main__":
    main()
