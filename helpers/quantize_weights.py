#!/usr/bin/env python3
"""
quantize_weights.py — Quantize raw BF16 non-expert matmul weights to 4-bit MLX format.

Reads model_weights.bin + model_weights.json, quantizes all 2D BF16 weight tensors
to the packed uint32 + bf16 scales + bf16 biases format the inference engine expects,
and writes a new model_weights.bin + model_weights.json.

Usage:
    python quantize_weights.py [--input DIR] [--output DIR]
"""

import argparse
import json
import os
import struct
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

GROUP_SIZE = 64
ALIGN = 64


def float32_to_bf16_bytes(arr_f32):
    u16 = (arr_f32.ravel().view(np.uint32) >> 16).astype(np.uint16)
    return u16.tobytes()


def bf16_bytes_to_f32(raw_bytes, shape):
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def quantize_4bit_affine(weight_f32):
    """Vectorized 4-bit affine quantization. Returns (packed_uint32, scales, biases)."""
    out_dim, in_dim = weight_f32.shape
    num_groups = in_dim // GROUP_SIZE
    n_packed = in_dim // 8

    scales = np.zeros((out_dim, num_groups), dtype=np.float32)
    biases = np.zeros_like(scales)
    packed = np.zeros(out_dim * n_packed, dtype=np.uint32)

    CHUNK = 64
    for r_start in range(0, out_dim, CHUNK):
        r_end = min(r_start + CHUNK, out_dim)
        chunk = weight_f32[r_start:r_end]
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
        q = q.reshape(r_end - r_start, n_packed, 8)
        for i in range(8):
            packed[r_start * n_packed:r_end * n_packed] |= \
                (q[:, :, i].ravel().astype(np.uint64) << (4 * i)).astype(np.uint32)

    return packed, scales, biases


def main():
    parser = argparse.ArgumentParser(
        description="Quantize raw BF16 non-expert matmul weights to 4-bit MLX format")
    parser.add_argument("--input", type=str, default=".", help="Directory with model_weights.bin/.json")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: same as input, overwrites)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    bin_path = input_dir / "model_weights.bin"
    json_path = input_dir / "model_weights.json"

    with open(json_path) as f:
        manifest = json.load(f)

    tensors = manifest["tensors"]

    # Identify tensors to quantize: 2D BF16 weight tensors
    to_quantize = []
    to_keep = []
    for name, info in tensors.items():
        if len(info["shape"]) == 2 and info["dtype"] == "BF16" and name.endswith(".weight"):
            to_quantize.append(name)
        else:
            to_keep.append(name)

    print(f"Tensors to quantize: {len(to_quantize)}")
    print(f"Tensors to keep as-is: {len(to_keep)}")

    # Compute new layout
    new_plan = []  # (name, offset, raw_bytes)
    offset = 0
    orig_total = 0
    new_total = 0

    for name in to_keep:
        info = tensors[name]
        sz = info["size"]
        orig_total += sz
        if offset % ALIGN != 0:
            offset += ALIGN - (offset % ALIGN)
        new_plan.append((name, offset, None))
        offset += sz
        new_total += sz

    # Plan quantized tensors: each becomes 3 entries (.weight, .scales, .biases)
    # also compute total bytes needed for the quantized data
    quant_bytes = {}
    for name in to_quantize:
        info = tensors[name]
        out_dim, in_dim = info["shape"]
        w_packed = out_dim * (in_dim // 2)  # bytes
        sb_size = out_dim * (in_dim // GROUP_SIZE) * 2  # bytes (bf16)
        total = w_packed + 2 * sb_size
        quant_bytes[name] = (w_packed, sb_size)
        orig_total += info["size"]
        new_total += total

    print(f"\nOriginal size: {orig_total / 1e9:.2f} GB")
    print(f"Quantized size: {new_total / 1e9:.2f} GB ({new_total/orig_total*100:.0f}%)")
    print(f"Savings: {(orig_total - new_total) / 1e9:.2f} GB")

    if args.dry_run:
        return

    # Read original binary data into memory for the tensors we're quantizing
    print("\nReading tensors to quantize...")
    quant_data = {}
    with open(bin_path, 'rb') as f:
        for name in tqdm(to_quantize):
            info = tensors[name]
            f.seek(info["offset"])
            raw = f.read(info["size"])
            quant_data[name] = (raw, info["shape"])

    # Compute final layout
    offset = 0
    new_manifest = {
        "model": manifest.get("model", ""),
        "config": manifest.get("config", {}),
    }
    new_tensors = {}

    # Write keep tensors (same layout)
    for name, keep_offset, _ in new_plan:
        info = tensors[name]
        new_tensors[name] = {
            "offset": keep_offset,
            "size": info["size"],
            "shape": info["shape"],
            "dtype": info["dtype"],
        }
        offset = max(offset, keep_offset + info["size"])

    # Quantize and plan
    print("\nQuantizing...")
    new_bin_path = output_dir / "model_weights.bin"
    out_bin_path = str(new_bin_path) + ".tmp"

    # Pre-compute all quantized data
    quant_results = {}
    for name in tqdm(to_quantize):
        raw, shape = quant_data[name]
        f32 = bf16_bytes_to_f32(raw, shape)
        packed, scales, biases = quantize_4bit_affine(f32)
        w_bytes = packed.tobytes()
        s_bytes = float32_to_bf16_bytes(scales)
        b_bytes = float32_to_bf16_bytes(biases)
        quant_results[name] = (w_bytes, s_bytes, b_bytes, shape)

    # Compute final offsets
    for name in to_quantize:
        w_bytes, s_bytes, b_bytes, shape = quant_results[name]
        out_dim, in_dim = shape
        num_groups = in_dim // GROUP_SIZE

        if offset % ALIGN != 0:
            offset += ALIGN - (offset % ALIGN)
        w_off = offset
        offset += len(w_bytes)

        if offset % ALIGN != 0:
            offset += ALIGN - (offset % ALIGN)
        s_off = offset
        offset += len(s_bytes)

        if offset % ALIGN != 0:
            offset += ALIGN - (offset % ALIGN)
        b_off = offset
        offset += len(b_bytes)

        new_tensors[name] = {
            "offset": w_off, "size": len(w_bytes),
            "shape": [out_dim, in_dim // 8], "dtype": "U32",
        }
        base_name = name[:-len(".weight")]
        new_tensors[base_name + ".scales"] = {
            "offset": s_off, "size": len(s_bytes),
            "shape": [out_dim, num_groups], "dtype": "BF16",
        }
        new_tensors[base_name + ".biases"] = {
            "offset": b_off, "size": len(b_bytes),
            "shape": [out_dim, num_groups], "dtype": "BF16",
        }

    new_manifest["tensors"] = new_tensors
    new_manifest["num_tensors"] = len(new_tensors)

    # Write new binary
    print(f"\nWriting {out_bin_path} ({new_total / 1e9:.2f} GB)...")
    with open(out_bin_path, 'wb') as f:
        # Pre-allocate
        f.seek(offset - 1)
        f.write(b'\x00')

        # Write keep tensors
        with open(bin_path, 'rb') as src:
            for name in tqdm(to_keep):
                info = tensors[name]
                src.seek(info["offset"])
                data = src.read(info["size"])
                f.seek(new_tensors[name]["offset"])
                f.write(data)

        # Write quantized tensors
        for name in tqdm(to_quantize):
            w_bytes, s_bytes, b_bytes, _ = quant_results[name]
            base = name[:-len(".weight")]
            f.seek(new_tensors[name]["offset"])
            f.write(w_bytes)
            f.seek(new_tensors[base + ".scales"]["offset"])
            f.write(s_bytes)
            f.seek(new_tensors[base + ".biases"]["offset"])
            f.write(b_bytes)

    # Rename
    os.rename(out_bin_path, new_bin_path)

    # Write new manifest
    new_json_path = output_dir / "model_weights.json"
    with open(new_json_path, 'w') as f:
        json.dump(new_manifest, f, indent=2)

    actual_size = os.path.getsize(new_bin_path)
    print(f"\nDone: {new_bin_path} ({actual_size / 1e9:.2f} GB)")
    print(f"Manifest: {new_json_path} ({len(new_tensors)} tensors)")


if __name__ == "__main__":
    main()
