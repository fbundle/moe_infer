#!/usr/bin/env python3
"""
compress_experts_lz4.py — Compress packed experts with LZ4 for faster SSD I/O.

Reads packed_experts/layer_XX.bin (concatenated raw expert data) and writes
packed_experts_lz4/layer_XX.bin with per-expert LZ4 blocks and offset header.

Header format:
  [u32 num_experts] [u32 off_0] [u32 off_1] ... [u32 off_{N-1}] [u32 total_size]
  Followed by compressed expert blobs.
  off[i] = byte offset of expert i's compressed blob from header end.
  Expert i decompressed size = expert_size (from config).

Usage:
    python helpers/compress_experts_lz4.py --model-dir data/models--mlx-community--Qwen3.5-35B-A3B-4bit
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

try:
    import lz4.block
except ImportError:
    print("ERROR: pip install lz4", file=sys.stderr)
    sys.exit(1)


def load_config(model_dir):
    """Load num_layers, num_experts, expert_size from config.json + packed_experts."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        raw = json.load(f)
    tc = raw.get("text_config", raw)
    num_layers = tc["num_hidden_layers"]
    num_experts = tc["num_experts"]
    # Infer expert_size from first layer file: file_size / num_experts
    packed_dir = os.path.join(model_dir, "packed_experts")
    first_layer = os.path.join(packed_dir, "layer_00.bin")
    file_size = os.path.getsize(first_layer)
    expert_size = file_size // num_experts
    return num_layers, num_experts, expert_size


def run(model_dir: str, min_ratio: float = 85.0):
    """Called from convert.py. model_dir is the output model directory.

    min_ratio: skip layer if compressed size >= min_ratio % of raw size.
               Default 85 means we require at least 15% savings.
    """
    model_dir = Path(model_dir)
    packed_dir = model_dir / "packed_experts"
    lz4_dir = model_dir / "packed_experts_lz4"

    if not packed_dir.is_dir():
        print(f"ERROR: {packed_dir} not found. Run repack_experts_4bit first.", file=sys.stderr)
        sys.exit(1)

    num_layers, num_experts, expert_size = load_config(model_dir)

    lz4_dir.mkdir(exist_ok=True)

    print(f"Compressing {num_layers} layers, {num_experts} experts each, {expert_size:,} bytes/expert")
    print(f"  Min ratio: {min_ratio:.0f}% (skip layer if larger)")
    print(f"  Source: {packed_dir}")
    print(f"  Dest:   {lz4_dir}")
    print()

    total_raw = 0
    total_lz4 = 0
    skipped = 0

    for layer in range(num_layers):
        src_path = packed_dir / f"layer_{layer:02d}.bin"
        dst_path = lz4_dir / f"layer_{layer:02d}.bin"

        if not src_path.exists():
            print(f"  Layer {layer:02d}: source not found, skipping")
            continue

        with open(src_path, "rb") as f:
            raw_data = f.read()

        total_raw += len(raw_data)

        # Compress each expert independently
        comp_blobs = []
        for e in range(num_experts):
            start = e * expert_size
            end = start + expert_size
            blob = raw_data[start:end]
            comp = lz4.block.compress(blob, mode="high_compression", store_size=False)
            comp_blobs.append(comp)

        total_comp = sum(len(b) for b in comp_blobs)
        file_size = 4 + (num_experts + 1) * 4 + total_comp
        ratio = file_size / max(len(raw_data), 1) * 100

        if ratio >= min_ratio:
            skipped += 1
            print(f"  Layer {layer:02d}: {ratio:.1f}% >= {min_ratio:.0f}%, skipping (will use raw)")
            continue

        # Build offsets and write
        offsets = []
        header_end = 4 + (num_experts + 1) * 4
        running = 0
        for comp in comp_blobs:
            offsets.append(header_end + running)
            running += len(comp)
        offsets.append(header_end + running)

        with open(dst_path, "wb") as f:
            f.write(struct.pack("<I", num_experts))
            for off in offsets:
                f.write(struct.pack("<I", off))
            for blob in comp_blobs:
                f.write(blob)

        total_lz4 += file_size
        print(f"  Layer {layer:02d}: {len(raw_data):,} -> {file_size:,} bytes ({ratio:.1f}%)")

    if total_raw > 0:
        overall = total_lz4 / total_raw * 100
        print(f"\n  Total: {total_raw:,} -> {total_lz4:,} bytes ({overall:.1f}%)")
        if skipped:
            print(f"  Skipped {skipped} layer(s) with ratio >= {min_ratio:.0f}%")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress packed experts with LZ4")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--min-ratio", type=float, default=85.0,
                        help="Skip layer if compressed size >= min_ratio %% of raw (default: 85)")
    args = parser.parse_args()
    run(args.model_dir, min_ratio=args.min_ratio)
