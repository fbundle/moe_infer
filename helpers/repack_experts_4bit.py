#!/usr/bin/env python3
"""
repack_experts_4bit.py — Repack MLX pre-quantized 4-bit experts into per-layer binaries.

Reads already-quantized switch_mlp tensors from an MLX-format safetensors model
(3D: [num_experts, out_dim, packed_in_dim]) and writes per-layer binary files
(packed_experts/layer_XX.bin) matching the layout expected by the C inference engine.

Usage:
    python repack_experts_4bit.py --model hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_safetensors_header(filepath):
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


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


def main():
    parser = argparse.ArgumentParser(
        description="Repack MLX 4-bit experts into per-layer binaries"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to MLX-format model directory"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: data/packed_experts)",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Layer range (e.g. '0-4' or '0,5,10')",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    cfg = load_hf_config(str(model_path))

    hidden_dim = cfg["hidden_size"]
    moe_intermediate = cfg["moe_intermediate_size"]
    num_experts = cfg["num_experts"]
    num_layers = cfg["num_hidden_layers"]

    # Expert component sizes (4-bit MLX format)
    # Weights: [out_dim, in_dim/8] U32  → out_dim * (in_dim/8) * 4 bytes
    # Scales:  [out_dim, in_dim/64] BF16 → out_dim * (in_dim/64) * 2 bytes
    GROUP_SIZE = 64
    gate_w_bytes = moe_intermediate * (hidden_dim // 8) * 4    # uint32
    gate_s_bytes = moe_intermediate * (hidden_dim // GROUP_SIZE) * 2  # bf16
    gate_b_bytes = gate_s_bytes
    up_w_bytes = gate_w_bytes
    up_s_bytes = gate_s_bytes
    up_b_bytes = gate_s_bytes
    down_w_bytes = hidden_dim * (moe_intermediate // 8) * 4
    down_s_bytes = hidden_dim * (moe_intermediate // GROUP_SIZE) * 2
    down_b_bytes = down_s_bytes

    expert_size = (gate_w_bytes + gate_s_bytes + gate_b_bytes +
                   up_w_bytes + up_s_bytes + up_b_bytes +
                   down_w_bytes + down_s_bytes + down_b_bytes)

    print(f"Model: {model_path}")
    print(f"  hidden={hidden_dim}, moe_inter={moe_intermediate}")
    print(f"  experts={num_experts}, layers={num_layers}")
    print(f"  expert_size={expert_size:,} bytes (4-bit)")
    layer_mb = num_experts * expert_size / 1e6
    total_gb = num_layers * layer_mb / 1e3
    print(f"  per-layer: {layer_mb:.0f} MB, total: {total_gb:.1f} GB")

    # Determine layers
    if args.layers is not None:
        layers = []
        for part in args.layers.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                layers.extend(range(int(a), int(b) + 1))
            else:
                layers.append(int(part))
        layers = sorted(set(layers))
    else:
        layers = list(range(num_layers))

    output_dir = Path(args.output) if args.output else Path("data/packed_experts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load weight map
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        wm = json.load(f)["weight_map"]

    # Map each layer to its safetensors file paths (MLX naming convention)
    layer_tensors = defaultdict(dict)  # layer -> {component: {name, file}}
    components = ["gate_proj", "up_proj", "down_proj"]
    sub_tensors = ["weight", "scales", "biases"]

    for layer in layers:
        for comp in components:
            for sub in sub_tensors:
                name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{comp}.{sub}"
                if name in wm:
                    layer_tensors[layer][f"{comp}.{sub}"] = {
                        "name": name, "file": wm[name]
                    }

    # Gather unique safetensors files
    unique_files = set()
    for lt in layer_tensors.values():
        for v in lt.values():
            unique_files.add(v["file"])

    print(f"\n  safetensors files: {len(unique_files)}")

    # Parse headers for all unique safetensors files
    file_info = {}
    for fname in tqdm(sorted(unique_files), desc="Parsing headers"):
        fpath = model_path / fname
        header, data_start = parse_safetensors_header(str(fpath))
        file_info[fname] = {"header": header, "data_start": data_start, "path": str(fpath)}

    t_total = time.time()

    # Process each layer
    for layer in tqdm(layers, desc="Layers", unit="layer"):
        t_layer = time.time()

        # Read all 9 tensors for this layer
        layer_data = {}
        for key, info in layer_tensors[layer].items():
            fname = info["file"]
            fi = file_info[fname]
            tensor_meta = fi["header"][info["name"]]
            byte_len = tensor_meta["data_offsets"][1] - tensor_meta["data_offsets"][0]

            with open(fi["path"], "rb") as f:
                f.seek(fi["data_start"] + tensor_meta["data_offsets"][0])
                raw = f.read(byte_len)

            # Parse shape from tensor metadata
            shape = tensor_meta["shape"]
            dtype = tensor_meta["dtype"]

            if dtype == "U32":
                layer_data[key] = np.frombuffer(raw, dtype=np.uint32).reshape(shape)
            elif dtype == "BF16":
                layer_data[key] = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            else:
                layer_data[key] = raw  # shouldn't happen

        # Build per-layer output buffer
        layer_buf = bytearray(num_experts * expert_size)

        for e in range(num_experts):
            base = e * expert_size
            off = 0

            # gate_proj: weight → scales → biases
            gw = layer_data["gate_proj.weight"][e]  # [512, 256] U32
            layer_buf[base + off : base + off + gate_w_bytes] = gw.tobytes()
            off += gate_w_bytes
            gs = layer_data["gate_proj.scales"][e]  # [512, 32] BF16
            layer_buf[base + off : base + off + gate_s_bytes] = gs.tobytes()
            off += gate_s_bytes
            gb = layer_data["gate_proj.biases"][e]  # [512, 32] BF16
            layer_buf[base + off : base + off + gate_b_bytes] = gb.tobytes()
            off += gate_b_bytes

            # up_proj
            uw = layer_data["up_proj.weight"][e]
            layer_buf[base + off : base + off + up_w_bytes] = uw.tobytes()
            off += up_w_bytes
            us = layer_data["up_proj.scales"][e]
            layer_buf[base + off : base + off + up_s_bytes] = us.tobytes()
            off += up_s_bytes
            ub = layer_data["up_proj.biases"][e]
            layer_buf[base + off : base + off + up_b_bytes] = ub.tobytes()
            off += up_b_bytes

            # down_proj
            dw = layer_data["down_proj.weight"][e]
            layer_buf[base + off : base + off + down_w_bytes] = dw.tobytes()
            off += down_w_bytes
            ds = layer_data["down_proj.scales"][e]
            layer_buf[base + off : base + off + down_s_bytes] = ds.tobytes()
            off += down_s_bytes
            db = layer_data["down_proj.biases"][e]
            layer_buf[base + off : base + off + down_b_bytes] = db.tobytes()
            off += down_b_bytes

        # Write layer file
        out_path = output_dir / f"layer_{layer:02d}.bin"
        with open(out_path, "wb") as f:
            f.write(layer_buf)

        elapsed = time.time() - t_layer
        tqdm.write(
            f"  Layer {layer:02d}: {len(layer_buf)/1e6:.0f} MB in {elapsed:.1f}s"
        )

    elapsed = time.time() - t_total
    total_mb = len(layers) * num_experts * expert_size / 1e6
    print(f"\nDone: {len(layers)} layers, {total_mb:.0f} MB, {elapsed:.1f}s "
          f"({total_mb/elapsed:.0f} MB/s)")


if __name__ == "__main__":
    main()
