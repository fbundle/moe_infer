#!/usr/bin/env python3
"""
Strip a large MoE model down to a few layers + few experts for verification.

Usage:
    python helpers/strip_model.py \
        --input hub/models--mlx-community--Qwen3.6-35B-A3B-4bit \
        --output data/models--mlx-community--Qwen3.6-35B-A3B-4bit-stripped \
        --num-layers 4 --num-experts 4
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def parse_safetensors_header(filepath):
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def load_tensor(filepath, header, data_start, tensor_name):
    """Load a single tensor from a safetensors file."""
    if tensor_name not in header:
        return None, None
    meta = header[tensor_name]
    off = meta["data_offsets"]
    byte_len = off[1] - off[0]
    dtype = meta["dtype"]
    shape = meta["shape"]
    with open(filepath, "rb") as f:
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


def get_expert_dim(model_cfg):
    """Get the expert weight dimensions."""
    hidden = model_cfg["hidden_size"]
    moe_inter = model_cfg["moe_intermediate_size"]
    # 4-bit packed: in_dim/8 U32 elements per row
    gate_up_packed_in = hidden // 8
    down_packed_in = moe_inter // 8
    group_size = 64  # MLX default
    gate_up_groups = hidden // group_size
    down_groups = moe_inter // group_size
    return {
        "gate_proj.weight": (moe_inter, gate_up_packed_in),
        "gate_proj.scales": (moe_inter, gate_up_groups),
        "gate_proj.biases": (moe_inter, gate_up_groups),
        "up_proj.weight": (moe_inter, gate_up_packed_in),
        "up_proj.scales": (moe_inter, gate_up_groups),
        "up_proj.biases": (moe_inter, gate_up_groups),
        "down_proj.weight": (hidden, down_packed_in),
        "down_proj.scales": (hidden, down_groups),
        "down_proj.biases": (hidden, down_groups),
    }


def run(input_dir, output_dir, num_layers, num_experts):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(input_path / "config.json") as f:
        full_config = json.load(f)

    cfg = full_config.get("text_config", full_config)
    full_layers = cfg["num_hidden_layers"]
    full_experts = cfg["num_experts"]
    hidden = cfg["hidden_size"]
    moe_inter = cfg["moe_intermediate_size"]

    print(f"Original: {full_layers} layers, {full_experts} experts, hidden={hidden}")
    print(f"Stripped: {num_layers} layers, {num_experts} experts")

    # Update config
    new_config = dict(full_config)
    if "text_config" in new_config:
        new_config["text_config"] = dict(new_config["text_config"])
        new_config["text_config"]["num_hidden_layers"] = num_layers
        new_config["text_config"]["num_experts"] = num_experts
        new_config["text_config"]["num_experts_per_tok"] = min(
            cfg.get("num_experts_per_tok", 8), num_experts
        )
        # Update layer_types
        interval = new_config["text_config"].get("full_attention_interval", 4)
        new_config["text_config"]["layer_types"] = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]
    else:
        new_config["num_hidden_layers"] = num_layers
        new_config["num_experts"] = num_experts
        new_config["num_experts_per_tok"] = min(
            cfg.get("num_experts_per_tok", 8), num_experts
        )
        interval = new_config.get("full_attention_interval", 4)
        new_config["layer_types"] = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]

    with open(output_path / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)
    print(f"Wrote config with {num_layers} layers, {num_experts} experts")

    # Load safetensors index
    with open(input_path / "model.safetensors.index.json") as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    # Determine which tensors to keep
    expert_dim = get_expert_dim(cfg)
    keep_tensors = []  # (name, filename)

    # lm_head
    for pfx in ["language_model.lm_head.", "lm_head."]:
        for name in sorted(wm.keys()):
            if name.startswith(pfx):
                keep_tensors.append(name)

    # Embedding
    for emb_name in ["language_model.model.embed_tokens.", "model.embed_tokens."]:
        for name in sorted(wm.keys()):
            if name.startswith(emb_name):
                keep_tensors.append(name)

    # Final norm
    for norm_pfx in ["language_model.model.norm.", "model.norm."]:
        for name in sorted(wm.keys()):
            if name.startswith(norm_pfx):
                keep_tensors.append(name)

    # Layer weights
    for layer_idx in range(num_layers):
        layer_pfx = f"language_model.model.layers.{layer_idx}."
        alt_pfx = f"model.layers.{layer_idx}."
        for name in sorted(wm.keys()):
            if name.startswith(layer_pfx) or name.startswith(alt_pfx):
                keep_tensors.append(name)

    print(f"Keeping {len(keep_tensors)} tensors")

    # Group by safetensors file
    by_file = defaultdict(list)
    for name in keep_tensors:
        by_file[wm[name]].append(name)

    # Process each safetensors file
    new_tensors = {}  # name -> (data_bytes, dtype_str, shape_tuple)
    expert_weights_stripped = 0

    for fname in tqdm(sorted(by_file.keys()), desc="Processing shards"):
        fpath = input_path / fname
        header, data_start = parse_safetensors_header(str(fpath))

        for tensor_name in by_file[fname]:
            arr, dtype = load_tensor(str(fpath), header, data_start, tensor_name)

            if arr is None:
                print(f"  WARNING: {tensor_name} not found in {fname}")
                continue

            # Slice tensors whose first dim is num_experts.
            # Only tensors under mlp.gate.* or mlp.switch_mlp.* have this property.
            # Avoid false matches on q_norm.weight etc. where head_dim==256==num_experts.
            is_expert_tensor = (
                ".mlp.gate." in tensor_name
                or ".mlp.switch_mlp." in tensor_name
                or ".mlp.experts." in tensor_name
            )
            if is_expert_tensor and len(arr.shape) >= 1 and arr.shape[0] == full_experts:
                arr = arr[:num_experts]
                expert_weights_stripped += 1

            new_tensors[tensor_name] = (arr.tobytes(), dtype, tuple(arr.shape))

    print(f"Stripped {expert_weights_stripped} expert tensors to {num_experts} experts")

    # Write single safetensors file
    print("Writing stripped model...")
    t0 = time.time()

    # Build safetensors header
    safetensors_header = {}
    offset = 0
    for name in sorted(new_tensors.keys()):
        raw, dtype, shape = new_tensors[name]
        length = len(raw)
        safetensors_header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + length],
        }
        offset += length

    header_json = json.dumps(safetensors_header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    header_len = struct.pack("<Q", len(header_bytes))

    out_file = output_path / "model.safetensors"
    with open(out_file, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        for name in sorted(new_tensors.keys()):
            f.write(new_tensors[name][0])

    size_mb = os.path.getsize(out_file) / 1e6
    elapsed = time.time() - t0
    print(f"Wrote {out_file} ({size_mb:.1f} MB) in {elapsed:.1f}s")

    # Write index
    new_index = {
        "metadata": {"total_size": os.path.getsize(out_file)},
        "weight_map": {name: "model.safetensors" for name in sorted(new_tensors.keys())},
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy tokenizer files
    for fname in ["tokenizer.json", "vocab.json", "tokenizer_config.json"]:
        src = input_path / fname
        if src.exists():
            import shutil
            shutil.copy2(src, output_path / fname)

    print(f"\nDone. Model ready at {output_path}")
    print(f"Total tensors: {len(new_tensors)}")
    total_mb = sum(len(v[0]) for v in new_tensors.values()) / 1e6
    print(f"Total weight size: {total_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Strip MoE model to fewer layers/experts")
    parser.add_argument("--input", type=str, required=True, help="Path to input model dir")
    parser.add_argument("--output", type=str, required=True, help="Path to output model dir")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=4)
    args = parser.parse_args()
    run(args.input, args.output, args.num_layers, args.num_experts)


if __name__ == "__main__":
    main()
