#!/usr/bin/env python3
"""
convert.py — Convert a HuggingFace Qwen3 MoE model to Flash-MoE format.

Runs the full conversion pipeline:
  1. Generate tokenizer.bin from tokenizer.json
  2. Generate model_config.json from config.json
  3. Extract non-expert weights → model_weights.bin + model_weights.json
  4. Repack 4-bit routed experts → packed_experts/

Usage:
    uv run python helpers/convert.py --model path/to/hf-model --output data

    Or step-by-step:
    uv run python helpers/convert.py --model ... --step tokenizer
    uv run python helpers/convert.py --model ... --step config
    uv run python helpers/convert.py --model ... --step weights
    uv run python helpers/convert.py --model ... --step experts
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: tokenizer.bin
# ─────────────────────────────────────────────────────────────────────────────

def convert_tokenizer(model_dir: str, output_dir: str):
    """Export HuggingFace tokenizer.json to compact binary format."""
    tok_path = Path(model_dir) / "tokenizer.json"
    if not tok_path.exists():
        print(f"ERROR: {tok_path} not found", file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join(output_dir, "tokenizer.bin")

    with open(tok_path, "r", encoding="utf-8") as f:
        t = json.load(f)

    model = t["model"]
    vocab = model["vocab"]  # str -> int
    merges = model["merges"]  # list of [str, str]
    added = t["added_tokens"]  # list of {id, content, ...}

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(out_path, "wb") as f:
        f.write(b"BPET")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", len(sorted_vocab)))
        f.write(struct.pack("<I", len(merges)))
        f.write(struct.pack("<I", len(added)))

        for token_str, token_id in sorted_vocab:
            b = token_str.encode("utf-8")
            f.write(struct.pack("<I", token_id))
            f.write(struct.pack("<H", len(b)))
            f.write(b)

        for pair in merges:
            a, b = pair[0], pair[1]
            ab = a.encode("utf-8")
            bb = b.encode("utf-8")
            f.write(struct.pack("<H", len(ab)))
            f.write(ab)
            f.write(struct.pack("<H", len(bb)))
            f.write(bb)

        for tok in added:
            b = tok["content"].encode("utf-8")
            f.write(struct.pack("<I", tok["id"]))
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    sz = os.path.getsize(out_path)
    print(f"  tokenizer.bin: {len(sorted_vocab)} vocab, {len(merges)} merges ({sz / 1024:.0f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: model_config.json
# ─────────────────────────────────────────────────────────────────────────────

GROUP_SIZE = 64
HEAD_DIM = 256
FULL_ATTN_INTERVAL = 4
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
PARTIAL_ROTARY = 0.25


def compute_expert_layout(hidden_dim: int, moe_intermediate: int) -> dict:
    """Compute 4-bit and 2-bit expert component sizes and offsets."""
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
    expert_size_4bit = down_b_off + down_sb

    gate_w2 = moe_intermediate * hidden_dim // 4
    up_w2 = gate_w2
    down_w2 = hidden_dim * moe_intermediate // 4

    gate_w_off_2 = 0
    gate_s_off_2 = gate_w2
    gate_b_off_2 = gate_w2 + gate_sb
    up_w_off_2 = gate_w2 + 2 * gate_sb
    up_s_off_2 = up_w_off_2 + up_w2
    up_b_off_2 = up_s_off_2 + up_sb
    down_w_off_2 = up_b_off_2 + up_sb
    down_s_off_2 = down_w_off_2 + down_w2
    down_b_off_2 = down_s_off_2 + down_sb
    expert_size_2bit = down_b_off_2 + down_sb

    return {
        "expert_size_4bit": expert_size_4bit,
        "expert_size_2bit": expert_size_2bit,
        "layout_4bit": {
            "gate_w_off": gate_w_off, "gate_s_off": gate_s_off, "gate_b_off": gate_b_off,
            "up_w_off": up_w_off, "up_s_off": up_s_off, "up_b_off": up_b_off,
            "down_w_off": down_w_off, "down_s_off": down_s_off, "down_b_off": down_b_off,
            "gate_w_size": gate_w, "gate_s_size": gate_sb, "gate_b_size": gate_sb,
            "up_w_size": up_w, "up_s_size": up_sb, "up_b_size": up_sb,
            "down_w_size": down_w, "down_s_size": down_sb, "down_b_size": down_sb,
        },
        "layout_2bit": {
            "gate_w_off": gate_w_off_2, "gate_s_off": gate_s_off_2, "gate_b_off": gate_b_off_2,
            "up_w_off": up_w_off_2, "up_s_off": up_s_off_2, "up_b_off": up_b_off_2,
            "down_w_off": down_w_off_2, "down_s_off": down_s_off_2, "down_b_off": down_b_off_2,
            "gate_w_size": gate_w2, "gate_s_size": gate_sb, "gate_b_size": gate_sb,
            "up_w_size": up_w2, "up_s_size": up_sb, "up_b_size": up_sb,
            "down_w_size": down_w2, "down_s_size": down_sb, "down_b_size": down_sb,
        },
    }


def convert_config(model_dir: str, output_dir: str):
    """Generate model_config.json from HF config.json."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = json.load(f)

    if "text_config" in cfg:
        cfg = cfg["text_config"]

    hidden_dim = cfg.get("hidden_size", 4096)
    num_layers = cfg.get("num_hidden_layers", 60)
    num_attn_heads = cfg.get("num_attention_heads", 32)
    num_kv_heads = cfg.get("num_key_value_heads", 2)
    vocab_size = cfg.get("vocab_size", 248320)
    num_experts = cfg.get("num_experts", 512)
    num_experts_per_tok = cfg.get("num_experts_per_tok", 10)
    moe_intermediate = cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 1024))
    shared_intermediate = cfg.get("shared_expert_intermediate_size", cfg.get("intermediate_size", 1024))
    linear_num_v_heads = cfg.get("linear_num_value_heads", 64)
    linear_num_k_heads = cfg.get("linear_num_key_heads", 16)

    rotary_dim = int(HEAD_DIM * PARTIAL_ROTARY)
    linear_total_key = linear_num_k_heads * LINEAR_KEY_DIM
    linear_total_value = linear_num_v_heads * LINEAR_VALUE_DIM
    linear_conv_dim = linear_total_key * 2 + linear_total_value
    num_full_attn_layers = num_layers // FULL_ATTN_INTERVAL
    num_linear_layers = num_layers - num_full_attn_layers

    layout = compute_expert_layout(hidden_dim, moe_intermediate)

    config = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_attn_heads": num_attn_heads,
        "num_kv_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate": moe_intermediate,
        "shared_intermediate": shared_intermediate,
        "linear_num_v_heads": linear_num_v_heads,
        "linear_num_k_heads": linear_num_k_heads,
        "rotary_dim": rotary_dim,
        "linear_total_key": linear_total_key,
        "linear_total_value": linear_total_value,
        "linear_conv_dim": linear_conv_dim,
        "num_full_attn_layers": num_full_attn_layers,
        "num_linear_layers": num_linear_layers,
        "expert_size_4bit": layout["expert_size_4bit"],
        "expert_size_2bit": layout["expert_size_2bit"],
        "expert_layout_4bit": layout["layout_4bit"],
        "expert_layout_2bit": layout["layout_2bit"],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model_config.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  model_config.json: {hidden_dim}d, {num_layers}L, {moe_intermediate}MoE_inter, {num_experts} experts")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: non-expert weights → model_weights.bin + manifest
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import re
from collections import defaultdict

def convert_weights(model_dir: str, output_dir: str):
    """Extract all non-expert weights into a single mmap-able binary."""
    import glob

    model = Path(model_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Find safetensors files
    st_files = sorted(glob.glob(str(model / "*.safetensors")))

    # Also check for a single model.safetensors
    if not st_files:
        single = model / "model.safetensors"
        if single.exists():
            st_files = [str(single)]

    if not st_files:
        print(f"ERROR: No safetensors files found in {model}", file=sys.stderr)
        sys.exit(1)

    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    # Collect all tensor metadata
    print(f"  Scanning {len(st_files)} safetensors file(s)...")
    all_tensors = {}  # name -> {dtype, shape, data_offset, file_idx}
    tensor_bytes = []  # per-tensor raw bytes
    tensor_order = []  # tensor names in layout order

    offset = 0
    for fi, st_path in enumerate(st_files):
        with safe_open(st_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Store as raw bytes in native dtype
                raw = tensor.numpy().tobytes()

                # Pad to 64-byte alignment
                pad = (64 - (len(raw) % 64)) % 64
                if pad:
                    raw = raw + b"\x00" * pad

                all_tensors[key] = {
                    "offset": offset,
                    "size": len(raw),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).split(".")[-1],
                }
                tensor_bytes.append(raw)
                tensor_order.append(key)
                offset += len(raw)

    # Sort by name for deterministic output
    tensor_order.sort()
    sorted_bytes = []
    sorted_offset = 0
    sorted_tensors = {}

    for name in tensor_order:
        info = all_tensors[name]
        raw = tensor_bytes[tensor_order.index(name)]
        sorted_tensors[name] = {
            "offset": sorted_offset,
            "size": info["size"],
            "shape": info["shape"],
            "dtype": translate_dtype(info["dtype"]),
        }
        sorted_bytes.append(raw)
        sorted_offset += info["size"]

    # Write binary blob
    bin_path = out / "model_weights.bin"
    total_size = 0
    with open(bin_path, "wb") as f:
        for raw in sorted_bytes:
            f.write(raw)
            total_size += len(raw)

    # Write manifest
    manifest = {"tensors": sorted_tensors}
    json_path = out / "model_weights.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  model_weights.bin: {total_size / 1e9:.2f} GB, {len(sorted_tensors)} tensors")


def translate_dtype(dtype: str) -> str:
    """Convert numpy dtype string to manifest dtype string."""
    mapping = {
        "float32": "F32",
        "float16": "F16",
        "bfloat16": "BF16",
        "uint32": "U32",
        "uint16": "U16",
        "uint8": "U8",
        "int32": "I32",
        "int64": "I64",
    }
    return mapping.get(dtype, dtype.upper())


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: repack 4-bit experts
# ─────────────────────────────────────────────────────────────────────────────

def convert_experts(model_dir: str, output_dir: str):
    """Repack 4-bit routed experts into per-layer binaries."""
    import glob

    model = Path(model_dir)
    out = Path(output_dir) / "packed_experts"
    out.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg_path = model / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        cfg = cfg["text_config"]

    hidden_dim = cfg.get("hidden_size", 4096)
    moe_inter = cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 1024))
    num_layers = cfg.get("num_hidden_layers", 60)
    num_experts = cfg.get("num_experts", 512)

    layout = compute_expert_layout(hidden_dim, moe_inter)
    expert_size = layout["expert_size_4bit"]
    lo = layout["layout_4bit"]

    # Find safetensors files
    st_files = sorted(glob.glob(str(model / "*.safetensors")))
    if not st_files:
        single = model / "model.safetensors"
        if single.exists():
            st_files = [str(single)]
    if not st_files:
        print(f"ERROR: No safetensors files found in {model}", file=sys.stderr)
        sys.exit(1)

    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    # Collect all expert tensors into a lookup
    # Pattern: model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.{weight,scales,biases}
    expert_re = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)_proj\.(\w+)")

    # Read all into memory
    print(f"  Reading expert tensors from {len(st_files)} file(s)...")
    expert_data = {}  # (layer, expert, component_type) -> np.array
    for st_path in st_files:
        with safe_open(st_path, framework="pt") as f:
            for key in f.keys():
                m = expert_re.match(key)
                if not m:
                    continue
                layer = int(m.group(1))
                expert = int(m.group(2))
                comp = m.group(3)  # gate, up, down
                ptype = m.group(4)  # weight, scales, biases
                tensor = f.get_tensor(key).numpy()
                expert_data[(layer, expert, comp, ptype)] = tensor

    if not expert_data:
        print(f"ERROR: No expert tensors found in safetensors files", file=sys.stderr)
        sys.exit(1)

    # For each layer, pack all experts into one binary
    for layer in range(num_layers):
        out_path = out / f"layer_{layer:04}_experts.bin"
        with open(out_path, "wb") as f:
            for expert in range(num_experts):
                # Gate: weight (U32), scales (U16), biases (U16)
                gw = expert_data.get((layer, expert, "gate", "weight"))
                gs = expert_data.get((layer, expert, "gate", "scales"))
                gb = expert_data.get((layer, expert, "gate", "biases"))
                # Up: weight (U32), scales (U16), biases (U16)
                uw = expert_data.get((layer, expert, "up", "weight"))
                us = expert_data.get((layer, expert, "up", "scales"))
                ub = expert_data.get((layer, expert, "up", "biases"))
                # Down: weight (U32), scales (U16), biases (U16)
                dw = expert_data.get((layer, expert, "down", "weight"))
                ds = expert_data.get((layer, expert, "down", "scales"))
                db = expert_data.get((layer, expert, "down", "biases"))

                # Write in layout order
                for arr in [gw, gs, gb, uw, us, ub, dw, ds, db]:
                    if arr is not None:
                        raw = arr.tobytes()
                        f.write(raw)
                    else:
                        # Fill with zeros if tensor is missing
                        f.write(b"\x00" * lo.get("gate_w_size", 0))

        sz = os.path.getsize(out_path)
        expected = expert_size * num_experts
        if sz != expected:
            print(f"  WARNING: layer {layer} size mismatch: {sz} != {expected}")

    per_layer = expert_size * num_experts
    total = per_layer * num_layers
    print(f"  packed_experts/: {num_layers} layers, {num_experts} experts each ({total / 1e9:.1f} GB total)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert HF Qwen3 MoE model to Flash-MoE format"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to HuggingFace model directory (with config.json, tokenizer.json, *.safetensors)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: <model>/../flash-moe-data)",
    )
    parser.add_argument(
        "--step", type=str, default=None,
        choices=["tokenizer", "config", "weights", "experts"],
        help="Run a single step (default: all)",
    )
    args = parser.parse_args()

    model_dir = str(Path(args.model).resolve())
    output_dir = args.output or os.path.join(os.path.dirname(model_dir), "flash-moe-data")
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Flash-MoE Converter")
    print(f"  Model:  {model_dir}")
    print(f"  Output: {output_dir}")
    print()

    steps = ["tokenizer", "config", "weights", "experts"]
    if args.step:
        steps = [args.step]

    t0 = time.time()

    for step in steps:
        header = f"Step {steps.index(step) + 1}/{len(steps)}: {step}"
        print(f"{'=' * 50}")
        print(header)
        print(f"{'=' * 50}")

        if step == "tokenizer":
            convert_tokenizer(model_dir, output_dir)
        elif step == "config":
            convert_config(model_dir, output_dir)
        elif step == "weights":
            convert_weights(model_dir, output_dir)
        elif step == "experts":
            convert_experts(model_dir, output_dir)

        print()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Model ready in: {output_dir}/")
    print()
    print("Next steps:")
    print(f"  cd moe_infer_rs")
    print(f"  cargo run --release -- --serve 8000 --model {output_dir}")


if __name__ == "__main__":
    main()
