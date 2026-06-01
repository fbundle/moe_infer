#!/usr/bin/env python3
"""
extract_weights.py — Extract all non-expert weights from Qwen3.5-397B-A17B-4bit
into a single binary file that the C inference engine can mmap.

Outputs:
  - model_weights.bin: binary blob containing all non-expert weight tensors
  - model_weights.json: manifest describing each tensor's location, shape, dtype

The binary format is simple:
  - Tensors are packed contiguously, 64-byte aligned
  - Each tensor is stored in its native format (U32 packed, BF16 as uint16, F32)
  - The JSON manifest maps tensor names to {offset, size, shape, dtype}

Usage:
    python extract_weights.py [--model PATH] [--output DIR]
"""

import json
import struct
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
import re
import numpy as np


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def bf16_to_f32(u16):
    """Convert bf16 (uint16) to float32."""
    return struct.unpack('!f', struct.pack('!I', u16 << 16))[0]


def f32_to_bf16(f32):
    """Convert float32 to bf16 (uint16) with round-to-nearest-even."""
    # Reinterpret f32 as uint32, add rounding bias, truncate to 16 bits
    i = struct.unpack('!I', struct.pack('!f', f32))[0]
    # Round to nearest even for the lower 16 bits
    round_bit = (i >> 15) & 1
    sticky = i & 0x7FFF
    if round_bit and (sticky or (i >> 16) & 1):
        i += 0x10000
    return (i >> 16) & 0xFFFF


def dequant_8bit_to_f32(weight_int8, scales_bf16, biases_bf16, out_dim, in_dim, group_size):
    """Dequantize MLX 8-bit tensor to float32.

    weight_int8: uint8 array [out_dim, in_dim]
    scales_bf16: uint16 array [out_dim, in_dim//group_size]
    biases_bf16: uint16 array [out_dim, in_dim//group_size]
    Returns float32 array [out_dim, in_dim]
    """
    num_groups = in_dim // group_size
    result = np.zeros(out_dim * in_dim, dtype=np.float32)
    w = weight_int8.reshape(out_dim, num_groups, group_size)
    s = scales_bf16.reshape(out_dim, num_groups)
    b = biases_bf16.reshape(out_dim, num_groups)

    for i in range(out_dim):
        for g in range(num_groups):
            scale = bf16_to_f32(int(s[i, g]))
            bias = bf16_to_f32(int(b[i, g]))
            start = (i * num_groups + g) * group_size
            result[start:start + group_size] = (
                w[i, g, :].astype(np.float32) * scale + bias
            )
    return result


def quant_f32_to_4bit_packed(values, out_dim, in_dim, group_size):
    """Quantize float32 array [out_dim, in_dim] to 4-bit packed.

    Returns (packed_uint32, scales_bf16, biases_bf16) in the MLX 4-bit layout:
      packed: uint32 array [out_dim * in_dim // 8]
      scales: uint16 array [out_dim * in_dim // group_size]
      biases: uint16 array [out_dim * in_dim // group_size]
    """
    num_groups = in_dim // group_size
    total = out_dim * in_dim
    packed = np.zeros(total // 8, dtype=np.uint32)
    scales = np.zeros(out_dim * num_groups, dtype=np.uint16)
    biases = np.zeros(out_dim * num_groups, dtype=np.uint16)

    v = values.reshape(out_dim, num_groups, group_size)

    for i in range(out_dim):
        for g in range(num_groups):
            chunk = v[i, g, :]
            vmin = float(chunk.min())
            vmax = float(chunk.max())
            if vmax == vmin:
                vmax = vmin + 1.0  # avoid div by zero

            fscale = (vmax - vmin) / 15.0
            fbias = vmin

            # Store scale/bias as bf16
            s_idx = i * num_groups + g
            scales[s_idx] = f32_to_bf16(fscale)
            biases[s_idx] = f32_to_bf16(fbias)

            # Quantize each value and pack into nibbles
            for j in range(group_size):
                q = int(round((float(chunk[j]) - fbias) / fscale))
                q = max(0, min(15, q))
                global_idx = (i * num_groups + g) * group_size + j
                word_idx = global_idx // 8
                nibble_shift = (global_idx % 8) * 4
                packed[word_idx] |= np.uint32(q & 0xF) << nibble_shift

    return packed, scales, biases


def run(model_path_str, output_dir_str, include_experts=False):
    model_path = Path(model_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the weight index
    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Filter: keep only language_model weights, skip vision and expert tensors
    # Raw BF16 models use .mlp.experts.{gate_up_proj,down_proj} (fused)
    # Pre-quantized models use .switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
    expert_pattern = re.compile(r'\.(switch_mlp|mlp\.experts)\.(gate_proj|up_proj|down_proj|gate_up_proj)',)
    mtp_expert_pattern = re.compile(r'mtp\.layers\.\d+\.mlp\.experts\.')
    vision_pattern = re.compile(r'^(vision_tower|model\.visual)')

    tensors_to_extract = {}  # name -> filename
    skipped_expert = 0
    skipped_vision = 0

    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            skipped_vision += 1
            continue
        if not include_experts and (expert_pattern.search(name) or mtp_expert_pattern.search(name)):
            skipped_expert += 1
            continue
        tensors_to_extract[name] = filename

    print(f"Model: {model_path}")
    print(f"Total weights in index: {len(weight_map)}")
    print(f"Skipped vision: {skipped_vision}")
    print(f"Skipped expert: {skipped_expert}")
    print(f"Extracting: {len(tensors_to_extract)} tensors")

    # Group by shard file for sequential I/O
    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    # Parse headers and plan layout
    print("\nParsing safetensors headers...")
    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))

    # Sanitize tensor names for the C engine.
    # MLX-quantized: "language_model.model.X" -> "model.X"
    #                "language_model.lm_head.X" -> "lm_head.X"
    # Raw Qwen3.5:   "model.language_model.X" -> "model.X"
    def sanitize_name(name):
        if name.startswith("language_model.model."):
            return "model." + name[len("language_model.model."):]
        if name.startswith("language_model."):
            return name[len("language_model."):]
        if name.startswith("model.language_model."):
            return "model." + name[len("model.language_model."):]
        return name

    # Plan the output layout
    # Sort tensors for deterministic output
    all_tensors = []  # (sanitized_name, original_name, filename)
    for name in sorted(tensors_to_extract.keys()):
        san_name = sanitize_name(name)
        all_tensors.append((san_name, name, tensors_to_extract[name]))

    # Write binary file
    bin_path = output_dir / 'model_weights.bin'
    # Auto-detect model config from HuggingFace config.json
    model_cfg_path = model_path / "config.json"
    if model_cfg_path.exists():
        with open(model_cfg_path) as f:
            hf_config = json.load(f)
        # Qwen3.5 MoE multimodal models nest text params under "text_config"
        if "text_config" in hf_config:
            hf_config = hf_config["text_config"]
        num_layers = hf_config.get("num_hidden_layers", 60)
        manifest = {
            "model": str(model_path),
            "num_tensors": len(all_tensors),
            "tensors": {},
            "config": {
                "hidden_size": hf_config.get("hidden_size", 4096),
                "num_hidden_layers": num_layers,
                "num_attention_heads": hf_config.get("num_attention_heads", 32),
                "num_key_value_heads": hf_config.get("num_key_value_heads", 2),
                "head_dim": hf_config.get("head_dim", hf_config.get("hidden_size", 4096) // hf_config.get("num_attention_heads", 32)),
                "vocab_size": hf_config.get("vocab_size", 248320),
                "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
                "num_experts": hf_config.get("num_experts", 512),
                "num_experts_per_tok": hf_config.get("num_experts_per_tok", 10),
                "moe_intermediate_size": hf_config.get("moe_intermediate_size", hf_config.get("intermediate_size", 1024)),
                "shared_expert_intermediate_size": hf_config.get("shared_expert_intermediate_size", hf_config.get("intermediate_size", 1024)),
                "full_attention_interval": hf_config.get("full_attention_interval", 4),
                "linear_num_value_heads": hf_config.get("linear_num_value_heads", 64),
                "linear_num_key_heads": hf_config.get("linear_num_key_heads", 16),
                "linear_key_head_dim": hf_config.get("linear_key_head_dim", 128),
                "linear_value_head_dim": hf_config.get("linear_value_head_dim", 128),
                "linear_conv_kernel_dim": hf_config.get("linear_conv_kernel_dim", 4),
                "partial_rotary_factor": hf_config.get("partial_rotary_factor", 0.25),
                "rope_theta": hf_config.get("rope_theta", 10000000.0),
            }
        }
    else:
        print(f"WARNING: {model_cfg_path} not found, using defaults")
        num_layers = 60
        manifest = {
            "model": str(model_path),
            "num_tensors": len(all_tensors),
            "tensors": {},
            "config": {
                "hidden_size": 4096,
                "num_hidden_layers": 60,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 1024,
                "shared_expert_intermediate_size": 1024,
                "full_attention_interval": 4,
                "linear_num_value_heads": 64,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
            }
        }

    # Layer type map
    layer_types = []
    for i in range(num_layers):
        if (i + 1) % manifest["config"]["full_attention_interval"] == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")
    manifest["config"]["layer_types"] = layer_types

    # Detect 8-bit quantized tensors from quantization config.
    # Qwen3.6+ uses INT8 for routing gate and shared_expert_gate tensors.
    # We dequantize and re-quantize to 4-bit so the C engine needs no changes.
    eight_bit_tensors = {}  # HF name -> {"group_size": N, "bits": 8}
    if model_cfg_path.exists():
        with open(model_cfg_path) as f:
            hf_full = json.load(f)
        # Check both "quantization" and "quantization_config" keys
        for qkey in ("quantization", "quantization_config"):
            qcfg = hf_full.get(qkey, {})
            for tensor_path, tcfg in qcfg.items():
                if isinstance(tcfg, dict) and tcfg.get("bits") == 8:
                    eight_bit_tensors[tensor_path] = tcfg

    # Pre-process 8-bit tensors: read weight+scales+biases, convert to 4-bit.
    # Result stored as sanitized_name -> raw bytes for weight/scales/biases.
    converted_4bit = {}  # sanitized_name -> bytes
    if eight_bit_tensors:
        print(f"\nConverting {len(eight_bit_tensors)} INT8 tensors to 4-bit...")
        for hf_name, qcfg in sorted(eight_bit_tensors.items()):
            gs = qcfg["group_size"]
            san_base = sanitize_name(hf_name)

            # Find the safetensors file containing this tensor
            hf_weight_name = hf_name + ".weight"
            if hf_weight_name not in weight_map:
                print(f"  WARNING: {hf_weight_name} not in weight map, skipping")
                continue
            sf_name = weight_map[hf_weight_name]
            sf_path = model_path / sf_name
            header, data_start = header_cache[sf_name]

            # Read weight (int8), scales (bf16), biases (bf16) from safetensors
            def read_tensor(hdr_name):
                if hdr_name not in header:
                    return None
                meta = header[hdr_name]
                off = meta['data_offsets']
                length = off[1] - off[0]
                shape = meta['shape']
                with open(sf_path, 'rb') as sf:
                    sf.seek(data_start + off[0])
                    return sf.read(length), shape

            w_data = read_tensor(hf_weight_name)
            s_data = read_tensor(hf_name + ".scales")
            b_data = read_tensor(hf_name + ".biases")

            if not all([w_data, s_data, b_data]):
                print(f"  WARNING: incomplete data for {hf_name}, skipping")
                continue

            w_raw, w_shape = w_data
            out_dim = w_shape[0]
            # MLX stores INT8 packed as U32 (4 uint8 per uint32).
            # The stated in_dim is U32 elements; actual int8 dim = in_dim * 4.
            in_dim_u32 = w_shape[1]
            in_dim = in_dim_u32 * 4  # actual int8 element count

            # Dequantize 8-bit -> float32
            weight_int8 = np.frombuffer(w_raw, dtype=np.uint8)
            scales_bf16 = np.frombuffer(s_data[0], dtype=np.uint16)
            biases_bf16 = np.frombuffer(b_data[0], dtype=np.uint16)

            f32_vals = dequant_8bit_to_f32(
                weight_int8, scales_bf16, biases_bf16,
                out_dim, in_dim, gs
            )

            # Re-quantize float32 -> 4-bit packed
            new_group_size = 64  # match GROUP_SIZE used by C engine
            packed, new_scales, new_biases = quant_f32_to_4bit_packed(
                f32_vals, out_dim, in_dim, new_group_size
            )

            converted_4bit[san_base + ".weight"] = packed.tobytes()
            converted_4bit[san_base + ".scales"] = new_scales.tobytes()
            converted_4bit[san_base + ".biases"] = new_biases.tobytes()

            print(f"  {san_base}: {out_dim}x{in_dim} INT8 -> 4-bit "
                  f"(packed={len(packed) * 4}B, s={len(new_scales) * 2}B, b={len(new_biases) * 2}B)")

    print(f"\nWriting {bin_path}...")
    t0 = time.time()
    offset = 0
    total_bytes = 0

    ALIGN = 64  # 64-byte alignment for Metal buffers

    with open(bin_path, 'wb') as out_f:
        for i, (san_name, orig_name, filename) in enumerate(all_tensors):
            # Check for 8-bit -> 4-bit converted data
            if san_name in converted_4bit:
                data = converted_4bit[san_name]
                byte_len = len(data)
                shape = []
                dtype = "U32" if ".weight" in san_name else "BF16"
            else:
                filepath = model_path / filename
                header, data_start = header_cache[filename]

                if orig_name not in header:
                    print(f"  WARNING: {orig_name} not found in {filename}, skipping")
                    continue

                meta = header[orig_name]
                tensor_offsets = meta['data_offsets']
                byte_len = tensor_offsets[1] - tensor_offsets[0]
                shape = meta['shape']
                dtype = meta['dtype']

                # Read tensor data from safetensors
                with open(filepath, 'rb') as sf:
                    sf.seek(data_start + tensor_offsets[0])
                    data = sf.read(byte_len)

            # Align offset
            if offset % ALIGN != 0:
                pad = ALIGN - (offset % ALIGN)
                out_f.write(b'\x00' * pad)
                offset += pad

            out_f.write(data)

            manifest["tensors"][san_name] = {
                "offset": offset,
                "size": byte_len,
                "shape": shape,
                "dtype": dtype,
            }

            offset += byte_len
            total_bytes += byte_len

            if (i + 1) % 100 == 0 or i == len(all_tensors) - 1:
                print(f"  [{i+1}/{len(all_tensors)}] {total_bytes / 1e9:.2f} GB written")

    elapsed = time.time() - t0
    throughput = total_bytes / elapsed / 1e9

    print(f"\nDone: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s ({throughput:.1f} GB/s)")
    print(f"Binary: {bin_path} ({os.path.getsize(bin_path) / 1e9:.2f} GB)")

    # Write manifest
    json_path = output_dir / 'model_weights.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {json_path}")

    # Print summary by category
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for san_name, info in manifest["tensors"].items():
        if "embed_tokens" in san_name:
            cat = "embedding"
        elif "norm.weight" in san_name and "layers." not in san_name:
            cat = "final_norm"
        elif "lm_head" in san_name:
            cat = "lm_head"
        elif "input_layernorm" in san_name or "post_attention_layernorm" in san_name:
            cat = "layer_norms"
        elif "linear_attn" in san_name:
            cat = "linear_attention"
        elif "self_attn" in san_name:
            cat = "full_attention"
        elif "mlp.gate." in san_name:
            cat = "routing_gate"
        elif "shared_expert." in san_name:
            cat = "shared_expert"
        elif "shared_expert_gate" in san_name:
            cat = "shared_expert_gate"
        elif "switch_mlp" in san_name:
            cat = "routed_experts"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories.keys()):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes']/1e6:8.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Extract non-expert weights to binary')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model directory (HuggingFace safetensors)')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory for model_weights.bin and .json')
    parser.add_argument('--include-experts', action='store_true',
                        help='Also extract expert weights (huge, not recommended)')
    args = parser.parse_args()
    run(args.model, args.output, args.include_experts)

if __name__ == '__main__':
    main()
