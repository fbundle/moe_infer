"""Dequantize BQ4 format back to HuggingFace safetensors format.

Requires the original (stripped) HF model as a shape reference, since BQ4
stores BF16 tensors flat without preserving 2D/3D shapes.

Usage::

    python -m moe_infer.dequantize data/Qwen3.6-35B-A3B-Strip/model_bq4 \
        --ref hub/models--Qwen--Qwen3.6-35B-A3B-Strip \
        --out hub/models--Qwen--Qwen3.6-35B-A3B-Strip-Dequant
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np

GROUP_SIZE = 64
FP8_GROUP_SIZE = 128


# ─── BF16 ↔ f32 ────────────────────────────────────────────────────────────

def bf16_to_f32(u16: int) -> float:
    return struct.unpack("!f", struct.pack("!I", u16 << 16))[0]


def bf16_bytes_to_f32(data: bytes) -> np.ndarray:
    u16 = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


# ─── INT4 dequantization ───────────────────────────────────────────────────

def dequant_int4(
    packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
    out_dim: int, in_dim: int,
) -> np.ndarray:
    num_groups = in_dim // GROUP_SIZE
    words_per_row = in_dim // 8
    scales_f32 = np.array([bf16_to_f32(int(s)) for s in scales], dtype=np.float32)
    biases_f32 = np.array([bf16_to_f32(int(b)) for b in biases], dtype=np.float32)
    result = np.zeros(out_dim * in_dim, dtype=np.float32)
    for row in range(out_dim):
        w_base = row * words_per_row
        s_base = row * num_groups
        for g in range(num_groups):
            scale = scales_f32[s_base + g]
            bias = biases_f32[s_base + g]
            out_base = row * in_dim + g * GROUP_SIZE
            for p in range(8):
                word = int(packed[w_base + g * 8 + p])
                for n in range(8):
                    nibble = (word >> (n * 4)) & 0xF
                    result[out_base + p * 8 + n] = float(nibble) * scale + bias
    return result.reshape(out_dim, in_dim)


# ─── INT8 dequantization ───────────────────────────────────────────────────

def dequant_int8(
    packed: np.ndarray, scales: np.ndarray,
    out_dim: int, in_dim: int,
) -> np.ndarray:
    result = np.zeros(out_dim * in_dim, dtype=np.float32)
    for row in range(out_dim):
        scale = scales[row]
        src = packed[row * in_dim:(row + 1) * in_dim]
        result[row * in_dim:(row + 1) * in_dim] = src.astype(np.float32) * scale
    return result.reshape(out_dim, in_dim)


# ─── FP4 / FP8 dequantization ──────────────────────────────────────────────

def _dequant_fp4_e2m1(
    packed: np.ndarray, scales: np.ndarray,
    out_dim: int, in_dim: int,
) -> np.ndarray:
    fp4_lut = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32)
    num_groups = in_dim // GROUP_SIZE
    words_per_row = in_dim // 8
    result = np.zeros(out_dim * in_dim, dtype=np.float32)
    for row in range(out_dim):
        w_base = row * words_per_row
        s_base = row * num_groups
        for g in range(num_groups):
            scale = bf16_to_f32(int(scales[s_base + g]))
            out_base = row * in_dim + g * GROUP_SIZE
            for p in range(8):
                word = int(packed[w_base + g * 8 + p])
                for n in range(8):
                    nibble = (word >> (n * 4)) & 0xF
                    result[out_base + p * 8 + n] = fp4_lut[nibble] * scale
    return result.reshape(out_dim, in_dim)


def _dequant_fp8_e4m3(
    packed: np.ndarray, scales: np.ndarray,
    out_dim: int, in_dim: int,
) -> np.ndarray:
    fp8_lut = np.zeros(256, dtype=np.float32)
    pow2 = np.array([1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4,
                     1.0 / 2, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
                     128.0, 256.0], dtype=np.float32)
    for i in range(256):
        sign = -1.0 if (i >> 7) else 1.0
        exp = (i >> 3) & 0xF
        mant = i & 0x7
        if exp == 0:
            mag = 0.0 if mant == 0 else mant / 512.0
        elif exp == 15:
            mag = 240.0
        else:
            mag = pow2[exp - 1] * (1.0 + mant / 8.0)
        fp8_lut[i] = sign * mag

    gs = FP8_GROUP_SIZE
    num_groups = in_dim // gs
    result = np.zeros(out_dim * in_dim, dtype=np.float32)
    for row in range(out_dim):
        s_base = row * num_groups
        for g in range(num_groups):
            scale = bf16_to_f32(int(scales[s_base + g]))
            g_base = row * in_dim + g * gs
            for j in range(gs):
                result[g_base + j] = fp8_lut[int(packed[g_base + j])] * scale
    return result.reshape(out_dim, in_dim)


# ─── Expert layout (matches src/engine/qwen35_moe/constants.rs) ────────────

def _expert_layout(hd: int, mi: int, gs: int) -> dict:
    gate_w = mi * hd // 2
    gate_sb = mi * (hd // gs) * 2
    up_w = mi * hd // 2
    up_sb = mi * (hd // gs) * 2
    down_w = hd * mi // 2
    down_sb = hd * (mi // gs) * 2
    return {
        "gate_w_off": 0,
        "gate_s_off": gate_w,
        "gate_b_off": gate_w + gate_sb,
        "up_w_off": gate_w + 2 * gate_sb,
        "up_s_off": gate_w + 2 * gate_sb + up_w,
        "up_b_off": gate_w + 2 * gate_sb + up_w + up_sb,
        "down_w_off": gate_w + 2 * gate_sb + up_w + 2 * up_sb,
        "down_s_off": gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w,
        "down_b_off": gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + down_sb,
        "expert_size": gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + 2 * down_sb,
    }


def _dequant_expert_section(
    data: memoryview, offset: int, out_dim: int, in_dim: int,
) -> np.ndarray:
    """Read packed+scales+biases for one expert sub-tensor from raw bytes."""
    num_groups = in_dim // GROUP_SIZE
    words_per_row = in_dim // 8
    w_bytes = out_dim * words_per_row * 4
    sb_bytes = out_dim * num_groups * 2

    packed = np.frombuffer(data[offset:offset + w_bytes], dtype=np.uint32).copy()
    scales = np.frombuffer(data[offset + w_bytes:offset + w_bytes + sb_bytes], dtype=np.uint16).copy()
    biases = np.frombuffer(data[offset + w_bytes + sb_bytes:offset + w_bytes + 2 * sb_bytes], dtype=np.uint16).copy()
    return dequant_int4(packed, scales, biases, out_dim, in_dim)


# ─── Name mapping ──────────────────────────────────────────────────────────

def _load_reverse_mapping(num_layers: int) -> dict[str, str]:
    """Load name_mapping.json and invert it (MLX → HF)."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "src", "quantize", "qwen35_moe", "name_mapping.json"),
        os.path.join(os.path.dirname(__file__), "..", "quant", "name_mapping.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c) as f:
                mapping: dict[str, str] = json.load(f)
            break
    else:
        print("[dequantize] WARNING: name_mapping.json not found — keeping MLX names")
        return {}

    expanded: dict[str, str] = {}
    for hf_pat, mlx_pat in mapping.items():
        if "{L}" in hf_pat:
            for l in range(num_layers):
                expanded[hf_pat.replace("{L}", str(l))] = mlx_pat.replace("{L}", str(l))
        elif "{B}" in hf_pat:
            for b in range(27):
                expanded[hf_pat.replace("{B}", str(b))] = mlx_pat.replace("{B}", str(b))
        else:
            expanded[hf_pat] = mlx_pat

    reverse: dict[str, str] = {}
    for hf_name, mlx_name in expanded.items():
        reverse[mlx_name] = hf_name
    return reverse


def _mlx_to_hf_name(mlx_name: str, reverse_map: dict[str, str]) -> str:
    """Convert MLX-internal tensor name to HF convention."""
    if mlx_name in reverse_map:
        return reverse_map[mlx_name]
    # Try with/without .weight suffix
    if mlx_name.endswith(".weight"):
        base = mlx_name[: -len(".weight")]
        if base + ".weight" in reverse_map:
            return reverse_map[base + ".weight"]
    else:
        if mlx_name + ".weight" in reverse_map:
            return reverse_map[mlx_name + ".weight"]
    return mlx_name  # fallback


# ─── Reference shape lookup ────────────────────────────────────────────────

def _read_ref_shapes(ref_dir: str) -> dict[str, tuple[int, ...]]:
    """Read tensor shapes from a HuggingFace safetensors model directory."""
    idx_path = os.path.join(ref_dir, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        # Single-file safetensors — parse header
        sf_path = os.path.join(ref_dir, "model.safetensors")
        if not os.path.exists(sf_path):
            print(f"[dequantize] WARNING: no safetensors found in {ref_dir}")
            return {}
        with open(sf_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode())
        shapes = {}
        for name, info in header.items():
            if name != "__metadata__":
                shapes[name] = tuple(info["shape"])
        return shapes

    with open(idx_path) as f:
        idx = json.load(f)

    shapes: dict[str, tuple[int, ...]] = {}
    # We need to read shard headers to get shapes
    shard_files = set(idx["weight_map"].values())
    for shard in shard_files:
        sf_path = os.path.join(ref_dir, shard)
        with open(sf_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode())
        for name, info in header.items():
            if name != "__metadata__":
                shapes[name] = tuple(info["shape"])
    return shapes


# ─── Main dequantize function ──────────────────────────────────────────────

# ─── Reverse sanitization ──────────────────────────────────────────────────

def _is_norm_weight(name: str) -> bool:
    """Check if a tensor name is a layernorm/rms-norm weight."""
    _norm_keys = (
        ".input_layernorm.weight", ".post_attention_layernorm.weight",
        "model.norm.weight", ".q_norm.weight", ".k_norm.weight",
        ".pre_fc_norm_hidden.weight", ".pre_fc_norm_embedding.weight",
        ".norm.weight",
    )
    return any(name.endswith(k) for k in _norm_keys)


def _is_qwen36(model_path: str) -> bool:
    """Detect Qwen3.6 (as opposed to 3.5) from the model id string."""
    return "3.6" in model_path or "Qwen3.6" in model_path


def _reverse_moveaxis_2_to_1(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reverse the moveaxis_2_to_1 sanitization: [c, s, k] → [c, k, s]."""
    c, s, k = shape
    out = np.zeros(c * k * s, dtype=arr.dtype)
    orig = arr.ravel().copy()
    for ci in range(c):
        for ki in range(k):
            for si in range(s):
                old_idx = ci * (s * k) + si * k + ki
                new_idx = ci * (k * s) + ki * s + si
                out[new_idx] = orig[old_idx]
    return out.reshape(c, k, s)


def dequantize(quant_dir: str, *, ref: str, out: str | None = None) -> Path:
    """Dequantize a BQ4 model back to HuggingFace safetensors format.

    Parameters
    ----------
    quant_dir : str
        Path to BQ4 directory (contains model_weights.bin, model_weights.json,
        packed_experts/).
    ref : str
        Path to original stripped HF model for shape reference.
    out : str or None
        Output directory. Defaults to ``{quant_dir}-Dequant``.
    """
    import safetensors.torch
    import torch

    qdir = Path(quant_dir)
    ref_dir = Path(ref)
    if out is None:
        out = f"{quant_dir}-Dequant"
    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)

    # ── 1. Read BQ4 manifest ─────────────────────────────────────────────
    with open(qdir / "model_weights.json") as f:
        manifest = json.load(f)

    cfg = manifest["config"]
    hd = cfg["hidden_size"]
    mi = cfg["moe_intermediate_size"]
    num_layers = cfg["num_hidden_layers"]
    num_experts = cfg["num_experts"]
    mtp_layers = cfg.get("mtp_num_hidden_layers", 0)
    total_layers = num_layers + mtp_layers

    print(f"[dequantize] {qdir} → {dst}")
    print(f"  hidden={hd} layers={num_layers} experts={num_experts} moe_inter={mi}")

    # ── 2. Read reference shapes ─────────────────────────────────────────
    ref_shapes = _read_ref_shapes(str(ref_dir))
    print(f"  Reference: {len(ref_shapes)} tensors from {ref_dir}")

    # ── 3. Reverse name mapping ─────────────────────────────────────────
    mlx_to_hf = _load_reverse_mapping(num_layers)

    # ── 4. Read model_weights.bin ───────────────────────────────────────
    with open(qdir / "model_weights.bin", "rb") as f:
        bin_data = memoryview(f.read())

    tensors_manifest: dict[str, dict] = manifest["tensors"]

    # ── 5. Classify tensors into primary vs companion ──────────────────
    companion_suffixes = {".scales", ".biases"}
    primary_names: list[str] = []
    for tname in tensors_manifest:
        is_companion = False
        for suf in companion_suffixes:
            if tname.endswith(suf):
                base = tname[: -len(suf)]
                if (base + ".weight") in tensors_manifest or base in tensors_manifest:
                    is_companion = True
                break
        if not is_companion:
            primary_names.append(tname)

    print(f"  {len(primary_names)} primary tensors from manifest")

    # ── 6. Dequantize each primary tensor ───────────────────────────────
    kept: dict[str, torch.Tensor] = {}

    for tname in primary_names:
        info = tensors_manifest[tname]
        offset = info["offset"]
        size = info["size"]
        shape: list[int] = [int(s) for s in info["shape"]]
        dtype_str = info["dtype"]
        data = bin_data[offset:offset + size]

        # Compute HF name
        hf_name = _mlx_to_hf_name(tname, mlx_to_hf)

        if dtype_str == "bf16":
            arr = bf16_bytes_to_f32(bytes(data))
            if len(shape) == 1 and shape[0] == len(arr):
                arr = arr.reshape(shape)

        elif dtype_str == "f32":
            arr = np.frombuffer(data, dtype=np.float32).copy().reshape(shape)

        elif dtype_str == "u32":
            # INT4 weight — needs .scales + .biases
            base = tname[: -len(".weight")] if tname.endswith(".weight") else tname
            scales_name = base + ".scales"
            biases_name = base + ".biases"
            if scales_name not in tensors_manifest or biases_name not in tensors_manifest:
                print(f"  SKIP {tname}: missing scales/biases")
                continue

            scales_info = tensors_manifest[scales_name]
            biases_info = tensors_manifest[biases_name]
            packed = np.frombuffer(data, dtype=np.uint32).copy()
            scales_raw = bytes(bin_data[scales_info["offset"]:scales_info["offset"] + scales_info["size"]])
            biases_raw = bytes(bin_data[biases_info["offset"]:biases_info["offset"] + biases_info["size"]])
            scales = np.frombuffer(scales_raw, dtype=np.uint16).copy()
            biases = np.frombuffer(biases_raw, dtype=np.uint16).copy()

            out_dim = int(scales_info["shape"][0])
            in_dim = int(scales_info["shape"][1]) * GROUP_SIZE  # padded in_dim
            arr = dequant_int4(packed, scales, biases, out_dim, in_dim)

            # Trim padding to match reference shape
            # Reference shape is e.g. [out_dim, original_in_dim]
            if hf_name in ref_shapes:
                ref_shape = ref_shapes[hf_name]
                if len(ref_shape) == 2:
                    ref_out, ref_in = int(ref_shape[0]), int(ref_shape[1])
                    if ref_out == out_dim and ref_in < in_dim:
                        arr = arr[:, :ref_in]

        elif dtype_str == "u8":
            # INT8 weight — needs .scales
            base = tname[: -len(".weight")] if tname.endswith(".weight") else tname
            scales_name = base + ".scales"
            if scales_name not in tensors_manifest:
                print(f"  SKIP {tname}: missing scales")
                continue

            scales_info = tensors_manifest[scales_name]
            packed = np.frombuffer(data, dtype=np.int8).copy()
            scales_raw = bytes(bin_data[scales_info["offset"]:scales_info["offset"] + scales_info["size"]])
            scales = np.frombuffer(scales_raw, dtype=np.float32).copy()

            out_dim = int(scales_info["shape"][0])
            in_dim = int(shape[1])
            arr = dequant_int8(packed, scales, out_dim, in_dim)

        elif dtype_str == "fp4_e2m1":
            base = tname[: -len(".weight")] if tname.endswith(".weight") else tname
            scales_name = base + ".scales"
            if scales_name not in tensors_manifest:
                print(f"  SKIP {tname}: missing scales")
                continue

            scales_info = tensors_manifest[scales_name]
            packed = np.frombuffer(data, dtype=np.uint32).copy()
            scales_raw = bytes(bin_data[scales_info["offset"]:scales_info["offset"] + scales_info["size"]])
            scales = np.frombuffer(scales_raw, dtype=np.uint16).copy()

            out_dim = int(scales_info["shape"][0])
            in_dim = int(scales_info["shape"][1]) * GROUP_SIZE
            arr = _dequant_fp4_e2m1(packed, scales, out_dim, in_dim)

        elif dtype_str == "fp8_e4m3":
            base = tname[: -len(".weight")] if tname.endswith(".weight") else tname
            scales_name = base + ".scales"
            if scales_name not in tensors_manifest:
                print(f"  SKIP {tname}: missing scales")
                continue

            scales_info = tensors_manifest[scales_name]
            packed = np.frombuffer(data, dtype=np.uint8).copy()
            scales_raw = bytes(bin_data[scales_info["offset"]:scales_info["offset"] + scales_info["size"]])
            scales = np.frombuffer(scales_raw, dtype=np.uint16).copy()

            out_dim = int(scales_info["shape"][0])
            in_dim = int(scales_info["shape"][1]) * FP8_GROUP_SIZE
            arr = _dequant_fp8_e4m3(packed, scales, out_dim, in_dim)

        else:
            print(f"  SKIP {tname}: unknown dtype {dtype_str}")
            continue

        # Use reference shape for final reshape (critical for BF16 flat tensors)
        #   BUT: for 3D conv1d tensors, the stored data is in sanitized order
        #   [c, s, k] (after moveaxis_2_to_1), not the original [c, k, s].
        #   Reshape flat data to sanitized shape, then reverse moveaxis.
        if hf_name in ref_shapes:
            target_shape = tuple(int(d) for d in ref_shapes[hf_name])
            if arr.size == int(np.prod(target_shape)):
                if "conv1d.weight" in hf_name and len(target_shape) == 3:
                    # Sanitized shape is [c, s, k] = [ref[0], ref[2], ref[1]]
                    san_shape = (target_shape[0], target_shape[2], target_shape[1])
                    arr = arr.reshape(san_shape)
                    arr = _reverse_moveaxis_2_to_1(arr, san_shape)
                    # arr is now [c, k, s] = reference shape
                else:
                    arr = arr.reshape(target_shape)

        # Reverse Qwen3.6 norm shift: quantization added +1.0 to norm weights.
        # The dequantized model should match the original HF model's unshifted norms.
        if _is_qwen36(str(qdir)) and _is_norm_weight(hf_name):
            arr = arr - 1.0

        # Convert to torch, cast to bf16
        t = torch.from_numpy(arr.astype(np.float32))
        if dtype_str not in ("f32",):
            t = t.to(torch.bfloat16)
        kept[hf_name] = t

    # ── 7. Dequantize expert tensors from packed_experts/ ──────────────
    layout = _expert_layout(hd, mi, GROUP_SIZE)
    expert_dir = qdir / "packed_experts"

    for layer in range(total_layers):
        layer_path = expert_dir / f"layer_{layer:02}.bin"
        if not layer_path.exists():
            continue

        with open(layer_path, "rb") as f:
            expert_data = memoryview(f.read())

        esize = layout["expert_size"]
        gate_up = np.zeros((num_experts, 2 * mi, hd), dtype=np.float32)
        down = np.zeros((num_experts, hd, mi), dtype=np.float32)

        for e in range(num_experts):
            base_off = e * esize
            gate_w = _dequant_expert_section(expert_data, base_off + layout["gate_w_off"], mi, hd)
            up_w = _dequant_expert_section(expert_data, base_off + layout["up_w_off"], mi, hd)
            down_w = _dequant_expert_section(expert_data, base_off + layout["down_w_off"], hd, mi)
            gate_up[e, :mi, :] = gate_w
            gate_up[e, mi:, :] = up_w
            down[e] = down_w

        mlx_gate_up = f"language_model.model.layers.{layer}.mlp.switch_mlp.gate_up_proj.weight"
        mlx_down = f"language_model.model.layers.{layer}.mlp.switch_mlp.down_proj.weight"
        hf_gate_up = _mlx_to_hf_name(mlx_gate_up, mlx_to_hf)
        hf_down = _mlx_to_hf_name(mlx_down, mlx_to_hf)

        kept[hf_gate_up] = torch.from_numpy(gate_up).to(torch.bfloat16)
        kept[hf_down] = torch.from_numpy(down).to(torch.bfloat16)

    # Verify against reference
    missing = set(ref_shapes.keys()) - set(kept.keys())
    extra = set(kept.keys()) - set(ref_shapes.keys())
    if missing:
        print(f"  Missing (in ref but not dequantized): {len(missing)}")
        for n in sorted(missing)[:5]:
            print(f"    - {n}")
    if extra:
        print(f"  Extra (dequantized but not in ref): {len(extra)}")
        for n in sorted(extra)[:5]:
            print(f"    + {n}")

    print(f"  Total kept: {len(kept)} tensors")

    # ── 8. Write safetensors ────────────────────────────────────────────
    out_weights = dst / "model.safetensors"
    safetensors.torch.save_file(kept, out_weights)

    out_index: dict[str, Any] = {
        "metadata": {"total_size": os.path.getsize(out_weights)},
        "weight_map": {name: "model.safetensors" for name in kept},
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(out_index, f, indent=2)

    # ── 9. Copy config + tokenizer from reference ───────────────────────
    copy_exts = {".json", ".txt", ".jinja", ".model"}
    for fname in sorted(os.listdir(ref_dir)):
        if fname.startswith("model") and fname.endswith(".safetensors"):
            continue
        if fname == "model.safetensors.index.json":
            continue
        p = ref_dir / fname
        if p.suffix in copy_exts or p.is_dir():
            if p.is_dir():
                if not (dst / fname).exists():
                    shutil.copytree(p, dst / fname)
            else:
                shutil.copy2(p, dst / fname)

    total_mb = os.path.getsize(out_weights) / 1e6
    print(f"  Wrote {total_mb:.0f} MB to {out_weights}")
    print("Done.")
    return dst


# ─── CLI ───────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Dequantize BQ4 model to HuggingFace safetensors")
    parser.add_argument("quant_dir", help="Path to BQ4 directory")
    parser.add_argument("--ref", required=True,
                        help="Path to original stripped HF model (for shapes)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: {quant_dir}-Dequant)")
    args = parser.parse_args()
    dequantize(args.quant_dir, ref=args.ref, out=args.out)


if __name__ == "__main__":
    _main()
