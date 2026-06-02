"""Dequantize Gemma 4 BQ4 → HuggingFace safetensors.

Mirrors moe_infer/dequantize.py (qwen35) but with Gemma-specific
naming and per-layer expert layout.

  - Tensor name reverse map: engine ``language_model.model.X`` →
    HF ``model.language_model.X``.
  - Expert tensors: per-layer blob → 3D
    ``model.language_model.layers.{L}.experts.gate_up_proj`` of shape
    ``[E, 2*moe_inter, hidden]`` and ``...experts.down_proj`` of shape
    ``[E, hidden, moe_inter]``.
  - NO +1 norm shift on the way out (Gemma 4 HF weights are stored
    absolute; nothing to undo at this point).

Output is a HF safetensors dir with the architecture set to
``Gemma4ForConditionalGeneration`` (drops the ``_Stripped`` marker) so
the result loads cleanly into MLX-VLM or transformers.

Usage::

    python -m moe_infer.gemma4_moe.dequantize \\
        data/gemma-4-26B-A4B-Strip/model_bq4 \\
        --ref hub/gemma4-26B-A4B-Strip \\
        --out hub/gemma4-26B-A4B-Strip-Dequant
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


# ─── BF16 ↔ f32 ────────────────────────────────────────────────────────────

def _bf16_bytes_to_f32(data: bytes) -> np.ndarray:
    u16 = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


# ─── INT4 dequant ─────────────────────────────────────────────────────────

def _dequant_int4(
    packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
    out_dim: int, in_dim: int,
) -> np.ndarray:
    """Match dtype.rs's int4_to_f32: row-major [out_dim, in_dim], group=64."""
    num_groups = in_dim // GROUP_SIZE
    words_per_row = in_dim // 8
    scales_f32 = (scales.astype(np.uint32) << 16).view(np.float32)
    biases_f32 = (biases.astype(np.uint32) << 16).view(np.float32)
    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    for row in range(out_dim):
        w_row = packed[row * words_per_row:(row + 1) * words_per_row]
        s_row = scales_f32[row * num_groups:(row + 1) * num_groups]
        b_row = biases_f32[row * num_groups:(row + 1) * num_groups]
        for g in range(num_groups):
            scale = float(s_row[g])
            bias = float(b_row[g])
            base = g * GROUP_SIZE
            for p in range(8):
                word = int(w_row[g * 8 + p])
                for n in range(8):
                    nibble = (word >> (n * 4)) & 0xF
                    result[row, base + p * 8 + n] = float(nibble) * scale + bias
    return result


def _dequant_expert_section(
    data: memoryview, offset: int, out_dim: int, in_dim: int,
) -> np.ndarray:
    num_groups = in_dim // GROUP_SIZE
    words_per_row = in_dim // 8
    w_bytes = out_dim * words_per_row * 4
    sb_bytes = out_dim * num_groups * 2
    packed = np.frombuffer(data[offset:offset + w_bytes], dtype=np.uint32).copy()
    scales = np.frombuffer(data[offset + w_bytes:offset + w_bytes + sb_bytes], dtype=np.uint16).copy()
    biases = np.frombuffer(
        data[offset + w_bytes + sb_bytes:offset + w_bytes + 2 * sb_bytes],
        dtype=np.uint16,
    ).copy()
    return _dequant_int4(packed, scales, biases, out_dim, in_dim)


def _expert_layout(hd: int, mi: int) -> dict[str, int]:
    """Per-expert byte layout (matches src/engine/gemma4_moe/constants.rs)."""
    gs = GROUP_SIZE
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


# ─── Engine → HF name map ─────────────────────────────────────────────────

def _engine_to_hf_name(engine_name: str) -> str:
    """`language_model.model.X` → `model.language_model.X`."""
    if engine_name.startswith("language_model.model."):
        return "model.language_model." + engine_name[len("language_model.model."):]
    if engine_name.startswith("language_model."):
        return "model.language_model." + engine_name[len("language_model."):]
    return engine_name


# ─── Reference shape lookup ────────────────────────────────────────────────

def _read_ref_shapes(ref_dir: str) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    idx_path = os.path.join(ref_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        shard_files = sorted(set(idx["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]
    for shard in shard_files:
        sf_path = os.path.join(ref_dir, shard)
        if not os.path.exists(sf_path):
            continue
        with open(sf_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode())
        for name, info in header.items():
            if name != "__metadata__":
                shapes[name] = tuple(info["shape"])
    return shapes


# ─── Main ─────────────────────────────────────────────────────────────────

def dequantize(quant_dir: str, *, ref: str, out: str | None = None) -> Path:
    """Dequantize a Gemma 4 BQ4 model back to HF safetensors format.

    Parameters
    ----------
    quant_dir : str
        Path to BQ4 dir (model_weights.bin + json + packed_experts/).
    ref : str
        Path to the original (unquantized) HF model for shape reference.
    out : str or None
        Output dir. Defaults to ``{quant_dir}-Dequant``.
    """
    import safetensors.torch
    import torch

    qdir = Path(quant_dir)
    ref_dir = Path(ref)
    if out is None:
        out = f"{quant_dir}-Dequant"
    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)

    # ── 1. BQ4 manifest ──────────────────────────────────────────────────
    with open(qdir / "model_weights.json") as f:
        manifest = json.load(f)
    cfg = manifest["config"]
    hd = cfg["hidden_size"]
    mi = cfg["moe_intermediate_size"]
    num_layers = cfg["num_hidden_layers"]
    num_experts = cfg["num_experts"]

    print(f"[dequant-gemma4] {qdir} → {dst}")
    print(f"  hidden={hd} layers={num_layers} experts={num_experts} moe_inter={mi}")

    ref_shapes = _read_ref_shapes(str(ref_dir))
    print(f"  reference: {len(ref_shapes)} tensors from {ref_dir}")

    # ── 2. Read model_weights.bin ────────────────────────────────────────
    with open(qdir / "model_weights.bin", "rb") as f:
        bin_data = memoryview(f.read())

    tensors_manifest: dict[str, dict] = manifest["tensors"]

    # ── 3. Classify primary vs companion (.scales/.biases) ──────────────
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
    print(f"  {len(primary_names)} primary tensors")

    # ── 4. Dequant each primary ────────────────────────────────────────
    kept: dict[str, torch.Tensor] = {}
    for tname in primary_names:
        info = tensors_manifest[tname]
        offset = info["offset"]
        size = info["size"]
        dtype_str = info["dtype"]
        data = bin_data[offset:offset + size]
        hf_name = _engine_to_hf_name(tname)

        if dtype_str == "bf16":
            arr = _bf16_bytes_to_f32(bytes(data))
        elif dtype_str == "u32":
            base = tname[: -len(".weight")] if tname.endswith(".weight") else tname
            s_name = base + ".scales"
            b_name = base + ".biases"
            if s_name not in tensors_manifest or b_name not in tensors_manifest:
                print(f"  SKIP {tname}: missing scales/biases")
                continue
            si = tensors_manifest[s_name]
            bi = tensors_manifest[b_name]
            packed = np.frombuffer(data, dtype=np.uint32).copy()
            scales = np.frombuffer(
                bytes(bin_data[si["offset"]:si["offset"] + si["size"]]),
                dtype=np.uint16,
            ).copy()
            biases = np.frombuffer(
                bytes(bin_data[bi["offset"]:bi["offset"] + bi["size"]]),
                dtype=np.uint16,
            ).copy()
            out_dim = int(si["shape"][0])
            in_dim = int(si["shape"][1]) * GROUP_SIZE
            arr = _dequant_int4(packed, scales, biases, out_dim, in_dim).reshape(-1)
            # Trim padding if reference has shorter in_dim.
            if hf_name in ref_shapes:
                rs = tuple(int(d) for d in ref_shapes[hf_name])
                if len(rs) == 2 and rs[0] == out_dim and rs[1] < in_dim:
                    arr = arr.reshape(out_dim, in_dim)[:, : rs[1]].reshape(-1)
        else:
            print(f"  SKIP {tname}: unknown dtype {dtype_str}")
            continue

        # Reshape to reference shape if known.
        if hf_name in ref_shapes:
            target = tuple(int(d) for d in ref_shapes[hf_name])
            if arr.size == int(np.prod(target)):
                arr = arr.reshape(target)

        kept[hf_name] = torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)

    # ── 5. Dequant per-layer expert blobs ───────────────────────────────
    layout = _expert_layout(hd, mi)
    expert_dir = qdir / "packed_experts"
    for layer in range(num_layers):
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
            up_w   = _dequant_expert_section(expert_data, base_off + layout["up_w_off"], mi, hd)
            down_w = _dequant_expert_section(expert_data, base_off + layout["down_w_off"], hd, mi)
            gate_up[e, :mi, :] = gate_w
            gate_up[e, mi:, :] = up_w
            down[e] = down_w
        gu_name = f"model.language_model.layers.{layer}.experts.gate_up_proj"
        dn_name = f"model.language_model.layers.{layer}.experts.down_proj"
        kept[gu_name] = torch.from_numpy(gate_up).to(torch.bfloat16)
        kept[dn_name] = torch.from_numpy(down).to(torch.bfloat16)

    # ── 6. Diff against reference ───────────────────────────────────────
    missing = sorted(set(ref_shapes.keys()) - set(kept.keys()))
    extra = sorted(set(kept.keys()) - set(ref_shapes.keys()))
    print(f"  kept={len(kept)} missing={len(missing)} extra={len(extra)}")
    for n in missing[:5]:
        print(f"    missing: {n}")
    for n in extra[:5]:
        print(f"    extra: {n}")

    # ── 7. Write safetensors ────────────────────────────────────────────
    out_weights = dst / "model.safetensors"
    safetensors.torch.save_file(kept, out_weights)
    out_index: dict[str, Any] = {
        "metadata": {"total_size": os.path.getsize(out_weights)},
        "weight_map": {name: "model.safetensors" for name in kept},
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(out_index, f, indent=2)

    # ── 8. Copy + adjust config/tokenizer ───────────────────────────────
    copy_files = (
        "tokenizer.json", "tokenizer_config.json",
        "generation_config.json", "processor_config.json",
        "chat_template.jinja",
    )
    for fname in copy_files:
        src = ref_dir / fname
        if src.exists():
            shutil.copy2(src, dst / fname)

    # config.json — keep ref's structure but drop the _Stripped marker so
    # MLX-VLM accepts the model directly.
    with open(ref_dir / "config.json") as f:
        cfg_json = json.load(f)
    archs = cfg_json.get("architectures", [])
    cfg_json["architectures"] = [
        a.removesuffix("_Stripped") if a.endswith("_Stripped") else a
        for a in archs
    ]
    with open(dst / "config.json", "w") as f:
        json.dump(cfg_json, f, indent=2)

    total_mb = os.path.getsize(out_weights) / 1e6
    print(f"  wrote {total_mb:.0f} MB → {out_weights}")
    return dst


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("quant_dir", help="BQ4 dir (model_weights.bin + packed_experts/)")
    p.add_argument("--ref", required=True, help="Reference HF model dir (for shapes)")
    p.add_argument("--out", default=None, help="Output dir (default: <quant>-Dequant)")
    args = p.parse_args()
    dequantize(args.quant_dir, ref=args.ref, out=args.out)


if __name__ == "__main__":
    _main()
