"""Strip a Gemma 4 26B-A4B MoE model to a small variant for validation.

Output: 4 decoder layers (3 sliding-attention + 1 full-attention,
matching the original layer-type pattern at indices 0..3 → 5) and 4
experts per layer (down from 128). Vision-tower and multimodal
projector tensors are dropped.

Strip dimensions match Qwen's stripper philosophy: tiny enough to load
both reference (MLX-VLM) and our engine on a 16 GB Mac, while keeping
ALL the architecturally interesting machinery (sliding + full attention,
dual-FFN, experts, partial RoPE on full layer).

Usage::

    python -m moe_infer.gemma4_moe.strip hub/models--google--gemma-4-26B-A4B

Output is written to ``<src>-Strip`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────

NUM_LAYERS = 6
NUM_EXPERTS = 4
TOP_K_EXPERTS = 4

# Keep one full attention period (layers 0..5 of the original 26B-A4B).
# Indices 0..4 are sliding-attention, index 5 is full-attention. This
# matches the engine's `is_full_attn_layer` default pattern check
# (period=6, full_at_index=5) without needing renumbering.
KEEP_SRC_LAYER_IDX = [0, 1, 2, 3, 4, 5]

# Tensors whose first axis is the expert count.
_EXPERT_TENSORS = (
    ".experts.gate_up_proj",   # [E, 2*moe_inter, hidden]
    ".experts.down_proj",      # [E, hidden, moe_inter]
    ".router.proj.weight",     # [E, hidden]
    ".router.per_expert_scale",  # [E]
)


def _is_vision(name: str) -> bool:
    return (
        "vision_tower" in name
        or "embed_vision" in name
        or "multimodal_projector" in name
    )


def _layer_idx(name: str) -> int | None:
    marker = "model.language_model.layers."
    if marker not in name:
        return None
    rest = name.split(marker, 1)[1]
    digits = rest.split(".")[0]
    try:
        return int(digits)
    except ValueError:
        return None


def _renumber_layer(name: str, new_idx: int) -> str:
    """Replace the layer index in ``model.language_model.layers.N.X`` with new_idx."""
    marker = "model.language_model.layers."
    if marker not in name:
        return name
    prefix, _, rest = name.partition(marker)
    # rest looks like "5.self_attn.q_proj.weight"
    old_idx_str, _, suffix = rest.partition(".")
    return f"{prefix}{marker}{new_idx}.{suffix}"


def strip(src_dir: str, *, out: str | None = None) -> Path:
    """Strip a Gemma 4 26B-A4B model.

    Parameters
    ----------
    src_dir : str
        Path to a local directory containing safetensors and
        ``model.safetensors.index.json``.
    out : str or None
        Output directory. Defaults to ``{src_dir}-Strip``.

    Returns
    -------
    Path
        Output directory path.
    """
    import safetensors.torch
    import torch

    src = Path(src_dir)
    if out is None:
        out = f"{src_dir}-Strip"
    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)

    keep_set = set(KEEP_SRC_LAYER_IDX)
    old_to_new = {old: new for new, old in enumerate(KEEP_SRC_LAYER_IDX)}

    # ── 1. Read index ─────────────────────────────────────────────────────
    with open(src / "model.safetensors.index.json") as f:
        src_index = json.load(f)
    src_weight_map: dict[str, str] = src_index["weight_map"]

    # ── 2. Figure out which shards we need ────────────────────────────────
    def keep(name: str) -> bool:
        if _is_vision(name):
            return False
        lidx = _layer_idx(name)
        if lidx is None:
            return True  # global tensors (embed_tokens, norm) — keep
        return lidx in keep_set

    needed_shards: set[str] = set()
    for name, shard in src_weight_map.items():
        if keep(name):
            needed_shards.add(shard)
    total_kept = sum(1 for n in src_weight_map if keep(n))
    print(f"Stripping {src} → {dst}")
    print(f"  source tensors: {len(src_weight_map)} → keeping {total_kept}")
    print(f"  reading {len(needed_shards)} shards")
    print(f"  layers kept (src→dst): {dict(zip(KEEP_SRC_LAYER_IDX, range(NUM_LAYERS)))}")

    # ── 3. Read + slice + renumber ────────────────────────────────────────
    kept: dict[str, torch.Tensor] = {}
    for shard_name in sorted(needed_shards):
        shard_path = src / shard_name
        with safetensors.torch.safe_open(str(shard_path), framework="pt") as f:
            for name in f.keys():
                if not keep(name):
                    continue
                data = f.get_tensor(name)
                # Expert-axis slice
                if any(name.endswith(s) for s in _EXPERT_TENSORS):
                    data = data[:NUM_EXPERTS].clone()
                # Renumber layer index
                lidx = _layer_idx(name)
                new_name = name
                if lidx is not None:
                    new_name = _renumber_layer(name, old_to_new[lidx])
                kept[new_name] = data
    print(f"  kept {len(kept)} tensors")

    # ── 4. Write safetensors ──────────────────────────────────────────────
    out_weights = dst / "model.safetensors"
    safetensors.torch.save_file(kept, out_weights)

    # ── 5. Write index.json ───────────────────────────────────────────────
    out_index: dict[str, Any] = {
        "metadata": {"total_size": os.path.getsize(out_weights)},
        "weight_map": {name: "model.safetensors" for name in kept},
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(out_index, f, indent=2)

    # ── 6. Update config.json ─────────────────────────────────────────────
    with open(src / "config.json") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    src_layer_types = tc.get("layer_types") or []
    new_layer_types = [src_layer_types[i] for i in KEEP_SRC_LAYER_IDX] if src_layer_types else [
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    ]
    tc["num_hidden_layers"] = NUM_LAYERS
    tc["num_experts"] = NUM_EXPERTS
    tc["top_k_experts"] = TOP_K_EXPERTS
    tc["num_experts_per_tok"] = TOP_K_EXPERTS
    tc["layer_types"] = new_layer_types
    # Drop vision_config so HF doesn't try to load a vision tower.
    cfg.pop("vision_config", None)
    cfg["architectures"] = ["Gemma4ForConditionalGeneration_Stripped"]
    # If the original wrapped vision/text under a multimodal config, rebuild
    # as a pure-text config so transformers loads it as Gemma4TextModel.
    with open(dst / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── 7. Copy tokenizer + chat template ─────────────────────────────────
    # Skip config.json (we wrote our own) and the index (we wrote our own).
    skip = {"config.json", "model.safetensors.index.json"}
    copy_exts = {".json", ".txt", ".jinja"}
    for fname in sorted(os.listdir(src)):
        if fname in skip:
            continue
        if any(fname.endswith(ext) for ext in copy_exts):
            import shutil
            shutil.copy2(src / fname, dst / fname)

    print(f"  wrote {dst.resolve()}")
    return dst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src", help="Source HF model directory")
    p.add_argument("--out", default=None, help="Output directory (default: <src>-Strip)")
    args = p.parse_args()
    strip(args.src, out=args.out)


if __name__ == "__main__":
    main()
