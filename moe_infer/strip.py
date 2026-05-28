"""Strip a Qwen3.5/3.6 MoE model to 4 layers (3 linear + 1 full attention)
and 4 experts — operates on HuggingFace safetensors format.

Usage::

    python -m moe_infer.strip hub/models--Qwen--Qwen3.6-35B-A3B
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────

NUM_LAYERS = 4
NUM_EXPERTS = 4
NUM_EXPERTS_PER_TOK = 4
HIDDEN_DIM = 2048
INTERMEDIATE = 512

# ─── Expert-slice rules ───────────────────────────────────────────────────

# Tensors whose first axis is the expert count.
# gate_up_proj: [E, 2*intermediate, hidden] → [4, ...]
# down_proj:    [E, hidden, intermediate]   → [4, ...]
# gate.weight:  [E, hidden]                 → [4, ...]
_EXPERT_TENSORS = {
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
    ".mlp.gate.weight",
}


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


def _keep_tensor(name: str) -> bool:
    if name.startswith("mtp."):
        return False
    if ".visual." in name:
        return False
    lidx = _layer_idx(name)
    return lidx is None or lidx < NUM_LAYERS


def _slice_expert_tensor(name: str, data: "torch.Tensor") -> "torch.Tensor":
    for suffix in _EXPERT_TENSORS:
        if name.endswith(suffix):
            return data[:NUM_EXPERTS].clone()
    return data


def strip(src_dir: str, *, out: str | None = None) -> Path:
    """Strip a Qwen MoE model to 4 layers / 4 experts.

    Parameters
    ----------
    src_dir : str
        Path to a local directory containing safetensors and
        ``model.safetensors.index.json``.
    out : str or None
        Output directory.  Defaults to ``{src_dir}-Strip``.

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

    # ── 1. Read index ─────────────────────────────────────────────────────

    with open(src / "model.safetensors.index.json") as f:
        src_index = json.load(f)

    src_weight_map: dict[str, str] = src_index["weight_map"]

    # ── 2. Find which shards contain tensors we need ──────────────────────

    needed_shards: set[str] = set()
    for name, shard in src_weight_map.items():
        if _keep_tensor(name):
            needed_shards.add(shard)

    total_kept = sum(1 for n in src_weight_map if _keep_tensor(n))
    print(f"Stripping {src} → {dst}")
    print(f"  {len(src_weight_map)} source tensors → keeping {total_kept}")
    print(f"  Reading {len(needed_shards)} shards...")

    # ── 3. Read shards and collect kept tensors ───────────────────────────

    kept: dict[str, torch.Tensor] = {}
    skipped_layers = 0
    skipped_visual = 0

    for shard_name in sorted(needed_shards):
        shard_path = src / shard_name
        with safetensors.torch.safe_open(str(shard_path), framework="pt") as f:
            for name in f.keys():
                if not _keep_tensor(name):
                    lidx = _layer_idx(name)
                    if lidx is not None and lidx >= NUM_LAYERS:
                        skipped_layers += 1
                    elif ".visual." in name:
                        skipped_visual += 1
                    continue

                data = f.get_tensor(name)
                if any(name.endswith(s) for s in _EXPERT_TENSORS):
                    data = _slice_expert_tensor(name, data)

                kept[name] = data

    print(f"  Kept {len(kept)} tensors"
          f" (skipped {skipped_layers} other-layer,"
          f" {skipped_visual} visual)")

    # ── 4. Write stripped safetensors ─────────────────────────────────────

    out_weights = dst / "model.safetensors"
    safetensors.torch.save_file(kept, out_weights)

    # ── 5. Write index.json ───────────────────────────────────────────────

    out_index: dict[str, Any] = {
        "metadata": {"total_size": os.path.getsize(out_weights)},
        "weight_map": {name: "model.safetensors" for name in kept},
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(out_index, f, indent=2)

    # ── 6. Write updated config.json ──────────────────────────────────────

    with open(src / "config.json") as f:
        cfg = json.load(f)

    tc = cfg.get("text_config", cfg)
    tc["num_hidden_layers"] = NUM_LAYERS
    tc["num_experts"] = NUM_EXPERTS
    tc["num_experts_per_tok"] = NUM_EXPERTS_PER_TOK
    tc["full_attention_interval"] = 4
    tc["layer_types"] = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    tc["mtp_num_hidden_layers"] = 0
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]

    with open(dst / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── 7. Copy tokenizer etc. ────────────────────────────────────────────

    copy_exts = {".json", ".txt", ".jinja"}
    for fname in sorted(os.listdir(src)):
        if fname.startswith("model") and fname.endswith(".safetensors"):
            continue
        if fname == "model.safetensors.index.json":
            continue
        if fname == "config.json":
            continue
        p = src / fname
        if p.suffix in copy_exts:
            shutil.copy2(p, dst / fname)

    total_mb = os.path.getsize(out_weights) / 1e6
    print(f"  Wrote {total_mb:.0f} MB to {out_weights}")
    print("Done.")
    return dst


# ─── CLI ───────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip Qwen MoE model to 4 layers / 4 experts")
    parser.add_argument("src", help="Path to HF model directory (local)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: {src}-Strip)")
    args = parser.parse_args()
    strip(args.src, out=args.out)


if __name__ == "__main__":
    _main()
