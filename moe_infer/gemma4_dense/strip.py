"""Strip a Gemma 4 12B dense model to a 6-layer variant for verification.

Keeps the first full attention period (layers 0..5: 5 sliding + 1 full).
Drops vision/audio tensors so the stripped model loads cleanly into
``transformers.AutoModelForCausalLM`` as a pure text LM (for use as a
ground-truth reference vs. our gemma4_dense engine).

Usage::

    python -m moe_infer.gemma4_dense.strip \\
        hub/models--google--gemma-4-12B/snapshots/<sha>

Output is written to ``<src>-Strip`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


NUM_LAYERS = 6
KEEP_SRC_LAYER_IDX = [0, 1, 2, 3, 4, 5]


def _is_multimodal(name: str) -> bool:
    """Drop vision / audio tensors so transformers loads it as text-only."""
    return any(prefix in name for prefix in (
        "vision_embedder", "embed_vision", "embed_audio", "vision_tower",
    ))


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
    marker = "model.language_model.layers."
    if marker not in name:
        return name
    prefix, _, rest = name.partition(marker)
    _, _, suffix = rest.partition(".")
    return f"{prefix}{marker}{new_idx}.{suffix}"


def strip(src_dir: str, *, out: str | None = None) -> Path:
    import safetensors.torch
    import torch  # noqa: F401  (safetensors needs the framework)

    src = Path(src_dir)
    if out is None:
        out = f"{src_dir}-Strip"
    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)

    keep_set = set(KEEP_SRC_LAYER_IDX)
    old_to_new = {old: new for new, old in enumerate(KEEP_SRC_LAYER_IDX)}

    def keep(name: str) -> bool:
        if _is_multimodal(name):
            return False
        lidx = _layer_idx(name)
        if lidx is None:
            return True  # globals: embed_tokens, language_model.model.norm
        return lidx in keep_set

    # ── Single-shard reads from the one model.safetensors ──
    shard_path = src / "model.safetensors"
    if not shard_path.exists():
        # Multi-shard fallback (shouldn't happen for gemma-4-12B, but cheap).
        idx_path = src / "model.safetensors.index.json"
        with open(idx_path) as f:
            src_index = json.load(f)
        src_shards = sorted({s for s in src_index["weight_map"].values()})
    else:
        src_shards = ["model.safetensors"]

    print(f"Stripping {src} → {dst}")
    print(f"  layers kept (src→dst): "
          f"{dict(zip(KEEP_SRC_LAYER_IDX, range(NUM_LAYERS)))}")
    print(f"  reading {len(src_shards)} shard(s)")

    kept: dict[str, Any] = {}
    for shard_name in src_shards:
        sp = src / shard_name
        with safetensors.torch.safe_open(str(sp), framework="pt") as f:
            for name in f.keys():
                if not keep(name):
                    continue
                data = f.get_tensor(name)
                lidx = _layer_idx(name)
                new_name = name
                if lidx is not None:
                    new_name = _renumber_layer(name, old_to_new[lidx])
                kept[new_name] = data
    print(f"  kept {len(kept)} tensors")

    # ── Write safetensors + index ──
    out_weights = dst / "model.safetensors"
    safetensors.torch.save_file(kept, out_weights)
    out_index: dict[str, Any] = {
        "metadata": {"total_size": os.path.getsize(out_weights)},
        "weight_map": {name: "model.safetensors" for name in kept},
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(out_index, f, indent=2)

    # ── Update config.json ──
    with open(src / "config.json") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    src_layer_types = tc.get("layer_types") or []
    new_layer_types = (
        [src_layer_types[i] for i in KEEP_SRC_LAYER_IDX]
        if src_layer_types
        else ["sliding_attention"] * 5 + ["full_attention"]
    )
    tc["num_hidden_layers"] = NUM_LAYERS
    tc["layer_types"] = new_layer_types

    # Drop vision/audio config so transformers loads as text-only.
    for key in ("vision_config", "audio_config"):
        cfg.pop(key, None)
    # Keep the ORIGINAL arch so HF transformers can load this directly as the
    # reference model. Our engine's verify script patches the architecture to
    # `..._Stripped` at runtime to dispatch to the small ModelConfig.
    # cfg["architectures"] is left as the source value (Gemma4UnifiedForConditionalGeneration).

    with open(dst / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Copy tokenizer / chat template ──
    skip = {"config.json", "model.safetensors", "model.safetensors.index.json"}
    copy_exts = {".json", ".txt", ".jinja"}
    for fname in sorted(os.listdir(src)):
        if fname in skip:
            continue
        if any(fname.endswith(ext) for ext in copy_exts):
            shutil.copy2(src / fname, dst / fname)

    print(f"  wrote {dst.resolve()}")
    return dst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src", help="Source HF model directory (snapshot path)")
    p.add_argument("--out", default=None,
                   help="Output directory (default: <src>-Strip)")
    args = p.parse_args()
    strip(args.src, out=args.out)


if __name__ == "__main__":
    main()
