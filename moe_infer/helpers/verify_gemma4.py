"""Verify our Gemma 4 engine against MLX-VLM on the stripped variant.

Strategy:
  1. Load stripped BF16 model via MLX-VLM as the reference.
  2. Load our quantized stripped model via our engine.
  3. Run the SAME token sequence through both, compare logits per token.
  4. Print cosine similarity, top-1 agreement, max abs diff.

Requires the stripped model to fit in 16 GB RAM:
  python -m moe_infer.gemma4_moe.strip <src_hub> --out <strip_hub>
  python -c "import _moe_infer_rs as r; r.gemma4_moe_quantize('<strip_hub>', '<data>/model_bq4')"

Usage:
  python -m moe_infer.helpers.verify_gemma4 \\
      --mlx-dir <strip_hub_with_canonical_arch> \\
      --our-dir <data>/model_bq4 \\
      --prompt 'Hello world'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_mlx_reference(mlx_dir: str, token_ids: list[int]) -> np.ndarray:
    """Run MLX-VLM forward on the stripped model, return logits [N, V].

    Bypasses the multimodal Gemma4 Model wrapper (which always builds a
    vision_tower) and loads LanguageModel directly from the safetensors.
    """
    import json
    from pathlib import Path

    import mlx.core as mx
    from mlx_vlm.models.gemma4.config import TextConfig
    from mlx_vlm.models.gemma4.language import LanguageModel

    mlx_path = Path(mlx_dir)
    cfg_data = json.loads((mlx_path / "config.json").read_text())
    text_cfg_data = cfg_data["text_config"]
    # TextConfig is a dataclass — needs model_type + the text-specific fields.
    text_cfg_data.setdefault("model_type", "gemma4")
    # Filter to only the fields the dataclass accepts.
    import dataclasses
    valid = {f.name for f in dataclasses.fields(TextConfig)}
    text_cfg = TextConfig(**{k: v for k, v in text_cfg_data.items() if k in valid})

    print(f"[mlx-vlm] building LanguageModel(num_hidden_layers={text_cfg.num_hidden_layers})", file=sys.stderr)
    lm = LanguageModel(text_cfg)

    # Load weights — strip the multimodal wrapper prefix and apply the same
    # rename rules mlx-vlm's Model.sanitize does for gemma4. We're loading
    # straight into LanguageModel (no `language_model.` prefix), so HF
    # `model.language_model.X` → `model.X`. Expert rename:
    #   experts.down_proj      → experts.switch_glu.down_proj.weight
    #   experts.gate_up_proj   → experts.switch_glu.{gate,up}_proj.weight
    weights: dict[str, "mx.array"] = {}
    for shard in sorted(mlx_path.glob("model*.safetensors")):
        print(f"[mlx-vlm] loading shard {shard.name}", file=sys.stderr)
        shard_weights = mx.load(str(shard))
        for k, v in shard_weights.items():
            if not k.startswith("model.language_model."):
                continue
            new_key = "model." + k[len("model.language_model."):]
            if new_key.endswith(".experts.down_proj"):
                new_key = new_key.replace(
                    ".experts.down_proj", ".experts.switch_glu.down_proj.weight",
                )
                weights[new_key] = v
                continue
            if new_key.endswith(".experts.gate_up_proj"):
                # HF: [E, 2*moe_inter, hidden] → split to gate / up
                # mlx-vlm: swap last two, split on last, swap back
                v = v.swapaxes(-1, -2)
                mid = v.shape[-1] // 2
                gate_key = new_key.replace(
                    ".experts.gate_up_proj", ".experts.switch_glu.gate_proj.weight",
                )
                up_key = new_key.replace(
                    ".experts.gate_up_proj", ".experts.switch_glu.up_proj.weight",
                )
                weights[gate_key] = v[..., :mid].swapaxes(-1, -2)
                weights[up_key] = v[..., mid:].swapaxes(-1, -2)
                continue
            weights[new_key] = v
    print(f"[mlx-vlm] loading {len(weights)} weight tensors into LanguageModel", file=sys.stderr)
    # Inspect what mlx-vlm expects vs what we have.
    expected = {k for k, _ in lm.parameters().items() if not k.endswith("_module")}
    expected_flat: set[str] = set()
    def _flat(name: str, val: object) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                _flat(f"{name}.{k}", v)
        elif isinstance(val, list):
            for i, v in enumerate(val):
                _flat(f"{name}.{i}", v)
        else:
            expected_flat.add(name)
    for k, v in lm.parameters().items():
        _flat(k, v)
    have = set(weights.keys())
    missing = sorted(expected_flat - have)
    extra = sorted(have - expected_flat)
    print(f"[mlx-vlm]   expected={len(expected_flat)} have={len(have)} missing={len(missing)} extra={len(extra)}", file=sys.stderr)
    if missing[:5]:
        print(f"[mlx-vlm]   missing examples: {missing[:5]}", file=sys.stderr)
    if extra[:5]:
        print(f"[mlx-vlm]   extra examples: {extra[:5]}", file=sys.stderr)
    lm.load_weights(list(weights.items()), strict=False)
    lm.eval()

    ids = mx.array([token_ids], dtype=mx.int32)
    print(f"[mlx-vlm] forward(ids shape={ids.shape})...", file=sys.stderr)
    # Capture per-layer hidden states so we can compare layer by layer.
    capture = list(range(text_cfg.num_hidden_layers))
    out = lm(ids, capture_layer_ids=capture, return_hidden=True)
    logits = out.logits if hasattr(out, "logits") else out
    arr = np.array(logits.astype(mx.float32), copy=True)
    if arr.ndim == 3:
        arr = arr[0]
    # Save hidden states to file so we can compare from outside.
    if out.hidden_states is not None:
        hiddens = [np.array(h.astype(mx.float32), copy=True) for h in out.hidden_states]
        np.savez(
            "/tmp/gemma4_mlx_hidden.npz",
            **{f"layer_{i}": h for i, h in enumerate(hiddens)},
        )
        print(f"[mlx-vlm] saved {len(hiddens)} per-layer hidden states", file=sys.stderr)
    return arr


def _load_our_engine_logits(our_dir: str, token_ids: list[int]) -> np.ndarray:
    """Run our engine's forward on the SAME token ids. Return logits [N, V]."""
    import _moe_infer_rs as r

    print(f"[ours] loading {our_dir}", file=sys.stderr)
    m = r.Model(our_dir)
    e = r.Engine(m, "Gemma4MoEFused", 0)
    c = r.Cache(m)
    arr = np.array(token_ids, dtype=np.int64)
    print(f"[ours] forward(N={len(token_ids)})...", file=sys.stderr)
    logits = e.forward(arr, c)
    return np.asarray(logits)


def _compare(mlx_logits: np.ndarray, our_logits: np.ndarray, label: str = "") -> None:
    """Report per-token cosine similarity, max abs diff, top-1 agreement."""
    assert mlx_logits.shape == our_logits.shape, (
        f"shape mismatch: mlx={mlx_logits.shape} ours={our_logits.shape}"
    )
    N, V = mlx_logits.shape
    print(f"\n── Comparison ({label}) ──")
    print(f"  shape: [{N}, {V}]")
    for i in range(N):
        a = mlx_logits[i].astype(np.float32)
        b = our_logits[i].astype(np.float32)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        mae = float(np.mean(np.abs(a - b)))
        max_diff = float(np.max(np.abs(a - b)))
        top1_a = int(a.argmax())
        top1_b = int(b.argmax())
        agree = "✓" if top1_a == top1_b else "✗"
        print(
            f"  tok={i:2d} cos={cos:.4f} mae={mae:.4f} max_diff={max_diff:.3f} "
            f"top1: mlx={top1_a:6d} ours={top1_b:6d} {agree}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mlx-dir", required=True,
        help="Stripped HF model dir with arch=Gemma4ForConditionalGeneration",
    )
    p.add_argument(
        "--our-dir", required=True,
        help="Our quantized model dir (data/<name>-Strip/model_bq4)",
    )
    p.add_argument("--prompt", default="Hello, my name is")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.mlx_dir)
    ids = tok.encode(args.prompt, add_special_tokens=True)
    print(f"prompt: {args.prompt!r}", file=sys.stderr)
    print(f"token ids: {ids}", file=sys.stderr)

    mlx_logits = _load_mlx_reference(args.mlx_dir, ids)
    our_logits = _load_our_engine_logits(args.our_dir, ids)
    _compare(mlx_logits, our_logits, label=f"N={len(ids)}")


if __name__ == "__main__":
    main()
