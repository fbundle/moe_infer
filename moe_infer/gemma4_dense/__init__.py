"""Gemma 4 12B dense — conversion and quantization.

The 12B variant of Gemma 4 is dense (not MoE) but still multimodal: the
"vision encoder" is a single patch_dense + pos_embedding + 2 LayerNorms,
and the "audio encoder" is literally one linear projection. We keep both
inline (no separate vision_tower) and INT4-quantize all their matmul
weights along with the language model.

Output layout::

    <output>/
    └── model_int4/
        ├── config.json
        ├── model_weights.bin
        └── model_weights.json
"""

from __future__ import annotations

import os as _os
import shutil as _shutil

import _moe_infer_rs as _rs  # type: ignore[import-untyped]


__all__ = ["convert", "extract_tokenizer", "quantize"]


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "config.json",
)


def extract_tokenizer(hub_path: str, output_dir: str) -> None:
    """Copy tokenizer + chat template files from a HF hub to *output_dir*."""
    _os.makedirs(output_dir, exist_ok=True)
    for name in _TOKENIZER_FILES:
        src = _os.path.join(hub_path, name)
        if _os.path.exists(src):
            _shutil.copy2(src, _os.path.join(output_dir, name))


def quantize(model_path: str, output_dir: str) -> None:
    """Quantize a HF Gemma 4 12B dense model.

    All 2D matmul weights → INT4 group=64. Norms / scalars / vision
    pos_embedding → BF16. Vision + audio projections are included (they're
    just matmuls; no separate vision_tower).
    """
    _rs.gemma4_dense_quantize(model_path, output_dir)


def convert(input: str, output: str | None = None) -> None:
    """Full conversion: HF hub → quantized model + tokenizer."""
    hub_path = input.rstrip("/")
    if output is None:
        output = f"data/{_os.path.basename(hub_path)}"

    model_dir = _os.path.join(output, "model_int4")
    print(f"[quantize] int4 → {model_dir}")
    quantize(hub_path, model_dir)

    print(f"[extract] Tokenizer → {output}/tokenizer")
    extract_tokenizer(hub_path, _os.path.join(output, "tokenizer"))

    print(f"\nDone → {output}/")
    print("  model_int4/")
    print("  tokenizer/")
