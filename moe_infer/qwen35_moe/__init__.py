"""Qwen3.5/3.6 MoE — conversion, extraction, and vision utilities."""

from __future__ import annotations

import os as _os

import _moe_infer_rs as _rs  # type: ignore[import-untyped]


def bq4_extract_tokenizer(hub_path: str, output_dir: str) -> None:
    """Copy tokenizer files from a HF hub to *output_dir*."""
    import shutil

    _TOKENIZER_FILES = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
    ]

    _os.makedirs(output_dir, exist_ok=True)
    for name in _TOKENIZER_FILES:
        src = _os.path.join(hub_path, name)
        if _os.path.exists(src):
            shutil.copy2(src, _os.path.join(output_dir, name))


def bq4_extract_vision(hub_path: str, output_dir: str) -> None:
    """Copy vision-encoder files from a HF hub to *output_dir*."""
    import json
    import shutil

    _os.makedirs(output_dir, exist_ok=True)

    for name in ("config.json", "preprocessor_config.json"):
        src = _os.path.join(hub_path, name)
        if _os.path.exists(src):
            shutil.copy2(src, _os.path.join(output_dir, name))

    index_path = _os.path.join(hub_path, "model.safetensors.index.json")
    if not _os.path.exists(index_path):
        return

    with open(index_path) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    vis_shards = sorted(
        {sn for k, sn in weight_map.items() if k.startswith("model.visual.")}
    )

    if not vis_shards:
        return

    shutil.copy2(index_path, _os.path.join(output_dir, "model.safetensors.index.json"))

    for shard_name in vis_shards:
        src = _os.path.join(hub_path, shard_name)
        if _os.path.exists(src):
            shutil.copy2(src, _os.path.join(output_dir, shard_name))

    print(
        f"[extract] Vision: {len(vis_shards)} shard(s) → {output_dir}",
        flush=True,
    )


def bq4_quantize(
    model_path: str,
    output_dir: str,
    *,
    version: str,
    strip_layers: int = 0,
    strip_experts: int = 0,
) -> None:
    """Quantize a HF Qwen3.5-MoE model to BQ4 format.

    Parameters
    ----------
    version : str
        Qwen generation: ``"3.5"`` or ``"3.6"``.
        Qwen3.6 applies a +1.0 norm-weight correction.
    """
    if version not in ("3.5", "3.6"):
        raise ValueError(f"version must be '3.5' or '3.6', got {version!r}")
    _rs.qwen35_moe_bq4_quantize(
        model_path,
        output_dir,
        version=version,
        strip_layers=strip_layers,
        strip_experts=strip_experts,
    )


def bq4_convert(
    input: str,
    output: str | None = None,
    *,
    version: str,
    strip: bool = False,
) -> None:
    """Full conversion: HF hub → model_bq4 + tokenizer + vision_encoder.

    Parameters
    ----------
    input : str
        Path to the HF hub directory.
    output : str or None
        Output root.  Defaults to ``data/<hub-basename>``.
    version : str
        Qwen generation: ``"3.5"`` or ``"3.6"``.
        Qwen3.6 applies a +1.0 norm-weight correction.
    strip : bool
        If True, create a test model with 4 layers × 4 experts.
    """
    hub_path = input.rstrip("/")
    if output is None:
        output = f"data/{_os.path.basename(hub_path)}"

    strip_layers = 4 if strip else 0
    strip_experts = 4 if strip else 0

    print(f"[1/3] Quantizing model → {output}/model_bq4")
    bq4_quantize(
        hub_path, _os.path.join(output, "model_bq4"),
        version=version,
        strip_layers=strip_layers,
        strip_experts=strip_experts,
    )

    print(f"[2/3] Extracting tokenizer → {output}/tokenizer")
    bq4_extract_tokenizer(hub_path, _os.path.join(output, "tokenizer"))

    print(f"[3/3] Extracting vision encoder → {output}/vision_encoder")
    bq4_extract_vision(hub_path, _os.path.join(output, "vision_encoder"))

    print(f"\nDone → {output}/")
    print(f"  model_bq4/      (quantized weights)")
    print(f"  tokenizer/      (AutoTokenizer-ready)")
    print(f"  vision_encoder/ (AutoImageProcessor + visual weights)")
