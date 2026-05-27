"""Qwen3.5/3.6 vision encoder loader."""

from __future__ import annotations

import json


def load_vision_encoder(hub_path: str):
    """Load the Qwen3.6 vision encoder from a HF hub.

    Only ``model.visual.*`` weights are loaded; language-model shards
    are skipped.  Returns the encoder in eval mode.

    Requires ``transformers``, ``torch``, and ``safetensors``.
    """
    import time

    import torch
    from safetensors.torch import load_file as load_safetensors
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeVisionConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeVisionModel,
    )

    t0 = time.time()

    with open(f"{hub_path}/model.safetensors.index.json") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    vis_shards = sorted(
        {sn for k, sn in weight_map.items() if k.startswith("model.visual.")}
    )

    state: dict[str, torch.Tensor] = {}
    for shard_name in vis_shards:
        for key, tensor in load_safetensors(f"{hub_path}/{shard_name}").items():
            if key.startswith("model.visual."):
                state[key.removeprefix("model.visual.")] = tensor

    cfg = Qwen3_5MoeVisionConfig.from_pretrained(hub_path)
    model = Qwen3_5MoeVisionModel(cfg)
    model.load_state_dict(state, strict=True)
    model.eval()

    dt = time.time() - t0
    print(f"[vision] Loaded {len(state)} tensors in {dt:.1f}s", flush=True)
    return model
