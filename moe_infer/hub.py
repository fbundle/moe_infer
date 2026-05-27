"""Load tokenizers from a HuggingFace model hub.

These are convenience factories — they pull heavy dependencies (transformers)
only when called, so text-only pipelines never import them.
"""

from __future__ import annotations

from moe_infer.qwen35_moe.vision import load_vision_encoder  # noqa: F401


def load_tokenizer(hub_path: str):
    """Load an ``AutoTokenizer`` from a HF hub directory.

    Parameters
    ----------
    hub_path : str
        Path to a directory containing HF config + tokenizer files
        (e.g. ``hub/models--Qwen--Qwen3.6-35B-A3B``).

    Returns
    -------
    transformers.PreTrainedTokenizer
    """
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    return AutoTokenizer.from_pretrained(hub_path)
