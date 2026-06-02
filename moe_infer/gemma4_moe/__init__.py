"""Gemma 4 26B-A4B MoE — conversion, extraction, and pipeline."""

from __future__ import annotations

import os as _os

import _moe_infer_rs as _rs  # type: ignore[import-untyped]

from moe_infer.gemma4_moe.pipeline import Gemma4MoEPipeline

__all__ = ["Gemma4MoEPipeline", "convert", "extract_tokenizer", "quantize"]


# ── Default chat template ──────────────────────────────────────────────────
#
# google/gemma-4-26B-A4B ships WITHOUT a chat_template.jinja and the
# tokenizer's `chat_template` field is unset. We provide a default based on
# the model's special tokens:
#   <|turn>     (id 105)  — start of turn (sot)
#   <turn|>     (id 106)  — end of turn  (eot)
#   <|channel>  (id 100)  — start of channel (role marker)
#   <channel|>  (id 101)  — end of channel
#
# The thought channel is `<|channel>thought\n...<channel|>` and the
# response follows on the same turn. This template generates:
#
#   <bos><|turn><|channel>user<channel|>
#   {user_message}<turn|>
#   <|turn><|channel>model<channel|>
#
# For multi-turn, only the FIRST message gets <bos>.
#
_GEMMA4_CHAT_TEMPLATE = """{% for message in messages -%}
{% if loop.first and message['role'] != 'system' -%}<bos>{% endif -%}
<|turn><|channel>{{ message['role'] }}<channel|>
{% if message['content'] is string -%}
{{ message['content'] }}
{%- else -%}
{% for item in message['content'] -%}
{% if item['type'] == 'text' -%}{{ item['text'] }}{% endif -%}
{% endfor -%}
{%- endif -%}
<turn|>
{% endfor -%}
{% if add_generation_prompt -%}
<|turn><|channel>model<channel|>
{% endif -%}"""


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "chat_template.json",
    "chat_template.jinja",
)


def extract_tokenizer(hub_path: str, output_dir: str) -> None:
    """Copy tokenizer files from a HF hub to *output_dir*.

    If the source has no chat template, write our default one.
    """
    import json
    import shutil

    _os.makedirs(output_dir, exist_ok=True)
    for name in _TOKENIZER_FILES:
        src = _os.path.join(hub_path, name)
        if _os.path.exists(src):
            shutil.copy2(src, _os.path.join(output_dir, name))

    # Provide default chat_template if neither .json nor .jinja exists AND
    # tokenizer_config.json doesn't already have a chat_template field.
    has_jinja = _os.path.exists(_os.path.join(output_dir, "chat_template.jinja"))
    has_json = _os.path.exists(_os.path.join(output_dir, "chat_template.json"))
    has_in_config = False
    cfg_path = _os.path.join(output_dir, "tokenizer_config.json")
    if _os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                tc = json.load(f)
            has_in_config = bool(tc.get("chat_template"))
        except (OSError, json.JSONDecodeError):
            pass
    if not (has_jinja or has_json or has_in_config):
        with open(_os.path.join(output_dir, "chat_template.jinja"), "w") as f:
            f.write(_GEMMA4_CHAT_TEMPLATE)
        print(
            f"[extract] No upstream chat template — wrote default Gemma 4 "
            f"template to {output_dir}/chat_template.jinja",
            flush=True,
        )


def quantize(model_path: str, output_dir: str) -> None:
    """Quantize a HF Gemma 4 26B-A4B model.

    INT4 group=64 for all matmul weights (attention, MLP, router, experts,
    embeddings). BF16 for norms, scalars, and full-layer o_proj. Per-layer
    expert blobs go to ``packed_experts/layer_XX.bin`` so the non-expert
    weight file stays under Apple Silicon's per-buffer length cap.
    """
    _rs.gemma4_moe_quantize(model_path, output_dir)


def convert(
    input: str,
    output: str | None = None,
) -> None:
    """Full conversion: HF hub → quantized model + tokenizer.

    Parameters
    ----------
    input : str
        Path to the HF hub directory.
    output : str or None
        Output root. Defaults to ``data/<hub-basename>``.

    Result::

        <output>/
        ├── model_bq4/
        │   ├── config.json
        │   ├── model_weights.bin     (~1.5 GB, non-experts)
        │   ├── model_weights.json
        │   └── packed_experts/
        │       └── layer_XX.bin      (~408 MB × 30 layers)
        └── tokenizer/
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── chat_template.jinja   (default if not in source)
            └── ...
    """
    hub_path = input.rstrip("/")
    if output is None:
        output = f"data/{_os.path.basename(hub_path)}"

    model_dir = _os.path.join(output, "model_bq4")
    print(f"[quantize] bq4 → {model_dir}")
    quantize(hub_path, model_dir)

    print(f"[extract] Tokenizer → {output}/tokenizer")
    extract_tokenizer(hub_path, _os.path.join(output, "tokenizer"))

    print(f"\nDone → {output}/")
    print("  model_bq4/")
    print("  tokenizer/")
