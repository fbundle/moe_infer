#!/usr/bin/env python3
"""
moe_infer.convert — Convert a HuggingFace Qwen3 MoE model to Flash-MoE format.

Usage:
    python -m moe_infer.convert --model path/to/hf-model --output data

    from moe_infer.convert import convert
    convert("path/to/hf-model", "data")
"""

import argparse
from pathlib import Path

from moe_infer.convert.gen_model_config import generate_json, load_hf_config
from moe_infer.convert.extract_weights import run as extract_weights
from moe_infer.convert.repack_experts_4bit import run as repack_4bit


def convert(model_path: str, output_dir: str = "data"):
    """Convert a HuggingFace Qwen3 MoE model to Flash-MoE format.

    Steps:
      1. Generate model_config.json from HF config.json
      2. Extract non-expert weights into model_weights.bin + manifest
      3. Repack 4-bit routed experts into per-layer binaries
    """
    model = Path(model_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Step 1: model_config.json
    print("=== Step 1/3: model_config.json ===")
    cfg = load_hf_config(str(model))
    generate_json(cfg, str(output))

    # Step 2: extract weights
    print("\n=== Step 2/3: Extract model weights ===")
    extract_weights(str(model), str(output))

    # Step 3: repack experts
    print("\n=== Step 3/3: Repack 4-bit experts ===")
    repack_4bit(str(model), str(output / "packed_experts"))

    print(f"\nDone. Model ready in: {output}/")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF Qwen3 MoE model to Flash-MoE format")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HF model directory")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory (default: data)")
    args = parser.parse_args()
    convert(args.model, args.output)


if __name__ == "__main__":
    main()
