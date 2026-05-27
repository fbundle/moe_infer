#!/usr/bin/env python3
"""Quantize HF BF16 Qwen MoE model → BQ4 format.

Thin CLI wrapper around ``moe_infer.qwen35_moe.bq4_quantize``.

Usage:
    python quantize.py --model hub/models--Qwen--Qwen3.6-35B-A3B --version 3.6 --output data/my-model
"""

import argparse

from moe_infer.qwen35_moe import bq4_quantize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize HF BF16 Qwen MoE model → BQ4 format")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to HuggingFace model directory (BF16 safetensors)')
    parser.add_argument('--output', type=str,
                        default='data/models--Qwen--Qwen3.6-35B-A3B-bq4',
                        help='Output directory')
    parser.add_argument('--version', type=str, required=True,
                        choices=['3.5', '3.6'],
                        help='Qwen generation: 3.5 or 3.6')
    parser.add_argument('--strip', action='store_true',
                        help='Strip to 4 layers × 4 experts for verification')
    args = parser.parse_args()

    strip_layers = 4 if args.strip else 0
    strip_experts = 4 if args.strip else 0

    bq4_quantize(
        args.model,
        args.output,
        version=args.version,
        strip_layers=strip_layers,
        strip_experts=strip_experts,
    )


if __name__ == '__main__':
    main()
