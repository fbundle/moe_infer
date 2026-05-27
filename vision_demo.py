#!/usr/bin/env python3
"""Vision demo — thin CLI wrapper around moe_infer.Pipeline."""

import argparse

from moe_infer.qwen35_moe.pipeline import Qwen35MoEPipeline


def main() -> None:
    default_hub = "hub/models--Qwen--Qwen3.6-35B-A3B"
    default_model = "data/models--Qwen--Qwen3.6-35B-A3B-bq4"

    parser = argparse.ArgumentParser(description="Qwen3.6-35B-A3B vision demo")
    parser.add_argument("--image", default="data/crycat-crying-cat.gif")
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--hub", default=default_hub)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--min-pixels", type=int, default=0,
        help="Lower bound on image pixels (0 = default: 65536)",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=0,
        help="Upper bound on image pixels (0 = default: 16777216)",
    )
    args = parser.parse_args()

    pipe = Qwen35MoEPipeline(args.model, hub=args.hub)

    print(f"[demo] Image: {args.image}")
    print(f"[demo] Question: {args.question}")
    print(f"[demo] Generating...")
    print()

    for token in pipe.chat(
        args.question,
        images=[args.image],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        min_image_pixels=args.min_pixels,
        max_image_pixels=args.max_pixels,
        stream=True,
    ):  # type: ignore[union-attr]
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
