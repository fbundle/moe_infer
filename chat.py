#!/usr/bin/env python3
"""Interactive chat demo — thin CLI wrapper around moe_infer.Pipeline."""

import argparse

from moe_infer import record_engine_telemetry
from moe_infer.qwen35_moe.pipeline import Qwen35MoEPipeline


def main() -> None:
    default_hub = "hub/models--Qwen--Qwen3.6-35B-A3B"
    default_model = "data/models--Qwen--Qwen3.6-35B-A3B-bq4"

    parser = argparse.ArgumentParser(description="MoE-Infer interactive chat")
    parser.add_argument(
        "--hub", default=default_hub,
        help=f"Path to HF hub (tokenizer + vision). Default: {default_hub}",
    )
    parser.add_argument(
        "--model", default=default_model,
        help=f"Path to quantized model. Default: {default_model}",
    )
    parser.add_argument(
        "--mode", default="Qwen35MoEBq4Exp2",
        choices=["Qwen35MoEBq4Exp1", "Qwen35MoEBq4Exp2"],
        help="Pipeline mode",
    )
    parser.add_argument(
        "--k", type=int, default=0,
        help="Active experts per token (0 = model default)",
    )
    parser.add_argument(
        "--telemetry", action="store_true",
        help="Enable per-layer GPU timing",
    )
    args = parser.parse_args()

    if args.telemetry:
        record_engine_telemetry(True)

    pipe = Qwen35MoEPipeline(
        args.model,
        hub=args.hub,
        mode=args.mode,
        k=0 if args.k == 0 else args.k,
    )

    while True:
        try:
            message = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        for token in pipe.chat(message, stream=True):  # type: ignore[union-attr]
            print(token, end="", flush=True)
        print()

        if args.telemetry:
            print()
            print(pipe.telemetry)


if __name__ == "__main__":
    main()
