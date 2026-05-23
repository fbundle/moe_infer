#!/usr/bin/env python3
"""Interactive chat demo for MoE-Infer."""

import argparse
import numpy as np
from transformers import AutoTokenizer
from moe_infer import Model, Engine, Cache, record_engine_telemetry  # type: ignore


# Maps model family names to response-extraction functions. Each function
# receives the raw decoded completion string and returns the final response.
# Add entries here when supporting new model families.
_RESPONSE_EXTRACTORS = {
    "qwen3": lambda c: c.removesuffix("<|im_end|>").split("</think>")[-1],
    "raw": lambda c: c,
}


def _get_response_extractor(model_name: str):
    """Pick the right response stripper based on a lowercase model-name hint."""
    name = model_name.lower()
    if "qwen" in name:
        return _RESPONSE_EXTRACTORS["qwen3"]
    return _RESPONSE_EXTRACTORS["raw"]


class Conversation:
    def __init__(self, tokenizer_path: str, model_path: str,
                 response_extractor, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = Model(model_path)
        self.engine = Engine(self.model, **kwargs)
        self.cache = Cache(self.model)
        self.extract_response = response_extractor
        self.messages: list[dict] = []

    def chat(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})

        input_ids = np.array(
            self.tokenizer.apply_chat_template(
                self.messages, add_generation_prompt=True,
                enable_thinking=False,
            ).input_ids,
            dtype=np.int64,
        )[self.cache.pos:]

        completion = ""
        completion_ids: list[int] = []

        for token, logits in self.engine.stream_generate(input_ids, self.cache):
            completion_ids.append(token)
            new_completion = self.tokenizer.decode(completion_ids)
            addon = new_completion[len(completion):]
            print(addon, end="", flush=True)
            completion = new_completion

        response = self.extract_response(completion)
        self.messages.append({"role": "assistant", "content": response})
        return response


def main():
    default_tokenizer = "hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
    default_model = "data/models--mlx-community--Qwen3.6-35B-A3B-4bit"

    parser = argparse.ArgumentParser(description="MoE-Infer interactive chat")
    parser.add_argument(
        "--tokenizer", default=default_tokenizer,
        help=f"Path to HF tokenizer (default: {default_tokenizer})",
    )
    parser.add_argument(
        "--model", default=default_model,
        help=f"Path to MoE-Infer model directory (default: {default_model})",
    )
    parser.add_argument(
        "--mode", default="FusedWoods",
        choices=["Cpu", "FusedExp", "FusedWoods"],
        help="Pipeline mode",
    )
    parser.add_argument(
        "--k", type=int, default=4,
        help="Active experts per token (0 = model default)",
    )
    parser.add_argument(
        "--telemetry", action="store_true",
        help="Enable per-layer timing telemetry",
    )
    args = parser.parse_args()

    record_engine_telemetry(args.k != 0 and args.telemetry)
    extractor = _get_response_extractor(args.model)
    k = 0 if args.k == 0 else args.k
    conv = Conversation(args.tokenizer, args.model, extractor,
                        pipeline_mode=args.mode, k=k)

    while True:
        try:
            message = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        conv.chat(message)
        print()
        if args.telemetry:
            print(conv.engine.telemetry())


if __name__ == "__main__":
    main()
