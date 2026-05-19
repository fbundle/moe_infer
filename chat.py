#!/usr/bin/env python3
"""Interactive chat using moe-infer-rs (Rust backend) with streaming output.

Usage:
  uv run chat.py --model data --tokenizer Qwen/Qwen3-8B
"""

import argparse
import readline
import time

from transformers import AutoTokenizer

import moe_infer_rs as _core


def generate_rust(model, cache, prompt_ids, max_tokens, eos_token_id,
                   temperature, top_k, top_p, min_p):
    """Run prefill then Rust-side autoregressive generation. Yields token_ids one at a time."""
    pos = cache.position
    new_ids = prompt_ids[pos:]
    n_prefill = len(new_ids) if new_ids else 1

    if new_ids:
        logits, _ = model.forward(new_ids, cache)
    else:
        logits, _ = model.forward([prompt_ids[-1]], cache)

    # logits is a flat list of [n_tokens * vocab_size] floats
    # Get the last vocab-sized slice and find argmax
    vocab = model.vocab_size
    last_logits = logits[-vocab:] if len(logits) >= vocab else logits
    first_id = max(range(len(last_logits)), key=lambda i: last_logits[i])

    # Rust generate returns all tokens at once — yield one at a time for streaming
    tokens = model.generate(
        first_id, cache, eos_token_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
    )
    yield from tokens


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Flash-MoE (Rust)")
    parser.add_argument("--model", "-m", default="data")
    parser.add_argument("--tokenizer", "-t", default=None)
    parser.add_argument("--max-tokens", "-n", type=int, default=512)
    parser.add_argument("--temperature", "-T", type=float, default=0.6)
    parser.add_argument("--top-k", "-k", type=int, default=0)
    parser.add_argument("--top-p", "-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    args = parser.parse_args()

    tok_path = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    print(f"Loading model from {args.model} ...")
    model = _core.Model(args.model)
    messages = []

    print(f"Ready. {model.num_layers} layers, "
          f"{model.hidden_dim} dim.\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = [int(t) for t in result["input_ids"]]

        cache = _core.Cache(model)
        t0 = time.monotonic()
        ttft = 0.0
        response_ids = []

        for token_id in generate_rust(model, cache, prompt_ids,
                                       args.max_tokens, tok.eos_token_id,
                                       args.temperature, args.top_k,
                                       args.top_p, args.min_p):
            if not response_ids:
                ttft = time.monotonic() - t0
            response_ids.append(token_id)
            print(tok.decode([token_id]), end="", flush=True)

        elapsed = time.monotonic() - t0
        n_tok = len(response_ids)
        tok_s = n_tok / elapsed if elapsed > 0 else 0

        response_text = tok.decode(response_ids, skip_special_tokens=False)
        messages.append({"role": "assistant", "content": response_text})

        print(f"\n[{n_tok} tokens, {tok_s:.1f} tok/s, "
              f"TTFT {ttft:.2f}s]\n")


if __name__ == "__main__":
    main()
