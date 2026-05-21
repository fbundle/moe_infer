#!/usr/bin/env python3
"""Benchmark the C inference engine — mirrors bench.rs exactly.
   Same prompt, same 100 tokens, same greedy argmax sampling."""
import sys
import time
import moe_infer.core as core

MODEL_PATH = "/Volumes/Hippopotamus/vault/code/flash-moe/data/models--mlx-community--Qwen3.5-35B-A3B-4bit"
PROMPT = "Hello, how are you?"
NUM_TOKENS = 100
EOS_1 = 248046
EOS_2 = 248044

def main():
    print(f"[bench_c] Initializing model from {MODEL_PATH}")
    t_init = time.time()
    model = core.init(MODEL_PATH)
    cache = core.cache_new(model)
    print(f"[bench_c] Init: { (time.time() - t_init) * 1000:.0f} ms")

    # Tokenize prompt using the Rust tokenizer (via model's BPE)
    # We need to tokenize. Import the Rust tokenizer via Python calling the .so
    from tokenizers import Tokenizer
    from pathlib import Path

    tok = Tokenizer.from_file(str(Path(MODEL_PATH) / "tokenizer.json"))
    prompt_str = (
        f"<|im_start|>system\nYou are a helpful assistant. /think<|im_end|>\n"
        f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    encoded = tok.encode(prompt_str)
    prompt_ids = encoded.ids
    print(f"[bench_c] Prompt tokens: {len(prompt_ids)}, target gen: {NUM_TOKENS}")

    # Prefill: process all prompt tokens through forward()
    print(f"[bench_c] Prefilling {len(prompt_ids)} tokens...")
    t_prefill = time.time()
    logits, cache = core.forward(prompt_ids, model, cache)
    prefill_ms = (time.time() - t_prefill) * 1000
    print(f"[bench_c] Prefill: {prefill_ms:.0f} ms ({len(prompt_ids)} tokens)")

    # Get first token from last logit (greedy argmax)
    import numpy as np
    last_logits = logits[-1, :]
    first_token = int(np.argmax(last_logits))
    print(f"[bench_c] First token: {first_token}")

    # Generation: use core.generate() — C-side autoregressive loop
    # Set temperature=0 for greedy argmax (matching Rust's argmax)
    print(f"[bench_c] Generating {NUM_TOKENS} tokens...")
    t_gen_start = time.time()
    gen_count = 0
    output_tokens = []
    for token_id in core.generate(first_token, model, cache,
                                    NUM_TOKENS, EOS_1,
                                    0.0,   # temperature=0 → greedy
                                    1,     # top_k=1
                                    1.0,   # top_p=1.0 (disabled)
                                    0.0):  # min_p=0.0 (disabled)
        output_tokens.append(token_id)
        gen_count += 1
        if token_id == EOS_1 or token_id == EOS_2:
            break

    gen_elapsed_ms = (time.time() - t_gen_start) * 1000
    tok_s = gen_count * 1000.0 / gen_elapsed_ms if gen_count > 0 else 0.0

    # Decode output
    decoded = tok.decode(output_tokens)

    print(f"\n[bench_c] Results:")
    print(f"[bench_c]   Generated: {gen_count} tokens")
    print(f"[bench_c]   Time:      {gen_elapsed_ms:.0f} ms")
    print(f"[bench_c]   Speed:     {tok_s:.2f} tok/s")
    print(f"[bench_c]   Output:    {decoded[:200]}")

    core.cache_free(cache)
    core.free_all(model)


if __name__ == "__main__":
    main()
