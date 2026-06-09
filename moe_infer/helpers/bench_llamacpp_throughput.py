"""Quick TTFT + decode tok/s probe for a llama.cpp GGUF.

Loads via llama-cpp-python (in-process, Metal offload), runs a few prompts of
varying length with streaming, and prints prefill time + decode rate.

No bench scoring here — this is just a "does the model actually run, and at
what speed?" check before wiring it into bench_axes.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=(
            "hub/models--google--gemma-4-12B-it-qat-q4_0-gguf/snapshots/"
            "f6e7774e6148da3b7f201e42ba37cf084c1db35f/"
            "gemma-4-12b-it-qat-q4_0.gguf"
        ),
    )
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="-1 = offload all layers to Metal")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    # Import after argparse so --help doesn't pay the import cost.
    from llama_cpp import Llama

    print(f"[load] model={args.model}")
    print(f"[load] n_ctx={args.n_ctx}  n_gpu_layers={args.n_gpu_layers}")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        verbose=False,
    )
    print(f"[load] loaded in {time.perf_counter() - t0:.1f}s\n")

    prompts = [
        # short
        ("short (~20 tok)",
         "Write one sentence about why the sky is blue."),
        # medium
        ("medium (~120 tok)",
         "Explain in 4-5 paragraphs the physical mechanism behind "
         "Rayleigh scattering, why it makes the sky appear blue during the "
         "day, why sunsets appear red, and how it differs from Mie scattering. "
         "Then list three practical implications for atmospheric science."),
        # longer prefill
        ("longer (~600 tok)",
         "Here is a passage on quantum mechanics: " + (
             "The Schrödinger equation describes how the quantum state of a "
             "non-relativistic physical system changes with time. It plays a "
             "role for matter waves analogous to the wave equation for light. "
         ) * 8 + "Summarize the passage in three sentences and identify two "
         "key concepts that distinguish it from classical mechanics."),
    ]

    for label, prompt in prompts:
        print(f"=== {label} ===")
        t_start = time.perf_counter()
        first_tok_t: float | None = None
        n_tok = 0
        text_parts: list[str] = []
        usage_prompt_tokens = None

        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            ch = chunk["choices"][0]
            delta = ch.get("delta", {})
            content = delta.get("content")
            if content:
                if first_tok_t is None:
                    first_tok_t = time.perf_counter()
                n_tok += 1
                text_parts.append(content)
            # Try to capture prompt-token count if exposed at the end.
            usage = chunk.get("usage")
            if usage and usage_prompt_tokens is None:
                usage_prompt_tokens = usage.get("prompt_tokens")

        t_end = time.perf_counter()
        ttft = (first_tok_t or t_end) - t_start
        decode = t_end - (first_tok_t or t_end)
        tok_per_s = (n_tok / decode) if decode > 0 else 0.0
        total = t_end - t_start

        # Try to get accurate prompt token count from llm.tokenize
        try:
            prompt_tok_count = len(llm.tokenize(prompt.encode()))
        except Exception:
            prompt_tok_count = usage_prompt_tokens or -1

        print(f"  prompt_tokens ≈ {prompt_tok_count}")
        print(f"  TTFT          = {ttft*1000:.0f} ms")
        print(f"  decode        = {decode:.2f} s for {n_tok} tokens "
              f"= {tok_per_s:.1f} tok/s")
        print(f"  total         = {total:.2f} s")
        snippet = "".join(text_parts).strip().replace("\n", " ")[:120]
        print(f"  preview       = {snippet!r}")
        print()


if __name__ == "__main__":
    main()
