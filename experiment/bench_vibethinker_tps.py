"""Quick tok/s test for VibeThinker-3B on oMLX vs raw mlx-lm.

Why: bench_axes saw 9.7 tok/s through oMLX. A BF16 3B model on M4 should
be closer to 30 tok/s single-stream. Test directly to see if the bottleneck
is oMLX overhead, the BF16 weight format, or VibeThinker's KV-cache layout.
"""
from __future__ import annotations
import json, statistics, sys, time, urllib.request
from pathlib import Path

PROMPT = "Write a Python function that computes the n-th Fibonacci number recursively. Include a docstring and an example."

def via_omlx(host, port, model_id, prompt, max_tokens):
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=600)
    text = resp.read().decode()
    wall = time.perf_counter() - t0
    j = json.loads(text)
    n_tok = j["usage"]["completion_tokens"]
    return wall, n_tok

def via_mlx(model_dir, prompt, max_tokens):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(model_dir)
    sampler = make_sampler(0.0)
    # Warmup
    for _ in stream_generate(model, tok, prompt, max_tokens=8, sampler=sampler):
        pass
    t0 = time.perf_counter()
    n = 0
    for _ in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=sampler):
        n += 1
    wall = time.perf_counter() - t0
    return wall, n

def main():
    mlx_dir = "data/VibeThinker-3B-mlx-bf16"
    max_tokens = 256
    trials = 3

    print(f"[setup] model=VibeThinker-3B BF16  max_tokens={max_tokens}  trials={trials}")
    print()

    print("=== oMLX (port 9100) ===")
    # Warmup
    via_omlx("127.0.0.1", 9100, "vibethinker-3b", PROMPT, 8)
    omlx_rates = []
    for i in range(trials):
        wall, n = via_omlx("127.0.0.1", 9100, "vibethinker-3b", PROMPT, max_tokens)
        rate = n / wall
        print(f"  trial {i+1}: wall={wall*1000:.0f}ms  tokens={n}  rate={rate:.2f} tok/s")
        omlx_rates.append(rate)

    print()
    print("=== mlx-lm raw ===")
    mlx_rates = []
    for i in range(trials):
        wall, n = via_mlx(mlx_dir, PROMPT, max_tokens)
        rate = n / wall
        print(f"  trial {i+1}: wall={wall*1000:.0f}ms  tokens={n}  rate={rate:.2f} tok/s")
        mlx_rates.append(rate)

    print()
    print("=" * 52)
    om = statistics.median(omlx_rates)
    mm = statistics.median(mlx_rates)
    print(f"{'engine':<14}{'median tok/s':>16}{'overhead':>22}")
    print("-" * 52)
    print(f"{'mlx-lm raw':<14}{mm:>16.2f}")
    print(f"{'oMLX REST':<14}{om:>16.2f}{(mm-om)/mm*100:>21.1f}%")


if __name__ == "__main__":
    main()
