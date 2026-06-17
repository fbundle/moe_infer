"""Single-request head-to-head: oMLX REST vs raw mlx-lm.

Same MLX model (mlx-community/Qwen3-4B-4bit), same prompt, same decode
budget. We expect oMLX to be at most slightly slower than raw mlx-lm on
a single request — its win is supposed to come from multi-request
batching, prefix-cache reuse, and SSD-cache persistence, none of which a
single hot request exercises. The measurement here pins the *baseline
overhead* of going through oMLX's server vs talking to mlx-lm directly.

Both endpoints are loaded warm; we run a small warmup, then 3 timed
trials of 100-token greedy decode. Reports tok/s (median of trials).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def via_omlx(host: str, port: int, model_id: str, prompt: str, max_tokens: int) -> tuple[float, int]:
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    text = resp.read().decode()
    wall = time.perf_counter() - t0
    j = json.loads(text)
    completion_tokens = j["usage"]["completion_tokens"]
    return wall, completion_tokens


def via_mlx_lm(model_dir: str, prompt: str, max_tokens: int):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(model_dir)
    sampler = make_sampler(0.0)
    # Warmup
    for _ in stream_generate(model, tok, prompt, max_tokens=8, sampler=sampler):
        pass
    t0 = time.perf_counter()
    last = None
    n = 0
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=sampler):
        last = r
        n += 1
    wall = time.perf_counter() - t0
    return wall, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omlx-host", default="127.0.0.1")
    ap.add_argument("--omlx-port", type=int, default=9100)
    ap.add_argument("--omlx-model", default="qwen3-4b")
    ap.add_argument("--mlx-model-dir", default="/Volumes/Hippopotamus/omlx_models/qwen3-4b")
    ap.add_argument("--prompt", default=(
        "Write a Python function that computes the n-th Fibonacci number "
        "recursively. Include a docstring and an example."
    ))
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    # ── oMLX over REST ────────────────────────────────────────────
    print(f"[omlx] warmup ...")
    via_omlx(args.omlx_host, args.omlx_port, args.omlx_model, args.prompt, 8)
    omlx_results = []
    for i in range(args.trials):
        wall, n = via_omlx(args.omlx_host, args.omlx_port, args.omlx_model, args.prompt, args.max_tokens)
        tps = n / wall
        print(f"        trial {i+1}: wall={wall*1000:.0f}ms  n={n}  rate={tps:.2f} tok/s")
        omlx_results.append(tps)

    # ── raw mlx-lm in-process ────────────────────────────────────
    print(f"\n[mlx-lm] loading + warmup ...")
    mlx_results = []
    for i in range(args.trials):
        wall, n = via_mlx_lm(args.mlx_model_dir, args.prompt, args.max_tokens)
        tps = n / wall
        print(f"         trial {i+1}: wall={wall*1000:.0f}ms  n={n}  rate={tps:.2f} tok/s")
        mlx_results.append(tps)

    # ── Summary ─────────────────────────────────────────────────
    om = statistics.median(omlx_results)
    mm = statistics.median(mlx_results)
    print(f"\n{'='*52}")
    print(f"{'engine':<14}{'median tok/s':>16}{'overhead %':>20}")
    print(f"{'-'*52}")
    print(f"{'mlx-lm raw':<14}{mm:>16.2f}")
    print(f"{'oMLX REST':<14}{om:>16.2f}{(mm-om)/mm*100:>19.1f}%")


if __name__ == "__main__":
    main()
