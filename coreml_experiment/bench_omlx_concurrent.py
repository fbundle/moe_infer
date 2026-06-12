"""Concurrent N-request test: oMLX (continuous batching) vs raw mlx-lm (sequential).

The whole point of an inference server is multiplexing concurrent
requests. mlx-lm in-process can only serve them sequentially (single
GPU stream). oMLX uses mlx-lm's BatchGenerator to fold concurrent
requests into a single batched forward pass.

We fire N requests, slightly different prompts (so the prefix cache
doesn't dominate the picture — we want to isolate the batching win).
Wall-clock time to receive all N responses, total tokens generated,
and effective aggregate tok/s.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


PROMPTS = [
    "Write a Python function that computes the n-th Fibonacci number recursively.",
    "Explain the difference between TCP and UDP in two short paragraphs.",
    "Summarize the plot of Hamlet in five sentences.",
    "List five practical applications of graph theory.",
    "Compare and contrast supervised vs unsupervised learning briefly.",
    "Describe how a hash table handles collisions, in three sentences.",
    "Give three concrete examples of dynamic programming problems.",
    "What is the difference between mutexes and semaphores? Be brief.",
]


def one_omlx(host: str, port: int, model_id: str, prompt: str, max_tokens: int):
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
    resp = urllib.request.urlopen(req, timeout=300)
    text = resp.read().decode()
    wall = time.perf_counter() - t0
    j = json.loads(text)
    return wall, j["usage"]["completion_tokens"]


def via_mlx_sequential(model_dir: str, prompts: list[str], max_tokens: int):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(model_dir)
    sampler = make_sampler(0.0)
    # warmup
    for _ in stream_generate(model, tok, prompts[0], max_tokens=8, sampler=sampler): pass

    t0 = time.perf_counter()
    total_tokens = 0
    for p in prompts:
        n = 0
        for _ in stream_generate(model, tok, p, max_tokens=max_tokens, sampler=sampler):
            n += 1
        total_tokens += n
    wall = time.perf_counter() - t0
    return wall, total_tokens


def via_omlx_concurrent(host: str, port: int, model_id: str, prompts: list[str], max_tokens: int):
    # Warmup with a single request
    one_omlx(host, port, model_id, prompts[0], 8)

    t0 = time.perf_counter()
    total_tokens = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futs = [ex.submit(one_omlx, host, port, model_id, p, max_tokens) for p in prompts]
        for f in concurrent.futures.as_completed(futs):
            wall_i, n_i = f.result()
            total_tokens += n_i
    wall = time.perf_counter() - t0
    return wall, total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--omlx-port", type=int, default=9100)
    ap.add_argument("--omlx-model", default="qwen3-4b")
    ap.add_argument("--mlx-model-dir", default="/Volumes/Hippopotamus/omlx_models/qwen3-4b")
    args = ap.parse_args()

    prompts = PROMPTS[:args.n]
    print(f"[setup] N={args.n} concurrent requests, max_tokens={args.max_tokens}, model=qwen3-4b\n")

    print(f"[mlx-lm sequential] N requests one after another ...")
    wall_seq, n_seq = via_mlx_sequential(args.mlx_model_dir, prompts, args.max_tokens)
    print(f"  total wall={wall_seq*1000:.0f}ms  tokens={n_seq}  agg rate={n_seq/wall_seq:.2f} tok/s")

    print(f"\n[oMLX concurrent] all N requests fired at once ...")
    wall_par, n_par = via_omlx_concurrent("127.0.0.1", args.omlx_port, args.omlx_model, prompts, args.max_tokens)
    print(f"  total wall={wall_par*1000:.0f}ms  tokens={n_par}  agg rate={n_par/wall_par:.2f} tok/s")

    speedup = (n_par / wall_par) / (n_seq / wall_seq)
    print(f"\n{'='*52}")
    print(f"{'aggregate speedup (oMLX/mlx-lm seq)':<38}{speedup:>10.2f}x")


if __name__ == "__main__":
    main()
