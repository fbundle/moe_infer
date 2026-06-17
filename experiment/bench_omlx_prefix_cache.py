"""Prefix-cache hit test: send the same long prompt twice to oMLX.

The second request should hit the prefix cache and skip prefill almost
entirely. mlx-lm in-process has no cross-request cache so each call pays
full prefill cost.

We deliberately use a LONG prompt so prefill dominates wall-clock; that
makes the cache win observable. Decode tokens are kept small for the
same reason.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


LONG_PROMPT = (
    "You are an experienced Python engineer. Read the following code review "
    "comments carefully and respond with a structured summary.\n\n"
    + ("Comment thread: discussing whether to refactor the request-batching "
       "logic in the inference server. Key arguments include memory usage, "
       "throughput trade-offs, KV cache management, prefix sharing, and the "
       "operational cost of running many concurrent requests on a single "
       "GPU. The original implementation processes requests one at a time, "
       "which is simple but underutilizes the accelerator. The proposed "
       "change introduces continuous batching at the iteration level, where "
       "active requests are merged into a single forward pass each step, "
       "and finished requests are evicted to make room for queued ones. "
       "Concerns raised: cache invalidation when sequences finish at "
       "different times; memory fragmentation from many small allocations; "
       "the additional complexity of the scheduler. ") * 5
    + "\n\nGiven this context, briefly answer: should we adopt continuous batching?"
)


def one_omlx(host, port, model_id, prompt, max_tokens):
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
    resp = urllib.request.urlopen(req, timeout=300)
    text = resp.read().decode()
    wall = time.perf_counter() - t0
    j = json.loads(text)
    return wall, j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"]


def via_mlx_lm(model_dir, prompt, max_tokens):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(model_dir)
    sampler = make_sampler(0.0)
    # warmup
    for _ in stream_generate(model, tok, prompt, max_tokens=4, sampler=sampler): pass

    t0 = time.perf_counter()
    n = 0
    for _ in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=sampler):
        n += 1
    wall = time.perf_counter() - t0
    return wall, len(tok.encode(prompt)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--mlx-model-dir", default="/Volumes/Hippopotamus/omlx_models/qwen3-4b")
    args = ap.parse_args()

    print(f"[setup] prompt is ~{len(LONG_PROMPT.split())} words long, max_tokens={args.max_tokens}\n")

    print(f"[oMLX] 1st call (cold prefix cache) ...")
    w1, pt1, n1 = one_omlx("127.0.0.1", args.port, args.model, LONG_PROMPT, args.max_tokens)
    print(f"       wall={w1*1000:.0f}ms  prompt_tokens={pt1}  completion_tokens={n1}")

    print(f"[oMLX] 2nd call (warm prefix cache, same prompt) ...")
    w2, pt2, n2 = one_omlx("127.0.0.1", args.port, args.model, LONG_PROMPT, args.max_tokens)
    print(f"       wall={w2*1000:.0f}ms  prompt_tokens={pt2}  completion_tokens={n2}")

    print(f"[oMLX] 3rd call (warm, sanity) ...")
    w3, pt3, n3 = one_omlx("127.0.0.1", args.port, args.model, LONG_PROMPT, args.max_tokens)
    print(f"       wall={w3*1000:.0f}ms  prompt_tokens={pt3}  completion_tokens={n3}")

    print(f"\n[mlx-lm] each call: full prefill (no cross-call cache)")
    wm1, mpt, mn = via_mlx_lm(args.mlx_model_dir, LONG_PROMPT, args.max_tokens)
    print(f"          wall={wm1*1000:.0f}ms  prompt_tokens={mpt}  completion_tokens={mn}")

    print(f"\n{'='*54}")
    print(f"{'engine':<24}{'wall (ms)':>12}{'tok/s':>10}")
    print(f"{'-'*54}")
    print(f"{'oMLX cold':<24}{w1*1000:>12.0f}{n1/w1:>10.1f}")
    print(f"{'oMLX warm':<24}{w2*1000:>12.0f}{n2/w2:>10.1f}")
    print(f"{'oMLX warm (2)':<24}{w3*1000:>12.0f}{n3/w3:>10.1f}")
    print(f"{'mlx-lm (always cold)':<24}{wm1*1000:>12.0f}{mn/wm1:>10.1f}")
    cache_saved_ms = (w1 - w2) * 1000
    print(f"\nprefix-cache savings (oMLX): {cache_saved_ms:.0f}ms = {(1 - w2/w1)*100:.1f}% off")


if __name__ == "__main__":
    main()
