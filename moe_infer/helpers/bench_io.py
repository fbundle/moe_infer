#!/usr/bin/env python3
"""bench_io.py — Measure how much per-token time is spent on expert disk I/O
vs GPU vs CPU routing, using the engine's built-in telemetry.

Two configurations:
  - no cache: every active expert misses, pure streaming workload
  - shared LRU cache 32: realistic steady state after warmup

Usage:
    python helpers/bench_io.py [--model PATH] [--tokens N] [--runs N]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import moe_infer
from moe_infer import Model, Engine, Cache


def random_tokens(n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(4, 50000, size=n).astype(np.int64)


def snapshot(eng) -> dict[str, float]:
    out = {}
    for k, v in eng.telemetry().items():
        if isinstance(v, list):
            v = float(np.sum(v))
        out[k] = float(v)
    return out


def delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: after[k] - before.get(k, 0.0) for k in after}


def run_config(model: Model, *, pipeline_mode: str, expert_cache_count: int,
               n_tokens: int, n_runs: int, warmup_tokens: int) -> dict:
    eng = moe_infer._core._rs.Engine(
        model._inner, pipeline_mode, 0, expert_cache_count=expert_cache_count
    )

    # Warmup (also primes the LRU cache when enabled)
    warm = random_tokens(warmup_tokens, seed=1)
    cache = moe_infer._core._rs.Cache(model._inner)
    eng.forward(warm, cache)

    all_telem: dict[str, list[float]] = {}
    wall_times: list[float] = []
    for r in range(n_runs):
        ids = random_tokens(n_tokens, seed=100 + r)
        c = moe_infer._core._rs.Cache(model._inner)

        # warm the new cache so we measure steady-state, not first-pos overhead
        eng.forward(ids[:1], c)
        before = snapshot(eng)

        t0 = time.perf_counter()
        eng.forward(ids[1:], c)
        wall = (time.perf_counter() - t0) * 1000.0

        d = delta(snapshot(eng), before)
        wall_times.append(wall)
        for k, v in d.items():
            all_telem.setdefault(k, []).append(v)
    return {
        "wall_ms_mean": float(np.mean(wall_times)),
        "wall_ms_std":  float(np.std(wall_times)),
        "telem": {k: (float(np.mean(v)), float(np.std(v))) for k, v in all_telem.items()},
        "n_tokens_per_run": n_tokens - 1,
        "n_runs": n_runs,
    }


def fmt_row(name: str, mean_ms: float, std_ms: float, total_ms: float) -> str:
    pct = 100.0 * mean_ms / total_ms if total_ms > 0 else 0
    return f"  {name:<32s}  {mean_ms:8.2f} ± {std_ms:6.2f} ms   ({pct:5.1f}%)"


def report(label: str, result: dict):
    print(f"\n{'='*72}")
    print(f"{label}   ({result['n_tokens_per_run']} measured tokens × {result['n_runs']} runs)")
    print('='*72)
    wall = result["wall_ms_mean"]
    print(f"  wall time: {wall:.2f} ms total  "
          f"({wall / result['n_tokens_per_run']:.2f} ms/token, "
          f"{1000.0 * result['n_tokens_per_run'] / wall:.1f} tok/s)")
    print()

    t = result["telem"]
    total = t.get("engine.total_ms", (wall, 0))[0]

    breakdown = [
        ("engine.total_ms",       "total (engine)"),
        ("engine.gpu_wait_ms",    "  GPU op1 wait"),
        ("engine.expert_io_ms",   "  expert routing+I/O (full)"),
        ("engine.expert_pread_ms","    └─ pread (disk only)"),
        ("engine.routing_cpu_ms", "  routing CPU (softmax+topk)"),
    ]
    for key, name in breakdown:
        if key in t:
            m, s = t[key]
            print(fmt_row(name, m, s, total))

    if "engine.expert_pread_bytes" in t:
        pb_mean, _ = t["engine.expert_pread_bytes"]
        pread_ms, _ = t.get("engine.expert_pread_ms", (0, 0))
        if pread_ms > 0:
            gbps = (pb_mean / 1e9) / (pread_ms / 1000.0)
            print(f"\n  pread bytes total: {pb_mean / 1e6:.1f} MB   "
                  f"throughput: {gbps:.2f} GB/s")

    if "cache.hits" in t and "cache.misses" in t:
        h, _ = t["cache.hits"]
        m, _ = t["cache.misses"]
        if h + m > 0:
            print(f"  cache hit rate: {100.0 * h / (h + m):.1f}%   "
                  f"({int(h)} hits / {int(m)} misses)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/Qwen3.6-35B-A3B/model_bq4")
    ap.add_argument("--tokens", type=int, default=64,
                    help="tokens per measured run (we drop the first to warm a fresh cache)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup-tokens", type=int, default=32)
    ap.add_argument("--sweep", type=str, default=None,
                    help="Comma-separated cache sizes to sweep, e.g. '0,32,64,128,256,512'")
    args = ap.parse_args()

    print(f"Loading model {args.model}...")
    model = Model(args.model)
    moe_infer.record_engine_telemetry(True)

    if args.sweep:
        sizes = [int(s) for s in args.sweep.split(",")]
        summary_rows = []
        for sz in sizes:
            label = "no cache" if sz == 0 else f"cache={sz}"
            res = run_config(
                model, pipeline_mode="Qwen35MoEFusedExp2", expert_cache_count=sz,
                n_tokens=args.tokens, n_runs=args.runs, warmup_tokens=args.warmup_tokens,
            )
            report(label, res)
            tok_per_s = 1000.0 * res["n_tokens_per_run"] / res["wall_ms_mean"]
            ms_per_tok = res["wall_ms_mean"] / res["n_tokens_per_run"]
            pread_ms_mean = res["telem"].get("engine.expert_pread_ms", (0, 0))[0]
            pread_pct = 100 * pread_ms_mean / res["telem"]["engine.total_ms"][0]
            gpu_pct = 100 * res["telem"]["engine.gpu_wait_ms"][0] / res["telem"]["engine.total_ms"][0]
            pread_bytes_mean = res["telem"].get("engine.expert_pread_bytes", (0, 0))[0]
            hits = res["telem"].get("cache.hits", (0, 0))[0]
            misses = res["telem"].get("cache.misses", (0, 0))[0]
            hit_rate = (100 * hits / (hits + misses)) if (hits + misses) > 0 else 0.0
            summary_rows.append((sz, tok_per_s, ms_per_tok, gpu_pct, pread_pct,
                                 pread_bytes_mean / 1e6, hit_rate))

        print(f"\n{'='*72}")
        print("SWEEP SUMMARY")
        print(f"{'='*72}")
        print(f"  {'cache':>6s}  {'tok/s':>6s}  {'ms/tok':>7s}  "
              f"{'GPU%':>5s}  {'pread%':>7s}  {'pread MB':>9s}  {'hit%':>5s}")
        for sz, tps, mst, g, p, pb, hr in summary_rows:
            label = "0" if sz == 0 else f"{sz}"
            print(f"  {label:>6s}  {tps:>6.2f}  {mst:>7.1f}  "
                  f"{g:>5.1f}  {p:>7.1f}  {pb:>9.1f}  {hr:>5.1f}")
        return

    # Configs to compare
    nocache = run_config(
        model, pipeline_mode="Qwen35MoEFusedExp2", expert_cache_count=0,
        n_tokens=args.tokens, n_runs=args.runs, warmup_tokens=args.warmup_tokens,
    )
    report("no expert cache (every active expert = pread)", nocache)

    cached = run_config(
        model, pipeline_mode="Qwen35MoEFusedExp2", expert_cache_count=32,
        n_tokens=args.tokens, n_runs=args.runs, warmup_tokens=args.warmup_tokens,
    )
    report("expert_cache_count=32 (steady state)", cached)


if __name__ == "__main__":
    main()
