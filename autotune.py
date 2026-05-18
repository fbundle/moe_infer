#!/usr/bin/env python3
"""Autotune experiment configurations for Flash-MoE 4-bit inference.

Sweeps the compile-time experiment parameter space, builds each config,
benchmarks it, and reports the optimal configuration.

Usage:
  python autotune.py                  # full sweep
  python autotune.py --quick          # fast sweep (fewer K values)
  python autotune.py --k 4 --k 8      # test specific K values only
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import List, Dict, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GEN_SCRIPT = os.path.join(REPO_ROOT, "helpers", "gen_config.py")
BIN_DIR = os.path.join(REPO_ROOT, "bin")
INFER_BIN = os.path.join(BIN_DIR, "infer")

# Benchmark prompt — short enough to finish fast, long enough to exercise the pipeline
BENCH_PROMPT = "Explain quantum computing in one sentence."
BENCH_TOKENS = 100


def generate_config(k: int, cache_mode: int, cache_entries: int,
                    gpu_linear: int, prediction: int) -> bool:
    """Generate src/config.h. Returns True if changed."""
    cmd = [
        sys.executable, GEN_SCRIPT,
        "--k", str(k),
        "--cache_mode", str(cache_mode),
        "--cache_entries", str(cache_entries),
        "--gpu_linear", str(gpu_linear),
        "--prediction", str(prediction),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if "unchanged" in result.stdout:
        return False
    print(result.stdout.strip())
    return True


def build() -> bool:
    """Build the inference binary. Returns True on success."""
    result = subprocess.run(
        ["make", "infer"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        print(f"BUILD FAILED:\n{result.stderr}")
        return False
    return True


def benchmark() -> tuple[Optional[float], str]:
    """Run inference benchmark. Returns (tok/s or None, generated text)."""
    cmd = [
        INFER_BIN,
        "--prompt", BENCH_PROMPT,
        "--tokens", str(BENCH_TOKENS),
    ]
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=300)
    elapsed = time.monotonic() - t0

    combined = result.stdout + result.stderr

    if result.returncode != 0:
        print(f"BENCHMARK FAILED (exit {result.returncode}):\n{result.stderr[-500:]}")
        return None, combined

    # Parse output for tok/s.  infer.m prints lines like:
    #   [stats] 100 tokens in 18.32s (5.46 tok/s)
    toks = None
    for line in combined.splitlines():
        m = re.search(r'([\d.]+)\s*tok/s', line)
        if m:
            toks = float(m.group(1))

    if toks is None:
        toks = BENCH_TOKENS / elapsed
        print(f"  (no tok/s in output, using wall time: {toks:.2f} tok/s)")

    # Extract the generated text (lines between [prompt] and [stats])
    lines = combined.splitlines()
    text_lines = []
    in_output = False
    for line in lines:
        if '[prompt]' in line or '[stats]' in line:
            in_output = not in_output
            continue
        if in_output and line.strip() and not line.startswith('['):
            text_lines.append(line.strip())
    generated_text = ' '.join(text_lines).strip()

    return toks, generated_text


def config_key(k: int, cm: int, gl: int, pred: int) -> str:
    return f"K={k}_CM={cm}_GL={gl}_P={pred}"


def main():
    parser = argparse.ArgumentParser(description="Autotune Flash-MoE 4-bit configs")
    parser.add_argument("--quick", action="store_true",
                        help="Fast sweep: skip K=2 (low quality) and prediction")
    parser.add_argument("--k", type=int, nargs="+",
                        default=[2, 4, 8],
                        help="K values to test (default: 2 4 8)")
    parser.add_argument("--gpu_linear", type=int, nargs="+",
                        default=[0, 1],
                        help="GPU linear values (default: 0 1)")
    parser.add_argument("--prediction", type=int, nargs="+",
                        default=[0, 1],
                        help="Prediction values (default: 0 1)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    ks = args.k
    if args.quick:
        ks = [k for k in ks if k != 2]
        # Skip prediction sweep in quick mode
        predictions = [0]
    else:
        predictions = args.prediction

    # CACHE_MODE: only test OS page cache (paper winner: +38% over custom caches).
    # Malloc cache with enough entries to matter (~2581) needs ~17GB — tight on 16GB.
    cache_modes = [0]

    # Shared expert baseline info
    total_configs = len(ks) * len(cache_modes) * len(args.gpu_linear) * len(predictions)
    print(f"=== Flash-MoE 4-bit Autotune ===")
    print(f"Sweep: K∈{ks}, CACHE_MODE∈{cache_modes}, GPU_LINEAR∈{args.gpu_linear}, "
          f"PREDICTION∈{predictions}")
    print(f"Total configs: {total_configs}")
    print(f"Prompt: \"{BENCH_PROMPT}\" ({BENCH_TOKENS} tokens)\n")

    results: List[Dict] = []
    best_toks = 0.0
    best_config = None

    for k in ks:
        for cm in cache_modes:
            for gl in args.gpu_linear:
                for pred in predictions:
                    key = config_key(k, cm, gl, pred)
                    cache_entries = 0  # OS page cache doesn't use malloc entries

                    print(f"[{len(results) + 1}/{total_configs}] {key} ", end="", flush=True)

                    changed = generate_config(k, cm, cache_entries, gl, pred)
                    if changed or not os.path.exists(INFER_BIN):
                        print(" building...", end=" ", flush=True)
                        if not build():
                            print("SKIP (build failed)")
                            continue
                    else:
                        print(" (unchanged, skip rebuild)", end=" ", flush=True)

                    print(" running...", end=" ", flush=True)
                    toks, text = benchmark()
                    if toks is None:
                        print("SKIP (benchmark failed)")
                        continue

                    print(f"{toks:.2f} tok/s")
                    if text:
                        # Show generated text dimmed so quality can be assessed
                        print(f"  \033[2;37m{text[:200]}\033[0m")
                    results.append({"k": k, "cache_mode": cm, "cache_entries": cache_entries,
                                    "gpu_linear": gl, "prediction": pred, "tok_s": round(toks, 2),
                                    "text": text})

                    if toks > best_toks:
                        best_toks = toks
                        best_config = results[-1]

    # Summary
    print(f"\n=== Results ({len(results)}/{total_configs} successful) ===\n")

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Sort by tok/s descending
    results.sort(key=lambda r: r["tok_s"], reverse=True)
    print(f"{'Rank':<5} {'K':<4} {'Cache':<7} {'GPU_Lin':<8} {'Pred':<5} {'tok/s':<8}")
    print("-" * 45)
    for i, r in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"{i+1:<5} {r['k']:<4} {r['cache_mode']:<7} {r['gpu_linear']:<8} "
              f"{r['prediction']:<5} {r['tok_s']:<8.2f}{marker}")

    if best_config:
        print(f"\nBest: K={best_config['k']}, CACHE_MODE={best_config['cache_mode']}, "
              f"GPU_LINEAR={best_config['gpu_linear']}, PREDICTION={best_config['prediction']} "
              f"→ {best_toks:.2f} tok/s")
        print(f"\nApply: python helpers/gen_config.py --k {best_config['k']} "
              f"--cache_mode {best_config['cache_mode']} "
              f"--gpu_linear {best_config['gpu_linear']} "
              f"--prediction {best_config['prediction']} && make")


if __name__ == "__main__":
    main()
