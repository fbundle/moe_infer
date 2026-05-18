#!/usr/bin/env python3
"""Autotune experiment configurations for Flash-MoE 4-bit inference.

Sweeps the compile-time experiment parameter space, builds each config,
benchmarks it, and reports the optimal configuration.

Usage:
  python autotune.py
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

BENCH_PROMPT = "Explain quantum computing in one sentence."
BENCH_TOKENS = 100


def generate_config(active_experts: int, expert_cache_mode: int,
                    expert_cache_entries: int, use_gpu_linear: int,
                    use_expert_prediction: int, malloc_weights: int) -> bool:
    """Generate src/config.h. Returns True if changed."""
    cmd = [
        sys.executable, GEN_SCRIPT,
        "--active_experts", str(active_experts),
        "--expert_cache_mode", str(expert_cache_mode),
        "--expert_cache_entries", str(expert_cache_entries),
        "--use_gpu_linear", str(use_gpu_linear),
        "--use_expert_prediction", str(use_expert_prediction),
        "--malloc_weights", str(malloc_weights),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if "unchanged" in result.stdout:
        return False
    print(result.stdout.strip())
    return True


def build() -> bool:
    """Build the inference binary. Returns True on success."""
    result = subprocess.run(
        ["make"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        print(f"BUILD FAILED:\n{result.stderr}")
        return False
    return True


def benchmark(retry: bool = True) -> tuple[Optional[float], str]:
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
        if retry:
            print("(retry)", end=" ", flush=True)
            return benchmark(retry=False)
        print(f"FAILED (exit {result.returncode}):\n{result.stderr[-300:]}")
        return None, ""

    # Parse tok/s from:  Generation:     X.X s (Y.YY tok/s)
    toks = None
    for line in combined.splitlines():
        m = re.search(r'([\d.]+)\s*tok/s', line)
        if m:
            toks = float(m.group(1))
            break

    if toks is None:
        toks = BENCH_TOKENS / max(elapsed, 0.1)

    # Extract generated text after [prompt], stop at timing lines
    text_lines = []
    capture = False
    for line in combined.splitlines():
        stripped = line.strip()
        if '[prompt]' in stripped:
            capture = True
            continue
        if capture:
            if any(tag in stripped for tag in
                   ('[ttft]', '[timing]', '---', 'Generation:', 'TTFT:', 'Tokens:')):
                break
            if stripped.startswith('[') or stripped.startswith('Config:'):
                continue
            cleaned = stripped.replace('Ġ', ' ').replace('Ċ', '\n')
            if cleaned.strip():
                text_lines.append(cleaned)

    generated_text = ' '.join(text_lines).strip()
    return toks, generated_text


def config_key(k: int, cm: int, gl: int, pred: int, umw: int) -> str:
    return f"K={k}_GL={gl}_P={pred}_UMW={umw}"


def main():
    parser = argparse.ArgumentParser(description="Autotune Flash-MoE 4-bit configs")
    parser.add_argument("--active_experts", type=int, nargs="+", default=[8],
                        help="Active experts values (default: 8)")
    parser.add_argument("--use_gpu_linear", type=int, nargs="+", default=[0, 1],
                        help="GPU linear values (default: 0 1)")
    parser.add_argument("--use_expert_prediction", type=int, nargs="+", default=[0, 1],
                        help="Expert prediction values (default: 0 1)")
    parser.add_argument("--malloc_weights", type=int, nargs="+", default=[0, 1],
                        help="Malloc weights values (default: 0 1)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    ks = args.active_experts
    predictions = args.use_expert_prediction
    use_malloc_opts = args.malloc_weights
    cache_modes = [0]  # Only OS page cache (paper winner)

    total_configs = (len(ks) * len(cache_modes) * len(args.use_gpu_linear) *
                     len(predictions) * len(use_malloc_opts))
    print(f"=== Flash-MoE 4-bit Autotune ===")
    print(f"Sweep: ACTIVE_EXPERTS∈{ks}, GPU_LINEAR∈{args.use_gpu_linear}, "
          f"PREDICTION∈{predictions}, MALLOC_WEIGHTS∈{use_malloc_opts}")
    print(f"Total configs: {total_configs}")
    print(f"Prompt: \"{BENCH_PROMPT}\" ({BENCH_TOKENS} tokens)\n")

    results: List[Dict] = []
    best_toks = 0.0
    best_config = None

    for k in ks:
        for cm in cache_modes:
            for gl in args.use_gpu_linear:
                for pred in predictions:
                    for umw in use_malloc_opts:
                        key = config_key(k, cm, gl, pred, umw)
                        ce = 0

                        print(f"[{len(results) + 1}/{total_configs}] {key} ",
                              end="", flush=True)

                        changed = generate_config(k, cm, ce, gl, pred, umw)
                        if changed or not os.path.exists(INFER_BIN):
                            print(" building...", end=" ", flush=True)
                            if not build():
                                print("SKIP (build failed)")
                                continue
                        else:
                            print("(unchanged, skip rebuild)", end=" ", flush=True)

                        print(" running...", end=" ", flush=True)
                        toks, text = benchmark()
                        if toks is None:
                            print("SKIP (benchmark failed)")
                            continue

                        print(f"{toks:.2f} tok/s")
                        if text:
                            print(f"  \033[2;37m{text[:200]}\033[0m")
                        results.append({
                            "active_experts": k, "expert_cache_mode": cm,
                            "use_gpu_linear": gl, "use_expert_prediction": pred,
                            "malloc_weights": umw,
                            "tok_s": round(toks, 2), "text": text,
                        })

                        if toks > best_toks:
                            best_toks = toks
                            best_config = results[-1]

    # Summary
    print(f"\n=== Results ({len(results)}/{total_configs} successful) ===\n")

    if args.json:
        print(json.dumps(results, indent=2))
        return

    results.sort(key=lambda r: r["tok_s"], reverse=True)
    header = f"{'Rank':<5} {'K':<4} {'GPU_Lin':<8} {'Pred':<5} {'MallocW':<8} {'tok/s':<8}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"{i+1:<5} {r['active_experts']:<4} {r['use_gpu_linear']:<8} "
              f"{r['use_expert_prediction']:<5} {r['malloc_weights']:<8} "
              f"{r['tok_s']:<8.2f}{marker}")

    if best_config:
        bc = best_config
        print(f"\nBest: ACTIVE_EXPERTS={bc['active_experts']}, "
              f"USE_GPU_LINEAR={bc['use_gpu_linear']}, "
              f"USE_EXPERT_PREDICTION={bc['use_expert_prediction']}, "
              f"MALLOC_WEIGHTS={bc['malloc_weights']} "
              f"→ {best_toks:.2f} tok/s")
        print(f"\nApply: python helpers/gen_config.py "
              f"--active_experts {bc['active_experts']} "
              f"--use_gpu_linear {bc['use_gpu_linear']} "
              f"--use_expert_prediction {bc['use_expert_prediction']} "
              f"--malloc_weights {bc['malloc_weights']} && make")


if __name__ == "__main__":
    main()
