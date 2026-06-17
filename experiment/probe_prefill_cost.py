"""Measure the prefill cost curve on Anemll's ctx=2048 Qwen2.5-0.5B.

Hypothesis: a static-shape ctx=2048 graph pays the full 2048-position
attention cost on every prefill regardless of how many tokens are actually
in the prompt. If that's true, prefill time is ~constant across prompt
lengths, which means a multi-cache-size dispatcher (separate compiled
graphs at exponentially-spaced ctx sizes — the user's design idea) would
give a real prefill speedup for short prompts.

This script runs Anemll's chat.py with prompts of increasing length and
parses the reported prefill time / rate to plot the curve.

Notes
-----
- Anemll's chat.py prints "Prefill: 547.0ms (75.0 t/s)" — we parse this.
- For a fair test we strip the chat template overhead by using the same
  prompt-content shape repeated at different lengths.
- We loop the model with `--prompt` only (no interactive mode).
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


def parse_prefill(stdout: str) -> tuple[float, float] | None:
    """Returns (ms, tok_per_s) parsed from Anemll's chat.py output, or None."""
    m = re.search(r"Prefill:\s*([\d.]+)ms\s*\(([\d.]+)\s*t/s\)", stdout)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def run_prefill_at_length(snap: Path, prompt_words: int, *, max_tokens: int = 1) -> tuple[float, float] | None:
    """Run chat.py with a prompt of approximately `prompt_words` words,
    return (prefill_ms, prefill_tok_per_s). max_tokens=1 minimises the
    decode work so prefill dominates the wall-clock."""
    # Repeat a neutral filler word; the exact content doesn't matter for
    # the attention/FFN op cost — only the token count does.
    filler = " ".join(["apple"] * prompt_words)
    cmd = [
        sys.executable, str(snap / "chat.py"),
        "--meta", str(snap / "meta.yaml"),
        "--prompt", filler,
        "--max-tokens", str(max_tokens),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    return parse_prefill(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", required=True, help="Path to Anemll model snapshot dir")
    ap.add_argument("--lengths", default="2,8,32,128,256,512,1024",
                    help="Comma-separated prompt-word counts to probe")
    ap.add_argument("--repeat", type=int, default=3,
                    help="Runs per length; report median")
    args = ap.parse_args()

    snap = Path(args.snap_dir)
    if not (snap / "chat.py").exists():
        sys.exit(f"chat.py not found at {snap}")

    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"\n{'prompt words':>13} {'prefill ms (median)':>22} {'prefill tok/s':>16}")
    print("-" * 56)

    for L in lengths:
        times = []
        rates = []
        for _ in range(args.repeat):
            r = run_prefill_at_length(snap, L)
            if r is None:
                continue
            times.append(r[0])
            rates.append(r[1])
        if not times:
            print(f"{L:>13} {'(failed)':>22}")
            continue
        ms = statistics.median(times)
        tps = statistics.median(rates)
        print(f"{L:>13} {ms:>22.1f} {tps:>16.1f}")


if __name__ == "__main__":
    main()
