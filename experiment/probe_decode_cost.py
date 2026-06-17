"""Measure decode tok/s as a function of current cache position.

The hypothesis multi-cache-size graphs are designed to fix:
  - ANE attention over a static ctx=2048 cache touches all 2048 positions
    every decode step (mask just zeros out unfilled positions, doesn't
    skip the compute).
  - So decode tok/s drops as the conversation gets longer.

If decode tok/s is flat across cache positions, then ANE is already doing
something smarter (e.g., compiled sparse attention) and multi-cache-size
graphs won't help. If it drops with position, then the optimization is real.

How we measure: run chat.py with a long initial prompt of N filler words
(this fills the cache to position N+template overhead), then ask for 50
decode tokens. The reported "Inference: X t/s" is the steady-state decode
rate from position ~N to position ~N+50. Plot tok/s vs N.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path


def parse_inference(stdout: str) -> float | None:
    m = re.search(r"Inference:\s*([\d.]+)\s*t/s", stdout)
    return float(m.group(1)) if m else None


def run_decode_at_position(snap: Path, pre_tokens: int, decode_tokens: int = 50) -> float | None:
    """Prefill ~pre_tokens of context, then time `decode_tokens` of decode."""
    filler = " ".join(["apple"] * pre_tokens)
    cmd = [
        sys.executable, str(snap / "chat.py"),
        "--meta", str(snap / "meta.yaml"),
        "--prompt", filler,
        "--max-tokens", str(decode_tokens),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return parse_inference(r.stdout + r.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", required=True)
    ap.add_argument("--positions", default="0,32,128,512,1024",
                    help="Cache positions (= number of words in initial prompt)")
    ap.add_argument("--decode", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    snap = Path(args.snap_dir)

    print(f"\n{'cache pos':>10} {'decode tok/s (median)':>25}")
    print("-" * 38)

    for P in [int(x) for x in args.positions.split(",")]:
        rates = []
        for _ in range(args.repeat):
            r = run_decode_at_position(snap, P, args.decode)
            if r is not None:
                rates.append(r)
        if rates:
            print(f"{P:>10} {statistics.median(rates):>25.1f}")
        else:
            print(f"{P:>10} {'(failed)':>25}")


if __name__ == "__main__":
    main()
