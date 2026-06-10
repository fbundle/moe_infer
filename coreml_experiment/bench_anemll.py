"""Benchmark a pre-converted Anemll model on the Apple Neural Engine.

This is the fastest validated path to running an LLM on ANE — Anemll
distributes pre-compiled .mlmodelc artifacts on HF Hub that side-step the
conversion-toolchain bugs we've hit with coremltools 8.x/9.x. We just
download one and call their bundled chat.py.

Validated configuration (Apple M4, macOS 26.4.1):
  - Python 3.12 venv `.venv-anemll/`
  - coremltools 8.3.0
  - transformers==4.55.0   (newer versions return BatchEncoding without
                            .size() so Anemll's chat.py breaks)
  - torch 2.5.0            (matches coremltools' tested version)
  - anemll==0.3.5 (installed from `vendor/anemll/` via `uv pip install -e`)

Results on M4 (Qwen2.5-0.5B-Instruct, INT4 LUT, ctx=2048):
  - Prefill: ~75 tok/s
  - Decode:  ~74 tok/s steady-state
  - Wall-clock w/ 200-tok decode: ~60 tok/s
  Coherent output (200-token short story, sensible content).

Usage
    source .venv-anemll/bin/activate
    python coreml_experiment/bench_anemll.py \\
        --model anemll/anemll-Qwen-Qwen2.5-0.5B-Instruct-ctx2048-monolithic_0.3.5 \\
        --prompt "Write a short story." --max-tokens 200
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Anemll HF model id (e.g. 'anemll/anemll-Qwen-...').")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/coreml_models"),
                    help="HF cache root (default: ~/coreml_models)")
    ap.add_argument("--prompt", default="Write a short story about a cat who learns to use a computer.")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    # Download (or locate) the snapshot directory.
    from huggingface_hub import snapshot_download
    print(f"[download] {args.model} -> {args.cache_dir}")
    snap = Path(snapshot_download(args.model, cache_dir=args.cache_dir))
    print(f"[locate]   snapshot = {snap}")
    meta = snap / "meta.yaml"
    chat = snap / "chat.py"
    if not meta.exists() or not chat.exists():
        sys.exit(f"missing meta.yaml or chat.py in {snap}")

    # Run Anemll's bundled chat (this is the path we know works on ANE).
    cmd = [
        sys.executable, str(chat),
        "--meta", str(meta),
        "--prompt", args.prompt,
        "--max-tokens", str(args.max_tokens),
    ]
    print(f"[run]      {' '.join(cmd[:3])} ...\n")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
