"""8B head-to-head: same DeepSeek-R1-Distill-Llama-8B model on both stacks.

  - Anemll/ANE  : anemll/anemll-DeepSeekR1-8B-ctx1024_0.2.0
  - MLX-LM/GPU  : mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit

Requires `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500
-o /tmp/power.log` already running in another terminal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


RE_ANEMLL_INFER = re.compile(r"Inference:\s*([\d.]+)\s*t/s")
RE_ANEMLL_PREFILL = re.compile(r"Prefill:\s*[\d.]+ms\s*\(([\d.]+)\s*t/s\)")
RE_SAMPLE_TS = re.compile(
    r"^\*\*\* Sampled system activity \(\w+ (?P<mon>\w+)\s+(?P<day>\d+) "
    r"(?P<H>\d+):(?P<M>\d+):(?P<S>\d+) (?P<Y>\d+)"
)
RE_POWER_LINE = re.compile(r"^(?P<dev>CPU|GPU|ANE) Power:\s*(?P<mW>\d+)\s*mW")
MONTHS = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,
          "Sep":9,"Oct":10,"Nov":11,"Dec":12}


@dataclass
class RunResult:
    label: str
    start: float
    end: float
    decode_tps: Optional[float]
    prefill_tps: Optional[float]


def parse_power_log(path):
    out, cur = [], None
    for line in open(path, errors="ignore"):
        m = RE_SAMPLE_TS.match(line)
        if m:
            if cur is not None: out.append(cur)
            ts = dt.datetime(int(m["Y"]), MONTHS[m["mon"]], int(m["day"]),
                             int(m["H"]), int(m["M"]), int(m["S"]))
            cur = {"ts": ts.timestamp(), "CPU": None, "GPU": None, "ANE": None}
            continue
        if cur is None: continue
        m = RE_POWER_LINE.match(line)
        if m: cur[m["dev"]] = float(m["mW"])
    if cur is not None: out.append(cur)
    return out


def slice_window(samples, start, end):
    sliced = [s for s in samples if start <= s["ts"] <= end]
    res = {"n": len(sliced)}
    for dev in ("CPU","GPU","ANE"):
        vals = [s[dev] for s in sliced if s[dev] is not None]
        if vals:
            res[f"{dev}_avg_mW"] = sum(vals)/len(vals)
            res[f"{dev}_peak_mW"] = max(vals)
    return res


def run_anemll_8b(prompt, max_tokens):
    repo = Path(__file__).resolve().parents[1]
    venv = repo / ".venv-anemll" / "bin" / "python"
    snap = next((Path(os.path.expanduser(
        "~/coreml_models/models--anemll--anemll-DeepSeekR1-8B-ctx1024_0.2.0/snapshots"
    ))).iterdir())
    chat = repo / "vendor" / "anemll" / "tests" / "chat.py"
    cmd = [str(venv), str(chat), "--meta", str(snap/"meta.yaml"),
           "--prompt", prompt, "--max-tokens", str(max_tokens)]
    print("[Anemll/ANE] DeepSeek-R1 8B running ...")
    start = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    end = time.time()
    text = out.stdout + out.stderr
    m = RE_ANEMLL_INFER.search(text); dec = float(m.group(1)) if m else None
    m = RE_ANEMLL_PREFILL.search(text); pre = float(m.group(1)) if m else None
    print(f"             decode={dec}  prefill={pre}  wall={end-start:.2f}s")
    return RunResult("Anemll/ANE (DeepSeek-R1 8B)", start, end, dec, pre)


def run_mlx_8b(prompt, max_tokens):
    repo = Path(__file__).resolve().parents[1]
    venv = repo / ".venv" / "bin" / "python"
    mdir = next((Path(os.path.expanduser(
        "~/coreml_models/models--mlx-community--DeepSeek-R1-Distill-Llama-8B-4bit/snapshots"
    ))).iterdir())
    code = f"""
import time, json
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model, tok = load({str(mdir)!r})
for _ in stream_generate(model, tok, {prompt!r}, max_tokens=8, sampler=make_sampler(0.0)): pass
last = None
t0 = time.perf_counter()
for r in stream_generate(model, tok, {prompt!r}, max_tokens={max_tokens}, sampler=make_sampler(0.0)):
    last = r
print(json.dumps({{
    "wall_s": time.perf_counter() - t0,
    "generation_tps": getattr(last, "generation_tps", None),
    "prompt_tps": getattr(last, "prompt_tps", None),
}}))
"""
    print("[MLX-LM/GPU] DeepSeek-R1 8B running ...")
    start = time.time()
    out = subprocess.run([str(venv), "-c", code], capture_output=True, text=True, timeout=1200)
    end = time.time()
    dec = pre = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                d = json.loads(line)
                dec = d.get("generation_tps"); pre = d.get("prompt_tps")
            except Exception: pass
    print(f"             decode={dec}  prefill={pre}  wall={end-start:.2f}s")
    return RunResult("MLX-LM/GPU (DeepSeek-R1 8B)", start, end, dec, pre)


def report(r, win):
    print(f"\n{'='*60}\n{r.label}\n{'='*60}")
    print(f"  decode tok/s     : {r.decode_tps}")
    print(f"  prefill tok/s    : {r.prefill_tps}")
    print(f"  wall window      : {r.end - r.start:.2f}s")
    print(f"  power samples in : {win.get('n')}")
    for dev in ("CPU","GPU","ANE"):
        avg = win.get(f"{dev}_avg_mW"); peak = win.get(f"{dev}_peak_mW")
        if avg is not None:
            print(f"  {dev:>3} power       : avg={avg:>6.0f} mW   peak={peak:>6.0f} mW")
    combined = sum(win.get(f"{d}_avg_mW", 0) for d in ("CPU","GPU","ANE"))
    if combined and r.decode_tps:
        ms_per_tok = 1000.0 / r.decode_tps
        print(f"  ≈ J/token (decode): {(combined/1000.0) * (ms_per_tok/1000.0):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=(
        "Write a short story about a curious cat who learns to use a "
        "computer and discovers the internet for the first time. "
        "Include vivid sensory details."
    ))
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--log", default="/tmp/power.log")
    ap.add_argument("--gap-s", type=float, default=5.0)
    args = ap.parse_args()
    if not Path(args.log).exists():
        sys.exit(f"power log not found: {args.log}")
    time.sleep(args.gap_s)
    anemll = run_anemll_8b(args.prompt, args.max_tokens)
    time.sleep(args.gap_s)
    mlx = run_mlx_8b(args.prompt, args.max_tokens)
    time.sleep(args.gap_s)
    samples = parse_power_log(args.log)
    print(f"\n[parse] {args.log}  → {len(samples)} samples")
    for r in (anemll, mlx):
        report(r, slice_window(samples, r.start, r.end))


if __name__ == "__main__":
    main()
