"""Controlled W/token bench: same Qwen3 across ANE and GPU.

Three runs, same prompt, same 100-token decode budget:
  1. Qwen3-0.6B on ANE   (anemll prebuilt, LUT-quantized)
  2. Qwen3-0.6B on GPU   (mlx-community, 4-bit)
  3. Qwen3-4B   on GPU   (mlx-community, 4-bit)  — for size-scaling

Requires `sudo powermetrics --samplers cpu_power,gpu_power,ane_power
-i 500 -o /tmp/power.log` already running in another terminal.

Reports tok/s, avg+peak CPU/GPU/ANE wattage during each run's window, and
joules-per-token. Same model (0.6B) on both stacks isolates the ANE-vs-GPU
energy story from the model-size confound that polluted the earlier
0.5B-Qwen2.5 vs 8B comparisons.
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


def run_anemll(snap_dir, prompt, max_tokens, label):
    repo = Path(__file__).resolve().parents[1]
    venv = repo / ".venv-anemll" / "bin" / "python"
    # Older Anemll snapshots ship a chat.py that doesn't accept --max-tokens.
    # Fall back to vendor/anemll/tests/chat.py when that's the case.
    bundled = snap_dir / "chat.py"
    vendor_chat = repo / "vendor" / "anemll" / "tests" / "chat.py"
    chat_script = bundled if bundled.exists() else vendor_chat
    # Probe whether the bundled chat.py accepts --max-tokens.
    if chat_script == bundled:
        try:
            help_out = subprocess.run(
                [str(venv), str(bundled), "--help"],
                capture_output=True, text=True, timeout=15,
            ).stdout + subprocess.run(
                [str(venv), str(bundled), "--help"],
                capture_output=True, text=True, timeout=15,
            ).stderr
            if "--max-tokens" not in help_out and vendor_chat.exists():
                chat_script = vendor_chat
        except Exception:
            pass
    cmd = [str(venv), str(chat_script),
           "--meta", str(snap_dir / "meta.yaml"),
           "--prompt", prompt, "--max-tokens", str(max_tokens)]
    print(f"[{label}] running ...")
    start = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    end = time.time()
    text = out.stdout + out.stderr
    m = RE_ANEMLL_INFER.search(text); dec = float(m.group(1)) if m else None
    m = RE_ANEMLL_PREFILL.search(text); pre = float(m.group(1)) if m else None
    print(f"           decode={dec}  prefill={pre}  wall={end-start:.2f}s")
    return RunResult(label, start, end, dec, pre)


def run_mlx(mlx_dir, prompt, max_tokens, label):
    repo = Path(__file__).resolve().parents[1]
    venv = repo / ".venv" / "bin" / "python"
    code = f"""
import time, json
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model, tok = load({str(mlx_dir)!r})
# Warmup
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
    print(f"[{label}] running ...")
    start = time.time()
    out = subprocess.run([str(venv), "-c", code], capture_output=True, text=True, timeout=600)
    end = time.time()
    dec = pre = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                d = json.loads(line)
                dec = d.get("generation_tps"); pre = d.get("prompt_tps")
            except Exception: pass
    print(f"           decode={dec}  prefill={pre}  wall={end-start:.2f}s")
    return RunResult(label, start, end, dec, pre)


def report(r: RunResult, win: dict):
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


def summary_table(results: list[tuple[RunResult, dict]]):
    print(f"\n{'='*72}")
    print(f"{'SUMMARY (Qwen3, same prompt, 100-tok decode)':^72}")
    print(f"{'='*72}")
    print(f"{'run':<32}{'decode tok/s':>14}{'combined W':>14}{'J/token':>12}")
    print(f"{'-'*72}")
    for r, win in results:
        if r.decode_tps is None:
            print(f"{r.label:<32}{'(failed)':>14}")
            continue
        combined = sum(win.get(f"{d}_avg_mW", 0) for d in ("CPU","GPU","ANE")) / 1000.0
        jpt = combined / r.decode_tps if r.decode_tps else float("nan")
        print(f"{r.label:<32}{r.decode_tps:>14.1f}{combined:>14.2f}{jpt:>12.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=(
        "Write a Python function that computes the n-th Fibonacci number "
        "recursively. Include a docstring and an example."
    ))
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--log", default="/tmp/power.log")
    ap.add_argument("--gap-s", type=float, default=4.0,
                    help="Idle gap between runs so power windows separate cleanly.")
    args = ap.parse_args()

    if not Path(args.log).exists():
        sys.exit(f"power log not found: {args.log}  (is `sudo powermetrics ... -o {args.log}` running?)")

    home = os.path.expanduser("~/coreml_models")

    def _maybe(path):
        p = Path(path)
        if not p.exists():
            return None
        try:
            return next(p.iterdir())
        except StopIteration:
            return None

    anemll_06b = _maybe(f"{home}/models--anemll--anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4/snapshots")
    mlx_06b    = _maybe(f"{home}/models--mlx-community--Qwen3-0.6B-4bit/snapshots")
    mlx_4b     = _maybe(f"{home}/models--mlx-community--Qwen3-4B-4bit/snapshots")
    mlx_8b     = _maybe(f"{home}/models--mlx-community--Qwen3-8B-4bit/snapshots")
    anemll_ds8 = _maybe(f"{home}/models--anemll--anemll-DeepSeekR1-8B-ctx1024_0.2.0/snapshots")
    mlx_ds8    = _maybe(f"{home}/models--mlx-community--DeepSeek-R1-Distill-Llama-8B-4bit/snapshots")

    runs = []
    if anemll_06b:
        time.sleep(args.gap_s)
        runs.append(run_anemll(anemll_06b, args.prompt, args.max_tokens, "Qwen3-0.6B / ANE (Anemll)"))
    if mlx_06b:
        time.sleep(args.gap_s)
        runs.append(run_mlx(mlx_06b, args.prompt, args.max_tokens, "Qwen3-0.6B / GPU (MLX)"))
    if mlx_4b:
        time.sleep(args.gap_s)
        runs.append(run_mlx(mlx_4b, args.prompt, args.max_tokens, "Qwen3-4B   / GPU (MLX)"))
    if mlx_8b:
        time.sleep(args.gap_s)
        runs.append(run_mlx(mlx_8b, args.prompt, args.max_tokens, "Qwen3-8B   / GPU (MLX)"))
    # 8B-class same-model head-to-head — only runs if both models are present.
    if anemll_ds8:
        time.sleep(args.gap_s)
        runs.append(run_anemll(anemll_ds8, args.prompt, args.max_tokens, "DeepSeek-R1-8B / ANE (Anemll)"))
    if mlx_ds8:
        time.sleep(args.gap_s)
        runs.append(run_mlx(mlx_ds8, args.prompt, args.max_tokens, "DeepSeek-R1-8B / GPU (MLX)"))
    time.sleep(args.gap_s)

    samples = parse_power_log(args.log)
    print(f"\n[parse] {args.log}  → {len(samples)} samples")

    paired = []
    for r in runs:
        win = slice_window(samples, r.start, r.end)
        report(r, win)
        paired.append((r, win))

    summary_table(paired)


if __name__ == "__main__":
    main()
