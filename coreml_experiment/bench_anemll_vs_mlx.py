"""Head-to-head ANE vs GPU bench for Qwen2.5-0.5B-Instruct.

Assumes powermetrics is already running in another window:
  sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -o /tmp/power.log

This script just runs the two inferences back-to-back, records wall-clock
windows for each, and then parses /tmp/power.log slicing by those windows.

For each run we report:
  - tok/s (decode steady-state)
  - average + peak ANE / GPU / CPU power (mW) during the inference window
  - joules per decode token (the "is the ANE pivot worth it" number)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Anemll's "Inference: X t/s" line is the decode rate (post-prefill).
RE_ANEMLL_INFER = re.compile(r"Inference:\s*([\d.]+)\s*t/s")
RE_ANEMLL_PREFILL = re.compile(r"Prefill:\s*[\d.]+ms\s*\(([\d.]+)\s*t/s\)")

# MLX-LM's generation_tps line.
RE_MLX_GEN_TPS = re.compile(r"generation_tps[\":]?\s*([\d.]+)")
RE_MLX_PROMPT_TPS = re.compile(r"prompt_tps[\":]?\s*([\d.]+)")


@dataclass
class RunResult:
    label: str
    start: float
    end: float
    decode_tps: Optional[float]
    prefill_tps: Optional[float]


def run_anemll(prompt: str, max_tokens: int) -> RunResult:
    repo = Path(__file__).resolve().parents[1]
    venv = repo / ".venv-anemll" / "bin" / "python"
    snap = next((Path(os.path.expanduser(
        "~/coreml_models/models--anemll--anemll-Qwen-Qwen2.5-0.5B-Instruct-ctx2048-monolithic_0.3.5/snapshots"
    ))).iterdir())
    cmd = [str(venv), str(snap / "chat.py"),
           "--meta", str(snap / "meta.yaml"),
           "--prompt", prompt, "--max-tokens", str(max_tokens)]
    print(f"[Anemll/ANE] running ...")
    start = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    end = time.time()
    text = out.stdout + out.stderr
    m_dec = RE_ANEMLL_INFER.search(text)
    m_pre = RE_ANEMLL_PREFILL.search(text)
    dec = float(m_dec.group(1)) if m_dec else None
    pre = float(m_pre.group(1)) if m_pre else None
    print(f"             decode={dec}  prefill={pre}  wall={end-start:.2f}s")
    return RunResult("Anemll/ANE", start, end, dec, pre)


def run_mlx(prompt: str, max_tokens: int) -> RunResult:
    repo = Path(__file__).resolve().parents[1]
    venv = repo / ".venv" / "bin" / "python"
    mdir = next((Path(os.path.expanduser(
        "~/coreml_models/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots"
    ))).iterdir())
    code = f"""
import time, json
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model, tok = load({str(mdir)!r})
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
    print(f"[MLX-LM/GPU] running ...")
    start = time.time()
    out = subprocess.run([str(venv), "-c", code], capture_output=True, text=True, timeout=600)
    end = time.time()
    dec = pre = None
    import json as _json
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                d = _json.loads(line)
                dec = d.get("generation_tps")
                pre = d.get("prompt_tps")
            except Exception:
                pass
    print(f"             decode={dec}  prefill={pre}  wall={end-start:.2f}s")
    return RunResult("MLX-LM/GPU", start, end, dec, pre)


# ── powermetrics text-log parser ─────────────────────────────────

RE_SAMPLE_TS = re.compile(
    r"^\*\*\* Sampled system activity \((?P<dow>\w+) (?P<mon>\w+)\s+"
    r"(?P<day>\d+) (?P<H>\d+):(?P<M>\d+):(?P<S>\d+) (?P<Y>\d+)"
)
RE_POWER_LINE = re.compile(r"^(?P<dev>CPU|GPU|ANE) Power:\s*(?P<mW>\d+)\s*mW")

MONTHS = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,
          "Sep":9,"Oct":10,"Nov":11,"Dec":12}


def parse_power_log(path: str) -> list[dict]:
    """Returns list of {ts: epoch_seconds, CPU/GPU/ANE: mW} per sample."""
    out, cur = [], None
    for line in open(path, errors="ignore"):
        m = RE_SAMPLE_TS.match(line)
        if m:
            if cur is not None:
                out.append(cur)
            ts = dt.datetime(int(m["Y"]), MONTHS[m["mon"]], int(m["day"]),
                             int(m["H"]), int(m["M"]), int(m["S"]))
            cur = {"ts": ts.timestamp(), "CPU": None, "GPU": None, "ANE": None}
            continue
        if cur is None:
            continue
        m = RE_POWER_LINE.match(line)
        if m:
            cur[m["dev"]] = float(m["mW"])
    if cur is not None:
        out.append(cur)
    return out


def slice_window(samples: list[dict], start: float, end: float) -> dict:
    """Average each device's mW during [start, end] (epoch seconds)."""
    sliced = [s for s in samples if start <= s["ts"] <= end]
    if not sliced:
        return {"n": 0}
    res = {"n": len(sliced)}
    for dev in ("CPU", "GPU", "ANE"):
        vals = [s[dev] for s in sliced if s[dev] is not None]
        if vals:
            res[f"{dev}_avg_mW"] = sum(vals) / len(vals)
            res[f"{dev}_peak_mW"] = max(vals)
    return res


def report(label: str, r: RunResult, win: dict) -> None:
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  decode tok/s     : {r.decode_tps}")
    print(f"  prefill tok/s    : {r.prefill_tps}")
    print(f"  wall window      : {r.end - r.start:.2f}s")
    print(f"  power samples in : {win.get('n')}")
    for dev in ("CPU", "GPU", "ANE"):
        avg = win.get(f"{dev}_avg_mW")
        peak = win.get(f"{dev}_peak_mW")
        if avg is not None:
            print(f"  {dev:>3} power       : avg={avg:>6.0f} mW   peak={peak:>6.0f} mW")
    # Joules per decoded token (decode-rate × avg combined power during the
    # window; not perfectly fair because the window includes prefill, but
    # for a decode-dominated long-prompt run it's close).
    combined_avg = sum(win.get(f"{d}_avg_mW", 0) for d in ("CPU","GPU","ANE"))
    if combined_avg and r.decode_tps:
        ms_per_tok = 1000.0 / r.decode_tps
        joules_per_tok = (combined_avg / 1000.0) * (ms_per_tok / 1000.0)
        print(f"  ≈ J/token (decode): {joules_per_tok:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=(
        "Write a short story about a curious cat who learns to use a "
        "computer and discovers the internet for the first time. "
        "Include vivid sensory details."
    ))
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--log", default="/tmp/power.log")
    ap.add_argument("--gap-s", type=float, default=4.0,
                    help="Idle gap between runs so windows are clearly separable.")
    args = ap.parse_args()

    if not Path(args.log).exists():
        sys.exit(f"power log not found: {args.log} -- is powermetrics running?")

    time.sleep(args.gap_s)
    anemll = run_anemll(args.prompt, args.max_tokens)
    time.sleep(args.gap_s)
    mlx = run_mlx(args.prompt, args.max_tokens)
    time.sleep(args.gap_s)  # let powermetrics emit one more sample

    print(f"\n[parse] {args.log}")
    samples = parse_power_log(args.log)
    print(f"        parsed {len(samples)} samples")

    for r in (anemll, mlx):
        win = slice_window(samples, r.start, r.end)
        report(f"{r.label}", r, win)


if __name__ == "__main__":
    main()
