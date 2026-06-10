#!/bin/bash
# Measure decode speed AND power for Qwen2.5-0.5B-Instruct on:
#   1. Apple Neural Engine via Anemll (ctx=2048, INT4 LUT)
#   2. Apple GPU via MLX-LM (4bit)
#
# Same base model, same prompt, same decode length, same hardware (M4).
# Power via `sudo powermetrics --samplers cpu_power,gpu_power,ane_power`.
#
# Requires sudo for powermetrics. The script prints decode tok/s and average
# wattages for each device during inference. Joules-per-token is the headline
# number that justifies (or doesn't) the ANE pivot.

set -e

REPO="/Volumes/Hippopotamus/vault/code/moe_infer"
cd "$REPO"

ANEMLL_SNAP=$(find ~/coreml_models/models--anemll--anemll-Qwen-Qwen2.5-0.5B-Instruct-ctx2048-monolithic_0.3.5/snapshots -maxdepth 1 -type d | tail -1)
MLX_DIR=$(find ~/coreml_models/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots -maxdepth 1 -type d | tail -1)

PROMPT="Write a short story about a curious cat who learns to use a computer and discovers the internet for the first time."
MAX_TOKENS=300

LOG_ANE="/tmp/power_ane.log"
LOG_MLX="/tmp/power_mlx.log"

run_with_power() {
    # Args: <log_file> <command...>
    local log="$1"; shift
    # Powermetrics samples every 500ms, writes to $log.
    sudo /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -o "$log" >/dev/null 2>&1 &
    local PM_PID=$!
    sleep 1  # let powermetrics warm up
    "$@"
    local RC=$?
    sleep 1
    sudo kill -2 $PM_PID 2>/dev/null || true
    wait $PM_PID 2>/dev/null || true
    return $RC
}

# ── 1. Anemll on ANE ──────────────────────────────────────────
echo "=========================================================="
echo "[1/2] Qwen2.5-0.5B-Instruct via Anemll on ANE"
echo "=========================================================="

run_with_power "$LOG_ANE" bash -c "
    source $REPO/.venv-anemll/bin/activate
    python '$ANEMLL_SNAP/chat.py' \
        --meta '$ANEMLL_SNAP/meta.yaml' \
        --prompt '$PROMPT' --max-tokens $MAX_TOKENS \
    | grep -E 'Prefill|Inference|Total|t/s'
"

# ── 2. MLX-LM on GPU ──────────────────────────────────────────
echo ""
echo "=========================================================="
echo "[2/2] Qwen2.5-0.5B-Instruct via MLX-LM on GPU"
echo "=========================================================="

run_with_power "$LOG_MLX" bash -c "
    source $REPO/.venv/bin/activate
    python -c \"
import time
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model, tok = load('$MLX_DIR')
# Warmup
for _ in stream_generate(model, tok, '$PROMPT', max_tokens=10, sampler=make_sampler(0.0)):
    pass
# Timed run
t0 = time.perf_counter()
last = None
for r in stream_generate(model, tok, '$PROMPT', max_tokens=$MAX_TOKENS, sampler=make_sampler(0.0)):
    last = r
elapsed = time.perf_counter() - t0
print(f'Wall-clock: {elapsed*1000:.1f}ms  ({$MAX_TOKENS/elapsed:.1f} t/s)')
if hasattr(last, 'generation_tps'):
    print(f'MLX-reported generation_tps: {last.generation_tps:.1f} t/s')
if hasattr(last, 'prompt_tps'):
    print(f'MLX-reported prompt_tps:     {last.prompt_tps:.1f} t/s')
\"
"

# ── Parse the power logs and print summary ────────────────────
echo ""
echo "=========================================================="
echo "Power summary (mW averages, from powermetrics)"
echo "=========================================================="

python3 - "$LOG_ANE" "$LOG_MLX" <<'PY'
import re
import statistics
import sys

def parse(log_path: str) -> dict:
    """Returns dict of device → list of mW samples during inference window."""
    text = open(log_path, errors="ignore").read()
    out = {"ANE": [], "GPU": [], "CPU": [], "Combined": []}
    for line in text.splitlines():
        m = re.match(r"ANE Power:\s*(\d+)\s*mW", line)
        if m: out["ANE"].append(float(m.group(1))); continue
        m = re.match(r"GPU Power:\s*(\d+)\s*mW", line)
        if m: out["GPU"].append(float(m.group(1))); continue
        m = re.match(r"CPU Power:\s*(\d+)\s*mW", line)
        if m: out["CPU"].append(float(m.group(1))); continue
        m = re.match(r"Combined Power.*?:\s*(\d+)\s*mW", line)
        if m: out["Combined"].append(float(m.group(1))); continue
    return out

for label, path in [("ANE (Anemll)", sys.argv[1]), ("GPU (MLX-LM)", sys.argv[2])]:
    data = parse(path)
    print(f"\n[{label}]  samples={len(data['ANE'])}")
    for k in ["ANE", "GPU", "CPU", "Combined"]:
        if data[k]:
            avg = statistics.mean(data[k])
            peak = max(data[k])
            print(f"  {k:>10}  avg={avg:>7.0f} mW   peak={peak:>7.0f} mW")
PY
