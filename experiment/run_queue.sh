#!/bin/bash
# Axis-major queue with per-model sampling/decode params.
# Each row is "model_id|temperature|top_p|max_tokens". top_k isn't in the
# OpenAI API surface, so it's omitted (Qwen3 thinking spec includes
# top_k=20 — applied at the server, not from here).
set -uo pipefail
cd "$(dirname "$0")"

DUMP=../data/bench_runs/go-bench
LOG=$DUMP/queue.log
mkdir -p "$DUMP"

MODELS=(
  # model_id|temperature|top_p|max_tokens
  "vibethinker-3b-q4|1.0|0.95|40960"
  "mlx-community--Qwen3.5-4B-4bit|0.6|0.95|40960"
  "lfm25-8b-a1b-mlx-8bit|0.2|1.0|40960"
)
AXES=("zebralogic" "kalshi" "gpqa_diamond")

for AXIS in "${AXES[@]}"; do
  for ROW in "${MODELS[@]}"; do
    IFS='|' read -r M TEMP TOPP MAXTOK <<< "$ROW"
    echo "[$(date +%s)] start $M $AXIS  temp=$TEMP top_p=$TOPP max_tok=$MAXTOK" >> "$LOG"
    ./bench --model "$M" --benches "$AXIS" \
      --concurrency 2 --max-tokens "$MAXTOK" --max-retries 5 \
      --temperature "$TEMP" --top-p "$TOPP" \
      --dump-dir "$DUMP" \
      >> "$DUMP/${M}.${AXIS}.bench.log" 2>&1
    echo "[$(date +%s)] done  $M $AXIS exit=$?" >> "$LOG"
  done
done
echo "[$(date +%s)] queue done" >> "$LOG"
