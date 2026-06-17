#!/bin/bash
# Sequentially bench a list of oMLX models, gated on a watched PID.
# Same dump dir for all (filenames key by model name → no collisions).
set -uo pipefail
cd "$(dirname "$0")"

WAIT_PID="${1:-}"
DUMP=../data/bench_runs/go-bench
LOG=$DUMP/queue.log
mkdir -p "$DUMP"

if [ -n "$WAIT_PID" ]; then
  echo "[$(date +%s)] waiting on PID=$WAIT_PID" >> "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[$(date +%s)] PID=$WAIT_PID gone, queue starting" >> "$LOG"
fi

for M in \
  "mlx-community--Qwen3.5-4B-4bit" \
  "lfm25-8b-a1b-mlx-8bit"
do
  echo "[$(date +%s)] start $M" >> "$LOG"
  ./bench --model "$M" --concurrency 2 --max-tokens 40960 --max-retries 5 --dump-dir "$DUMP" \
    >> "$DUMP/${M}.bench.log" 2>&1
  echo "[$(date +%s)] done  $M exit=$?" >> "$LOG"
done
echo "[$(date +%s)] queue done" >> "$LOG"
