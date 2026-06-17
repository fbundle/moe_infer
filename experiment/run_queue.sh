#!/bin/bash
# Model-major queue: for each model, run all axes, then next model.
# Manages its own oMLX server: kills whatever holds port 9100, spawns
# a fresh `omlx serve` between every (model, axis), waits for ready,
# then runs the bench. Pidfiles let kill_queue.sh stop cleanly.
set -uo pipefail
cd "$(dirname "$0")"

DUMP=../data/bench_runs/go-bench
LOG=$DUMP/queue.log
OMLX_LOG=$DUMP/omlx.log
QUEUE_PID="$DUMP/.queue.pid"
BENCH_PID="$DUMP/.bench.pid"
OMLX_PID="$DUMP/.omlx.pid"
mkdir -p "$DUMP"
echo $$ > "$QUEUE_PID"
trap 'rm -f "$QUEUE_PID" "$BENCH_PID"' EXIT

MODELS=(
  # model_id|temperature|top_p|max_tokens|tolerant_json|concurrency
  "vibethinker-3b-q4|1.0|0.95|40960|false|2"
  "mlx-community--gemma-4-e4b-it-qat-4bit|0.2|1.0|40960|true|4"
  "lfm25-8b-a1b-mlx-8bit|0.2|1.0|40960|true|2"
  # Qwen3.5-4B on hold: even at conc=1, thinking mode burns 30k+ tokens per puzzle → ~20 min/ex.
  # "mlx-community--Qwen3.5-4B-4bit|0.6|0.95|40960|false|1"
)
AXES=("zebralogic" "bandit" "gpqa_diamond")

restart_omlx() {
  # Kill whoever holds port 9100 (could be a prior `omlx serve` we
  # spawned, or an orphaned server from before the queue started).
  local pid
  pid=$(lsof -ti :9100 2>/dev/null | head -1 || true)
  if [ -n "$pid" ]; then
    echo "[$(date +%s)] stopping oMLX PID=$pid" >> "$LOG"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  # Spawn fresh
  echo "[$(date +%s)] starting omlx serve" >> "$LOG"
  nohup omlx serve --paged-ssd-cache-dir "$(pwd)/omlx-cache" >> "$OMLX_LOG" 2>&1 &
  echo $! > "$OMLX_PID"
  # Wait for /v1/models
  for _ in $(seq 1 90); do
    if curl -sf -o /dev/null http://127.0.0.1:9100/v1/models; then
      echo "[$(date +%s)] oMLX ready PID=$(cat $OMLX_PID)" >> "$LOG"
      return 0
    fi
    sleep 2
  done
  echo "[$(date +%s)] oMLX did not become ready in 180s" >> "$LOG"
  return 1
}

for ROW in "${MODELS[@]}"; do
  for AXIS in "${AXES[@]}"; do
    IFS='|' read -r M TEMP TOPP MAXTOK TOL CONC <<< "$ROW"
    restart_omlx || continue
    TOL_FLAG=""
    [ "$TOL" = "true" ] && TOL_FLAG="--tolerant-json"
    echo "[$(date +%s)] start $M $AXIS  temp=$TEMP top_p=$TOPP max_tok=$MAXTOK tolerant=$TOL conc=$CONC" >> "$LOG"
    ./bench --model "$M" --benches "$AXIS" \
      --concurrency "$CONC" --max-tokens "$MAXTOK" --max-steps 60 \
      --temperature "$TEMP" --top-p "$TOPP" $TOL_FLAG \
      --dump-dir "$DUMP" \
      >> "$DUMP/${M}.${AXIS}.bench.log" 2>&1 &
    echo $! > "$BENCH_PID"
    wait $!
    EXIT_CODE=$?
    echo "[$(date +%s)] done  $M $AXIS exit=$EXIT_CODE" >> "$LOG"
    rm -f "$BENCH_PID"
  done
done
echo "[$(date +%s)] queue done" >> "$LOG"
