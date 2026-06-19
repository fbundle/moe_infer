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
  # model_id|temperature|top_p|max_tokens|tolerant_json|concurrency|assistant_fmt|extra_flags
  # VibeThinker-3B BF16 (oMLX id: vibethinker-3b, weights 6.04GB) — authoritative baseline at authors' canonical dtype. Tests whether quant artifact explains the q4 numbers. conc=1 (BF16 + KV cache).
  "vibethinker-3b|1.0|0.95|40960|false|1|qwen3|"
  # Finish thinking-off Qwen3.5-4B first (resumes partial bandit, fresh GPQA), then thinking-on (slower) on separate JSONLs.
  "mlx-community--Qwen3.5-4B-4bit|0.7|0.8|40960|false|2|qwen3|--thinking-off"
  "mlx-community--Qwen3.5-4B-4bit|0.6|0.95|40960|false|1|qwen3|--dump-tag -think"
  # VibeThinker-3B q4 bandit terminated (degenerate arm-1 policy); ZL+GPQA already final.
  # "vibethinker-3b-q4|1.0|0.95|40960|false|2|qwen3|"
  # Other local models on hold:
  # "lfm25-8b-a1b-mlx-8bit|0.2|1.0|40960|true|2|qwen3|"
  # "mlx-community--gemma-4-e4b-it-qat-4bit|0.2|1.0|40960|true|4|gemma4|"
)
# Bandit skipped for the BF16 round — VibeThinker bandit q4 was already documented as degenerate (arm-1 sticky), BF16 won't change that and would burn 50+ hr.
AXES=("zebralogic" "gpqa_diamond")

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
  # Spawn fresh. Use absolute venv path since nohup's child shell may not
  # inherit a PATH that includes `omlx` (e.g. after a host reboot).
  echo "[$(date +%s)] starting omlx serve" >> "$LOG"
  nohup ../.venv/bin/omlx serve --memory-guard aggressive --paged-ssd-cache-dir "$(pwd)/omlx-cache" >> "$OMLX_LOG" 2>&1 &
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
    IFS='|' read -r M TEMP TOPP MAXTOK TOL CONC FMT EXTRA <<< "$ROW"
    restart_omlx || continue
    TOL_FLAG=""
    [ "$TOL" = "true" ] && TOL_FLAG="--tolerant-json"
    echo "[$(date +%s)] start $M $AXIS  temp=$TEMP top_p=$TOPP max_tok=$MAXTOK tolerant=$TOL conc=$CONC fmt=$FMT extra='${EXTRA:-}'" >> "$LOG"
    ./bench --model "$M" --benches "$AXIS" \
      --concurrency "$CONC" --max-tokens "$MAXTOK" --max-steps 60 \
      --temperature "$TEMP" --top-p "$TOPP" $TOL_FLAG \
      --assistant-fmt "$FMT" $EXTRA \
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
