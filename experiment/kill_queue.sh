#!/bin/bash
# Stop the queue + any in-flight bench cleanly using pidfiles written by
# run_queue.sh. No pgrep, no text matching.
set -uo pipefail
cd "$(dirname "$0")"

DUMP=../data/bench_runs/go-bench
QUEUE_PID="$DUMP/.queue.pid"
BENCH_PID="$DUMP/.bench.pid"
OMLX_PID="$DUMP/.omlx.pid"

# Stop bench first so the queue doesn't immediately spawn the next one.
if [ -f "$BENCH_PID" ]; then
  PID=$(cat "$BENCH_PID")
  if kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID"
    echo "sent SIGTERM to bench PID=$PID"
  fi
fi

if [ -f "$QUEUE_PID" ]; then
  PID=$(cat "$QUEUE_PID")
  if kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID"
    echo "sent SIGTERM to queue PID=$PID"
  fi
fi

# Wait briefly for both to exit, then report
for _ in $(seq 1 10); do
  alive=0
  for pf in "$QUEUE_PID" "$BENCH_PID"; do
    if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
      alive=1
    fi
  done
  [ "$alive" = "0" ] && break
  sleep 1
done

if [ -f "$QUEUE_PID" ] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
  echo "queue still alive — sending SIGKILL"
  kill -KILL "$(cat "$QUEUE_PID")"
fi
if [ -f "$BENCH_PID" ] && kill -0 "$(cat "$BENCH_PID")" 2>/dev/null; then
  echo "bench still alive — sending SIGKILL"
  kill -KILL "$(cat "$BENCH_PID")"
fi
# Also stop the oMLX server we spawned (if any)
if [ -f "$OMLX_PID" ]; then
  PID=$(cat "$OMLX_PID")
  if kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID" 2>/dev/null
    echo "sent SIGTERM to omlx PID=$PID"
  fi
fi
rm -f "$QUEUE_PID" "$BENCH_PID" "$OMLX_PID"
echo "stopped"
