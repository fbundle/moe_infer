"""Per-section decode timer for FusedExp5 on Qwen3.6-35B-A3B.

Wires `record_engine_telemetry(True)` and reads back the accumulated
phase counters after a warmup + measured run:

  phase.op1_encode_ms      — CPU encoding of attention CB (per-layer × layers)
  phase.op1_commit_wait_ms — GPU work for attention + the deferred op2 of L-1
  phase.route_cpu_ms       — softmax + topk over 256 expert scores
  engine.expert_io_ms      — pread / cache resolution for top-k experts
  phase.op2_encode_ms      — CPU encoding of post-expert CB (gate+up+swiglu+down+combine)
  phase.hidden_wait_ms     — final commit + wait for the last pending CB
  phase.lm_head_ms         — final norm + lm_head matvec (per token)

The lm_head/hidden_wait/route_cpu are per-token; op1/op2/expert_io
accumulate across all 40 layers per token. The report divides each by
the number of measured steps so we see milliseconds-per-token.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from moe_infer import _core as _rs
from moe_infer.qwen35_moe.pipeline import Qwen35MoEPipeline


_PHASES = [
    "phase.op1_encode_ms",
    "phase.op1_commit_wait_ms",
    "phase.route_cpu_ms",
    "engine.expert_io_ms",
    "phase.op2_encode_ms",
    "phase.hidden_wait_ms",
    "phase.lm_head_ms",
    "engine.total_ms",
]

_CACHE_KEYS = ["cache.hits", "cache.misses"]


def _read(t: dict, k: str) -> float:
    v = t.get(k, 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/Qwen3.6-35B-A3B")
    ap.add_argument("--engine", default="Qwen35MoEFusedExp5")
    ap.add_argument("--prompt",
                    default="Once upon a time in a small village by the sea there lived a young girl who loved to")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--expert-cache", type=int, default=0,
                    help="LRU expert-cache slot count (0=disabled, 256+=one layer cached, etc.)")
    args = ap.parse_args()

    print(f"[profile] loading {args.model} with engine={args.engine}")
    pipe = Qwen35MoEPipeline(args.model, mode=args.engine,
                             expert_cache_count=args.expert_cache)
    eng = pipe._engine
    cache = pipe._cache
    tok = pipe._tokenizer

    ids = np.array(tok.encode(args.prompt, add_special_tokens=False),
                   dtype=np.int64)
    print(f"[profile] prefill {len(ids)} tokens (no telemetry)")
    logits = eng.forward(ids, cache, mtp=False)
    last_logits = logits[-1] if logits.ndim == 2 else logits

    # Warmup: capture none.
    print(f"[profile] warmup {args.warmup} steps")
    for _ in range(args.warmup):
        t = int(np.argmax(last_logits))
        nl = eng.forward(np.array([t], dtype=np.int64), cache, mtp=False)
        last_logits = nl[-1] if nl.ndim == 2 else nl

    # Reset the telemetry by enabling now (existing entries will keep being
    # accumulated; we just snapshot the deltas).
    _rs.record_engine_telemetry(True)
    t_before = eng.telemetry()

    print(f"[profile] measured {args.steps} steps")
    t0 = time.perf_counter()
    for _ in range(args.steps):
        t = int(np.argmax(last_logits))
        nl = eng.forward(np.array([t], dtype=np.int64), cache, mtp=False)
        last_logits = nl[-1] if nl.ndim == 2 else nl
    wall_ms = (time.perf_counter() - t0) * 1000.0

    t_after = eng.telemetry()

    # ── Delta and per-step breakdown ─────────────────────────────────────
    print()
    print(f"[profile] wall time: {wall_ms:.0f} ms over {args.steps} steps"
          f"  =  {wall_ms / args.steps:.1f} ms/step  =  {args.steps * 1000.0 / wall_ms:.2f} tok/s")
    print()

    total = _read(t_after, "engine.total_ms") - _read(t_before, "engine.total_ms")
    if total <= 0:
        print("[profile] no telemetry recorded — was record_engine_telemetry(True) honored?")
        return

    print(f"{'section':<32} {'ms/step':>10}  {'pct':>6}")
    print("-" * 52)
    deltas = {}
    for k in _PHASES:
        d = _read(t_after, k) - _read(t_before, k)
        deltas[k] = d
        if k == "engine.total_ms":
            continue
        per_step = d / args.steps
        pct = 100.0 * d / total
        print(f"{k:<32} {per_step:>9.2f}   {pct:>5.1f}%")
    total_per_step = total / args.steps
    print("-" * 52)
    print(f"{'engine.total_ms':<32} {total_per_step:>9.2f}   100.0%")

    # ── Implied bandwidth check ──────────────────────────────────────────
    summed = sum(deltas[k] for k in _PHASES if k != "engine.total_ms")
    gap = total - summed
    if gap > 0:
        print(f"\n[profile] unaccounted (total - sum of phases): "
              f"{gap / args.steps:.2f} ms/step  ({100 * gap / total:.1f}%)")
        print("  ↑ this is residual: op1_full setup, embed lookup, gate readback,")
        print("    `init_hidden`, telemetry overhead, etc.")

    # ── Cache stats ──────────────────────────────────────────────────────
    print()
    hits = _read(t_after, "cache.hits") - _read(t_before, "cache.hits")
    misses = _read(t_after, "cache.misses") - _read(t_before, "cache.misses")
    if hits + misses > 0:
        rate = hits / (hits + misses)
        print(f"[cache] expert-cache size = {args.expert_cache}  "
              f"hits={int(hits)} misses={int(misses)} hit-rate={rate:.1%}")
    else:
        print(f"[cache] LRU disabled (expert_cache_count={args.expert_cache})")


if __name__ == "__main__":
    main()
