"""MTP acceptance-rate benchmark for Qwen3.6-35B-A3B.

Question: is multi-token prediction (MTP) speculative decoding worth wiring
up on this model? The user's prior belief is "no — caching is good enough,
the model is too well-trained for spec decoding to land."

This script gives a clean number to confirm or refute that.

Method (acceptance-rate proxy, no batched verify needed):

  for each generation step:
      real_tok      = argmax( main_forward(prev_tok) )      # ground truth
      draft_next    = argmax( mtp_forward(real_tok) )       # cheap draft

  Acceptance rate = P(draft_next == real_tok of NEXT step).

That number is the *ceiling* for any MTP-driven spec-decode loop: even if
we add batched verify later, we cannot accept more drafts than the MTP head
actually gets right.

We also time main forward vs MTP forward to compute the breakeven point:

  if accept_rate * main_step_ms > mtp_step_ms + verify_overhead
      → MTP is worth wiring up
  else
      → don't bother

The engine we use is FusedExp2 (the only Qwen3.6 variant that exposes
mtp_forward today).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from moe_infer import _core as _rs
from moe_infer.qwen35_moe.pipeline import Qwen35MoEPipeline


def _argmax_int(x: np.ndarray) -> int:
    return int(np.argmax(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/Qwen3.6-35B-A3B")
    ap.add_argument("--engine", default="Qwen35MoEFusedExp2",
                    help="must support mtp_forward; FusedExp2 is the only one wired today")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    print(f"[bench_mtp] loading {args.model} with engine={args.engine}")
    pipe = Qwen35MoEPipeline(args.model, mode=args.engine)
    eng = pipe._engine
    cache = pipe._cache
    tok = pipe._tokenizer

    if not pipe._has_mtp:
        print("[bench_mtp] FAIL: model does not expose MTP weights")
        return

    # Prefill prompt with mtp=True so last_h_pre_norm gets captured.
    ids = np.array(tok.encode(args.prompt, add_special_tokens=False),
                   dtype=np.int64)
    print(f"[bench_mtp] prefill {len(ids)} tokens")
    logits = eng.forward(ids, cache, mtp=True)
    last_logits = logits[-1] if logits.ndim == 2 else logits

    eng.mtp_reset()

    main_ms: list[float] = []
    mtp_ms: list[float] = []
    accepts = 0
    pending_draft: int | None = None
    pending_real: int | None = None

    print(f"[bench_mtp] {args.warmup} warmup + {args.steps} measured steps")

    total_steps = args.warmup + args.steps
    generated: list[int] = []

    for step in range(total_steps):
        measure = step >= args.warmup

        # 1) main forward on previous real_tok (greedy) — sample from last_logits
        real_tok = _argmax_int(last_logits)
        generated.append(real_tok)

        # Compare last step's draft against this step's real_tok.
        if pending_draft is not None and measure:
            if pending_draft == real_tok:
                accepts += 1

        # 2) main forward to get logits for position +1
        t0 = time.perf_counter()
        next_logits = eng.forward(np.array([real_tok], dtype=np.int64),
                                  cache, mtp=True)
        t1 = time.perf_counter()
        if measure:
            main_ms.append((t1 - t0) * 1000.0)
        last_logits = next_logits[-1] if next_logits.ndim == 2 else next_logits

        # 3) MTP draft for position +2 (consumes last_h_pre_norm captured above)
        t2 = time.perf_counter()
        mtp_logits = eng.mtp_forward(real_tok)
        t3 = time.perf_counter()
        if measure:
            mtp_ms.append((t3 - t2) * 1000.0)

        if mtp_logits.size == 0:
            print("[bench_mtp] mtp_forward returned empty — MTP not wired in this engine")
            return

        pending_draft = _argmax_int(mtp_logits)

    # ── Report ───────────────────────────────────────────────────────────
    main_arr = np.array(main_ms)
    mtp_arr = np.array(mtp_ms)
    accept_rate = accepts / max(1, len(main_ms))

    print()
    print(f"[bench_mtp] generated: {tok.decode(generated[:24])!r} ...")
    print(f"[bench_mtp] main step:   mean={main_arr.mean():.1f} ms  "
          f"p50={np.median(main_arr):.1f}  p90={np.percentile(main_arr, 90):.1f}")
    print(f"[bench_mtp] mtp  step:   mean={mtp_arr.mean():.1f} ms  "
          f"p50={np.median(mtp_arr):.1f}  p90={np.percentile(mtp_arr, 90):.1f}")
    print(f"[bench_mtp] mtp / main ratio: {mtp_arr.mean() / main_arr.mean():.2%}")
    print(f"[bench_mtp] accept rate (greedy argmax): {accept_rate:.2%}  "
          f"({accepts}/{len(main_ms)})")

    # ── Implied speedup if batched verify costs ~0 ────────────────────────
    # In a 1-token-draft scheme: cost per accepted token =
    #   main_step + mtp_step + (1 - accept) * 0_savings
    # tokens/sec = 1 / mean(main + mtp) * (1 + accept_rate)
    base_tps = 1000.0 / main_arr.mean()
    spec_tps = (1.0 + accept_rate) / ((main_arr.mean() + mtp_arr.mean()) / 1000.0)
    print(f"[bench_mtp] baseline tok/s (no spec):                 {base_tps:.2f}")
    print(f"[bench_mtp] hypothetical spec tok/s (1-draft, free verify): {spec_tps:.2f}")
    print(f"[bench_mtp] hypothetical speedup:                    {spec_tps / base_tps:.2%}")
    print()
    print("note: real spec decoding needs *batched* verify in the engine to")
    print("hit the hypothetical number; we don't have it. So the speedup")
    print("above is an upper bound, not what you'd measure today.")


if __name__ == "__main__":
    main()
