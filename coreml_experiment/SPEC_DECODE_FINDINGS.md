# Speculative decoding on Apple Silicon — what we measured

## Validated: MLX-only spec decode

Both drafter and verifier on the GPU, single Python process.

| Verifier | Drafter | K | α (acc/step) | Speedup vs plain | Notes |
|---|---|---|---|---|---|
| Qwen3-4B-4bit | Qwen3-0.6B-4bit | 2 | 1.24 / 2 | **1.26×** | sweet spot, code prompt |
| Qwen3-4B-4bit | Qwen3-0.6B-4bit | 3 | 1.61 / 3 | 1.11× | creative prompt |
| Qwen3-4B-4bit | Qwen3-0.6B-4bit | 4 | 1.78 / 4 | 0.95× | creative — break-even |
| Qwen3-4B-4bit | Qwen3-0.6B-4bit | 4 | 2.12 / 4 | 1.05× | code prompt |
| Qwen3-4B-4bit | Qwen3-0.6B-4bit | 6 | 2.64 / 6 | 0.89× | regression |
| Qwen3-4B-4bit | Qwen3-0.6B-4bit | 8 | 2.74 / 8 | 0.67× | regression |

All runs bit-identical to plain greedy decode. **Best practical setting: K=2.**

Honest takeaway: 1.2–1.3× speedup is real but modest. To unlock K=4-8 territory we'd need a more aligned drafter (larger, or distilled from the verifier).

## ANE+MLX integration: why we didn't ship it

Measured per-token rates on the same Qwen3-0.6B drafter:

| Stack | Decode tok/s | Prefill tok/s |
|---|---|---|
| MLX/GPU  | 244.5 | 608  |
| ANE      | 77.6  | 1307 |

The ANE drafter is **3.15× slower per token on decode** than the same model on the GPU. So **synchronous** ANE+MLX spec decoding loses to MLX+MLX on every dimension except instantaneous power draw.

Synchronous step cost (K=2, theoretical with no overhead):

| Variant | Draft 2 tok | Verify 3 tok | Step total |
|---|---|---|---|
| MLX + MLX | 8 ms | 10 ms | **18 ms** |
| ANE + MLX | 26 ms | 10 ms | **36 ms** |

ANE+MLX is 2× slower in wall-clock under synchronous scheduling. The only winning architecture is **async pipelining**: while the verifier is verifying step N, the ANE drafter is already drafting step N+1. With pipelining the step time becomes `max(draft, verify)` instead of `draft + verify`:

| Variant | max(draft, verify) | Effective rate |
|---|---|---|
| MLX + MLX, pipelined | max(8, 10) = 10 ms | ~220 tok/s (constrained by verifier) |
| ANE + MLX, pipelined | max(26, 10) = 26 ms | ~85 tok/s |

So even **pipelined** ANE+MLX (~85 tok/s) loses to **pipelined** MLX+MLX (~220 tok/s). The GPU's much higher draft throughput dominates.

**The one scenario where ANE-draft wins:** when the GPU is busy with *other work* (rendering, another model's inference, etc.) and we want spec decoding without contending for GPU cycles. In that case the ANE drafter is "free" because it's on a separate accelerator that isn't being used for anything else.

For a single-model laptop chat workload, **MLX-only spec decoding with K=2 is the right pattern**. ANE makes sense as the *verifier* (the model you actually want to deploy) for the power-efficient sustained-chat case, not as a drafter.

## What we didn't do (and why)

- **Persistent ANE-drafter subprocess server**: would have been ~3-4 hours of work building a stdin/stdout protocol around Anemll's CoreML MLModel. The math above made the verdict clear; building the integration would have just produced a slower number.
- **Larger drafter (Qwen3-1.7B)**: might lift acceptance to 65-75%, K=4 viable, speedup to 1.6-1.8×. Worth trying next session.
- **Async pipelining of MLX+MLX**: real ~2× win available but requires futures + careful KV-cache management. Sketched in the table above; not yet implemented.

## Files

- `coreml_experiment/spec_decode.py` — validated MLX-only spec decode + α measurement
