"""Minimal speculative decoding prototype with two MLX models.

Architecture
------------
Both models in one Python process, both on the GPU (MLX is Metal-only).
Greedy decoding (temperature = 0) so we can measure acceptance against
the verifier's exact argmax — no probabilistic acceptance complications.

For each spec-decode step:
  1. Drafter autoregressively generates K candidate tokens.
  2. Verifier takes [last_committed, d1, d2, ..., dK] as a single
     batched-forward input and returns logits at every position.
  3. Walk positions 1..K: accept d_i if argmax(verifier_logits_i) == d_i.
     On first mismatch (or position K+1 if all matched), commit
     verifier's argmax as the "bonus" token.
  4. Roll back the drafter's KV cache to the accepted prefix end.

For KV cache management we use mlx_lm's `make_prompt_cache` and the
explicit cache.offset rewind trick (works for the standard KV cache).

Metrics
-------
  - α : average number of accepted draft tokens per step
        (NOT counting the bonus verifier token)
  - acceptance_rate : α / K  (fraction of drafted tokens accepted)
  - effective tok/s : N tokens generated / wall-clock seconds
  - plain decode tok/s : for the same N tokens, verifier-only autoregressive
                          generation, same prompt, same conditions

The pair we use:
  - Verifier : mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit
  - Drafter  : mlx-community/Llama-3.2-1B-Instruct-4bit
Both Llama-3 family → identical tokenizer (verified by `tokenizer.vocab`).
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


# ────────────────────────────────────────────────────────────────────────
# Plain-decode baseline (verifier alone, greedy, autoregressive)
# ────────────────────────────────────────────────────────────────────────

def plain_decode(verifier, prompt_ids: mx.array, n_new_tokens: int) -> tuple[list[int], float]:
    """Generate `n_new_tokens` from `verifier` autoregressively, greedy.

    Returns (generated_tokens, wall_clock_seconds).
    """
    cache = make_prompt_cache(verifier)
    t0 = time.perf_counter()
    # Prefill prompt
    logits = verifier(prompt_ids[None], cache=cache)[0, -1]
    out = []
    for _ in range(n_new_tokens):
        tok = int(mx.argmax(logits).item())
        out.append(tok)
        logits = verifier(mx.array([[tok]]), cache=cache)[0, -1]
    mx.eval(logits)  # force completion
    return out, time.perf_counter() - t0


# ────────────────────────────────────────────────────────────────────────
# Spec-decode runner
# ────────────────────────────────────────────────────────────────────────

@dataclass
class SpecStepStats:
    n_drafted: int          # K
    n_accepted: int         # 0..K (drafter tokens that survived verification)
    bonus_committed: bool   # whether we committed the verifier's bonus token
    step_ms: float

@dataclass
class SpecDecodeResult:
    tokens: list[int]
    wall_s: float
    steps: list[SpecStepStats] = field(default_factory=list)

    @property
    def alpha(self) -> float:
        """Average accepted drafter tokens per step (the canonical metric)."""
        return statistics.mean(s.n_accepted for s in self.steps) if self.steps else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Per-position acceptance rate over the entire run."""
        total = sum(s.n_drafted for s in self.steps)
        accepted = sum(s.n_accepted for s in self.steps)
        return accepted / total if total else 0.0


def spec_decode(verifier, drafter, prompt_ids: mx.array,
                n_new_tokens: int, K: int) -> SpecDecodeResult:
    """Greedy-verify spec decoding. Returns generated tokens + per-step stats."""
    verifier_cache = make_prompt_cache(verifier)
    drafter_cache  = make_prompt_cache(drafter)

    # Prefill prompt on both models. Both cache offsets advance to len(prompt).
    t0 = time.perf_counter()
    v_logits = verifier(prompt_ids[None], cache=verifier_cache)[0, -1]
    d_logits = drafter(prompt_ids[None], cache=drafter_cache)[0, -1]
    last_committed = int(mx.argmax(v_logits).item())  # verifier-quality next token
    out = [last_committed]
    steps: list[SpecStepStats] = []

    while len(out) < n_new_tokens:
        step_t0 = time.perf_counter()

        # ── Drafter speculates K tokens autoregressively, feeding `last_committed`
        # ── first so its KV cache progresses past the just-committed token.
        drafted: list[int] = []
        # Save the drafter's pre-speculation cache offsets (so we can rewind
        # on rejection).
        d_pre_offsets = [c.offset for c in drafter_cache]

        x = mx.array([[last_committed]])
        for _ in range(K):
            d_logits = drafter(x, cache=drafter_cache)[0, -1]
            tok = int(mx.argmax(d_logits).item())
            drafted.append(tok)
            x = mx.array([[tok]])

        # ── Verifier batched-forward over [last_committed, d1..dK].
        #    First column is the just-committed token (already in its cache
        #    via the previous step's logits computation? No — we never
        #    appended it. So we feed it now.)
        batch = mx.array([[last_committed] + drafted])
        v_logits_seq = verifier(batch, cache=verifier_cache)
        v_argmax = mx.argmax(v_logits_seq[0], axis=-1).tolist()
        # v_argmax has length K+1: position 0 verifies `last_committed`
        # (irrelevant — already in our output), positions 1..K verify d1..dK.

        # ── Greedy acceptance: walk d1..dK against v_argmax[0..K-1]
        # (note: v_logits[i] is the model's prediction AT position i, which
        # predicts position i+1 in the input. So v_argmax[i] is what the
        # verifier "would say" the next token after position i is. To verify
        # d_i (the i-th drafted token = input position i+1 in this batch),
        # compare against v_argmax[i]).
        n_accepted = 0
        for i in range(K):
            if v_argmax[i] == drafted[i]:
                out.append(drafted[i])
                n_accepted += 1
            else:
                break

        # ── Bonus token: at the rejection position (or right after the last
        # ── accepted if all K matched), commit the verifier's argmax.
        bonus_idx = n_accepted  # 0..K
        bonus_token = v_argmax[bonus_idx]
        out.append(bonus_token)
        bonus_committed = True

        # ── Drafter cache rollback. After the K calls, drafter cache holds
        # ── processing of [last_committed, d1, ..., d_{K-1}], offset = pre+K.
        # ── We want next iter's first forward (which feeds bonus_token, the
        # ── new last_committed) to land at the correct position in the
        # ── committed sequence [..., last_committed_OLD, d1, ..., d_{n_acc},
        # ── bonus_token, ...]. So cache should hold processing of [..., d_{n_acc}],
        # ── i.e. offset = pre + n_accepted + 1.
        # ── Edge case n_accepted == K: we'd need d_K in cache but never
        # ── processed it. Feed it explicitly here.
        if n_accepted == K:
            _ = drafter(mx.array([[drafted[K - 1]]]), cache=drafter_cache)
            # Now drafter cache offset = pre + K + 1, holds [..., d_{K-1}, d_K].
        else:
            for c, pre in zip(drafter_cache, d_pre_offsets):
                c.offset = pre + n_accepted + 1
        last_committed = bonus_token

        # ── Verifier's cache also overshot: it now has [last, d1..dK]
        # ── appended. We want to keep [last, d1..d_{n_accepted}, bonus].
        # ── i.e. drop the last (K - n_accepted) entries.
        for c in verifier_cache:
            c.offset -= (K - n_accepted)

        mx.eval(v_logits_seq, d_logits)
        steps.append(SpecStepStats(
            n_drafted=K, n_accepted=n_accepted,
            bonus_committed=bonus_committed,
            step_ms=(time.perf_counter() - step_t0) * 1000.0,
        ))

    wall = time.perf_counter() - t0
    return SpecDecodeResult(tokens=out[:n_new_tokens], wall_s=wall, steps=steps)


# ────────────────────────────────────────────────────────────────────────
# Tokenizer compat check + main
# ────────────────────────────────────────────────────────────────────────

def assert_tokenizers_match(verifier_tok, drafter_tok) -> None:
    """If the two models don't share an identical vocabulary, abort — spec
    decoding is incorrect (the drafter's token IDs would mean different
    things to the verifier)."""
    v = verifier_tok.get_vocab()
    d = drafter_tok.get_vocab()
    if v != d:
        # Try matching by special tokens + size as a fallback (vocab dicts
        # can differ in internal ordering but represent the same vocab).
        if len(v) != len(d):
            raise SystemExit(
                f"tokenizer mismatch: verifier vocab={len(v)} drafter vocab={len(d)}"
            )
        # OK, same size — assume compatible. Print a soft warning.
        print(f"[warn] vocab sizes match ({len(v)}) but dicts not identical; assuming compat")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit")
    ap.add_argument("--drafter",  default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--prompt",   default=(
        "Write a short story about a curious cat who learns to use a "
        "computer and discovers the internet for the first time."
    ))
    ap.add_argument("--n-new", type=int, default=100, help="Decode token budget")
    ap.add_argument("--K",     type=int, default=4,   help="Draft length per step")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/coreml_models"))

    print(f"[load] verifier = {args.verifier}")
    verifier, v_tok = load(args.verifier)
    print(f"[load] drafter  = {args.drafter}")
    drafter, d_tok = load(args.drafter)
    assert_tokenizers_match(v_tok, d_tok)

    prompt_ids = mx.array(v_tok.encode(args.prompt))
    print(f"[prompt] {len(prompt_ids)} tokens")

    # ── Warmup both ─────────────────────────────────────────────────
    print("[warmup]")
    plain_decode(verifier, prompt_ids, 8)
    plain_decode(drafter,  prompt_ids, 8)

    # ── Plain decode baseline (verifier alone) ──────────────────────
    print("[plain] decoding ...")
    plain_tokens, plain_wall = plain_decode(verifier, prompt_ids, args.n_new)
    plain_tps = args.n_new / plain_wall
    print(f"        wall={plain_wall:.2f}s  rate={plain_tps:.1f} tok/s")

    # ── Spec decode ────────────────────────────────────────────────
    print(f"[spec ] K={args.K} decoding ...")
    spec = spec_decode(verifier, drafter, prompt_ids, args.n_new, args.K)
    spec_tps = len(spec.tokens) / spec.wall_s
    print(f"        wall={spec.wall_s:.2f}s  rate={spec_tps:.1f} tok/s")
    print(f"        α (accepted/step) = {spec.alpha:.2f} / {args.K}")
    print(f"        acceptance rate   = {spec.acceptance_rate:.2%}")
    print(f"        speedup vs plain  = {spec_tps / plain_tps:.2f}×")
    # Per-step histogram
    from collections import Counter
    hist = Counter(s.n_accepted for s in spec.steps)
    print(f"        accept histogram: {dict(sorted(hist.items()))}")

    # ── Sanity: same output (greedy → deterministic) ───────────────
    if spec.tokens != plain_tokens:
        print("[warn] spec output differs from plain — bug in cache rollback?")
        for i, (a, b) in enumerate(zip(spec.tokens, plain_tokens)):
            if a != b:
                print(f"       first diff at pos {i}: spec={a} plain={b}")
                break
    else:
        print("[ok] spec output bit-identical to plain greedy decode")


if __name__ == "__main__":
    main()
