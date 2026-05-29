#!/usr/bin/env python3
"""Compare _moe_infer_rs logits against dequantized HF model on stripped model."""

import json
import os
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

HF_DIR = os.path.join(ROOT, "hub", "models--Qwen--Qwen3.6-35B-A3B-Strip-Dequant")
BQ4_DIR = os.path.join(ROOT, "data", "Qwen3.6-35B-A3B-Strip", "model_bq4")

TOKENS = [
    248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
    26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
    488, 30, 248046, 198, 248045, 74455, 198, 248068, 198,
]


def run_transformers():
    """Load dequantized HF model via transformers, return last-position logits."""
    import torch
    from transformers import AutoModelForCausalLM

    print("[nway] Loading dequantized HF model...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(HF_DIR, dtype=torch.bfloat16)
    model.eval()

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
    logits = out.logits[0, -1].float().numpy()

    elapsed = time.time() - t0
    print(f"[nway] transformers     : {elapsed*1000:5.0f} ms  "
          f"min={logits.min():.4f} max={logits.max():.4f} "
          f"mean={logits.mean():.4f} NaNs={np.isnan(logits).sum()}")
    return logits


def run_rust(mode="Qwen35MoEFusedExp2"):
    """Load BQ4 model via _moe_infer_rs, return last-position logits."""
    import _moe_infer_rs as _rs  # type: ignore[import-untyped]

    # Patch config for StrippedModel dispatch, restore after
    cfg_path = os.path.join(BQ4_DIR, "config.json")
    with open(cfg_path) as f:
        orig = json.load(f)
    patched = dict(orig)
    patched["architectures"] = ["Qwen3_5MoeForConditionalGeneration_Stripped"]
    with open(cfg_path, "w") as f:
        json.dump(patched, f)

    try:
        t0 = time.time()

        model = _rs.Model(BQ4_DIR)
        engine = _rs.Engine(model, mode, 0)
        cache = _rs.Cache(model)

        ids_arr = np.array(TOKENS, dtype=np.int64)
        logits_all = engine.forward(ids_arr, cache)

        elapsed = time.time() - t0
        logits = np.array(logits_all[-1], dtype=np.float32)

        print(f"[nway] Rust {mode:<12}: {elapsed*1000:5.0f} ms  "
              f"min={logits.min():.4f} max={logits.max():.4f} "
              f"mean={logits.mean():.4f} NaNs={np.isnan(logits).sum()}")
        return logits
    finally:
        with open(cfg_path, "w") as f:
            json.dump(orig, f)


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def compare(label1, logits1, label2, logits2, eps=1e-3):
    """Compare two logit arrays — raw-logit + distribution metrics."""
    n = min(len(logits1), len(logits2))
    a = logits1[:n].astype(np.float64)
    b = logits2[:n].astype(np.float64)
    diff = np.abs(a - b)
    idx = diff.argmax()

    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)
    rel_diff = diff / denom

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = float(np.dot(a, b) / max(a_norm * b_norm, 1e-12))

    match = int((diff < eps).sum())

    # ── Distribution metrics on softmax(logits) ──────────────────────────
    pa = _softmax(a)
    pb = _softmax(b)

    # Top-k agreement
    def top_k(p, k):
        return np.argpartition(p, -k)[-k:]
    a_top1 = int(np.argmax(a))
    b_top1 = int(np.argmax(b))
    top1_match = (a_top1 == b_top1)

    def overlap(k):
        return len(set(top_k(pa, k).tolist()) & set(top_k(pb, k).tolist()))

    o10, o50 = overlap(10), overlap(50)

    # KL(P_a || P_b) and KL(P_b || P_a) — clip to avoid log(0)
    eps_p = 1e-30
    kl_ab = float(np.sum(pa * (np.log(pa + eps_p) - np.log(pb + eps_p))))
    kl_ba = float(np.sum(pb * (np.log(pb + eps_p) - np.log(pa + eps_p))))

    # Total variation distance: 0.5 * sum |pa - pb|, range [0, 1]
    tv = float(0.5 * np.sum(np.abs(pa - pb)))

    # Probability mass that B puts on A's top-k (and vice versa)
    a_top10_idx = top_k(pa, 10)
    b_top10_idx = top_k(pb, 10)
    mass_b_on_a_top10 = float(pb[a_top10_idx].sum())
    mass_a_on_b_top10 = float(pa[b_top10_idx].sum())
    mass_a_top10 = float(pa[a_top10_idx].sum())  # for reference
    mass_b_top10 = float(pb[b_top10_idx].sum())

    print(f"\n  {label1} vs {label2}:")
    print(f"    -- raw logits --")
    print(f"    max_diff={diff.max():.6f} at idx {idx}"
          f" ({label1}={a[idx]:.6f}, {label2}={b[idx]:.6f})")
    print(f"    mean_diff={diff.mean():.6f}")
    print(f"    max_rel_diff={rel_diff.max():.6f}")
    print(f"    cosine_sim={cos:.8f}")
    print(f"    within {eps}: {match}/{n} ({100.*match/n:.1f}%)")
    print(f"    -- distribution (softmax) --")
    print(f"    top-1 match:      {'YES' if top1_match else 'NO '}"
          f"  ({label1}_top1={a_top1}, {label2}_top1={b_top1})")
    print(f"    top-10 overlap:   {o10}/10")
    print(f"    top-50 overlap:   {o50}/50")
    print(f"    KL({label1}||{label2}) = {kl_ab:.6f}")
    print(f"    KL({label2}||{label1}) = {kl_ba:.6f}")
    print(f"    total variation:  {tv:.6f}")
    print(f"    mass({label2}) on {label1}'s top-10:"
          f" {mass_b_on_a_top10:.4f}  (vs self={mass_a_top10:.4f})")
    print(f"    mass({label1}) on {label2}'s top-10:"
          f" {mass_a_on_b_top10:.4f}  (vs self={mass_b_top10:.4f})")
    return diff.max()


def main():
    print("=" * 60)
    print("Logit Verification: dequantized HF vs _moe_infer_rs")
    print(f"Tokens: {len(TOKENS)}")
    print("=" * 60)

    hf = run_transformers()

    for mode in ("Qwen35MoEFusedExp2", "Qwen35MoEFusedExp1"):
        try:
            rust = run_rust(mode)
            compare("transformers", hf, mode, rust)
        except Exception as e:
            print(f"\n[nway] Rust {mode}: FAILED — {e}")


if __name__ == "__main__":
    main()
