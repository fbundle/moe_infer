#!/usr/bin/env python3
"""Verify Qwen35Dense engine vs HF transformers (BF16) on the full 4B model.

Same shape as verify_vs_original.py (the canonical qwen35_moe check), but
runs the WHOLE model — no strip needed, 4B fits in memory.

Usage:
    python -m moe_infer.helpers.verify_qwen35_dense
"""

import json
import os
import sys
import time

import numpy as np

HF_DIR  = "/Volumes/Hippopotamus/vault/code/moe_infer/data/Qwen3.5-4B/source"
INT4_DIR = "/Volumes/Hippopotamus/vault/code/moe_infer/data/Qwen3.5-4B/model_int4"
BQ4_DIR  = "/Volumes/Hippopotamus/vault/code/moe_infer/data/Qwen3.5-4B/model_bq4"

# A 29-token sequence — matches the qwen35_moe verify shape.
# Decodes to a Q/A style probe that exercises both linear-attn (24 layers)
# and full-attn (8 layers).
TOKENS = [
    760, 6511, 314, 9338, 369, 117, 13, 281, 1043, 11,
    369, 282, 1043, 281, 9338, 28, 117, 198, 9338, 314,
    260, 16043, 1216, 280, 9338, 13, 1043, 314, 260,
]


def run_transformers():
    import torch
    from transformers import AutoModelForCausalLM
    print("[v] Loading HF Qwen3.5-4B BF16...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(HF_DIR, dtype=torch.bfloat16)
    model.eval()
    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
    logits = out.logits[0, -1].float().numpy()
    elapsed = time.time() - t0
    print(f"[v] transformers (BF16)            : {elapsed*1000:7.0f} ms  "
          f"min={logits.min():+.4f} max={logits.max():+.4f} "
          f"mean={logits.mean():+.4f} NaNs={np.isnan(logits).sum()}")
    return logits


def run_rust(weights_dir, label):
    import _moe_infer_rs as _rs
    t0 = time.time()
    model = _rs.Model(weights_dir)
    engine = _rs.Engine(model, "Qwen35DenseFused", 0, expert_cache_count=0)
    cache = _rs.Cache(model)
    ids_arr = np.array(TOKENS, dtype=np.int64)
    logits_all = engine.forward(ids_arr, cache)
    elapsed = time.time() - t0
    logits = np.array(logits_all[-1], dtype=np.float32)
    print(f"[v] Rust {label:<26s}: {elapsed*1000:7.0f} ms  "
          f"min={logits.min():+.4f} max={logits.max():+.4f} "
          f"mean={logits.mean():+.4f} NaNs={np.isnan(logits).sum()}")
    return logits


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def compare(label1, logits1, label2, logits2):
    n = min(len(logits1), len(logits2))
    a = logits1[:n].astype(np.float64)
    b = logits2[:n].astype(np.float64)
    diff = np.abs(a - b)
    idx = int(diff.argmax())
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    pa = _softmax(a); pb = _softmax(b)
    a_top1 = int(np.argmax(a)); b_top1 = int(np.argmax(b))
    def top_k(p, k): return np.argpartition(p, -k)[-k:]
    o10 = len(set(top_k(pa, 10).tolist()) & set(top_k(pb, 10).tolist()))
    o50 = len(set(top_k(pa, 50).tolist()) & set(top_k(pb, 50).tolist()))
    eps_p = 1e-30
    kl_ab = float(np.sum(pa * (np.log(pa + eps_p) - np.log(pb + eps_p))))
    kl_ba = float(np.sum(pb * (np.log(pb + eps_p) - np.log(pa + eps_p))))
    tv = float(0.5 * np.sum(np.abs(pa - pb)))

    print(f"\n  {label1} vs {label2}:")
    print(f"    max_diff = {diff.max():.6f} at idx {idx} "
          f"({label1}={a[idx]:.4f}, {label2}={b[idx]:.4f})")
    print(f"    mean_diff = {diff.mean():.6f}")
    print(f"    cosine_sim = {cos:.8f}")
    print(f"    top-1 match: {'YES' if a_top1 == b_top1 else 'NO '} "
          f"({label1}_top1={a_top1}, {label2}_top1={b_top1})")
    print(f"    top-10 overlap: {o10}/10    top-50 overlap: {o50}/50")
    print(f"    KL({label1}||{label2}) = {kl_ab:.6f}    "
          f"KL({label2}||{label1}) = {kl_ba:.6f}")
    print(f"    total variation: {tv:.6f}")


def main():
    print("=" * 70)
    print("Verification: HF BF16 Qwen3.5-4B vs Rust Qwen35DenseFused (full model)")
    print(f"Tokens: {len(TOKENS)} ({TOKENS[:6]}...)")
    print("=" * 70)

    hf = run_transformers()
    print()

    if os.path.isdir(INT4_DIR):
        rust_int4 = run_rust(INT4_DIR, "INT4")
        compare("transformers", hf, "INT4", rust_int4)

    if os.path.isdir(BQ4_DIR):
        rust_bq4 = run_rust(BQ4_DIR, "BQ4")
        compare("transformers", hf, "BQ4", rust_bq4)


if __name__ == "__main__":
    main()
