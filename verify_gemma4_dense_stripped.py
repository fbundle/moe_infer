#!/usr/bin/env python3
"""Verify gemma4_dense engine vs HF transformers on the 6-layer stripped model.

Setup (already done):
  * hub/models--google--gemma-4-12B-Strip/   -- 6-layer BF16 HF safetensors
  * data/Gemma-4-12B-Strip/model_int4/        -- 6-layer INT4 our quantize

The stripped model keeps the original `Gemma4UnifiedForConditionalGeneration`
arch in config.json so transformers loads it directly. When loading via our
engine, we temporarily patch the arch string to `..._Stripped` so the dispatch
hits `FusedGemma4Dense::<Gemma4Dense12BStripped>`.

This is the same pattern as `verify_nway.py` (for qwen35_moe) but tuned for
the Gemma-4-12B-dense / 6-layer / Gemma4Unified arch.
"""

import json
import os
import time

import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
HF_DIR = os.path.join(ROOT, "hub", "models--google--gemma-4-12B-Strip")
OUR_DIR = os.path.join(ROOT, "data", "Gemma-4-12B-Strip", "model_int4")

TOKENS = [1, 7, 100, 1000, 24]


def run_transformers():
    print("[verify] loading stripped HF model via transformers...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        HF_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[verify] transformers loaded in {time.time()-t0:.1f}s")

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    logits = out.logits[0, -1].float().cpu().numpy()
    print(f"[verify] transformers logits: shape={logits.shape}  "
          f"min={logits.min():.3f} max={logits.max():.3f}  "
          f"NaNs={int(np.isnan(logits).sum())}")
    # Hidden states: tuple of (n_layers+1,) tensors, each [batch, seq, hidden]
    hs = [h[0, -1].float().cpu().numpy() for h in out.hidden_states]
    return logits, hs


def run_ours():
    print("[verify] loading stripped INT4 model via our engine...")
    import _moe_infer_rs as _rs

    # Patch config to _Stripped so the dispatch hits Gemma4Dense12BStripped.
    cfg_path = os.path.join(OUR_DIR, "config.json")
    with open(cfg_path) as f:
        orig = json.load(f)
    patched = dict(orig)
    patched["architectures"] = ["Gemma4UnifiedForConditionalGeneration_Stripped"]
    with open(cfg_path, "w") as f:
        json.dump(patched, f, indent=2)

    try:
        t0 = time.time()
        m = _rs.Model(OUR_DIR)
        eng = _rs.Engine(m, "Gemma4DenseFused", 0)
        cache = _rs.Cache(m)
        print(f"[verify] our engine init in {time.time()-t0:.1f}s")

        t0 = time.time()
        logits_all = eng.forward(np.array(TOKENS, dtype=np.int64), cache, mtp=False)
        elapsed = time.time() - t0

        logits = np.asarray(logits_all[-1], dtype=np.float32)
        print(f"[verify] our engine {len(TOKENS)} toks in {elapsed*1000:.0f} ms  "
              f"logits shape={logits.shape}  "
              f"min={logits.min():.3f} max={logits.max():.3f}  "
              f"NaNs={int(np.isnan(logits).sum())}")
        return logits
    finally:
        with open(cfg_path, "w") as f:
            json.dump(orig, f, indent=2)


def compare(a_label, a, b_label, b):
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a_f = a[finite]; b_f = b[finite]
    if len(a_f) < n:
        print(f"  WARN: {n - len(a_f)} non-finite positions excluded")

    cos = float(np.dot(a_f, b_f) / max(
        np.linalg.norm(a_f) * np.linalg.norm(b_f), 1e-12))
    diff = np.abs(a_f - b_f)

    def softmax(x):
        x = x - x.max(); e = np.exp(x); return e / e.sum()
    pa = softmax(a); pb = softmax(b)
    def topk(p, k): return set(np.argpartition(p, -k)[-k:].tolist())
    o10 = len(topk(pa, 10) & topk(pb, 10))

    print(f"\n  {a_label} vs {b_label}")
    print(f"    cos_sim         = {cos:.6f}")
    print(f"    max_abs_diff    = {diff.max():.4f}")
    print(f"    mean_abs_diff   = {diff.mean():.4f}")
    print(f"    {a_label}_top1  = {int(np.argmax(a))}")
    print(f"    {b_label}_top1  = {int(np.argmax(b))}")
    print(f"    top10 overlap   = {o10}/10")
    return cos


def main():
    print("=" * 64)
    print("Verify (stripped): transformers (BF16) vs our INT4 engine")
    print(f"Tokens: {TOKENS}")
    print("=" * 64)

    # Our engine FIRST (frees memory before transformers loads its big BF16).
    ours = run_ours()
    print()
    hf, hf_hidden = run_transformers()

    cos = compare("transformers", hf, "ours", ours)
    print()
    print(f"hidden_states from HF: {len(hf_hidden)} layers (embed + 6 decoder)")
    for i, h in enumerate(hf_hidden):
        print(f"  layer {i}: shape={h.shape}  norm={np.linalg.norm(h):.3f}")

    if cos > 0.99:
        print("\nPASS: cos_sim > 0.99")
    elif cos > 0.5:
        print(f"\nPARTIAL: cos_sim {cos:.3f}")
    else:
        print(f"\nFAIL: cos_sim {cos:.3f}")


if __name__ == "__main__":
    main()
