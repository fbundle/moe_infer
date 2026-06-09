#!/usr/bin/env python3
"""Compare our gemma4_dense engine logits against llama-cpp-python's GGUF.

The reference (llama.cpp via llama-cpp-python) runs the same GGUF (q4_0)
that's also our INT4 quantization source: the dequantized values should
match to within numerical noise. If our engine matches llama.cpp's logits
within cos_sim ~0.99 and top-1 agreement, the implementation is correct.

Usage:
    python verify_gemma4_dense.py
"""

import os
import time

import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
GGUF_PATH = os.path.join(
    ROOT, "hub", "models--google--gemma-4-12B-it-qat-q4_0-gguf",
    "snapshots", "f6e7774e6148da3b7f201e42ba37cf084c1db35f",
    "gemma-4-12b-it-qat-q4_0.gguf",
)
OUR_DIR = os.path.join(ROOT, "data", "Gemma-4-12B", "model_int4")

# Small token set — keep short so per-position comparison is cheap.
# Using lowercase ASCII / numbers to avoid tokenizer-specific issues.
TOKENS = [1, 7, 100, 1000, 24]


def run_llamacpp():
    """Run llama-cpp-python with logits_all=True, return final-position logits."""
    print("[verify] loading GGUF with llama-cpp-python (Metal)...")
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(
        model_path=GGUF_PATH,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
        logits_all=True,  # required to get per-token logits
    )
    print(f"[verify] loaded in {time.time()-t0:.1f}s")

    # Use the raw eval API so we can pass token IDs and read logits at any position.
    # llama_cpp.Llama exposes `eval(tokens)` which advances internal state and
    # populates `llm.scores` for every position when logits_all=True.
    t0 = time.time()
    llm.reset()
    llm.eval(TOKENS)
    elapsed = time.time() - t0

    # Final-position logits.
    # llama_cpp stores scores as a numpy array [n_ctx, vocab_size] when
    # logits_all=True; the last evaluated position is at index n_eval-1.
    n = len(TOKENS)
    scores = np.asarray(llm.eval_logits, dtype=np.float32)
    logits = scores[-1]
    print(f"[verify] llama.cpp eval {n} toks in {elapsed*1000:.0f} ms  "
          f"logits shape={logits.shape}  "
          f"min={logits.min():.3f} max={logits.max():.3f}  "
          f"NaNs={int(np.isnan(logits).sum())}")
    return logits


def run_ours():
    """Run our gemma4_dense engine on the same token IDs."""
    print("[verify] loading our INT4 model + engine...")
    import _moe_infer_rs as _rs
    t0 = time.time()
    m = _rs.Model(OUR_DIR)
    eng = _rs.Engine(m, "Gemma4DenseFused", 0)
    cache = _rs.Cache(m)
    print(f"[verify] engine init in {time.time()-t0:.1f}s")

    t0 = time.time()
    logits_all = eng.forward(np.array(TOKENS, dtype=np.int64), cache, mtp=False)
    elapsed = time.time() - t0

    logits = np.asarray(logits_all[-1], dtype=np.float32)
    print(f"[verify] our engine {len(TOKENS)} toks in {elapsed*1000:.0f} ms  "
          f"logits shape={logits.shape}  "
          f"min={logits.min():.3f} max={logits.max():.3f}  "
          f"NaNs={int(np.isnan(logits).sum())}")
    return logits


def compare(a_label, a, b_label, b):
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < n:
        print(f"  WARN: {n - int(finite.sum())} non-finite positions "
              "excluded from comparison")
    a_f = a[finite]; b_f = b[finite]

    cos = float(np.dot(a_f, b_f) / max(
        np.linalg.norm(a_f) * np.linalg.norm(b_f), 1e-12))
    diff = np.abs(a_f - b_f)
    max_diff_idx = int(np.argmax(diff))
    a_top1 = int(np.argmax(a))
    b_top1 = int(np.argmax(b))

    def softmax(x):
        x = x - x.max()
        e = np.exp(x); return e / e.sum()
    pa = softmax(a); pb = softmax(b)
    def topk(p, k): return set(np.argpartition(p, -k)[-k:].tolist())
    o10 = len(topk(pa, 10) & topk(pb, 10))

    print(f"\n  {a_label} vs {b_label}")
    print(f"    cos_sim            = {cos:.6f}")
    print(f"    max_abs_diff       = {diff.max():.4f}  (at finite-idx {max_diff_idx})")
    print(f"    mean_abs_diff      = {diff.mean():.4f}")
    print(f"    {a_label}_top1     = {a_top1}")
    print(f"    {b_label}_top1     = {b_top1}")
    print(f"    top10 overlap      = {o10}/10")
    return cos


def main():
    print("=" * 60)
    print("Verify: gemma4_dense (our INT4) vs llama.cpp (q4_0 GGUF)")
    print(f"Token IDs: {TOKENS}")
    print("=" * 60)

    ours = run_ours()
    print()
    llamacpp = run_llamacpp()

    cos = compare("llamacpp", llamacpp, "ours", ours)

    if cos > 0.99:
        print("\nPASS: cos_sim > 0.99 — engines agree to numerical noise.")
    elif cos > 0.5:
        print("\nPARTIAL: cos_sim > 0.5 but < 0.99 — directional agreement, "
              "but quantization or layer-level bugs present.")
    else:
        print("\nFAIL: cos_sim too low — engine has structural bugs to fix.")


if __name__ == "__main__":
    main()
