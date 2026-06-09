#!/usr/bin/env python3
"""Direct layer-by-layer hidden-state comparison vs transformers.

Runs our engine ONCE with GEMMA4_DENSE_CAPTURE_HIDDEN=1. The engine writes the
residual stream into buf_logits after each layer (and the initial embed at
slot 0). We unpack and compare against transformers.output_hidden_states
element-by-element.

This pinpoints WHICH layer first diverges and by how much — sharper than
comparing final-projected logits.
"""

import json
import os
import time

import numpy as np
import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
HF_DIR = os.path.join(ROOT, "hub", "models--google--gemma-4-12B-Strip")
OUR_DIR = os.path.join(ROOT, "data", "Gemma-4-12B-Strip", "model_int4")
TOKENS = [1, 7, 100, 1000, 24]

HIDDEN = 3840
N_LAYERS = 6


def run_ours_hidden():
    """Returns shape (N_LAYERS+1, HIDDEN). Slot 0 = post-embed-scaling; slot k+1 = post-layer-k."""
    import _moe_infer_rs as _rs

    cfg_path = os.path.join(OUR_DIR, "config.json")
    with open(cfg_path) as f:
        orig = json.load(f)
    patched = dict(orig)
    patched["architectures"] = ["Gemma4UnifiedForConditionalGeneration_Stripped"]
    with open(cfg_path, "w") as f:
        json.dump(patched, f, indent=2)

    try:
        os.environ["GEMMA4_DENSE_CAPTURE_HIDDEN"] = "1"
        m = _rs.Model(OUR_DIR)
        eng = _rs.Engine(m, "Gemma4DenseFused", 0)
        cache = _rs.Cache(m)
        # Engine writes hidden states into buf_logits; "logits" returned here
        # is really the buf_logits content as a (n_tokens, vocab_size) array.
        logits = eng.forward(np.array(TOKENS, dtype=np.int64), cache, mtp=False)
        # The capture happens once per token-forward; we want the LAST token's
        # captured states (the engine reuses buf_logits across tokens, so the
        # last one is what's left).
        flat = np.asarray(logits[-1], dtype=np.float32)
        hidden = flat[: (N_LAYERS + 1) * HIDDEN].reshape(N_LAYERS + 1, HIDDEN)
        return hidden
    finally:
        os.environ.pop("GEMMA4_DENSE_CAPTURE_HIDDEN", None)
        with open(cfg_path, "w") as f:
            json.dump(orig, f, indent=2)


def run_hf_hidden():
    """Returns shape (N_LAYERS+1, HIDDEN). Slot 0..N-1 = output of layer i-1
    (matches our captures). Slot N = output of LAST layer PRE-FINAL-NORM
    (transformers' hs[N] is POST-final-norm, which we don't capture)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        HF_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    captured = [None] * (N_LAYERS + 1)
    def mk_hook(idx):
        def h(mod, args, output):
            o = output[0] if isinstance(output, tuple) else output
            captured[idx + 1] = o[0, -1].float().cpu().numpy()
        return h
    for i, L in enumerate(model.model.language_model.layers):
        L.register_forward_hook(mk_hook(i))

    with torch.no_grad():
        out = model(torch.tensor([TOKENS]), output_hidden_states=True)
    captured[0] = out.hidden_states[0][0, -1].float().cpu().numpy()
    return np.stack(captured, axis=0).astype(np.float32)


def compare_row(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
    return cos, rel


def main():
    print("Loading our engine (with hidden-state capture) ...")
    t0 = time.time()
    ours = run_ours_hidden()
    print(f"  ours captured in {time.time()-t0:.1f}s  shape={ours.shape}")

    print("\nLoading HF transformers ...")
    t0 = time.time()
    hf = run_hf_hidden()
    print(f"  HF captured in {time.time()-t0:.1f}s  shape={hf.shape}")

    print(f"\n{'slot':>5} {'meaning':<22} {'cos_sim':>10} {'rel_err':>10} {'our_norm':>10} {'hf_norm':>10}")
    print("-" * 70)
    for k in range(N_LAYERS + 1):
        meaning = "embed (post-scale)" if k == 0 else f"post-layer-{k-1}"
        cos, rel = compare_row(ours[k], hf[k])
        on = float(np.linalg.norm(ours[k]))
        hn = float(np.linalg.norm(hf[k]))
        marker = "  <-- diverges" if cos < 0.99 else ""
        print(f"{k:>5} {meaning:<22} {cos:>10.6f} {rel:>10.4f} {on:>10.3f} {hn:>10.3f}{marker}")


if __name__ == "__main__":
    main()
