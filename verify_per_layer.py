#!/usr/bin/env python3
"""Per-layer cos-sim comparison vs transformers (stripped 6-layer Gemma-4-12B).

For each STOP_LAYER in 0..=6:
  1. Run our engine with GEMMA4_DENSE_STOP_LAYER=N.
  2. Compute HF's equivalent: take hidden_states[N] (post-layer N-1 output),
     pass through HF's `language_model.norm` + tied lm_head + softcap.
  3. Compare logits cos sim, top-1 agreement.

This pinpoints WHICH layer first introduces a large divergence. The expected
pattern when only the final full-attn layer is broken is: cos_sim stays >0.9
through STOP=5, then drops sharply at STOP=6.
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


def run_ours(stop_layer):
    import _moe_infer_rs as _rs

    cfg_path = os.path.join(OUR_DIR, "config.json")
    with open(cfg_path) as f:
        orig = json.load(f)
    patched = dict(orig)
    patched["architectures"] = ["Gemma4UnifiedForConditionalGeneration_Stripped"]
    with open(cfg_path, "w") as f:
        json.dump(patched, f, indent=2)
    try:
        m = _rs.Model(OUR_DIR)
        eng = _rs.Engine(m, "Gemma4DenseFused", 0)
        cache = _rs.Cache(m)
        os.environ["GEMMA4_DENSE_STOP_LAYER"] = str(stop_layer)
        logits = eng.forward(np.array(TOKENS, dtype=np.int64), cache, mtp=False)
        return np.asarray(logits[-1], dtype=np.float32)
    finally:
        os.environ.pop("GEMMA4_DENSE_STOP_LAYER", None)
        with open(cfg_path, "w") as f:
            json.dump(orig, f, indent=2)


def hf_partial_logits(stop_layer):
    """Run HF transformers forward, then take hidden_states[stop_layer] and
    project it through model.norm + tied lm_head + softcap to get the
    equivalent of 'STOP_LAYER=N then final_norm + lm_head + softcap'."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(HF_DIR, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([TOKENS]), output_hidden_states=True)
        # hidden_states[i] = post-layer-(i-1). hidden_states[0] = embed.
        # We want hidden_states[stop_layer]: post-(stop_layer-1)th layer.
        hs = out.hidden_states[stop_layer][0, -1]
        # Apply final norm
        normed = model.model.language_model.norm(hs.unsqueeze(0).unsqueeze(0))
        # Tied lm_head (transformers does it inside `model.lm_head`)
        logits = model.lm_head(normed)
        # Softcap (read from config)
        cap = float(model.config.text_config.final_logit_softcapping or 30.0)
        logits = torch.tanh(logits / cap) * cap
        return logits[0, 0].float().cpu().numpy()


def compare(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a_f = a[finite]; b_f = b[finite]
    cos = float(np.dot(a_f, b_f) / max(
        np.linalg.norm(a_f) * np.linalg.norm(b_f), 1e-12))
    return cos, int(np.argmax(a)), int(np.argmax(b))


def main():
    print("=" * 70)
    print("Per-layer comparison: gemma4_dense (INT4) vs transformers (BF16)")
    print(f"Tokens: {TOKENS}")
    print("=" * 70)

    print(f"\n{'stop_layer':>10}  {'hf_top1':>10}  {'ours_top1':>10}  {'cos_sim':>10}")
    for stop in range(0, 7):
        hf = hf_partial_logits(stop)
        ours = run_ours(stop)
        cos, hf_top, ours_top = compare(hf, ours)
        marker = " <-- divergence" if cos < 0.9 else ""
        print(f"{stop:>10}  {hf_top:>10}  {ours_top:>10}  {cos:>10.4f}{marker}")


if __name__ == "__main__":
    main()
