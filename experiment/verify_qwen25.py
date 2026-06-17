"""Verify our minimal Qwen2.5 matches HF transformers on a real input.

If this passes, we have a clean PyTorch model under our control that we can
then trace + convert to CoreML without fighting HF's complex internals.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from qwen25 import build_qwen25


def main():
    hf_cache = os.path.expanduser("~/coreml_models")
    snap = Path(hf_cache) / "models--Qwen--Qwen2.5-3B-Instruct" / "snapshots"
    hf_dir = next(snap.iterdir())

    print(f"[load] minimal qwen25 from {hf_dir}")
    model, (cos, sin) = build_qwen25(hf_dir, max_seq=128)

    # Drive both ours and HF with the same single-token input.
    input_ids = [9707, 11, 1879, 0]  # "Hello, world!"
    cos_pt = cos
    sin_pt = sin

    # ── Our forward (token-at-a-time decode) ───────────────────────
    k_caches = [torch.zeros(1, model.cfg["num_key_value_heads"], 128,
                            model.cfg["head_dim"], dtype=torch.float16)
                for _ in range(model.cfg["num_hidden_layers"])]
    v_caches = [t.clone() for t in k_caches]

    ours_logits_per_step = []
    with torch.no_grad():
        for pos, tok in enumerate(input_ids):
            t = torch.tensor([[tok]], dtype=torch.long)
            logits, k_caches, v_caches = model(t, pos, k_caches, v_caches, cos_pt, sin_pt)
            ours_logits_per_step.append(logits[0, 0].float().cpu().numpy())

    # ── HF reference (single forward over all tokens) ──────────────
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", cache_dir=hf_cache,
        dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    with torch.no_grad():
        out = hf(torch.tensor([input_ids]))
    hf_logits = out.logits[0].float().cpu().numpy()

    print(f"\n{'step':>5} {'token':>6} {'cos_sim':>10} {'top1_match':>12}  ours_top1  hf_top1")
    for i, tok in enumerate(input_ids):
        a = ours_logits_per_step[i]
        b = hf_logits[i]
        cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))
        top1_ours = int(np.argmax(a))
        top1_hf = int(np.argmax(b))
        match = "✓" if top1_ours == top1_hf else "✗"
        print(f"{i:>5} {tok:>6} {cos:>10.4f} {match:>12}  {top1_ours:>9} {top1_hf:>8}")


if __name__ == "__main__":
    main()
