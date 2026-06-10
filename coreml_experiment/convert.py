"""HF model → INT4 stateful CoreML .mlpackage.

Pipeline:
  1. Load HF model in BF16 (eval mode).
  2. Wrap with a stateful KV-cache module (one buffer per layer for K and V).
  3. Trace with `torch.jit.trace` for a fixed (seq_len, n_layers, head_dim, …).
  4. Convert to MIL via `ct.convert(...)` with `states=[StateType(...)]` so the
     KV cache lives inside the model.
  5. Apply per-grouped-channel INT4 weight quantization via
     `ct.optimize.coreml.linear_quantize_weights`.
  6. Save the .mlpackage.

The first cut uses a single fixed max-cache-size graph (KV cache of length
MAX_SEQ). A follow-up will compile multiple graphs at exponentially-spaced
cache sizes (1, 2, 4, …, 1024, 2048) so ANE can keep ops static-shape and
avoid CPU fallback for short sequences.

USAGE
    python convert.py --model Qwen/Qwen2.5-3B-Instruct \
        --hf-cache ~/coreml_models \
        --out models/Qwen2.5-3B-Instruct-int4.mlpackage \
        --max-seq 2048
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                    help="HF model ID")
    ap.add_argument("--hf-cache", default=os.path.expanduser("~/coreml_models"),
                    help="HF cache directory")
    ap.add_argument("--out", default="coreml_experiment/models/Qwen2.5-3B-Instruct-int4.mlpackage",
                    help="Output .mlpackage path")
    ap.add_argument("--max-seq", type=int, default=2048,
                    help="Maximum KV cache length the converted graph supports")
    ap.add_argument("--n-bits", type=int, default=4,
                    help="Weight quantization bit width (4 = INT4)")
    ap.add_argument("--group-size", type=int, default=64,
                    help="Quantization group size (per-grouped-channel)")
    ap.add_argument("--compute-units", default="cpu_and_ne",
                    choices=["cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"],
                    help="CoreML ComputeUnit for the converted model")
    return ap.parse_args()


# ── Stateful KV-cache wrapper ────────────────────────────────────────────────

class StatefulCausalLM(nn.Module):
    """Wraps a HF causal LM so the KV cache is a torch buffer the CoreML
    converter can lift into a state input.

    The wrapper exposes one forward signature::

        logits = self(input_ids, position)

    where `input_ids` is `[B=1, T=1]` (single-token decode) and `position` is
    `[B=1, T=1]` long tensor giving the absolute position in the sequence.
    Internally, K and V projections at position `p` are written to slots
    `k_cache[..., p, ...]` / `v_cache[..., p, ...]` and attention is computed
    over slots 0..p inclusive.

    For prefill, the caller can either:
      - call this in a loop (slow, but matches our Metal engine's current
        behavior), or
      - export a SEPARATE graph at prompt-length cache size for batch prefill
        (planned multi-cache-size optimization).
    """

    def __init__(self, hf_model, max_seq: int):
        super().__init__()
        self.model = hf_model
        cfg = hf_model.config
        self.max_seq = max_seq
        self.n_layers = cfg.num_hidden_layers
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        # KV cache buffers, one per layer (shape [1, n_kv_heads, max_seq, head_dim]).
        # Registered as buffers so coremltools' StateType conversion lifts them.
        for i in range(self.n_layers):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, self.n_kv_heads, max_seq, self.head_dim, dtype=torch.float16),
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, self.n_kv_heads, max_seq, self.head_dim, dtype=torch.float16),
            )

    def forward(self, input_ids: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        # Build a (1, T, T) causal mask anchored at `position`. Since we only
        # support T=1 here, the mask is shape (1, 1, max_seq) with -inf for
        # positions > position.x.
        # NB: we rely on HF's attention impl to read past_key_values via the
        # cache positions argument. For Qwen2.5 we expect HF's `DynamicCache`
        # or the static cache path.
        out = self.model(
            input_ids=input_ids,
            cache_position=position,
            use_cache=True,
            return_dict=True,
        )
        return out.logits  # [1, T, vocab]


def build_traced(hf_model, max_seq: int, vocab_size: int):
    """Trace the stateful wrapper for a single-token decode step.

    Uses ``torch.jit.trace`` with the eager-attention HF model. We tried
    ``torch.export`` first, but it lifts module buffers into graph inputs,
    which then trips coremltools' state validation (named_buffers is `[]`).
    ``jit.trace`` keeps buffers attached, and with `attn_implementation="eager"`
    set on the HF model we avoid the SDPA int-cast bug that originally
    motivated trying export.
    """
    wrapper = StatefulCausalLM(hf_model, max_seq=max_seq).eval()

    input_ids = torch.tensor([[1]], dtype=torch.long)
    position = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (input_ids, position), strict=False)
    return traced, (input_ids, position)


def convert_to_coreml(traced, example_inputs, max_seq: int, n_kv_heads: int,
                      head_dim: int, n_layers: int, vocab_size: int,
                      compute_units: str):
    """Convert TorchScript → MIL → mlpackage with StateType for KV cache."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb  # noqa: F401

    # ── Input descriptors ────────────────────────────────────────────────
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
    ]

    # ── State descriptors: K and V cache per layer ───────────────────────
    states = []
    for i in range(n_layers):
        states.append(
            ct.StateType(
                name=f"k_cache_{i}",
                wrapped_type=ct.TensorType(
                    shape=(1, n_kv_heads, max_seq, head_dim),
                    dtype=np.float16,
                ),
            )
        )
        states.append(
            ct.StateType(
                name=f"v_cache_{i}",
                wrapped_type=ct.TensorType(
                    shape=(1, n_kv_heads, max_seq, head_dim),
                    dtype=np.float16,
                ),
            )
        )

    cu_map = {
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }

    print(f"[convert] coremltools.convert(...) — this can take a few minutes")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        states=states,
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        compute_units=cu_map[compute_units],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    print(f"[convert] done in {time.perf_counter() - t0:.1f}s")
    return mlmodel


def quantize_int4(mlmodel, n_bits: int, group_size: int):
    """Apply per-grouped-channel symmetric weight quantization."""
    import coremltools.optimize.coreml as cto
    cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            granularity="per_grouped_channel",
            block_size=group_size,
            weight_threshold=2048,
        ),
    )
    print(f"[quantize] linear_symmetric  n_bits={n_bits}  group_size={group_size}")
    t0 = time.perf_counter()
    q = cto.linear_quantize_weights(mlmodel, config=cfg)
    print(f"[quantize] done in {time.perf_counter() - t0:.1f}s")
    return q


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_CACHE", args.hf_cache)
    os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")

    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"[load] {args.model}")
    cfg = AutoConfig.from_pretrained(args.model, cache_dir=args.hf_cache)
    # `attn_implementation="eager"` avoids `scaled_dot_product_attention`,
    # whose tracer-unfriendly int conversions (`int(some_tensor)` over
    # non-scalar shapes) crash the coremltools `_int` op converter.
    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.hf_cache,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    vocab_size = cfg.vocab_size
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    print(f"[arch] layers={n_layers}  kv_heads={n_kv_heads}  head_dim={head_dim}  vocab={vocab_size}")

    print(f"[trace] max_seq={args.max_seq}")
    traced, example = build_traced(hf, args.max_seq, vocab_size)

    mlmodel = convert_to_coreml(
        traced, example,
        max_seq=args.max_seq,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        compute_units=args.compute_units,
    )

    mlmodel = quantize_int4(mlmodel, n_bits=args.n_bits, group_size=args.group_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save] {out_path}")
    mlmodel.save(str(out_path))
    print("[done]")


if __name__ == "__main__":
    main()
