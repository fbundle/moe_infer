# coreml_experiment — Current state

## What works (validated on Apple M4, macOS 26.4.1, coremltools 9.0)

- **Toolchain**: Python 3.13 venv (`.venv-coreml/`), coremltools 9.0 with torch 2.7.
- **All compute devices visible**: `MLNeuralEngineComputeDevice`, `MLGPUComputeDevice`, `MLCPUComputeDevice`.
- **End-to-end pipeline on a tiny model** (`smoke_test.py`): TinyMLP (Linear → ReLU → Linear) → `torch.jit.trace` → `ct.convert` → `linear_quantize_weights` (INT4, group=64, symmetric) → `.mlpackage` → load with each ComputeUnit. INT4 rel_err 0.0073 vs reference (excellent).

## What's blocked

**Full HF Qwen2.5-3B-Instruct conversion**. Two paths attempted:

1. **`torch.jit.trace(hf_model)`** — fails inside coremltools' `_int` op converter (`TypeError: only 0-dimensional arrays can be converted to Python scalars`). HF's transformer code does `int(some_tensor)` somewhere on a non-scalar tensor; the converter's constant-folding chokes. Setting `attn_implementation="eager"` does not fix it (the int-cast is not SDPA-specific).

2. **`torch.export(hf_model)` + `run_decompositions({})`** — converts further into the pipeline, but the exporter lifts module buffers into graph inputs, so coremltools' StateType validation (`hf_model.named_buffers() == []`) fails. There's no `preserve_module_call_signature`-style escape that keeps both the stateful KV cache AND the conversion happy.

3. **From-scratch Qwen2.5** (`qwen25.py`) — loads HF safetensors weights into a hand-written minimal module designed for trace-cleanliness. Weights load correctly (embed output matches HF bit-exactly). But the transformer block forward produces 0 cos_sim vs HF — there's an unfixed bug somewhere in RoPE / attention. Did not finish debugging this session.

## Recommended next steps

Three viable paths forward, ranked by time-to-result:

### A. Use Anemll (third-party LLM-to-ANE pipeline) — fastest path

[Anemll](https://github.com/Anemll/Anemll) is a project specifically built to convert open LLMs to CoreML for ANE deployment. It handles all the conversion edge cases we've been hitting. Try this first:

```sh
git clone https://github.com/Anemll/Anemll vendor/anemll
# Follow their Qwen conversion recipe
```

If it works, we get a working Qwen2.5-3B on ANE today and can move on to whatever's next.

### B. Use `huggingface/exporters` (HF's official CoreML exporter)

```sh
uv pip install git+https://github.com/huggingface/exporters
python -m exporters.coreml ...
```

Less mature than Anemll for LLMs specifically, but officially supported.

### C. Finish the from-scratch path

Continue debugging `qwen25.py`. The bug is in `forward()` — embeddings load correctly but the transformer block produces wrong output. Likely culprits in priority order:

1. **RoPE convention mismatch**: HF Qwen2.5 may use the "interleaved" rotary pattern rather than "split-half". Inspect HF's `apply_rotary_pos_emb` in `transformers/models/qwen2/modeling_qwen2.py` and align.
2. **GQA repeat order**: HF uses `repeat_interleave` along the head dim; verify ours matches.
3. **Numerical precision**: All-fp16 may accumulate enough error to corrupt the result. Try fp32 internal math and see if the cos_sim climbs.

This path gives the most control but each iteration is slow due to the 11s weight-load.

### Apple's "multi-cache-size graphs" optimization (deferred)

User suggestion (worth doing AFTER one of A/B/C above works): compile separate static-shape CoreML graphs at exponentially-spaced cache sizes — 1, 2, 4, 8, ..., 1024, 2048. At inference, pick the smallest graph that fits the current sequence length. ANE strongly prefers static shapes; a single dynamic-shape graph often forces CPU fallback. This is the production pattern (Apple Intelligence uses something similar).

Implementation: trace+convert N times with different `max_seq` constants baked in, save N `.mlpackage`s, write a tiny dispatcher in `bench.py` that routes each forward call to the right one based on position.

## Files

```
coreml_experiment/
├── README.md           — project goals + architecture choice
├── STATUS.md           — this file
├── smoke_test.py       — VALIDATED end-to-end pipeline test on TinyMLP
├── convert.py          — HF → CoreML conversion (BLOCKED, see above)
├── qwen25.py           — From-scratch trace-friendly Qwen2.5 (weights OK, forward broken)
└── verify_qwen25.py    — Compares qwen25.py output to HF reference (currently shows 0 cos_sim)
```

## Dev environment

```sh
source .venv-coreml/bin/activate
export HF_HUB_CACHE=~/coreml_models     # 5.8GB of Qwen2.5-3B-Instruct lives here
```
