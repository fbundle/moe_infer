# coreml_experiment — Current state

## Working: ANE inference end-to-end ✅

We have **two pre-converted LLMs running on the Apple Neural Engine** via Anemll's distribution on Hugging Face. Validated 2026-06-10 on Apple M4 + macOS 26.4.1:

| Model | Class | Decode | Prefill | Wall-clock (200 tok decode) | Quant |
|---|---|---|---|---|---|
| `anemll/anemll-Qwen-Qwen2.5-0.5B-Instruct-ctx2048-monolithic_0.3.5` | 0.5B Qwen2.5 | **74.0 tok/s** | 75.0 tok/s | ~60 tok/s | INT4 LUT (LUT=4) + LM-head LUT=6 |
| `anemll/anemll-Hermes-3.2-3B-iOS-0.1.1` | 3B Llama-3.2 | **15.6 tok/s** | 20.0 tok/s | ~12 tok/s | INT4 LUT (FFN=6) + LM-head LUT=8 |

Outputs are coherent — short stories, factual statements about ANE, etc.

**Power perspective (the actual point):** ANE is roughly half the GPU's decode speed on 3B-class, but uses ~5-10× less power per token. For "always-on" use cases (chat assistant in the background, on battery) this is the right tradeoff. For maximum throughput plugged in, GPU (MLX or our Metal engine) wins.

## Working: end-to-end PyTorch → CoreML pipeline on a tiny model ✅

`smoke_test.py` validates that the toolchain itself works:
- TinyMLP (Linear → ReLU → Linear) → `torch.jit.trace` → `ct.convert` → `linear_quantize_weights` (INT4 sym, group=64) → `.mlpackage` → reload across CPU_ONLY / CPU_AND_GPU / CPU_AND_NE / ALL.
- All compute units load and dispatch. INT4 rel_err 0.0073 vs reference (excellent).
- ANE / GPU / CPU all visible: `MLNeuralEngineComputeDevice`, `MLGPUComputeDevice`, `MLCPUComputeDevice`.

## Blocked: converting Qwen2.5-3B-Instruct ourselves

Anemll's conversion script (`convert_model.sh`) failed at "Step 3: Converting FFN" with:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
  in coremltools.../torch/ops.py:3022 in _int → _cast
```

Tried across:
- coremltools 9.0 + Python 3.13 + torch 2.5/2.7 → same error
- coremltools 8.3 + Python 3.12 + torch 2.5/2.7 → same error
- transformers 4.55 / 5.10 → same error

So it's **not** a Python/coremltools/torch version mismatch; it's a real bug in coremltools' `_int` converter that fires when the FFN graph has a specific op pattern. Anemll's pre-converted models prove their pipeline ran successfully against an earlier coremltools build — they ship binary `.mlmodelc` outputs that bypass the bug at consumer time.

For 3B-on-ANE today, **use the pre-converted Hermes 3B** (or wait for Anemll 0.3.5 to ship a Qwen2.5-3B). For converting our own, the next step is to instrument the failing FFN trace to find which op produces the non-scalar value and either patch coremltools or rewrite the offending PyTorch op.

## Files

```
coreml_experiment/
├── README.md           — project goals + architecture choice
├── STATUS.md           — this file
├── smoke_test.py       — VALIDATED end-to-end pipeline on TinyMLP
├── bench_anemll.py     — VALIDATED — downloads + runs Anemll pre-converted model on ANE
├── convert.py          — Our own conversion (BLOCKED, see above)
├── qwen25.py           — From-scratch trace-friendly Qwen2.5 (loads weights but forward broken)
└── verify_qwen25.py    — Compares qwen25.py to HF reference
```

## Dev environments

We have **two** venvs because Anemll and our own scripts want different versions:

| Venv | Python | coremltools | torch | transformers | Use for |
|---|---|---|---|---|---|
| `.venv-anemll/` | 3.12 | 8.3.0 | 2.5.0 | 4.55.0 | Anemll workflow (chat / bench / convert) |
| `.venv-coreml/` | 3.13 | 9.0 | 2.7.1 | 5.10.2 | Our own `smoke_test.py` and `convert.py` experiments |

```sh
# To run Anemll bench:
source .venv-anemll/bin/activate
python coreml_experiment/bench_anemll.py --model anemll/anemll-Qwen-Qwen2.5-0.5B-Instruct-ctx2048-monolithic_0.3.5

# To run our own pipeline tests:
source .venv-coreml/bin/activate
python coreml_experiment/smoke_test.py
```

## What's next, in priority order

1. **Power measurement.** The whole point of ANE is ~5-10× less power per token than GPU. Validate this concretely:
   - Use `powermetrics` (`sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000`)
   - Run the Anemll Qwen2.5-0.5B on ANE for 1 minute, capture wattage
   - Run the same prompt+model on our Metal engine on GPU, capture wattage
   - Compare W per token.
2. **Multi-cache-size graphs** (user's earlier idea, deferred until conversion works). Compile separate static-shape models at sequence lengths 1, 2, 4, 8, ..., 1024, 2048 and dispatch per current position. Avoids the CPU fallback that dynamic-shape graphs trigger on ANE. Apple Intelligence uses this pattern.
3. **3B-class Qwen2.5 on ANE.** Either:
   - Wait for Anemll 0.3.5 to publish a converted Qwen2.5-3B (they already have 0.5B), or
   - Debug our conversion: identify the failing `_int` op via printing the offending PyTorch operator name, then either rewrite the op in our `qwen25.py` or patch coremltools' op converter to handle non-scalar constants in `_int`.
4. **Larger context windows.** Both pre-converted models are ctx=1024 or 2048. Apple Intelligence ships ~8K. Worth converting (or downloading) a longer-ctx variant once 3B works.

## Reference

- Anemll repo (vendored at `vendor/anemll/`): https://github.com/Anemll/Anemll
- Anemll HF org: https://huggingface.co/anemll
- Apple's coremltools docs (stateful models): https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
