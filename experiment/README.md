# coreml_experiment

Power-efficient on-laptop LLM inference via CoreML on the Apple Neural Engine.

This is a side experiment to the main `moe_infer` Rust+Metal engine. It targets:
- **Dense models** that fit comfortably in memory (Qwen2.5-3B, Llama-3.2-3B class)
- **Power efficiency** over raw throughput — ANE uses ~5–10× less power per token than GPU
- **Pure-Python conversion pipeline** + Python inference; no Rust (coremltools is Python-only)

For the main MoE Metal engine, see `../src/engine/`. MoE is *not* targeted here — CoreML's static-graph design clashes with dynamic expert routing, and there's no analog for our expert-streaming-from-disk feature.

## Layout

- `convert.py` — HF model → INT4 stateful `.mlpackage`
- `bench.py` — load + greedy decode + tok/s and (eventually) wattage
- `models/` — converted `.mlpackage` artifacts (gitignored)

## Approach

1. **Stateful CoreML** for the KV cache (macOS 15+, supported by coremltools 8.0+). KV cache lives inside the model as a state input — caller advances it across forward calls without re-allocating.
2. **INT4 weight quantization** via `ct.optimize.coreml.linear_quantize_weights` with `mode="linear_symmetric"`, `n_bits=4`, per-grouped-channel (group_size=64). Matches the bit budget of our Metal engine's INT4 format.
3. **Multi-cache-size graphs** (planned optimization, not in v0): build separate static-shape graphs at sequence lengths 1, 2, 4, 8, 16, 32, …, 2048 and route each request to the smallest fitting graph. ANE strongly prefers static shapes; a single dynamic-shape graph often forces CPU fallback on attention ops. This is the pattern Apple uses in production.
4. **ANE-first compile target**: `ct.ComputeUnit.CPU_AND_NE` for first runs. Drop to `CPU_AND_GPU` or `ALL` only if ANE fallback proves catastrophic.

## Why a separate venv

coremltools 9.0 ships binary modules built against Python 3.13. Our main `.venv/` is Python 3.14 (mlx-lm requirement). `../.venv-coreml/` is the 3.13 venv used by this experiment.

## Compute device check

```python
from coremltools.models.compute_device import MLComputeDevice
[type(d).__name__ for d in MLComputeDevice.get_all_compute_devices()]
# → ['MLNeuralEngineComputeDevice', 'MLGPUComputeDevice', 'MLCPUComputeDevice']
```
