# MoE-Infer

High-performance inference engine for Mixture-of-Experts models on Apple Silicon. Streams expert weights from SSD on demand — no Python ML frameworks at runtime, just Rust and hand-tuned Metal shaders.

Supports `mlx-community/Qwen3.5-35B-A3B-4bit` and `mlx-community/Qwen3.6-35B-A3B-4bit`.

The `FusedWoods` pipeline mode is named after Dan Woods, author of the original C/Metal inference engine this project builds upon.

## Hardware Requirements

- Mac with Apple Silicon (M1/M2/M3/M4)
- ~20 GB free SSD space for model weights
- macOS 14+ (for Metal 3)

## Quick Start

### 1. Convert the model

```bash
python helpers/convert.py --model hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
```

### 2. Build and install Python bindings and Run inference

```bash
maturin develop --release -m moe_infer_rs/Cargo.toml
python chat.py
```


## Python API

```python
from moe_infer import Model, Engine, Cache, record_engine_telemetry
```

### Model

| Method | Description |
|--------|-------------|
| `model = Model(model_path)` | Load model weights, config, and expert file handles (data only) |

### Engine

| Method | Description |
|--------|-------------|
| `engine = Engine(model, pipeline_mode="FusedExp", k=0)` | Initialize Metal GPU resources. `k` selects active experts per token (0 = model default) |
| `engine.forward(input_ids, cache)` | Forward pass, returns `[n_tokens, vocab_size]` float32 logits |
| `engine.stream_generate(input_ids, cache, ...)` | Autoregressive generation, yields `(token_id, logits)` tuples incrementally |
| `engine.telemetry()` | Returns dict: `prefill_ms`, `total_ms`, `tokens_generated`, `tokens_per_sec`, plus engine-specific metrics |

### Telemetry

| Function | Description |
|----------|-------------|
| `record_engine_telemetry(on: bool)` | Enable/disable engine-level timing (e.g. `engine.expert_io_ms`) |

### Cache

| Method | Description |
|--------|-------------|
| `cache = Cache(model)` | Create KV caches + linear attention state for the given model |
| `cache.reset()` | Reset position, KV caches, and linear attention states |
| `cache.pos` | Current sequence position (read-only) |

### Pipeline Modes

| Mode | Description |
|------|-------------|
| `Cpu` | Pure CPU reference. All operations on CPU. Slow but useful for debugging. |
| `Gpu` | GPU kernels with individual dispatch. No command buffer fusion. |
| `FusedExp` | Linear attention fused into one command buffer. MoE experts dispatched individually. |
| `FusedWoods` | Full 3-command-buffer pipeline (CMD1 + CMD2 + async CMD3). **Recommended.** |

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 256 | Maximum new tokens to generate |
| `temperature` | 0.0 | < 0.01 for greedy, > 0 for sampling |
| `top_k` | 0 | Keep top-k logits (0 = disabled) |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `min_p` | 0.0 | Minimum probability relative to max |
| `eos_token_ids` | [248046, 248044] | Stop tokens |

## Verification

`verify_nway.py` checks numerical correctness across all engines on a stripped 4-layer model:

```bash
python verify_nway.py
```

Runs `Cpu`, `Gpu`, `FusedWoods`, `FusedExp` (Rust), `C` (C bench), and `mlx-lm` on the same token sequence, then compares logits pairwise. Outputs an N×N `max_diff` matrix with per-engine status (IDENTICAL / MATCH / CLOSE / DIVERGE).

Expected results: all non-mlx engines agree within `2.6e-05` (ULP-level). mlx-lm diverges at `~0.11` due to bf16 precision. C bench and Rust FusedWoods are byte-for-byte identical.

## Benchmarking

`bench.py` benchmarks C and Rust GPU pipelines on the full 40-layer model:

```bash
python bench.py
```

Tests forward passes at 100, 200, and 300 tokens across `Gpu`, `FusedWoods`, `FusedExp` (Rust) and the C bench. Prints per-engine latency and throughput, plus speedup vs C.

Requires the full model in `data/models--mlx-community--Qwen3.5-35B-A3B-4bit`.

## Model Format

MoE-Infer expects a model directory with:

```
model_dir/
├── config.json                # HF config (read directly by Rust engine)
├── model_weights.bin          # Mmap'd: all non-expert weights (embeddings, norms, projections, shared experts)
├── model_weights.json         # Tensor manifest (name → offset, size, shape, dtype)
├── packed_experts/            # Per-layer expert files (required)
│   ├── layer_00.bin
│   ├── layer_01.bin
│   └── ...
├── packed_experts_lz4/        # LZ4-compressed experts (optional, faster SSD I/O)
│   ├── layer_00.bin
│   ├── layer_01.bin
│   └── ...
├── tokenizer.json             # HF tokenizer (used by Python bindings)
└── vocab.json
```

The engine auto-detects `packed_experts_lz4/` at load time and falls back to `packed_experts/` if not present.

Helper scripts in `helpers/` convert from HuggingFace/MLX format:
- `convert.py` — One-step conversion: config, weights, and expert repacking
- `extract_weights.py` — Non-expert weights → `model_weights.bin` + `model_weights.json`
- `repack_experts_4bit.py` — MLX 4-bit experts → `packed_experts/` per-layer files
- `compress_experts_lz4.py` — Compress packed experts with LZ4 → `packed_experts_lz4/` (~40-55% smaller)
- `repack_experts_2bit.py` — Requantize experts to 2-bit → `packed_experts_2bit/` (experimental)
- `quantize_from_hf.py` — Convert HuggingFace unquantized models → MoE-Infer format
- `strip_model.py` — Build a small 4-layer model for fast verification

## Performance

Apple M4, Qwen3.5-35B-A3B-4bit (40 layers, 256 experts, K=8), 32-token prompt, 100-token greedy generation:

| Mode | tok/s |
|------|-------|
| FusedWoods | 2.69 |
| FusedExp | 2.14 |
| Gpu | 1.70 |
| Cpu | 0.15 |

Expert I/O (SSD reads) dominates at ~72% of per-layer time.

## Project Structure

```
moe_infer_rs/              Rust engine + Python bindings
  src/
    engine.rs              Engine trait, TelemetryValue, record_engine_telemetry
    engine/
      cpu.rs               CPU engine (self-contained, pure f32)
      fusedexp.rs          FusedExp pipeline (per-phase telemetry, K configurable)
      fusedwoods.rs        FusedWoods pipeline (3-CMD, recommended)
    model/
      mod.rs               Model struct (loads all files at startup)
      config.rs            ModelConfig derived from HF config.json
      weights.rs           Mmap'd weight file + tensor lookup
      expert.rs            ExpertFile enum (Raw pread / Lz4 decompress)
    math_util.rs           Math utilities (rms_norm, softmax, sigmoid, dequant, RoPE, etc.)
    metal_util/
      context.rs           Metal device init, pipeline creation, ExpertCache LRU, scratch bufs
      kernels.rs           Metal kernel dispatch (matvec, SwiGLU, conv1d, SSM, attention)
      shaders.metal         Metal compute shaders (embedded at compile time via include_str!)
    cache.rs               KV cache + linear attention state
    constants.rs           Architecture constants
    timer.rs               Wall-clock timer
    python_bindings.rs     PyO3 bindings (Model, Engine, Cache, stream_generate)
    lib.rs                 Module declarations + Python module init
    error.rs               Error types
  Cargo.toml

helpers/                   Model conversion scripts
  convert.py               One-step convert: config + weights + experts
  extract_weights.py       Non-expert weights → model_weights.bin + model_weights.json
  repack_experts_4bit.py   MLX 4-bit experts → packed_experts/
  compress_experts_lz4.py  packed_experts/ → packed_experts_lz4/ (~40-55% compression)
  repack_experts_2bit.py   packed_experts/ → packed_experts_2bit/ (experimental)
  strip_model.py           Build 4-layer stripped model for fast verification
  quantize_from_hf.py      Convert HuggingFace unquantized models → MoE-Infer format

bench.py                   Multi-engine performance benchmark
verify_nway.py             N-way logit comparison (Cpu, FusedExp, FusedWoods, C, mlx-lm)
chat.py                    Interactive chat demo
```

## License

MIT
