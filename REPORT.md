# Rust Port Report: Flash-MoE → Qwen3.5-35B-A3B-4bit

## Status

The Rust port (`moe_infer_rs/`) builds, runs, and generates coherent text on Apple M4 via PyO3 Python bindings. The original vendor C code lives in `moe_infer_c/` (patched for the 35B model) and serves as the performance baseline.

**Fused3 is now complete.** All 40 layers run with 2 sync points (matching the C engine's 3-CMD architecture). `Fused3` is available as a pipeline mode.

## Architecture Comparison

| Aspect | C (`moe_infer_c/infer.m`) | Rust (`moe_infer_rs/`) |
|--------|--------------------------|------------------------|
| Model config | `#define` compile-time constants | JSON-driven at runtime |
| Weight loading | mmap + zero-copy Metal buffer | Same (`newBufferWithBytesNoCopy`) |
| Shader compilation | Runtime (`newLibraryWithSource`) | Same (embedded via `include_str!`) |
| Expert I/O | `pread` from per-layer files | Same (`libc::pread`) |
| Linear attention | Fused CMD1: qkv/z/b/a + conv1d + SSM | Fused CMD1 + out_proj + residual (same) |
| Full attention | GPU batched CMD2: attn + o_proj + residual + norm + gate | Same |
| MoE routing | CMD2: o_proj + residual + norm + gate | Same (full-attn in CMD2, linear in CMD1) |
| Expert dispatch | CMD3: async commit + GPU combine | Same |
| KV cache | GPU bf16 buffers | CPU f32 buffers |
| Python bindings | None (C-only) | PyO3 + Maturin (Context, Cache classes) |

## Pipeline Mode: Fused3 — COMPLETE

Fused3 matches the C engine's 3-command-buffer architecture. All steps complete:

| Feature | Status |
|---------|--------|
| Fused CMD1 (linear attention: qkv/z/b/a + conv1d + SSM + gated_norm + out_proj + residual) | Done |
| GPU batched full attention (scores + softmax + values + sigmoid) | Done |
| GPU moe_combine_residual (expert weighted sum + shared expert + residual in one kernel) | Done |
| Async CMD3 (commit without wait, complete on next layer) | Done |
| CMD2 fusion for full-attn (batched attn + o_proj + residual + norm + gate + shared) | Done |
| CMD1 out_proj fusion for linear (out_proj + residual in CMD1, not separate CMD) | Done |
| PyO3 bindings (Context, Cache, telemetry, stream_generate) | Done |
| Ctrl-C interrupt handling (`py.check_signals()`) | Done |

## Sync points per layer

- **Linear attention** (30/40): 2 sync points — CMD1 (fused linear + out_proj + residual) + router CMD + async CMD3
- **Full attention** (10/40): 2 sync points — QKV CMD + CMD2 (batched attn + o_proj + residual + norm + gate) + async CMD3

**Matches C engine: 2 sync points for all 40 layers.**

## Remaining gaps (non-critical)

| Gap | Description |
|-----|-------------|
| GPU KV cache | KV cache stored as CPU f32 buffers, uploaded per layer. C engine stores on GPU persistently |
| GPU RoPE | Q/K norms and RoPE are CPU-side (C engine also does this on CPU) |
| Linear router CMD fusion | Gate + shared projections are a separate CMD in linear layers. Could fuse into CMD1 or CMD2 — would not reduce sync points further |

## Output Coherence

The Rust engine produces coherent, sensible output verified against the same prompt. The CMD2 combine bug (attention contribution lost in full-attn CMD2 path) has been fixed — `temp_buf` (h_mid + attn_out) is now passed through as `hmid_gpu` to the moe_combine_residual kernel.

## Python API

```python
from moe_infer import Context, Cache

ctx = Context()
ctx.load_model("/path/to/model", pipeline_mode="Fused3")
cache = ctx.new_cache()

# Forward / generate / stream
logits = ctx.forward(input_ids, cache)
new_ids = ctx.generate(input_ids, cache, max_tokens=256, temperature=0.7)
results = ctx.stream_generate(input_ids, cache, max_tokens=256)

# Telemetry
info = ctx.telemetry()
# {"ttft_ms": ..., "tokens_per_sec": ..., "tokens_generated": ...}
```

## Next Steps

1. **GPU KV cache** — store K/V caches persistently on GPU instead of uploading per layer
2. **GPU RoPE kernel** — port C `apply_rope` shader (stretch goal; C engine also does RoPE on CPU)
3. **GPU SSM state** — linear attention SSM state currently uploaded/downloaded per layer in non-fused path
