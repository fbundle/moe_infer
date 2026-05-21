# Port Report: Flash-MoE → Rust (Qwen3.5-35B-A3B-4bit)

## Status

Rust port of the Flash-MoE Metal inference engine. **Builds and runs**, generating coherent text at **4.24 tok/s** on Apple M4 (16GB). The C baseline achieves ~5.2 tok/s — the Rust port is at ~82% of C speed, with the gap coming from incomplete command-buffer fusion.

## Architecture

The Rust crate (`moe_infer_rs/`) mirrors the C codebase's layer structure:

| C file | Rust module | Status |
|--------|------------|--------|
| `main.m` / `metal_init()` | `metal_context.rs` | Done |
| `shaders.metal` | `shaders/shaders.metal` | Done (embedded via `include_str!`) |
| `layer_forward.h` / `fused_layer_forward` | `gpu_forward.rs` | Partial (see gaps below) |
| `attention.h` / `linear_attention_forward` | `gpu_forward.rs` | Partial |
| `expert.h` / `run_expert_forward` | `expert.rs` | Done (CPU fallback path only) |
| `server.h` / `tokenizer` | `server.rs` / `tokenizer.rs` | Done |
| `config.h` / `model_config.h` | `config.rs` | Done (JSON-driven, no compile-time defines) |
| `weights.c` / `weight_cache.c` | `weights.rs` | Done (mmap + zero-copy Metal buffer) |

### Key crates used
- `metal` 0.33 — Apple Metal bindings
- `objc` 0.2 — ObjC runtime interop
- `memmap2` 0.9 — mmap for zero-copy weight buffer
- `rayon` — parallel CPU expert dispatch

### Weight buffer architecture
Model weights (1.38 GB) are mmap'd and wrapped in a `MTLBuffer` via `newBufferWithBytesNoCopy`, giving the GPU zero-copy access. Tensor lookups resolve byte offsets within this buffer via a JSON manifest. Expert weights live in separate per-layer files (`packed_experts/layer_XX.bin`).

## Current Performance

```
Benchmark: cargo run --release --bin bench -- --model <model_dir> --tokens 100
Result:    4.24 tok/s (100 tokens in 23606 ms)
Prefill:   9131 ms for 29 tokens
Device:    Apple M4, unified memory
```

The benchmark (`src/bin/bench.rs`) is a pure Rust binary — no HTTP overhead, no Python. It generates 100+ tokens for stable timing per the methodology used for the C benchmark.

## What's Working

- **Model loading**: JSON-driven config, mmap'd weight file, HF tokenizer
- **Metal setup**: Device/queue/library/pipeline creation, shader compilation (2 ms)
- **Weight buffer**: Zero-copy GPU access via `newBufferWithBytesNoCopy`
- **All GPU kernels**: dequant matvec (v3/v4/v5/2bit), SwiGLU, RMS norm, weighted sum, attention, conv1d, gated delta net, residual add, sigmoid gate
- **Linear attention (GatedDeltaNet)**: QKV/Z/B/A projections, conv1d, q/k split, RMS norm, SSM recurrence, gated RMS norm, out_proj
- **Full attention**: Every 4th layer with KV cache, RoPE, Q/K norm, sigmoid gate, o_proj
- **MoE routing**: GPU routing (gate projection → top-K), expert dispatch with wait
- **Shared expert**: Sigmoid gate, up/gate projections, SwiGLU, down projection
- **Deferred expert infrastructure**: `DeferredExperts` struct and `prev_deferred` plumbing threaded through all layers — ready for true async dispatch
- **Pure Rust benchmark**: 100+ token generation without HTTP overhead

## Gaps vs C Code (Performance Opportunities)

### 1. Incomplete CMD1 fusion
The C code's `fused_layer_forward` encodes 5 GPU dispatches into a single `MTLCommandBuffer` for linear attention layers:
```
CMD1: Q_proj + K_proj + V_proj + Z_proj + B_proj + A_proj + conv1d + split + RMSnorm + SSM + gatedRMSnorm
```
The Rust `gpu_forward.rs` has the fused path coded but the **bench.rs benchmark does not use it**. Instead, bench.rs copies CPU helper functions that dispatch each projection as a separate command buffer (encode → commit → wait per projection). Wire bench.rs to call `gpu_forward.rs`'s fused path.

### 2. No CMD2 fusion
In C, CMD2 batches `o_proj + residual_add + rms_norm + routing_gate` into one command buffer. In Rust, these are currently individual dispatches. The `moe_layer_forward` in Rust dispatches routing on GPU synchronously but o_proj/residual/norm are scattered across different code paths.

### 3. No CMD3 async expert dispatch
In C, CMD3 dispatches all K experts in one command buffer and commits **without waiting** — results are collected in the *next* layer's forward pass via `deferred_experts_complete()`. This overlaps GPU expert compute with CPU work for the current layer.

In Rust, experts are dispatched synchronously (`wait_until_completed`), so the GPU sits idle between each expert dispatch. The `DeferredExperts` struct is in place as a placeholder. To implement true async dispatch:
- Store the expert `CommandBuffer` in `DeferredExperts` using `unsafe` ObjC `retain`/`release` (metal-rs `CommandBuffer` is a ZST marker — actual references are `&CommandBufferRef`)
- Commit CMD3 without waiting
- In the next layer's forward, wait on the previous deferred command buffer and read back results

### 4. CPU full attention in bench.rs
The benchmark has inline CPU implementations of full attention (q/k/v projections, RoPE, softmax, o_proj) that use individual Metal dispatches. The C code fuses all full-attention projections into a single command buffer. The `gpu_forward.rs` fused path handles full-attention layers too — it should be used by the benchmark.

### 5. No GPU-side combine (`moe_combine_residual`)
The kernel exists in shaders and the pipeline is created, but the Rust code does CPU-side accumulation of expert outputs. The C code optionally does GPU-side combine in CMD3. Enabling this would eliminate CPU→GPU round-trips for expert output accumulation.

## Fixes Applied (This Session)

- **Metal context impl block**: Missing closing brace in `metal_context.rs` after `init_linear_attn_buffers()`
- **Kernel offsets**: `encode_gated_delta_net_step` now accepts `q_offset`, `k_offset`, `v_offset` parameters — the fused path passes correct offsets into `buf_conv_output` for q/k/v regions
- **DeferredExperts plumbing**: All layer-forward call sites (server.rs, bench.rs) thread `prev_deferred: &mut Option<DeferredExperts>` and call `.complete()` after the last layer
- **Re-exports**: `GpuWeightCtx`, `metal_buf_shared`, `DeferredExperts` re-exported from `lib.rs`
- **bench binary**: `[[bin]]` entry in `Cargo.toml`, forced add through `.gitignore`

## Next Steps (Priority Order)

1. **Wire bench.rs to use `gpu_forward.rs` fused path** — replace the inline CPU attention functions with calls to `linear_attention_forward` / `full_attention_forward` from `gpu_forward.rs`. This alone should close most of the gap to C speed.

2. **Implement CMD3 async expert dispatch** — store `&CommandBufferRef` in `DeferredExperts` (using `unsafe` ObjC retain), commit without wait, complete in next layer. The user explicitly approved `unsafe` for this.

3. **Implement CMD2 fusion** — batch `o_proj + residual_add + rms_norm + routing_gate` into a single encoder.

4. **GPU-side combine** — use `moe_combine_residual` kernel in CMD3 instead of CPU accumulation.

5. **Full prefill optimization** — the C code uses fused kernels for batched prefill. Rust currently does token-by-token CPU prefill which takes 9+ seconds for 29 tokens.

## Running

```bash
# Build
cd moe_infer_rs
cargo build --release

# Benchmark (100 tokens)
cargo run --release --bin bench -- \
  --model /Volumes/Hippopotamus/vault/code/flash-moe/data/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --tokens 100 --verbose

# HTTP server
cargo run --release --bin moe-infer -- \
  --model /Volumes/Hippopotamus/vault/code/flash-moe/data/models--mlx-community--Qwen3.5-35B-A3B-4bit
```

## Key File Index

| File | Purpose |
|------|---------|
| `src/gpu_forward.rs` | Fused layer forward, linear attention, MoE routing, DeferredExperts |
| `src/kernels.rs` | GPU kernel dispatch wrappers (set buffers, set bytes, dispatch threadgroups) |
| `src/metal_context.rs` | Metal init, pipeline creation, GpuWeightCtx, buffer helpers |
| `src/server.rs` | HTTP server, per-layer loop, token generation |
| `src/bin/bench.rs` | Pure Rust benchmark (no HTTP) — **currently has duplicate attention code** |
| `src/weights.rs` | Weight file mmap + JSON manifest + tensor lookup |
| `src/config.rs` | Model config from HuggingFace `config.json` |
| `src/tokenizer.rs` | BPE tokenizer (HF `tokenizer.json`) |
| `src/expert.rs` | CPU 4-bit expert forward |
| `shaders/shaders.metal` | All Metal compute shaders |
