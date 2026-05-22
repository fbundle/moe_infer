# MoE-Infer: Technical Report

## Overview

MoE-Infer is a high-performance inference engine for Mixture-of-Experts models on Apple Silicon. It streams expert weights from SSD on demand (with optional LZ4 compression), uses hand-tuned Metal compute shaders for all GPU operations, and exposes Python bindings via PyO3. No Python ML frameworks at runtime — just Rust, Metal, and ~0.65 GB of mmap'd weights.

Built on the 3-command-buffer GPU pipeline architecture designed by Dan Woods in the original C/Metal engine. The `FusedWoods` pipeline mode is named in his honor.

**Supported models**: `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.6-35B-A3B-4bit`

**Hardware**: Apple Silicon (M1–M4) with unified memory. Tested on M4.

## Architecture

### Model Structure

Qwen3.5-35B-A3B-4bit: 40 layers, 256 experts, K=8 active experts per token.

| Parameter | Value |
|-----------|-------|
| Hidden dim | 2048 |
| Vocab size | 248,320 |
| Layers | 40 (30 linear attention + 10 full attention, every 4th layer) |
| Experts | 256 (8 active per token) |
| Expert intermediate | 512 |
| Shared expert intermediate | 512 |
| Linear attention | 16 K-heads (dim 128), 32 V-heads (dim 128), conv kernel 4 |
| Full attention | 16 Q-heads (dim 256), 2 KV-heads (dim 256), RoPE dim 64 |
| Quantization | 4-bit affine (group_size=64), nibble * scale + bias |
| Weight format | U32 packed weights + BF16 scales/biases |

### Data Flow (per token, per layer)

```
Input → RMS Norm → Attention (linear or full) → Residual Add
  → Post-Attention Norm → MoE Gate → Top-K Routing
  → Expert I/O (SSD pread) → Expert Matvecs (Gate/Up, SwiGLU, Down)
  → Shared Expert (SwiGLU + Down) → MoE Combine + Residual → Output
```

### Expert I/O

Expert weights (~19 GB 4-bit) live on SSD in per-layer files (`packed_experts/layer_NN.bin`). Only K=8 active experts are read per layer (~1.77 MB each) via parallel `pread()` across 4 threads, with an LRU cache (32 entries) to avoid re-reading repeated experts.

**LZ4 compression** (optional): `helpers/compress_experts_lz4.py` compresses the per-layer expert files with LZ4, reducing total expert size by ~40-55%. The engine auto-detects `packed_experts_lz4/` at load time and transparently decompresses on read via `lz4_flex`. This is a drop-in replacement for the raw packed files and reduces SSD bandwidth by roughly 30-50%. Both `ExpertFile::Raw` and `ExpertFile::Lz4` variants share the same `read_expert()` interface.

#### Why `pread()` and not `mmap()`

Non-expert weights (0.65 GB) use `mmap()` — they fit in memory and are accessed every layer, every token. Experts (19 GB, 30× larger) use `pread()` into pre-allocated Metal buffers. The reasoning:

**1. DMA alignment (3.6× speedup).** Expert data buffers are allocated with `posix_memalign(..., 2MB)` and wrapped via `newBufferWithBytesNoCopy`. The DMA controller that handles `pread()` from Apple's SSD achieves 3.6× higher throughput with 2 MB alignment vs the 16 KB page alignment that `mmap()` guarantees. This is the single biggest factor.

**2. One syscall, not 110 page faults.** A 1.77 MB expert spans ~110 pages (16 KB each on Apple Silicon). With `mmap()`, the first access to each page triggers a synchronous kernel trap: page fault → I/O dispatch → TLB fill. That's 110 individual round-trips through the kernel. With `pread()`, the kernel reads the entire blob in a single efficient I/O operation — one syscall, one I/O submission, one completion.

**3. Double-buffering for prediction preads.** The C engine uses an A/B buffer pair (`buf_multi_expert_data` / `buf_multi_expert_data_B`). While the GPU processes expert A's results, prediction preads fill the B buffer for the next layer. `mmap()` can't provide independent buffer copies — it's a single mapping. The double-buffer scheme is essential for overlapping I/O with compute (see Expert Prediction in §C vs Rust FusedWoods).

**4. Explicit eviction control.** The LRU cache (32 entries) decides which experts stay resident based on application-level routing patterns. With `mmap()` + memory pressure, the kernel's page reclaimer makes that decision instead — and it has no knowledge of MoE routing. Under the wrong access pattern, the kernel evicts the wrong pages and thrashing results. With `pread()`, eviction is deterministic and application-controlled.

**5. Scale mismatch.** Non-expert weights (0.65 GB) are small enough to mmap once at startup and keep resident forever — the `newBufferWithBytesNoCopy` Metal buffer wrapping the mmap'd region is valid for the lifetime of the process. Experts (19 GB) can't be kept resident alongside KV caches, activations, and scratch buffers. `pread()` is the correct primitive for "read this blob, use it on GPU, discard it."

### Metal Compute Pipeline

All matrix-vector multiplies run on GPU via Metal compute shaders:
- **4-bit dequant matvec**: `nibble * scale + bias` fused with dot product via FMA
- **Fused linear attention (CMD1)**: QKV/Z/B/A projections + Conv1d + Q/K RMS norms + SSM state update + Gated RMS norm — single command buffer
- **Fused full attention (CMD2)**: QKV projections + Q/K norms + RoPE (CPU) + Batched attention (scores, softmax, values) + Sigmoid gate + o_proj + Residual add + Post-attn norm + MoE gate + Shared expert projections — single command buffer
- **Expert dispatch (CMD3)**: K expert gate/up + SwiGLU + down + Shared expert SwiGLU + down + MoE combine + residual — async commit, completed next layer

## Weight File Format

MoE-Infer uses a custom binary weight format optimized for mmap and pread, converted from the HuggingFace/MLX safetensors format. The conversion is done by three helper scripts in `helpers/`.

### HF/MLX Format (Input)

The source model is stored in the standard MLX-quantized safetensors layout:

- **Multiple `.safetensors` files** with a `model.safetensors.index.json` index mapping tensor names to shard files.
- **4-bit affine quantization**: weights stored as nibble-packed U32 arrays `[out_dim, in_dim/8]`, scales and biases as BF16 arrays `[out_dim, in_dim/64]`. Group size is 64.
- **Expert tensors** use 3D shapes: `[num_experts, out_dim, packed_in_dim]` for each of gate_proj, up_proj, down_proj (weight + scales + biases = 9 tensors per layer).
- **Tensor naming**: `language_model.model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}` for experts; `language_model.model.layers.N.self_attn.*` for full attention; `language_model.model.layers.N.linear_attn.*` for linear attention.
- **Gate tensors** (router and shared expert gate) may use 8-bit quantization (`{weight,scales,biases}` with INT8 dtype) on newer models (Qwen3.6+). These are dequantized and re-quantized to 4-bit during extraction.

### MoE-Infer Non-Expert Weights

Single mmap'd file `model_weights.bin` + JSON manifest `model_weights.json`. Produced by `helpers/extract_weights.py`.

**`model_weights.bin`**: All non-expert tensors packed contiguously with 64-byte alignment. Each tensor stored in its native format (U32 packed for 4-bit, BF16 for scales/biases, F32 for norms). The file is mmap'd at startup for zero-copy GPU access via `newBufferWithBytesNoCopy`.

**`model_weights.json`**: Manifest mapping sanitized tensor names to `{offset, size, shape, dtype}`. Also includes a `config` block with all model dimensions (hidden_size, num_layers, head counts, MoE params, etc.) and per-layer types. The C/Rust engines use this to resolve tensors by name at runtime.

Key differences from HF format:
- **Single file** vs multi-shard: all non-expert tensors in one contiguous binary.
- **Name sanitization**: `language_model.model.layers.N.X` → `model.layers.N.X`; `language_model.lm_head` → `lm_head`. The C engine's `get_tensor_ptr()` looks up tensors by sanitized name.
- **8-bit → 4-bit conversion**: INT8 gate tensors are dequantized to f32 then re-quantized to 4-bit affine (group_size=64) to match the engine's uniform 4-bit code path.
- **Excluded**: vision tower weights, expert tensors (stored separately), and MTP (Multi-Token Prediction) expert layers.

### MoE-Infer Expert Weights

Per-layer flat binary files `packed_experts/layer_NN.bin`. Produced by `helpers/repack_experts_4bit.py`.

Each layer file is a concatenation of expert weight blobs:

```
[expert_0][expert_1]...[expert_{num_experts-1}]
```

Where each expert blob is:

```
gate_proj.weight (U32)  gate_proj.scales (BF16)  gate_proj.biases (BF16)
up_proj.weight   (U32)  up_proj.scales   (BF16)  up_proj.biases   (BF16)
down_proj.weight (U32)  down_proj.scales (BF16)  down_proj.biases (BF16)
```

Sizes for a typical expert (hidden_dim=2048, moe_intermediate=512, group_size=64):

| Component | Dims | Bytes |
|-----------|------|-------|
| gate_proj.weight | 512 × 256 U32 | 524,288 |
| gate_proj.scales | 512 × 32 BF16 | 32,768 |
| gate_proj.biases | 512 × 32 BF16 | 32,768 |
| up_proj.weight | 512 × 256 U32 | 524,288 |
| up_proj.scales | 512 × 32 BF16 | 32,768 |
| up_proj.biases | 512 × 32 BF16 | 32,768 |
| down_proj.weight | 2048 × 64 U32 | 524,288 |
| down_proj.scales | 2048 × 8 BF16 | 32,768 |
| down_proj.biases | 2048 × 8 BF16 | 32,768 |
| **Total per expert** | | **~1.77 MB** |

Key differences from HF format:
- **Per-layer files** vs 3D tensors in multi-shard safetensors: the 3D `[num_experts, out_dim, packed_in_dim]` arrays are sliced by expert and repacked into flat per-expert layouts.
- **Flat binary** vs safetensors container: no JSON header, no tensor metadata — just raw concatenated blobs. Offsets are known from `model_config.json` (see below).
- **pread-friendly**: fixed-size records at known offsets enable direct `pread(expert_id * expert_size)` from SSD without parsing.

### Model Config

The engine reads HF `config.json` directly at runtime via `model::config::load_model_config()`. All dimensions and expert layout offsets are derived from HF fields — no intermediate config format needed. The C engine has these as compile-time `#define`s.

Derived fields computed at load time:
- `expert_size_4bit` / `expert_size_2bit`: total bytes per expert in the packed layer files.
- `expert_layout_4bit` / `expert_layout_2bit`: byte offsets within each expert blob for each projection's weight/scales/biases.
- `rotary_dim`, `linear_total_key`, `linear_total_value`, `linear_conv_dim`: derived from head counts and dimensions.
- `num_full_attn_layers`, `num_linear_layers`: computed from `num_layers` and `full_attention_interval`.

### Conversion Pipeline

```
HF config.json ──► copied directly ──► config.json

HF safetensors/ ──► helpers/extract_weights.py ──► model_weights.bin
                                                   model_weights.json

HF safetensors/ ──► helpers/repack_experts_4bit.py ──► packed_experts/layer_00.bin
                                                       packed_experts/layer_01.bin
                                                       ...

              ┌──► helpers/compress_experts_lz4.py ──► packed_experts_lz4/
              │                                       (optional, ~40-55% compression)
              │
              └──► helpers/repack_experts_2bit.py   ──► packed_experts_2bit/
                                                      (experimental, 2-bit quant)
```

All scripts read from the same MLX-format model directory and output to a single MoE-Infer model directory. The conversion is a one-time offline step; at inference time only the binary files are needed.

`helpers/convert.py` automates the entire pipeline with a single command. `helpers/quantize_from_hf.py` converts directly from HuggingFace unquantized models. `helpers/strip_model.py` builds a small 4-layer model for fast verification iteration.

## Pipeline Modes

| Mode | Description | Sync Points/Layer |
|------|-------------|-------------------|
| `Cpu` | Pure CPU reference. All dequant matvecs, norms, attention, SSM on CPU. | N/A (sequential) |
| `Gpu` | Individual GPU kernel dispatch per operation. No command buffer fusion. | 10+ |
| `FusedExp` | Linear attention fused into CMD1. MoE experts dispatched individually. | 4–6 |
| `FusedWoods` | Full C-engine architecture: CMD1 (linear attn) + CMD2 (full attn + o_proj + routing) + async CMD3 (experts + combine). | 2–3 |

`FusedWoods` (named after Dan Woods, author of the original C inference engine) is the recommended mode and matches the C engine's 3-command-buffer design.

### FusedWoods Command Buffer Layout

**Linear attention layers (30/40)**:
- CMD1: QKV/Z/B/A projections → Conv1d → Q/K RMS norms → SSM → Gated RMS norm → out_proj → Residual add
- CMD2: Post-attn norm → Gate + Shared expert projections + Shared expert gate
- CMD3 (async): Expert gate/up + SwiGLU + down × K → Shared SwiGLU + down → MoE combine + residual → Input norm for next layer

**Full attention layers (10/40)**:
- CMD1: Q/K/V projections → Q/K norms → RoPE (CPU) → KV cache append
- CMD2: Batched attention (scores, softmax, values) → Sigmoid gate → o_proj → Residual add → Post-attn norm → Gate + Shared expert projections
- CMD3 (async): Same as linear, plus explicit input norm for next layer

## Performance

Benchmarked on Apple M4, Qwen3.5-35B-A3B-4bit full model, K=8 experts, 32-token prompt, 100-token greedy generation.

| Metric | Value |
|--------|-------|
| Generation speed (FusedWoods) | 2.69 tok/s |
| Expert I/O (disk read) | ~5.8 ms/layer |
| Expert I/O share of per-layer time | ~72% |
| Weight file (mmap) | 0.65 GB |
| Expert files (SSD) | ~19 GB |

Expert I/O dominates latency. The LRU cache and parallel pread bring Rust within 4% of the C baseline.

## Numerical Verification

### Cpu vs MLX-LM (Stripped 4-Layer Model)

All verification uses the stripped model (4 layers, 4 experts) to enable fast iteration.

**Algorithmic bugs found and fixed**:
1. **RoPE element pairing** (2026-05-22): `apply_rope()` used traditional consecutive pairs (d, d+1) instead of NeoX-style pairs (i, i + dims/2) used by MLX's `nn.RoPE(traditional=False)`. Fix reduced logit max_diff from 0.835 to 0.113 (7.4× improvement).

2. **CpuOnly full-attention MoE bug** (2026-05-21): In CpuOnly mode, the GPU attention path returned early without adding attention output to hidden, causing MoE to use pre-attention hidden as residual. Attention contribution was lost (max_diff 4.88 vs FusedWoods).

**Per-operation verification** (Layer 0, token 0): Every intermediate tensor in GatedDeltaNet compared between Rust f32 and MLX bf16. All operations match within bf16 precision limits (~0.4% relative). No remaining algorithmic bugs.

**Current state**: After all fixes, max logit diff = 0.113, cos_sim = 0.99996. The residual divergence is entirely attributable to bf16 vs f32 precision differences across ~40 operations per token. Per-layer hidden state error is bounded at ~2e-3 and does not grow across layers. The lm_head projection (2048 → 248320) amplifies this to the observed 0.113 logit max_diff.

### C vs Rust FusedWoods (Stripped Model)

C and Rust FusedWoods are numerically identical (max_diff < 1e-5, 100% within 1e-3). The C→Rust port is faithful at the algorithmic level. However, C retains several performance optimizations that Rust does not yet implement.

#### Architecture (Identical)

Both engines implement the same 3-command-buffer pipeline per layer:

- **CMD1**: Attention projections (QKV/Z/B/A or Q/K/V) + conv1d (linear) + Q/K RMS norms + SSM/decay-beta + gated RMS norm. For linear attention with GPU path, all 6 encoders run in one command buffer; the gated output lands in `batch_out[6]` for CMD2.
- **CMD2**: out_proj/o_proj + residual_add + rms_norm + gate + shared_expert projections — fused into a single command buffer. For full-attention layers at seq_len ≥ 32 (C only), attention dispatches (scores, softmax, values, sigmoid gate) are prepended.
- **CMD3** (async, deferred commit): K expert gate/up + SwiGLU + down + shared SwiGLU + down + moe_combine_residual + GPU-side input_norm for next layer. The `DeferredExperts` struct (Rust) mirrors C's `g_deferred` state; the next layer's CMD1 submits immediately after CMD3 commit, with the GPU queue serializing CMD3(N-1) then CMD1(N).

Both also share the GPU-side input_norm optimization: CMD3 computes RMS norm of the combined output into `buf_input`, so the next layer skips the CPU upload of normed hidden state. Both also have equivalent LRU expert caches (C has two: `expert_cache` MTLBuffer-based and `malloc_cache`; Rust has one via `ExpertCache` in `metal_context.rs`).

#### Performance Differences (C advantages)

1. **Expert prediction** (C only). During CMD1 wait, C predicts the next layer's experts (based on the previous token's routing) and starts async `pread()` into `buf_multi_expert_data_B`. By the time CMD3 runs, if prediction was correct (experts often repeat between consecutive tokens), the I/O is already done. Rust has no prediction infrastructure — every layer always pays the full pread latency in CMD3.

2. **LZ4-compressed experts**. Both C and Rust now support LZ4-compressed experts. Rust auto-detects `packed_experts_lz4/` at model load time and transparently decompresses via `lz4_flex`. Both engines achieve roughly the same SSD bandwidth reduction.

3. **2-bit quantization** (C only). The `dequant_matvec_2bit` Metal shader exists in Rust's embedded shaders and `model/config.rs` computes 2-bit layout offsets, but no engine code path dispatches it. `helpers/repack_experts_2bit.py` generates `packed_experts_2bit/` files. Expert size drops from ~1.77 MB to ~1.44 MB at the cost of ~4% relative logit error (pilot experiments). Wiring the engine to use 2-bit experts is the remaining work.

4. **Persistent I/O thread pool**. C uses a pre-spawned `io_pool` of `NUM_IO_THREADS` pthreads coordinated via cond_wait/cond_broadcast. Tasks are dispatched synchronously (`io_pool_dispatch` blocks until all threads report completion). For async I/O, C uses GCD `dispatch_group_async`. Rust uses `rayon::scope()` which spawns threads per invocation. On macOS, GCD may have lower wake-up latency due to kernel integration.

5. **Full attention GPU gating**. C uses batched GPU attention only when `seq_len >= 32` — below 32 tokens, CPU attention is faster because GPU encoder overhead dominates. Rust unconditionally uses GPU attention if pipelines are available, which may be slower for very short sequences.

6. **CPU BLAS acceleration**. C's CPU fallback uses Apple Accelerate (`cblas_sgemv`, `cblas_sger`, `cblas_sscal`) for the linear attention SSM state update. Rust's CPU path uses hand-written loops. Accelerate BLAS is highly optimized for Apple Silicon and can be 2-5× faster for the SSM matvec operations.

7. **Per-phase timing telemetry**. C has fine-grained per-phase timing (`cmd1_submit`, `cmd1_wait`, `cpu_attn`, `cmd2_encode`, `cmd2_wait`, `expert_io`, `cmd3_encode`, `deferred_wait`, `deferred_cpu`, `spec_route`, `pred_read`, `routing_io`). Rust's `FusedExp` engine captures `engine.expert_io_ms`, `engine.full_attention_layer`, `engine.linear_group`, and `engine.total_ms`. FusedWoods currently has no engine-level telemetry. Timing is controlled via `moe_infer.record_engine_telemetry(True)`.

8. **Expert dispatch encoding pattern**. C creates 2 separate compute encoders per expert (gate+up in one, SwiGLU+down in another), which may allow the Metal driver to schedule gate/up of expert 1 concurrently with SwiGLU/down of expert 0. Rust puts all K experts in a single encoder (4 dispatches each), serializing all expert work within CMD3.

9. **Shared expert gate/up on GPU**. Both C and Rust avoid a CPU→GPU re-upload for shared expert results. C keeps `shared_gate` and `shared_up` results on GPU (`buf_shared_gate`, `buf_shared_up`) — SwiGLU reads them directly in CMD3. Rust passes them through CMD3 via `keep_alive` references to prevent buffer recycling. Both are functionally equivalent.

10. **`h_mid` handling in CMD3 combine**. C reads `h_mid` from `buf_h_mid` (populated by CMD2's `residual_add` kernel) — no CPU round-trip. Rust reads `h_mid` from CMD2's `temp_buf` via `hmid_gpu_override`, also avoiding the CPU round-trip. Both are equivalent.

## Key Design Decisions

1. **SSD expert streaming over GPU preloading**: Expert weights are too large (~19 GB) for unified memory alongside KV caches and activations. On-demand SSD reads with LRU caching are the pragmatic choice.

2. **CPU KV cache**: KV caches stored as CPU f32 buffers rather than GPU persistent buffers. Adds ~0.03 ms/layer upload overhead but simplifies memory management. The C engine does the same for full attention layers.

3. **CPU RoPE**: RoPE rotations computed on CPU (both Rust and C). The rotary dimension is only 64 elements per head — GPU kernel launch overhead exceeds CPU compute time.

4. **Single mmap for non-expert weights**: All 0.65 GB of non-expert weights (embeddings, norms, attention projections, shared experts, gates) in one mmap'd file. Zero-copy GPU access via `newBufferWithBytesNoCopy`.

5. **Per-layer expert files**: Each layer's 256 experts in a separate file (`packed_experts/layer_NN.bin`). Enables `pread()` with offset — no seeking needed.

## Known Limitations

1. **No batched inference**: Single-token-at-a-time generation. The prefill processes tokens sequentially rather than in parallel.

2. **No continuous batching**: One sequence per `Engine`. Multiple concurrent users require multiple `Engine` instances.

3. **No expert prediction**: C engine predicts experts for the next token to overlap pread with attention compute. Rust does not implement this.

4. **No 2-bit expert engine path**: The `dequant_matvec_2bit` Metal shader and `repack_experts_2bit.py` helper exist, but the engine code paths in `fusedwoods.rs` and `fusedexp.rs` do not dispatch it yet.

5. **No KV cache quantization**: KV cache stored as f32. Quantizing to bf16 or int8 would reduce memory and upload bandwidth.

6. **FusedWoods lacks per-phase telemetry**: FusedExp captures `engine.expert_io_ms`, `engine.full_attention_layer`, `engine.linear_group`, and `engine.total_ms` when telemetry is enabled. FusedWoods has no engine-level telemetry yet.

## Project Structure

```
moe_infer_rs/              Rust engine + Python bindings
  src/
    engine.rs              Engine trait, TelemetryValue, set_record_engine_telemetry
    engine/
      cpu.rs               CPU engine (self-contained, pure f32)
      fusedexp.rs          FusedExp pipeline (per-phase telemetry, K configurable)
      fusedwoods.rs        FusedWoods pipeline (3-CMD, recommended, no telemetry yet)
    model/
      mod.rs               Model struct (loads all files at startup)
      config.rs            ModelConfig derived from HF config.json
      weights.rs           Mmap'd weight file + tensor lookup (model_weights.bin/json)
      expert.rs            ExpertFile enum (Raw pread / Lz4 decompress)
    cache.rs               KV cache + linear attention state
    constants.rs           Architecture constants (MAX_SEQ, GROUP_SIZE, etc.)
    math_util.rs           Math utilities (rms_norm, softmax, sigmoid, dequant, RoPE, etc.)
    metal_util/
      context.rs           Metal device init, pipeline creation, ExpertCache LRU, scratch bufs
      kernels.rs           Metal kernel dispatch (matvec, SwiGLU, conv1d, SSM, attention)
      shaders.metal         Metal compute shaders (embedded at compile time via include_str!)
    timer.rs               Wall-clock timer (SystemTime, ms precision)
    python_bindings.rs     PyO3 bindings (Model, Engine, Cache, stream_generate, telemetry)
    lib.rs                 Module declarations + Python module init
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
verify_nway.py             N-way logit comparison (Cpu, Gpu, FusedExp, FusedWoods, C, mlx-lm)
chat.py                    Interactive chat demo
```
