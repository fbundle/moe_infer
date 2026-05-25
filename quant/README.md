# BQ4: Block Quantization for MoE

BQ4 classifies every weight tensor by its **block** — the dot-separated path
before the last segment — and assigns a quantization format per block.
Sensitive blocks stay BF16; large, redundant blocks use affine INT4;
some blocks use symmetric per-channel INT8.

## Naming convention

Tensors follow the MLX convention (`language_model.` prefix).  Split on the
**last dot**: everything before is the block, the last segment is the kind.
Blocks may contain dots; kinds never do.

```
language_model.model.layers.{L}. self_attn.q_proj.  weight
└─────────── prefix ───────────┘└──── block ─────┘└─ kind ─┘

language_model.model.layers.{L}.  mlp.switch_mlp.gate_proj.  weight
└─────────── prefix ───────────┘└────────── block ────────┘ └ kind ┘

language_model. lm_head. weight
└── prefix ───┘└ block ┘└ kind ┘
```

Kinds are one of: `weight`, `scales`, `biases`, `bias`, `A_log`, `dt_bias`.

The prefix (`language_model.model.layers.{L}.`, `vision_tower.blocks.{B}.`,
`mtp.layers.{L}.`, etc.) is stripped to get the **relative block** used for
classification.

## Quantization rules

```haskell
data Quant = BF16 | BF16Pass | INT4 | INT8 | FP32

matrixTable :: String -> Quant
matrixTable "self_attn.q_proj" = BF16Pass
matrixTable "self_attn.k_proj" = BF16Pass
matrixTable "self_attn.v_proj" = BF16Pass
matrixTable "self_attn.o_proj" = BF16Pass
matrixTable "mlp.gate"         = BF16Pass
matrixTable "lm_head"          = INT8          -- per-channel symmetric
matrixTable "attn.qkv"         = BF16Pass
matrixTable "attn.proj"        = BF16Pass
matrixTable "patch_embed.proj" = BF16Pass
matrixTable "pos_embed"        = BF16Pass
matrixTable _                  = INT4

bq4 :: String -> Quant
bq4 name
  | kind == "A_log"   = FP32
  | kind == "weight"  = matrixTable block
  | kind == "scales"  = BF16
  | kind == "biases"  = BF16
  | kind == "bias"    = BF16
  | kind == "dt_bias" = BF16
  where
    (prefix, kind) = splitOnLastDot name
    block          = stripLayerPrefix prefix
```

1. **Scalars** (`A_log`) → FP32
2. **Vectors** (`scales`, `biases`, `bias`, `dt_bias`, and `weight` with ndim ≠ 2) → BF16
3. **Matrices** (`weight` with ndim = 2) → look up the block in the matrix table.
   If found, use the table format; otherwise INT4.

### Rationale

**BF16Pass matrices** — attention projections (`q_proj`, `k_proj`, `v_proj`,
`o_proj`, `qkv`, `proj`), router (`mlp.gate`), projection embeddings
(`patch_embed.proj`), and positional embeddings (`pos_embed`).  Attention Q·Kᵀ
amplifies quantization noise quadratically; router error misroutes tokens across
expert passes.  BF16Pass skips sanitization (no norm shift or moveaxis) since
these are pure 2D weight matrices.

**INT4 matrices** — experts (`mlp.switch_mlp.*`, `mlp.shared_expert.*`),
linear attention projections (`linear_attn.in_proj_*`, `out_proj`), embeddings
(`embed_tokens`), vision FFN (`mlp.linear_fc*`), and MTP projection (`fc`).
These are the bulk of the model (256 experts × 3 matrices × 40 layers).
Affine INT4 with per-group (64) scale + bias.

**INT8 matrix** — `lm_head` only.  Per-channel symmetric quantization: one
float32 scale per output channel, signed int8 weights centered on zero.
Motivation: the lm_head is the single largest matrix (~947 MB BF16 → ~484 MB
INT8 + 0.97 MB scales), applied once at the final layer so quantization error
does not compound.

**BF16** — everything else: all vectors (norms, conv1d, dt_bias) and all
quantization metadata (`scales`, `biases`, `bias`).

## Affine INT4

Standard affine per-group quantization: each group of 64 contiguous weights
is quantized independently.

```
scale = (max - min) / 15
bias  = min
w_q   = round((w_f32 - bias) / scale)  clamped to [0, 15]
```

Dequant: `w_f32 = nibble × scale + bias`

**Storage per group (64 weights):**
- 32 bytes packed nibbles (4-bit, 8 per uint32, LSB-first)
- 2 bytes BF16 scale
- 2 bytes BF16 bias
- Total: 36 bytes per group = 4.5 bits per weight

## INT8 (lm_head)

Per-channel symmetric quantization: signed int8 weights, one float32 scale per
output channel (vocab entry).

```
scale[i] = max(|w[i,:]|) / 127
w_q      = round(w_f32 / scale[i])   clamped to [-127, 127]
```

Dequant: `w_f32 = int8(w_q) × scale[i]`

No zero-point — symmetric around zero.  This keeps the GPU kernel simple
(one multiply-add per element, no bias term) and matches the distribution
characteristics of well-trained output projection weights, which tend to
be roughly zero-centered.

**Storage for lm_head [248320, 2048]:**
- 484 MB packed int8 weights
- 0.97 MB float32 scales (one per output channel)
- Total: ~485 MB vs 947 MB BF16 (49% reduction)

## Kernel dispatch

Dispatch lives in `WeightBuffer::encode_matvec_into()`.  Each tensor's dtype
(from the weight manifest JSON) determines the Metal pipeline:

| dtype    | Kernel                   | Dequant                             |
|----------|--------------------------|-------------------------------------|
| `"u32"`  | `dequant_matvec_4bit_*`  | `nibble × scale + bias` (per-group) |
| `"bf16"` | `matvec_bf16`            | direct dot product, no dequant      |
| `"u8"`   | `matvec_int8` **(planned)** | `int8(w) × scale[i]` (per-channel) |

No engine variant needed — one engine dispatches per-tensor.  Mixed
quantization is a property of the weight file, not the runtime.

## Weight conversion

Split on the last dot to get the block and kind, then feed the name through
`bq4` above.  The resulting `Quant` is written as the `dtype` in the manifest
JSON.

The `.weight` suffix is **preserved** in the manifest for BF16/BF16Pass/INT8
1D and 2D weight tensors (e.g. `language_model.model.layers.0.input_layernorm.weight`).
For INT4 tensors, the suffix is stripped to form a base name, then three
separate entries are written: `{.weight, .scales, .biases}`.

## Expert router

The expert router is a single linear projection `W_gate ∈ R[num_experts × hidden_dim]`
stored as `language_model.model.layers.{L}.mlp.gate.weight`.

**Forward pass:**
1. Post-attention hidden state is RMS-normed to produce `h ∈ R[hidden_dim]`
2. GPU: `scores = W_gate · h` (into `buf_gate_scores`), executed inside the
   op1 encoder alongside attention projections
3. CPU: softmax → top-k → normalize → select expert buffers

**Why BF16Pass.**  The gate is `[256 × 2048]` = ~500K floats.  Quantizing it
saves ~1.5MB total across all 40 layers — negligible.  But a single bit-flip
can reroute a token from expert 47 to expert 231, wasting all subsequent expert
computation.  The error multiplier makes this the most expensive quantization
in the model per byte saved.

## Qwen3.5 vs Qwen3.6 norm weight convention

Qwen3.6 changed the convention for RMS norm weights: they are shifted by -1.0
relative to Qwen3.5.  MLX-LM's sanitizer bakes a +1.0 correction into the
quantized weights so the runtime formula `y = x * w` works for both.

Our engines follow the **Qwen3.5 convention** (no runtime shift).  To quantize
a Qwen3.6 model, pass `--qwen36` to `quantize.py`.  This normalizes the norm
weights to Qwen3.5 convention at quantization time.

Without `--qwen36` on a Qwen3.6 model, norm weights will be ~1.0 too low,
causing all RMS norm operations to produce near-zero outputs and the model
to generate garbage.
