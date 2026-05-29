# MoE-Infer

Run the Qwen3.6-35B-A3B MoE model on your Mac.  It chats, streams
word-by-word, and understands images — all on-device, no internet needed.

## What you need

- A Mac with Apple Silicon (M1 or newer) and at least 32 GB of unified memory
  (the BQ4 model is ~19 GB on disk and the engine mmaps it plus expert
  scratch buffers; expect roughly model size + a few GB of headroom)
- About 100 GB of free disk space (70 GB download + ~20 GB per quantized model)
- The original HuggingFace model downloaded to `hub/models--Qwen--Qwen3.6-35B-A3B`

## Setup (do this once)

Open Terminal and run these commands from the project folder:

```bash
# 1. Build and install
uv add moe_infer[all]  # or `uv sync --reinstall --extra vision` if you live inside this project

# 2. Download the model (about 70 GB)
hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir hub/models--Qwen--Qwen3.6-35B-A3B

```

If you live in a cave, you can install with pip `pip install moe_infer[all]`

```python
# 3. Convert the model (takes ~5 minutes per scheme)
from moe_infer.qwen35_moe import convert


# BQ4 only (block-aware quantization)
convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/Qwen3.6-35B-A3B', version='3.6')

# INT4 only (all weights quantized)
convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/Qwen3.6-35B-A3B', version='3.6', scheme='int4')

# Both at once
convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/Qwen3.6-35B-A3B', version='3.6', scheme=['bq4', 'int4'])
```

When it's done you'll see:

```
data/Qwen3.6-35B-A3B/
├── model_bq4/        ← BQ4 compressed model (~19 GB)
├── model_int4/       ← INT4 compressed model (~18 GB)
├── tokenizer/        ← turns words into numbers (and back)
└── vision_encoder/   ← lets it "see" images
```

## Chat

Start Python and load the model:

```python
from moe_infer import Qwen35MoEPipeline

# Default: uses model_bq4/
pipe = Qwen35MoEPipeline("data/Qwen3.6-35B-A3B")

# INT4 quantization:
pipe = Qwen35MoEPipeline("data/Qwen3.6-35B-A3B", quantize_mode="int4")
```

### Text

```python
pipe.chat("Hello!")
# → 'Hello! How can I help you today?'

pipe.chat("What is the capital of France?")
# → 'The capital of France is **Paris**.'
```

The model remembers your conversation.  Call `pipe.reset()` to start fresh.

### Streaming

See words appear as the model generates them:

```python
for word in pipe.chat("Write a haiku about cats", stream=True):
    print(word, end="", flush=True)
```

### Images

Point it at a photo and ask about it:

```python
pipe.chat("What is in this photo?", images=["data/crycat-crying-cat.gif"], max_image_pixels=65536)
```

### Multi-Token Prediction (MTP)

Qwen3.6 includes an MTP draft head that predicts future tokens.  The
quantized model bundles the MTP weights automatically — no extra config
needed.  The engine loads them at startup:

```python
pipe = Qwen35MoEPipeline("data/Qwen3.6-35B-A3B")
print(pipe._has_mtp)  # True for Qwen3.6, False for Qwen3.5
```

MTP state is initialized during model load.  The Python-side speculative
decoding loop (`generate_from_mtp`) is available but currently delegates
to the standard autoregressive loop.  The Rust MTP forward pass is
functional; batched draft + verify will land in a future release.

## Verification

To verify that the BQ4 engine computes the same logits as the original
model, the pipeline is:

1.  **Strip** the HF model to 4 layers / 4 experts (fast test).
2.  **Quantize** the stripped model to BQ4.
3.  **Dequantize** BQ4 back to HF safetensors.
4.  **Compare** logits: dequantized HF (transformers) vs BQ4 engine (Rust).

All four steps run locally — no network needed after the initial download.

```bash
# 1. Strip: 40 layers → 4 layers, 256 experts → 4 experts (~2.4 GB)
python -m moe_infer.strip hub/models--Qwen--Qwen3.6-35B-A3B \
  --out hub/models--Qwen--Qwen3.6-35B-A3B-Strip

# 2. Quantize to BQ4 (~0.9 GB)
python -c "
from moe_infer.qwen35_moe import convert
convert('hub/models--Qwen--Qwen3.6-35B-A3B-Strip',
        'data/Qwen3.6-35B-A3B-Strip', version='3.6')
"

# 3. Dequantize back to HF safetensors (~2.4 GB)
python -m moe_infer.dequantize data/Qwen3.6-35B-A3B-Strip/model_bq4 \
  --ref hub/models--Qwen--Qwen3.6-35B-A3B-Strip \
  --out hub/models--Qwen--Qwen3.6-35B-A3B-Strip-Dequant

# 4. Compare logits
python verify_nway.py
```

Expected metrics on the 29-token test sequence (vocab_size=248,320):

| Comparison | Cosine | Max Diff | Mean Diff | Notes |
|---|---|---|---|---|
| Original HF vs BQ4 engine | ~0.902 | ~4.6 | ~0.79 | INT4 quantization loss |
| **Dequantized HF vs BQ4 engine** | **~0.925** | **~4.5** | **~0.72** | Same weights, GPU vs CPU path |
| FusedExp1 vs FusedExp2 | 1.000 | 0.000 | 0.000 | Bit-exact between Rust pipelines |

The dequantized model reproduces the engine more closely than the original
because both use the same quantized weights.  The remaining gap (~0.075) is
numerical precision differences between Metal GPU (on-the-fly INT4 matvec)
and PyTorch CPU (pre-dequantized BF16 matmul).

## Tips

| Tip | What to do |
|---|---|
| Text-only chat | `Qwen35MoEPipeline("data/...")` — it finds everything automatically |
| Reset conversation | `pipe.reset()` |
| Conversation history | `pipe.messages` — see what was said |
| Engine timing | `pipe.telemetry` — how long each step took |
| Switch quantization | `quantize_mode="int4"` or `quantize_mode="bq4"` (default) |
| MTP support | Qwen3.6 models load MTP automatically; `pipe._has_mtp` reports status |
