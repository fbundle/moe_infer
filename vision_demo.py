#!/usr/bin/env python3
"""Vision demo for Qwen3.6-35B-A3B. Reads vision encoder directly from HF safetensors."""

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from moe_infer import Model, Engine, Cache

# ─── Vision config (from config.json vision_config) ───

HUB = Path("hub/models--Qwen--Qwen3.6-35B-A3B")

HIDDEN    = 1152
HEADS     = 16
HEAD_DIM  = 72     # 1152 / 16
INTER     = 4304
DEPTH     = 27
PATCH     = 16
TEMPORAL  = 2
MERGE     = 2
IMAGE_PX  = 768    # sqrt(2304 * 16²) — pos_embed has 2304 entries
NPATCH    = IMAGE_PX // PATCH  # 48 per side
NPATCHES  = NPATCH * NPATCH   # 2304
NMERGED   = (NPATCH // MERGE) ** 2  # 576

# Token IDs
VISION_START = 248053
VISION_END   = 248054
IMAGE_PAD    = 248055
IM_START     = "<|im_start|>"
IM_END       = "<|im_end|>"

# ─── Safetensors reader ──────────────────────────────────────────────────────

def _read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Read all tensors from a single safetensors file."""
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_off = 8 + hdr_len

    tensors = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        off = meta["data_offsets"]
        dtype = meta["dtype"]
        shape = meta["shape"]
        f.seek(data_off + off[0])
        raw = f.read(off[1] - off[0])
        if dtype == "BF16":
            arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            arr = (arr << 16).view(np.float32).reshape(shape)
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        else:
            raise ValueError(f"Unknown dtype {dtype}")
        tensors[name] = arr
    return tensors


def _load_shards(hub: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load all safetensors shards listed in model.safetensors.index.json."""
    with open(hub / "model.safetensors.index.json") as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    shard_names = sorted(set(wm.values()))
    shards = {}
    for sn in shard_names:
        shards[sn] = _read_safetensors(hub / sn)
    return shards


# ─── Image preprocessing ─────────────────────────────────────────────────────

def preprocess(path: str) -> np.ndarray:
    """Load image, resize to IMAGE_PX×IMAGE_PX, normalize, extract patches.
    Returns [NPATCHES, 3, PATCH, PATCH]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_PX, IMAGE_PX), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32) / 255.0            # [0,1]
    pixels = (pixels - 0.5) / 0.5                                # [-1, 1]
    pixels = pixels.transpose(2, 0, 1)                           # [3, H, W]

    patches = np.zeros((NPATCHES, 3, PATCH, PATCH), dtype=np.float32)
    idx = 0
    for i in range(NPATCH):
        for j in range(NPATCH):
            patches[idx] = pixels[:, i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH]
            idx += 1
    return patches


# ─── Activation ───────────────────────────────────────────────────────────────

def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# ─── Vision encoder ──────────────────────────────────────────────────────────

class VisionEncoder:
    def __init__(self, hub: Path):
        print("[vision] Loading weights from HF safetensors...", flush=True)
        t0 = time.time()
        shards = _load_shards(hub)
        wm = json.loads(open(hub / "model.safetensors.index.json").read())["weight_map"]

        def get(name: str) -> np.ndarray:
            return shards[wm[name]][name]

        self.patch_w = get("model.visual.patch_embed.proj.weight")  # [1152, 3, 2, 16, 16]
        self.patch_b = get("model.visual.patch_embed.proj.bias")    # [1152]
        self.pos_emb = get("model.visual.pos_embed.weight")          # [2304, 1152]

        self.blocks = []
        for b in range(DEPTH):
            p = f"model.visual.blocks.{b}"
            self.blocks.append({
                "n1w": get(f"{p}.norm1.weight"), "n1b": get(f"{p}.norm1.bias"),
                "qw":  get(f"{p}.attn.qkv.weight"), "qb": get(f"{p}.attn.qkv.bias"),
                "pw":  get(f"{p}.attn.proj.weight"), "pb": get(f"{p}.attn.proj.bias"),
                "n2w": get(f"{p}.norm2.weight"), "n2b": get(f"{p}.norm2.bias"),
                "f1w": get(f"{p}.mlp.linear_fc1.weight"), "f1b": get(f"{p}.mlp.linear_fc1.bias"),
                "f2w": get(f"{p}.mlp.linear_fc2.weight"), "f2b": get(f"{p}.mlp.linear_fc2.bias"),
            })

        self.m_nw = get("model.visual.merger.norm.weight")
        self.m_nb = get("model.visual.merger.norm.bias")
        self.m_f1w = get("model.visual.merger.linear_fc1.weight")
        self.m_f1b = get("model.visual.merger.linear_fc1.bias")
        self.m_f2w = get("model.visual.merger.linear_fc2.weight")
        self.m_f2b = get("model.visual.merger.linear_fc2.bias")

        total_mb = sum(arr.nbytes for arr in [self.patch_w, self.patch_b, self.pos_emb]
                       + [v for b in self.blocks for v in b.values()]
                       + [self.m_nw, self.m_nb, self.m_f1w, self.m_f1b, self.m_f2w, self.m_f2b])
        print(f"[vision] Loaded {total_mb/1e6:.1f} MB in {time.time()-t0:.1f}s", flush=True)

    # ── LayerNorm (vision uses LayerNorm, not RMSNorm) ──────────────────

    @staticmethod
    def _ln(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * w + b

    # ── Multi-head self-attention ───────────────────────────────────────

    def _attn(self, x: np.ndarray, blk: dict) -> np.ndarray:
        """x: [seq, HIDDEN] → [seq, HIDDEN]"""
        seq = x.shape[0]
        qkv = x @ blk["qw"].T + blk["qb"]                         # [seq, 3456]
        q, k, v = np.split(qkv, 3, axis=-1)                       # each [seq, 1152]
        q = q.reshape(seq, HEADS, HEAD_DIM)                        # [seq, 16, 72]
        k = k.reshape(seq, HEADS, HEAD_DIM)
        v = v.reshape(seq, HEADS, HEAD_DIM)

        scale = 1.0 / np.sqrt(HEAD_DIM)
        scores = q @ k.transpose(0, 2, 1) * scale                  # [seq, 16, seq]
        scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        scores /= scores.sum(axis=-1, keepdims=True)

        out = (scores @ v).reshape(seq, HIDDEN)                    # [seq, 1152]
        return out @ blk["pw"].T + blk["pb"]

    # ── MLP ────────────────────────────────────────────────────────────

    @staticmethod
    def _mlp(x: np.ndarray, blk: dict) -> np.ndarray:
        h = x @ blk["f1w"].T + blk["f1b"]
        h = gelu_tanh(h)
        return h @ blk["f2w"].T + blk["f2b"]

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, patches: np.ndarray) -> np.ndarray:
        """patches: [NPATCHES, 3, 16, 16] → vision features [NMERGED, 2048]"""
        t0 = time.time()

        # Patch embed: Conv3D [1152, 3, 2, 16, 16], stride [2, 16, 16]
        # Arrange patches into spatial grid, duplicate for temporal dim
        grid = patches.reshape(NPATCH, NPATCH, 3, PATCH, PATCH)  # [48, 48, 3, 16, 16]
        frame = np.zeros((3, IMAGE_PX, IMAGE_PX), dtype=np.float32)
        for i in range(NPATCH):
            for j in range(NPATCH):
                frame[:, i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH] = grid[i, j].transpose(1, 2, 0)
        frames = np.stack([frame, frame], axis=1)                  # [3, 2, 768, 768]

        # Apply Conv3D: for each output channel and spatial position
        w = self.patch_w.reshape(HIDDEN, -1)                       # [1152, 1536]
        x = np.zeros((HIDDEN, NPATCH, NPATCH), dtype=np.float32)
        for pi in range(NPATCH):
            for pj in range(NPATCH):
                si, sj = pi * PATCH, pj * PATCH
                patch = frames[:, :, si:si+PATCH, sj:sj+PATCH].reshape(-1)
                x[:, pi, pj] = w @ patch + self.patch_b

        x = x.reshape(HIDDEN, -1).T + self.pos_emb                 # [2304, 1152]

        # Transformer blocks
        for bi, blk in enumerate(self.blocks):
            if bi % 9 == 0:
                print(f"[vision] block {bi}/{DEPTH} ({time.time()-t0:.1f}s)", flush=True)
            x = x + self._attn(self._ln(x, blk["n1w"], blk["n1b"]), blk)
            x = x + self._mlp(self._ln(x, blk["n2w"], blk["n2b"]), blk)

        # Spatial merge: 2×2 groups → concatenate → [NMERGED, HIDDEN*4]
        x_2d = x.reshape(NPATCH, NPATCH, HIDDEN)
        merged = np.zeros((NMERGED, MERGE * MERGE * HIDDEN), dtype=np.float32)
        gh = NPATCH // MERGE
        for i in range(gh):
            for j in range(gh):
                group = x_2d[i*MERGE:(i+1)*MERGE, j*MERGE:(j+1)*MERGE, :]
                merged[i*gh + j] = group.reshape(-1)

        # Merger: LN → fc1 → GELU → fc2 → [NMERGED, 2048]
        h = self._ln(merged, self.m_nw, self.m_nb)
        h = gelu_tanh(h @ self.m_f1w.T + self.m_f1b)
        h = h @ self.m_f2w.T + self.m_f2b

        print(f"[vision] forward done in {time.time()-t0:.1f}s", flush=True)
        return h


# ─── Sampling ─────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def sample(logits: np.ndarray, temperature: float = 0.0) -> int:
    if temperature < 0.01:
        return int(np.argmax(logits))
    probs = softmax(logits / temperature)
    return int(np.random.choice(len(probs), p=probs))


# ─── Demo ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-35B-A3B vision demo")
    parser.add_argument("--image", default="vendor/crycat-crying-cat.gif")
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--model", default="data/models--Qwen--Qwen3.6-35B-A3B-bq4")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--hub", default=str(HUB))
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    hub = Path(args.hub)

    # 1. Preprocess image
    print(f"[demo] Loading image: {args.image}")
    patches = preprocess(args.image)

    # 2. Vision encoder forward
    vision = VisionEncoder(hub)
    vis_feats = vision.forward(patches)                            # [576, 2048]

    # 3. Load language model
    print(f"[demo] Loading model: {args.model}")
    model = Model(args.model)
    engine = Engine(model, pipeline_mode="Qwen35MoEBq4Exp2", k=0)
    cache = Cache(model)
    tokenizer = AutoTokenizer.from_pretrained(str(hub))

    # 4. Build input with vision tokens
    # Chat template: <|im_start|>user\n<|vision_start|><|image_pad|>×576<|vision_end|>\n{question}<|im_end|>\n<|im_start|>assistant\n
    vision_tokens = [VISION_START] + [IMAGE_PAD] * NMERGED + [VISION_END]
    text = f"{IM_START}user\n{'<|vision_start|>' + '<|image_pad|>' * NMERGED + '<|vision_end|>'}\n{args.question}{IM_END}\n{IM_START}assistant\n"
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    # Encode text tokens to embeddings
    print(f"[demo] Building embeddings ({len(input_ids)} tokens)...")
    text_embeds = engine.embed_lookup(np.array(input_ids, dtype=np.int64))  # [N, 2048]

    # Locate IMAGE_PAD positions in the token sequence
    pad_mask = np.array(input_ids) == IMAGE_PAD
    pad_indices = np.where(pad_mask)[0]

    if len(pad_indices) == 0:
        embeds = text_embeds
    elif len(pad_indices) != NMERGED:
        print(f"[demo] WARNING: found {len(pad_indices)} IMAGE_PAD tokens, expected {NMERGED}")
        # Use min of both
        n_use = min(len(pad_indices), NMERGED)
        embeds = text_embeds.copy()
        embeds[pad_indices[:n_use]] = vis_feats[:n_use]
    else:
        embeds = text_embeds
        embeds[pad_indices] = vis_feats

    # 5. Forward through LM
    print(f"[demo] Running LM forward...")
    t0 = time.time()
    logits = engine.forward_hidden(embeds, cache)                  # [N, vocab]
    print(f"[demo] Prefill: {time.time()-t0:.1f}s", flush=True)

    # 6. Generate
    last = np.asarray(logits[-1])
    eos_ids = {248046, 248044}  # <|im_end|>, <|endoftext|>
    generated: list[int] = []

    print(f"[demo] Generating (max {args.max_tokens} tokens)...")
    t_gen = time.time()
    for _ in range(args.max_tokens):
        tok = sample(last, args.temperature)
        if tok in eos_ids:
            break
        generated.append(tok)
        print(tokenizer.decode([tok]), end="", flush=True)
        logits = engine.forward_hidden(
            engine.embed_lookup(np.array([tok], dtype=np.int64)),
            cache,
        )
        last = np.asarray(logits[0])

    gen_s = time.time() - t_gen
    n_gen = len(generated)
    print(f"\n\n[demo] {n_gen} tokens in {gen_s:.1f}s ({n_gen/gen_s:.1f} tok/s)" if n_gen > 0
          else f"\n\n[demo] 0 tokens generated")


if __name__ == "__main__":
    main()
