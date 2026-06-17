"""Minimal trace-friendly Qwen2.5 implementation for CoreML export.

We avoid all of `transformers`' complex cache-position / shape-inference code
paths — those are what trip the coremltools `_int` op converter. Instead we
implement the architecture from scratch using only ops the converter handles
cleanly (matmul, add, mul, softmax, layernorm, gather, slice).

Weights are loaded directly from the HF safetensors. Architecture is
standard Qwen2.5: tied or untied embeddings + N × (RMSNorm + GQA-with-RoPE
+ SwiGLU MLP) + final norm + LM head. Single-token decode with externally-
maintained KV cache. No SDPA, no dynamic shapes.

The class only supports single-token decode (T=1). Prefill is the caller's
problem (loop or build a separate fixed-T graph at a power-of-two prompt
length, per the multi-cache-size plan in the README).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] or [B, T, H, Dh] — normalize over last dim.
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def precompute_rope_cache(head_dim: int, max_seq: int, base: float = 1000000.0):
    """Returns (cos, sin) of shape [max_seq, head_dim/2] each."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [max_seq, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope_single(q_or_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a single-position tensor.

    q_or_k: [B, H, 1, head_dim]
    cos, sin: [1, 1, 1, head_dim/2] (caller has indexed into the cache).
    """
    d = q_or_k.shape[-1]
    half = d // 2
    x1 = q_or_k[..., :half]
    x2 = q_or_k[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    out_x = q_or_k * torch.cat([cos, cos], dim=-1) + rotated * torch.cat([sin, sin], dim=-1)
    return out_x


class GQAAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 use_qk_bias: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=use_qk_bias)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=use_qk_bias)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=use_qk_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,              # [B=1, T=1, hidden]
        k_cache: torch.Tensor,        # [B=1, n_kv_heads, max_seq, head_dim]
        v_cache: torch.Tensor,        # [B=1, n_kv_heads, max_seq, head_dim]
        position: int,                # 0..max_seq-1 (constant at trace time)
        cos: torch.Tensor,            # [max_seq, head_dim/2]
        sin: torch.Tensor,            # [max_seq, head_dim/2]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # [B, H, T=1, Dh]
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE for this single position.
        cos_p = cos[position:position+1].view(1, 1, 1, -1)
        sin_p = sin[position:position+1].view(1, 1, 1, -1)
        q = apply_rope_single(q, cos_p, sin_p)
        k = apply_rope_single(k, cos_p, sin_p)

        # Update KV cache (out-of-place — caller persists the new cache).
        # We use scatter so the position dimension can be a tensor at trace time.
        k_cache_new = k_cache.clone()
        v_cache_new = v_cache.clone()
        k_cache_new[:, :, position:position+1, :] = k
        v_cache_new[:, :, position:position+1, :] = v

        # Repeat KV heads to match Q heads.
        if self.n_heads != self.n_kv_heads:
            rep = self.n_heads // self.n_kv_heads
            k_full = k_cache_new.repeat_interleave(rep, dim=1)
            v_full = v_cache_new.repeat_interleave(rep, dim=1)
        else:
            k_full = k_cache_new
            v_full = v_cache_new

        # Scaled dot product (T=1 query against max_seq keys).
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # [B, H, 1, max_seq]

        # Causal mask: positions > `position` get -inf.
        max_seq = k_cache.shape[-2]
        mask = torch.arange(max_seq) > position
        mask = mask.view(1, 1, 1, max_seq).to(scores.dtype) * torch.tensor(-1e4, dtype=scores.dtype)
        scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_full)  # [B, H, 1, Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), k_cache_new, v_cache_new


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.attn = GQAAttention(
            cfg["hidden_size"], cfg["num_attention_heads"],
            cfg["num_key_value_heads"], cfg["head_dim"],
            use_qk_bias=cfg.get("qkv_bias", True),
        )
        self.post_norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp = SwiGLUMLP(cfg["hidden_size"], cfg["intermediate_size"])

    def forward(self, x, k_cache, v_cache, position, cos, sin):
        attn_out, k_new, v_new = self.attn(self.input_norm(x), k_cache, v_cache, position, cos, sin)
        x = x + attn_out
        x = x + self.mlp(self.post_norm(x))
        return x, k_new, v_new


class Qwen25Decoder(nn.Module):
    """Single-token decode-only Qwen2.5 model.

    Forward signature: (input_id, position, k_caches, v_caches, cos, sin) →
    (logits, new_k_caches, new_v_caches). All caches are tuples/lists of
    per-layer tensors so torch.jit.trace can handle them.
    """

    def __init__(self, cfg, max_seq: int):
        super().__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        if cfg.get("tie_word_embeddings", False):
            self.lm_head = None  # use self.embed.weight at forward time
        else:
            self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def forward(self, input_id, position, k_caches, v_caches, cos, sin):
        x = self.embed(input_id)  # [1, 1, hidden]
        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            x, k_new, v_new = layer(x, k_caches[i], v_caches[i], position, cos, sin)
            new_k.append(k_new)
            new_v.append(v_new)
        x = self.norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.embed.weight)
        else:
            logits = self.lm_head(x)
        return logits, new_k, new_v


def load_weights(model: Qwen25Decoder, hf_dir: Path, cfg) -> None:
    """Load HF safetensors weights into our decoder."""
    # Build the file → tensor index from safetensors.index.json.
    import json
    idx = json.loads((hf_dir / "model.safetensors.index.json").read_text()) \
        if (hf_dir / "model.safetensors.index.json").exists() else None
    if idx is None:
        files = [hf_dir / "model.safetensors"]
        weight_map = None
    else:
        weight_map = idx["weight_map"]
        files = sorted({hf_dir / f for f in weight_map.values()})

    def get(name):
        if weight_map is not None:
            f = hf_dir / weight_map[name]
        else:
            f = files[0]
        with safe_open(str(f), framework="pt") as st:
            return st.get_tensor(name)

    # Map HF → ours
    sd = {}
    sd["embed.weight"] = get("model.embed_tokens.weight")
    sd["norm.weight"] = get("model.norm.weight")
    if not model.cfg.get("tie_word_embeddings", False):
        sd["lm_head.weight"] = get("lm_head.weight")
    for i in range(model.cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        sd[f"layers.{i}.input_norm.weight"] = get(f"{p}.input_layernorm.weight")
        sd[f"layers.{i}.post_norm.weight"] = get(f"{p}.post_attention_layernorm.weight")
        sd[f"layers.{i}.attn.q_proj.weight"] = get(f"{p}.self_attn.q_proj.weight")
        sd[f"layers.{i}.attn.k_proj.weight"] = get(f"{p}.self_attn.k_proj.weight")
        sd[f"layers.{i}.attn.v_proj.weight"] = get(f"{p}.self_attn.v_proj.weight")
        sd[f"layers.{i}.attn.o_proj.weight"] = get(f"{p}.self_attn.o_proj.weight")
        # Qwen2.5 q/k/v projections do have biases.
        sd[f"layers.{i}.attn.q_proj.bias"] = get(f"{p}.self_attn.q_proj.bias")
        sd[f"layers.{i}.attn.k_proj.bias"] = get(f"{p}.self_attn.k_proj.bias")
        sd[f"layers.{i}.attn.v_proj.bias"] = get(f"{p}.self_attn.v_proj.bias")
        sd[f"layers.{i}.mlp.gate_proj.weight"] = get(f"{p}.mlp.gate_proj.weight")
        sd[f"layers.{i}.mlp.up_proj.weight"] = get(f"{p}.mlp.up_proj.weight")
        sd[f"layers.{i}.mlp.down_proj.weight"] = get(f"{p}.mlp.down_proj.weight")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:   print(f"[load] missing keys (allowed for tied lm_head): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected: print(f"[load] unexpected keys: {unexpected[:5]}")


def build_qwen25(hf_dir: Path, max_seq: int) -> tuple[Qwen25Decoder, tuple]:
    """Load the HF model into our minimal Qwen2.5 implementation and return
    (model, (cos, sin)) where (cos, sin) are the precomputed RoPE tables."""
    import json
    cfg_raw = json.loads((hf_dir / "config.json").read_text())
    cfg = {
        "vocab_size": cfg_raw["vocab_size"],
        "hidden_size": cfg_raw["hidden_size"],
        "intermediate_size": cfg_raw["intermediate_size"],
        "num_hidden_layers": cfg_raw["num_hidden_layers"],
        "num_attention_heads": cfg_raw["num_attention_heads"],
        "num_key_value_heads": cfg_raw["num_key_value_heads"],
        "head_dim": cfg_raw.get("head_dim", cfg_raw["hidden_size"] // cfg_raw["num_attention_heads"]),
        "rms_norm_eps": cfg_raw.get("rms_norm_eps", 1e-6),
        "rope_theta": cfg_raw.get("rope_theta", 1000000.0),
        "tie_word_embeddings": cfg_raw.get("tie_word_embeddings", False),
        "qkv_bias": True,  # Qwen2.5 has biases on q/k/v
    }
    model = Qwen25Decoder(cfg, max_seq).eval().half()
    load_weights(model, hf_dir, cfg)
    cos, sin = precompute_rope_cache(cfg["head_dim"], max_seq, cfg["rope_theta"])
    cos = cos.half()
    sin = sin.half()
    return model, (cos, sin)
