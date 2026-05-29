"""Autoregressive token generation loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from moe_infer._core import Engine, Cache
from moe_infer.sampling import sample


def generate_from(
    first_logits: np.ndarray,
    engine: Engine,
    cache: Cache,
    tokenizer: Any,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    eos_ids: tuple[int, ...] = (248046, 248044),
    on_token: Callable[[int], None] | None = None,
    mtp: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Generate tokens autoregressively from pre-computed logits.

    Parameters
    ----------
    first_logits : np.ndarray
        Logits from the prefill step, shape ``[vocab_size]`` or
        ``[1, vocab_size]``.  The last row is used for the first sample.
    engine : Engine
        GPU inference engine.
    cache : Cache
        KV-cache (mutated in-place as tokens are generated).
    tokenizer : Any
        HF-compatible tokenizer with a ``.decode(ids)`` method.
    max_tokens : int
        Maximum new tokens to generate.
    temperature : float
        Sampling temperature.  0.0 = greedy.
    top_k : int
        Top-k filtering.  0 = disabled.
    top_p : float
        Nucleus (top-p) filtering.  1.0 = disabled.
    min_p : float
        Min-p filtering.  0.0 = disabled.
    eos_ids : tuple[int, ...]
        Token ids that signal end-of-sequence.
    on_token : callable or None
        Called with each token id as it is sampled (for streaming output).
    mtp : bool
        Forwarded to ``engine.forward_hidden`` to enable MTP state capture.

    Returns
    -------
    (text, stats) : tuple[str, dict]
        Decoded text and a stats dict with keys ``tokens``, ``seconds``,
        ``tok_per_s``.
    """
    t0 = time.time()
    last = np.asarray(
        first_logits[-1] if first_logits.ndim == 2 else first_logits
    )
    generated: list[int] = []

    for _ in range(max_tokens):
        tok = sample(last, temperature, top_k, top_p, min_p)
        if tok in eos_ids:
            break
        generated.append(tok)
        if on_token is not None:
            on_token(tok)
        last = engine.forward(
            np.array([tok], dtype=np.int64), cache, mtp=mtp,
        )[0]

    dt = time.time() - t0
    n = len(generated)
    text = tokenizer.decode(generated)
    stats: dict[str, Any] = {
        "tokens": n,
        "seconds": dt,
        "tok_per_s": n / dt if n > 0 else 0,
    }
    return text, stats


def generate_from_mtp(
    first_logits: np.ndarray,
    engine: Engine,
    cache: Cache,
    tokenizer: Any,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    eos_ids: tuple[int, ...] = (248046, 248044),
    on_token: Callable[[int], None] | None = None,
    mtp: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Generate tokens with MTP-aware engine (standard loop for now).

    The engine initialises MTP state during model load.  This generator
    uses the standard autoregressive loop; speculative batching will be
    added once KV-cache consistency is handled at the engine level.

    Parameters match :func:`generate_from`.
    """
    return generate_from(
        first_logits, engine, cache, tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        eos_ids=eos_ids,
        on_token=on_token,
        mtp=mtp,
    )
