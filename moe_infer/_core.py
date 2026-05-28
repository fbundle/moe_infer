"""Private bindings over _moe_infer_rs — not part of the public API.

These thin wrappers delegate to the Rust extension via __getattr__,
adding docstrings and type annotations.  User code imports from
``moe_infer``, never from here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import _moe_infer_rs as _rs  # type: ignore[import-untyped]


# ── Model ─────────────────────────────────────────────────────────────────────

class Model:
    """A quantized MoE language model loaded from disk.

    Parameters
    ----------
    path : str
        Directory containing ``config.json``, ``model_weights.bin``,
        ``model_weights.json``, and ``packed_experts/``.
    """

    def __init__(self, path: str) -> None:
        self._inner = _rs.Model(path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return self._inner.__repr__()


# ── Engine ────────────────────────────────────────────────────────────────────

class Engine:
    """GPU inference engine backed by hand-tuned Metal compute shaders.

    Parameters
    ----------
    model : Model
        A loaded :class:`Model` instance.
    pipeline_mode : str
        One of ``"Qwen35MoEFusedExp1"`` or ``"Qwen35MoEFusedExp2"``.
    k : int
        Active experts per token.  0 means "use model default" (8 for Qwen3.6).
    """

    def __init__(
        self,
        model: Model,
        pipeline_mode: str = "Qwen35MoEFusedExp2",
        k: int = 0,
    ) -> None:
        self._inner = _rs.Engine(model._inner, pipeline_mode, k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return self._inner.__repr__()


# ── Cache ─────────────────────────────────────────────────────────────────────

class Cache:
    """KV-cache and linear-attention (DeltaNet) state for a conversation.

    Parameters
    ----------
    model : Model
        A loaded :class:`Model` instance used to size the caches.
    """

    def __init__(self, model: Model) -> None:
        self._inner = _rs.Cache(model._inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return self._inner.__repr__()


# ── Top-level functions ──────────────────────────────────────────────────────

def record_engine_telemetry(on: bool) -> None:
    """Enable or disable per-layer GPU timing telemetry globally.

    When enabled, :meth:`Engine.telemetry` returns a dict with keys like
    ``"engine.expert_io_ms"``, ``"engine.total_ms"``, etc.
    """
    _rs.record_engine_telemetry(on)


# ── Qwen-specific re-exports (moved to moe_infer.qwen35_moe) ─────────────────

from moe_infer.qwen35_moe import (  # noqa: E402
    convert as qwen35_moe_convert,
    extract_tokenizer as qwen35_moe_extract_tokenizer,
    extract_vision as qwen35_moe_extract_vision,
    quantize as qwen35_moe_quantize,
)
