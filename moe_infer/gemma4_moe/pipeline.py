"""Gemma 4 26B-A4B MoE pipeline — Gemma-specific defaults."""

from __future__ import annotations

from typing import Any

from moe_infer.pipeline import Pipeline


class Gemma4MoEPipeline(Pipeline):
    """Pipeline pre-configured for Gemma 4 26B-A4B MoE.

    Sets Gemma-specific defaults: ``Gemma4MoEFused`` engine mode, EOS tokens
    (``<eos>`` = 1 and ``<turn|>`` = 106), and a continuation format that
    re-uses the existing KV cache for multi-turn chat.

    Auto-discovery of ``model_bq4/`` and ``tokenizer/`` is handled by the
    base :class:`Pipeline`.

    Usage::

        from moe_infer.gemma4_moe import Gemma4MoEPipeline

        pipe = Gemma4MoEPipeline("data/gemma-4-26B-A4B")
        pipe.chat("Hello!")
    """

    # `<eos>` (1) is the official terminator. `<turn|>` (106) ends each turn
    # in our chat template; sampling it should stop the turn.
    eos_ids: tuple[int, ...] = (1, 106)

    # Continuation format for follow-up turns: re-encode without <bos>,
    # since the first turn already injected it.
    _continuation_fmt: str = (
        "<turn|>\n<|turn><|channel>user<channel|>\n"
        "{message}<turn|>\n<|turn><|channel>model<channel|>\n"
    )

    @classmethod
    def _extract_response(cls, raw: str) -> str:
        """Strip the trailing turn marker and an optional thought channel.

        Gemma 4 reasoning mode emits ``<|channel>thought\\n...<channel|>``
        before the user-visible answer. We drop the thought block.
        """
        text = raw.removesuffix("<turn|>")
        # Strip thought channel if present.
        if "<|channel>thought" in text and "<channel|>" in text:
            head, _, rest = text.partition("<|channel>thought")
            _, _, answer = rest.partition("<channel|>")
            text = head + answer
        return text

    @staticmethod
    def _load_vision_encoder(hub_path: str) -> Any:
        # Vision support is not yet wired for Gemma 4. The vision_tower
        # tensors are skipped at quantize time (WeightKind::Skip). When
        # we add it, this method should load the vision encoder from
        # the converted vision_encoder/ subdirectory.
        raise NotImplementedError(
            "Gemma 4 vision encoder is not yet wired. The Phase 2 forward "
            "handles text only. The vision_tower tensors are dropped at "
            "quantize time."
        )

    def __init__(
        self,
        model_path: str,
        *,
        hub: str | None = None,
        tokenizer: Any = None,
        num_active_experts: int = 0,
        expert_cache_count: int = 0,
    ) -> None:
        super().__init__(
            model_path,
            hub=hub,
            tokenizer=tokenizer,
            mode="Gemma4MoEFused",
            num_active_experts=num_active_experts,
            quantize_mode="bq4",
            expert_cache_count=expert_cache_count,
        )
