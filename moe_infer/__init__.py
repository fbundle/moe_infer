"""
MoE-Infer: High-performance MoE inference engine for Apple Silicon.

Re-exports the native _moe_infer_rs module as a clean public API.
"""

from _moe_infer_rs import (  # type: ignore
    Model,
    Engine,
    Cache,
    record_engine_telemetry,
    qwen35_moe_bq4_quantize,
)

__all__ = [
    "Model",
    "Engine",
    "Cache",
    "record_engine_telemetry",
    "qwen35_moe_bq4_quantize",
]
