#!/usr/bin/env python3
"""Run Rust engine with CpuOnly on stripped model with 1 token, capture layer-0 debug output."""
import numpy as np
from moe_infer import Context, Cache

MODEL_DIR = "/Volumes/Hippopotamus/vault/code/flash-moe/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit-stripped"

# Use just 1 token
token_ids = [248045]  # BOS token

print("Creating context...")
ctx = Context()
ctx.load_model(MODEL_DIR, pipeline_mode="CpuOnly")
cache = ctx.new_cache()

ids_arr = np.array(token_ids, dtype=np.int64)
logits_all = ctx.forward(ids_arr, cache)

print("Done.")
ctx.unload_model()
