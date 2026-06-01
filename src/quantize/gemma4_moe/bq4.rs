//! Gemma 4 26B-A4B BQ4 quantization pipeline (scaffold).
//!
//! Mirrors `quantize/qwen35_moe/bq4.rs` in structure but with the Gemma 4
//! tensor name map (see `name_mapping.json` in this directory) and the
//! Gemma-specific differences:
//!
//!   - Tied input/output embeddings — only emit one packed weight; the
//!     engine derives `lm_head` from it at load time.
//!   - No shared expert tensors to handle.
//!   - Per-layer pre/post FFN norms (`pre_feedforward_layernorm`,
//!     `post_feedforward_layernorm`) in addition to the usual input/post-
//!     attention norms.
//!   - Per-head Q/K norms (`q_norm`, `k_norm`) — emit as BF16, no shift.
//!   - Possible fused `gate_up_proj` per expert (need to verify against
//!     actual safetensors header — code is structured to accept either
//!     fused or split layouts via a runtime check).
//!
//! Status: scaffold only — no quantize function body yet. The structure
//! follows BQ4: non-expert tensors packed into `model_weights.bin` +
//! `model_weights.json` manifest, per-layer expert blobs into
//! `packed_experts/layer_XX.bin`.

#![allow(dead_code)]

use crate::error::MoEError;

/// Quantize a Gemma 4 26B-A4B HF model directory into the engine's BQ4 format.
///
/// Inputs:
///   `hf_dir` — path to HuggingFace model dir (config.json + sharded safetensors).
///   `out_dir` — output dir. Will contain `config.json`, `model_weights.bin`,
///       `model_weights.json`, `packed_experts/layer_XX.bin`.
///
/// Status: TODO. Mirror `quantize/qwen35_moe/bq4.rs`'s `quantize()` after
/// (a) verifying the tensor names in name_mapping.json against an actual
/// model and (b) deciding whether to fuse gate/up or keep them split.
pub fn quantize(_hf_dir: &str, _out_dir: &str) -> Result<(), MoEError> {
    Err(MoEError::Config(
        "Gemma 4 BQ4 quantize pipeline is not yet implemented (scaffold only).".into()
    ))
}
