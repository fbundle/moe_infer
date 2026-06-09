#[path = "math_util.rs"] mod math;
#[path = "engine.rs"] pub mod engine;
#[path = "model.rs"] pub mod model;

mod cache;
mod constants;
mod error;
pub mod hf_util;
mod dtype;
pub mod quantize;
mod safetensors;
mod timer;

// Qwen35MoE quantization schemes (directory layout is for organization only)
#[path = "quantize/qwen35_moe/common.rs"] mod qwen35_moe_common;
#[path = "quantize/qwen35_moe/bq4.rs"] pub mod bq4;
#[path = "quantize/qwen35_moe/int4.rs"] pub mod int4;

// Qwen3.5 dense variant (e.g. Qwen3.5-4B) — INT4 only, and selective BQ4.
#[path = "quantize/qwen35_dense/int4.rs"] pub mod qwen35_dense_int4;
#[path = "quantize/qwen35_dense/bq4.rs"]  pub mod qwen35_dense_bq4;

// Gemma 4 quantize pipeline — BF16 passthrough (first-cut).
#[path = "quantize/gemma4_moe/bq4.rs"] pub mod gemma4_bq4;

// Gemma 4 12B dense INT4 — multimodal projections kept inline (audio = single
// matmul; vision = patch_dense + pos_embedding + 2 LayerNorms).
#[path = "quantize/gemma4_dense/int4.rs"] pub mod gemma4_dense_int4;

#[cfg(feature = "python-bindings")]
mod python_bindings;

#[cfg(feature = "python-bindings")]
#[pyo3::pymodule]
fn _moe_infer_rs(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    use pyo3::wrap_pyfunction;
    m.add_class::<python_bindings::Model>()?;
    m.add_class::<python_bindings::Engine>()?;
    m.add_class::<python_bindings::PyEngineSnapshot>()?;
    m.add_class::<python_bindings::Cache>()?;
    m.add_function(wrap_pyfunction!(python_bindings::record_engine_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(python_bindings::qwen35_moe_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(python_bindings::qwen35_dense_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(python_bindings::gemma4_moe_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(python_bindings::gemma4_dense_quantize, m)?)?;
    m.add_class::<python_bindings::PyHfRepo>()?;
    Ok(())
}
