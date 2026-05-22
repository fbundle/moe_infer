pub mod config;
pub mod constants;
pub mod engine;
pub mod error;
pub mod pipeline_gpu;
pub mod metal_kernels;
pub mod metal_context;
pub mod pipeline_common;
pub mod pipeline_cpu;
pub mod pipeline_fusedwoods;
pub mod pipeline_fusedexp;
pub mod timer;
pub mod weights;

#[cfg(feature = "python-bindings")]
mod python_bindings;

// Re-export key types
pub use config::{load_model_config, ExpertLayout, ModelConfig};
pub use constants::*;
pub use error::MoEError;

pub use pipeline_gpu::{moe_layer_forward, linear_attention_forward, full_attention_forward};
pub use pipeline_common::{LinearAttnState, FullAttnCache, FullAttnCmd2State, DeferredExperts, PipelineMode, bf16_to_f32, cpu_dequant_matvec_4bit, cpu_rms_norm};
pub use pipeline_fusedwoods::LinearAttnFusedWoodsState;
pub use metal_context::{MetalContext, GpuWeightCtx, ExpertIOState, ExpertCache, metal_buf_shared};
pub use timer::now_ms;
pub use weights::WeightFile;

#[cfg(feature = "python-bindings")]
#[pyo3::pymodule]
fn moe_infer(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    m.add_class::<python_bindings::Model>()?;
    m.add_class::<python_bindings::Engine>()?;
    m.add_class::<python_bindings::Cache>()?;
    Ok(())
}
