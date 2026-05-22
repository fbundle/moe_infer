#[path = "math/util.rs"] mod math;
pub mod engine;
pub mod model;

mod cache;
mod constants;
mod error;
#[path = "metal_util/kernels.rs"] mod metal_kernels;
#[path = "metal_util/context.rs"] mod metal_context;
mod timer;

#[cfg(feature = "python-bindings")]
mod python_bindings;

#[cfg(feature = "python-bindings")]
#[pyo3::pymodule]
fn moe_infer(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    m.add_class::<python_bindings::Model>()?;
    m.add_class::<python_bindings::Engine>()?;
    m.add_class::<python_bindings::Cache>()?;
    Ok(())
}
