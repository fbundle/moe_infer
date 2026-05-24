#[path = "qwen35_moe/constants.rs"]
pub mod constants;
#[path = "qwen35_moe/cpu.rs"]
pub mod cpu;
#[path = "qwen35_moe/fusedexp.rs"]
pub mod fusedexp;
#[path = "qwen35_moe/metal_context.rs"]
pub mod metal_context;
#[path = "qwen35_moe/metal_kernels.rs"]
pub mod metal_kernels;

pub use constants::{ModelConfig, FullModel, StrippedModel};
pub use cpu::CpuEngine;
pub use fusedexp::FusedExp;

/// Type alias for stripped model variant.
pub type FusedExpStripped<'a> = FusedExp<'a, StrippedModel>;
