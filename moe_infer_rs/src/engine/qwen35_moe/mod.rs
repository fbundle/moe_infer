pub mod constants;
pub mod fusedexp;
pub mod metal_context;
pub mod metal_kernels;

pub use constants::{ModelConfig, FullModel, StrippedModel};
pub use fusedexp::FusedExp;

/// Type alias for stripped model variant.
pub type FusedExpStripped<'a> = FusedExp<'a, StrippedModel>;
