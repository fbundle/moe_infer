pub mod constants;
pub mod fusedexp;
pub mod fusedwoods;

pub use constants::{ModelConfig, FullModel, StrippedModel};
pub use fusedexp::FusedExp;
pub use fusedwoods::{FusedWoods, FusedWoodsScratch};

/// Type aliases for stripped model variants.
pub type FusedExpStripped<'a> = FusedExp<'a, StrippedModel>;
pub type FusedWoodsStripped<'a> = FusedWoods<'a, StrippedModel>;
pub type FusedWoodsScratchStripped = FusedWoodsScratch<StrippedModel>;
