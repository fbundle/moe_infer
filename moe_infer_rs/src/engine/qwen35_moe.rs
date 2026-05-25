#[path = "qwen35_moe/constants.rs"]
pub mod constants;
#[path = "qwen35_moe/cpu.rs"]
pub mod cpu;
#[path = "qwen35_moe/fused_4bit.rs"]
pub mod fused_4bit;
#[path = "qwen35_moe/fused_4bit_exp1.rs"]
pub mod fused_4bit_exp1;
#[path = "qwen35_moe/fused_4bit_exp2.rs"]
pub mod fused_4bit_exp2;
#[path = "qwen35_moe/fused_4bit_exp3.rs"]
pub mod fused_4bit_exp3;
#[path = "qwen35_moe/metal_context.rs"]
pub mod metal_context;
#[path = "qwen35_moe/metal_kernels.rs"]
pub mod metal_kernels;

pub use constants::{ModelConfig, FullModel, StrippedModel};
pub use cpu::CpuEngine;
pub use fused_4bit::Fused4bit;
pub use fused_4bit_exp1::Fused4bitExp1;
pub use fused_4bit_exp2::Fused4bitExp2;
pub use fused_4bit_exp3::Fused4bitExp3;

/// Type alias for stripped model variant.
pub type Fused4bitStripped<'a> = Fused4bit<'a, StrippedModel>;

/// Type alias for experimental stripped model variants.
pub type Fused4bitExp1Stripped<'a> = Fused4bitExp1<'a, StrippedModel>;
pub type Fused4bitExp2Stripped<'a> = Fused4bitExp2<'a, StrippedModel>;
pub type Fused4bitExp3Stripped<'a> = Fused4bitExp3<'a, StrippedModel>;
