//! Gemma 4 MoE engine.
//!
//! Scaffolded but not yet implemented. The Engine trait impl lives in
//! `engine.rs`; the kernel surface (sliding-window attention, dual RoPE,
//! GELU, final-logit softcap) is sketched in `shaders.metal` and
//! `metal_kernels.rs`. None of this is wired into `DynEngine` yet — that
//! happens once at least one forward pass works.
//!
//! Layout mirrors `qwen35_moe/`:
//!
//!   constants.rs         — ModelConfig trait + Gemma4_26B_A4B marker
//!   metal_context.rs     — MetalContext + buffer allocation (mostly TODO)
//!   metal_kernels.rs     — Rust dispatch wrappers (mostly TODO)
//!   shaders.metal        — Gemma 4 specific Metal kernels (mostly TODO)
//!   engine.rs            — FusedExp struct + Engine trait impl (mostly TODO)
//!   batched.rs           — eventually, batched-prefill path (deferred)

pub mod constants;
pub mod metal_context;
pub mod metal_kernels;
pub mod engine;
