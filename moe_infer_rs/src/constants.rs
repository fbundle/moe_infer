// Compile-time constants (mirrors config.h)
#![allow(unused)]

pub const NUM_ACTIVE_EXPERTS: usize = 8;
pub const EXPERT_CACHE_MODE: i32 = 0;
pub const USE_GPU_LINEAR: i32 = 1;
pub const MAX_K: usize = 8;

// Optimization flags
pub const USE_KV_CACHE_BF16: bool = true;
pub const USE_HEAP_TOPK: bool = true;
