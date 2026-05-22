/// Compile-time constants (from gen_config.py concepts).
///
/// These are fixed at compile time and control buffer sizes, array dimensions, etc.
/// All model-specific dimensions come from model_config.json at runtime.

/// Maximum number of active experts per token.
/// Controls fixed-size buffer arrays throughout the engine.
pub const MAX_K: usize = 8;

/// Number of parallel I/O threads for expert weight pread.
pub const NUM_IO_THREADS: usize = 8;

/// Maximum active experts (upper bound for buffer allocation).
pub const MAX_ACTIVE_EXPERTS: usize = 64;

/// Number of output rows per threadgroup in the v3/v5 shaders.
pub const ROWS_PER_TG: u32 = 8;


/// Threadgroup size for optimized kernels.
pub const TG_SIZE: u32 = 256;

/// SIMD width (Apple GPU = 32).
pub const SIMD_WIDTH: u32 = 32;
