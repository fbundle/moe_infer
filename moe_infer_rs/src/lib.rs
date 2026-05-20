// Flash-MoE inference engine — Rust port
//
// Architecture: single compilation unit style like the C codebase,
// but with proper Rust module boundaries. The main context struct
// FlashMoEContext holds ALL state (one per loaded model).

pub mod constants;
pub mod types;
pub mod config;
pub mod kernels;
pub mod weights;
pub mod embeddings;
pub mod attention;
pub mod expert_io;
pub mod generate;
pub mod metal;
pub mod gpu_ops;
pub mod gpu_forward;
pub mod cpu_forward;

use constants::MAX_K;
use types::*;
use std::fs::File;
use std::os::fd::AsRawFd;
#[cfg(feature = "timing")]
use std::time::Instant;

// ============================================================================
// Per-generation cache — one per concurrent sequence
// ============================================================================

/// Per-generation cache — one per concurrent sequence
#[derive(Debug)]
pub struct FlashMoECache {
    pub kv_caches: Vec<Option<KVCache>>,         // [num_layers] — Some for full-attn layers
    pub linear_states: Vec<Option<LinearAttnState>>, // [num_layers] — Some for linear layers
    pub pos: i32,
}

impl FlashMoECache {
    pub fn new(cfg: &ModelConfig) -> Self {
        let nl = cfg.num_layers as usize;
        let mut kv_caches = Vec::with_capacity(nl);
        let mut linear_states = Vec::with_capacity(nl);

        for i in 0..nl {
            let is_full = ((i + 1) % cfg.full_attn_interval as usize) == 0;
            if is_full {
                kv_caches.push(Some(KVCache::new(
                    cfg.max_seq_len as usize,
                    cfg.num_kv_heads as usize,
                    cfg.head_dim as usize,
                )));
                linear_states.push(None);
            } else {
                kv_caches.push(None);
                linear_states.push(Some(LinearAttnState::new(
                    cfg.conv_kernel_size,
                    cfg.linear_conv_dim,
                    cfg.linear_num_v_heads,
                    cfg.linear_value_dim,
                    cfg.linear_key_dim,
                )));
            }
        }
        Self { kv_caches, linear_states, pos: 0 }
    }

    pub fn reset(&mut self, cfg: &ModelConfig) {
        for i in 0..cfg.num_layers as usize {
            if let Some(ref mut kv) = self.kv_caches[i] {
                kv.len = 0;
            }
            if let Some(ref mut st) = self.linear_states[i] {
                st.conv_state.fill(0.0);
                st.ssm_state.fill(0.0);
            }
        }
        self.pos = 0;
    }
}

// ============================================================================
// Main model context — holds ALL state for one loaded model
// ============================================================================

pub struct FlashMoEContext {
    pub cfg: ModelConfig,
    pub model_path: String,
    pub wf: WeightFile,
    pub ht_data: weights::OwnedTensorHashTable,

    // Expert file I/O
    pub layer_fds: Vec<Option<File>>,
    pub layer_mmaps: Vec<*mut u8>,      // mmap'd expert files (zero-syscall access)
    pub layer_mmap_sizes: Vec<usize>,

    // Working buffers
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub final_norm_w: Vec<u16>,

    // Expert cache
    pub expert_cache: Option<expert_io::ExpertLRUCache>,
    pub malloc_cache: Option<expert_io::MallocExpertCache>,

    // Persistent I/O thread pool (4 workers)
    pub io_pool: Option<expert_io::IOPool>,

    // Metal GPU context (required — no CPU fallback)
    pub metal_ctx: metal::MetalCtx,

    // Per-layer weight caches (pre-computed offsets)
    pub layer_caches: Vec<gpu_forward::LayerWeightCache>,

    // Per-token scratch (reused across layers)
    pub layer_scratch: cpu_forward::CpuForwardScratch,

    // Deferred CMD3 for async expert pipeline (None when no pending CMD3)
    pub deferred: Option<gpu_forward::DeferredCmd3>,

    // GPU pipeline mode
    pub gpu_mode: gpu_forward::GpuMode,

    // Temporal prediction — previous token's routing predicts next token's
    pub pred_experts: Vec<i32>,       // [num_layers * MAX_K]
    pub pred_count: Vec<i32>,         // [num_layers]
    pub pred_valid: bool,
    pub pred_generating: bool,
    pub pred_hits: u64,
    pub pred_misses: u64,

    // Flags
    pub use_2bit: bool,
    pub initialized: bool,
}

impl FlashMoEContext {
    pub fn vocab_size(&self) -> i32 { self.cfg.vocab_size }
    pub fn hidden_dim(&self) -> i32 { self.cfg.hidden_dim }
    pub fn num_layers(&self) -> i32 { self.cfg.num_layers }
}

impl Drop for FlashMoEContext {
    fn drop(&mut self) {
        for i in 0..self.layer_mmaps.len() {
            if !self.layer_mmaps[i].is_null() {
                unsafe { expert_io::munmap(self.layer_mmaps[i], self.layer_mmap_sizes[i]); }
            }
        }
    }
}

// SAFETY: FlashMoEContext owns all its resources (mmap, files, GPU context).
// It is designed for single-owner use, wrapped in a Mutex for Python threading.
unsafe impl Send for FlashMoEContext {}
unsafe impl Sync for FlashMoEContext {}

// ============================================================================
// Forward pass — one token through all layers
// ============================================================================

/// Full forward pass: process n_tokens through all layers.
/// Updates cache in-place. Writes logits to logits_out.
pub fn flashmoe_forward(
    m: &mut FlashMoEContext,
    input_ids: &[i32],
    n_tokens: i32,
    logits_out: &mut [f32],
    cache: &mut FlashMoECache,
) -> Result<(), String> {
    let hidden_dim = m.cfg.hidden_dim as usize;
    let vocab_size = m.cfg.vocab_size as usize;
    let mut pos = cache.pos;
    let mode = m.gpu_mode;

    #[cfg(feature = "timing")]
    let t_total = Instant::now();
    #[cfg(feature = "timing")]
    let mut t_layers_us = 0u128;
    #[cfg(feature = "timing")]
    let mut t_lmhead_us = 0u128;

    // Reset prediction state at start of generation
    if cache.pos == 0 {
        m.pred_valid = false;
        m.pred_experts.fill(-1);
        m.pred_count.fill(0);
        m.pred_hits = 0;
        m.pred_misses = 0;
    }

    for tok in 0..n_tokens as usize {
        // Embedding lookup
        embeddings::embed_lookup(
            &m.wf, &m.ht_data, input_ids[tok],
            m.cfg.hidden_dim, &mut m.hidden,
        );
        // Layer loop — GPU when Metal available, CPU fallback
        let gpu_ctx = &m.metal_ctx;
        #[cfg(feature = "timing")]
        let t_layers = Instant::now();
        for layer in 0..m.cfg.num_layers as usize {
            let is_full = ((layer + 1) % m.cfg.full_attn_interval as usize) == 0;

            let kv = if is_full {
                cache.kv_caches[layer].as_mut()
            } else {
                None
            };
            let la_state = if !is_full {
                cache.linear_states[layer].as_mut()
            } else {
                None
            };

            let packed_fd = m.layer_fds[layer].as_ref();
            let layer_mmap = m.layer_mmaps[layer];

            unsafe {
                gpu_forward::gpu_layer_forward(
                    &m.cfg,
                    m.wf.data,
                    &m.layer_caches,
                    &mut m.hidden,
                    kv,
                    la_state,
                    pos,
                    packed_fd,
                    layer_mmap,
                    m.use_2bit,
                    &mut m.layer_scratch,
                    gpu_ctx,
                    m.expert_cache.as_mut(),
                    m.malloc_cache.as_mut(),
                    m.io_pool.as_ref(),
                    layer,
                    &mut m.deferred,
                    mode,
                    &mut m.pred_experts,
                    &mut m.pred_count,
                    &mut m.pred_valid,
                    m.pred_generating,
                );
            }

        }

        // Complete any pending deferred CMD3 from the last layer
        unsafe {
            gpu_forward::complete_deferred_experts(
                &m.metal_ctx,
                &mut m.deferred,
                &mut m.hidden,
                hidden_dim,
                &mut m.layer_scratch,
                m.expert_cache.as_mut(),
            );
        }

        pos += 1;

        // Final LayerNorm
        if !m.final_norm_w.is_empty() {
            let mut normed = vec![0.0f32; hidden_dim];
            kernels::cpu_rms_norm(
                &m.hidden, &m.final_norm_w, &mut normed,
                hidden_dim, m.cfg.rms_norm_eps,
            );
            m.hidden.copy_from_slice(&normed);
        }

        // LM head (GPU when Metal available)
        #[cfg(feature = "timing")]
        let t_lm = Instant::now();
        embeddings::lm_head_forward(
            &m.wf, &m.ht_data, &m.hidden,
            &mut logits_out[tok * vocab_size..(tok + 1) * vocab_size],
            m.cfg.vocab_size, m.cfg.hidden_dim, m.cfg.group_size,
            m.metal_ctx.as_ref(),
        );
        #[cfg(feature = "timing")]
        {
            t_layers_us += t_layers.elapsed().as_micros();
            t_lmhead_us += t_lm.elapsed().as_micros();
        }
    }

    #[cfg(feature = "timing")]
    eprintln!("[timing] {} tokens: tot={}ms, layers={}ms, lmhead={}ms",
        n_tokens,
        t_total.elapsed().as_millis(),
        t_layers_us / 1000,
        t_lmhead_us / 1000,
    );

    cache.pos = pos;
    Ok(())
}

// ============================================================================
// Init / Free
// ============================================================================

pub fn flashmoe_init(model_path: &str) -> Result<FlashMoEContext, String> {
    let cfg = config::load_config(model_path)?;

    // Build paths
    let weights_path = format!("{}/model_weights.bin", model_path);
    let manifest_path = format!("{}/model_weights.json", model_path);

    // Load weights
    let wf = weights::open_weights(&weights_path, &manifest_path)?;
    let ht = weights::OwnedTensorHashTable::new(&wf.manifest);

    // ------ Metal GPU init (required — no CPU fallback) ------
    let mut metal_ctx = metal::MetalCtx::new(&cfg)
        .map_err(|e| format!("Metal GPU unavailable: {}", e))?;

    // Wrap weight file in a Metal buffer (unified memory — no copy)
    metal_ctx.set_weights(wf.data, wf.size);

    // ------ Expert LRU cache (GPU only) ------
    let detect_2bit = {
        let probe = format!("{}/packed_experts_2bit/layer_00.bin", model_path);
        std::path::Path::new(&probe).exists()
    };
    let esz = if detect_2bit { cfg.expert_size_2bit as usize } else { cfg.expert_size_4bit as usize };
    let max_entries = 32; // small cache — mmap provides zero-syscall reads for misses

    // malloc_cache: posix_memalign + zero-copy Metal wrappers (always used)
    let malloc_cache = Some(expert_io::MallocExpertCache::new(
        cfg.num_layers, cfg.num_experts, max_entries, esz, &metal_ctx.device));
    let expert_cache = None; // only used as fallback when malloc_cache is None

    // ------ Build per-layer weight caches ------
    let layer_caches = unsafe {
        gpu_forward::build_layer_cache(
            &cfg,
            &ht,
            wf.data,
        )
    };

    // ------ Allocate per-token scratch ------
    let layer_scratch = cpu_forward::CpuForwardScratch::new(&cfg);

    // ------ Open expert files ------
    let expert_dir = if detect_2bit { "packed_experts_2bit" } else { "packed_experts" };

    let num_layers = cfg.num_layers as usize;
    let mut layer_fds: Vec<Option<File>> = Vec::with_capacity(num_layers);

    // mmap expert files for zero-syscall access (matches C's pattern)
    let mut layer_mmaps: Vec<*mut u8> = Vec::with_capacity(num_layers);
    let mut layer_mmap_sizes: Vec<usize> = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let path = format!("{}/{}/layer_{:02}.bin", model_path, expert_dir, i);
        let fd = File::open(&path);
        if let Ok(f) = fd {
            let raw_fd = f.as_raw_fd();
            // Disable read-ahead (random access pattern for experts)
            unsafe { libc::fcntl(raw_fd, libc::F_RDAHEAD, 0); }
            let sz = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
            if sz > 0 {
                let ptr = unsafe { expert_io::mmap(std::ptr::null_mut(), sz, expert_io::PROT_READ, expert_io::MAP_PRIVATE, raw_fd, 0) };
                if ptr != expert_io::MAP_FAILED {
                    layer_mmaps.push(ptr);
                    layer_mmap_sizes.push(sz);
                } else {
                    layer_mmaps.push(std::ptr::null_mut());
                    layer_mmap_sizes.push(0);
                }
            } else {
                layer_mmaps.push(std::ptr::null_mut());
                layer_mmap_sizes.push(0);
            }
            layer_fds.push(Some(f));
        } else {
            layer_fds.push(None);
            layer_mmaps.push(std::ptr::null_mut());
            layer_mmap_sizes.push(0);
        }
    }

    // ------ Allocate working buffers ------
    let hidden = vec![0.0f32; cfg.hidden_dim as usize];
    let logits = vec![0.0f32; cfg.vocab_size as usize];
    let final_norm_w = ht.find("model.norm.weight")
        .map(|t| {
            let ptr = unsafe { wf.data.add(t.offset as usize) as *const u16 };
            unsafe { std::slice::from_raw_parts(ptr, cfg.hidden_dim as usize) }.to_vec()
        })
        .unwrap_or_default();

    // Init persistent I/O thread pool
    let io_pool = Some(expert_io::IOPool::new());
    println!("[init] Persistent I/O pool started ({} threads)", expert_io::NUM_IO_THREADS);

    // ------ Init prediction state ------
    let nl = cfg.num_layers as usize;
    let pred_experts = vec![-1i32; nl * MAX_K];
    let pred_count = vec![0i32; nl];

    println!("[init] Temporal prediction enabled (prev-token routing → next-token prefetch)");
    println!("[init] Model loaded: {} layers ({} full + {} linear)",
        cfg.num_layers, cfg.num_full_attn_layers, cfg.num_linear_layers);
    println!("[init] Metal GPU context initialized");

    Ok(FlashMoEContext {
        cfg,
        model_path: model_path.to_string(),
        wf,
        ht_data: ht,
        layer_fds,
        layer_mmaps,
        layer_mmap_sizes,
        hidden,
        logits,
        final_norm_w,
        expert_cache,
        malloc_cache,
        io_pool,
        metal_ctx,
        layer_caches,
        layer_scratch,
        deferred: None,
        gpu_mode: gpu_forward::GpuMode::ThreeCommand,
        use_2bit: detect_2bit,
        initialized: true,
        pred_experts,
        pred_count,
        pred_valid: false,
        pred_generating: true,
        pred_hits: 0,
        pred_misses: 0,
    })
}

// ============================================================================
// PyO3 bindings — Python-accessible API
// ============================================================================

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;
    use numpy::{PyArray1, PyArray2, PyArrayMethods};
    use std::sync::Mutex;

    use super::*;

    /// Python-exposed model handle.
    #[pyclass]
    pub struct Model {
        ctx: Mutex<FlashMoEContext>,
    }

    #[pyclass]
    pub struct Cache {
        inner: Mutex<FlashMoECache>,
        cfg: ModelConfig, // copy needed for reset
    }

    /// Streaming token-by-token iterator returned by `Model.generate()`.
    /// Each `__next__` call runs one forward+sample step in Rust and returns
    /// immediately, so tokens are produced as they're sampled (true streaming).
    #[pyclass]
    pub struct GenerateIterator {
        model: Py<Model>,
        cache: Py<Cache>,
        state: Mutex<GenerateIterState>,
    }

    struct GenerateIterState {
        next_id: i32,
        remaining: i32,
        eos_token_id: i32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        logits_buf: Vec<f32>,
        finished: bool,
    }

    #[pymethods]
    impl GenerateIterator {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&self, py: Python<'_>) -> PyResult<Option<i32>> {
            // Snapshot params + take the reusable logits buffer under state lock
            let (mut next, eos_id, temp, t_k, t_p, m_p, mut logits_buf) = {
                let mut s = self.state.lock().unwrap();
                if s.finished || s.remaining <= 0 {
                    return Ok(None);
                }
                (s.next_id, s.eos_token_id, s.temperature, s.top_k, s.top_p, s.min_p,
                 std::mem::take(&mut s.logits_buf))
            };

            // Lock model + cache, run one forward + sample step
            let model_ref = self.model.bind(py).borrow();
            let cache_ref = self.cache.bind(py).borrow();
            let mut ctx = model_ref.ctx.lock().unwrap();
            let mut c = cache_ref.inner.lock().unwrap();

            generate::generate_step(
                &mut ctx, &mut c, &mut next, &mut logits_buf,
                eos_id, temp, t_k, t_p, m_p,
            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Write back state
            {
                let mut s = self.state.lock().unwrap();
                s.next_id = next;
                s.logits_buf = logits_buf;
                s.remaining -= 1;
                if next == eos_id {
                    s.finished = true;
                }
            }

            Ok(Some(next))
        }
    }

    #[pymethods]
    impl Cache {
        #[new]
        #[pyo3(signature = (model=None))]
        fn new(model: Option<&Model>) -> PyResult<Self> {
            if let Some(m) = model {
                let ctx = m.ctx.lock().unwrap();
                Ok(Self {
                    inner: Mutex::new(FlashMoECache::new(&ctx.cfg)),
                    cfg: ctx.cfg.clone(),
                })
            } else {
                Err(pyo3::exceptions::PyRuntimeError::new_err("Cache requires a Model"))
            }
        }

        #[pyo3(signature = (model=None))]
        fn reset(&self, model: Option<&Model>) -> PyResult<()> {
            let cfg = if let Some(m) = model {
                m.ctx.lock().unwrap().cfg.clone()
            } else {
                self.cfg.clone()
            };
            self.inner.lock().unwrap().reset(&cfg);
            Ok(())
        }

        #[getter]
        fn position(&self) -> PyResult<i32> {
            Ok(self.inner.lock().unwrap().pos)
        }
    }

    #[pymethods]
    impl Model {
        #[new]
        fn new(model_path: &str) -> PyResult<Self> {
            let ctx = flashmoe_init(model_path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Ok(Self { ctx: Mutex::new(ctx) })
        }

        fn forward(&self, py: Python<'_>, input_ids: Vec<i32>, cache: &Cache) -> PyResult<(Py<PyArray2<f32>>, Py<PyAny>)> {
            let mut ctx = self.ctx.lock().unwrap();
            let mut c = cache.inner.lock().unwrap();
            let vocab = ctx.cfg.vocab_size as usize;
            let n = input_ids.len();

            let mut logits = vec![0.0f32; n * vocab];
            flashmoe_forward(&mut ctx, &input_ids, n as i32, &mut logits, &mut c)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            let arr1d = PyArray1::from_vec(py, logits);
            let arr = arr1d.reshape([n, vocab])
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            Ok((arr.into(), py.None()))
        }

        /// Full autoregressive loop in Rust with true streaming.
        /// Returns a Python iterator that yields tokens one by one as they are
        /// sampled — each `__next__` call runs one forward + sample step in Rust.
        /// Much faster than calling `sample()` in a Python loop.
        #[pyo3(signature = (first_token_id, cache, max_tokens, eos_token_id,
                            temperature, top_k, top_p, min_p))]
        fn generate(
            self_: Bound<'_, Self>,
            first_token_id: i32,
            cache: Bound<'_, Cache>,
            max_tokens: i32,
            eos_token_id: i32,
            temperature: f32,
            top_k: i32,
            top_p: f32,
            min_p: f32,
        ) -> PyResult<Py<GenerateIterator>> {
            let py = self_.py();
            let guard = self_.borrow();
            let vocab = guard.ctx.lock().unwrap().cfg.vocab_size as usize;
            drop(guard);
            let iter = GenerateIterator {
                model: self_.clone().unbind(),
                cache: cache.unbind(),
                state: Mutex::new(GenerateIterState {
                    next_id: first_token_id,
                    remaining: max_tokens,
                    eos_token_id,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    logits_buf: vec![0.0f32; vocab],
                    finished: false,
                }),
            };
            Py::new(py, iter)
        }

        /// Single forward + sample step. Takes the current token_id, runs one
        /// forward pass, samples the next token, and returns it.  Call in a loop
        /// for true token-by-token streaming.
        fn sample(
            &self,
            token_id: i32,
            cache: &Cache,
            temperature: f32,
            top_k: i32,
            top_p: f32,
            min_p: f32,
        ) -> PyResult<i32> {
            let mut ctx = self.ctx.lock().unwrap();
            let mut c = cache.inner.lock().unwrap();
            let vocab = ctx.cfg.vocab_size as usize;

            let mut logits_buf = vec![0.0f32; vocab];
            let mut next_id = token_id;

            generate::generate_step(
                &mut ctx, &mut c, &mut next_id, &mut logits_buf,
                -1, // eos not checked here — caller decides when to stop
                temperature, top_k, top_p, min_p,
            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(next_id)
        }

        #[getter]
        fn num_layers(&self) -> PyResult<i32> {
            Ok(self.ctx.lock().unwrap().cfg.num_layers)
        }

        #[getter]
        fn hidden_dim(&self) -> PyResult<i32> {
            Ok(self.ctx.lock().unwrap().cfg.hidden_dim)
        }

        #[getter]
        fn vocab_size(&self) -> PyResult<i32> {
            Ok(self.ctx.lock().unwrap().cfg.vocab_size)
        }

        #[getter]
        fn has_gpu(&self) -> PyResult<bool> {
            Ok(true) // always true — we fail at init if no GPU
        }
    }

    /// Python module entry point
    #[pymodule]
    fn moe_infer_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Model>()?;
        m.add_class::<Cache>()?;
        m.add_class::<GenerateIterator>()?;
        Ok(())
    }
}
