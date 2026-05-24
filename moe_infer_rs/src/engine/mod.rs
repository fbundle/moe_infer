use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::cache::Cache;
use crate::error::MoEError;
use crate::engine::qwen35_moe::metal_context::{WeightBuffer, MetalContext, ExpertBuffer};
use crate::model::Model;

pub mod qwen35_moe;

/// Signal check callback: returns true if processing should abort (e.g. Ctrl-C).
pub type SignalCheckFn<'a> = &'a mut dyn FnMut() -> bool;

/// Global toggle for engine-level telemetry recording.
static RECORD_TELEMETRY: AtomicBool = AtomicBool::new(false);

/// Enable or disable engine-level telemetry globally.
pub fn set_record_telemetry(on: bool) {
    RECORD_TELEMETRY.store(on, Ordering::Relaxed);
}

/// Check whether engine-level telemetry is enabled.
pub fn record_telemetry() -> bool {
    RECORD_TELEMETRY.load(Ordering::Relaxed)
}

/// A telemetry value: either a scalar or a list of per-invocation measurements.
#[derive(Clone)]
pub enum TelemetryValue {
    Scalar(f64),
    List(Vec<f64>),
}

pub trait Engine {
    /// Upload CPU cache → GPU buffers before forward. No-op if pos == 0.
    fn upload_cache(&self, cache: &Cache);
    /// Download GPU buffers → CPU cache after forward.
    fn download_cache(&self, cache: &mut Cache);

    /// Process `input_ids` through all layers. Returns logits [n, vocab_size].
    fn forward(
        &mut self,
        input_ids: &[i64],
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError>;

    /// Per-engine telemetry. Keys are like `engine.*`.
    /// Values can be scalars or per-invocation lists.
    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        BTreeMap::new()
    }
}

// ─── Pipeline mode ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub enum PipelineMode {
    FusedExp,
    FusedExpStripped,
}

// ─── Type-erased engine ─────────────────────────────────────────────────────

use qwen35_moe::{FullModel, StrippedModel, FusedExp};

/// Type-erased engine holding one of the engine variants.
pub enum ErasedEngine {
    FusedExp(FusedExp<'static, FullModel>),
    FusedExpStripped(FusedExp<'static, StrippedModel>),
}

impl ErasedEngine {
    /// Create an erased engine.
    ///
    /// # Safety
    /// `model`, `ctx`, `gpu_wf`, and `expert_gpu_buffer` must remain live and
    /// unmoved for as long as the returned `ErasedEngine` exists. In the PyO3
    /// binding, the `Engine` pyclass is heap-allocated and never moved, so
    /// references to its fields are stable.
    pub unsafe fn new(
        model: &Model,
        ctx: &MetalContext,
        gpu_wf: &WeightBuffer,
        expert_gpu_buffer: Option<&mut ExpertBuffer>,
        k: usize,
        mode: PipelineMode,
    ) -> Result<Self, MoEError> {
        let model_ref: &Model = &*(model as *const Model);
        let ctx_ref: &MetalContext = &*(ctx as *const MetalContext);
        let gpu_wf_ref: &WeightBuffer = &*(gpu_wf as *const WeightBuffer);

        Ok(match mode {
            PipelineMode::FusedExp => {
                let e = FusedExp::new(
                    model_ref, ctx_ref, gpu_wf_ref,
                    expert_gpu_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                ErasedEngine::FusedExp(e)
            }
            PipelineMode::FusedExpStripped => {
                let e = FusedExp::new(
                    model_ref, ctx_ref, gpu_wf_ref,
                    expert_gpu_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                ErasedEngine::FusedExpStripped(e)
            }
        })
    }

    pub fn upload_cache(&self, cache: &Cache) {
        match self {
            ErasedEngine::FusedExp(e) => Engine::upload_cache(e, cache),
            ErasedEngine::FusedExpStripped(e) => Engine::upload_cache(e, cache),
        }
    }

    pub fn download_cache(&self, cache: &mut Cache) {
        match self {
            ErasedEngine::FusedExp(e) => Engine::download_cache(e, cache),
            ErasedEngine::FusedExpStripped(e) => Engine::download_cache(e, cache),
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &[i64],
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        match self {
            ErasedEngine::FusedExp(e) => Engine::forward(e, input_ids, check_signal),
            ErasedEngine::FusedExpStripped(e) => Engine::forward(e, input_ids, check_signal),
        }
    }

    pub fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        match self {
            ErasedEngine::FusedExp(e) => Engine::telemetry(e),
            ErasedEngine::FusedExpStripped(e) => Engine::telemetry(e),
        }
    }
}
