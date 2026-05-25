use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::cache::Cache;
use crate::engine::qwen35_moe::{Qwen35MoEFullModel, Qwen35MoEStrippedModel};
use crate::error::MoEError;
use crate::engine::qwen35_moe::metal_context::{WeightBuffer, ExpertBuffer};
use crate::engine::qwen35_moe::Qwen35MoEMetalContext;
use crate::model::Model;

#[path = "engine/qwen35_moe.rs"]
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

// ─── Engine type ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum EngineEnum {
    Qwen35MoEFused4bit,
    Qwen35MoEFused4bitStripped,
    Qwen35MoEFused4bitExp1,
    Qwen35MoEFused4bitExp1Stripped,
    Qwen35MoEFused4bitExp2,
    Qwen35MoEFused4bitExp2Stripped,
    Qwen35MoEFused4bitExp3,
    Qwen35MoEFused4bitExp3Stripped,
}

impl EngineEnum {
    /// Resolve (engine_name, architecture) and init GPU in one step.
    pub fn resolve_and_init(
        engine_type: &str,
        model: &Model,
        k: usize,
    ) -> Result<(Self, Qwen35MoEMetalContext, WeightBuffer, ExpertBuffer), MoEError> {
        let arch = model.config.resolve("architectures")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        match (engine_type, arch) {
            ("Qwen35MoEFused4bit", "Qwen3_5MoeForConditionalGeneration") => {
                let (ctx, wb, eb) = Qwen35MoEMetalContext::init_gpu::<Qwen35MoEFullModel>(&model.weight_file, k, engine_type)?;
                Ok((EngineEnum::Qwen35MoEFused4bit, ctx, wb, eb))
            }
            ("Qwen35MoEFused4bit", "Qwen3_5MoeForConditionalGeneration_Stripped") => {
                let (ctx, wb, eb) = Qwen35MoEMetalContext::init_gpu::<Qwen35MoEStrippedModel>(&model.weight_file, k, engine_type)?;
                Ok((EngineEnum::Qwen35MoEFused4bitStripped, ctx, wb, eb))
            }
            ("Qwen35MoEFused4bitExp1", "Qwen3_5MoeForConditionalGeneration") => {
                let (ctx, wb, eb) = Qwen35MoEMetalContext::init_gpu::<Qwen35MoEFullModel>(&model.weight_file, k, engine_type)?;
                Ok((EngineEnum::Qwen35MoEFused4bitExp1, ctx, wb, eb))
            }
            ("Qwen35MoEFused4bitExp1", "Qwen3_5MoeForConditionalGeneration_Stripped") => {
                let (ctx, wb, eb) = Qwen35MoEMetalContext::init_gpu::<Qwen35MoEStrippedModel>(&model.weight_file, k, engine_type)?;
                Ok((EngineEnum::Qwen35MoEFused4bitExp1Stripped, ctx, wb, eb))
            }
            ("Qwen35MoEFused4bitExp2", "Qwen3_5MoeForConditionalGeneration") => {
                let (ctx, wb, eb) = Qwen35MoEMetalContext::init_gpu::<Qwen35MoEFullModel>(&model.weight_file, k, engine_type)?;
                Ok((EngineEnum::Qwen35MoEFused4bitExp2, ctx, wb, eb))
            }
            ("Qwen35MoEFused4bitExp2", "Qwen3_5MoeForConditionalGeneration_Stripped") => {
                let (ctx, wb, eb) = Qwen35MoEMetalContext::init_gpu::<Qwen35MoEStrippedModel>(&model.weight_file, k, engine_type)?;
                Ok((EngineEnum::Qwen35MoEFused4bitExp2Stripped, ctx, wb, eb))
            }
            ("Qwen35MoEFused4bitExp3", "Qwen3_5MoeForConditionalGeneration") => {
                let (ctx, wb, eb) = Qwen35MoEMetalContext::init_gpu::<Qwen35MoEFullModel>(&model.weight_file, k, engine_type)?;
                Ok((EngineEnum::Qwen35MoEFused4bitExp3, ctx, wb, eb))
            }
            ("Qwen35MoEFused4bitExp3", "Qwen3_5MoeForConditionalGeneration_Stripped") => {
                let (ctx, wb, eb) = Qwen35MoEMetalContext::init_gpu::<Qwen35MoEStrippedModel>(&model.weight_file, k, engine_type)?;
                Ok((EngineEnum::Qwen35MoEFused4bitExp3Stripped, ctx, wb, eb))
            }
            _ => Err(MoEError::Config(format!(
                "Unknown engine: engine_type={:?}, arch={:?}", engine_type, arch
            ))),
        }
    }
}

// ─── Type-erased engine ─────────────────────────────────────────────────────

use qwen35_moe::{Qwen35MoEFused4bit, Qwen35MoEFused4bitExp1, Qwen35MoEFused4bitExp2, Qwen35MoEFused4bitExp3};

/// Type-erased engine holding one of the engine variants.
pub enum DynEngine {
    Qwen35MoEFused4bit(Qwen35MoEFused4bit<'static, Qwen35MoEFullModel>),
    Qwen35MoEFused4bitStripped(Qwen35MoEFused4bit<'static, Qwen35MoEStrippedModel>),
    Qwen35MoEFused4bitExp1(Qwen35MoEFused4bitExp1<'static, Qwen35MoEFullModel>),
    Qwen35MoEFused4bitExp1Stripped(Qwen35MoEFused4bitExp1<'static, Qwen35MoEStrippedModel>),
    Qwen35MoEFused4bitExp2(Qwen35MoEFused4bitExp2<'static, Qwen35MoEFullModel>),
    Qwen35MoEFused4bitExp2Stripped(Qwen35MoEFused4bitExp2<'static, Qwen35MoEStrippedModel>),
    Qwen35MoEFused4bitExp3(Qwen35MoEFused4bitExp3<'static, Qwen35MoEFullModel>),
    Qwen35MoEFused4bitExp3Stripped(Qwen35MoEFused4bitExp3<'static, Qwen35MoEStrippedModel>),
}

impl DynEngine {
    /// Create an erased engine.
    ///
    /// # Safety
    /// `model`, `ctx`, `weight_buffer`, and `expert_buffer` must remain live and
    /// unmoved for as long as the returned `DynEngine` exists. In the PyO3
    /// binding, the `Engine` pyclass is heap-allocated and never moved, so
    /// references to its fields are stable.
    pub unsafe fn new(
        model: &Model,
        ctx: &Qwen35MoEMetalContext,
        weight_buffer: &WeightBuffer,
        expert_buffer: Option<&mut ExpertBuffer>,
        k: usize,
        engine_type: EngineEnum,
    ) -> Result<Self, MoEError> {
        let model_ref: &Model = &*(model as *const Model);
        let ctx_ref: &Qwen35MoEMetalContext = &*(ctx as *const Qwen35MoEMetalContext);
        let weight_buffer_ref: &WeightBuffer = &*(weight_buffer as *const WeightBuffer);

        Ok(match engine_type {
            EngineEnum::Qwen35MoEFused4bit => {
                let e = Qwen35MoEFused4bit::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Qwen35MoEFused4bit(e)
            }
            EngineEnum::Qwen35MoEFused4bitStripped => {
                let e = Qwen35MoEFused4bit::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Qwen35MoEFused4bitStripped(e)
            }
            EngineEnum::Qwen35MoEFused4bitExp1 => {
                let e = Qwen35MoEFused4bitExp1::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Qwen35MoEFused4bitExp1(e)
            }
            EngineEnum::Qwen35MoEFused4bitExp1Stripped => {
                let e = Qwen35MoEFused4bitExp1::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Qwen35MoEFused4bitExp1Stripped(e)
            }
            EngineEnum::Qwen35MoEFused4bitExp2 => {
                let e = Qwen35MoEFused4bitExp2::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Qwen35MoEFused4bitExp2(e)
            }
            EngineEnum::Qwen35MoEFused4bitExp2Stripped => {
                let e = Qwen35MoEFused4bitExp2::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Qwen35MoEFused4bitExp2Stripped(e)
            }
            EngineEnum::Qwen35MoEFused4bitExp3 => {
                let e = Qwen35MoEFused4bitExp3::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Qwen35MoEFused4bitExp3(e)
            }
            EngineEnum::Qwen35MoEFused4bitExp3Stripped => {
                let e = Qwen35MoEFused4bitExp3::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Qwen35MoEFused4bitExp3Stripped(e)
            }
        })
    }

    pub fn upload_cache(&self, cache: &Cache) {
        match self {
            DynEngine::Qwen35MoEFused4bit(e) => Engine::upload_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitStripped(e) => Engine::upload_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp1(e) => Engine::upload_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp1Stripped(e) => Engine::upload_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp2(e) => Engine::upload_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp2Stripped(e) => Engine::upload_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp3(e) => Engine::upload_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp3Stripped(e) => Engine::upload_cache(e, cache),
        }
    }

    pub fn download_cache(&self, cache: &mut Cache) {
        match self {
            DynEngine::Qwen35MoEFused4bit(e) => Engine::download_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitStripped(e) => Engine::download_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp1(e) => Engine::download_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp1Stripped(e) => Engine::download_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp2(e) => Engine::download_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp2Stripped(e) => Engine::download_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp3(e) => Engine::download_cache(e, cache),
            DynEngine::Qwen35MoEFused4bitExp3Stripped(e) => Engine::download_cache(e, cache),
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &[i64],
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        match self {
            DynEngine::Qwen35MoEFused4bit(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Qwen35MoEFused4bitStripped(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Qwen35MoEFused4bitExp1(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Qwen35MoEFused4bitExp1Stripped(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Qwen35MoEFused4bitExp2(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Qwen35MoEFused4bitExp2Stripped(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Qwen35MoEFused4bitExp3(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Qwen35MoEFused4bitExp3Stripped(e) => Engine::forward(e, input_ids, check_signal),
        }
    }

    pub fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        match self {
            DynEngine::Qwen35MoEFused4bit(e) => Engine::telemetry(e),
            DynEngine::Qwen35MoEFused4bitStripped(e) => Engine::telemetry(e),
            DynEngine::Qwen35MoEFused4bitExp1(e) => Engine::telemetry(e),
            DynEngine::Qwen35MoEFused4bitExp1Stripped(e) => Engine::telemetry(e),
            DynEngine::Qwen35MoEFused4bitExp2(e) => Engine::telemetry(e),
            DynEngine::Qwen35MoEFused4bitExp2Stripped(e) => Engine::telemetry(e),
            DynEngine::Qwen35MoEFused4bitExp3(e) => Engine::telemetry(e),
            DynEngine::Qwen35MoEFused4bitExp3Stripped(e) => Engine::telemetry(e),
        }
    }
}
