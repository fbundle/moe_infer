use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::cache::Cache;
use crate::engine::qwen35_moe::constants::{FullModel, StrippedModel, ModelConfig};
use crate::error::MoEError;
use crate::engine::qwen35_moe::metal_context::{WeightBuffer, MetalContext, ExpertBuffer};
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
    Fused4bit,
    Fused4bitStripped,
    Fused4bitExp1,
    Fused4bitExp1Stripped,
    Fused4bitExp2,
    Fused4bitExp2Stripped,
    Fused4bitExp3,
    Fused4bitExp3Stripped,
}

impl EngineEnum {
    /// Initialize GPU resources (Metal context, weight buffer, expert buffer)
    /// for this engine type on the given model.
    pub fn init_gpu(
        self,
        model: &Model,
        k: usize,
    ) -> Result<(MetalContext, WeightBuffer, ExpertBuffer), MoEError> {
        let is_stripped = matches!(self, EngineEnum::Fused4bitStripped | EngineEnum::Fused4bitExp1Stripped | EngineEnum::Fused4bitExp2Stripped | EngineEnum::Fused4bitExp3Stripped);
        let (num_layers, num_experts, num_experts_per_tok, num_linear_layers, linear_conv_dim,
             linear_num_v_heads, linear_total_value, linear_key_dim, linear_value_dim,
             hidden_dim, shared_intermediate, moe_intermediate, expert_size_4bit,
             num_full_attn_layers, kv_dim, num_attn_heads, head_dim) =
            if is_stripped {
                (StrippedModel::NUM_LAYERS, StrippedModel::NUM_EXPERTS, StrippedModel::NUM_EXPERTS_PER_TOK,
                 StrippedModel::NUM_LINEAR_LAYERS, StrippedModel::LINEAR_CONV_DIM,
                 StrippedModel::LINEAR_NUM_V_HEADS, StrippedModel::LINEAR_TOTAL_VALUE,
                 StrippedModel::LINEAR_KEY_DIM, StrippedModel::LINEAR_VALUE_DIM,
                 StrippedModel::HIDDEN_DIM, StrippedModel::SHARED_INTERMEDIATE,
                 StrippedModel::MOE_INTERMEDIATE, StrippedModel::EXPERT_SIZE_4BIT,
                 StrippedModel::NUM_FULL_ATTN_LAYERS,
                 StrippedModel::NUM_KV_HEADS * StrippedModel::HEAD_DIM,
                 StrippedModel::NUM_ATTN_HEADS, StrippedModel::HEAD_DIM)
            } else {
                (FullModel::NUM_LAYERS, FullModel::NUM_EXPERTS, FullModel::NUM_EXPERTS_PER_TOK,
                 FullModel::NUM_LINEAR_LAYERS, FullModel::LINEAR_CONV_DIM,
                 FullModel::LINEAR_NUM_V_HEADS, FullModel::LINEAR_TOTAL_VALUE,
                 FullModel::LINEAR_KEY_DIM, FullModel::LINEAR_VALUE_DIM,
                 FullModel::HIDDEN_DIM, FullModel::SHARED_INTERMEDIATE,
                 FullModel::MOE_INTERMEDIATE, FullModel::EXPERT_SIZE_4BIT,
                 FullModel::NUM_FULL_ATTN_LAYERS,
                 FullModel::NUM_KV_HEADS * FullModel::HEAD_DIM,
                 FullModel::NUM_ATTN_HEADS, FullModel::HEAD_DIM)
            };

        let k = if k == 0 { num_experts_per_tok } else { k };
        if k > num_experts_per_tok {
            return Err(MoEError::Config(format!(
                "k ({}) must not exceed model's num_experts_per_tok ({})", k, num_experts_per_tok
            )));
        }

        let mut ctx = MetalContext::init()?;
        ctx.init_linear_attn_buffers(
            num_linear_layers, linear_conv_dim, linear_num_v_heads,
            linear_total_value, linear_key_dim, linear_value_dim,
            hidden_dim, num_experts, shared_intermediate,
            num_full_attn_layers, kv_dim,
            num_attn_heads, head_dim,
            num_attn_heads * 2 * head_dim,
        );
        let expert_buffer = ctx.init_expert_buffers(
            expert_size_4bit, hidden_dim, moe_intermediate, shared_intermediate,
        );
        let weight_buffer = WeightBuffer::new(&ctx.device, &model.weight_file);

        eprintln!(
            "[engine] {} layers hidden={} experts={} mode={:?}",
            num_layers, hidden_dim, num_experts, self
        );

        Ok((ctx, weight_buffer, expert_buffer))
    }
}

// ─── Type-erased engine ─────────────────────────────────────────────────────

use qwen35_moe::{Fused4bit, Fused4bitExp1, Fused4bitExp2, Fused4bitExp3};

/// Type-erased engine holding one of the engine variants.
pub enum DynEngine {
    Fused4bit(Fused4bit<'static, FullModel>),
    Fused4bitStripped(Fused4bit<'static, StrippedModel>),
    Fused4bitExp1(Fused4bitExp1<'static, FullModel>),
    Fused4bitExp1Stripped(Fused4bitExp1<'static, StrippedModel>),
    Fused4bitExp2(Fused4bitExp2<'static, FullModel>),
    Fused4bitExp2Stripped(Fused4bitExp2<'static, StrippedModel>),
    Fused4bitExp3(Fused4bitExp3<'static, FullModel>),
    Fused4bitExp3Stripped(Fused4bitExp3<'static, StrippedModel>),
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
        ctx: &MetalContext,
        weight_buffer: &WeightBuffer,
        expert_buffer: Option<&mut ExpertBuffer>,
        k: usize,
        engine_type: EngineEnum,
    ) -> Result<Self, MoEError> {
        let model_ref: &Model = &*(model as *const Model);
        let ctx_ref: &MetalContext = &*(ctx as *const MetalContext);
        let weight_buffer_ref: &WeightBuffer = &*(weight_buffer as *const WeightBuffer);

        Ok(match engine_type {
            EngineEnum::Fused4bit => {
                let e = Fused4bit::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Fused4bit(e)
            }
            EngineEnum::Fused4bitStripped => {
                let e = Fused4bit::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Fused4bitStripped(e)
            }
            EngineEnum::Fused4bitExp1 => {
                let e = Fused4bitExp1::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Fused4bitExp1(e)
            }
            EngineEnum::Fused4bitExp1Stripped => {
                let e = Fused4bitExp1::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Fused4bitExp1Stripped(e)
            }
            EngineEnum::Fused4bitExp2 => {
                let e = Fused4bitExp2::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Fused4bitExp2(e)
            }
            EngineEnum::Fused4bitExp2Stripped => {
                let e = Fused4bitExp2::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Fused4bitExp2Stripped(e)
            }
            EngineEnum::Fused4bitExp3 => {
                let e = Fused4bitExp3::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Fused4bitExp3(e)
            }
            EngineEnum::Fused4bitExp3Stripped => {
                let e = Fused4bitExp3::new(
                    model_ref, ctx_ref, weight_buffer_ref,
                    expert_buffer.map(|b| &mut *(b as *mut ExpertBuffer)),
                    k,
                )?;
                DynEngine::Fused4bitExp3Stripped(e)
            }
        })
    }

    pub fn upload_cache(&self, cache: &Cache) {
        match self {
            DynEngine::Fused4bit(e) => Engine::upload_cache(e, cache),
            DynEngine::Fused4bitStripped(e) => Engine::upload_cache(e, cache),
            DynEngine::Fused4bitExp1(e) => Engine::upload_cache(e, cache),
            DynEngine::Fused4bitExp1Stripped(e) => Engine::upload_cache(e, cache),
            DynEngine::Fused4bitExp2(e) => Engine::upload_cache(e, cache),
            DynEngine::Fused4bitExp2Stripped(e) => Engine::upload_cache(e, cache),
            DynEngine::Fused4bitExp3(e) => Engine::upload_cache(e, cache),
            DynEngine::Fused4bitExp3Stripped(e) => Engine::upload_cache(e, cache),
        }
    }

    pub fn download_cache(&self, cache: &mut Cache) {
        match self {
            DynEngine::Fused4bit(e) => Engine::download_cache(e, cache),
            DynEngine::Fused4bitStripped(e) => Engine::download_cache(e, cache),
            DynEngine::Fused4bitExp1(e) => Engine::download_cache(e, cache),
            DynEngine::Fused4bitExp1Stripped(e) => Engine::download_cache(e, cache),
            DynEngine::Fused4bitExp2(e) => Engine::download_cache(e, cache),
            DynEngine::Fused4bitExp2Stripped(e) => Engine::download_cache(e, cache),
            DynEngine::Fused4bitExp3(e) => Engine::download_cache(e, cache),
            DynEngine::Fused4bitExp3Stripped(e) => Engine::download_cache(e, cache),
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &[i64],
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        match self {
            DynEngine::Fused4bit(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Fused4bitStripped(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Fused4bitExp1(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Fused4bitExp1Stripped(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Fused4bitExp2(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Fused4bitExp2Stripped(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Fused4bitExp3(e) => Engine::forward(e, input_ids, check_signal),
            DynEngine::Fused4bitExp3Stripped(e) => Engine::forward(e, input_ids, check_signal),
        }
    }

    pub fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        match self {
            DynEngine::Fused4bit(e) => Engine::telemetry(e),
            DynEngine::Fused4bitStripped(e) => Engine::telemetry(e),
            DynEngine::Fused4bitExp1(e) => Engine::telemetry(e),
            DynEngine::Fused4bitExp1Stripped(e) => Engine::telemetry(e),
            DynEngine::Fused4bitExp2(e) => Engine::telemetry(e),
            DynEngine::Fused4bitExp2Stripped(e) => Engine::telemetry(e),
            DynEngine::Fused4bitExp3(e) => Engine::telemetry(e),
            DynEngine::Fused4bitExp3Stripped(e) => Engine::telemetry(e),
        }
    }
}
