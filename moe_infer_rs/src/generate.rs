// Autoregressive token generation with sampling -- single-step.
// Port of moe_infer_mlx/core_src/generate.h

use rand::Rng;
use std::cmp::Ordering;

use crate::{FlashMoEContext, FlashMoECache};

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

/// Sample a single token from the logits vector.
///
/// `logits` is used as scratch space and its contents are overwritten.
/// - temperature <= 0: greedy argmax
/// - temperature > 0: temperature scale -> softmax -> sort descending ->
///   top-k -> top-p -> min-p -> renormalize -> random sample
fn sample_token(
    logits: &mut [f32],
    vocab_size: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
) -> i32 {
    // --- Greedy ----------------------------------------------------------
    if temperature <= 0.0 {
        let mut best = logits[0];
        let mut best_i = 0_i32;
        for i in 1..vocab_size {
            if logits[i] > best {
                best = logits[i];
                best_i = i as i32;
            }
        }
        return best_i;
    }

    // --- Temperature scaling + numerically stable softmax ----------------
    let max_l = logits[..vocab_size]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let inv_t = 1.0 / temperature;
    let mut sum = 0.0_f32;
    for i in 0..vocab_size {
        logits[i] = f32::exp((logits[i] - max_l) * inv_t);
        sum += logits[i];
    }

    if sum <= 0.0 {
        // Degenerate distribution -- fall back to argmax on the exponentiated values.
        let mut best = logits[0];
        let mut best_i = 0_i32;
        for i in 1..vocab_size {
            if logits[i] > best {
                best = logits[i];
                best_i = i as i32;
            }
        }
        return best_i;
    }

    // --- Normalise -------------------------------------------------------
    let norm = 1.0 / sum;
    for i in 0..vocab_size {
        logits[i] *= norm;
    }

    // --- Build sorted (prob, idx) pairs, descending by prob -------------
    let mut sorted: Vec<(f32, i32)> = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        sorted.push((logits[i], i as i32));
    }
    sorted.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
    });

    // --- Top-k -----------------------------------------------------------
    let mut cutoff = vocab_size;
    if top_k > 0 && (top_k as usize) < cutoff {
        cutoff = top_k as usize;
    }

    // --- Top-p (nucleus) -------------------------------------------------
    if top_p > 0.0 && top_p < 1.0 {
        let mut cum = 0.0_f32;
        for i in 0..cutoff {
            cum += sorted[i].0;
            if cum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
    }

    // --- Min-p -----------------------------------------------------------
    if min_p > 0.0 {
        let max_p = sorted[0].0;
        let thresh = min_p * max_p;
        for i in 0..cutoff {
            if sorted[i].0 < thresh {
                cutoff = i;
                break;
            }
        }
    }

    if cutoff < 1 {
        cutoff = 1;
    }

    // --- Renormalize and sample -----------------------------------------
    let cum2: f32 = sorted[..cutoff].iter().map(|p| p.0).sum();
    let inv_cum = if cum2 > 0.0 { 1.0 / cum2 } else { 1.0 };

    let r: f32 = rand::thread_rng().gen(); // uniform in [0, 1)
    let mut acc = 0.0_f32;
    let mut token = sorted[0].1;
    for i in 0..cutoff {
        acc += sorted[i].0 * inv_cum;
        if r < acc {
            token = sorted[i].1;
            break;
        }
    }

    token
}

// ---------------------------------------------------------------------------
// Single-step generation
// ---------------------------------------------------------------------------

/// One autoregressive step: feed `*next_id` to the model, sample the next
/// token, and write the result back into `*next_id`.
///
/// `logits_buf` must be pre-allocated with at least `vocab_size` floats; its
/// contents are overwritten (used as scratch).
///
/// Returns `Ok(())` on success, `Err(msg)` on forward-pass failure.
pub fn generate_step(
    model: &mut FlashMoEContext,
    cache: &mut FlashMoECache,
    next_id: &mut i32,
    logits_buf: &mut [f32],
    _eos_token_id: i32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
) -> Result<(), String> {
    let prev_id = *next_id;
    let vocab_size = model.cfg.vocab_size as usize;

    crate::flashmoe_forward(model, &[prev_id], 1, logits_buf, cache)
        .map_err(|e| format!("forward pass failed: {}", e))?;

    let token = sample_token(
        logits_buf,
        vocab_size,
        temperature,
        top_k,
        top_p,
        min_p,
    );
    *next_id = token;

    Ok(())
}
