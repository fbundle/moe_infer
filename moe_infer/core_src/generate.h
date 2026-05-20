#ifndef GENERATE_H
#define GENERATE_H

// ============================================================================
// Autoregressive token generation with sampling — single-step and batched.
// Included into moe_infer_c.m after moe_infer_c.h.
// ============================================================================

#include "common.h"
#include "moe_infer_c.h"

typedef struct {
    float prob;
    int idx;
} _ProbIdx;

static int _cmp_prob_desc(const void *a, const void *b) {
    float pa = ((const _ProbIdx *)a)->prob;
    float pb = ((const _ProbIdx *)b)->prob;
    return (pa < pb) - (pa > pb);
}

// Sample from logits (1 x vocab_size). Writes the chosen token into *next_id.
// logits buffer is reused as scratch; its contents are overwritten.
static void _sample_token(FlashMoE_Context *m, float *logits,
                          int *next_id, float temperature,
                          int top_k, float top_p, float min_p)
{
    int V = m->cfg.vocab_size;

    if (temperature <= 0.0f) {
        // Greedy
        float best = logits[0];
        int best_i = 0;
        for (int i = 1; i < V; i++) {
            if (logits[i] > best) { best = logits[i]; best_i = i; }
        }
        *next_id = best_i;
        return;
    }

    // Temperature scaling + softmax (numerically stable)
    float max_l = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > max_l) max_l = logits[i];

    float inv_t = 1.0f / temperature;
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf((logits[i] - max_l) * inv_t);
        sum += logits[i];
    }

    if (sum <= 0.0f) {
        float best = logits[0];
        int best_i = 0;
        for (int i = 1; i < V; i++) {
            if (logits[i] > best) { best = logits[i]; best_i = i; }
        }
        *next_id = best_i;
        return;
    }

    // Build sorted index on stack — use a VLA; V is ~150K so ~1.2 MB on stack.
    // That's borderline. Use heap instead for safety.
    _ProbIdx *sorted = malloc(V * sizeof(_ProbIdx));
    if (!sorted) {
        *next_id = 0;
        return;
    }

    float norm = 1.0f / sum;
    for (int i = 0; i < V; i++) {
        sorted[i].prob = logits[i] * norm;
        sorted[i].idx = i;
    }

    // Sort descending by probability
    qsort(sorted, V, sizeof(_ProbIdx), _cmp_prob_desc);

    // Apply top-k
    int cutoff = V;
    if (top_k > 0 && top_k < cutoff) cutoff = top_k;

    // Apply top-p (nucleus)
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            cum += sorted[i].prob;
            if (cum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    // Apply min-p
    if (min_p > 0.0f) {
        float max_p = sorted[0].prob;
        float thresh = min_p * max_p;
        for (int i = 0; i < cutoff; i++) {
            if (sorted[i].prob < thresh) {
                cutoff = i;
                break;
            }
        }
    }

    if (cutoff < 1) cutoff = 1;

    // Renormalize and sample
    float cum2 = 0.0f;
    for (int i = 0; i < cutoff; i++) cum2 += sorted[i].prob;
    float inv_cum = cum2 > 0.0f ? 1.0f / cum2 : 1.0f;

    float r = (float)arc4random() / (float)UINT32_MAX;
    float acc = 0.0f;
    *next_id = sorted[0].idx;
    for (int i = 0; i < cutoff; i++) {
        acc += sorted[i].prob * inv_cum;
        if (r < acc) {
            *next_id = sorted[i].idx;
            break;
        }
    }

    free(sorted);
}

// One step: feed *next_id to model, sample, write result back into *next_id.
// Returns 0 on success, -1 on error.
static int generate_step(FlashMoE_Context *m, FlashMoE_Cache *cache,
                         int *next_id, float *logits,
                         int eos_token_id, float temperature,
                         int top_k, float top_p, float min_p)
{
    int prev_id = *next_id;
    if (flashmoe_forward(m, &prev_id, 1, logits, cache) != 0)
        return -1;
    _sample_token(m, logits, next_id, temperature, top_k, top_p, min_p);
    return 0;
}

#endif // GENERATE_H
