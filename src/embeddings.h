#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

// ============================================================================
// Embedding lookup (4-bit quantized)
// ============================================================================

static void embed_lookup(WeightFile *wf, int token_id, float *out) {
    // Embedding: weight[vocab_size, hidden_dim/8] (U32), scales[vocab_size, groups], biases[vocab_size, groups]
    // For embedding lookup, we just need one row.
    // But the embedding is quantized: each row has hidden_dim/8 uint32 values (packed 4-bit)
    // plus scales and biases per group

    TensorInfo *w_info = get_tensor_info(wf, "model.embed_tokens.weight");
    TensorInfo *s_info = get_tensor_info(wf, "model.embed_tokens.scales");
    TensorInfo *b_info = get_tensor_info(wf, "model.embed_tokens.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: embedding tensors not found\n");
        memset(out, 0, HIDDEN_DIM * sizeof(float));
        return;
    }

    // w shape: [248320, 512] U32 -> each row has 512 uint32 = 4096 packed 4-bit values
    int packed_cols = w_info->shape[1];  // 512
    int num_groups = s_info->shape[1];   // 64

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    const uint32_t *w_row = W + (size_t)token_id * packed_cols;
    const uint16_t *s_row = S + (size_t)token_id * num_groups;
    const uint16_t *b_row = B + (size_t)token_id * num_groups;

    int group_size = HIDDEN_DIM / num_groups;  // 4096/64 = 64
    int packed_per_group = group_size / 8;     // 8

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias = bf16_to_f32(b_row[g]);

        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[g * packed_per_group + p];
            int base = g * group_size + p * 8;

            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                out[base + n] = (float)nibble * scale + bias;
            }
        }
    }
}

// ============================================================================
// LM head (logits projection)
// ============================================================================

static void lm_head_forward(WeightFile *wf, const float *hidden, float *logits) {
    // lm_head: [hidden_dim=4096] -> [vocab_size=248320]
    // This is a HUGE matmul. For 248320 output dims, it will be slow on CPU.
    // Optimization: only compute top candidates

    TensorInfo *w_info = get_tensor_info(wf, "lm_head.weight");
    TensorInfo *s_info = get_tensor_info(wf, "lm_head.scales");
    TensorInfo *b_info = get_tensor_info(wf, "lm_head.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: lm_head tensors not found\n");
        return;
    }

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    // Full matmul — use GPU if available (248320 output rows!)
    fast_dequant_matvec(W, S, B, hidden, logits, VOCAB_SIZE, HIDDEN_DIM, GROUP_SIZE);
}


#endif // EMBEDDINGS_H
