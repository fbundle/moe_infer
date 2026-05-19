/*
 * moe_infer_c.m — C wrapper API for Flash-MoE inference engine.
 */

#include "util.h"
#include "model_weights.h"
#include "cpu_kernels.h"
#include "metal_setup.h"
#include "gpu_ops.h"
#include "attention.h"
#include "embeddings.h"
#include "expert_io.h"
#include "layer_forward.h"
#include "moe_infer_c.h"

// ---- Global runtime config ----
ModelConfig g_cfg = {0};

// ---- Cache ----

typedef struct FlashMoE_Cache {
    KVCache **kv_caches;          // [g_cfg.num_layers] — non-NULL for full-attn layers
    void    **layer_states;       // [g_cfg.num_layers] — LinearAttnState* for linear layers
    int       pos;                // sequence position for RoPE
} FlashMoE_Cache;

// ---- File-scope state (was local to main.m) ----

static WeightFile *g_wf = NULL;

static int   *g_layer_fds = NULL;
static void **g_layer_mmaps = NULL;
static size_t *g_layer_mmap_sizes = NULL;

static float    *g_hidden = NULL;
static float    *g_logits = NULL;
static uint16_t *g_final_norm_w = NULL;

static int g_K = 4;
static int g_initialized = 0;

// ---- flashmoe_cache_new ----

FlashMoE_Cache *flashmoe_cache_new(void) {
    FlashMoE_Cache *c = calloc(1, sizeof(FlashMoE_Cache));
    if (!c) return NULL;
    c->kv_caches    = calloc(g_cfg.num_layers, sizeof(KVCache *));
    c->layer_states = calloc(g_cfg.num_layers, sizeof(void *));
    c->pos = 0;
    for (int i = 0; i < g_cfg.num_layers; i++) {
        int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
        if (is_full) {
            c->kv_caches[i] = kv_cache_new();
        } else {
            c->layer_states[i] = linear_attn_state_new();
        }
    }
    return c;
}

FlashMoE_Cache *flashmoe_cache_clone(FlashMoE_Cache *src) {
    // Only needed if the caller wants snapshot semantics.
    // For now we don't copy — just return src. Python ensures single path.
    (void)src;
    return NULL;  // not implemented yet
}

void flashmoe_cache_free(FlashMoE_Cache *c) {
    if (!c) return;
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (c->kv_caches[i])    kv_cache_free(c->kv_caches[i]);
        if (c->layer_states[i]) linear_attn_state_free(c->layer_states[i]);
    }
    free(c->kv_caches);
    free(c->layer_states);
    free(c);
}

void flashmoe_cache_reset(FlashMoE_Cache *c) {
    if (!c) return;
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (c->kv_caches[i]) {
            c->kv_caches[i]->len = 0;
        }
        if (c->layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)c->layer_states[i];
            memset(s->conv_state, 0,
                   (CONV_KERNEL_SIZE - 1) * g_cfg.linear_conv_dim * sizeof(float));
            memset(s->ssm_state, 0,
                   g_cfg.linear_num_v_heads * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float));
        }
    }
    c->pos = 0;
    reset_delta_net_state();
}

// ---- flashmoe_init ----

int flashmoe_init(const char *model_path) {
    if (g_initialized) return -1;

    snprintf(g_model_path, sizeof(g_model_path), "%s", model_path);

    char weights_path[1024], manifest_path[1024];
    snprintf(weights_path,  sizeof(weights_path),  "%s/model_weights.bin",   model_path);
    snprintf(manifest_path, sizeof(manifest_path), "%s/model_weights.json",  model_path);

    if (model_config_load(model_path) != 0) return -1;
    util_arrays_alloc();

    // Metal
    g_metal = metal_setup();
    if (!g_metal) {
        fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
    }

    // I/O thread pool
    io_pool_init();

    // Detect 2-bit experts
    int use_2bit = g_use_2bit;
    if (!use_2bit) {
        char probe[1024];
        snprintf(probe, sizeof(probe), "%s/packed_experts_2bit/layer_00.bin", model_path);
        int pfd = open(probe, O_RDONLY);
        if (pfd >= 0) {
            close(pfd);
            snprintf(probe, sizeof(probe), "%s/packed_experts/layer_00.bin", model_path);
            int pfd4 = open(probe, O_RDONLY);
            if (pfd4 < 0) {
                g_use_2bit = 1;
                printf("[auto] Using 2-bit experts (4-bit not found)\n");
            } else {
                close(pfd4);
            }
        }
    }

    // Load weights
    g_wf = open_weights(weights_path, manifest_path);
    if (!g_wf) {
        fprintf(stderr, "ERROR: Failed to load weights\n");
        return -1;
    }
    if (g_metal) {
        metal_set_weights(g_metal, g_wf->data, g_wf->size);
    }

    // Open expert files
    g_layer_fds = calloc(g_cfg.num_layers, sizeof(int));
    g_layer_mmaps = calloc(g_cfg.num_layers, sizeof(void *));
    g_layer_mmap_sizes = calloc(g_cfg.num_layers, sizeof(size_t));
    for (int i = 0; i < g_cfg.num_layers; i++) g_layer_fds[i] = -1;

    for (int i = 0; i < g_cfg.num_layers; i++) {
        g_layer_fds_cold[i] = -1;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                 g_use_2bit ? "packed_experts_2bit" : "packed_experts", i);
        g_layer_fds[i] = open(path, O_RDONLY);
        g_layer_mmaps[i] = MAP_FAILED;
        g_layer_mmap_sizes[i] = 0;
        if (g_layer_fds[i] >= 0) {
            fcntl(g_layer_fds[i], F_RDAHEAD, 0);
            struct stat st;
            if (fstat(g_layer_fds[i], &st) == 0 && st.st_size > 0) {
                g_layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ,
                                        MAP_PRIVATE, g_layer_fds[i], 0);
                if (g_layer_mmaps[i] != MAP_FAILED) {
                    g_layer_mmap_sizes[i] = st.st_size;
                }
            }
        }
    }

    // LZ4 detection
    {
        char lz4_probe[1024];
        snprintf(lz4_probe, sizeof(lz4_probe), "%s/packed_experts_lz4/layer_00.bin", model_path);
        if (!g_use_2bit && access(lz4_probe, R_OK) == 0) {
            for (int i = 0; i < g_cfg.num_layers; i++) {
                char lz4_path[1024];
                snprintf(lz4_path, sizeof(lz4_path),
                         "%s/packed_experts_lz4/layer_%02d.bin", model_path, i);
                int lz4_fd = open(lz4_path, O_RDONLY);
                if (lz4_fd >= 0) {
                    g_lz4_index[i] = malloc(g_cfg.num_experts * sizeof(LZ4IndexEntry));
                    ssize_t nr = pread(lz4_fd, g_lz4_index[i],
                                       g_cfg.num_experts * sizeof(LZ4IndexEntry), 0);
                    if (nr == g_cfg.num_experts * (ssize_t)sizeof(LZ4IndexEntry)) {
                        close(g_layer_fds[i]);
                        g_layer_fds[i] = lz4_fd;
                        fcntl(lz4_fd, F_RDAHEAD, 1);
                    } else {
                        free(g_lz4_index[i]);
                        g_lz4_index[i] = NULL;
                        close(lz4_fd);
                    }
                }
            }
            if (g_lz4_index[0] != NULL || g_lz4_index[1] != NULL) {  // at least some layers
                g_use_lz4 = 1;
                for (int k = 0; k < MAX_K; k++) {
                    g_lz4_comp_bufs[k] = malloc(g_cfg.expert_size_4bit + 4096);
                }
            }
        }
    }

    // Warm page cache
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (g_layer_fds[i] >= 0) {
            char dummy[4096];
            pread(g_layer_fds[i], dummy, sizeof(dummy), 0);
        }
    }

    // Working buffers
    g_hidden = calloc(g_cfg.hidden_dim, sizeof(float));
    g_logits = calloc(g_cfg.vocab_size, sizeof(float));
    g_final_norm_w = get_tensor_ptr(g_wf, "model.norm.weight");
    g_K = NUM_ACTIVE_EXPERTS;

    // Allocate scratch buffers (g_cfg.hidden_dim / g_cfg.num_experts dependent)
    init_layer_scratch();

    // Build layer weight cache
    if (!layer_cache_built) build_layer_cache(g_wf);

    g_initialized = 1;
    return 0;
}

// ---- flashmoe_forward ----

int flashmoe_forward(const int *input_ids, int n_tokens,
                     float *logits_out, FlashMoE_Cache *cache) {
    if (!g_initialized || !cache || !input_ids || n_tokens < 1 || !logits_out)
        return -1;

    if (g_cache_telemetry_enabled) cache_telemetry_note_token();

    int pos = cache->pos;

    for (int tok = 0; tok < n_tokens; tok++) {
        embed_lookup(g_wf, input_ids[tok], g_hidden);

        for (int layer = 0; layer < g_cfg.num_layers; layer++) {
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            fused_layer_forward(g_wf, layer, g_hidden,
                                is_full ? cache->kv_caches[layer] : NULL,
                                is_full ? NULL : cache->layer_states[layer],
                                pos,
                                g_layer_mmaps[layer] != MAP_FAILED
                                    ? g_layer_mmaps[layer] : NULL,
                                g_K, g_layer_fds[layer]);
        }
        complete_deferred_experts();
        pos++;

        if (g_final_norm_w) {
            float *normed = malloc(g_cfg.hidden_dim * sizeof(float));
            cpu_rms_norm(g_hidden, g_final_norm_w, normed, g_cfg.hidden_dim, RMS_NORM_EPS);
            memcpy(g_hidden, normed, g_cfg.hidden_dim * sizeof(float));
            free(normed);
        }

        lm_head_forward(g_wf, g_hidden, logits_out + (size_t)tok * g_cfg.vocab_size);
    }

    cache->pos = pos;
    return 0;
}

// ---- Accessors ----

int flashmoe_cache_position(FlashMoE_Cache *c) {
    return c ? c->pos : 0;
}

int flashmoe_vocab_size(void)  { return g_cfg.vocab_size; }
int flashmoe_hidden_dim(void)  { return g_cfg.hidden_dim; }
int flashmoe_num_layers(void)  { return g_cfg.num_layers; }

// ---- flashmoe_free ----

void flashmoe_free(void) {
    if (!g_initialized) return;

    io_pool_shutdown();
    if (g_malloc_cache) {
        malloc_cache_free(g_malloc_cache);
        g_malloc_cache = NULL;
    }
    if (g_expert_cache) {
        expert_cache_free(g_expert_cache);
        g_expert_cache = NULL;
    }
    // Close layer FDs, unmap mmaps, free LZ4 indices BEFORE util_arrays_free
    // because util_arrays_free() frees g_layer_fds_cold and g_lz4_index.
    for (int i = 0; i < g_cfg.num_layers; i++) {
        if (g_layer_mmaps[i] != MAP_FAILED) {
            munmap(g_layer_mmaps[i], g_layer_mmap_sizes[i]);
        }
        if (g_layer_fds[i] >= 0) close(g_layer_fds[i]);
        if (g_layer_fds_cold && g_layer_fds_cold[i] >= 0) close(g_layer_fds_cold[i]);
        if (g_lz4_index && g_lz4_index[i]) {
            free(g_lz4_index[i]);
            g_lz4_index[i] = NULL;
        }
    }
    for (int k = 0; k < MAX_K; k++) {
        free(g_lz4_comp_bufs[k]);
        g_lz4_comp_bufs[k] = NULL;
    }
    free(g_layer_fds);
    free(g_layer_mmaps);
    free(g_layer_mmap_sizes);
    g_layer_fds = NULL;
    g_layer_mmaps = NULL;
    g_layer_mmap_sizes = NULL;
    util_arrays_free();
    free(g_hidden);
    free(g_logits);
    g_hidden = NULL;
    g_logits = NULL;
    g_wf = NULL;
    // Free Metal arrays (allocated dynamically after VLA→pointer conversion).
    // Each buffer has +1 retain from newBufferWithLength: plus +1 from CFRetain
    // (see metal_setup.h) — CFRelease before freeing the array.
    if (g_metal) {
        if (g_metal->buf_kv_k) {
            for (int i = 0; i < g_cfg.num_full_attn_layers; i++)
                if (g_metal->buf_kv_k[i]) CFRelease((__bridge CFTypeRef)g_metal->buf_kv_k[i]);
            free(g_metal->buf_kv_k);
        }
        if (g_metal->buf_kv_v) {
            for (int i = 0; i < g_cfg.num_full_attn_layers; i++)
                if (g_metal->buf_kv_v[i]) CFRelease((__bridge CFTypeRef)g_metal->buf_kv_v[i]);
            free(g_metal->buf_kv_v);
        }
        if (g_metal->buf_delta_state) {
            for (int i = 0; i < g_cfg.num_linear_layers; i++)
                if (g_metal->buf_delta_state[i]) CFRelease((__bridge CFTypeRef)g_metal->buf_delta_state[i]);
            free(g_metal->buf_delta_state);
        }
        if (g_metal->buf_conv_state) {
            for (int i = 0; i < g_cfg.num_linear_layers; i++)
                if (g_metal->buf_conv_state[i]) CFRelease((__bridge CFTypeRef)g_metal->buf_conv_state[i]);
            free(g_metal->buf_conv_state);
        }
        free(g_metal);
    }
    g_metal = NULL;
    free(layer_cache);
    layer_cache = NULL;
    layer_cache_built = 0;
    g_initialized = 0;
}
