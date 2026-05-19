#ifndef UTIL_H
#define UTIL_H

// Shared utilities: system includes, global state, cache telemetry, timing.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/wait.h>
#include <compression.h>

#include "config.h"
#include "model_config.h"

// ============================================================================
// Timing helper
// ============================================================================

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ============================================================================
// Per-phase timing accumulators for fused_layer_forward
// Tracks time spent in each pipeline phase across all layers per token.
// Reset at token boundary, printed as summary.
// ============================================================================

typedef struct {
    double deferred_wait;    // waiting for previous CMD3 GPU
    double deferred_cpu;     // CPU readback + combine for deferred experts
    double input_norm;       // CPU RMS norm + CMD1 prep
    double cmd1_submit;      // CMD1 encode + commit
    double cmd1_wait;        // CMD1 waitUntilCompleted
    double cpu_attn;         // CPU attention compute (delta-net or full-attn)
    double cmd2_encode;      // CMD2 encode (o_proj + residual + norm + routing)
    double cmd2_wait;        // CMD2 commit + waitUntilCompleted
    double routing_cpu;      // CPU softmax + topK
    double spec_route;       // speculative early routing (gate matvec + topK)
    double expert_io;        // parallel pread + cache lookup
    double cmd3_encode;      // CMD3 encode experts + submit (deferred)
    double total;            // total per-layer time
    int count;               // number of layers timed
} LayerTimingAccum;

static LayerTimingAccum g_timing = {0};
static int g_timing_enabled = 0;

static char g_model_path[1024] = "data";

// Temporal prediction pipeline counters (declared early for timing_print access)
static int g_pred_enabled = USE_EXPERT_PREDICTION;
static int g_pred_generating = 0;   // only set to 1 after prefill (predictions only help during generation)
static uint64_t g_pred_hits = 0;
static uint64_t g_pred_misses = 0;
static uint64_t g_pred_layers = 0;
static int *g_pred_experts = NULL;  // [g_cfg.num_layers * MAX_K] alloc'd at init
static int *g_pred_count = NULL;    // [g_cfg.num_layers] alloc'd at init

// Routing data collection for training an expert predictor
// Binary format per sample: int32 layer_idx, int32 K, float32[4096] hidden, int32[K] expert_indices
static FILE *g_routing_log = NULL;
static int g_routing_log_samples = 0;

// LZ4 compressed expert support
// File format: [LZ4IndexEntry × 512] + [compressed blobs]
typedef struct {
    uint64_t offset;
    uint32_t comp_size;
    uint32_t raw_size;
} LZ4IndexEntry;

static LZ4IndexEntry **g_lz4_index = NULL;  // [g_cfg.num_layers] alloc'd at init
static void *g_lz4_comp_bufs[8];             // pre-allocated compressed read buffers (MAX_K=8)
static int g_use_lz4 = 0;                    // auto-detected from packed_experts_lz4/

// ============================================================================
// Expert frequency tracking (diagnostic: --freq flag)
// ============================================================================

static int *g_expert_freq = NULL;  // [g_cfg.num_layers * g_cfg.num_experts] alloc'd at init
static int g_freq_tracking = 0;
static int g_use_2bit = 0;
static int g_cache_telemetry_enabled = 0;
static int g_think_budget = 2048;

// Tiered I/O: cold fds (F_NOCACHE) for first reads, warm fds (page cached) for repeats
static int *g_layer_fds_cold = NULL;  // [g_cfg.num_layers] alloc'd at init
static uint8_t *g_expert_seen = NULL; // [g_cfg.num_layers * (g_cfg.num_experts/8)] alloc'd at init

// Async pread state defined after InferPreadTask (see below)

static inline int expert_is_seen(int layer, int expert) {
    return (g_expert_seen[layer * (g_cfg.num_experts / 8) + (expert >> 3)] >> (expert & 7)) & 1;
}
static inline void expert_mark_seen(int layer, int expert) {
    g_expert_seen[layer * (g_cfg.num_experts / 8) + (expert >> 3)] |= (1 << (expert & 7));
}
// Pick fd for expert read. Currently: always use warm fd (OS page cache).
// Tiered I/O (cold F_NOCACHE for first reads) was tested but OS page cache
// without any bypass outperforms all custom caching strategies.
static inline int expert_pick_fd(int layer, int expert, int warm_fd) {
    (void)layer; (void)expert;
    return warm_fd;
}

// Active expert size based on quantization mode
static inline size_t active_expert_size(void) {
    return g_use_2bit ? g_cfg.expert_size_2bit : g_cfg.expert_size_4bit;
}
static int g_freq_total_tokens = 0;  // total tokens processed while tracking

typedef struct {
    uint64_t token_clock;
    uint64_t unique_experts_touched;
    uint64_t cold_misses;
    uint64_t eviction_misses;
    uint64_t evictions;
    uint64_t reuse_le_1;
    uint64_t reuse_le_4;
    uint64_t reuse_le_16;
    uint64_t reuse_le_64;
    uint64_t reuse_gt_64;
    uint64_t reuse_distance_sum;
    uint64_t reuse_distance_samples;
} CacheTelemetry;

static CacheTelemetry g_cache_telemetry = {0};
static uint8_t  *g_cache_seen = NULL;
static uint64_t *g_cache_last_touch_token = NULL;
static uint64_t *g_cache_last_evict_token = NULL;

void cache_telemetry_reset(void) {
    memset(&g_cache_telemetry, 0, sizeof(g_cache_telemetry));
    if (g_cache_seen)
        memset(g_cache_seen, 0, (size_t)g_cfg.num_layers * g_cfg.num_experts * sizeof(uint8_t));
    if (g_cache_last_touch_token)
        memset(g_cache_last_touch_token, 0, (size_t)g_cfg.num_layers * g_cfg.num_experts * sizeof(uint64_t));
    if (g_cache_last_evict_token)
        memset(g_cache_last_evict_token, 0, (size_t)g_cfg.num_layers * g_cfg.num_experts * sizeof(uint64_t));
}

static void cache_telemetry_note_token(void) {
    if (!g_cache_telemetry_enabled) return;
    g_cache_telemetry.token_clock++;
}

static void cache_telemetry_touch(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= g_cfg.num_layers || expert_idx < 0 || expert_idx >= g_cfg.num_experts) return;
    if (!g_cache_seen[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)]) {
        g_cache_seen[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)] = 1;
        g_cache_telemetry.unique_experts_touched++;
    }
    g_cache_last_touch_token[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)] = g_cache_telemetry.token_clock;
}

static void cache_telemetry_miss(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= g_cfg.num_layers || expert_idx < 0 || expert_idx >= g_cfg.num_experts) return;
    if (!g_cache_seen[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)]) {
        g_cache_telemetry.cold_misses++;
        g_cache_seen[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)] = 1;
        g_cache_telemetry.unique_experts_touched++;
    } else {
        g_cache_telemetry.eviction_misses++;
        uint64_t dist = 0;
        if (g_cache_last_evict_token[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)] > 0 &&
            g_cache_telemetry.token_clock >= g_cache_last_evict_token[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)]) {
            dist = g_cache_telemetry.token_clock - g_cache_last_evict_token[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)];
        }
        if (dist <= 1) g_cache_telemetry.reuse_le_1++;
        else if (dist <= 4) g_cache_telemetry.reuse_le_4++;
        else if (dist <= 16) g_cache_telemetry.reuse_le_16++;
        else if (dist <= 64) g_cache_telemetry.reuse_le_64++;
        else g_cache_telemetry.reuse_gt_64++;
        g_cache_telemetry.reuse_distance_sum += dist;
        g_cache_telemetry.reuse_distance_samples++;
    }
    g_cache_last_touch_token[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)] = g_cache_telemetry.token_clock;
}

static void cache_telemetry_evict(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= g_cfg.num_layers || expert_idx < 0 || expert_idx >= g_cfg.num_experts) return;
    g_cache_telemetry.evictions++;
    g_cache_last_evict_token[(size_t)(layer_idx) * g_cfg.num_experts + (expert_idx)] = g_cache_telemetry.token_clock;
}

void cache_telemetry_print(uint64_t hits, uint64_t misses) {
    if (!g_cache_telemetry_enabled) return;
    uint64_t total = hits + misses;
    fprintf(stderr, "\n=== Cache Telemetry ===\n");
    fprintf(stderr, "Tokens tracked: %llu\n", g_cache_telemetry.token_clock);
    fprintf(stderr, "Unique experts touched: %llu / %d (%.1f%%)\n",
            g_cache_telemetry.unique_experts_touched,
            g_cfg.num_layers * g_cfg.num_experts,
            100.0 * g_cache_telemetry.unique_experts_touched / (g_cfg.num_layers * g_cfg.num_experts));
    fprintf(stderr, "Miss breakdown: cold %llu (%.1f%% of misses), eviction %llu (%.1f%% of misses)\n",
            g_cache_telemetry.cold_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.cold_misses / misses : 0.0,
            g_cache_telemetry.eviction_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.eviction_misses / misses : 0.0);
    fprintf(stderr, "Evictions: %llu\n", g_cache_telemetry.evictions);
    fprintf(stderr, "Eviction reuse distance: <=1 tok %llu, <=4 %llu, <=16 %llu, <=64 %llu, >64 %llu",
            g_cache_telemetry.reuse_le_1,
            g_cache_telemetry.reuse_le_4,
            g_cache_telemetry.reuse_le_16,
            g_cache_telemetry.reuse_le_64,
            g_cache_telemetry.reuse_gt_64);
    if (g_cache_telemetry.reuse_distance_samples > 0) {
        fprintf(stderr, " (avg %.1f tok)\n",
                (double)g_cache_telemetry.reuse_distance_sum / g_cache_telemetry.reuse_distance_samples);
    } else {
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "Effective hit rate: %.1f%%\n",
            total > 0 ? 100.0 * hits / total : 0.0);
}

void timing_reset(void) {
    memset(&g_timing, 0, sizeof(g_timing));
}

void timing_print(void) {
    if (g_timing.count == 0) return;
    int n = g_timing.count;
    fprintf(stderr, "\n[timing] Per-layer breakdown (avg of %d layers, ms):\n", n);
    fprintf(stderr, "  deferred_wait:  %6.3f\n", g_timing.deferred_wait / n);
    fprintf(stderr, "  deferred_cpu:   %6.3f\n", g_timing.deferred_cpu / n);
    fprintf(stderr, "  input_norm:     %6.3f\n", g_timing.input_norm / n);
    fprintf(stderr, "  cmd1_submit:    %6.3f\n", g_timing.cmd1_submit / n);
    fprintf(stderr, "  cmd1_wait:      %6.3f\n", g_timing.cmd1_wait / n);
    fprintf(stderr, "  spec_route:     %6.3f\n", g_timing.spec_route / n);
    fprintf(stderr, "  cpu_attn:       %6.3f\n", g_timing.cpu_attn / n);
    fprintf(stderr, "  cmd2_encode:    %6.3f\n", g_timing.cmd2_encode / n);
    fprintf(stderr, "  cmd2_wait:      %6.3f\n", g_timing.cmd2_wait / n);
    fprintf(stderr, "  routing_cpu:    %6.3f\n", g_timing.routing_cpu / n);
    fprintf(stderr, "  expert_io:      %6.3f\n", g_timing.expert_io / n);
    fprintf(stderr, "  cmd3_encode:    %6.3f\n", g_timing.cmd3_encode / n);
    fprintf(stderr, "  total_layer:    %6.3f\n", g_timing.total / n);
    fprintf(stderr, "  sum_phases:     %6.3f\n",
            (g_timing.deferred_wait + g_timing.deferred_cpu + g_timing.input_norm +
             g_timing.cmd1_submit + g_timing.cmd1_wait + g_timing.spec_route +
             g_timing.cpu_attn +
             g_timing.cmd2_encode + g_timing.cmd2_wait + g_timing.routing_cpu +
             g_timing.expert_io + g_timing.cmd3_encode) / n);
    fprintf(stderr, "  cmd_buffers:    %d (3 per layer: CMD1+CMD2+CMD3)\n", n * 3);
    fprintf(stderr, "  sync_waits:     %d (2 per layer: CMD1+CMD2, CMD3 deferred)\n", n * 2);
    fprintf(stderr, "  gpu_encoders:   ~%d per layer (CMD1:3-4, CMD2:8-12, CMD3:~10)\n",
            22);  // approximate
    if (g_pred_enabled && g_pred_layers > 0) {
        uint64_t total = g_pred_hits + g_pred_misses;
        double hit_rate = total > 0 ? (double)g_pred_hits / total * 100.0 : 0;
        fprintf(stderr, "  [predict] hits=%llu misses=%llu rate=%.1f%% layers=%llu\n",
                g_pred_hits, g_pred_misses, hit_rate, g_pred_layers);
    }
}

// ============================================================================
// bf16 <-> f32 conversion (CPU side)
// ============================================================================

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

__attribute__((unused))
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

// ---- Dynamic array allocation (called after model_config_load) ----

static int util_arrays_alloc(void) {
    int N = g_cfg.num_layers;
    int E = g_cfg.num_experts;

    g_lz4_index = calloc(N, sizeof(LZ4IndexEntry *));

    g_layer_fds_cold = calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) g_layer_fds_cold[i] = -1;

    g_expert_seen = calloc((size_t)N * (E / 8), 1);

    g_expert_freq = calloc((size_t)N * E, sizeof(int));

    g_cache_seen = calloc((size_t)N * E, 1);
    g_cache_last_touch_token = calloc((size_t)N * E, sizeof(uint64_t));
    g_cache_last_evict_token = calloc((size_t)N * E, sizeof(uint64_t));

    g_pred_experts = calloc((size_t)N * MAX_K, sizeof(int));
    g_pred_count   = calloc(N, sizeof(int));

    return 0;
}

static void util_arrays_free(void) {
    free(g_lz4_index);      g_lz4_index = NULL;
    free(g_layer_fds_cold); g_layer_fds_cold = NULL;
    free(g_expert_seen);    g_expert_seen = NULL;
    free(g_expert_freq);    g_expert_freq = NULL;
    free(g_cache_seen);     g_cache_seen = NULL;
    free(g_cache_last_touch_token); g_cache_last_touch_token = NULL;
    free(g_cache_last_evict_token); g_cache_last_evict_token = NULL;
    free(g_pred_experts);  g_pred_experts = NULL;
    free(g_pred_count);    g_pred_count = NULL;
}

#endif // UTIL_H
