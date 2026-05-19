// Runtime model configuration — loaded from model_config.json at init time.
// All model dimensions are accessed via g_cfg.* directly — no compatibility macros.
// This enables multi-model support with different configs.

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#import <Foundation/Foundation.h>

#include "config.h"

// ---- Expert layout (per quantization) ----
typedef struct {
    int gate_w_off, gate_s_off, gate_b_off;
    int up_w_off, up_s_off, up_b_off;
    int down_w_off, down_s_off, down_b_off;
    int gate_w_size, gate_s_size, gate_b_size;
    int up_w_size, up_s_size, up_b_size;
    int down_w_size, down_s_size, down_b_size;
} ExpertLayout;

// ---- Model dimensions ----
typedef struct {
    int hidden_dim, num_layers, num_attn_heads, num_kv_heads;
    int vocab_size, num_experts, num_experts_per_tok;
    int moe_intermediate, shared_intermediate;
    int linear_num_v_heads, linear_num_k_heads;
    int rotary_dim, linear_total_key, linear_total_value, linear_conv_dim;
    int num_full_attn_layers, num_linear_layers;
    int expert_size_4bit, expert_size_2bit;
    ExpertLayout layout_4bit, layout_2bit;
} ModelConfig;

extern ModelConfig g_cfg;

// ---- Load (implementation in this header) ----

static int json_int(NSDictionary *d, NSString *key, int fallback) {
    NSNumber *v = d[key];
    return v ? [v intValue] : fallback;
}

static void parse_layout(ExpertLayout *l, NSDictionary *d) {
    l->gate_w_off = json_int(d, @"gate_w_off", 0);
    l->gate_s_off = json_int(d, @"gate_s_off", 0);
    l->gate_b_off = json_int(d, @"gate_b_off", 0);
    l->up_w_off   = json_int(d, @"up_w_off", 0);
    l->up_s_off   = json_int(d, @"up_s_off", 0);
    l->up_b_off   = json_int(d, @"up_b_off", 0);
    l->down_w_off = json_int(d, @"down_w_off", 0);
    l->down_s_off = json_int(d, @"down_s_off", 0);
    l->down_b_off = json_int(d, @"down_b_off", 0);
    l->gate_w_size = json_int(d, @"gate_w_size", 0);
    l->gate_s_size = json_int(d, @"gate_s_size", 0);
    l->gate_b_size = json_int(d, @"gate_b_size", 0);
    l->up_w_size   = json_int(d, @"up_w_size", 0);
    l->up_s_size   = json_int(d, @"up_s_size", 0);
    l->up_b_size   = json_int(d, @"up_b_size", 0);
    l->down_w_size = json_int(d, @"down_w_size", 0);
    l->down_s_size = json_int(d, @"down_s_size", 0);
    l->down_b_size = json_int(d, @"down_b_size", 0);
}

static int model_config_load(const char *model_path) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model_config.json", model_path);

    NSString *ns_path = [NSString stringWithUTF8String:path];
    NSData *data = [NSData dataWithContentsOfFile:ns_path];
    if (!data) {
        fprintf(stderr, "ERROR: Cannot read %s\n", path);
        return -1;
    }

    NSError *err = nil;
    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data
                                                         options:0
                                                           error:&err];
    if (!json) {
        fprintf(stderr, "ERROR: Invalid JSON in %s: %s\n",
                path, [[err localizedDescription] UTF8String]);
        return -1;
    }

    g_cfg.hidden_dim      = json_int(json, @"hidden_dim", 2048);
    g_cfg.num_layers      = json_int(json, @"num_layers", 40);
    g_cfg.num_attn_heads  = json_int(json, @"num_attn_heads", 16);
    g_cfg.num_kv_heads    = json_int(json, @"num_kv_heads", 2);
    g_cfg.vocab_size      = json_int(json, @"vocab_size", 248320);
    g_cfg.num_experts     = json_int(json, @"num_experts", 256);
    g_cfg.num_experts_per_tok = json_int(json, @"num_experts_per_tok", 8);
    g_cfg.moe_intermediate     = json_int(json, @"moe_intermediate", 512);
    g_cfg.shared_intermediate  = json_int(json, @"shared_intermediate", 512);
    g_cfg.linear_num_v_heads   = json_int(json, @"linear_num_v_heads", 32);
    g_cfg.linear_num_k_heads   = json_int(json, @"linear_num_k_heads", 16);
    g_cfg.rotary_dim           = json_int(json, @"rotary_dim", 64);
    g_cfg.linear_total_key     = json_int(json, @"linear_total_key", 2048);
    g_cfg.linear_total_value   = json_int(json, @"linear_total_value", 4096);
    g_cfg.linear_conv_dim      = json_int(json, @"linear_conv_dim", 8192);
    g_cfg.num_full_attn_layers = json_int(json, @"num_full_attn_layers", 10);
    g_cfg.num_linear_layers    = json_int(json, @"num_linear_layers", 30);
    g_cfg.expert_size_4bit     = json_int(json, @"expert_size_4bit", 1769472);
    g_cfg.expert_size_2bit     = json_int(json, @"expert_size_2bit", 983040);

    NSDictionary *l4 = json[@"expert_layout_4bit"];
    if (l4) parse_layout(&g_cfg.layout_4bit, l4);
    NSDictionary *l2 = json[@"expert_layout_2bit"];
    if (l2) parse_layout(&g_cfg.layout_2bit, l2);

    printf("[config] hidden_dim=%d, num_layers=%d (%d full + %d linear)\n",
           g_cfg.hidden_dim, g_cfg.num_layers,
           g_cfg.num_full_attn_layers, g_cfg.num_linear_layers);
    printf("  experts=%d (K=%d), moe_inter=%d, shared_inter=%d\n",
           g_cfg.num_experts, g_cfg.num_experts_per_tok,
           g_cfg.moe_intermediate, g_cfg.shared_intermediate);
    printf("  attn_heads=%d, kv_heads=%d, vocab=%d, head_dim=%d\n",
           g_cfg.num_attn_heads, g_cfg.num_kv_heads,
           g_cfg.vocab_size, HEAD_DIM);
    printf("  linear: v_heads=%d, k_heads=%d\n",
           g_cfg.linear_num_v_heads, g_cfg.linear_num_k_heads);
    printf("  expert_size=%d bytes (4-bit), %d bytes (2-bit)\n",
           g_cfg.expert_size_4bit, g_cfg.expert_size_2bit);

    return 0;
}

#endif // MODEL_CONFIG_H
