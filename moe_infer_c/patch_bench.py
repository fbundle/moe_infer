#!/usr/bin/env python3
"""Transform infer.m (397B) into bench.m (35B) with hardcoded prompt IDs."""
import re
import json

MODEL_DIR = "/Volumes/Hippopotamus/vault/code/flash-moe/data/models--mlx-community--Qwen3.5-35B-A3B-4bit"

# Prompt token IDs (pre-computed)
PROMPT_IDS = [
    248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
    26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
    488, 30, 248046, 198, 248045, 74455, 198, 248068, 198,
]
N_PROMPT = len(PROMPT_IDS)

with open("model_config.json") as f:
    cfg = json.load(f)

lay = cfg["expert_layout_4bit"]

REPLACEMENTS = [
    # Model dimensions — use exact original strings (literal, not regex)
    ('#define HIDDEN_DIM          4096', '#define HIDDEN_DIM          2048'),
    ('#define NUM_LAYERS          60', '#define NUM_LAYERS          40'),
    ('#define NUM_ATTN_HEADS      32', '#define NUM_ATTN_HEADS      16'),
    ('#define NUM_EXPERTS         512', '#define NUM_EXPERTS         256'),
    ('#define NUM_EXPERTS_PER_TOK 10', '#define NUM_EXPERTS_PER_TOK 8'),
    ('#define MOE_INTERMEDIATE    1024', '#define MOE_INTERMEDIATE    512'),
    ('#define SHARED_INTERMEDIATE 1024', '#define SHARED_INTERMEDIATE 512'),
    ('#define LINEAR_NUM_V_HEADS  64', '#define LINEAR_NUM_V_HEADS  32'),
    ('#define EXPERT_SIZE         7077888', '#define EXPERT_SIZE         1769472'),

    # Expert layout (replace all hardcoded 397B offsets with 35B values)
    ('gate_w_off = 0;        gate_s_off = 2097152;  gate_b_off = 2228224;',
     f'gate_w_off = {lay["gate_w_off"]};        gate_s_off = {lay["gate_s_off"]};  gate_b_off = {lay["gate_b_off"]};'),
    ('up_w_off   = 2359296;  up_s_off   = 4456448;  up_b_off   = 4587520;',
     f'up_w_off   = {lay["up_w_off"]};  up_s_off   = {lay["up_s_off"]};  up_b_off   = {lay["up_b_off"]};'),
    ('down_w_off = 4718592;  down_s_off = 6815744;  down_b_off = 6946816;',
     f'down_w_off = {lay["down_w_off"]};  down_s_off = {lay["down_s_off"]};  down_b_off = {lay["down_b_off"]};'),

    ('NSUInteger gate_w_off = 0;',
     f'NSUInteger gate_w_off = {lay["gate_w_off"]};'),
    ('NSUInteger gate_s_off = 2097152;',
     f'NSUInteger gate_s_off = {lay["gate_s_off"]};'),
    ('NSUInteger gate_b_off = 2228224;',
     f'NSUInteger gate_b_off = {lay["gate_b_off"]};'),
    ('NSUInteger up_w_off   = 2359296;',
     f'NSUInteger up_w_off   = {lay["up_w_off"]};'),
    ('NSUInteger up_s_off   = 4456448;',
     f'NSUInteger up_s_off   = {lay["up_s_off"]};'),
    ('NSUInteger up_b_off   = 4587520;',
     f'NSUInteger up_b_off   = {lay["up_b_off"]};'),
    ('NSUInteger down_w_off = 4718592;',
     f'NSUInteger down_w_off = {lay["down_w_off"]};'),
    ('NSUInteger down_s_off = 6815744;',
     f'NSUInteger down_s_off = {lay["down_s_off"]};'),
    ('NSUInteger down_b_off = 6946816;',
     f'NSUInteger down_b_off = {lay["down_b_off"]};'),

    # Layer counts (derived — 60→40 layers means 15→10 full, 45→30 linear)
    ('#define NUM_FULL_ATTN_LAYERS 15', '#define NUM_FULL_ATTN_LAYERS 10'),
    ('#define NUM_LINEAR_LAYERS 45', '#define NUM_LINEAR_LAYERS 30'),

    # Hardcoded delta-net dimensions in buffer allocation (397B → 35B)
    ('64*128*128*sizeof(float)', '32*128*128*sizeof(float)'),  # delta_state per layer
    ('3*12288*sizeof(float)', '3*8192*sizeof(float)'),         # conv_state per layer
    ('memset([ctx->buf_delta_state[i] contents], 0, 64*128*128*sizeof(float))',
     'memset([ctx->buf_delta_state[i] contents], 0, 32*128*128*sizeof(float))'),
    ('memset([ctx->buf_conv_state[i] contents], 0, 3*12288*sizeof(float))',
     'memset([ctx->buf_conv_state[i] contents], 0, 3*8192*sizeof(float))'),
    # g_decay / beta buffers: 64 → 32 v_heads
    ('64*sizeof(float)    options:MTLResourceStorageModeShared];  // buf_delta_g_decay',
     '32*sizeof(float)    options:MTLResourceStorageModeShared];  // buf_delta_g_decay'),
    ('64*sizeof(float)    options:MTLResourceStorageModeShared];  // buf_delta_beta',
     '32*sizeof(float)    options:MTLResourceStorageModeShared];  // buf_delta_beta'),
    # q/k/v/output/conv buffers
    ('8192*sizeof(float)  options:MTLResourceStorageModeShared];  // buf_delta_v',
     '4096*sizeof(float)  options:MTLResourceStorageModeShared];  // buf_delta_v'),
    ('8192*sizeof(float)  options:MTLResourceStorageModeShared];  // buf_delta_output',
     '4096*sizeof(float)  options:MTLResourceStorageModeShared];  // buf_delta_output'),
    ('12288*sizeof(float) options:MTLResourceStorageModeShared];  // buf_conv_input',
     '8192*sizeof(float) options:MTLResourceStorageModeShared];  // buf_conv_input'),
    ('12288*sizeof(float) options:MTLResourceStorageModeShared];  // buf_conv_output',
     '8192*sizeof(float) options:MTLResourceStorageModeShared];  // buf_conv_output'),

    # Model path
    ('/Users/danielwoods/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3',
     MODEL_DIR),

    # ── Stub vocab (benchmark doesn't need token decoding) ──
    # Replace the fopen-failure return NULL with a dummy vocab
    ('if (!f) {\n        fprintf(stderr, "ERROR: Cannot open vocab %s\\n", path);\n        return NULL;\n    }',
     'if (!f) {\n        fprintf(stderr, "[vocab] No vocab.bin, using stub (token IDs only)\\n");\n        Vocabulary *v = calloc(1, sizeof(Vocabulary));\n        v->num_tokens = 256000;\n        v->tokens = calloc(1, sizeof(char *));\n        v->lengths = calloc(1, sizeof(int));\n        return v;\n    }'),

    # Replace all printf decode_token patterns with integer output in CLI path
    ('printf("%s", decode_token(vocab, next_token));',
     'printf("%d ", next_token);'),
    ('fprintf(stderr, "  token %d (\\"%s\\") logit=%.4f\\n",\n                        top5[i], decode_token(vocab, top5[i]), topv[i]);',
     'fprintf(stderr, "  token %d logit=%.4f\\n", top5[i], topv[i]);'),
]

with open("infer.m") as f:
    content = f.read()

for old, new in REPLACEMENTS:
    content = content.replace(old, new)

# ── Replace tokenizer init/encode with hardcoded prompt IDs ──

# Remove the tokenizer.h include
content = content.replace('#include "tokenizer.h"', '// tokenizer.h not needed — hardcoded prompt IDs')

# Remove TOKENIZER_IMPL define
content = content.replace('#define TOKENIZER_IMPL', '// #define TOKENIZER_IMPL  // disabled — hardcoded prompt IDs')

# Replace bpe_tokenizer type with void* (we don't need it)
content = content.replace('static bpe_tokenizer g_tokenizer;', 'static void *g_tokenizer_unused;  // tokenizer disabled')
content = content.replace('static int g_tokenizer_loaded = 0;', 'static int g_tokenizer_loaded = 0;  // always 0')

# Replace init_tokenizer() body with no-op
content = re.sub(
    r'static void init_tokenizer\(void\) \{.*?\n\}',
    'static void init_tokenizer(void) { return; }  // prompt IDs hardcoded',
    content, flags=re.DOTALL
)

# Replace encode_prompt_text_to_tokens with hardcoded version
old_encode_fn = """static PromptTokens *encode_prompt_text_to_tokens(const char *text) {
    init_tokenizer();
    if (!g_tokenizer_loaded) return NULL;

    // Allocate output buffer (generous: 4 tokens per character worst case)
    int max_ids = (int)strlen(text) * 4 + 256;
    uint32_t *ids = malloc(max_ids * sizeof(uint32_t));
    if (!ids) return NULL;

    int n = bpe_encode(&g_tokenizer, text, ids, max_ids);
    if (n < 0) { free(ids); return NULL; }

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    pt->ids = ids;
    pt->count = n;

    fprintf(stderr, "Tokens (%d): [", n);
    for (int i = 0; i < n && i < 20; i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%u", ids[i]);
    }
    if (n > 20) fprintf(stderr, ", ...");
    fprintf(stderr, "]\\n");

    return pt;
}"""

new_encode_fn = f"""static PromptTokens *encode_prompt_text_to_tokens(const char *text) {{
    // Hardcoded prompt token IDs for benchmark (no tokenizer needed)
    (void)text;
    static const uint32_t hardcoded[] = {{{', '.join(str(x) for x in PROMPT_IDS)}}};
    int n = {N_PROMPT};

    uint32_t *ids = malloc(n * sizeof(uint32_t));
    memcpy(ids, hardcoded, n * sizeof(uint32_t));

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    pt->ids = ids;
    pt->count = n;

    fprintf(stderr, "Tokens (%d): [", n);
    for (int i = 0; i < n && i < 20; i++) {{
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%u", ids[i]);
    }}
    if (n > 20) fprintf(stderr, ", ...");
    fprintf(stderr, "]\\n");

    return pt;
}}"""

content = content.replace(old_encode_fn, new_encode_fn)

# ── Also add the model_config.json copy note (it's already in the MODEL_DIR) ──

with open("bench.m", "w") as f:
    f.write(content)

print(f"Wrote bench.m ({len(content)} bytes)")
print(f"Prompt IDs: {N_PROMPT} tokens")
print(f"Model path: {MODEL_DIR}")
print(f"Expert size 4-bit: {cfg['expert_size_4bit']}")
