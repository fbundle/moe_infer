#!/usr/bin/env python3
"""Performance benchmark: C vs Rust on the full Qwen3.5-35B-A3B-4bit model."""
import subprocess, struct, sys, os, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.join(ROOT, "moe_infer_c")
MODEL_DIR = os.path.join(ROOT, "data", "models--mlx-community--Qwen3.5-35B-A3B-4bit")
VOCAB_SIZE = 248320

PROMPT = "Explain the theory of relativity in simple terms."
NUM_TOKENS = 100
K_EXPERTS = 8
EOS_1 = 248046
EOS_2 = 248044


def get_encoded_prompt():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    prompt_str = (
        f"<|im_start|>system\nYou are a helpful assistant. /think<|im_end|>\n"
        f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    )
    encoded = tok.encode(prompt_str)
    return encoded.ids, tok


def write_prompt_tokens(path, token_ids):
    data = struct.pack(f"<i{len(token_ids)}i", len(token_ids), *token_ids)
    with open(path, "wb") as f:
        f.write(data)


def ensure_clean_bench():
    """Revert verify-logits patch if present, recompile bench."""
    bench_path = os.path.join(C_DIR, "bench.m")
    with open(bench_path) as f:
        content = f.read()

    if "g_verify_logits_path" not in content:
        return True  # already clean

    print("[bench_perf] Reverting verify-logits patch from bench.m...")
    content = content.replace(
        "static int g_think_budget = 2048;\nstatic const char *g_verify_logits_path = NULL;",
        "static int g_think_budget = 2048;",
    )
    content = content.replace(
        '{"predict",       no_argument,       0, \'D\'},\n            {"verify-logits", required_argument, 0, \'V\'},',
        '{"predict",       no_argument,       0, \'D\'},',
    )
    content = content.replace(
        '"m:w:j:v:p:P:t:k:C:M:R:B:V:LSTFE2Gh"',
        '"m:w:j:v:p:P:t:k:C:M:R:B:LSTFE2Gh"',
    )
    content = content.replace(
        "case 'D': g_pred_enabled = 1; break;\n                case 'V': g_verify_logits_path = optarg; break;",
        "case 'D': g_pred_enabled = 1; break;",
    )
    old_block = (
        "double t_lm = now_ms();\n"
        "        lm_head_forward(wf, hidden, logits);\n"
        "        if (g_verify_logits_path) {\n"
        "            FILE *vf = fopen(g_verify_logits_path, \"wb\");\n"
        "            if (vf) {\n"
        '                fwrite(logits, sizeof(float), VOCAB_SIZE, vf);\n'
        "                fclose(vf);\n"
        "            }\n"
        '            fprintf(stderr, "[verify] Dumped %d logits to %s\\n", (int)VOCAB_SIZE, g_verify_logits_path);\n'
        "            exit(0);\n"
        "        }\n"
        "        double lm_ms = now_ms() - t_lm;"
    )
    new_block = (
        "double t_lm = now_ms();\n"
        "        lm_head_forward(wf, hidden, logits);\n"
        "        double lm_ms = now_ms() - t_lm;"
    )
    content = content.replace(old_block, new_block, 1)

    with open(bench_path, "w") as f:
        f.write(content)

    result = subprocess.run(
        [
            "clang", "-O2", "-Wall", "-fobjc-arc",
            "-framework", "Metal", "-framework", "Foundation",
            "-framework", "Accelerate",
            bench_path, "-lpthread", "-lcompression", "-o", "bench",
        ],
        cwd=C_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        errors = [l for l in result.stderr.splitlines() if "error:" in l]
        print(f"[bench_perf] C compilation FAILED:")
        for e in errors:
            print(f"  {e}")
        return False
    print("[bench_perf] C bench recompiled OK")
    return True


def bench_c(prompt_ids, tok):
    tokens_path = os.path.join(C_DIR, "bench_tokens.bin")
    write_prompt_tokens(tokens_path, prompt_ids)

    print(f"\n{'='*60}")
    print("C Engine Benchmark")
    print(f"{'='*60}")
    print(f"  Prompt tokens: {len(prompt_ids)}, target: {NUM_TOKENS}, K={K_EXPERTS}")

    args = [
        "./bench", "--prompt-tokens", "bench_tokens.bin",
        "--tokens", str(NUM_TOKENS), "--k", str(K_EXPERTS), "--timing",
    ]
    print(f"  Running: {' '.join(args)}")
    t0 = time.time()
    result = subprocess.run(args, cwd=C_DIR, capture_output=True, text=True, timeout=300)
    wall_ms = (time.time() - t0) * 1000

    metrics = {"wall_ms": wall_ms}

    for line in result.stderr.splitlines() + result.stdout.splitlines():
        line = line.strip()
        if "[ttft]" in line:
            metrics["ttft_ms"] = float(line.split()[1])
        elif "TTFT:" in line:
            metrics["ttft_ms"] = float(line.split()[1])
        elif "Total time:" in line:
            metrics["total_s"] = float(line.split()[2])
        elif "Generation:" in line:
            parts = line.split()
            metrics["gen_s"] = float(parts[1])
            metrics["tok_s"] = float(parts[3].replace("(", "").replace(")", ""))
        elif "Tokens:" in line and "generated" in line:
            metrics["tokens_generated"] = int(line.split()[1])

    # Per-layer timing
    in_timing = False
    for line in result.stderr.splitlines():
        if "[timing]" in line:
            in_timing = True
            continue
        if in_timing and ":" in line and line.strip():
            parts = line.strip().split(":")
            if len(parts) >= 2:
                key = parts[0].strip()
                try:
                    metrics[f"c_{key}"] = float(parts[1].strip().split()[0])
                except ValueError:
                    pass
        elif in_timing and not line.strip():
            break

    # Print relevant output
    for line in result.stderr.splitlines():
        line_stripped = line.strip()
        if any(kw in line_stripped.lower() for kw in [
            "ttft", "total time", "generation:", "timing", "tok/s",
            "generated", "eos", "per-layer",
        ]):
            print(f"  {line_stripped}")
    for line in result.stdout.splitlines():
        line_stripped = line.strip()
        if any(kw in line_stripped.lower() for kw in [
            "ttft", "total time", "generation:", "statistics", "tok/s",
        ]):
            print(f"  {line_stripped}")

    # Decode output tokens
    gen_parts = result.stdout.replace("\n", " ").split()
    gen_ids = [int(p) for p in gen_parts if p.lstrip("-").isdigit()]
    if gen_ids:
        decoded = tok.decode(gen_ids)
        print(f"\n  Output: {decoded[:300]}...")

    return metrics


def bench_rust(prompt_ids, tok):
    sys.path.insert(0, os.path.join(ROOT, ".venv/lib/python3.14/site-packages"))
    from moe_infer import Context, Cache

    print(f"\n{'='*60}")
    print("Rust Engine Benchmark")
    print(f"{'='*60}")
    print(f"  Prompt tokens: {len(prompt_ids)}, target: {NUM_TOKENS}")

    t_init = time.time()
    ctx = Context()
    ctx.load_model(MODEL_DIR, pipeline_mode="Fused3")
    cache = ctx.new_cache()
    init_ms = (time.time() - t_init) * 1000
    print(f"  Init: {init_ms:.0f} ms")

    # Prefill
    print(f"  Prefilling {len(prompt_ids)} tokens...")
    ids_arr = np.array(prompt_ids, dtype=np.int64)
    t0 = time.time()
    logits_all = ctx.forward(ids_arr, cache)
    prefill_ms = (time.time() - t0) * 1000
    print(f"  Prefill: {prefill_ms:.0f} ms")

    last_logits = np.array(logits_all[-1], dtype=np.float32)
    first_token = int(np.argmax(last_logits))
    print(f"  First token: {first_token}")

    # Generation (one token at a time — forward() expects full sequence, uses cache.pos to skip)
    print(f"  Generating up to {NUM_TOKENS} tokens (greedy)...")
    all_ids = list(prompt_ids) + [first_token]
    t_gen_start = time.time()

    for _ in range(NUM_TOKENS):
        ids_arr = np.array(all_ids, dtype=np.int64)
        logits_all = ctx.forward(ids_arr, cache)
        last_logits = np.array(logits_all[-1], dtype=np.float32)
        next_token = int(np.argmax(last_logits))
        all_ids.append(next_token)
        if next_token == EOS_1 or next_token == EOS_2:
            break

    gen_ms = (time.time() - t_gen_start) * 1000
    gen_tokens = all_ids[len(prompt_ids):]  # all generated tokens
    gen_count = len(gen_tokens)
    tok_s = gen_count * 1000.0 / gen_ms if gen_ms > 0 else 0

    decoded = tok.decode(gen_tokens)
    print(f"  Generated: {gen_count} tokens in {gen_ms:.0f} ms")
    print(f"  Speed:     {tok_s:.2f} tok/s")
    print(f"  Output:    {decoded[:300]}...")

    ctx.unload_model()

    return {
        "init_ms": init_ms,
        "prefill_ms": prefill_ms,
        "ttft_ms": prefill_ms,
        "gen_ms": gen_ms,
        "gen_tokens": gen_count,
        "tok_s": tok_s,
    }


def main():
    print("=" * 60)
    print("Flash-MoE Performance: C vs Rust")
    print(f"Model: Qwen3.5-35B-A3B-4bit (40 layers, 256 experts)")
    print(f"Prompt: {PROMPT}")
    print("=" * 60)

    if not ensure_clean_bench():
        print("[bench_perf] C bench compilation failed, aborting")
        sys.exit(1)

    prompt_ids, tok = get_encoded_prompt()
    print(f"Encoded prompt: {len(prompt_ids)} tokens")

    c = bench_c(prompt_ids, tok)
    r = bench_rust(prompt_ids, tok)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'C':>12} {'Rust':>12} {'Ratio':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    for label, c_key, r_key in [
        ("TTFT (ms)", "ttft_ms", "ttft_ms"),
        ("Gen tokens", "tokens_generated", "gen_tokens"),
        ("Gen speed (tok/s)", "tok_s", "tok_s"),
    ]:
        c_val = c.get(c_key, 0) or 0
        r_val = r.get(r_key, 0) or 0
        ratio = f"{r_val/c_val:.2f}x" if c_val > 0 else "N/A"
        print(f"{label:<25} {c_val:>12.1f} {r_val:>12.1f} {ratio:>10}")

    # C per-layer breakdown
    tl_keys = [
        "deferred_wait", "deferred_cpu", "input_norm", "cmd1_submit",
        "cmd1_wait", "cpu_attn", "cmd2_encode", "cmd2_wait",
        "routing_cpu", "expert_io", "cmd3_encode", "total_layer",
    ]
    if any(f"c_{k}" in c for k in tl_keys):
        print(f"\n  C per-layer breakdown (ms):")
        for k in tl_keys:
            val = c.get(f"c_{k}")
            if val is not None:
                print(f"    {k:<20} {val:.3f}")


if __name__ == "__main__":
    main()
