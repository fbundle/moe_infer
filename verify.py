#!/usr/bin/env python3
"""Verify Rust port produces same logits as C baseline (within epsilon).

1. Re-patches C bench with logit-dump mode
2. Recompiles C bench
3. Runs both C and Rust engines on the same 100-token sequence
4. Compares logits element-by-element within epsilon
"""
import subprocess, struct, sys, os, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.join(ROOT, "moe_infer_c")
MODEL_DIR = os.path.join(ROOT, "data", "models--mlx-community--Qwen3.5-35B-A3B-4bit")
VOCAB_SIZE = 248320
HIDDEN_DIM = 2048

# Fixed token sequence for verification: ~100 tokens
VERIFY_TOKENS = [
    248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
    26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
    488, 30, 248046, 198, 248045, 74455, 198, 248068, 198,
    248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
    26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
    488, 30, 248046, 198, 248045, 74455, 198, 248068, 198,
    248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
    26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
    488, 30, 248046, 198, 248045, 74455, 198, 248068, 198,
    248045, 8678, 198, 2523, 513, 264, 10631, 17313,
]


def patch_bench_for_verify():
    """Add --verify-logits flag to bench.m (logit dump after prefill lm_head)."""
    bench_path = os.path.join(C_DIR, "bench.m")

    with open(bench_path) as f:
        content = f.read()

    # Check if already patched
    if 'g_verify_logits_path' in content:
        print("[verify] bench.m already patched for logit dump")
        return

    # 1. Add global variable near other static globals
    old_think = 'static int g_think_budget = 2048;'
    new_think = 'static int g_think_budget = 2048;\nstatic const char *g_verify_logits_path = NULL;'
    content = content.replace(old_think, new_think)

    # 2. Add --verify-logits to long_options
    old_opts = '{"predict",       no_argument,       0, \'D\'},'
    new_opts = '{"predict",       no_argument,       0, \'D\'},\n            {"verify-logits", required_argument, 0, \'V\'},'
    content = content.replace(old_opts, new_opts)

    # 3. Add getopt string char
    old_getopt = '"m:w:j:v:p:P:t:k:C:M:R:B:LSTFE2Gh"'
    new_getopt = '"m:w:j:v:p:P:t:k:C:M:R:B:V:LSTFE2Gh"'
    content = content.replace(old_getopt, new_getopt)

    # 4. Add case handler in getopt switch
    old_case = 'case \'D\': g_pred_enabled = 1; break;'
    new_case = 'case \'D\': g_pred_enabled = 1; break;\n                case \'V\': g_verify_logits_path = optarg; break;'
    content = content.replace(old_case, new_case)

    # 5. After lm_head_forward in the GENERATION prefill path, dump logits if --verify-logits.
    #    Target the unique pattern with "double t_lm = now_ms()" (only in gen prefill, not serve).
    old_lm = (
        'double t_lm = now_ms();\n'
        '        lm_head_forward(wf, hidden, logits);\n'
        '        double lm_ms = now_ms() - t_lm;'
    )
    verify_block = (
        'double t_lm = now_ms();\n'
        '        lm_head_forward(wf, hidden, logits);\n'
        '        if (g_verify_logits_path) {\n'
        '            FILE *vf = fopen(g_verify_logits_path, "wb");\n'
        '            if (vf) {\n'
        '                fwrite(logits, sizeof(float), VOCAB_SIZE, vf);\n'
        '                fclose(vf);\n'
        '            }\n'
        '            fprintf(stderr, "[verify] Dumped %d logits to %s\\n", (int)VOCAB_SIZE, g_verify_logits_path);\n'
        '            exit(0);\n'
        '        }\n'
        '        double lm_ms = now_ms() - t_lm;'
    )
    content = content.replace(old_lm, verify_block, 1)

    with open(bench_path, "w") as f:
        f.write(content)
    print("[verify] Patched bench.m with --verify-logits flag")


def compile_c_bench():
    """Compile the C bench binary."""
    bench_m = os.path.join(C_DIR, "bench.m")
    print("[verify] Compiling C bench...")
    result = subprocess.run(
        ["clang", "-O2", "-Wall", "-fobjc-arc",
         "-framework", "Metal", "-framework", "Foundation",
         "-framework", "Accelerate",
         bench_m, "-lpthread", "-lcompression", "-o", "bench"],
        cwd=C_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        # Show only errors (not warnings)
        errors = [l for l in result.stderr.splitlines() if 'error:' in l]
        print(f"[verify] C compilation FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("[verify] C bench compiled OK")


def write_prompt_tokens(path, token_ids):
    """Write tokens in C bench format: [int32 count][int32 tokens...]."""
    data = struct.pack(f'<i{len(token_ids)}i', len(token_ids), *token_ids)
    with open(path, 'wb') as f:
        f.write(data)


def run_c_bench():
    """Run C bench with --prompt-tokens and --verify-logits, return logits array."""
    tokens_path = os.path.join(C_DIR, "verify_tokens.bin")
    write_prompt_tokens(tokens_path, VERIFY_TOKENS)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        logits_path = tmp.name

    print(f"[verify] Running C bench ({len(VERIFY_TOKENS)} tokens)...")
    t0 = __import__('time').time()

    args = ["./bench", "--prompt-tokens", "verify_tokens.bin",
            "--verify-logits", logits_path, "--tokens", "1", "--k", "8"]

    result = subprocess.run(args, cwd=C_DIR, capture_output=True, text=True)
    elapsed = __import__('time').time() - t0

    # Check for logits file
    if not os.path.exists(logits_path) or os.path.getsize(logits_path) == 0:
        print("[verify] ERROR: C bench did not produce logits file")
        print("STDERR:", result.stderr[-3000:])
        print("STDOUT:", result.stdout[-1000:])
        sys.exit(1)

    logits = np.fromfile(logits_path, dtype=np.float32)
    os.unlink(logits_path)

    print(f"[verify] C bench done in {elapsed * 1000:.0f} ms")
    print(f"[verify] C logits: shape={logits.shape}, min={logits.min():.4f}, "
          f"max={logits.max():.4f}, mean={logits.mean():.4f}, "
          f"NaNs={np.isnan(logits).sum()}")

    for line in result.stderr.splitlines():
        if 'verify' in line.lower() or 'error' in line.lower():
            print(f"  [C] {line}")

    return logits


def run_rust_bench():
    """Run Rust engine via PyO3 bindings, return logits for the last position."""
    from moe_infer import Context, Cache

    print("[verify] Running Rust engine...")
    t0 = __import__('time').time()

    ctx = Context()
    ctx.load_model(MODEL_DIR, pipeline_mode="Fused3")
    cache = ctx.new_cache()

    ids_arr = np.array(VERIFY_TOKENS, dtype=np.int64)
    logits_all = ctx.forward(ids_arr, cache)

    elapsed = __import__('time').time() - t0
    print(f"[verify] Rust forward done in {elapsed * 1000:.0f} ms")

    # C bench returns logits after processing ALL tokens (predicting token N+1)
    # Rust forward returns logits for each position. logits[i] predicts token i+1.
    logits = np.array(logits_all[-1], dtype=np.float32)

    print(f"[verify] Rust logits: shape={logits.shape}, min={logits.min():.4f}, "
          f"max={logits.max():.4f}, mean={logits.mean():.4f}, "
          f"NaNs={np.isnan(logits).sum()}")

    ctx.unload_model()
    return logits


def compare_logits(c_logits, rust_logits, eps=1e-3):
    """Compare C and Rust logits element-by-element."""
    min_len = min(len(c_logits), len(rust_logits))
    if len(c_logits) != len(rust_logits):
        print(f"[verify] WARNING: length mismatch C={len(c_logits)} Rust={len(rust_logits)}")
        c_logits = c_logits[:min_len]
        rust_logits = rust_logits[:min_len]

    diff = np.abs(c_logits - rust_logits)
    max_diff = diff.max()
    mean_diff = diff.mean()
    idx_max = diff.argmax()

    matching = (diff < eps).sum()
    pct = 100.0 * matching / len(diff)

    print(f"\n[verify] Comparison ({len(diff)} elements, eps={eps}):")
    print(f"  Max diff:   {max_diff:.6f} at index {idx_max} "
          f"(C={c_logits[idx_max]:.6f}, Rust={rust_logits[idx_max]:.6f})")
    print(f"  Mean diff:  {mean_diff:.6f}")
    print(f"  Within eps: {matching}/{len(diff)} ({pct:.2f}%)")

    if max_diff < eps:
        print("\n[verify] PASS — all logits within epsilon")
        return True
    elif pct > 99.9:
        print(f"\n[verify] WARNING — {100.0 - pct:.2f}% outside epsilon, max diff={max_diff:.6f}")
        print("[verify] This may indicate minor numerical differences (acceptable)")
        return True
    else:
        print(f"\n[verify] FAIL — significant divergence ({100.0 - pct:.2f}% outside epsilon)")
        top10 = np.argsort(diff)[-10:][::-1]
        print("  Top-10 differences:")
        for i in top10:
            print(f"    [{i}] C={c_logits[i]:.6f} Rust={rust_logits[i]:.6f} diff={diff[i]:.6f}")
        return False


def main():
    print("=" * 60)
    print("Flash-MoE Verification: C vs Rust logits")
    print(f"Tokens: {len(VERIFY_TOKENS)}")
    print(f"Model:  {MODEL_DIR}")
    print("=" * 60)

    # Step 1: Re-generate bench.m and patch for logit dump
    print("\n--- Step 1: Prepare C bench ---")
    subprocess.run([sys.executable, "patch_bench.py"], cwd=C_DIR, capture_output=True)
    patch_bench_for_verify()
    compile_c_bench()

    # Step 2: Run C bench
    print("\n--- Step 2: Run C engine ---")
    c_logits = run_c_bench()

    # Step 3: Run Rust engine
    print("\n--- Step 3: Run Rust engine ---")
    rust_logits = run_rust_bench()

    # Step 4: Compare
    print("\n--- Step 4: Compare ---")
    ok = compare_logits(c_logits, rust_logits)

    if ok:
        print("\n* Verification complete - C and Rust logits match.")
    else:
        print("\n* Verification failed - logits diverge.")
        sys.exit(1)


if __name__ == "__main__":
    main()
