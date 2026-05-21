#!/usr/bin/env python3
"""Quick 2-token verification: C vs Rust logits."""
import subprocess, struct, sys, os, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.join(ROOT, "moe_infer_c")
MODEL_DIR = os.path.join(ROOT, "data", "models--mlx-community--Qwen3.5-35B-A3B-4bit")
VOCAB_SIZE = 248320
HIDDEN_DIM = 2048

# First 2 tokens from verify sequence
TOKENS_2 = [248045, 8678]


def write_prompt_tokens(path, token_ids):
    data = struct.pack(f'<i{len(token_ids)}i', len(token_ids), *token_ids)
    with open(path, 'wb') as f:
        f.write(data)


def run_c(logits_path):
    """Run C bench with 2 prompt tokens, dump logits."""
    tokens_path = os.path.join(C_DIR, "verify_2tok.bin")
    write_prompt_tokens(tokens_path, TOKENS_2)
    args = ["./bench", "--prompt-tokens", "verify_2tok.bin",
            "--verify-logits", logits_path, "--tokens", "1", "--k", "8"]
    result = subprocess.run(args, cwd=C_DIR, capture_output=True, text=True)
    print("=== C STDERR ===")
    for line in result.stderr.splitlines():
        if any(tag in line for tag in ['[C-', '[verify]', 'Error', 'error', 'WARNING']):
            print(f"  {line}")
    if os.path.exists(logits_path) and os.path.getsize(logits_path) > 0:
        logits = np.fromfile(logits_path, dtype=np.float32)
        print(f"C logits: shape={logits.shape}, min={logits.min():.4f}, max={logits.max():.4f}")
        return logits
    else:
        print("C FAILED to produce logits")
        print("STDERR tail:", result.stderr[-1000:])
        return None


def run_rust():
    """Run Rust forward with 2 tokens, return logits for last position."""
    from moe_infer import Context, Cache
    ctx = Context()
    ctx.load_model(MODEL_DIR, pipeline_mode="Fused3")
    cache = ctx.new_cache()
    ids_arr = np.array(TOKENS_2, dtype=np.int64)
    logits_all = ctx.forward(ids_arr, cache)
    # logits_all has shape (n, vocab). logits_all[i] predicts token i+1.
    # For 2 tokens, logits_all[1] predicts token 2.
    logits = np.array(logits_all[-1], dtype=np.float32)
    print(f"Rust logits: shape={logits.shape}, min={logits.min():.4f}, max={logits.max():.4f}")

    # Also check first token's logits
    logits_0 = np.array(logits_all[0], dtype=np.float32)
    print(f"Rust logits[0]: min={logits_0.min():.4f}, max={logits_0.max():.4f}")

    ctx.unload_model()
    return logits


def compare(c_logits, rust_logits, eps=1e-3):
    min_len = min(len(c_logits), len(rust_logits))
    c_logits = c_logits[:min_len]
    rust_logits = rust_logits[:min_len]
    diff = np.abs(c_logits - rust_logits)
    max_diff = diff.max()
    idx_max = diff.argmax()
    matching = (diff < eps).sum()
    pct = 100.0 * matching / len(diff)
    print(f"\nComparison: max_diff={max_diff:.6f} at idx {idx_max}, within_eps={pct:.2f}%")
    print(f"  C[{idx_max}]={c_logits[idx_max]:.6f} Rust[{idx_max}]={rust_logits[idx_max]:.6f}")
    if max_diff < eps:
        print("PASS")
    else:
        print("FAIL")


def main():
    print("=" * 60)
    print("2-token verification")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        logits_path = tmp.name

    c_logits = run_c(logits_path)
    if c_logits is None:
        sys.exit(1)
    os.unlink(logits_path)

    rust_logits = run_rust()
    compare(c_logits, rust_logits)


if __name__ == "__main__":
    main()
