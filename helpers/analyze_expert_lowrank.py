#!/usr/bin/env python3
"""analyze_expert_lowrank.py — SVD-based low-rank feasibility study for MoE experts.

Reads a single layer's packed_experts/layer_XX.bin, dequantizes each expert's
gate/up/down matrices to float32, and answers:

    For a target variance-keep ratio alpha in [0, 1], what rank does each
    matrix need, what is the storage/bandwidth saving, and how much does
    the matvec output drift?

Variance keep: smallest r such that sum(sigma[:r]**2) / sum(sigma**2) >= alpha.

Usage:
    python helpers/analyze_expert_lowrank.py \
        --model data/Qwen3.6-35B-A3B/model_bq4 \
        --layer 0 --variance 0.95
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np


GROUP_SIZE = 64


# ─── Fast vectorised dequant ───────────────────────────────────────────────

def bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def dequant_int4_fast(
    packed: np.ndarray, scales_u16: np.ndarray, biases_u16: np.ndarray,
    out_dim: int, in_dim: int,
) -> np.ndarray:
    """Vectorised INT4 dequant. Returns float32 [out_dim, in_dim]."""
    num_groups = in_dim // GROUP_SIZE
    scales = bf16_u16_to_f32(scales_u16).reshape(out_dim, num_groups)
    biases = bf16_u16_to_f32(biases_u16).reshape(out_dim, num_groups)
    shifts = (np.arange(8, dtype=np.uint32) * 4)
    nibbles = ((packed[:, None] >> shifts) & np.uint32(0xF)).astype(np.float32)
    nibbles = nibbles.reshape(out_dim, num_groups, GROUP_SIZE)
    return (nibbles * scales[:, :, None] + biases[:, :, None]).reshape(out_dim, in_dim)


# ─── Expert layout (copied from moe_infer.dequantize) ──────────────────────

def expert_layout(hd: int, mi: int) -> dict:
    gs = GROUP_SIZE
    gate_w = mi * hd // 2
    gate_sb = mi * (hd // gs) * 2
    up_w = mi * hd // 2
    up_sb = mi * (hd // gs) * 2
    down_w = hd * mi // 2
    down_sb = hd * (mi // gs) * 2
    return {
        "gate_w": 0,
        "gate_s": gate_w,
        "gate_b": gate_w + gate_sb,
        "up_w": gate_w + 2 * gate_sb,
        "up_s": gate_w + 2 * gate_sb + up_w,
        "up_b": gate_w + 2 * gate_sb + up_w + up_sb,
        "down_w": gate_w + 2 * gate_sb + up_w + 2 * up_sb,
        "down_s": gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w,
        "down_b": gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + down_sb,
        "size": gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + 2 * down_sb,
    }


def read_section(buf: memoryview, off: int, out_dim: int, in_dim: int) -> np.ndarray:
    num_groups = in_dim // GROUP_SIZE
    w_bytes = out_dim * (in_dim // 8) * 4
    sb_bytes = out_dim * num_groups * 2
    packed = np.frombuffer(buf[off:off + w_bytes], dtype=np.uint32)
    scales = np.frombuffer(buf[off + w_bytes:off + w_bytes + sb_bytes], dtype=np.uint16)
    biases = np.frombuffer(buf[off + w_bytes + sb_bytes:off + w_bytes + 2 * sb_bytes], dtype=np.uint16)
    return dequant_int4_fast(packed, scales, biases, out_dim, in_dim)


# ─── SVD analysis ──────────────────────────────────────────────────────────

def rank_for_variance(sigma: np.ndarray, alpha: float) -> int:
    energy = (sigma ** 2).cumsum()
    total = energy[-1]
    r = int(np.searchsorted(energy, alpha * total) + 1)
    return min(r, len(sigma))


def lowrank_reconstruction(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, r: int) -> np.ndarray:
    return (U[:, :r] * S[:r]) @ Vt[:r, :]


def output_cosine(W: np.ndarray, W_lr: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-sample cosine sim between W @ x and W_lr @ x."""
    Y = W @ X       # [out_dim, num_samples]
    Y_lr = W_lr @ X
    num = (Y * Y_lr).sum(axis=0)
    den = np.linalg.norm(Y, axis=0) * np.linalg.norm(Y_lr, axis=0) + 1e-12
    return num / den


# ─── Storage accounting ────────────────────────────────────────────────────

def int4_bytes(out_dim: int, in_dim: int) -> int:
    """INT4 packed + BF16 scales/biases at GROUP_SIZE=64."""
    w = out_dim * in_dim // 2
    sb = 2 * out_dim * (in_dim // GROUP_SIZE) * 2  # scales + biases, BF16
    return w + sb


def lowrank_int4_bytes(out_dim: int, in_dim: int, r: int) -> int:
    """U is (out_dim, r), V is (r, in_dim). Both INT4-quantised along in_dim.
    For SVD ranks not divisible by 64 we round in_dim of U up to next group."""
    r_padded = ((r + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    return int4_bytes(out_dim, r_padded) + int4_bytes(r, in_dim)


# ─── Main ──────────────────────────────────────────────────────────────────

def analyze_one_matrix(W: np.ndarray, name: str, alpha: float, X: np.ndarray) -> dict:
    """SVD W, pick rank for variance alpha, report fidelity + storage."""
    out_dim, in_dim = W.shape
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    r = rank_for_variance(S, alpha)
    W_lr = lowrank_reconstruction(U, S, Vt, r)

    frob = np.linalg.norm(W - W_lr) / (np.linalg.norm(W) + 1e-12)
    cos = output_cosine(W, W_lr, X)
    orig_bytes = int4_bytes(out_dim, in_dim)
    lr_bytes = lowrank_int4_bytes(out_dim, in_dim, r)
    ratio = lr_bytes / orig_bytes

    # Also tabulate rank at standard variance targets for context
    targets = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    rank_table = {a: rank_for_variance(S, a) for a in targets}

    return {
        "name": name,
        "shape": (out_dim, in_dim),
        "max_rank": min(out_dim, in_dim),
        "rank": r,
        "rank_table": rank_table,
        "frob_rel_err": float(frob),
        "cos_mean": float(cos.mean()),
        "cos_p1": float(np.percentile(cos, 1)),
        "cos_min": float(cos.min()),
        "orig_bytes": orig_bytes,
        "lr_bytes": lr_bytes,
        "size_ratio": ratio,
        "sigma_top": S[:5].tolist(),
        "sigma_tail": S[-5:].tolist(),
    }


def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def analyze_layer(layer_idx: int, model_path: Path, layout: dict,
                  hd: int, mi: int, num_experts: int,
                  alpha: float, X_hd: np.ndarray, X_mi: np.ndarray) -> list[dict]:
    layer_path = model_path / "packed_experts" / f"layer_{layer_idx:02d}.bin"
    if not layer_path.exists():
        raise SystemExit(f"missing {layer_path}")
    with open(layer_path, "rb") as f:
        buf = memoryview(f.read())

    per_expert = []
    for e in range(num_experts):
        base = e * layout["size"]
        gate_W = read_section(buf, base + layout["gate_w"], mi, hd)
        up_W   = read_section(buf, base + layout["up_w"],   mi, hd)
        down_W = read_section(buf, base + layout["down_w"], hd, mi)
        per_expert.append({
            "expert": e,
            "gate": analyze_one_matrix(gate_W, "gate", alpha, X_hd),
            "up":   analyze_one_matrix(up_W,   "up",   alpha, X_hd),
            "down": analyze_one_matrix(down_W, "down", alpha, X_mi),
        })
    return per_expert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="BQ4 model dir (contains packed_experts/)")
    ap.add_argument("--layer", type=int, default=0,
                    help="Single layer to analyse (ignored if --all-layers)")
    ap.add_argument("--all-layers", action="store_true",
                    help="Sweep every layer file and print a compact summary table")
    ap.add_argument("--variance", type=float, default=0.95,
                    help="Target variance-keep ratio in [0, 1]")
    ap.add_argument("--num-experts", type=int, default=None,
                    help="Limit to first N experts (default: all)")
    ap.add_argument("--num-samples", type=int, default=256,
                    help="Random activations for output-cosine probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=str, default=None,
                    help="Optional: write per-expert results as JSON")
    args = ap.parse_args()

    model_path = Path(args.model)
    with open(model_path / "model_weights.json") as f:
        cfg = json.load(f)["config"]

    hd = cfg["hidden_size"]
    mi = cfg["moe_intermediate_size"]
    total_layers = cfg["num_hidden_layers"]
    num_experts = cfg["num_experts"]
    if args.num_experts is not None:
        num_experts = min(num_experts, args.num_experts)

    print(f"Model:    {model_path}")
    print(f"Hidden:   {hd}    MoE inter: {mi}    Total layers: {total_layers}")
    print(f"Experts per layer: {num_experts}    Target variance keep: {args.variance:.3f}")
    print()

    layout = expert_layout(hd, mi)
    rng = np.random.default_rng(args.seed)
    X_hd = rng.standard_normal((hd, args.num_samples)).astype(np.float32)
    X_mi = rng.standard_normal((mi, args.num_samples)).astype(np.float32)

    if args.all_layers:
        layer_ids = list(range(total_layers))
        # Per-matrix sub-headers, then one compact row per layer.
        print(f"  {'layer':>5s}  "
              f"{'gate rank':>10s} {'cos':>7s} {'ratio':>7s}   "
              f"{'up rank':>10s} {'cos':>7s} {'ratio':>7s}   "
              f"{'down rank':>10s} {'cos':>7s} {'ratio':>7s}   "
              f"{'per-expert':>10s}")
        all_layer_summaries = []
        t0 = time.time()
        for li in layer_ids:
            per_expert = analyze_layer(li, model_path, layout, hd, mi,
                                       num_experts, args.variance, X_hd, X_mi)
            summary = {"layer": li}
            for name in ("gate", "up", "down"):
                ranks = np.array([pe[name]["rank"] for pe in per_expert])
                coses = np.array([pe[name]["cos_mean"] for pe in per_expert])
                ratios = np.array([pe[name]["size_ratio"] for pe in per_expert])
                summary[name] = {
                    "rank_mean": float(ranks.mean()),
                    "rank_max": float(ranks.max()),
                    "max_rank": per_expert[0][name]["max_rank"],
                    "cos_mean": float(coses.mean()),
                    "ratio_mean": float(ratios.mean()),
                }
            orig_per_expert = sum(
                int4_bytes(*per_expert[0][m]["shape"]) for m in ("gate", "up", "down")
            )
            lr_per_expert = np.array([
                sum(pe[m]["lr_bytes"] for m in ("gate", "up", "down"))
                for pe in per_expert
            ])
            summary["per_expert_ratio"] = float(lr_per_expert.mean() / orig_per_expert)
            all_layer_summaries.append(summary)

            print(f"  {li:5d}  "
                  f"{summary['gate']['rank_mean']:6.1f}/{summary['gate']['max_rank']:<3d} "
                  f"{summary['gate']['cos_mean']:7.4f} {summary['gate']['ratio_mean']:7.3f}   "
                  f"{summary['up']['rank_mean']:6.1f}/{summary['up']['max_rank']:<3d} "
                  f"{summary['up']['cos_mean']:7.4f} {summary['up']['ratio_mean']:7.3f}   "
                  f"{summary['down']['rank_mean']:6.1f}/{summary['down']['max_rank']:<3d} "
                  f"{summary['down']['cos_mean']:7.4f} {summary['down']['ratio_mean']:7.3f}   "
                  f"{summary['per_expert_ratio']:10.3f}")

        elapsed = time.time() - t0
        print()
        print(f"  swept {len(layer_ids)} layers in {elapsed:.1f}s")
        # Final aggregate across layers
        per_layer_ratio = np.array([s["per_expert_ratio"] for s in all_layer_summaries])
        per_layer_cos_min = np.array([
            min(s[m]["cos_mean"] for m in ("gate", "up", "down"))
            for s in all_layer_summaries
        ])
        winners = (per_layer_ratio < 1.0).sum()
        losers = (per_layer_ratio >= 1.0).sum()
        print(f"  layers where low-rank saves bytes: {winners}/{len(layer_ids)}")
        print(f"  layers where it adds overhead:     {losers}/{len(layer_ids)}")
        print(f"  per-expert bandwidth ratio: mean {per_layer_ratio.mean():.3f}  "
              f"min {per_layer_ratio.min():.3f}  max {per_layer_ratio.max():.3f}")
        print(f"  min per-matrix cosine across layers: {per_layer_cos_min.min():.4f}")

        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({
                    "config": {
                        "model": str(model_path), "variance": args.variance,
                        "num_experts_sampled": num_experts, "hidden_size": hd,
                        "moe_intermediate_size": mi, "num_samples": args.num_samples,
                    },
                    "layers": all_layer_summaries,
                }, f, indent=2)
            print(f"  wrote {args.json_out}")
        return

    # Single-layer mode (original behaviour)
    print(f"Layer: {args.layer}")
    per_expert = []
    t0 = time.time()
    layer_path = model_path / "packed_experts" / f"layer_{args.layer:02d}.bin"
    if not layer_path.exists():
        raise SystemExit(f"missing {layer_path}")
    with open(layer_path, "rb") as f:
        buf = memoryview(f.read())

    for e in range(num_experts):
        base = e * layout["size"]
        gate_W = read_section(buf, base + layout["gate_w"], mi, hd)
        up_W   = read_section(buf, base + layout["up_w"],   mi, hd)
        down_W = read_section(buf, base + layout["down_w"], hd, mi)

        gate_r = analyze_one_matrix(gate_W, "gate", args.variance, X_hd)
        up_r   = analyze_one_matrix(up_W,   "up",   args.variance, X_hd)
        down_r = analyze_one_matrix(down_W, "down", args.variance, X_mi)
        per_expert.append({"expert": e, "gate": gate_r, "up": up_r, "down": down_r})

        if e < 3 or (e + 1) % 32 == 0 or e == num_experts - 1:
            elapsed = time.time() - t0
            rate = (e + 1) / elapsed
            print(f"  expert {e:3d}: "
                  f"gate r={gate_r['rank']:3d} cos={gate_r['cos_mean']:.4f}  "
                  f"up r={up_r['rank']:3d} cos={up_r['cos_mean']:.4f}  "
                  f"down r={down_r['rank']:3d} cos={down_r['cos_mean']:.4f}  "
                  f"[{rate:.1f} expert/s]")

    print()
    print("=" * 78)
    print(f"AGGREGATE (layer {args.layer}, variance={args.variance:.3f}, "
          f"N={num_experts} experts)")
    print("=" * 78)

    def agg(matrix_name: str):
        ranks = np.array([pe[matrix_name]["rank"] for pe in per_expert])
        cos_means = np.array([pe[matrix_name]["cos_mean"] for pe in per_expert])
        cos_p1s = np.array([pe[matrix_name]["cos_p1"] for pe in per_expert])
        cos_mins = np.array([pe[matrix_name]["cos_min"] for pe in per_expert])
        frobs = np.array([pe[matrix_name]["frob_rel_err"] for pe in per_expert])
        ratios = np.array([pe[matrix_name]["size_ratio"] for pe in per_expert])
        shape = per_expert[0][matrix_name]["shape"]
        max_rank = per_expert[0][matrix_name]["max_rank"]

        print(f"\n  {matrix_name}_proj  shape={shape[0]}x{shape[1]}  (max rank={max_rank})")
        print(f"    rank:          mean={ranks.mean():6.1f}  std={ranks.std():5.1f}  "
              f"min={ranks.min()}  max={ranks.max()}  "
              f"(mean/max = {fmt_pct(ranks.mean() / max_rank)})")
        print(f"    output cos:    mean={cos_means.mean():.4f}  "
              f"worst-expert-mean={cos_means.min():.4f}  "
              f"p1={cos_p1s.mean():.4f}  worst-sample={cos_mins.min():.4f}")
        print(f"    frob rel err:  mean={frobs.mean():.4f}  max={frobs.max():.4f}")
        print(f"    bytes ratio:   mean={ratios.mean():.3f}  "
              f"(saves {fmt_pct(1 - ratios.mean())})")

        # Rank table at standard variance targets (averaged across experts)
        targets = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
        rt = np.array([[pe[matrix_name]["rank_table"][a] for a in targets]
                       for pe in per_expert])
        print(f"    rank for variance keep:")
        for i, a in enumerate(targets):
            print(f"      {a:.3f} -> mean rank {rt[:, i].mean():6.1f}  "
                  f"(max {rt[:, i].max()})  "
                  f"= {fmt_pct(rt[:, i].mean() / max_rank)} of max")

    for name in ("gate", "up", "down"):
        agg(name)

    # Overall per-expert bandwidth saving
    orig_per_expert = sum(
        int4_bytes(*per_expert[0][m]["shape"]) for m in ("gate", "up", "down")
    )
    lr_per_expert = np.array([
        sum(pe[m]["lr_bytes"] for m in ("gate", "up", "down"))
        for pe in per_expert
    ])
    print()
    print(f"  Per-expert total (gate+up+down):")
    print(f"    INT4 baseline:  {orig_per_expert/1024:.1f} KiB")
    print(f"    INT4 low-rank:  mean {lr_per_expert.mean()/1024:.1f} KiB "
          f"(min {lr_per_expert.min()/1024:.1f}, max {lr_per_expert.max()/1024:.1f})")
    print(f"    Bandwidth ratio: mean {lr_per_expert.mean()/orig_per_expert:.3f}  "
          f"({orig_per_expert / lr_per_expert.mean():.2f}x speedup if bandwidth-bound)")

    print()
    print("NOTE: output-cosine uses isotropic Gaussian inputs — a proxy for real")
    print("      activations. Real activations are not isotropic; a more rigorous")
    print("      test would forward real text through the model and probe with")
    print("      actual hidden states. Treat these numbers as upper bounds on cos.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "config": {
                    "model": str(model_path), "layer": args.layer,
                    "variance": args.variance, "num_experts": num_experts,
                    "hidden_size": hd, "moe_intermediate_size": mi,
                    "num_samples": args.num_samples,
                },
                "per_expert": per_expert,
            }, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
