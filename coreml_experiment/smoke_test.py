"""Tiny end-to-end CoreML pipeline smoke test.

Goal: validate that the toolchain (coremltools 9.0 + torch 2.7 on macOS 26
with M4 ANE) actually works before sinking time into HF model surgery.

Steps:
  1. Build a tiny PyTorch model — single Linear layer.
  2. Trace + convert to .mlpackage.
  3. Apply INT4 weight quantization.
  4. Load with each ComputeUnit (CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL).
  5. Run inference and confirm the answer is sensible.
  6. Report which devices each variant actually uses (via load_spec).
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
import coremltools.optimize.coreml as cto


class TinyMLP(nn.Module):
    """Two-layer MLP. Big enough to have INT4-quantizable Linear weights."""
    def __init__(self, dim_in: int = 256, dim_hidden: int = 2048, dim_out: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def main() -> None:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    example = torch.randn(1, 256, dtype=torch.float16)
    model = model.half()

    # ── Convert ────────────────────────────────────────────────
    print("[1/4] trace + convert ...")
    traced = torch.jit.trace(model, example, strict=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 256), dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    print("    ok")

    # ── Quantize ──────────────────────────────────────────────
    print("[2/4] linear-symmetric INT4 quantize ...")
    cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            granularity="per_block",
            block_size=64,
            weight_threshold=2048,
        ),
    )
    mlmodel_q = cto.linear_quantize_weights(mlmodel, config=cfg)
    print("    ok")

    # ── Save + reload per ComputeUnit ────────────────────────
    print("[3/4] save .mlpackage + reload per device ...")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tiny.mlpackage"
        mlmodel_q.save(str(path))

        x_np = example.float().numpy()
        # PyTorch reference
        with torch.no_grad():
            ref = model(example).float().numpy()

        results = {}
        for unit_name, unit in [
            ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
            ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
            ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
            ("ALL", ct.ComputeUnit.ALL),
        ]:
            try:
                m = ct.models.MLModel(str(path), compute_units=unit)
                # Warmup
                m.predict({"x": x_np})
                # Time
                N = 50
                t0 = time.perf_counter()
                for _ in range(N):
                    out = m.predict({"x": x_np})
                ms = (time.perf_counter() - t0) / N * 1000
                y_np = out["y"]
                rel = float(np.linalg.norm(y_np - ref) / max(np.linalg.norm(ref), 1e-6))
                results[unit_name] = (ms, rel)
                print(f"    {unit_name:<14} {ms:6.2f} ms/run  rel_err={rel:.4f}")
            except Exception as e:
                print(f"    {unit_name:<14} FAILED: {type(e).__name__}: {e}")
                results[unit_name] = None

    print("[4/4] summary")
    print(f"    PyTorch reference output norm: {float(np.linalg.norm(ref)):.4f}")
    for unit, r in results.items():
        if r is not None:
            print(f"    {unit:<14} {r[0]:6.2f} ms/run  rel_err={r[1]:.4f}")
    print("\nALL test passed if rel_err < ~0.05 across all units.")


if __name__ == "__main__":
    main()
