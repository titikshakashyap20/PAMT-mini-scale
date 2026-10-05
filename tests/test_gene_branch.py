"""
Stage-2 verification script: GENE branch (pathway embedding + Transformer
encoder, depth=2) on real preprocessed data. Checks forward shape AND that
gradients actually flow back through every parameter (a shape-only check
can pass while the model is silently broken for training).

Usage (from project root):
    python test_gene_branch.py

Adjust sys.path.insert / import lines to match where you place these three
files in your project (e.g. src/gene_branch/).
"""

import glob
import os

import numpy as np
import torch
import sys

sys.path.insert(0, "src")
from gene_branch.gene_branch import GeneBranch

GENE_DIR = "data/processed/gene"
BATCH_SIZE = 15


def load_real_batch(batch_size=BATCH_SIZE):
    files = sorted(glob.glob(os.path.join(GENE_DIR, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {GENE_DIR}")
    files = files[:batch_size]

    arrs = []
    for f in files:
        a = np.load(f)
        assert a.shape == (186, 4942), f"{f} has shape {a.shape}, expected (186, 4942)"
        arrs.append(a)

    batch = np.stack(arrs, axis=0)  # (B, 186, 4942)
    return torch.tensor(batch, dtype=torch.float32), files


def run_check(x, label):
    print(f"--- {label} ---")
    print(f"Input shape:  {tuple(x.shape)}  (expect (B, 186, 4942))")

    model = GeneBranch()
    model.train()

    print(f"pos_gene_embed shape: {tuple(model.pos_gene_embed.shape)}  (expect (1, 186, 256))")

    out = model(x)
    print(f"Output shape: {tuple(out.shape)}  (expect (B, 186, 256))")
    assert out.shape == (x.shape[0], 186, 256), "Output shape mismatch"

    ok_finite = not torch.isnan(out).any() and not torch.isinf(out).any()
    print(f"No NaN/Inf in output: {'OK' if ok_finite else 'FAILED'}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}  (expect 6826392)")

    loss = out.sum()
    loss.backward()
    n_total = sum(1 for _ in model.parameters())
    n_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"Params with nonzero gradient: {n_grad} / {n_total}")

    passed = (
        out.shape == (x.shape[0], 186, 256)
        and ok_finite
        and total_params == 6826392
        and n_grad == n_total
    )
    print(f"{label}: {'PASSED' if passed else 'CHECK OUTPUT ABOVE'}\n")
    return passed


def main():
    x_synth = torch.randn(8, 186, 4942)
    run_check(x_synth, "Synthetic check")

    x_real, files = load_real_batch(BATCH_SIZE)
    print(f"Loaded {len(files)} patients from {GENE_DIR}")
    run_check(x_real, "Real data check")


if __name__ == "__main__":
    main()