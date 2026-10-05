"""
Stage 1 verification.

Run: python test_gene_embedding.py

What this checks:
  1. Synthetic data -- guaranteed to run regardless of your file layout.
     Confirms output shape (B, 186, 256), no NaN/Inf, and that every
     parameter receives a gradient.
  2. (Optional) Real data -- adjust GENE_DIR and the loader in
     load_real_batch() to your actual per-patient file layout, same as the
     Stage 4 test pattern. If GENE_DIR doesn't exist, this section is
     skipped with a note rather than failing.
"""

import os
import torch
import sys

sys.path.insert(0, "src")
from gene_branch.pathway_embedding import PathwayEmbedding

# ---- adjust to your layout ----
GENE_DIR = "gene_features"  # directory of per-patient (186, 4942) arrays
# --------------------------------


def synthetic_check():
    print("--- Synthetic check ---")
    torch.manual_seed(0)
    B = 8
    x = torch.randn(B, 186, 4942)

    model = PathwayEmbedding(in_features=4942, hidden_features=1000, out_features=256, drop=0.1)
    out = model(x)

    print(f"Input shape:  {tuple(x.shape)}  (expect (B, 186, 4942))")
    print(f"Output shape: {tuple(out.shape)}  (expect (B, 186, 256))")
    assert out.shape == (B, 186, 256), "Output shape mismatch"

    assert torch.isfinite(out).all(), "Non-finite values in output"
    print("No NaN/Inf in output: OK")

    total_params = sum(p.numel() for p in model.parameters())
    named = list(model.named_parameters())
    print(f"Total parameters: {total_params}  across {len(named)} tensors")
    expected = (4942 * 1000 + 1000) + (1000 * 256 + 256)
    print(f"Expected (fc1 + fc2, weights+biases): {expected}")
    assert total_params == expected, "Parameter count doesn't match expected architecture"

    out.sum().backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.any(p.grad != 0))
    print(f"Params with nonzero gradient: {n_grad} / {len(named)}")
    assert n_grad == len(named), "Some parameters did not receive gradients"

    print("Synthetic check: PASSED\n")


def load_real_batch():
    """
    Adjust this to your actual file format. Expected: a stack of per-patient
    arrays, each (186, 4942), matching the pathway-based gene expression
    matrix your preprocessing stage already produces.
    """
    import numpy as np

    files = sorted(f for f in os.listdir(GENE_DIR) if f.endswith(".npy"))
    arrays = [np.load(os.path.join(GENE_DIR, f)) for f in files]
    patient_ids = [f.replace(".npy", "") for f in files]
    batch = torch.tensor(np.stack(arrays), dtype=torch.float32)
    return batch, patient_ids


def real_data_check():
    print("--- Real data check ---")
    if not os.path.isdir(GENE_DIR):
        print(f"GENE_DIR '{GENE_DIR}' not found -- skipping real-data check.")
        print("Adjust GENE_DIR and load_real_batch() to your layout, then rerun.\n")
        return

    batch, patient_ids = load_real_batch()
    print(f"Loaded {len(patient_ids)} patients, batch shape {tuple(batch.shape)}")
    assert batch.shape[1:] == (186, 4942), (
        f"Expected per-patient shape (186, 4942), got {tuple(batch.shape[1:])}"
    )

    model = PathwayEmbedding()
    out = model(batch)
    print(f"Output shape: {tuple(out.shape)}  (expect ({len(patient_ids)}, 186, 256))")
    assert out.shape == (len(patient_ids), 186, 256)
    assert torch.isfinite(out).all()
    print("Real data check: PASSED\n")


if __name__ == "__main__":
    synthetic_check()
    real_data_check()