"""
Stage 3 — real-data verification for WSIBranch.
Loads all 15 real WSI patch files, pads to the batch max, and checks:
shape, NaN/Inf, param count, gradient flow, and — the substantive check —
that padding a WSI into a larger batch never changes that WSI's own
real-token outputs versus running it alone.
"""

import glob
import os

import numpy as np
import torch
import sys

sys.path.insert(0, "src")
from wsi_branch.wsi_branch import WSIBranch

WSI_DIR = "data/processed/wsi"
MAX_PATCHES = 162
EMBED_DIM = 384


def load_real_wsi_files():
    files = sorted(glob.glob(os.path.join(WSI_DIR, "*_patches.npz")))
    if not files:
        raise FileNotFoundError(f"No *_patches.npz files found in {WSI_DIR}")

    per_wsi = []
    for f in files:
        data = np.load(f)
        for key in ("embeddings", "cluster_labels", "coords"):
            assert key in data, f"{f} missing key '{key}'"

        emb = data["embeddings"]
        assert emb.ndim == 2 and emb.shape[1] == EMBED_DIM, (
            f"{f}: embeddings shape {emb.shape}, expected (N, {EMBED_DIM})"
        )
        n = emb.shape[0]
        assert 94 <= n <= MAX_PATCHES, (
            f"{f}: N={n} patches, outside expected range [94, {MAX_PATCHES}]"
        )

        patient_id = os.path.basename(f).replace("_patches.npz", "")
        per_wsi.append((patient_id, torch.tensor(emb, dtype=torch.float32)))

    return per_wsi


def build_padded_batch(per_wsi, max_patches=MAX_PATCHES, embed_dim=EMBED_DIM):
    B = len(per_wsi)
    x = torch.zeros(B, max_patches, embed_dim)
    mask = torch.ones(B, max_patches, dtype=torch.bool)  # True = padding

    patient_ids, patch_counts = [], []
    for i, (pid, emb) in enumerate(per_wsi):
        n = emb.shape[0]
        x[i, :n] = emb
        mask[i, :n] = False
        patient_ids.append(pid)
        patch_counts.append(n)

    return x, mask, patient_ids, patch_counts


def run_gradient_and_shape_check(model, x, mask):
    model.train()
    out = model(x, key_padding_mask=mask)
    print(f"  Output shape: {tuple(out.shape)}  (expect (15, 162, 384))")
    assert out.shape == (15, MAX_PATCHES, EMBED_DIM)

    ok_finite = not torch.isnan(out).any() and not torch.isinf(out).any()
    print(f"  No NaN/Inf in output: {'OK' if ok_finite else 'FAILED'}")

    loss = out.sum()
    loss.backward()
    n_total = sum(1 for _ in model.parameters())
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Params with nonzero gradient: {n_grad} / {n_total}")

    return ok_finite and n_grad == n_total


def run_masking_correctness_check(model, per_wsi, x, mask):
    """For every one of the 15 real WSIs: compare its real-token outputs
    inside the padded batch against running that same WSI alone, unpadded.
    Padding/masking must have zero effect on real tokens."""
    model.eval()
    with torch.no_grad():
        out_batched = model(x, key_padding_mask=mask)  # (15, 162, 384)

        max_diffs = []
        for i, (pid, emb) in enumerate(per_wsi):
            n = emb.shape[0]
            out_alone = model(emb.unsqueeze(0), key_padding_mask=None)  # (1, n, 384)
            out_from_batch = out_batched[i, :n].unsqueeze(0)            # (1, n, 384)
            diff = (out_alone - out_from_batch).abs().max().item()
            max_diffs.append((pid, n, diff))

    worst_pid, worst_n, worst = max(max_diffs, key=lambda t: t[2])
    print(f"  Worst-case per-WSI max abs diff: {worst:.8f}  (patient={worst_pid}, N={worst_n})")
    passed = worst < 1e-5
    print(f"  Masking correctness across all 15 real WSIs: {'PASSED' if passed else 'FAILED'}")
    return passed


def main():
    print("--- Loading real WSI data ---")
    per_wsi = load_real_wsi_files()
    print(f"Loaded {len(per_wsi)} WSIs from {WSI_DIR}")

    x, mask, patient_ids, patch_counts = build_padded_batch(per_wsi)
    print(f"Patch counts: min={min(patch_counts)}, max={max(patch_counts)}  (expect within [94, 162])")
    print(f"Padded batch shape: {tuple(x.shape)}  (expect (15, 162, 384))")
    print(f"Mask shape: {tuple(mask.shape)}  (expect (15, 162))")

    model = WSIBranch()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}  (expect 10708992)")

    print("\n--- Gradient / shape / NaN check (train mode) ---")
    ok1 = run_gradient_and_shape_check(model, x, mask)

    print("\n--- Masking correctness check (eval mode, all 15 real WSIs) ---")
    ok2 = run_masking_correctness_check(model, per_wsi, x, mask)

    print()
    all_passed = ok1 and ok2 and total_params == 10708992
    print("Stage 3 real-data check: PASSED" if all_passed else "Stage 3 real-data check: CHECK OUTPUT ABOVE")


if __name__ == "__main__":
    main()