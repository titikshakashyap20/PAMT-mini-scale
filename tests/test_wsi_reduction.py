"""
Stage 3.5 — real-data verification for WSIReduction.
Runs the 15 real WSIs through the frozen WSIBranch (Stage 3), then
through WSIReduction (Stage 3.5), and checks: shape, NaN/Inf, param
count, gradient flow, and that padding still has zero effect on real
tokens after this new stage is added.
"""

import glob
import os

import numpy as np
import torch
import sys

sys.path.insert(0, "src")
from wsi_branch.wsi_branch import WSIBranch
from wsi_branch.wsi_reduction import WSIReduction

WSI_DIR = "data/processed/wsi"
MAX_PATCHES = 162
EMBED_DIM_IN = 384
EMBED_DIM_OUT = 256


def load_real_wsi_files():
    files = sorted(glob.glob(os.path.join(WSI_DIR, "*_patches.npz")))
    if not files:
        raise FileNotFoundError(f"No *_patches.npz files found in {WSI_DIR}")

    per_wsi = []
    for f in files:
        data = np.load(f)
        emb = data["embeddings"]
        assert emb.ndim == 2 and emb.shape[1] == EMBED_DIM_IN, (
            f"{f}: embeddings shape {emb.shape}, expected (N, {EMBED_DIM_IN})"
        )
        n = emb.shape[0]
        assert 94 <= n <= MAX_PATCHES, f"{f}: N={n} outside [94, {MAX_PATCHES}]"
        patient_id = os.path.basename(f).replace("_patches.npz", "")
        per_wsi.append((patient_id, torch.tensor(emb, dtype=torch.float32)))

    return per_wsi


def build_padded_batch(per_wsi, max_patches=MAX_PATCHES, embed_dim=EMBED_DIM_IN):
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


def run_shape_nan_grad_check(reduction, x_384, mask):
    reduction.train()
    out, mask_out = reduction(x_384, key_padding_mask=mask)

    print(f"  Output shape: {tuple(out.shape)}  (expect (15, 162, 256))")
    assert out.shape == (15, MAX_PATCHES, EMBED_DIM_OUT)

    print(f"  Mask unchanged: {torch.equal(mask_out, mask)}  (expect True)")
    assert torch.equal(mask_out, mask), "key_padding_mask was modified"

    ok_finite = not torch.isnan(out).any() and not torch.isinf(out).any()
    print(f"  No NaN/Inf in output: {'OK' if ok_finite else 'FAILED'}")

    total_params = sum(p.numel() for p in reduction.parameters())
    print(f"  Total parameters: {total_params}  (expect 410496)")

    loss = out.sum()
    loss.backward()
    n_total = sum(1 for _ in reduction.parameters())
    n_grad = sum(1 for p in reduction.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Params with nonzero gradient: {n_grad} / {n_total}")

    return ok_finite and total_params == 410496 and n_grad == n_total


def run_padding_invariance_check(wsi_branch, reduction, per_wsi, x_384_padded, mask):
    """Extends Stage 3's masking-correctness check one stage further:
    confirm that real-token outputs after WSIReduction are identical
    whether that WSI was run alone or inside the padded 15-WSI batch."""
    wsi_branch.eval()
    reduction.eval()

    with torch.no_grad():
        encoder_out_batched = wsi_branch(x_384_padded, key_padding_mask=mask)      # (15, 162, 384)
        reduced_batched, _ = reduction(encoder_out_batched, key_padding_mask=mask)  # (15, 162, 256)

        max_diffs = []
        for i, (pid, emb) in enumerate(per_wsi):
            n = emb.shape[0]
            encoder_out_alone = wsi_branch(emb.unsqueeze(0), key_padding_mask=None)  # (1, n, 384)
            reduced_alone, _ = reduction(encoder_out_alone, key_padding_mask=None)   # (1, n, 256)

            reduced_from_batch = reduced_batched[i, :n].unsqueeze(0)                 # (1, n, 256)
            diff = (reduced_alone - reduced_from_batch).abs().max().item()
            max_diffs.append((pid, n, diff))

    worst_pid, worst_n, worst = max(max_diffs, key=lambda t: t[2])
    print(f"  Worst-case per-WSI max abs diff (alone vs. padded+masked, post-reduction): "
          f"{worst:.8f}  (patient={worst_pid}, N={worst_n})")
    passed = worst < 1e-5
    print(f"  Padding invariance through Stage 3.5: {'PASSED' if passed else 'FAILED'}")
    return passed


def main():
    print("--- Loading real WSI data ---")
    per_wsi = load_real_wsi_files()
    print(f"Loaded {len(per_wsi)} WSIs from {WSI_DIR}")

    x_384, mask, patient_ids, patch_counts = build_padded_batch(per_wsi)
    print(f"Patch counts: min={min(patch_counts)}, max={max(patch_counts)}")
    print(f"Padded input shape: {tuple(x_384.shape)}  (expect (15, 162, 384))")

    wsi_branch = WSIBranch()          # frozen Stage 3, unmodified
    reduction = WSIReduction()        # new Stage 3.5

    print("\n--- Running Stage 3 (frozen) -> Stage 3.5 ---")
    with torch.no_grad():
        stage3_out = wsi_branch(x_384, key_padding_mask=mask)
    print(f"Stage 3 output shape: {tuple(stage3_out.shape)}  (expect (15, 162, 384))")

    print("\n--- Shape / NaN / param / gradient check (Stage 3.5) ---")
    ok1 = run_shape_nan_grad_check(reduction, stage3_out, mask)

    print("\n--- Padding invariance check (Stage 3 + 3.5 combined, eval mode) ---")
    ok2 = run_padding_invariance_check(wsi_branch, reduction, per_wsi, x_384, mask)

    print()
    print("Stage 3.5 real-data check: PASSED" if (ok1 and ok2) else "Stage 3.5 real-data check: CHECK OUTPUT ABOVE")


if __name__ == "__main__":
    main()