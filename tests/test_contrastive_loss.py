"""
Stage 4 real-data verification -- OFFICIAL-CODE-FAITHFUL version.

Stages 1-3.5 are reused exactly as already verified.

Official-code-faithful Stage 4:
    - raw dot-product similarity
    - top_h = 2
    - no learnable temperature / tau
    - padding excluded from top-k and softmax
    - binary Y_h labels
    - gradient flow through GeneBranch, WSIBranch, WSIReduction
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from gene_branch.gene_branch import GeneBranch
from gene_branch.contrastive_loss import PathwayPatchContrastiveLoss
from wsi_branch.wsi_branch import WSIBranch
from wsi_branch.wsi_reduction import WSIReduction


GENE_DIR = Path("data/processed/gene")
WSI_DIR = Path("data/processed/wsi")


# ============================================================
# REAL DATA LOADING
# ============================================================

def match_patients(gene_dir: Path, wsi_dir: Path):
    """
    Match each gene .npy file to its corresponding WSI .npz file.

    Gene:
        TCGA-XXXX.npy

    WSI:
        TCGA-XXXX-01Z-00-DX1.<UUID>_patches.npz
    """
    gene_files = sorted(gene_dir.glob("*.npy"))
    wsi_files = sorted(wsi_dir.glob("*_patches.npz"))

    pairs = []

    for gene_file in gene_files:
        patient_id = gene_file.stem

        matches = [
            wsi_file
            for wsi_file in wsi_files
            if wsi_file.stem.startswith(patient_id + "-")
        ]

        if matches:
            pairs.append(
                (patient_id, gene_file, matches[0])
            )

    return pairs


def load_15_patients():
    """
    Load all matched real patients.

    Returns
    -------
    gene:
        (15, 186, 4942)

    wsi:
        (15, 162, 384)
        padded to the maximum real WSI length

    key_padding_mask:
        (15, 162)
        True = padding
        False = real patch
    """

    pairs = match_patients(GENE_DIR, WSI_DIR)

    print(
        f"Matched {len(pairs)} gene<->WSI patient pairs "
        f"out of {len(list(GENE_DIR.glob('*.npy')))} gene files."
    )

    assert len(pairs) == 15, (
        f"Expected 15 matched patients, found {len(pairs)}"
    )

    gene_arrays = []
    wsi_arrays = []
    patch_counts = []
    patient_ids = []

    for patient_id, gene_file, wsi_file in pairs:

        gene_array = np.load(gene_file)

        wsi_data = np.load(wsi_file)

        assert "embeddings" in wsi_data, (
            f"{wsi_file} missing 'embeddings'"
        )

        wsi_embeddings = wsi_data["embeddings"]

        assert gene_array.shape == (186, 4942), (
            f"{patient_id}: gene shape {gene_array.shape}, "
            f"expected (186,4942)"
        )

        assert (
            wsi_embeddings.ndim == 2
            and wsi_embeddings.shape[1] == 384
        ), (
            f"{patient_id}: WSI shape {wsi_embeddings.shape}, "
            f"expected (N,384)"
        )

        assert 94 <= wsi_embeddings.shape[0] <= 162, (
            f"{patient_id}: patch count {wsi_embeddings.shape[0]} "
            f"outside expected range [94,162]"
        )

        print(
            f"  {patient_id} -> {wsi_file.name}"
        )

        gene_arrays.append(gene_array)
        wsi_arrays.append(wsi_embeddings)
        patch_counts.append(wsi_embeddings.shape[0])
        patient_ids.append(patient_id)

    # --------------------------------------------------------
    # Gene batch
    # --------------------------------------------------------

    gene = torch.from_numpy(
        np.stack(gene_arrays, axis=0)
    ).float()

    # --------------------------------------------------------
    # WSI padded batch
    # --------------------------------------------------------

    batch_size = len(wsi_arrays)
    max_patches = max(patch_counts)
    embed_dim = wsi_arrays[0].shape[1]

    padded_wsi = np.zeros(
        (batch_size, max_patches, embed_dim),
        dtype=np.float32,
    )

    key_padding_mask = np.ones(
        (batch_size, max_patches),
        dtype=bool,
    )

    for i, embeddings in enumerate(wsi_arrays):

        n = embeddings.shape[0]

        padded_wsi[i, :n, :] = embeddings

        # False = real patch
        key_padding_mask[i, :n] = False

    wsi = torch.from_numpy(padded_wsi).float()

    key_padding_mask = torch.from_numpy(
        key_padding_mask
    ).bool()

    print()
    print("Patient IDs:")
    print(patient_ids)

    print()
    print("Gene batch shape:", tuple(gene.shape))
    print(
        "WSI batch shape:",
        tuple(wsi.shape),
        "(padded to max)"
    )
    print("Patch counts:", patch_counts)

    assert gene.shape == (15, 186, 4942)
    assert wsi.shape == (15, 162, 384)
    assert key_padding_mask.shape == (15, 162)

    return gene, wsi, key_padding_mask, patch_counts


# ============================================================
# STAGES 1-3.5
# ============================================================

def run_frozen_pipeline():

    gene, wsi_384, key_padding_mask, patch_counts = (
        load_15_patients()
    )

    # --------------------------------------------------------
    # Already-verified modules
    # --------------------------------------------------------

    gene_branch = GeneBranch(
        in_features=4942,
        hidden_features=1000,
        embed_dim=256,
        num_pathways=186,
        depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.1,
    )

    wsi_branch = WSIBranch(
        embed_dim=384,
        max_patches=162,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.1,
    )

    wsi_reduction = WSIReduction(
        in_features=384,
        hidden_features=640,
        out_features=256,
        drop_rate=0.1,
    )

    # --------------------------------------------------------
    # GENE: Stage 1 + Stage 2
    # --------------------------------------------------------

    gene_branch.train()

    xP = gene_branch(gene)

    print()
    print(
        "GENE Stage-2 output:",
        tuple(xP.shape),
        "(expect (15,186,256))"
    )

    assert xP.shape == (15, 186, 256)

    # --------------------------------------------------------
    # WSI: Stage 3
    # --------------------------------------------------------

    wsi_branch.train()

    xI_384 = wsi_branch(
        wsi_384,
        key_padding_mask=key_padding_mask,
    )

    print(
        "WSI Stage-3 output:",
        tuple(xI_384.shape),
        "(expect (15,162,384))"
    )

    assert xI_384.shape == (15, 162, 384)

    # --------------------------------------------------------
    # WSI: Stage 3.5
    # --------------------------------------------------------

    wsi_reduction.train()

    reduction_result = wsi_reduction(
        xI_384,
        key_padding_mask=key_padding_mask,
    )

    # Your verified WSIReduction returns:
    #     xI, key_padding_mask

    if isinstance(reduction_result, tuple):

        xI, returned_mask = reduction_result

        assert returned_mask is key_padding_mask or torch.equal(
            returned_mask,
            key_padding_mask,
        )

    else:
        xI = reduction_result

    print(
        "WSI Stage-3.5 output:",
        tuple(xI.shape),
        "(expect (15,162,256))"
    )

    assert xI.shape == (15, 162, 256)

    print(
        "Patch mask preserved: True"
    )

    return (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        xP,
        xI,
        key_padding_mask,
        patch_counts,
    )


# ============================================================
# GRADIENT REPORT
# ============================================================

def gradient_report(module, name):

    params = list(module.parameters())

    nonzero = sum(
        1
        for p in params
        if (
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
        )
    )

    print(
        f"{name}: "
        f"{nonzero}/{len(params)} parameters "
        f"have finite nonzero gradients"
    )

    assert nonzero == len(params), (
        f"{name}: not all parameters have "
        f"finite nonzero gradients"
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        xP,
        xI,
        key_padding_mask,
        patch_counts,
    ) = run_frozen_pipeline()

    B, M, D = xP.shape
    _, N, _ = xI.shape

    # --------------------------------------------------------
    # Stage 4 loss
    # --------------------------------------------------------

    loss_fn = PathwayPatchContrastiveLoss(
        top_h=2
    )

    loss_fn.train()

    # --------------------------------------------------------
    # IMPORTANT:
    # Official-code-faithful version has NO learnable tau.
    # --------------------------------------------------------

    n_params = sum(
        1 for _ in loss_fn.parameters()
    )

    print()
    print(
        "PathwayPatchContrastiveLoss parameters:",
        n_params,
        "(expect 0)"
    )

    assert n_params == 0, (
        "Contrastive loss should have no "
        "learnable parameters in official-code-faithful version."
    )

    # --------------------------------------------------------
    # Run L3
    # --------------------------------------------------------

    loss, info = loss_fn(
        xP,
        xI,
        key_padding_mask,
    )

    S = info["S"]
    S_prime = info["S_prime"]
    Y_h = info["Y_h"]
    topk_idx = info["topk_idx"]

    print()
    print("--- Stage 4 / L3 ---")

    print(
        "L3 value:",
        loss.item()
    )

    print(
        "S shape:",
        tuple(S.shape),
        f"(expect (15,186,{N}))"
    )

    print(
        "S_prime shape:",
        tuple(S_prime.shape)
    )

    print(
        "Y_h shape:",
        tuple(Y_h.shape)
    )

    print(
        "topk_idx shape:",
        tuple(topk_idx.shape),
        "(expect (15,186,2))"
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert S.shape == (15, 186, N)
    assert S_prime.shape == (15, 186, N)
    assert Y_h.shape == (15, 186, N)
    assert topk_idx.shape == (15, 186, 2)

    # --------------------------------------------------------
    # Numerical checks
    # --------------------------------------------------------

    print()
    print("--- Numerical checks ---")

    print(
        "L3 finite:",
        torch.isfinite(loss).item()
    )

    print(
        "S finite:",
        torch.isfinite(S).all().item()
    )

    print(
        "S_prime finite:",
        torch.isfinite(S_prime).all().item()
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(S).all()
    assert torch.isfinite(S_prime).all()

    # --------------------------------------------------------
    # Padding checks
    # --------------------------------------------------------

    padding_expanded = (
        key_padding_mask
        .unsqueeze(1)
        .expand_as(S_prime)
    )

    padded_prob = S_prime[
        padding_expanded
    ]

    padded_yh = Y_h[
        padding_expanded
    ]

    max_padded_prob = (
        padded_prob.max().item()
        if padded_prob.numel()
        else 0.0
    )

    max_padded_yh = (
        padded_yh.max().item()
        if padded_yh.numel()
        else 0.0
    )

    print()
    print("--- Padding checks ---")

    print(
        "Maximum padded S_prime:",
        f"{max_padded_prob:.2e}",
        "(expect 0)"
    )

    print(
        "Maximum padded Y_h:",
        max_padded_yh,
        "(expect 0)"
    )

    assert max_padded_prob == 0.0
    assert max_padded_yh == 0.0

    # --------------------------------------------------------
    # Softmax normalization
    # --------------------------------------------------------

    row_sums = S_prime.sum(dim=-1)

    print()
    print("--- Softmax check ---")

    print(
        "Valid S_prime row sums:",
        row_sums.min().item(),
        "to",
        row_sums.max().item(),
    )

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-4,
    )

    print(
        "Softmax normalization: PASSED"
    )

    # --------------------------------------------------------
    # Top-h / Y_h checks
    # --------------------------------------------------------

    positives_per_row = Y_h.sum(dim=-1)

    print()
    print("--- Y_h / top-h check ---")

    print(
        "Y_h row sums:",
        positives_per_row.min().item(),
        "to",
        positives_per_row.max().item(),
        "(expect exactly 2.0)"
    )

    assert torch.all(
        positives_per_row == 2
    )

    print(
        "Exactly 2 positives per pathway: PASSED"
    )

    print(
        "Padding excluded from positives: PASSED"
    )

    # --------------------------------------------------------
    # Gradient check
    # --------------------------------------------------------

    print()
    print(
        "--- Gradient check: "
        "GENE -> WSI -> reduction -> L3 ---"
    )

    loss.backward()

    gradient_report(
        gene_branch,
        "GeneBranch"
    )

    gradient_report(
        wsi_branch,
        "WSIBranch"
    )

    gradient_report(
        wsi_reduction,
        "WSIReduction"
    )

    # Loss itself has no parameters.
    print(
        "ContrastiveLoss: "
        "0 parameters (expected 0)"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print(
        "STAGE 4 "
        "(OFFICIAL-CODE-FAITHFUL) "
        "REAL-DATA CHECK: PASSED"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()