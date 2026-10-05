"""
Stage 5A — real-data verification for Pathway-to-Patch Fusion.

Official-code-faithful interpretation:

    xP  : (B, 186, 256)
    xI  : (B, N, 256)

    Qpath = FC(xP)
    Kpatch = FC(xI)
    Vpatch = FC(xI)

    CA = SoftMax(QK^T / sqrt(d)) V

    Output:
        CA = (B, 186, 256)

Padding convention:
    True  = padding
    False = valid patch

Stages 1–3.5 are reused and NOT modified.
"""

import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# IMPORTS
# ============================================================

sys.path.insert(0, "src")

from gene_branch.gene_branch import GeneBranch
from wsi_branch.wsi_branch import WSIBranch
from wsi_branch.wsi_reduction import WSIReduction
from fusion import PathwayToPatchFusion


GENE_DIR = Path("data/processed/gene")
WSI_DIR = Path("data/processed/wsi")


# ============================================================
# DATA LOADING
# ============================================================

def match_patients(gene_dir: Path, wsi_dir: Path):
    gene_files = sorted(gene_dir.glob("*.npy"))
    wsi_files = sorted(wsi_dir.glob("*_patches.npz"))

    pairs = []

    for gf in gene_files:
        patient_id = gf.stem

        matches = [
            wf
            for wf in wsi_files
            if wf.stem.startswith(patient_id + "-")
        ]

        if matches:
            pairs.append(
                (patient_id, gf, matches[0])
            )

    return pairs


def load_15_patients():
    """
    Load all 15 matched real patients.

    Returns:
        patient_ids
        x_gene             : (15, 186, 4942)
        x_wsi              : (15, 162, 384)
        key_padding_mask   : (15, 162), True = padding
        patch_counts
    """

    pairs = match_patients(
        GENE_DIR,
        WSI_DIR
    )

    print(
        f"Matched {len(pairs)} gene<->WSI patient pairs."
    )

    assert len(pairs) == 15, (
        f"Expected 15 matched patients, got {len(pairs)}"
    )

    gene_arrays = []
    wsi_arrays = []
    patch_counts = []
    patient_ids = []

    # --------------------------------------------------------
    # Load each matched patient
    # --------------------------------------------------------

    for patient_id, gene_file, wsi_file in pairs:

        gene = np.load(gene_file)

        wsi_data = np.load(wsi_file)
        embeddings = wsi_data["embeddings"]

        assert gene.shape == (
            186,
            4942
        ), (
            f"{patient_id}: unexpected gene shape "
            f"{gene.shape}"
        )

        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 384

        n = embeddings.shape[0]

        assert 94 <= n <= 162, (
            f"{patient_id}: N={n}, "
            f"expected range [94,162]"
        )

        patient_ids.append(patient_id)
        gene_arrays.append(gene)
        wsi_arrays.append(embeddings)
        patch_counts.append(n)

    # --------------------------------------------------------
    # Gene batch
    # --------------------------------------------------------

    x_gene = torch.from_numpy(
        np.stack(gene_arrays, axis=0)
    ).float()

    # --------------------------------------------------------
    # WSI padded batch
    # --------------------------------------------------------

    B = len(wsi_arrays)
    N_max = max(patch_counts)
    D = 384

    padded = np.zeros(
        (B, N_max, D),
        dtype=np.float32
    )

    key_padding_mask = np.ones(
        (B, N_max),
        dtype=bool
    )

    for i, embeddings in enumerate(wsi_arrays):

        n = embeddings.shape[0]

        padded[
            i,
            :n,
            :
        ] = embeddings

        key_padding_mask[
            i,
            :n
        ] = False

    x_wsi = torch.from_numpy(
        padded
    ).float()

    key_padding_mask = torch.from_numpy(
        key_padding_mask
    ).bool()

    print("\nPatient IDs:")
    print(patient_ids)

    print(
        "\nGene batch shape:",
        tuple(x_gene.shape)
    )

    print(
        "WSI batch shape:",
        tuple(x_wsi.shape),
        "(padded to max)"
    )

    print(
        "Patch counts:",
        patch_counts
    )

    print(
        "Padding mask shape:",
        tuple(key_padding_mask.shape)
    )

    return (
        patient_ids,
        x_gene,
        x_wsi,
        key_padding_mask,
        patch_counts,
    )


# ============================================================
# FROZEN STAGES 2 → 3 → 3.5
# ============================================================

def run_frozen_pipeline():

    (
        patient_ids,
        gene,
        wsi_384,
        key_padding_mask,
        patch_counts,
    ) = load_15_patients()

    # --------------------------------------------------------
    # Stage 2 — GENE
    # --------------------------------------------------------

    gene_branch = GeneBranch()

    xP = gene_branch(gene)

    print(
        "\nGENE Stage-2 output:",
        tuple(xP.shape),
        "(expect (15,186,256))"
    )

    assert xP.shape == (
        15,
        186,
        256
    )

    # --------------------------------------------------------
    # Stage 3 — WSI Transformer
    # --------------------------------------------------------

    wsi_branch = WSIBranch()

    xI_384 = wsi_branch(
        wsi_384,
        key_padding_mask=key_padding_mask
    )

    print(
        "WSI Stage-3 output:",
        tuple(xI_384.shape),
        "(expect (15,162,384))"
    )

    assert xI_384.shape == (
        15,
        162,
        384
    )

    # --------------------------------------------------------
    # Stage 3.5 — WSI reduction
    # --------------------------------------------------------

    wsi_reduction = WSIReduction()

    xI, returned_mask = wsi_reduction(
        xI_384,
        key_padding_mask
    )

    print(
        "WSI Stage-3.5 output:",
        tuple(xI.shape),
        "(expect (15,162,256))"
    )

    assert xI.shape == (
        15,
        162,
        256
    )

    print(
        "Patch mask preserved:",
        returned_mask is key_padding_mask
    )

    assert returned_mask is key_padding_mask

    return (
        patient_ids,
        gene,
        wsi_384,
        patch_counts,
        key_padding_mask,
        gene_branch,
        wsi_branch,
        wsi_reduction,
        xP,
        xI,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STAGE 5A — PATHWAY-TO-PATCH FUSION")
    print("=" * 60)

    (
        patient_ids,
        gene,
        wsi_384,
        patch_counts,
        key_padding_mask,
        gene_branch,
        wsi_branch,
        wsi_reduction,
        xP,
        xI,
    ) = run_frozen_pipeline()

    # ========================================================
    # FUSION MODULE
    # ========================================================

    print("\n--- Stage 5A / Fusion ---")

    fusion = PathwayToPatchFusion(
        dim=256,
        num_heads=16,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Run initial forward in eval mode so dropout cannot
    # interfere with the numerical/padding checks.
    # --------------------------------------------------------

    gene_branch.eval()
    wsi_branch.eval()
    wsi_reduction.eval()
    fusion.eval()

    with torch.no_grad():

        CA, attn = fusion(
            xP,
            xI,
            key_padding_mask
        )

    # ========================================================
    # SHAPE CHECK
    # ========================================================

    B, M, D = xP.shape
    N = xI.shape[1]

    print(
        f"CA shape: {tuple(CA.shape)} "
        f"(expected (15,186,256))"
    )

    print(
        f"attn shape: {tuple(attn.shape)} "
        f"(expected (15,16,186,{N}))"
    )

    assert CA.shape == (
        15,
        186,
        256
    )

    assert attn.shape == (
        15,
        16,
        186,
        N
    )

    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    n_params = sum(
        p.numel()
        for p in fusion.parameters()
    )

    print(
        "\nFusion parameters:",
        n_params
    )

    assert n_params > 0

    # ========================================================
    # FINITE CHECK
    # ========================================================

    print("\n--- Numerical checks ---")

    CA_finite = torch.isfinite(CA).all()
    attn_finite = torch.isfinite(attn).all()

    print(
        "CA finite:",
        bool(CA_finite)
    )

    print(
        "Attention finite:",
        bool(attn_finite)
    )

    assert CA_finite
    assert attn_finite

    # ========================================================
    # ATTENTION NORMALIZATION
    # ========================================================

    print("\n--- Attention normalization ---")

    row_sums = attn.sum(dim=-1)

    print(
        "Attention row sums:",
        row_sums.min().item(),
        "to",
        row_sums.max().item()
    )

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-4
    )

    print(
        "Attention normalization: PASSED"
    )

    # ========================================================
    # PADDING ATTENTION
    # ========================================================

    print("\n--- Padding attention check ---")

    padding_expanded = (
        key_padding_mask
        [:, None, None, :]
        .expand_as(attn)
    )

    padded_attn = attn[
        padding_expanded
    ]

    max_padded = (
        padded_attn.max().item()
        if padded_attn.numel()
        else 0.0
    )

    print(
        "Maximum padded attention:",
        f"{max_padded:.2e}",
        "(expect 0)"
    )

    assert max_padded == 0.0

    print(
        "Padding excluded from attention: PASSED"
    )

    # ========================================================
    # GRADIENT CHECK
    # ========================================================

    print(
        "\n--- Gradient check: "
        "GENE -> WSI -> reduction -> fusion ---"
    )

    # Need a fresh forward with gradients enabled.
    gene_branch.train()
    wsi_branch.train()
    wsi_reduction.train()
    fusion.train()

    # Clear any old gradients.
    for module in (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
    ):
        module.zero_grad(set_to_none=True)

    xP_grad = gene_branch(gene)

    xI_384_grad = wsi_branch(
        wsi_384,
        key_padding_mask
    )

    xI_grad, _ = wsi_reduction(
        xI_384_grad,
        key_padding_mask
    )

    CA_grad, _ = fusion(
        xP_grad,
        xI_grad,
        key_padding_mask
    )

    loss = CA_grad.sum()

    assert torch.isfinite(loss)

    loss.backward()

    for name, module in (
        ("GeneBranch", gene_branch),
        ("WSIBranch", wsi_branch),
        ("WSIReduction", wsi_reduction),
        ("Fusion", fusion),
    ):

        params = list(
            module.parameters()
        )

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
            f"{nonzero}/{len(params)} "
            f"parameters have finite nonzero gradients"
        )

        assert nonzero == len(params), (
            f"{name} has missing, zero, "
            f"or non-finite gradients"
        )

    # ========================================================
    # PADDING INVARIANCE
    # ========================================================

    print(
        "\n--- Padding invariance check ---"
    )

    # Choose a patient that actually contains padding.
    idx = next(
        i
        for i, n in enumerate(patch_counts)
        if n < N
    )

    n_valid = patch_counts[idx]

    # --------------------------------------------------------
    # Run the selected patient ALONE.
    # --------------------------------------------------------

    gene_single = gene[
        idx:idx + 1
    ]

    wsi_single = wsi_384[
        idx,
        :n_valid,
    ].unsqueeze(0)

    mask_single = torch.zeros(
        1,
        n_valid,
        dtype=torch.bool
    )

    # --------------------------------------------------------
    # Everything in eval mode to eliminate dropout noise.
    # --------------------------------------------------------

    gene_branch.eval()
    wsi_branch.eval()
    wsi_reduction.eval()
    fusion.eval()

    with torch.no_grad():

        xP_single = gene_branch(
            gene_single
        )

        xI_384_single = wsi_branch(
            wsi_single,
            mask_single
        )

        xI_single, _ = wsi_reduction(
            xI_384_single,
            mask_single
        )

        CA_alone, _ = fusion(
            xP_single,
            xI_single,
            mask_single
        )

        # ----------------------------------------------------
        # Compare against the same patient inside padded batch
        # ----------------------------------------------------

        xP_batch = gene_branch(
            gene[idx:idx + 1]
        )

        xI_384_batch = wsi_branch(
            wsi_384[idx:idx + 1],
            key_padding_mask[idx:idx + 1]
        )

        xI_batch, _ = wsi_reduction(
            xI_384_batch,
            key_padding_mask[idx:idx + 1]
        )

        CA_batch, _ = fusion(
            xP_batch,
            xI_batch,
            key_padding_mask[idx:idx + 1]
        )

        max_diff = (
            CA_alone - CA_batch
        ).abs().max().item()

    print(
        f"Patient: {patient_ids[idx]}"
    )

    print(
        f"Valid patches: {n_valid}"
    )

    print(
        "Padding invariance max abs diff:",
        f"{max_diff:.8f}"
    )

    assert max_diff < 1e-5

    print(
        "Padding invariance: PASSED"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("STAGE 5A REAL-DATA CHECK: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()