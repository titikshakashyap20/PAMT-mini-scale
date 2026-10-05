"""
Stage 5B real-data verification -- OFFICIAL-CODE-FAITHFUL (Option A).

Pipeline:
    GENE Stage 2
        xP: (B,186,256)
              \
               -> RiskHead -> R: (B,1)
              /
    Stage 5A CA: (B,186,256)

Risk head:
    xP  -> AdaptiveAvgPool2d((186,1)) -> (B,186,1)
    CA  -> AdaptiveAvgPool2d((186,1)) -> (B,186,1)
    concat along token dimension -> (B,372)
    Linear(372,1) -> (B,1)

No LayerNorm, hidden MLP, activation, dropout, or extra projection.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Project imports
sys.path.insert(0, "src")

from gene_branch.gene_branch import GeneBranch
from wsi_branch.wsi_branch import WSIBranch
from wsi_branch.wsi_reduction import WSIReduction
from fusion import PathwayToPatchFusion
from scripts.risk_head import SurvivalRiskHead


# ============================================================
# REAL DATA
# ============================================================

GENE_DIR = Path("data/processed/gene")
WSI_DIR = Path("data/processed/wsi")


def match_patients(gene_dir: Path, wsi_dir: Path):
    """Match each gene file with its corresponding WSI file."""

    gene_files = sorted(gene_dir.glob("*.npy"))
    wsi_files = sorted(wsi_dir.glob("*_patches.npz"))

    pairs = []

    for gf in gene_files:
        patient_id = gf.stem

        matches = [
            wf for wf in wsi_files
            if wf.stem.startswith(patient_id + "-")
        ]

        if matches:
            pairs.append((patient_id, gf, matches[0]))

    return pairs


def load_15_patients():
    """
    Load all matched real patients.

    Returns:
        x_gene:
            (15,186,4942)

        x_wsi:
            (15,162,384), zero-padded

        patch_mask:
            (15,162), bool
            True = padding
            False = valid patch

        patient_ids:
            list of 15 patient IDs

        patch_counts:
            original number of patches for each patient
    """

    pairs = match_patients(GENE_DIR, WSI_DIR)

    if len(pairs) != 15:
        raise RuntimeError(
            f"Expected 15 matched patients, found {len(pairs)}."
        )

    gene_arrays = []
    patch_arrays = []
    patch_counts = []
    patient_ids = []

    for patient_id, gene_file, wsi_file in pairs:

        gene = np.load(gene_file)

        wsi_data = np.load(wsi_file)
        patches = wsi_data["embeddings"]

        gene_arrays.append(gene)
        patch_arrays.append(patches)
        patch_counts.append(patches.shape[0])
        patient_ids.append(patient_id)

    # -------------------------
    # Gene batch
    # -------------------------

    x_gene = torch.from_numpy(
        np.stack(gene_arrays, axis=0)
    ).float()

    # -------------------------
    # WSI padding
    # -------------------------

    B = len(patch_arrays)
    N_max = max(patch_counts)
    D = patch_arrays[0].shape[1]

    padded = np.zeros(
        (B, N_max, D),
        dtype=np.float32
    )

    patch_mask = np.ones(
        (B, N_max),
        dtype=bool
    )

    for i, patches in enumerate(patch_arrays):

        n = patches.shape[0]

        padded[i, :n, :] = patches
        patch_mask[i, :n] = False

    x_wsi = torch.from_numpy(padded).float()
    patch_mask = torch.from_numpy(patch_mask).bool()

    return (
        x_gene,
        x_wsi,
        patch_mask,
        patient_ids,
        patch_counts,
    )


# ============================================================
# FROZEN STAGES 1–5A
# ============================================================

def run_frozen_pipeline():

    (
        gene,
        wsi_384,
        key_padding_mask,
        patient_ids,
        patch_counts,
    ) = load_15_patients()

    print("=" * 60)
    print("STAGE 5B — SURVIVAL RISK HEAD")
    print("=" * 60)

    print("\nMatched patients:", len(patient_ids))

    print("\nPatient IDs:")
    print(patient_ids)

    print("\nGene batch shape:", tuple(gene.shape))
    print(
        "WSI batch shape:",
        tuple(wsi_384.shape),
        "(padded to max)"
    )

    print("Patch counts:", patch_counts)
    print(
        "Padding mask shape:",
        tuple(key_padding_mask.shape)
    )

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

    # --------------------------------------------------------
    # Stage 3 — WSI
    # --------------------------------------------------------

    wsi_branch = WSIBranch(
        embed_dim=384,
        max_patches=162,
        depth=6,
        num_heads=16,
    )

    xI_384 = wsi_branch(
        wsi_384,
        key_padding_mask=key_padding_mask,
    )

    print(
        "WSI Stage-3 output:",
        tuple(xI_384.shape),
        "(expect (15,162,384))"
    )

    # --------------------------------------------------------
    # Stage 3.5 — WSI reduction
    # --------------------------------------------------------

    wsi_reduction = WSIReduction()

    xI = wsi_reduction(
        xI_384,
        key_padding_mask=key_padding_mask,
    )

    # WSIReduction returns (tensor, mask)
    if isinstance(xI, tuple):
        xI, returned_mask = xI

        assert torch.equal(
            returned_mask,
            key_padding_mask
        ), "Stage 3.5 changed the padding mask."

    print(
        "WSI Stage-3.5 output:",
        tuple(xI.shape),
        "(expect (15,162,256))"
    )

    # --------------------------------------------------------
    # Stage 5A — fusion
    # --------------------------------------------------------

    fusion = PathwayToPatchFusion(
        dim=256,
        num_heads=16,
    )

    CA, attention = fusion(
        xP,
        xI,
        key_padding_mask,
    )

    print(
        "Stage-5A CA output:",
        tuple(CA.shape),
        "(expect (15,186,256))"
    )

    # Make sure gradients are enabled for the full chain.
    for module in (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
    ):
        for p in module.parameters():
            p.requires_grad_(True)

    return (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        xP,
        CA,
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        xP,
        CA,
    ) = run_frozen_pipeline()

    # ========================================================
    # 1. INPUT SHAPES
    # ========================================================

    print("\n--- Input shape checks ---")

    print(
        f"xP shape: {tuple(xP.shape)} "
        "(expected (15,186,256))"
    )

    print(
        f"CA shape: {tuple(CA.shape)} "
        "(expected (15,186,256))"
    )

    assert xP.shape == (15, 186, 256)
    assert CA.shape == (15, 186, 256)

    # ========================================================
    # 2. RISK HEAD
    # ========================================================

    risk_head = SurvivalRiskHead(
        num_pathway_tokens=186,
        num_classes=1,
    )

    for p in risk_head.parameters():
        p.requires_grad_(True)

    # ========================================================
    # 3. PARAMETER COUNT
    # ========================================================

    n_params = sum(
        p.numel()
        for p in risk_head.parameters()
    )

    print(
        f"\nRiskHead parameters: {n_params} "
        "(expected 373)"
    )

    assert n_params == 373, (
        f"Expected 373 parameters, "
        f"got {n_params}"
    )

    # ========================================================
    # 4. POOLED SHAPES
    # ========================================================

    gene_pooled = risk_head.gap_gene(xP)
    fusion_pooled = risk_head.gap_fusion(CA)

    print(
        "Pooled xP shape:",
        tuple(gene_pooled.shape),
        "(expected (15,186,1))"
    )

    print(
        "Pooled CA shape:",
        tuple(fusion_pooled.shape),
        "(expected (15,186,1))"
    )

    assert gene_pooled.shape == (15, 186, 1)
    assert fusion_pooled.shape == (15, 186, 1)

    # ========================================================
    # 5. CONCATENATION
    # ========================================================

    fused = torch.cat(
        [gene_pooled, fusion_pooled],
        dim=1,
    ).squeeze(-1)

    print(
        "Fused shape:",
        tuple(fused.shape),
        "(expected (15,372))"
    )

    assert fused.shape == (15, 372)

    # ========================================================
    # 6. RISK OUTPUT
    # ========================================================

    R = risk_head(xP, CA)

    print(
        "Risk R shape:",
        tuple(R.shape),
        "(expected (15,1))"
    )

    assert R.shape == (15, 1)

    # ========================================================
    # 7. FINITE OUTPUT
    # ========================================================

    finite = torch.isfinite(R).all().item()

    print(
        "Risk output finite:",
        finite
    )

    assert finite

    print(
        "Sample risk values:",
        R.detach().flatten()[:5].tolist()
    )

    # ========================================================
    # 8. GRADIENT CHECK
    # ========================================================

    print(
        "\n--- Gradient check: "
        "GENE -> WSI -> reduction -> fusion -> risk ---"
    )

    loss = R.sum()

    loss.backward()

    # --------------------------------------------------------
    # Risk head
    # --------------------------------------------------------

    rh_params = list(
        risk_head.parameters()
    )

    rh_nonzero = sum(
        1
        for p in rh_params
        if p.grad is not None
        and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
    )

    print(
        f"RiskHead: "
        f"{rh_nonzero}/{len(rh_params)} "
        "parameters have finite nonzero gradients"
    )

    assert rh_nonzero == len(rh_params)

    # --------------------------------------------------------
    # Upstream modules
    # --------------------------------------------------------

    for name, module in (
        ("GeneBranch", gene_branch),
        ("WSIBranch", wsi_branch),
        ("WSIReduction", wsi_reduction),
        ("Fusion", fusion),
    ):

        params = list(module.parameters())

        nonzero = sum(
            1
            for p in params
            if p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
        )

        print(
            f"{name}: "
            f"{nonzero}/{len(params)} "
            "parameters have finite nonzero gradients"
        )

        assert nonzero == len(params), (
            f"{name} has parameters "
            "with missing/zero/non-finite gradients"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 50)
    print("STAGE 5B REAL-DATA CHECK: PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()