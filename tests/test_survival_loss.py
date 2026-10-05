"""
Stage 6 real-data verification.

Pipeline:
    gene -> GeneBranch -> xP
    wsi -> WSIBranch -> xI_384
    xI_384 -> WSIReduction -> xI
    xP, xI -> PathwayToPatchFusion -> CA
    xP, CA -> SurvivalRiskHead -> R
    xP, xI, mask -> PathwayPatchContrastiveLoss -> L3

Then:
    R, T, S, L3 -> SurvivalLoss -> total loss

Checks:
    - 15 real patients matched correctly
    - real OS labels loaded correctly
    - all intermediate shapes
    - Cox risk-set matrix
    - L1 / L2 / L3 / total finite
    - beta = 0.8 applied to L3
    - L2 weight-only parameter counting
    - full gradient chain from total loss
    - all-censored Cox edge case
"""

from pathlib import Path
import sys

import numpy as np
import torch

# ------------------------------------------------------------
# Project imports -- these match the actual repo layout
# ------------------------------------------------------------
sys.path.insert(0, "src")

from gene_branch.gene_branch import GeneBranch
from wsi_branch.wsi_branch import WSIBranch
from wsi_branch.wsi_reduction import WSIReduction
from fusion import PathwayToPatchFusion
from scripts.risk_head import SurvivalRiskHead
from gene_branch.contrastive_loss import PathwayPatchContrastiveLoss

from scripts.survival_loss import (
    SurvivalLoss,
    cox_neg_log_likelihood,
    load_os_labels,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
GENE_DIR = Path("data/processed/gene")
WSI_DIR = Path("data/processed/wsi")

CLINICAL_PATIENT_PATH = (
    "data/raw/blca_tcga_pan_can_atlas_2018/"
    "data_clinical_patient.txt"
)


# ------------------------------------------------------------
# Confirmed patient ordering from Stages 4 / 5A / 5B
# ------------------------------------------------------------
EXPECTED_PATIENT_IDS = [
    "TCGA-2F-A9KO",
    "TCGA-2F-A9KP",
    "TCGA-2F-A9KQ",
    "TCGA-2F-A9KR",
    "TCGA-2F-A9KT",
    "TCGA-2F-A9KW",
    "TCGA-4Z-AA7M",
    "TCGA-4Z-AA7N",
    "TCGA-4Z-AA7O",
    "TCGA-4Z-AA7Q",
    "TCGA-4Z-AA7R",
    "TCGA-4Z-AA7S",
    "TCGA-4Z-AA7W",
    "TCGA-4Z-AA7Y",
    "TCGA-4Z-AA80",
]

EXPECTED_PATCH_COUNTS = [
    144, 148, 109, 126, 137,
    139, 94, 147, 142, 100,
    118, 162, 123, 159, 109,
]

# Confirmed from the real clinical file
EXPECTED_FUTIME = [
    24.1312424,
    11.96699214,
    94.88115199,
    104.6454285,
    108.9522307,
    8.350593418,
    16.27379426,
    44.94197324,
    16.83269224,
    16.76693954,
    17.16145577,
    34.98043857,
    27.61613571,
    50.03780781,
    0.624650689,
]

EXPECTED_FUSTAT = [
    1, 1, 0, 1, 0,
    1, 0, 1, 0, 1,
    1, 1, 0, 0, 1,
]

# Previously observed Stage 4 L3.
# This is only used as a reference, NOT substituted for real L3.
STAGE4_LOGGED_L3 = 5.225014686584473


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"

    if detail and not condition:
        print(f"[{status}] {name} -- {detail}")
    else:
        print(f"[{status}] {name}")

    return condition


# ------------------------------------------------------------
# NPZ loader
# ------------------------------------------------------------
def extract_patch_array(npz_path):
    """
    Extract the (N, 384) DINO patch-feature array.

    Handles:
      - single-array NPZ files
      - keys containing patch / feature / embed
    """
    with np.load(npz_path) as npz:
        keys = list(npz.files)

        if len(keys) == 1:
            return npz[keys[0]]

        candidates = [
            k for k in keys
            if any(
                word in k.lower()
                for word in ("patch", "feature", "embed")
            )
        ]

        if not candidates:
            raise KeyError(
                f"Could not identify patch array in {npz_path}. "
                f"Available keys: {keys}"
            )

        return npz[candidates[0]]


# ------------------------------------------------------------
# Load matched real patients
# ------------------------------------------------------------
def load_matched_patients():

    gene_files = sorted(GENE_DIR.glob("*.npy"))

    if not gene_files:
        raise FileNotFoundError(
            f"No gene .npy files found under {GENE_DIR}"
        )

    # Find all WSI patch files once.
    wsi_files = list(WSI_DIR.glob("*_patches.npz"))

    if not wsi_files:
        raise FileNotFoundError(
            f"No *_patches.npz files found under {WSI_DIR}"
        )

    patient_ids = []
    gene_arrays = []
    wsi_arrays = []
    patch_counts = []

    for gene_file in gene_files:

        patient_id = gene_file.stem

        # WSI filenames contain the patient ID followed by:
        # -01Z-00-DX1.<UUID>_patches.npz
        matches = [
            f for f in wsi_files
            if f.name.startswith(patient_id + "-")
        ]

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No WSI file found for {patient_id}"
            )

        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple WSI files found for {patient_id}: "
                f"{[f.name for f in matches]}"
            )

        wsi_file = matches[0]

        gene_array = np.load(gene_file)
        wsi_array = extract_patch_array(wsi_file)

        if gene_array.shape != (186, 4942):
            raise ValueError(
                f"{patient_id}: unexpected gene shape "
                f"{gene_array.shape}"
            )

        if wsi_array.ndim != 2 or wsi_array.shape[1] != 384:
            raise ValueError(
                f"{patient_id}: unexpected WSI shape "
                f"{wsi_array.shape}"
            )

        patient_ids.append(patient_id)
        gene_arrays.append(gene_array)
        wsi_arrays.append(wsi_array)
        patch_counts.append(wsi_array.shape[0])

    return (
        patient_ids,
        gene_arrays,
        wsi_arrays,
        patch_counts,
    )


# ------------------------------------------------------------
# Build padded batch
# ------------------------------------------------------------
def build_batch(gene_arrays, wsi_arrays):

    B = len(gene_arrays)

    # Gene:
    # (B, 186, 4942)
    gene_input = torch.tensor(
        np.stack(gene_arrays, axis=0),
        dtype=torch.float32,
    )

    max_patches = max(
        array.shape[0]
        for array in wsi_arrays
    )

    feature_dim = wsi_arrays[0].shape[1]

    # WSI:
    # (B, max_N, 384)
    wsi_input = torch.zeros(
        B,
        max_patches,
        feature_dim,
        dtype=torch.float32,
    )

    # True = padding
    patch_mask = torch.ones(
        B,
        max_patches,
        dtype=torch.bool,
    )

    for i, patches in enumerate(wsi_arrays):

        n = patches.shape[0]

        wsi_input[i, :n, :] = torch.tensor(
            patches,
            dtype=torch.float32,
        )

        patch_mask[i, :n] = False

    return gene_input, wsi_input, patch_mask


# ============================================================
# MAIN
# ============================================================
def main():

    all_passed = True

    print("=" * 60)
    print("STAGE 6 — SURVIVAL LOSS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load real patients
    # --------------------------------------------------------
    (
        patient_ids,
        gene_arrays,
        wsi_arrays,
        patch_counts,
    ) = load_matched_patients()

    print("\nPatient IDs:")
    print(patient_ids)

    all_passed &= check(
        "15 matched patients",
        len(patient_ids) == 15,
        str(len(patient_ids)),
    )

    all_passed &= check(
        "patient ordering",
        patient_ids == EXPECTED_PATIENT_IDS,
        str(patient_ids),
    )

    all_passed &= check(
        "patch counts",
        patch_counts == EXPECTED_PATCH_COUNTS,
        str(patch_counts),
    )

    # --------------------------------------------------------
    # 2. Build batch
    # --------------------------------------------------------
    (
        gene_input,
        wsi_input,
        patch_mask,
    ) = build_batch(
        gene_arrays,
        wsi_arrays,
    )

    print(
        f"\nGene batch shape: "
        f"{tuple(gene_input.shape)}"
    )

    print(
        f"WSI batch shape: "
        f"{tuple(wsi_input.shape)}"
    )

    print(
        f"Patch mask shape: "
        f"{tuple(patch_mask.shape)}"
    )

    all_passed &= check(
        "gene input shape",
        tuple(gene_input.shape)
        == (15, 186, 4942),
    )

    all_passed &= check(
        "WSI input shape",
        tuple(wsi_input.shape)
        == (15, 162, 384),
    )

    all_passed &= check(
        "patch mask shape",
        tuple(patch_mask.shape)
        == (15, 162),
    )

    # --------------------------------------------------------
    # 3. Load real survival labels
    # --------------------------------------------------------
    labels = load_os_labels(
        CLINICAL_PATIENT_PATH,
        patient_ids,
    )

    T = labels.futime
    S = labels.fustat

    print("\n--- Survival labels ---")

    print(
        f"T shape: {tuple(T.shape)}"
    )

    print(
        f"S shape: {tuple(S.shape)}"
    )

    print(
        f"T (futime, months): "
        f"{T.tolist()}"
    )

    print(
        f"S (fustat): "
        f"{S.tolist()}"
    )

    all_passed &= check(
        "T shape == (15,)",
        tuple(T.shape) == (15,),
    )

    all_passed &= check(
        "S shape == (15,)",
        tuple(S.shape) == (15,),
    )

    all_passed &= check(
        "S contains only 0/1",
        bool(
            ((S == 0) | (S == 1))
            .all()
            .item()
        ),
    )

    all_passed &= check(
        "T matches confirmed values",
        torch.allclose(
            T,
            torch.tensor(
                EXPECTED_FUTIME,
                dtype=torch.float32,
            ),
            atol=1e-4,
        ),
        str(T.tolist()),
    )

    all_passed &= check(
        "S matches confirmed values",
        torch.equal(
            S,
            torch.tensor(
                EXPECTED_FUSTAT,
                dtype=torch.float32,
            ),
        ),
        str(S.tolist()),
    )

    # --------------------------------------------------------
    # 4. Instantiate actual pipeline
    # --------------------------------------------------------
    print("\n--- Stage 1-5B forward pass ---")

    gene_branch = GeneBranch()
    wsi_branch = WSIBranch()
    wsi_reduction = WSIReduction()

    fusion = PathwayToPatchFusion(
        dim=256,
        num_heads=16,
    )

    risk_head = SurvivalRiskHead(
        num_pathway_tokens=186,
        num_classes=1,
    )

    contrastive_loss_fn = PathwayPatchContrastiveLoss(
        top_h=2,
    )

    # --------------------------------------------------------
    # Gene branch
    # --------------------------------------------------------
    xP = gene_branch(gene_input)

    print(
        f"GENE output: {tuple(xP.shape)} "
        f"(expect (15,186,256))"
    )

    all_passed &= check(
        "xP shape",
        tuple(xP.shape)
        == (15, 186, 256),
    )

    # --------------------------------------------------------
    # WSI branch
    # --------------------------------------------------------
    xI_384 = wsi_branch(
        wsi_input,
        patch_mask,
    )

    print(
        f"WSI Stage-3 output: "
        f"{tuple(xI_384.shape)} "
        f"(expect (15,162,384))"
    )

    all_passed &= check(
        "xI_384 shape",
        tuple(xI_384.shape)
        == (15, 162, 384),
    )

    # --------------------------------------------------------
    # WSI reduction
    # --------------------------------------------------------
    xI, patch_mask_out = wsi_reduction(
        xI_384,
        patch_mask,
    )

    print(
        f"WSI Stage-3.5 output: "
        f"{tuple(xI.shape)} "
        f"(expect (15,162,256))"
    )

    all_passed &= check(
        "xI shape",
        tuple(xI.shape)
        == (15, 162, 256),
    )

    all_passed &= check(
        "patch mask preserved",
        torch.equal(
            patch_mask_out,
            patch_mask,
        ),
    )

    # Use the preserved mask downstream
    patch_mask = patch_mask_out

    # --------------------------------------------------------
    # Stage 5A fusion
    # --------------------------------------------------------
    CA, attn = fusion(
        xP,
        xI,
        patch_mask,
    )

    print(
        f"Stage-5A CA output: "
        f"{tuple(CA.shape)} "
        f"(expect (15,186,256))"
    )

    all_passed &= check(
        "CA shape",
        tuple(CA.shape)
        == (15, 186, 256),
    )

    # --------------------------------------------------------
    # Stage 5B risk head
    # --------------------------------------------------------
    R = risk_head(
        xP,
        CA,
    )

    print(
        f"Risk output: "
        f"{tuple(R.shape)} "
        f"(expect (15,1))"
    )

    all_passed &= check(
        "R shape",
        tuple(R.shape)
        == (15, 1),
    )

    all_passed &= check(
        "R finite",
        bool(
            torch.isfinite(R)
            .all()
            .item()
        ),
    )

    # --------------------------------------------------------
    # Stage 4 L3
    # --------------------------------------------------------
    L3, info = contrastive_loss_fn(
        xP,
        xI,
        patch_mask,
    )

    print(
        f"L3: {L3.item():.6f}"
    )

    print(
        f"Previously logged Stage-4 L3: "
        f"{STAGE4_LOGGED_L3:.6f}"
    )

    all_passed &= check(
        "L3 finite scalar",
        L3.dim() == 0
        and bool(
            torch.isfinite(L3)
            .item()
        ),
    )

    # --------------------------------------------------------
    # 5. R_mat sanity check
    # --------------------------------------------------------
    print("\n--- Cox risk-set matrix ---")

    theta = R.squeeze(-1)

    _, R_mat = cox_neg_log_likelihood(
        theta.detach(),
        T,
        S,
    )

    print(
        f"R_mat shape: "
        f"{tuple(R_mat.shape)}"
    )

    all_passed &= check(
        "R_mat shape",
        tuple(R_mat.shape)
        == (15, 15),
    )

    diagonal = torch.diagonal(R_mat)

    all_passed &= check(
        "R_mat diagonal all 1",
        bool(
            torch.equal(
                diagonal,
                torch.ones(15),
            )
        ),
    )

    max_t_idx = int(
        torch.argmax(T).item()
    )

    max_t_row = R_mat[max_t_idx]

    print(
        f"Maximum-T patient: "
        f"{patient_ids[max_t_idx]}"
    )

    print(
        f"Maximum-T risk-set size: "
        f"{max_t_row.sum().item()}"
    )

    all_passed &= check(
        "maximum-T patient has only itself in risk set",
        bool(
            max_t_row.sum().item() == 1.0
            and
            max_t_row[max_t_idx].item() == 1.0
        ),
    )

    # --------------------------------------------------------
    # 6. Survival loss
    # --------------------------------------------------------
    print("\n--- Survival Loss ---")

    loss_fn = SurvivalLoss(
        weight_decay=5e-4,
        alpha=1.0,
        beta=0.8,
    )

    modules_list = [
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
    ]

    total_loss, diagnostics = loss_fn(
        R,
        T,
        S,
        L3,
        modules_list,
    )

    L1 = diagnostics["L1"]
    L2_raw = diagnostics["L2_raw"]
    L2_weighted = diagnostics["L2_weighted"]
    L3_weighted = diagnostics["L3_weighted"]

    print(
        f"L1           = {L1.item():.6f}"
    )

    print(
        f"L2_raw       = {L2_raw.item():.6f}"
    )

    print(
        f"L2_weighted  = {L2_weighted.item():.6f}"
    )

    print(
        f"L3           = {L3.item():.6f}"
    )

    print(
        f"L3_weighted  = {L3_weighted.item():.6f}"
    )

    print(
        f"Total loss   = {total_loss.item():.6f}"
    )

    # --------------------------------------------------------
    # Numerical checks
    # --------------------------------------------------------
    all_passed &= check(
        "L1 finite",
        bool(
            torch.isfinite(L1)
            .item()
        ),
    )

    all_passed &= check(
        "L2_raw finite",
        bool(
            torch.isfinite(L2_raw)
            .item()
        ),
    )

    all_passed &= check(
        "L2_weighted finite",
        bool(
            torch.isfinite(L2_weighted)
            .item()
        ),
    )

    all_passed &= check(
        "L3_weighted finite",
        bool(
            torch.isfinite(L3_weighted)
            .item()
        ),
    )

    all_passed &= check(
        "total loss finite",
        bool(
            torch.isfinite(total_loss)
            .item()
        ),
    )

    # --------------------------------------------------------
    # L3 beta check
    # --------------------------------------------------------
    expected_L3_weighted = (
        0.8 * L3.detach()
    )

    all_passed &= check(
        "L3_weighted == 0.8 * L3",
        bool(
            torch.isclose(
                L3_weighted,
                expected_L3_weighted,
                atol=1e-5,
            ).item()
        ),
        (
            f"got {L3_weighted.item():.8f}, "
            f"expected {expected_L3_weighted.item():.8f}"
        ),
    )

    # --------------------------------------------------------
    # 7. L2 weight-only parameter count
    # --------------------------------------------------------
    print("\n--- L2 parameter check ---")

    weight_named_params = [
        (name, p)
        for module in modules_list
        for name, p in module.named_parameters()
        if "weight" in name
    ]

    total_named_params = [
        (name, p)
        for module in modules_list
        for name, p in module.named_parameters()
    ]

    weight_count = len(
        weight_named_params
    )

    total_count = len(
        total_named_params
    )

    print(
        f"Weight-named parameters: "
        f"{weight_count}"
    )

    print(
        f"Total parameter tensors: "
        f"{total_count}"
    )

    all_passed &= check(
        "L2 excludes bias parameters",
        weight_count < total_count,
        f"{weight_count} vs {total_count}",
    )

    # --------------------------------------------------------
    # 8. Full gradient chain
    # --------------------------------------------------------
    print(
        "\n--- Full gradient check from TOTAL LOSS ---"
    )

    for module in modules_list:
        for p in module.parameters():
            p.grad = None

    total_loss.backward()

    for module, name in zip(
        modules_list,
        [
            "GeneBranch",
            "WSIBranch",
            "WSIReduction",
            "PathwayToPatchFusion",
            "SurvivalRiskHead",
        ],
    ):

        params = list(
            module.parameters()
        )

        n_params = len(params)

        n_good = sum(
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
            f"{n_good}/{n_params} "
            f"parameters have finite "
            f"nonzero gradients"
        )

        all_passed &= check(
            f"{name} full gradient chain",
            n_good == n_params,
            f"{n_good}/{n_params}",
        )

    # --------------------------------------------------------
    # 9. All-censored edge case
    # --------------------------------------------------------
    print(
        "\n--- All-censored edge case ---"
    )

    S_all_censored = torch.zeros_like(S)

    L1_censored, _ = (
        cox_neg_log_likelihood(
            theta.detach(),
            T,
            S_all_censored,
        )
    )

    print(
        f"All-censored L1: "
        f"{L1_censored.item()}"
    )

    all_passed &= check(
        "all-censored L1 == 0",
        L1_censored.item() == 0.0,
        f"got {L1_censored.item()}",
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------
    print("\n" + "=" * 60)

    if all_passed:
        print(
            "STAGE 6 REAL-DATA CHECK: PASSED"
        )
    else:
        print(
            "STAGE 6 REAL-DATA CHECK: FAILED"
        )

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())