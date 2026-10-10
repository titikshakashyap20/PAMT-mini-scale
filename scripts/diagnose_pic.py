"""
PAMT MINI-SCALE — PIC DIAGNOSTIC

Pathway-Identity Consistency (PIC)
----------------------------------

This is DIAGNOSTIC ONLY.

It does NOT:
- train or fine-tune the model
- modify best_model.pt
- modify any PAMT module
- add any parameters
- change the original attention pipeline

Goal:
Test whether pathway-indexed cross-attention maps retain enough
pathway identity after random WSI patch removal to justify a future
PIC regularization experiment.

Procedure:
1. Load the existing epoch-467 best checkpoint.
2. Compute clean pathway-to-patch attention.
3. For each patient, perform 5 random perturbations.
4. Each perturbation removes exactly 20% of valid WSI patches.
5. Re-run the WSI branch + reduction + cross-attention.
6. Compare each clean pathway p against every perturbed pathway q
   using Jensen-Shannon divergence.
7. Re-identify pathway p as the perturbed pathway with minimum JS.
8. Report re-identification accuracy overall and per patient.

PASS CRITERIA:
- Overall re-ID accuracy must be between 0.05 and 0.80.
- At least 10/15 patients must individually fall in that range.

FAIL CRITERIA:
- Overall accuracy < 0.02
- Overall accuracy > 0.90
- Fewer than 10/15 patients fall in the required range.

The thresholds are fixed BEFORE seeing the result.
"""

from pathlib import Path
import sys
import csv
import random

import numpy as np
import torch


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


from train import (
    load_matched_patients,
    build_batch,
    build_modules,
    BEST_MODEL_PATH,
    EXPECTED_PATIENT_IDS,
)


# ================================================================
# CONFIGURATION
# ================================================================

NUM_PATHWAYS = 186
NUM_HEADS = 16

NUM_DRAWS = 5
REMOVAL_FRACTION = 0.20

BASE_SEED = 20261007

LOW_PASS = 0.05
HIGH_PASS = 0.80

HARD_FAIL_LOW = 0.02
HARD_FAIL_HIGH = 0.90

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pic_diagnostic"

PATIENT_CSV = OUTPUT_DIR / "patient_reidentification.csv"
DRAW_CSV = OUTPUT_DIR / "draw_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "pic_summary.csv"


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ================================================================
# MODEL LOADING
# ================================================================

def load_best_model():

    modules, trainable_modules = build_modules()

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found:\n{BEST_MODEL_PATH}\n"
            "Run scripts/train.py first."
        )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
    ) = trainable_modules

    gene_branch.load_state_dict(checkpoint["GeneBranch"])
    wsi_branch.load_state_dict(checkpoint["WSIBranch"])
    wsi_reduction.load_state_dict(checkpoint["WSIReduction"])
    fusion.load_state_dict(checkpoint["Fusion"])
    risk_head.load_state_dict(checkpoint["SurvivalRiskHead"])

    for module in trainable_modules:
        module.eval()

    return modules, trainable_modules, checkpoint


# ================================================================
# ATTENTION EXTRACTION
# ================================================================

@torch.no_grad()
def get_attention(
    xP,
    wsi_input,
    patch_mask,
    wsi_branch,
    wsi_reduction,
    fusion,
):
    """
    Run:

        WSI input
          ↓
        WSIBranch
          ↓
        WSIReduction
          ↓
        Pathway-to-Patch Fusion

    Returns:
        attention: (B, H, P, N)
        output_mask: (B, N)
    """

    xI_384 = wsi_branch(
        wsi_input,
        patch_mask,
    )

    xI, output_mask = wsi_reduction(
        xI_384,
        patch_mask,
    )

    _, attention = fusion(
        xP,
        xI,
        output_mask,
    )

    if attention.ndim != 4:
        raise RuntimeError(
            f"Unexpected attention shape: {tuple(attention.shape)}"
        )

    return attention, output_mask


# ================================================================
# HEAD-MEAN ATTENTION
# ================================================================

def head_mean_attention(attention, valid_mask):
    """
    attention:
        (H, P, N)

    valid_mask:
        (N,), True = padding

    Returns:
        (P, N_valid)

    Attention is averaged over heads and only valid patches
    are retained.
    """

    # Average over attention heads.
    attn = attention.mean(dim=0)

    # Keep only real patches.
    valid = ~valid_mask

    attn = attn[:, valid]

    # Numerical safety.
    attn = torch.clamp(attn, min=0.0)

    # Re-normalize each pathway's attention distribution.
    row_sum = attn.sum(dim=-1, keepdim=True)

    if torch.any(row_sum <= 0):
        raise RuntimeError(
            "Encountered pathway attention row with zero mass."
        )

    attn = attn / row_sum

    return attn


# ================================================================
# JENSEN-SHANNON DIVERGENCE
# ================================================================

def js_divergence_matrix(clean, perturbed, eps=1e-12):
    """
    Compute JS divergence between every clean pathway and every
    perturbed pathway.

    clean:
        (P, N)

    perturbed:
        (P, N)

    Returns:
        (P, P)

    Entry [p, q] = JS(clean pathway p, perturbed pathway q)
    """

    clean = clean.clamp_min(eps)
    perturbed = perturbed.clamp_min(eps)

    # Normalize again for safety.
    clean = clean / clean.sum(dim=-1, keepdim=True)
    perturbed = perturbed / perturbed.sum(dim=-1, keepdim=True)

    # Add pairwise dimensions.
    #
    # clean:
    #   P x 1 x N
    #
    # perturbed:
    #   1 x P x N
    #
    p = clean[:, None, :]
    q = perturbed[None, :, :]

    m = 0.5 * (p + q)

    kl_pm = (
        p * (
            torch.log(p) - torch.log(m)
        )
    ).sum(dim=-1)

    kl_qm = (
        q * (
            torch.log(q) - torch.log(m)
        )
    ).sum(dim=-1)

    js = 0.5 * (kl_pm + kl_qm)

    return js


# ================================================================
# RANDOM PATCH REMOVAL
# ================================================================

def make_perturbed_patient(
    wsi_input,
    patch_mask,
    removal_fraction,
    rng,
):
    """
    Remove a random fraction of VALID patches.

    Removed patches are:
    - marked as padding
    - zeroed explicitly

    Returns:
        perturbed_wsi
        perturbed_mask
        removed_indices
    """

    wsi = wsi_input.clone()
    mask = patch_mask.clone()

    valid_indices = torch.nonzero(
        ~mask,
        as_tuple=False,
    ).flatten().tolist()

    valid_n = len(valid_indices)

    if valid_n < 2:
        raise RuntimeError(
            f"Not enough valid patches: {valid_n}"
        )

    num_remove = max(
        1,
        int(round(valid_n * removal_fraction)),
    )

    num_remove = min(
        num_remove,
        valid_n - 1,
    )

    removed = rng.sample(
        valid_indices,
        num_remove,
    )

    removed_tensor = torch.tensor(
        removed,
        dtype=torch.long,
    )

    mask[removed_tensor] = True
    wsi[removed_tensor] = 0.0

    return wsi, mask, removed


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 78)
    print("PAMT MINI-SCALE — PATHWAY-IDENTITY CONSISTENCY DIAGNOSTIC")
    print("=" * 78)

    print("\nIMPORTANT:")
    print("This is diagnostic only.")
    print("No training or fine-tuning will be performed.")
    print("The original best checkpoint will not be modified.\n")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(BASE_SEED)

    # ------------------------------------------------------------
    # 1. Load exact PAMT cohort
    # ------------------------------------------------------------

    (
        patient_ids,
        gene_arrays,
        wsi_arrays,
    ) = load_matched_patients()

    if patient_ids != EXPECTED_PATIENT_IDS:
        raise RuntimeError(
            "Patient ordering mismatch with train.py."
        )

    print(f"Patients: {len(patient_ids)}")

    # ------------------------------------------------------------
    # 2. Build the normal batch
    # ------------------------------------------------------------

    (
        gene_input,
        wsi_input,
        patch_mask,
    ) = build_batch(
        gene_arrays,
        wsi_arrays,
    )

    print(
        f"Gene input: {tuple(gene_input.shape)}"
    )

    print(
        f"WSI input:  {tuple(wsi_input.shape)}"
    )

    print(
        f"Mask:       {tuple(patch_mask.shape)}"
    )

    # ------------------------------------------------------------
    # 3. Load checkpoint
    # ------------------------------------------------------------

    (
        modules,
        trainable_modules,
        checkpoint,
    ) = load_best_model()

    (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
    ) = trainable_modules

    best_epoch = checkpoint.get(
        "epoch",
        "unknown",
    )

    print(f"Checkpoint: {BEST_MODEL_PATH}")
    print(f"Checkpoint epoch: {best_epoch}")

    # ------------------------------------------------------------
    # 4. Compute gene representation ONCE
    # ------------------------------------------------------------

    with torch.no_grad():

        xP = gene_branch(
            gene_input
        )

    print(
        f"Gene representation: {tuple(xP.shape)}"
    )

    # ------------------------------------------------------------
    # 5. Clean attention
    # ------------------------------------------------------------

    print("\nComputing clean attention...")

    with torch.no_grad():

        clean_attention, clean_mask = get_attention(
            xP,
            wsi_input,
            patch_mask,
            wsi_branch,
            wsi_reduction,
            fusion,
        )

    print(
        f"Clean attention: {tuple(clean_attention.shape)}"
    )

    # ------------------------------------------------------------
    # 6. Storage
    # ------------------------------------------------------------

    patient_results = []
    draw_results = []

    total_correct = 0
    total_cases = 0

    # ------------------------------------------------------------
    # 7. Patient loop
    # ------------------------------------------------------------

    for patient_idx, patient_id in enumerate(patient_ids):

        print("\n" + "-" * 78)
        print(
            f"[{patient_idx + 1}/{len(patient_ids)}] "
            f"{patient_id}"
        )

        clean_patient_attention = clean_attention[
            patient_idx
        ]

        clean_patient_mask = clean_mask[
            patient_idx
        ]

        valid_indices = torch.nonzero(
            ~clean_patient_mask,
            as_tuple=False,
        ).flatten()

        valid_n = int(
            valid_indices.numel()
        )

        print(
            f"Valid patches: {valid_n}"
        )

        # --------------------------------------------------------
        # Clean head-mean attention
        # --------------------------------------------------------

        clean_distribution = head_mean_attention(
            clean_patient_attention,
            clean_patient_mask,
        )

        # --------------------------------------------------------
        # Draw perturbations
        # --------------------------------------------------------

        patient_correct = 0
        patient_cases = 0

        for draw_idx in range(NUM_DRAWS):

            draw_seed = (
                BASE_SEED
                + patient_idx * 100
                + draw_idx
            )

            rng = random.Random(
                draw_seed
            )

            (
                perturbed_wsi,
                perturbed_mask,
                removed_indices,
            ) = make_perturbed_patient(
                wsi_input[patient_idx],
                patch_mask[patient_idx],
                REMOVAL_FRACTION,
                rng,
            )

            # Add batch dimension.
            perturbed_wsi = (
                perturbed_wsi
                .unsqueeze(0)
            )

            perturbed_mask = (
                perturbed_mask
                .unsqueeze(0)
            )

            xP_patient = (
                xP[patient_idx]
                .unsqueeze(0)
            )

            # ----------------------------------------------------
            # Re-run WSI branch + reduction + fusion.
            # ----------------------------------------------------

            with torch.no_grad():

                perturbed_attention, perturbed_output_mask = (
                    get_attention(
                        xP_patient,
                        perturbed_wsi,
                        perturbed_mask,
                        wsi_branch,
                        wsi_reduction,
                        fusion,
                    )
                )

            perturbed_attention = (
                perturbed_attention[0]
            )

            perturbed_output_mask = (
                perturbed_output_mask[0]
            )

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # Compare only patches that survived the perturbation.
            #
            # Clean attention is restricted to the SAME retained
            # patch positions and renormalized.
            # ----------------------------------------------------

            retained_mask = ~perturbed_output_mask

            clean_restricted = (
                clean_patient_attention[
                    :,
                    :,
                ].mean(dim=0)
            )

            clean_restricted = (
                clean_restricted[
                    :,
                    retained_mask,
                ]
            )

            clean_restricted = (
                clean_restricted.clamp_min(1e-12)
            )

            clean_restricted = (
                clean_restricted
                / clean_restricted.sum(
                    dim=-1,
                    keepdim=True,
                )
            )

            # ----------------------------------------------------
            # Perturbed head-mean attention
            # ----------------------------------------------------

            perturbed_distribution = (
                head_mean_attention(
                    perturbed_attention,
                    perturbed_output_mask,
                )
            )

            # ----------------------------------------------------
            # Check dimensions
            # ----------------------------------------------------

            if (
                clean_restricted.shape
                != perturbed_distribution.shape
            ):
                raise RuntimeError(
                    "Clean/perturbed attention dimension mismatch: "
                    f"{tuple(clean_restricted.shape)} vs "
                    f"{tuple(perturbed_distribution.shape)}"
                )

            # ----------------------------------------------------
            # Pairwise JS divergence
            # ----------------------------------------------------

            js_matrix = js_divergence_matrix(
                clean_restricted,
                perturbed_distribution,
            )

            # ----------------------------------------------------
            # Re-identification
            #
            # For each clean pathway p:
            # choose perturbed pathway q with minimum JS.
            # ----------------------------------------------------

            predicted_pathways = (
                torch.argmin(
                    js_matrix,
                    dim=1,
                )
            )

            true_pathways = torch.arange(
                NUM_PATHWAYS,
                dtype=torch.long,
            )

            correct = (
                predicted_pathways
                == true_pathways
            )

            draw_accuracy = (
                correct.float()
                .mean()
                .item()
            )

            draw_correct = int(
                correct.sum().item()
            )

            draw_cases = NUM_PATHWAYS

            patient_correct += draw_correct
            patient_cases += draw_cases

            total_correct += draw_correct
            total_cases += draw_cases

            # ----------------------------------------------------
            # Mean diagonal / off-diagonal JS
            # ----------------------------------------------------

            diagonal_js = (
                torch.diag(js_matrix)
                .mean()
                .item()
            )

            off_diagonal_mask = (
                ~torch.eye(
                    NUM_PATHWAYS,
                    dtype=torch.bool,
                )
            )

            between_js = (
                js_matrix[
                    off_diagonal_mask
                ]
                .mean()
                .item()
            )

            num_removed = len(
                removed_indices
            )

            num_remaining = int(
                retained_mask.sum().item()
            )

            print(
                f"  Draw {draw_idx + 1}/{NUM_DRAWS}: "
                f"removed={num_removed}, "
                f"remaining={num_remaining}, "
                f"re-ID={draw_accuracy:.4f}"
            )

            draw_results.append(
                {
                    "patient_id": patient_id,
                    "draw": draw_idx + 1,
                    "seed": draw_seed,
                    "valid_patches": valid_n,
                    "removed_patches": num_removed,
                    "remaining_patches": num_remaining,
                    "reidentification_accuracy": draw_accuracy,
                    "diagonal_js": diagonal_js,
                    "between_pathway_js": between_js,
                }
            )

        # --------------------------------------------------------
        # Patient-level accuracy
        # --------------------------------------------------------

        patient_accuracy = (
            patient_correct
            / patient_cases
        )

        patient_pass = (
            LOW_PASS
            <= patient_accuracy
            <= HIGH_PASS
        )

        print(
            f"  Patient re-ID accuracy: "
            f"{patient_accuracy:.4f}"
        )

        print(
            f"  Patient diagnostic range: "
            f"{'PASS' if patient_pass else 'FAIL'}"
        )

        patient_results.append(
            {
                "patient_id": patient_id,
                "valid_patches": valid_n,
                "num_draws": NUM_DRAWS,
                "reidentification_accuracy": patient_accuracy,
                "within_required_range": patient_pass,
            }
        )

    # ============================================================
    # 8. Overall result
    # ============================================================

    overall_accuracy = (
        total_correct
        / total_cases
    )

    patients_in_range = sum(
        row["within_required_range"]
        for row in patient_results
    )

    hard_fail = (
        overall_accuracy < HARD_FAIL_LOW
        or overall_accuracy > HARD_FAIL_HIGH
    )

    range_requirement = (
        LOW_PASS
        <= overall_accuracy
        <= HIGH_PASS
    )

    patient_requirement = (
        patients_in_range >= 10
    )

    if hard_fail:
        overall_decision = "FAIL"
        reason = (
            "Overall re-identification accuracy is outside "
            "the hard diagnostic bounds."
        )

    elif not range_requirement:
        overall_decision = "FAIL"
        reason = (
            "Overall re-identification accuracy is not in "
            "the predefined 0.05–0.80 range."
        )

    elif not patient_requirement:
        overall_decision = "FAIL"
        reason = (
            "Fewer than 10/15 patients fall within the "
            "predefined 0.05–0.80 range."
        )

    else:
        overall_decision = "PASS"
        reason = (
            "The predefined diagnostic criteria were satisfied."
        )

    # ============================================================
    # 9. Save patient CSV
    # ============================================================

    with open(
        PATIENT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "valid_patches",
                "num_draws",
                "reidentification_accuracy",
                "within_required_range",
            ],
        )

        writer.writeheader()
        writer.writerows(patient_results)

    # ============================================================
    # 10. Save draw CSV
    # ============================================================

    with open(
        DRAW_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "patient_id",
            "draw",
            "seed",
            "valid_patches",
            "removed_patches",
            "remaining_patches",
            "reidentification_accuracy",
            "diagonal_js",
            "between_pathway_js",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(draw_results)

    # ============================================================
    # 11. Save summary
    # ============================================================

    summary_rows = [
        ("checkpoint_epoch", best_epoch),
        ("num_patients", len(patient_ids)),
        ("num_pathways", NUM_PATHWAYS),
        ("num_attention_heads", NUM_HEADS),
        ("num_draws_per_patient", NUM_DRAWS),
        ("removal_fraction", REMOVAL_FRACTION),
        ("overall_reidentification_accuracy", overall_accuracy),
        ("patients_in_required_range", patients_in_range),
        ("required_patients_in_range", 10),
        ("overall_low_threshold", LOW_PASS),
        ("overall_high_threshold", HIGH_PASS),
        ("hard_fail_low", HARD_FAIL_LOW),
        ("hard_fail_high", HARD_FAIL_HIGH),
        ("decision", overall_decision),
        ("reason", reason),
    ]

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)

    # ============================================================
    # 12. Final report
    # ============================================================

    print("\n")
    print("=" * 78)
    print("PIC DIAGNOSTIC RESULT")
    print("=" * 78)

    print(
        f"Overall re-identification accuracy: "
        f"{overall_accuracy:.6f}"
    )

    print(
        f"Patients in required range "
        f"[{LOW_PASS:.2f}, {HIGH_PASS:.2f}]: "
        f"{patients_in_range}/15"
    )

    print(
        f"Hard-fail bounds: "
        f"< {HARD_FAIL_LOW:.2f} or > {HARD_FAIL_HIGH:.2f}"
    )

    print(
        f"\nDECISION: {overall_decision}"
    )

    print(
        f"Reason: {reason}"
    )

    print("\nOutputs:")
    print(f"  {PATIENT_CSV}")
    print(f"  {DRAW_CSV}")
    print(f"  {SUMMARY_CSV}")

    print("\n" + "=" * 78)

    if overall_decision == "PASS":
        print(
            "PIC DIAGNOSTIC PASSED."
        )
        print(
            "Do NOT train yet. Inspect the results first."
        )
    else:
        print(
            "PIC DIAGNOSTIC FAILED."
        )
        print(
            "Do NOT fine-tune PIC."
        )

    print("=" * 78)


if __name__ == "__main__":
    main()