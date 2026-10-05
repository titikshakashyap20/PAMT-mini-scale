"""
Post-hoc modality ablation for the PAMT mini-scale model.

IMPORTANT:
This is a POST-HOC MODALITY-MASKING experiment using the already-trained
best_model.pt. It does not retrain separate gene-only/WSI-only models.

Conditions:
    1. full_multimodal
    2. gene_only_masked_wsi
    3. wsi_only_masked_gene

The experiment is intended as a lightweight contribution/ablation analysis
for the existing 15-patient checkpoint. All metrics are training-set /
exploratory because the checkpoint was trained on these same 15 patients.

Run from the project root:
    python scripts/modality_ablation.py
"""

from pathlib import Path
import sys
import csv

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
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
    load_os_status_strings,
    build_modules,
    EXPECTED_PATIENT_IDS,
    BEST_MODEL_PATH,
    CLINICAL_PATIENT_PATH,
)
from scripts.survival_loss import SurvivalLoss, load_os_labels
from c_index import concordance_index


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modality_ablation"
RESULTS_CSV = OUTPUT_DIR / "modality_ablation_results.csv"
RISK_CSV = OUTPUT_DIR / "modality_ablation_risk_scores.csv"


def load_best_model():
    """Load the exact best checkpoint using train.py's model construction."""
    modules, trainable_modules = build_modules()

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_MODEL_PATH}\n"
            "Run the existing training pipeline first."
        )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu")

    gene_branch, wsi_branch, wsi_reduction, fusion, risk_head = trainable_modules

    gene_branch.load_state_dict(checkpoint["GeneBranch"])
    wsi_branch.load_state_dict(checkpoint["WSIBranch"])
    wsi_reduction.load_state_dict(checkpoint["WSIReduction"])
    fusion.load_state_dict(checkpoint["Fusion"])
    risk_head.load_state_dict(checkpoint["SurvivalRiskHead"])

    for module in trainable_modules:
        module.eval()

    return modules, trainable_modules, checkpoint


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PAMT MINI-SCALE — POST-HOC MODALITY ABLATION")
    print("=" * 72)
    print("This uses the existing best_model.pt; no retraining is performed.")
    print("All metrics are training-set / exploratory.\n")

    # -------------------------------------------------------------
    # 1. Load the exact same 15-patient batch used by train.py
    # -------------------------------------------------------------
    patient_ids, gene_arrays, wsi_arrays = load_matched_patients()

    if patient_ids != EXPECTED_PATIENT_IDS:
        raise RuntimeError(
            "Patient ordering mismatch. Aborting rather than producing "
            "misaligned risk scores."
        )

    gene_input, wsi_input, patch_mask = build_batch(
        gene_arrays,
        wsi_arrays,
    )

    labels = load_os_labels(
        CLINICAL_PATIENT_PATH,
        patient_ids,
    )
    T, S = labels.futime, labels.fustat
    os_status_strings = load_os_status_strings(
        CLINICAL_PATIENT_PATH,
        patient_ids,
    )

    # -------------------------------------------------------------
    # 2. Load best checkpoint
    # -------------------------------------------------------------
    modules, trainable_modules, checkpoint = load_best_model()

    (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
        contrastive_loss_fn,
    ) = modules

    print(f"Checkpoint: {BEST_MODEL_PATH}")
    print(f"Best epoch: {checkpoint.get('epoch')}")
    print(f"Gene input: {tuple(gene_input.shape)}")
    print(f"WSI input:  {tuple(wsi_input.shape)}")

    # -------------------------------------------------------------
    # 3. Obtain the shared branch representations ONCE
    # -------------------------------------------------------------
    with torch.no_grad():
        xP = gene_branch(gene_input)

        xI_384 = wsi_branch(
            wsi_input,
            patch_mask,
        )

        xI, mask_out = wsi_reduction(
            xI_384,
            patch_mask,
        )

        # Normal full multimodal fusion.
        CA_full, attn_full = fusion(
            xP,
            xI,
            mask_out,
        )

        # Full model risk.
        R_full = risk_head(
            xP,
            CA_full,
        )

        # ---------------------------------------------------------
        # Gene-only condition:
        # remove the WSI contribution at the risk-head input.
        #
        # xP remains real; CA is replaced by zeros.
        # ---------------------------------------------------------
        CA_zero = torch.zeros_like(CA_full)

        R_gene = risk_head(
            xP,
            CA_zero,
        )

        # ---------------------------------------------------------
        # WSI-only condition:
        # remove gene information BEFORE fusion.
        #
        # Because q has no bias in the official-code-faithful
        # fusion module, zero xP produces a neutral/uniform query.
        # This is therefore a post-hoc WSI-only masking analysis,
        # NOT a separately retrained WSI-only model.
        # ---------------------------------------------------------
        xP_zero = torch.zeros_like(xP)

        CA_wsi, _ = fusion(
            xP_zero,
            xI,
            mask_out,
        )

        R_wsi = risk_head(
            xP_zero,
            CA_wsi,
        )

    # -------------------------------------------------------------
    # 4. Compute Cox L1 only.
    #
    # We intentionally do NOT compare full SurvivalLoss totals here:
    # L2 depends on the complete model parameter set and L3 is a
    # multimodal alignment loss, so they are not meaningful as
    # post-hoc modality-masking metrics.
    # -------------------------------------------------------------
    cox_only = SurvivalLoss(
        weight_decay=0.0,
        alpha=1.0,
        beta=0.0,
    )

    conditions = {
        "full_multimodal": R_full,
        "gene_only_masked_wsi": R_gene,
        "wsi_only_masked_gene": R_wsi,
    }

    rows = []
    risk_rows = []

    for condition_name, risk in conditions.items():
        with torch.no_grad():
            # SurvivalLoss returns L1 plus zero-weighted L2/L3 here.
            loss_value, diagnostics = cox_only(
                risk,
                T,
                S,
                torch.tensor(0.0),
                trainable_modules,
            )

            c_idx = concordance_index(
                risk.squeeze(-1),
                T,
                S,
            )

        rows.append(
            {
                "condition": condition_name,
                "cox_L1": float(diagnostics["L1"].item()),
                "c_index": float(c_idx),
            }
        )

        risk_values = risk.squeeze(-1).tolist()

        for pid, months, status, score in zip(
            patient_ids,
            T.tolist(),
            os_status_strings,
            risk_values,
        ):
            risk_rows.append(
                {
                    "condition": condition_name,
                    "patient_id": pid,
                    "OS_MONTHS": months,
                    "OS_STATUS": status,
                    "risk_score": score,
                }
            )

    results = pd.DataFrame(rows)

    # Delta C-index relative to the full model.
    full_cindex = float(
        results.loc[
            results["condition"] == "full_multimodal",
            "c_index",
        ].iloc[0]
    )
    results["delta_c_index_vs_full"] = (
        results["c_index"] - full_cindex
    )

    results.to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(risk_rows).to_csv(RISK_CSV, index=False)

    # -------------------------------------------------------------
    # 5. Console summary
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(results.to_string(index=False))

    print("\nRisk-score files:")
    print(f"  {RISK_CSV}")
    print(f"Summary:")
    print(f"  {RESULTS_CSV}")

    print("\nInterpretation:")
    print(
        "  - full_multimodal = original trained model."
    )
    print(
        "  - gene_only_masked_wsi = WSI contribution removed at the risk head."
    )
    print(
        "  - wsi_only_masked_gene = gene representation removed before fusion."
    )
    print(
        "  - These are post-hoc masking results, not separately retrained "
        "single-modality models."
    )
    print(
        "  - C-index is exploratory because the same 15 patients were used "
        "to train the checkpoint."
    )


if __name__ == "__main__":
    main()
