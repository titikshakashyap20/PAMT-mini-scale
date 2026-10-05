"""
Evaluate the best-training-loss checkpoint (data/processed/model_checkpoints/best_model.pt)
on all 15 real patients.

This script does NOT retrain anything and does NOT modify any verified
Stage 1-6 module (GeneBranch, WSIBranch, WSIReduction, Fusion,
SurvivalRiskHead, PathwayPatchContrastiveLoss, SurvivalLoss). It reuses
train.py's own data-loading and model-construction functions directly, so
there is no second, drifting copy of that logic.

IMPORTANT LABELING:
"best_model.pt" = the checkpoint with the LOWEST total_loss seen during
training (see train.py). There is no held-out validation or test set in
this 15-patient mini-scale project. Every number this script prints or
saves is therefore a TRAINING-SET / EXPLORATORY number -- never a
validation or generalization result. This is stated again in the final
printed summary so it can never be read out of context.
"""

from pathlib import Path
import sys
import csv

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

# Reuse train.py's own loading + model-construction logic directly --
# no re-derivation, no duplication.
from train import (
    load_matched_patients,
    build_batch,
    load_os_status_strings,
    build_modules,
    forward_pipeline,
    EXPECTED_PATIENT_IDS,
    BEST_MODEL_PATH,
    CLINICAL_PATIENT_PATH,
    TRAINING_OUT_DIR,
)
from scripts.survival_loss import SurvivalLoss, load_os_labels
from c_index import concordance_index


BEST_RISK_CSV = TRAINING_OUT_DIR / "best_risk_scores.csv"


def main():
    print("=" * 60)
    print("BEST TRAINING-LOSS CHECKPOINT EVALUATION")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Load the same 15 patients, in the exact same order as train.py
    #    (load_matched_patients() itself asserts this order internally)
    # ------------------------------------------------------------
    patient_ids, gene_arrays, wsi_arrays = load_matched_patients()
    assert patient_ids == EXPECTED_PATIENT_IDS, (
        "Patient ordering does not match train.py's confirmed order -- "
        "aborting rather than risk misaligned risk scores."
    )

    gene_input, wsi_input, patch_mask = build_batch(gene_arrays, wsi_arrays)
    labels = load_os_labels(CLINICAL_PATIENT_PATH, patient_ids)
    T, S = labels.futime, labels.fustat
    os_status_strings = load_os_status_strings(CLINICAL_PATIENT_PATH, patient_ids)

    print(f"Loaded {len(patient_ids)} patients: {patient_ids}")

    # ------------------------------------------------------------
    # 2. Reconstruct the exact Stage 1-5B architecture used by train.py
    # ------------------------------------------------------------
    modules, trainable_modules = build_modules()
    gene_branch, wsi_branch, wsi_reduction, fusion, risk_head = trainable_modules

    # ------------------------------------------------------------
    # 3. Load the best checkpoint
    # ------------------------------------------------------------
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {BEST_MODEL_PATH} -- run scripts/train.py first."
        )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu")

    gene_branch.load_state_dict(checkpoint["GeneBranch"])
    wsi_branch.load_state_dict(checkpoint["WSIBranch"])
    wsi_reduction.load_state_dict(checkpoint["WSIReduction"])
    fusion.load_state_dict(checkpoint["Fusion"])
    risk_head.load_state_dict(checkpoint["SurvivalRiskHead"])

    # ------------------------------------------------------------
    # 4. Recover / report the saved best epoch
    # ------------------------------------------------------------
    best_epoch = checkpoint.get("epoch", None)
    checkpoint_total_loss = checkpoint.get("total_loss", None)
    checkpoint_c_index = checkpoint.get("c_index_mini_scale_exploratory", None)

    print(f"\nLoaded checkpoint: {BEST_MODEL_PATH}")
    print(f"Checkpoint epoch (best total_loss seen): {best_epoch}")
    if checkpoint_total_loss is not None:
        print(f"Checkpoint-time total_loss: {checkpoint_total_loss:.6f}")
    if checkpoint_c_index is not None:
        print(f"Checkpoint-time C_index (mini-scale, exploratory): {checkpoint_c_index:.6f}")

    # ------------------------------------------------------------
    # 5. Inference on all 15 patients, eval mode, no grad
    # ------------------------------------------------------------
    for m in trainable_modules:
        m.eval()

    loss_fn = SurvivalLoss(weight_decay=5e-4, alpha=1.0, beta=0.8)

    with torch.no_grad():
        R, L3 = forward_pipeline(modules, gene_input, wsi_input, patch_mask)
        total_loss, diagnostics = loss_fn(R, T, S, L3, trainable_modules)
        c_index = concordance_index(R.squeeze(-1), T, S)

    # ------------------------------------------------------------
    # 6. Print all computed values
    # ------------------------------------------------------------
    print("\n--- Re-evaluated on all 15 patients (this run) ---")
    print(f"risk scores shape: {tuple(R.shape)}")
    print(f"L1          = {diagnostics['L1'].item():.6f}")
    print(f"L2_raw      = {diagnostics['L2_raw'].item():.6f}")
    print(f"L2_weighted = {diagnostics['L2_weighted'].item():.6f}")
    print(f"L3          = {L3.item():.6f}")
    print(f"L3_weighted = {diagnostics['L3_weighted'].item():.6f}")
    print(f"total_loss  = {total_loss.item():.6f}")
    print(f"C_index     = {c_index:.6f}")

    # ------------------------------------------------------------
    # 7. Save best_risk_scores.csv, patient/label/score alignment preserved
    # ------------------------------------------------------------
    risk_scores = R.squeeze(-1).tolist()

    TRAINING_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_RISK_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "OS_MONTHS", "OS_STATUS", "risk_score"])
        for pid, months, status, risk in zip(patient_ids, T.tolist(), os_status_strings, risk_scores):
            writer.writerow([pid, months, status, risk])

    print(f"\nBest-checkpoint risk scores saved to: {BEST_RISK_CSV}")

    # ------------------------------------------------------------
    # 8-9. Final summary -- explicit, unambiguous labeling
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BEST TRAINING-LOSS CHECKPOINT -- TRAINING-SET / EXPLORATORY")
    print("This is NOT validation performance and NOT generalization")
    print("performance. All 15 patients shown here were also used to")
    print("train this checkpoint. No held-out patients exist in this")
    print("15-patient mini-scale project.")
    print("=" * 60)


if __name__ == "__main__":
    main()