"""
Stage 7 -- Training loop (revised).

Trains the full Stage 1-5B pipeline against SurvivalLoss (Stage 6) using
ALL 15 real patients as a single full batch (no train/val split), because
the Cox partial-likelihood risk set is defined patient-to-patient within
one batch -- splitting the batch would compute a smaller, less faithful
risk set.

C-index is computed on the SAME 15 patients the model was trained on. This
is therefore a MINI-SCALE / EXPLORATORY metric only -- it reflects how well
the model fits the training data, NOT generalization to unseen patients.
It is never reported or treated as a statistically meaningful validation
result anywhere in this script or its outputs.

Outputs:
    data/processed/training/training_history.csv   (per-epoch, every epoch)
    data/processed/training/final_risk_scores.csv   (15 patients, final eval)
    data/processed/model_checkpoints/best_model.pt   (lowest train total_loss seen)
    data/processed/model_checkpoints/final_model.pt  (last epoch)

Scope note: with only 15 real patients and no WSI-resampling augmentation
(the paper uses G=80x augmented resampling per epoch; this project uses
fixed, precomputed patch selections per patient), this is a toy-scale
demonstration of the training mechanics, not a reproduction of the paper's
reported performance.
"""

from pathlib import Path
import sys
import csv
import argparse

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, "src")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gene_branch.gene_branch import GeneBranch
from wsi_branch.wsi_branch import WSIBranch
from wsi_branch.wsi_reduction import WSIReduction
from fusion import PathwayToPatchFusion
from scripts.risk_head import SurvivalRiskHead
from gene_branch.contrastive_loss import PathwayPatchContrastiveLoss
from scripts.survival_loss import SurvivalLoss, load_os_labels
from c_index import concordance_index


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
GENE_DIR = Path("data/processed/gene")
WSI_DIR = Path("data/processed/wsi")
CLINICAL_PATIENT_PATH = (
    "data/raw/blca_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
)

TRAINING_OUT_DIR = Path("data/processed/training")
CHECKPOINT_DIR = Path("data/processed/model_checkpoints")

HISTORY_CSV = TRAINING_OUT_DIR / "training_history.csv"
FINAL_RISK_CSV = TRAINING_OUT_DIR / "final_risk_scores.csv"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_model.pt"

# Confirmed patient ordering (matches test_survival_loss.py) -- preserved
# throughout, including in final_risk_scores.csv.
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

SEED = 42


# ------------------------------------------------------------
# Data loading (same logic/contract as test_survival_loss.py)
# ------------------------------------------------------------
def extract_patch_array(npz_path):
    with np.load(npz_path) as npz:
        keys = list(npz.files)

        if len(keys) == 1:
            return npz[keys[0]]

        candidates = [
            k
            for k in keys
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


def load_matched_patients():
    gene_files = sorted(GENE_DIR.glob("*.npy"))

    if not gene_files:
        raise FileNotFoundError(
            f"No gene .npy files found under {GENE_DIR}"
        )

    wsi_files = list(WSI_DIR.glob("*_patches.npz"))

    if not wsi_files:
        raise FileNotFoundError(
            f"No *_patches.npz files found under {WSI_DIR}"
        )

    patient_ids = []
    gene_arrays = []
    wsi_arrays = []

    for gene_file in gene_files:
        patient_id = gene_file.stem

        matches = [
            f
            for f in wsi_files
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

        gene_array = np.load(gene_file)
        wsi_array = extract_patch_array(matches[0])

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

    if patient_ids != EXPECTED_PATIENT_IDS:
        raise RuntimeError(
            f"Patient ordering does not match the confirmed order.\n"
            f"Got:      {patient_ids}\n"
            f"Expected: {EXPECTED_PATIENT_IDS}"
        )

    return patient_ids, gene_arrays, wsi_arrays


def build_batch(gene_arrays, wsi_arrays):
    B = len(gene_arrays)

    gene_input = torch.tensor(
        np.stack(gene_arrays, axis=0),
        dtype=torch.float32,
    )

    max_patches = max(
        a.shape[0]
        for a in wsi_arrays
    )

    feature_dim = wsi_arrays[0].shape[1]

    wsi_input = torch.zeros(
        B,
        max_patches,
        feature_dim,
        dtype=torch.float32,
    )

    patch_mask = torch.ones(
        B,
        max_patches,
        dtype=torch.bool,
    )  # True = padding

    for i, patches in enumerate(wsi_arrays):
        n = patches.shape[0]

        wsi_input[i, :n, :] = torch.tensor(
            patches,
            dtype=torch.float32,
        )

        patch_mask[i, :n] = False

    return gene_input, wsi_input, patch_mask


def load_os_status_strings(clinical_patient_path, patient_ids):
    """
    Re-read the raw OS_STATUS string (e.g. '1:DECEASED') per patient, for
    final_risk_scores.csv -- load_os_labels() already converts this to a
    0/1 float, but the CSV spec asks for OS_STATUS as given, so we read the
    raw field separately rather than reconstructing it from the 0/1 value.
    """

    with open(clinical_patient_path, "r", newline="") as f:
        raw_lines = f.readlines()

    header_idx = None

    for i, line in enumerate(raw_lines):
        if line.split("\t", 1)[0].strip() == "PATIENT_ID":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Could not find PATIENT_ID header in clinical file."
        )

    header = (
        raw_lines[header_idx]
        .rstrip("\n")
        .split("\t")
    )

    col = {
        name: i
        for i, name in enumerate(header)
    }

    rows = {}

    reader = csv.reader(
        raw_lines[header_idx + 1:],
        delimiter="\t",
    )

    for fields in reader:
        if not fields or fields[0].startswith("#"):
            continue

        if len(fields) <= max(col.values()):
            continue

        rows[fields[col["PATIENT_ID"]]] = fields

    return [
        rows[pid][col["OS_STATUS"]].strip()
        for pid in patient_ids
    ]


# ------------------------------------------------------------
# Forward pass through the full Stage 1-5B pipeline
# ------------------------------------------------------------
def forward_pipeline(
    modules,
    gene_input,
    wsi_input,
    patch_mask,
):
    (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
        contrastive_loss_fn,
    ) = modules

    xP = gene_branch(gene_input)

    xI_384 = wsi_branch(
        wsi_input,
        patch_mask,
    )

    xI, mask_out = wsi_reduction(
        xI_384,
        patch_mask,
    )

    CA, _attn = fusion(
        xP,
        xI,
        mask_out,
    )

    R = risk_head(
        xP,
        CA,
    )

    L3, _info = contrastive_loss_fn(
        xP,
        xI,
        mask_out,
    )

    return R, L3


def build_modules():
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

    modules = (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
        contrastive_loss_fn,
    )

    trainable_modules = [
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
    ]

    return modules, trainable_modules


def save_checkpoint(
    path,
    trainable_modules,
    epoch,
    total_loss,
    c_index,
):
    (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
    ) = trainable_modules

    torch.save(
        {
            "epoch": epoch,
            "total_loss": float(total_loss),
            "c_index_mini_scale_exploratory": (
                float(c_index)
                if not np.isnan(c_index)
                else None
            ),
            "GeneBranch": gene_branch.state_dict(),
            "WSIBranch": wsi_branch.state_dict(),
            "WSIReduction": wsi_reduction.state_dict(),
            "Fusion": fusion.state_dict(),
            "SurvivalRiskHead": risk_head.state_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    # pamt_training config block (mirrors config.yaml -- see README)
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if args.optimizer.lower() != "adamw":
        raise ValueError(
            "Only 'adamw' is currently supported by this script."
        )

    if args.weight_decay != 0.0:
        raise ValueError(
            "weight_decay must be 0.0 -- SurvivalLoss already applies "
            "the paper's L2 regularization manually; setting it on the "
            "optimizer too would double-apply it."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    TRAINING_OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------
    # 1. Load ALL 15 real patients as one full batch
    # ----------------------------------------------------------------
    patient_ids, gene_arrays, wsi_arrays = load_matched_patients()

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

    print(
        f"Loaded {len(patient_ids)} patients "
        f"(full batch, no split): {patient_ids}"
    )

    print(
        f"Gene batch shape: {tuple(gene_input.shape)}"
    )

    print(
        f"WSI batch shape:  {tuple(wsi_input.shape)}"
    )

    # ----------------------------------------------------------------
    # 2. Build model + loss + optimizer
    # ----------------------------------------------------------------
    modules, trainable_modules = build_modules()

    loss_fn = SurvivalLoss(
        weight_decay=5e-4,
        alpha=1.0,
        beta=0.8,
    )

    all_params = [
        p
        for m in trainable_modules
        for p in m.parameters()
    ]

    optimizer = optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=0.0,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0.0,
    )

    # ----------------------------------------------------------------
    # 3. Training loop
    # ----------------------------------------------------------------
    best_total_loss = float("inf")

    # IMPORTANT:
    # Keep the entire epoch loop INSIDE this with-block so that
    # writer/f remain open while every epoch is written.
    with open(
        HISTORY_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "learning_rate",

            "train_L1",
            "train_L2_raw",
            "train_L2_weighted",
            "train_L3",
            "train_L3_weighted",
            "train_total_loss",

            "eval_L1",
            "eval_L2_raw",
            "eval_L2_weighted",
            "eval_L3",
            "eval_L3_weighted",
            "eval_total_loss",
            "eval_C_index",
        ])

        # ------------------------------------------------------------
        # Epoch loop
        # ------------------------------------------------------------
        for epoch in range(
            1,
            args.epochs + 1,
        ):

            # --------------------------------------------------------
            # Train-mode pass
            # Dropout is active here.
            # --------------------------------------------------------
            for m in trainable_modules:
                m.train()

            optimizer.zero_grad()

            R_train, L3_train = forward_pipeline(
                modules,
                gene_input,
                wsi_input,
                patch_mask,
            )

            train_total_loss, train_diag = loss_fn(
                R_train,
                T,
                S,
                L3_train,
                trainable_modules,
            )

            train_total_loss.backward()

            optimizer.step()

            current_lr = scheduler.get_last_lr()[0]

            scheduler.step()

            # --------------------------------------------------------
            # Eval-mode pass
            # Dropout disabled.
            # This gives deterministic metrics.
            # --------------------------------------------------------
            for m in trainable_modules:
                m.eval()

            with torch.no_grad():

                R_eval, L3_eval = forward_pipeline(
                    modules,
                    gene_input,
                    wsi_input,
                    patch_mask,
                )

                eval_total_loss, eval_diag = loss_fn(
                    R_eval,
                    T,
                    S,
                    L3_eval,
                    trainable_modules,
                )

                eval_c_idx = concordance_index(
                    R_eval.squeeze(-1),
                    T,
                    S,
                )

            # Restore train mode before the next epoch.
            for m in trainable_modules:
                m.train()

            # --------------------------------------------------------
            # Save train-mode + eval-mode metrics
            # --------------------------------------------------------
            writer.writerow([
                epoch,
                current_lr,

                # Train-mode metrics
                train_diag["L1"].item(),
                train_diag["L2_raw"].item(),
                train_diag["L2_weighted"].item(),
                L3_train.item(),
                train_diag["L3_weighted"].item(),
                train_total_loss.item(),

                # Eval-mode metrics
                eval_diag["L1"].item(),
                eval_diag["L2_raw"].item(),
                eval_diag["L2_weighted"].item(),
                L3_eval.item(),
                eval_diag["L3_weighted"].item(),
                eval_total_loss.item(),
                eval_c_idx,
            ])

            f.flush()

            # --------------------------------------------------------
            # Best checkpoint selection
            #
            # IMPORTANT:
            # "Best" means LOWEST TRAINING-MODE TOTAL LOSS.
            # It is NOT a validation-best checkpoint.
            # --------------------------------------------------------
            if train_total_loss.item() < best_total_loss:

                best_total_loss = train_total_loss.item()

                save_checkpoint(
                    BEST_MODEL_PATH,
                    trainable_modules,
                    epoch,
                    train_total_loss.item(),
                    eval_c_idx,
                )

            # --------------------------------------------------------
            # Console logging
            # --------------------------------------------------------
            if (
                epoch % args.log_every == 0
                or epoch == 1
                or epoch == args.epochs
            ):

                print(
                    f"epoch {epoch:4d} | "
                    f"lr {current_lr:.6f} | "

                    f"train_total "
                    f"{train_total_loss.item():.4f} "
                    f"(L1={train_diag['L1'].item():.4f}, "
                    f"L2={train_diag['L2_weighted'].item():.6f}, "
                    f"L3={train_diag['L3_weighted'].item():.4f}) | "

                    f"eval_total "
                    f"{eval_total_loss.item():.4f} "
                    f"(L1={eval_diag['L1'].item():.4f}, "
                    f"L2={eval_diag['L2_weighted'].item():.6f}, "
                    f"L3={eval_diag['L3_weighted'].item():.4f}) | "

                    f"eval_C_index "
                    f"{eval_c_idx:.4f}"
                )

    # ----------------------------------------------------------------
    # Save final checkpoint
    # ----------------------------------------------------------------
    save_checkpoint(
        FINAL_MODEL_PATH,
        trainable_modules,
        args.epochs,
        train_total_loss.item(),
        eval_c_idx,
    )

    # ----------------------------------------------------------------
    # 4. Final evaluation
    # Eval mode, all 15 patients, final/last epoch weights.
    # ----------------------------------------------------------------
    for m in trainable_modules:
        m.eval()

    with torch.no_grad():

        R_final, L3_final = forward_pipeline(
            modules,
            gene_input,
            wsi_input,
            patch_mask,
        )

        final_total_loss, final_diag = loss_fn(
            R_final,
            T,
            S,
            L3_final,
            trainable_modules,
        )

        final_c_index = concordance_index(
            R_final.squeeze(-1),
            T,
            S,
        )

    print("\n" + "=" * 60)

    print(
        "FINAL EVALUATION "
        "(all 15 patients, final-epoch weights)"
    )

    print(
        "MINI-SCALE / EXPLORATORY -- trained and evaluated "
        "on the same"
    )

    print(
        "15 patients. This is NOT a validation or "
        "generalization result."
    )

    print("=" * 60)

    print(
        f"total_loss = "
        f"{final_total_loss.item():.6f}"
    )

    print(
        f"L1         = "
        f"{final_diag['L1'].item():.6f}"
    )

    print(
        f"L2_raw     = "
        f"{final_diag['L2_raw'].item():.6f}"
    )

    print(
        f"L2_weighted= "
        f"{final_diag['L2_weighted'].item():.6f}"
    )

    print(
        f"L3         = "
        f"{L3_final.item():.6f}"
    )

    print(
        f"L3_weighted= "
        f"{final_diag['L3_weighted'].item():.6f}"
    )

    print(
        f"C_index    = "
        f"{final_c_index:.6f}  "
        f"(mini-scale, exploratory)"
    )

    # ----------------------------------------------------------------
    # 5. Save final risk scores per patient
    # Preserving confirmed patient order.
    # ----------------------------------------------------------------
    risk_scores = R_final.squeeze(-1).tolist()

    with open(
        FINAL_RISK_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "patient_id",
            "OS_MONTHS",
            "OS_STATUS",
            "risk_score",
        ])

        for pid, months, status, risk in zip(
            patient_ids,
            T.tolist(),
            os_status_strings,
            risk_scores,
        ):

            writer.writerow([
                pid,
                months,
                status,
                risk,
            ])

    print(
        f"\nTraining history saved to:   "
        f"{HISTORY_CSV}"
    )

    print(
        f"Final risk scores saved to:  "
        f"{FINAL_RISK_CSV}"
    )

    print(
        f"Best checkpoint saved to:    "
        f"{BEST_MODEL_PATH} "
        f"(lowest total_loss={best_total_loss:.6f})"
    )

    print(
        f"Final checkpoint saved to:   "
        f"{FINAL_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()