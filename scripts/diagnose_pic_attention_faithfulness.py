"""
TEMPORARY DIAGNOSTIC — PIC-checkpoint pathway-attention faithfulness

Purpose
-------
This script does NOT change PAMT, retrain the model, or modify any existing
Stage 1-7 source file. It is a diagnostic to decide whether PAMT's
pathway-to-patch attention is actually associated with the survival-risk
prediction.

For each patient and pathway:
  1. Run the trained PAMT checkpoint normally and obtain Stage-5A attention.
  2. Select the top-attended WSI patches for that pathway.
  3. Re-run the model after treating those patches as removed/padded.
  4. Measure the absolute change in the patient's predicted risk.
  5. Repeat with a random set of the same number of patches as a control.

The main diagnostic is whether attention-guided deletion changes risk more
than matched random deletion.

Default intervention: remove the top 10% of valid patches for each pathway.
This is intentionally a DIAGNOSTIC ONLY. If the results show a meaningful
faithfulness gap, a separate training loss can be designed afterward.

Run from project root:
    python scripts/diagnose_attention_faithfulness.py

Optional:
    python scripts/diagnose_attention_faithfulness.py --fraction 0.05
    python scripts/diagnose_attention_faithfulness.py --fraction 0.20
    python scripts/diagnose_attention_faithfulness.py --batch_size 4

Outputs:
    outputs/pic_faithfulness_diagnostic/pathway_faithfulness.csv
    outputs/pic_faithfulness_diagnostic/patient_summary.csv
    outputs/pic_faithfulness_diagnostic/overall_summary.csv
"""

from pathlib import Path
import argparse
import random
import sys

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------
# Project imports — reuse the exact model/data construction already used by
# train.py and attention_patch_ranking.py. No duplicate model definitions.
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import (  # noqa: E402
    BEST_MODEL_PATH,
    build_batch,
    build_modules,
    load_matched_patients,
)


PIC_CHECKPOINT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "pic_infonce_finetune" / "best_pic_combined.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pic_faithfulness_diagnostic"
PATHWAY_COUNT = 186
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_best_model():
    """Load the exact Stage 1-5B modules from the final PIC checkpoint."""
    modules, trainable_modules = build_modules()

    if not PIC_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"PIC checkpoint not found at {PIC_CHECKPOINT_PATH}. "
            "Confirm that scripts/train_pic_infonce_finetune.py completed successfully."
        )

    checkpoint = torch.load(PIC_CHECKPOINT_PATH, map_location="cpu")

    gene_branch, wsi_branch, wsi_reduction, fusion, risk_head = (
        trainable_modules
    )

    gene_branch.load_state_dict(checkpoint["GeneBranch"])
    wsi_branch.load_state_dict(checkpoint["WSIBranch"])
    wsi_reduction.load_state_dict(checkpoint["WSIReduction"])
    fusion.load_state_dict(checkpoint["Fusion"])
    risk_head.load_state_dict(checkpoint["SurvivalRiskHead"])

    for module in trainable_modules:
        module.eval()

    return modules, checkpoint


def forward_from_precomputed_gene(
    xP: torch.Tensor,
    wsi_branch,
    wsi_reduction,
    fusion,
    risk_head,
    wsi_input: torch.Tensor,
    patch_mask: torch.Tensor,
):
    """Run the WSI -> reduction -> fusion -> risk path with fixed xP."""
    xI_384 = wsi_branch(wsi_input, patch_mask)
    xI, mask_out = wsi_reduction(xI_384, patch_mask)
    CA, attn = fusion(xP, xI, mask_out)
    R = risk_head(xP, CA)
    return R, attn, mask_out


def make_deleted_batch(
    wsi_input: torch.Tensor,
    original_mask: torch.Tensor,
    delete_indices,
):
    """Mark selected valid patch indices as padding for one patient."""
    new_input = wsi_input.clone()
    new_mask = original_mask.clone()

    if len(delete_indices) == 0:
        return new_input, new_mask

    idx = torch.tensor(delete_indices, dtype=torch.long)
    new_mask[idx] = True
    # Zeroing is not strictly required because masked positions are excluded
    # downstream, but doing so makes the intervention explicit and prevents
    # accidental use if a future module ignores the mask.
    new_input[idx] = 0.0
    return new_input, new_mask


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic for PIC-checkpoint pathway-attention faithfulness."
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Fraction of valid patches to remove per pathway (default: 0.10).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of pathway interventions evaluated together (default: 4).",
    )
    args = parser.parse_args()

    if not (0.0 < args.fraction < 1.0):
        raise ValueError("--fraction must be > 0 and < 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("PAMT MINI-SCALE — PIC CHECKPOINT ATTENTION FAITHFULNESS DIAGNOSTIC")
    print("=" * 76)
    print("DIAGNOSTIC ONLY: no training and no existing PAMT files are modified.")
    print(f"Deletion fraction: {args.fraction:.2%}")
    print(f"Checkpoint: {PIC_CHECKPOINT_PATH}")

    # -------------------------------------------------------------
    # 1. Load the exact 15-patient data and best checkpoint.
    # -------------------------------------------------------------
    patient_ids, gene_arrays, wsi_arrays = load_matched_patients()
    gene_input, wsi_input, patch_mask = build_batch(gene_arrays, wsi_arrays)
    modules, checkpoint = load_best_model()

    (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
        _contrastive_loss_fn,
    ) = modules

    print(f"Patients: {len(patient_ids)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")

    # -------------------------------------------------------------
    # 2. Baseline forward pass.
    #    Gene branch is computed once because it is unchanged by WSI deletion.
    # -------------------------------------------------------------
    with torch.no_grad():
        xP = gene_branch(gene_input)
        xI_384 = wsi_branch(wsi_input, patch_mask)
        xI, mask_out = wsi_reduction(xI_384, patch_mask)
        baseline_CA, baseline_attn = fusion(xP, xI, mask_out)
        baseline_R = risk_head(xP, baseline_CA).squeeze(-1)

    if baseline_attn.ndim != 4:
        raise RuntimeError(
            f"Unexpected attention tensor shape: {tuple(baseline_attn.shape)}"
        )

    B, H, P, N = baseline_attn.shape
    if P != PATHWAY_COUNT:
        raise RuntimeError(
            f"Expected {PATHWAY_COUNT} pathway queries, got {P}."
        )

    # Average heads, matching the project's existing attention interpretation.
    mean_attn = baseline_attn.mean(dim=1)  # (B, P, N)

    all_rows = []

    # -------------------------------------------------------------
    # 3. Patient-by-patient, pathway-by-pathway interventions.
    # -------------------------------------------------------------
    for patient_idx, patient_id in enumerate(patient_ids):
        valid_indices = torch.where(~mask_out[patient_idx])[0].tolist()
        n_valid = len(valid_indices)
        n_delete = max(1, int(np.ceil(args.fraction * n_valid)))

        if n_delete >= n_valid:
            raise RuntimeError(
                f"{patient_id}: intervention would remove all valid patches."
            )

        base_risk = float(baseline_R[patient_idx].item())

        print(
            f"\n[{patient_idx + 1:02d}/{B}] {patient_id}: "
            f"{n_valid} valid patches; removing {n_delete} per pathway"
        )

        # Build all intervention records first. Each record contains the
        # pathway-specific top-attention deletion and a matched random deletion.
        interventions = []

        for pathway_idx in range(P):
            scores = mean_attn[patient_idx, pathway_idx, valid_indices]
            order = torch.argsort(scores, descending=True).tolist()
            top_delete = [valid_indices[j] for j in order[:n_delete]]

            # Deterministic matched random control using a pathway-specific RNG.
            rng = np.random.default_rng(SEED + patient_idx * 1000 + pathway_idx)
            random_delete = rng.choice(
                np.asarray(valid_indices),
                size=n_delete,
                replace=False,
            ).tolist()

            top_mass = float(scores[torch.argsort(scores, descending=True)[:n_delete]].sum().item())

            interventions.append(
                {
                    "pathway_idx": pathway_idx,
                    "top_delete": top_delete,
                    "random_delete": random_delete,
                    "attention_mass_top": top_mass,
                }
            )

        # Evaluate in small batches to avoid exploding WSI attention memory.
        for start in range(0, len(interventions), args.batch_size):
            chunk = interventions[start : start + args.batch_size]

            top_inputs = []
            top_masks = []
            random_inputs = []
            random_masks = []

            for item in chunk:
                ti, tm = make_deleted_batch(
                    wsi_input[patient_idx],
                    patch_mask[patient_idx],
                    item["top_delete"],
                )
                ri, rm = make_deleted_batch(
                    wsi_input[patient_idx],
                    patch_mask[patient_idx],
                    item["random_delete"],
                )
                top_inputs.append(ti)
                top_masks.append(tm)
                random_inputs.append(ri)
                random_masks.append(rm)

            top_wsi = torch.stack(top_inputs, dim=0)
            top_mask = torch.stack(top_masks, dim=0)
            random_wsi = torch.stack(random_inputs, dim=0)
            random_mask = torch.stack(random_masks, dim=0)

            # xP is identical for every intervention in this patient.
            xP_chunk = xP[patient_idx].unsqueeze(0).expand(len(chunk), -1, -1)

            with torch.no_grad():
                top_R, _, _ = forward_from_precomputed_gene(
                    xP_chunk,
                    wsi_branch,
                    wsi_reduction,
                    fusion,
                    risk_head,
                    top_wsi,
                    top_mask,
                )
                random_R, _, _ = forward_from_precomputed_gene(
                    xP_chunk,
                    wsi_branch,
                    wsi_reduction,
                    fusion,
                    risk_head,
                    random_wsi,
                    random_mask,
                )

            top_risks = top_R.squeeze(-1).cpu().numpy()
            random_risks = random_R.squeeze(-1).cpu().numpy()

            for j, item in enumerate(chunk):
                top_risk = float(top_risks[j])
                random_risk = float(random_risks[j])
                top_delta = abs(base_risk - top_risk)
                random_delta = abs(base_risk - random_risk)

                all_rows.append(
                    {
                        "patient_id": patient_id,
                        "pathway_index": item["pathway_idx"],
                        "n_valid_patches": n_valid,
                        "n_deleted": n_delete,
                        "attention_mass_top": item["attention_mass_top"],
                        "baseline_risk": base_risk,
                        "attention_delete_risk": top_risk,
                        "attention_delete_abs_delta": top_delta,
                        "random_delete_risk": random_risk,
                        "random_delete_abs_delta": random_delta,
                        "faithfulness_gain": top_delta - random_delta,
                        "faithfulness_ratio": (
                            top_delta / random_delta
                            if random_delta > 1e-12
                            else np.nan
                        ),
                    }
                )

        patient_done = pd.DataFrame(
            [r for r in all_rows if r["patient_id"] == patient_id]
        )
        print(
            "  mean |Δrisk|: "
            f"attention={patient_done['attention_delete_abs_delta'].mean():.6f}, "
            f"random={patient_done['random_delete_abs_delta'].mean():.6f}"
        )

    # -------------------------------------------------------------
    # 4. Save raw pathway-level results.
    # -------------------------------------------------------------
    raw_df = pd.DataFrame(all_rows)
    raw_path = OUTPUT_DIR / "pathway_faithfulness.csv"
    raw_df.to_csv(raw_path, index=False)

    # -------------------------------------------------------------
    # 5. Patient summaries.
    # -------------------------------------------------------------
    patient_summary = (
        raw_df.groupby("patient_id", as_index=False)
        .agg(
            pathways=("pathway_index", "count"),
            mean_attention_mass_top=("attention_mass_top", "mean"),
            mean_attention_abs_delta=("attention_delete_abs_delta", "mean"),
            mean_random_abs_delta=("random_delete_abs_delta", "mean"),
            mean_faithfulness_gain=("faithfulness_gain", "mean"),
            median_faithfulness_gain=("faithfulness_gain", "median"),
            mean_faithfulness_ratio=("faithfulness_ratio", "mean"),
        )
    )
    patient_path = OUTPUT_DIR / "patient_summary.csv"
    patient_summary.to_csv(patient_path, index=False)

    # -------------------------------------------------------------
    # 6. Overall diagnostic summary.
    # -------------------------------------------------------------
    top = raw_df["attention_delete_abs_delta"].to_numpy()
    rnd = raw_df["random_delete_abs_delta"].to_numpy()
    gain = raw_df["faithfulness_gain"].to_numpy()

    overall = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "deletion_fraction": args.fraction,
        "patients": len(patient_ids),
        "pathways_per_patient": P,
        "total_pathway_interventions": len(raw_df),
        "mean_attention_abs_delta": float(np.mean(top)),
        "mean_random_abs_delta": float(np.mean(rnd)),
        "median_attention_abs_delta": float(np.median(top)),
        "median_random_abs_delta": float(np.median(rnd)),
        "mean_faithfulness_gain": float(np.mean(gain)),
        "median_faithfulness_gain": float(np.median(gain)),
        "fraction_attention_gt_random": float(np.mean(top > rnd)),
        "mean_faithfulness_ratio": float(
            np.nanmean(raw_df["faithfulness_ratio"].to_numpy())
        ),
    }

    # Spearman between attention mass and deletion effect. Use pandas ranks to
    # avoid adding scipy as a project dependency.
    ranked_a = pd.Series(raw_df["attention_mass_top"]).rank(method="average")
    ranked_d = pd.Series(raw_df["attention_delete_abs_delta"]).rank(method="average")
    spearman = ranked_a.corr(ranked_d)
    overall["spearman_attention_mass_vs_abs_delta"] = float(spearman)

    overall_df = pd.DataFrame([overall])
    overall_path = OUTPUT_DIR / "overall_summary.csv"
    overall_df.to_csv(overall_path, index=False)

    # -------------------------------------------------------------
    # 7. Human-readable conclusion (diagnostic, not a scientific claim).
    # -------------------------------------------------------------
    print("\n" + "=" * 76)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 76)
    print(f"Mean |Δrisk| — attention deletion : {overall['mean_attention_abs_delta']:.6f}")
    print(f"Mean |Δrisk| — random deletion    : {overall['mean_random_abs_delta']:.6f}")
    print(f"Mean attention-vs-random gain     : {overall['mean_faithfulness_gain']:.6f}")
    print(f"Median attention-vs-random gain   : {overall['median_faithfulness_gain']:.6f}")
    print(f"Attention > random (fraction)     : {overall['fraction_attention_gt_random']:.3f}")
    print(f"Spearman(attention mass, |Δrisk|) : {spearman:.6f}")
    print("\nInterpretation:")
    print("- Positive gain means attention-guided deletion changed risk more than")
    print("  a matched random deletion on average.")
    print("- This is evidence about the EXISTING model's attention behavior only.")
    print("- It is NOT proof of causal pathology importance and NOT a validation")
    print("  or test-set result.")
    print("- Do NOT add a new training loss yet. Use these results to decide")
    print("  whether a faithfulness-based extension is justified.")
    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {patient_path}")
    print(f"  {overall_path}")


if __name__ == "__main__":
    main()
