"""
Attention-based WSI patch ranking for the PAMT mini-scale model.

Uses the EXISTING best_model.pt and the actual Stage 5A cross-attention
weights returned by PathwayToPatchFusion.

Attention tensor:
    (B, 16 heads, 186 pathways, N patches)

For each patient, a global patch score is computed as:
    mean over attention heads and pathway queries

Padded WSI positions are excluded.

Outputs:
    outputs/attention_patch_ranking/patch_attention_scores.csv
    outputs/attention_patch_ranking/top_patches.csv
    outputs/attention_patch_ranking/attention_summary.csv
    outputs/attention_patch_ranking/top10_attention_heatmap.png
    outputs/attention_patch_ranking/<patient>_top_patches.png

Run from the project root:
    python scripts/attention_patch_ranking.py

Optional:
    python scripts/attention_patch_ranking.py --top_k 10
    python scripts/attention_patch_ranking.py --patient TCGA-2F-A9KO
"""

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


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
    build_modules,
    BEST_MODEL_PATH,
    WSI_DIR,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "attention_patch_ranking"
ALL_SCORES_CSV = OUTPUT_DIR / "patch_attention_scores.csv"
TOP_PATCHES_CSV = OUTPUT_DIR / "top_patches.csv"
SUMMARY_CSV = OUTPUT_DIR / "attention_summary.csv"
HEATMAP_PNG = OUTPUT_DIR / "top10_attention_heatmap.png"


def extract_patch_array(npz_path):
    """Use the same patch-array identification rule as train.py."""
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


def find_wsi_file(patient_id):
    matches = list(WSI_DIR.glob(f"{patient_id}-*_patches.npz"))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No WSI patch file found for {patient_id} under {WSI_DIR}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple WSI patch files found for {patient_id}: "
            f"{matches}"
        )

    return matches[0]


def load_coords(npz_path, expected_n):
    """
    Try to recover patch coordinates from the same NPZ.

    The ranking itself does NOT depend on coordinates. If the NPZ does
    not contain coords, x/y are left blank in the CSV.
    """
    with np.load(npz_path) as npz:
        if "coords" not in npz.files:
            return [None] * expected_n

        coords = np.asarray(npz["coords"])

    if coords.ndim != 2 or coords.shape[0] != expected_n or coords.shape[1] < 2:
        return [None] * expected_n

    return [
        (float(coords[i, 0]), float(coords[i, 1]))
        for i in range(expected_n)
    ]


def load_best_model():
    modules, trainable_modules = build_modules()

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_MODEL_PATH}"
        )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location="cpu",
    )

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


def make_patient_figure(
    patient_df,
    patient_id,
    output_path,
    top_k,
):
    valid = patient_df.sort_values(
        "patch_rank",
        ascending=True,
    )

    coords_available = (
        valid["x"].notna().all()
        and valid["y"].notna().all()
    )

    plt.figure(figsize=(8, 6))

    if coords_available:
        scatter = plt.scatter(
            valid["x"],
            valid["y"],
            c=valid["attention_score"],
            s=40,
        )
        plt.colorbar(scatter, label="Mean attention")
        plt.xlabel("Patch X coordinate")
        plt.ylabel("Patch Y coordinate")
        plt.title(
            f"{patient_id} — attention-based patch ranking"
        )

        top = valid.head(top_k)
        plt.scatter(
            top["x"],
            top["y"],
            s=90,
            facecolors="none",
            edgecolors="black",
        )

        for _, row in top.iterrows():
            plt.annotate(
                f"#{int(row['patch_rank'])}",
                (row["x"], row["y"]),
                xytext=(4, 4),
                textcoords="offset points",
            )

    else:
        top = valid.head(top_k)
        plt.bar(
            top["patch_rank"].astype(str),
            top["attention_score"],
        )
        plt.xlabel("Patch rank")
        plt.ylabel("Mean attention")
        plt.title(
            f"{patient_id} — top attention-ranked patches"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top patches per patient to save.",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help=(
            "Optional patient ID for the detailed figure. "
            "Default: first patient."
        ),
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("PAMT MINI-SCALE — ATTENTION-BASED PATCH RANKING")
    print("=" * 72)

    # -------------------------------------------------------------
    # 1. Load exact project batch
    # -------------------------------------------------------------
    patient_ids, gene_arrays, wsi_arrays = load_matched_patients()

    gene_input, wsi_input, patch_mask = build_batch(
        gene_arrays,
        wsi_arrays,
    )

    # -------------------------------------------------------------
    # 2. Load exact best checkpoint
    # -------------------------------------------------------------
    modules, checkpoint = load_best_model()

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
    print(f"Patients: {len(patient_ids)}")

    # -------------------------------------------------------------
    # 3. Recompute Stage 5A attention
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

        CA, attn = fusion(
            xP,
            xI,
            mask_out,
        )

    expected_shape = (
        len(patient_ids),
        fusion.num_heads,
        186,
        wsi_input.shape[1],
    )

    if tuple(attn.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected attention shape: {tuple(attn.shape)}; "
            f"expected {expected_shape}"
        )

    # -------------------------------------------------------------
    # 4. Global patch attention
    #
    # attn: (B, heads, pathways, patches)
    # mean over heads and pathways -> (B, patches)
    # -------------------------------------------------------------
    patch_attention = attn.mean(dim=(1, 2))

    all_rows = []
    top_rows = []
    summary_rows = []

    for patient_index, patient_id in enumerate(patient_ids):
        real_n = int((~mask_out[patient_index]).sum().item())

        scores = patch_attention[
            patient_index,
            :real_n,
        ].cpu().numpy()

        # Numerical sanity check.
        score_sum = float(scores.sum())

        if not np.isclose(
            score_sum,
            1.0,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"{patient_id}: valid patch attention sums to "
                f"{score_sum:.8f}, not 1."
            )

        order = np.argsort(
            -scores,
            kind="stable",
        )

        wsi_path = find_wsi_file(patient_id)
        coords = load_coords(
            wsi_path,
            real_n,
        )

        ranked_rows = []

        for rank_zero, patch_index in enumerate(order):
            rank = rank_zero + 1
            coord = coords[int(patch_index)]

            x = None if coord is None else coord[0]
            y = None if coord is None else coord[1]

            row = {
                "patient_id": patient_id,
                "patch_index": int(patch_index),
                "patch_rank": rank,
                "attention_score": float(
                    scores[int(patch_index)]
                ),
                "x": x,
                "y": y,
            }

            all_rows.append(row)
            ranked_rows.append(row)

            if rank <= args.top_k:
                top_rows.append(row)

        top_scores = scores[order]

        summary = {
            "patient_id": patient_id,
            "num_real_patches": real_n,
            "top1_attention_mass": float(top_scores[:1].sum()),
            "top3_attention_mass": float(
                top_scores[: min(3, real_n)].sum()
            ),
            "top5_attention_mass": float(
                top_scores[: min(5, real_n)].sum()
            ),
            "top10_attention_mass": float(
                top_scores[: min(10, real_n)].sum()
            ),
            "mean_attention": float(scores.mean()),
            "max_attention": float(scores.max()),
        }

        summary_rows.append(summary)

    all_df = pd.DataFrame(all_rows)
    top_df = pd.DataFrame(top_rows)
    summary_df = pd.DataFrame(summary_rows)

    all_df.to_csv(
        ALL_SCORES_CSV,
        index=False,
    )
    top_df.to_csv(
        TOP_PATCHES_CSV,
        index=False,
    )
    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # -------------------------------------------------------------
    # 5. Heatmap: top-10 attention mass per patient
    # -------------------------------------------------------------
    heatmap = np.zeros(
        (len(patient_ids), args.top_k),
        dtype=float,
    )

    for i, patient_id in enumerate(patient_ids):
        scores = (
            top_df.loc[
                top_df["patient_id"] == patient_id,
                "attention_score",
            ]
            .to_numpy()
        )

        heatmap[i, : len(scores)] = scores[: args.top_k]

    plt.figure(
        figsize=(
            max(8, args.top_k * 0.8),
            max(5, len(patient_ids) * 0.35),
        )
    )
    plt.imshow(
        heatmap,
        aspect="auto",
    )
    plt.colorbar(label="Attention score")
    plt.xticks(
        range(args.top_k),
        [f"#{i}" for i in range(1, args.top_k + 1)],
    )
    plt.yticks(
        range(len(patient_ids)),
        patient_ids,
    )
    plt.xlabel("Patch rank")
    plt.ylabel("Patient")
    plt.title("Top patch attention scores")
    plt.tight_layout()
    plt.savefig(
        HEATMAP_PNG,
        dpi=180,
    )
    plt.close()

    # -------------------------------------------------------------
    # 6. Detailed figure for one patient
    # -------------------------------------------------------------
    selected_patient = args.patient or patient_ids[0]

    if selected_patient not in patient_ids:
        raise ValueError(
            f"Unknown patient {selected_patient}. "
            f"Choose one of: {patient_ids}"
        )

    patient_df = all_df[
        all_df["patient_id"] == selected_patient
    ].copy()

    figure_path = (
        OUTPUT_DIR
        / f"{selected_patient}_top_patches.png"
    )

    make_patient_figure(
        patient_df,
        selected_patient,
        figure_path,
        args.top_k,
    )

    # -------------------------------------------------------------
    # 7. Console summary
    # -------------------------------------------------------------
    print("\nAttention tensor:")
    print(f"  {tuple(attn.shape)}")

    print("\nPatch ranking:")
    print(
        "  score = mean attention over 16 heads and 186 "
        "pathway queries"
    )
    print(
        "  padded patches are excluded before ranking"
    )

    print("\nTop-patch concentration:")
    print(
        summary_df[
            [
                "patient_id",
                "num_real_patches",
                "top1_attention_mass",
                "top3_attention_mass",
                "top5_attention_mass",
                "top10_attention_mass",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {ALL_SCORES_CSV}")
    print(f"  {TOP_PATCHES_CSV}")
    print(f"  {SUMMARY_CSV}")
    print(f"  {HEATMAP_PNG}")
    print(f"  {figure_path}")


if __name__ == "__main__":
    main()
