"""
Generate implementation-based visualizations for PAMT Stages 4–5B.

Creates:
    1. Stage 4 pathway-to-patch similarity heatmap
    2. Stage 5A pathway-to-patch cross-attention heatmap
    3. Stage 5B survival-risk output bar chart

Uses the SAME real 15-patient data and model architecture used
in the verified Stage 4, 5A and 5B tests.

IMPORTANT:
- Does not modify any model files.
- Uses one real patient for the heatmaps.
- Padded WSI patches are excluded from heatmaps.
- Risk values are shown only as verification outputs.
"""

from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================
# IMPORT VERIFIED MODULES
# ============================================================

from gene_branch.gene_branch import GeneBranch
from gene_branch.contrastive_loss import PathwayPatchContrastiveLoss

from wsi_branch.wsi_branch import WSIBranch
from wsi_branch.wsi_reduction import WSIReduction

from fusion import PathwayToPatchFusion
from scripts.risk_head import SurvivalRiskHead


# ============================================================
# DATA PATHS
# ============================================================

GENE_DIR = PROJECT_ROOT / "data" / "processed" / "gene"
WSI_DIR = PROJECT_ROOT / "data" / "processed" / "wsi"

OUTPUT_DIR = PROJECT_ROOT / "stage4_5b_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD / MATCH PATIENTS
# ============================================================

def match_patients(gene_dir: Path, wsi_dir: Path):
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


def load_batch(pairs):
    gene_arrays = []
    patch_arrays = []
    patch_counts = []
    patient_ids = []

    for patient_id, gene_file, wsi_file in pairs:

        gene = np.load(gene_file)

        z = np.load(wsi_file)

        # Your existing verified WSI files use "embeddings".
        if "embeddings" not in z:
            raise KeyError(
                f"'embeddings' not found in {wsi_file.name}. "
                f"Available keys: {list(z.keys())}"
            )

        patches = z["embeddings"]

        gene_arrays.append(gene)
        patch_arrays.append(patches)
        patch_counts.append(patches.shape[0])
        patient_ids.append(patient_id)

    # --------------------------------------------------------
    # Gene
    # --------------------------------------------------------

    x_gene = torch.from_numpy(
        np.stack(gene_arrays, axis=0)
    ).float()

    # --------------------------------------------------------
    # WSI padding
    # --------------------------------------------------------

    batch_size = len(patch_arrays)
    max_patches = max(patch_counts)
    patch_dim = patch_arrays[0].shape[1]

    padded = np.zeros(
        (batch_size, max_patches, patch_dim),
        dtype=np.float32,
    )

    mask = np.ones(
        (batch_size, max_patches),
        dtype=bool,
    )

    for i, patches in enumerate(patch_arrays):

        n = patches.shape[0]

        padded[i, :n] = patches
        mask[i, :n] = False

    x_wsi = torch.from_numpy(padded).float()
    patch_mask = torch.from_numpy(mask).bool()

    return (
        patient_ids,
        x_gene,
        x_wsi,
        patch_mask,
        patch_counts,
    )


# ============================================================
# BUILD VERIFIED PIPELINE
# ============================================================

def run_pipeline(
    x_gene,
    x_wsi,
    patch_mask,
):
    """
    Runs:

    GENE
      ↓
    xP

    WSI
      ↓
    xI_384
      ↓
    WSIReduction
      ↓
    xI

    xP + xI
      ↓
    L3

    xP + xI
      ↓
    Fusion
      ↓
    CA

    xP + CA
      ↓
    Risk Head
      ↓
    R
    """

    gene_branch = GeneBranch()

    wsi_branch = WSIBranch(
        embed_dim=384,
        max_patches=162,
        depth=6,
        num_heads=16,
    )

    wsi_reduction = WSIReduction(
        in_features=384,
        hidden_features=640,
        out_features=256,
        drop_rate=0.1,
    )

    contrastive_loss = PathwayPatchContrastiveLoss(
        top_h=2,
    )

    fusion = PathwayToPatchFusion(
        dim=256,
        num_heads=16,
    )

    risk_head = SurvivalRiskHead(
        num_pathway_tokens=186,
        num_classes=1,
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    gene_branch.eval()
    wsi_branch.eval()
    wsi_reduction.eval()
    contrastive_loss.eval()
    fusion.eval()
    risk_head.eval()

    with torch.no_grad():

        # Stage 2
        xP = gene_branch(x_gene)

        # Stage 3
        xI_384 = wsi_branch(
            x_wsi,
            key_padding_mask=patch_mask,
        )

        # Stage 3.5
        xI = wsi_reduction(
            xI_384,
            key_padding_mask=patch_mask,
        )

        # Depending on your verified WSIReduction implementation,
        # it may return either xI or (xI, mask).
        if isinstance(xI, tuple):
            xI = xI[0]

        # Stage 4
        L3, info = contrastive_loss(
            xP,
            xI,
            patch_mask,
        )

        # Stage 5A
        CA, attn = fusion(
            xP,
            xI,
            patch_mask,
        )

        # Stage 5B
        R = risk_head(
            xP,
            CA,
        )

    return {
        "xP": xP,
        "xI_384": xI_384,
        "xI": xI,
        "L3": L3,
        "info": info,
        "CA": CA,
        "attn": attn,
        "R": R,
    }


# ============================================================
# FIGURE 1 — STAGE 4 SIMILARITY HEATMAP
# ============================================================

def generate_stage4_heatmap(
    patient_id,
    info,
    patch_mask,
    patient_index,
    patch_count,
):
    """
    Visualizes S_prime for one real patient.

    Rows:
        186 pathways

    Columns:
        valid WSI patches only

    Values:
        normalized pathway-to-patch similarity
    """

    S_prime = info["S_prime"][patient_index].detach().cpu().numpy()

    Y_h = info["Y_h"][patient_index].detach().cpu().numpy()

    # Only real patches
    valid_n = patch_count

    S_valid = S_prime[:, :valid_n]
    Y_valid = Y_h[:, :valid_n]

    plt.figure(figsize=(12, 8))

    plt.imshow(
        S_valid,
        aspect="auto",
        interpolation="nearest",
    )

    plt.xlabel("WSI Patch")
    plt.ylabel("Pathway")
    plt.title(
        f"Stage 4 — Pathway-to-Patch Similarity\n"
        f"{patient_id} ({valid_n} valid patches)"
    )

    plt.colorbar(
        label="S′ similarity probability"
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "stage4_similarity_heatmap.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Optional second visualization: top-h selections
    # --------------------------------------------------------

    plt.figure(figsize=(12, 8))

    plt.imshow(
        S_valid,
        aspect="auto",
        interpolation="nearest",
    )

    # Plot top-h positive positions
    positive_rows, positive_cols = np.where(Y_valid > 0)

    plt.scatter(
        positive_cols,
        positive_rows,
        marker="o",
        facecolors="none",
        edgecolors="white",
        s=18,
        linewidths=0.7,
    )

    plt.xlabel("WSI Patch")
    plt.ylabel("Pathway")

    plt.title(
        f"Stage 4 — Top-2 Pathway-to-Patch Selections\n"
        f"{patient_id}"
    )

    plt.colorbar(
        label="S′ similarity probability"
    )

    plt.tight_layout()

    output_topk = OUTPUT_DIR / "stage4_top2_selections.png"

    plt.savefig(
        output_topk,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nStage 4 visualization:")
    print(f"  Patient: {patient_id}")
    print(f"  Valid patches: {valid_n}")
    print(f"  S_prime used: {S_valid.shape}")
    print(f"  Saved: {output}")
    print(f"  Saved: {output_topk}")


# ============================================================
# FIGURE 2 — STAGE 5A CROSS-ATTENTION
# ============================================================

def generate_stage5a_heatmap(
    patient_id,
    attn,
    patch_mask,
    patient_index,
    patch_count,
    head=0,
):
    """
    Visualizes one attention head.

    Shape before slicing:
        (16, 186, N)

    Visualization:
        186 pathways × valid WSI patches
    """

    attention = (
        attn[
            patient_index,
            head,
            :,
            :patch_count,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    plt.figure(figsize=(12, 8))

    plt.imshow(
        attention,
        aspect="auto",
        interpolation="nearest",
    )

    plt.xlabel("WSI Patch")
    plt.ylabel("Pathway")

    plt.title(
        f"Stage 5A — Pathway-to-Patch Cross-Attention\n"
        f"{patient_id} | Attention Head {head + 1}"
    )

    plt.colorbar(
        label="Attention weight"
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "stage5a_cross_attention_heatmap.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nStage 5A visualization:")
    print(f"  Patient: {patient_id}")
    print(f"  Attention head: {head + 1}")
    print(f"  Valid patches: {patch_count}")
    print(f"  Attention used: {attention.shape}")
    print(f"  Saved: {output}")


# ============================================================
# FIGURE 3 — STAGE 5B RISK OUTPUTS
# ============================================================

def generate_stage5b_risk_plot(
    patient_ids,
    risk,
):
    """
    Displays the 15 risk outputs from the verification batch.

    These are NOT trained survival predictions.
    They are simply the outputs produced by the current model
    on the real verification data.
    """

    risk_values = (
        risk.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    plt.figure(figsize=(14, 6))

    x = np.arange(len(patient_ids))

    plt.bar(
        x,
        risk_values,
    )

    plt.xticks(
        x,
        patient_ids,
        rotation=45,
        ha="right",
    )

    plt.xlabel("Patient")
    plt.ylabel("Predicted Risk R")

    plt.title(
        "Stage 5B — Survival Risk Outputs\n"
        "Real 15-patient verification batch"
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "stage5b_risk_outputs.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nStage 5B visualization:")
    print(f"  Risk tensor: {tuple(risk.shape)}")
    print(f"  Saved: {output}")

    print("\nRisk values:")

    for pid, value in zip(
        patient_ids,
        risk_values,
    ):
        print(
            f"  {pid}: {value:.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PAMT STAGE 4–5B VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Match patients
    # --------------------------------------------------------

    pairs = match_patients(
        GENE_DIR,
        WSI_DIR,
    )

    if len(pairs) != 15:
        raise RuntimeError(
            f"Expected 15 matched patients, got {len(pairs)}"
        )

    print(
        f"\nMatched patients: {len(pairs)}"
    )

    patient_ids, x_gene, x_wsi, patch_mask, patch_counts = (
        load_batch(pairs)
    )

    print("\nPatient IDs:")

    for pid in patient_ids:
        print(f"  {pid}")

    print("\nGene input:")
    print(f"  {tuple(x_gene.shape)}")

    print("\nWSI input:")
    print(f"  {tuple(x_wsi.shape)}")

    print("\nPatch counts:")
    print(f"  {patch_counts}")

    # --------------------------------------------------------
    # Run complete pipeline
    # --------------------------------------------------------

    outputs = run_pipeline(
        x_gene,
        x_wsi,
        patch_mask,
    )

    xP = outputs["xP"]
    xI = outputs["xI"]
    L3 = outputs["L3"]
    info = outputs["info"]
    CA = outputs["CA"]
    attn = outputs["attn"]
    R = outputs["R"]

    # --------------------------------------------------------
    # Print verified shapes
    # --------------------------------------------------------

    print("\n--- Pipeline outputs ---")

    print(
        f"xP:       {tuple(xP.shape)}"
    )

    print(
        f"xI:       {tuple(xI.shape)}"
    )

    print(
        f"S_prime:  {tuple(info['S_prime'].shape)}"
    )

    print(
        f"Y_h:      {tuple(info['Y_h'].shape)}"
    )

    print(
        f"CA:       {tuple(CA.shape)}"
    )

    print(
        f"Attention:{tuple(attn.shape)}"
    )

    print(
        f"Risk R:   {tuple(R.shape)}"
    )

    print(
        f"L3:       {L3.item():.6f}"
    )

    # --------------------------------------------------------
    # Select ONE patient for heatmaps
    # --------------------------------------------------------

    patient_index = 0

    patient_id = patient_ids[patient_index]

    patch_count = patch_counts[patient_index]

    print("\n--- Visualization patient ---")

    print(
        f"Patient: {patient_id}"
    )

    print(
        f"Valid WSI patches: {patch_count}"
    )

    # --------------------------------------------------------
    # Stage 4
    # --------------------------------------------------------

    generate_stage4_heatmap(
        patient_id=patient_id,
        info=info,
        patch_mask=patch_mask,
        patient_index=patient_index,
        patch_count=patch_count,
    )

    # --------------------------------------------------------
    # Stage 5A
    # --------------------------------------------------------

    generate_stage5a_heatmap(
        patient_id=patient_id,
        attn=attn,
        patch_mask=patch_mask,
        patient_index=patient_index,
        patch_count=patch_count,
        head=0,
    )

    # --------------------------------------------------------
    # Stage 5B
    # --------------------------------------------------------

    generate_stage5b_risk_plot(
        patient_ids=patient_ids,
        risk=R,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FIGURES GENERATED")
    print("=" * 60)

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print("\nFiles:")

    for file in sorted(
        OUTPUT_DIR.glob("*.png")
    ):
        print(
            f"  {file.name}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()