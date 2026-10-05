import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

PATIENT_ID = "TCGA-2F-A9KO"

GENE_DIR = Path("data/processed/gene")
WSI_DIR = Path("data/processed/wsi")
OUTPUT_DIR = Path("C:/Users/visha/Documents/pamt_mini")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND FILES
# ============================================================

gene_path = GENE_DIR / f"{PATIENT_ID}.npy"

# Find matching WSI .npz using patient ID
wsi_matches = list(WSI_DIR.glob(f"{PATIENT_ID}*_patches.npz"))

if not gene_path.exists():
    raise FileNotFoundError(f"Gene file not found:\n{gene_path}")

if not wsi_matches:
    raise FileNotFoundError(
        f"No WSI .npz found for patient {PATIENT_ID}"
    )

wsi_path = wsi_matches[0]

print("Gene file:", gene_path)
print("WSI file :", wsi_path)


# ============================================================
# LOAD GENE DATA
# ============================================================

gene_data = np.load(gene_path)

print("\nGene data")
print("Shape:", gene_data.shape)

# Expected:
# (186, 4942)


# ============================================================
# LOAD WSI DATA
# ============================================================

wsi_data = np.load(wsi_path)

print("\nWSI data keys:", wsi_data.files)

embeddings = wsi_data["embeddings"]
coords = wsi_data["coords"]
cluster_labels = wsi_data["cluster_labels"]

print("WSI embeddings:", embeddings.shape)
print("Coordinates:", coords.shape)
print("Cluster labels:", cluster_labels.shape)


# ============================================================
# CREATE FIGURE
# ============================================================

fig = plt.figure(figsize=(16, 9))

fig.suptitle(
    f"PAMT Model Input — {PATIENT_ID}",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# 1. GENE PATHWAY MATRIX
# ============================================================

ax1 = plt.subplot2grid((2, 3), (0, 0))

# Display a downsampled/normalized version so the structure
# is visually understandable.
gene_vis = gene_data.copy()

# Normalize each pathway for visualization
row_min = gene_vis.min(axis=1, keepdims=True)
row_max = gene_vis.max(axis=1, keepdims=True)

gene_vis = (gene_vis - row_min) / (
    row_max - row_min + 1e-8
)

ax1.imshow(
    gene_vis,
    aspect="auto",
    interpolation="nearest"
)

ax1.set_title(
    f"Gene pathway representation\n{gene_data.shape[0]} pathways × "
    f"{gene_data.shape[1]} genes"
)

ax1.set_xlabel("Gene index")
ax1.set_ylabel("Pathway index")


# ============================================================
# 2. DINO EMBEDDINGS
# ============================================================

ax2 = plt.subplot2grid((2, 3), (0, 1))

# Visualize the DINO embedding matrix
im = ax2.imshow(
    embeddings,
    aspect="auto",
    interpolation="nearest"
)

ax2.set_title(
    f"WSI DINO embeddings\n"
    f"{embeddings.shape[0]} patches × {embeddings.shape[1]} dimensions"
)

ax2.set_xlabel("DINO feature dimension")
ax2.set_ylabel("Patch index")

plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)


# ============================================================
# 3. PATCH / CLUSTER DISTRIBUTION
# ============================================================

ax3 = plt.subplot2grid((2, 3), (0, 2))

unique_clusters, counts = np.unique(
    cluster_labels,
    return_counts=True
)

ax3.bar(
    unique_clusters,
    counts
)

ax3.set_title(
    f"WSI patch clusters\n"
    f"{len(unique_clusters)} clusters"
)

ax3.set_xlabel("Cluster ID")
ax3.set_ylabel("Number of patches")


# ============================================================
# 4. WSI COORDINATES
# ============================================================

ax4 = plt.subplot2grid((2, 3), (1, 0))

ax4.scatter(
    coords[:, 0],
    coords[:, 1],
    s=10
)

ax4.invert_yaxis()

ax4.set_title(
    f"Selected WSI patch locations\n"
    f"{len(coords)} patches"
)

ax4.set_xlabel("X coordinate")
ax4.set_ylabel("Y coordinate")


# ============================================================
# 5. DINO FEATURE DISTRIBUTION
# ============================================================

ax5 = plt.subplot2grid((2, 3), (1, 1))

feature_means = embeddings.mean(axis=0)

ax5.plot(feature_means)

ax5.set_title(
    "Mean DINO feature activation"
)

ax5.set_xlabel("DINO dimension")
ax5.set_ylabel("Mean feature value")


# ============================================================
# 6. MODEL INPUT SUMMARY
# ============================================================

ax6 = plt.subplot2grid((2, 3), (1, 2))

ax6.axis("off")

summary = (
    f"PATIENT\n"
    f"{PATIENT_ID}\n\n"

    f"GENE BRANCH\n"
    f"Pathways: {gene_data.shape[0]}\n"
    f"Genes: {gene_data.shape[1]}\n"
    f"Input: {gene_data.shape}\n\n"

    f"WSI BRANCH\n"
    f"Real patches: {embeddings.shape[0]}\n"
    f"DINO dimension: {embeddings.shape[1]}\n"
    f"Input: {embeddings.shape}\n\n"

    f"NEXT\n"
    f"GENE → Pathway Embedding\n"
    f"WSI → WSI Transformer"
)

ax6.text(
    0.05,
    0.95,
    summary,
    transform=ax6.transAxes,
    fontsize=11,
    verticalalignment="top",
    family="monospace"
)


# ============================================================
# SAVE
# ============================================================

output_path = OUTPUT_DIR / f"{PATIENT_ID}_model_input_panel.png"

plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(output_path)