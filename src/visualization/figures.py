"""
Publication-quality preprocessing figures for demonstrating the WSI branch.

    Figure 1 - Whole slide image thumbnail
    Figure 2 - Detected tissue regions (grid before vs. after tissue filtering)
    Figure 3 - Example extracted tissue patches
    Figure 4 - Representative patches selected after K-means (by cluster)
    Figure 5 - (Optional) 2D projection of DINO embeddings (PCA or t-SNE),
               colored by cluster

All figures are saved at 300 DPI as standalone PNG files.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for servers / CI
import matplotlib.pyplot as plt

from utils.logging_utils import get_logger

log = get_logger(__name__)

_DPI = 300


def _project_coords_to_thumbnail(
    coords_level: List[Tuple[int, int]],
    level_dimensions: Tuple[int, int],
    thumbnail_shape: Tuple[int, int],
) -> np.ndarray:
    """Maps (x, y) top-left coordinates from level-L pixel space onto the
    (smaller) thumbnail's pixel space, for overlay plotting."""
    level_w, level_h = level_dimensions
    thumb_h, thumb_w = thumbnail_shape[:2]

    coords = np.array(coords_level, dtype=np.float64)
    if len(coords) == 0:
        return coords

    coords[:, 0] *= thumb_w / level_w
    coords[:, 1] *= thumb_h / level_h
    return coords


def figure1_thumbnail(thumbnail: np.ndarray, save_path: Path) -> Path:
    """Figure 1: the raw WSI thumbnail, for orientation."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(thumbnail)
    ax.set_title("Figure 1. Whole Slide Image Thumbnail", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def figure2_tissue_regions(
    thumbnail: np.ndarray,
    all_grid_coords: List[Tuple[int, int]],
    kept_coords: List[Tuple[int, int]],
    level_dimensions: Tuple[int, int],
    save_path: Path,
) -> Path:
    """Figure 2: full candidate grid (before) vs. tissue-filtered grid (after)."""
    all_thumb = _project_coords_to_thumbnail(all_grid_coords, level_dimensions, thumbnail.shape)
    kept_thumb = _project_coords_to_thumbnail(kept_coords, level_dimensions, thumbnail.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(thumbnail)
    if len(all_thumb):
        axes[0].scatter(all_thumb[:, 0], all_thumb[:, 1], s=1, c="red", alpha=0.35)
    axes[0].set_title(f"Before tissue filtering\n({len(all_grid_coords)} grid cells)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(thumbnail)
    if len(kept_thumb):
        axes[1].scatter(kept_thumb[:, 0], kept_thumb[:, 1], s=1, c="lime", alpha=0.5)
    axes[1].set_title(f"After tissue filtering\n({len(kept_coords)} tissue regions)", fontsize=12)
    axes[1].axis("off")

    fig.suptitle("Figure 2. Detected Tissue Regions", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def figure3_example_patches(patches: np.ndarray, save_path: Path, n: int = 16, seed: int = 0) -> Path:
    """Figure 3: a grid of example extracted tissue patches, randomly sampled."""
    rng = np.random.default_rng(seed)
    n = min(n, len(patches))
    idx = rng.choice(len(patches), size=n, replace=False)

    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).reshape(-1)

    for ax, i in zip(axes, idx):
        ax.imshow(patches[i])
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Figure 3. Example Extracted Tissue Patches (n={n})", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def figure4_representative_patches(
    patches: np.ndarray,
    cluster_labels: np.ndarray,
    representative_indices: np.ndarray,
    save_path: Path,
    n_clusters_to_show: int = 8,
    patches_per_cluster_to_show: int = 5,
    seed: int = 0,
) -> Path:
    """Figure 4: representative patches after K-means, grouped by cluster
    (one row per cluster) so the visual diversity captured by clustering
    is directly inspectable."""
    rng = np.random.default_rng(seed)

    rep_labels = cluster_labels[representative_indices]
    unique_clusters = np.unique(rep_labels)
    show_clusters = rng.choice(
        unique_clusters, size=min(n_clusters_to_show, len(unique_clusters)), replace=False
    )
    show_clusters = sorted(show_clusters.tolist())

    fig, axes = plt.subplots(
        len(show_clusters),
        patches_per_cluster_to_show,
        figsize=(patches_per_cluster_to_show * 1.8, len(show_clusters) * 1.8),
    )
    axes = np.array(axes).reshape(len(show_clusters), patches_per_cluster_to_show)

    for row, cluster_id in enumerate(show_clusters):
        cluster_rep_idx = representative_indices[rep_labels == cluster_id]
        n_show = min(patches_per_cluster_to_show, len(cluster_rep_idx))
        chosen = rng.choice(cluster_rep_idx, size=n_show, replace=False)

        for col in range(patches_per_cluster_to_show):
            ax = axes[row, col]
            if col < n_show:
                ax.imshow(patches[chosen[col]])
            ax.axis("off")
        axes[row, 0].set_ylabel(f"Cluster {cluster_id}", fontsize=9, rotation=0,
                                 labelpad=35, va="center")

    fig.suptitle("Figure 4. Representative Patches Selected After K-means", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def figure5_embedding_projection(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Path,
    method: str = "pca",
    seed: int = 0,
) -> Path:
    """Figure 5 (optional): 2D projection of DINO embeddings, colored by
    K-means cluster, to visually sanity-check cluster separability."""
    if method == "pca":
        from sklearn.decomposition import PCA
        projector = PCA(n_components=2, random_state=seed)
        coords_2d = projector.fit_transform(embeddings)
        method_label = "PCA"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        projector = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30)
        coords_2d = projector.fit_transform(embeddings)
        method_label = "t-SNE"
    else:
        raise ValueError(f"Unknown projection method: {method}")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1], c=cluster_labels, cmap="tab20", s=8, alpha=0.8
    )
    ax.set_title(f"Figure 5. DINO Embeddings ({method_label} projection, colored by cluster)", fontsize=12)
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    fig.colorbar(scatter, ax=ax, label="Cluster ID", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_all_figures(artifacts, output_dir: Path, embedding_projection: str = "pca") -> dict:
    """Generates all five figures for one patient's WSI artifacts and saves
    them into `output_dir`. Returns a dict of {figure_name: saved_path}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    log.info("Generating preprocessing figures...")

    paths["figure1_thumbnail"] = figure1_thumbnail(
        artifacts.thumbnail, output_dir / f"{artifacts.patient_id}_fig1_thumbnail.png"
    )

    paths["figure2_tissue_regions"] = figure2_tissue_regions(
        artifacts.thumbnail,
        artifacts.full_grid_coords,
        artifacts.tissue_kept_coords,
        artifacts.level_dimensions,
        output_dir / f"{artifacts.patient_id}_fig2_tissue_regions.png",
    )

    paths["figure3_example_patches"] = figure3_example_patches(
        artifacts.candidate_patches, output_dir / f"{artifacts.patient_id}_fig3_example_patches.png"
    )

    paths["figure4_representative_patches"] = figure4_representative_patches(
        artifacts.candidate_patches,
        artifacts.cluster_labels,
        artifacts.representative_indices,
        output_dir / f"{artifacts.patient_id}_fig4_representative_patches.png",
    )

    paths["figure5_embedding_projection"] = figure5_embedding_projection(
        artifacts.embeddings,
        artifacts.cluster_labels,
        output_dir / f"{artifacts.patient_id}_fig5_embedding_{embedding_projection}.png",
        method=embedding_projection,
    )

    log.info(f"  Saved {len(paths)} figures to {output_dir}")
    return paths
