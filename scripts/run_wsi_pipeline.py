"""
Run from the project root:

python scripts/run_wsi_pipeline.py --slide-path data/raw/wsi_data/tcga_blca/TCGA-XXXX.svs
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from utils.logging_utils import get_logger, stage
from utils.config import load_config
from visualization.figures import generate_all_figures
from wsi_branch.loader import open_slide, get_thumbnail_array
from wsi_branch.patch_extraction import extract_tissue_patches, pick_level_for_magnification
from wsi_branch.dino_embedding import load_dino, embed_patches
from wsi_branch.patch_clustering import cluster_and_select

log = get_logger(__name__)


@dataclass
class WSIArtifacts:
    """Bundles everything figures.generate_all_figures() needs for one slide."""
    patient_id: str
    thumbnail: np.ndarray
    full_grid_coords: list
    tissue_kept_coords: list
    level_dimensions: tuple
    candidate_patches: list
    cluster_labels: np.ndarray
    representative_indices: np.ndarray
    embeddings: np.ndarray


def process_one_slide(slide_path, cfg, wcfg, model, device, figures_dir, out_dir, level_override=None):
    """Runs one slide end-to-end: extraction -> embedding -> clustering ->
    figures -> save. Assumes model/device are already loaded by the caller."""
    slide_id = Path(slide_path).stem

    with stage(log, f"[{slide_id}] Opening Whole Slide Image"):
        slide = open_slide(slide_path)
        log.info(f"  Slide dimensions: {slide.dimensions}")
        thumbnail = get_thumbnail_array(slide)

    level = level_override
    if level is None:
        with stage(log, f"[{slide_id}] Selecting pyramid level"):
            level = pick_level_for_magnification(slide, wcfg["target_magnification"])
    level_dimensions = slide.level_dimensions[level]

    with stage(log, f"[{slide_id}] Extracting tissue patches"):
        patches, all_grid_coords = extract_tissue_patches(
            slide, level=level, patch_size=wcfg["patch_size"],
            max_patches=wcfg["max_candidate_patches"],
            bg_intensity_cutoff=wcfg["tissue_bg_intensity_cutoff"],
            bg_fraction_threshold=wcfg["tissue_bg_fraction_threshold"],
        )
        kept_coords = [(x, y) for x, y, _ in patches]
        log.info(f"  Extracted {len(patches)} tissue patches (from {len(all_grid_coords)} grid cells).")

    with stage(log, f"[{slide_id}] Computing DINO embeddings"):
        embeddings = embed_patches(patches, model, device, wcfg["embedding_batch_size"])

    with stage(log, f"[{slide_id}] Clustering and selecting representative patches"):
        _, cluster_labels, representative_indices = cluster_and_select(
            patches, embeddings, n_clusters=wcfg["n_clusters"],
            patches_per_cluster=wcfg["patches_per_cluster"], seed=cfg["sample"]["seed"],
        )
        log.info(f"  Selected {len(representative_indices)} representative patches.")

    with stage(log, f"[{slide_id}] Generating preprocessing figures"):
        artifacts = WSIArtifacts(
            patient_id=slide_id, thumbnail=thumbnail, full_grid_coords=all_grid_coords,
            tissue_kept_coords=kept_coords, level_dimensions=level_dimensions,
            candidate_patches=[p[2] for p in patches], cluster_labels=cluster_labels,
            representative_indices=representative_indices, embeddings=embeddings,
        )
        generate_all_figures(artifacts, figures_dir)

    with stage(log, f"[{slide_id}] Saving outputs"):
        rep_coords = np.array([(patches[i][0], patches[i][1]) for i in representative_indices])
        out_path = out_dir / f"{slide_id}_patches.npz"
        np.savez(out_path, coords=rep_coords, embeddings=embeddings[representative_indices],
                  cluster_labels=cluster_labels[representative_indices])
        log.info(f"  Saved: {out_path}")

    slide.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-path", required=True, help="Path to one .svs slide")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--dino-checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config()
    wcfg = cfg["wsi_branch"]

    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["paths"]["processed_wsi_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.dino_checkpoint
    if checkpoint is None and "dino_training" in cfg:
        default_ckpt = Path(cfg["dino_training"]["checkpoint_dir"]) / f"{wcfg['dino_model']}_mini_latest.pt"
        if default_ckpt.exists():
            checkpoint = str(default_ckpt)
    model, device = load_dino(wcfg["dino_repo"], wcfg["dino_model"], checkpoint_path=checkpoint)

    process_one_slide(args.slide_path, cfg, wcfg, model, device, figures_dir, out_dir, args.level)


if __name__ == "__main__":
    main()