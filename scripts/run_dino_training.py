"""
Builds the pooled DINO training corpus (~60 tissue patches per slide, capped
at 800 total across the MINI cohort) and runs self-supervised DINO training.

Run from the project root:
    python scripts/run_dino_training.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from utils.logging_utils import get_logger, stage
from utils.config import load_config
from wsi_branch.loader import open_slide
from wsi_branch.patch_extraction import extract_tissue_patches, pick_level_for_magnification
from wsi_branch.dino_pretraining import train_dino

log = get_logger(__name__)


def find_slide_paths(wsi_root: Path, manifest_path: Path) -> list[Path]:
    """Reads manifest.tsv, dedupes by 12-char patient barcode (keeping the
    first listed slide per patient), and resolves each to its file on disk."""
    manifest = pd.read_csv(manifest_path, sep="\t")
    manifest["patient_id"] = manifest["file_name"].str[:12]
    manifest = manifest.drop_duplicates(subset="patient_id", keep="first")

    paths = []
    for file_name in manifest["file_name"]:
        p = wsi_root / file_name
        if p.exists():
            paths.append(p)
        else:
            log.info(f"  Skipping {file_name}: not found under {wsi_root}")
    return paths


def build_training_corpus(cfg: dict, slide_paths: list[Path]) -> list:
    """Pools patches_per_wsi_for_corpus tissue patches from each slide, up to
    total_corpus_cap patches overall. Returns a flat list of PIL.Image patches."""
    wcfg = cfg["wsi_branch"]
    dcfg = cfg["dino_training"]
    per_wsi = dcfg["patches_per_wsi_for_corpus"]
    cap = dcfg["total_corpus_cap"]

    corpus = []
    for slide_path in slide_paths:
        if len(corpus) >= cap:
            break
        with stage(log, f"Pooling patches from {slide_path.name}"):
            slide = open_slide(slide_path)
            level = pick_level_for_magnification(slide, wcfg["target_magnification"])
            patches, _ = extract_tissue_patches(
                slide,
                level=level,
                patch_size=wcfg["patch_size"],
                max_patches=per_wsi,
                bg_intensity_cutoff=wcfg["tissue_bg_intensity_cutoff"],
                bg_fraction_threshold=wcfg["tissue_bg_fraction_threshold"],
            )
            slide.close()

            remaining = cap - len(corpus)
            images = [p[2] for p in patches][:remaining]
            corpus.extend(images)
            log.info(f"  Pooled {len(images)} patches (corpus size: {len(corpus)}/{cap})")

    return corpus


def main():
    cfg = load_config()
    wcfg = cfg["wsi_branch"]
    dcfg = cfg["dino_training"]

    log.info("=" * 60)
    log.info("PAMT Mini Implementation — DINO Self-Supervised Pretraining")
    log.info("=" * 60)

    with stage(log, "Building pooled training corpus"):
        slide_paths = find_slide_paths(
            Path(cfg["paths"]["wsi_root"]), Path(cfg["paths"]["manifest"])
        )
        log.info(f"  Found {len(slide_paths)} unique-patient slides in manifest")
        corpus = build_training_corpus(cfg, slide_paths)
        log.info(f"  Final training corpus size: {len(corpus)} patches")

    with stage(log, "Training DINO"):
        checkpoint_path = train_dino(
            corpus,
            cfg=dcfg,
            model_name=wcfg["dino_model"],
            embed_dim=384,
        )
        log.info(f"  Trained checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()