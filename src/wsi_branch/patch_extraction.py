"""
WSI Patch Extraction Module

Implements the preprocessing described in Section III-B of the PAMT paper.

Pipeline:
    Whole-slide image (.svs)
        ↓
    Pyramid level selection
        ↓
    Non-overlapping 256×256 patch extraction
        ↓
    Background filtering
        ↓
    Sharpness/artifact filtering
        ↓
    Tissue patch selection

Output:
    List of (x, y, PIL.Image) tissue patches
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import openslide
from scipy import ndimage

from utils.logging_utils import get_logger
from wsi_branch.loader import get_thumbnail_array
from PIL import Image

log = get_logger(__name__)

_MPP_TIMES_MAG_CONSTANT = 0.25 * 40.0  # 10.0

def _base_objective_power(slide) -> Optional[float]:
    value = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _base_mpp_x(slide) -> Optional[float]:
    value = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if value is None:
        return None
    try:
        mpp = float(value)
        return mpp if mpp > 0 else None
    except (TypeError, ValueError):
        return None


def pick_level_for_magnification(slide, target_mag=20):
    downsamples = slide.level_downsamples

    base_mag = _base_objective_power(slide)

    if base_mag is not None:
        method = "objective-power"
    else:
        mpp = _base_mpp_x(slide)

        if mpp is not None:
            base_mag = _MPP_TIMES_MAG_CONSTANT / mpp
            method = "mpp-estimate"
        else:
            log.info(
                "Automatic magnification detection unavailable. "
                "Falling back to pyramid level 0."
            )
            return 0

    effective_mags = [base_mag / d for d in downsamples]

    diffs = [abs(m - target_mag) for m in effective_mags]

    best_level = int(np.argmin(diffs))

    log.info(
        f"Selected pyramid level {best_level} "
        f"({effective_mags[best_level]:.1f}x, {method})"
    )

    return best_level

def has_tissue(
    patch,
    bg_intensity_cutoff=220,
    bg_fraction_threshold=0.8,
):
    gray = np.array(patch.convert("L"))
    background_fraction = (gray > bg_intensity_cutoff).mean()
    return background_fraction < bg_fraction_threshold


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — a standard, cheap sharpness/artifact
    check. Real tissue has lots of fine texture and edges (high variance).
    Motion-blur streaks and dust/debris are locally smooth or near-flat
    despite not being background-white, so they slip past has_tissue()
    but score low here."""
    lap = ndimage.laplace(gray.astype(np.float64))
    return float(lap.var())


def is_valid_tissue_patch(
    patch,
    bg_intensity_cutoff=220,
    bg_fraction_threshold=0.8,
    min_sharpness=50.0,
):
    """Combines the background check, sharpness/blur check, and
    ink/debris check. Thresholds are heuristics — calibrate against your
    own known-good vs. known-bad patches rather than trusting defaults."""
    gray = np.array(patch.convert("L"))
    background_fraction = (gray > bg_intensity_cutoff).mean()
    if background_fraction >= bg_fraction_threshold:
        return False, "background"

    is_artifact, reason = has_ink_or_debris(patch)
    if is_artifact:
        return False, reason

    if _laplacian_variance(gray) < min_sharpness:
        return False, "low_sharpness"

    return True, "ok"

def has_ink_or_debris(patch, dark_pixel_threshold=50, dark_pixel_max_fraction=0.008,
                        nonH_and_E_hue_max_fraction=0.03):
    """Catches artifacts the sharpness filter can't: high-contrast dark
    debris/hair/folds (sharp, not blurry — passes Laplacian fine) and
    colored ink/marker pen (green/blue, not the pink-purple-white H&E
    palette). Both are heuristic thresholds — calibrate against your own
    flagged patches before trusting the defaults."""
    arr = np.array(patch.convert("RGB")).astype(np.float64)

    # near-black debris/hair/fold: all channels dark simultaneously
    near_black = (arr.max(axis=-1) < dark_pixel_threshold)
    if near_black.mean() > dark_pixel_max_fraction:
        return True, "dark_debris"

    # ink/marker: hue outside the pink-purple-white H&E range (roughly
    # hue 280-360 or 0-20 in a 0-360 scale; green/blue ink sits ~90-240)
    hsv = np.array(patch.convert("HSV")).astype(np.float64)
    hue = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1]
    ink_like = (hue > 80) & (hue < 260) & (sat > 60)
    if ink_like.mean() > nonH_and_E_hue_max_fraction:
        return True, "ink_marker"

    return False, "ok"


def _thumbnail_tissue_fraction(thumb_gray, mx0, my0, mx1, my1, bg_intensity_cutoff):
    """Cheap tissue estimate for one grid cell's projected footprint on the thumbnail."""
    region = thumb_gray[my0:my1, mx0:mx1]
    if region.size == 0:
        return 0.0
    background_fraction = (region > bg_intensity_cutoff).mean()
    return 1.0 - background_fraction


def extract_tissue_patches(
    slide,
    level,
    patch_size=256,
    max_patches=3000,
    bg_intensity_cutoff=220,
    bg_fraction_threshold=0.8,
    min_sharpness=50.0,
    thumbnail_max_size=2048,
    seed=0,
):
    width, height = slide.level_dimensions[level]
    stride = patch_size

    # FIX (bug 1): include the final row/column that fits exactly.
    candidates = [
        (x, y)
        for y in range(0, height - patch_size + 1, stride)
        for x in range(0, width - patch_size + 1, stride)
    ]
    log.info(f"Candidate patch locations: {len(candidates)}")

    # --- Fast low-resolution pre-filter (fixes bug 3: performance) -----
    thumb_gray = np.array(
        Image.fromarray(
            get_thumbnail_array(slide, thumbnail_max_size)
        ).convert("L")
    )
    thumb_h, thumb_w = thumb_gray.shape
    scale_x = thumb_w / width
    scale_y = thumb_h / height

    exact_min_tissue_fraction = 1.0 - bg_fraction_threshold          # e.g. 0.2
    prefilter_min_tissue_fraction = max(exact_min_tissue_fraction - 0.1, 0.0)  # more lenient, e.g. 0.1

    likely_tissue_coords = []
    for x, y in candidates:
        mx0, my0 = int(x * scale_x), int(y * scale_y)
        mx1 = max(mx0 + 1, int((x + patch_size) * scale_x))
        my1 = max(my0 + 1, int((y + patch_size) * scale_y))
        tissue_fraction = _thumbnail_tissue_fraction(
            thumb_gray, mx0, my0, mx1, my1, bg_intensity_cutoff
        )
        if tissue_fraction >= prefilter_min_tissue_fraction:
            likely_tissue_coords.append((x, y))

    log.info(f"Passed low-resolution pre-filter: {len(likely_tissue_coords)}")

    # --- Random cap BEFORE full-resolution reads (fixes bug 2: bias) ---
    rng = np.random.default_rng(seed)
    if len(likely_tissue_coords) > max_patches:
        oversample_n = min(len(likely_tissue_coords), int(max_patches * 1.5))
        idx = rng.choice(len(likely_tissue_coords), size=oversample_n, replace=False)
        likely_tissue_coords = [likely_tissue_coords[i] for i in idx]

    # --- Exact confirmation at full resolution -------------------------
    downsample = slide.level_downsamples[level]
    tissue_patches = []
    rejected_background = 0
    rejected_ink_debris = 0
    rejected_sharpness = 0

    for x, y in likely_tissue_coords:
        if len(tissue_patches) >= max_patches:
            break
        region = slide.read_region(
            (int(x * downsample), int(y * downsample)), level, (patch_size, patch_size)
        ).convert("RGB")
        valid, reason = is_valid_tissue_patch(
            region, bg_intensity_cutoff, bg_fraction_threshold, min_sharpness
        )
        if valid:
            tissue_patches.append((x, y, region))
        elif reason == "background":
            rejected_background += 1
        elif reason == "low_sharpness":
            rejected_sharpness += 1
        else:
            rejected_ink_debris += 1

    if len(tissue_patches) < max_patches:
        log.info(
            f"Note: only {len(tissue_patches)} tissue patches confirmed "
            f"(target was {max_patches}) — this slide may have limited tissue."
        )
    log.info(
        f"Tissue patches extracted: {len(tissue_patches)} "
        f"(rejected: {rejected_background} background, {rejected_sharpness} blurry, "
        f"{rejected_ink_debris} ink/debris)"
    )
    return tissue_patches, candidates