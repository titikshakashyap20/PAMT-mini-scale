"""
WSI loading utilities.

Centralizes opening a slide and producing a cheap low-resolution
thumbnail, so the same thumbnail logic isn't duplicated between the
tissue pre-filter in patch_extraction.py and Figure 1 in figures.py.
"""

from pathlib import Path

import numpy as np
import openslide

from utils.logging_utils import get_logger

log = get_logger(__name__)


def open_slide(path) -> openslide.OpenSlide:
    """Opens a WSI file, raising a clear error if it's missing or unreadable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WSI file not found: {path}")
    try:
        return openslide.OpenSlide(str(path))
    except openslide.OpenSlideError as exc:
        raise RuntimeError(f"OpenSlide could not read '{path}': {exc}") from exc


def get_thumbnail_array(slide: openslide.OpenSlide, max_size: int = 2048) -> np.ndarray:
    """Cheap RGB thumbnail as a numpy array (H, W, 3), longest side <= max_size."""
    thumb = slide.get_thumbnail((max_size, max_size))
    return np.array(thumb.convert("RGB"))