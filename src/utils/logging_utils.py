"""
Logging utilities shared by every preprocessing module.

Goal: replace ad-hoc debugging `print()` statements with a single,
consistent, demo-friendly logging style:

    [PAMT] Loading Whole Slide Image...
    [PAMT] Selecting pyramid level...
    [PAMT]   -> Level 1 selected (20.1x, closest to target 20.0x)
    [PAMT] Extracting tissue patches...
    [PAMT]   Detected 42318 tissue regions.
    ...

Usage:
    from utils.logging_utils import get_logger, stage

    log = get_logger(__name__)
    with stage(log, "Loading Whole Slide Image"):
        ...
"""

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path


_CONFIGURED = False


def get_logger(name: str, log_file: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Returns a configured logger. Safe to call repeatedly across modules."""
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _CONFIGURED:
        root = logging.getLogger()
        root.setLevel(level)

        formatter = logging.Formatter("[PAMT] %(message)s")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            root.addHandler(file_handler)

        _CONFIGURED = True

    return logger


@contextmanager
def stage(logger: logging.Logger, message: str):
    """
    Prints a clean stage banner, runs the wrapped block, then reports elapsed time.

    Example:
        with stage(log, "Computing DINO embeddings"):
            embeddings = compute_embeddings(patches)
    """
    logger.info(f"{message}...")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"  Done in {elapsed:.1f}s.")


def log_kv(logger: logging.Logger, label: str, value) -> None:
    """Consistent 'key: value' style line, indented under a stage banner."""
    logger.info(f"  {label}: {value}")
