"""Load config.yaml once, resolve paths relative to project root."""
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config.yaml") -> dict:
    cfg_path = PROJECT_ROOT / config_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve all "paths" entries to absolute paths so scripts can be run
    # from any working directory (VS Code sometimes runs from repo root,
    # sometimes from the file's own folder).
    for key, rel_path in cfg["paths"].items():
        cfg["paths"][key] = str(PROJECT_ROOT / rel_path)

    return cfg
