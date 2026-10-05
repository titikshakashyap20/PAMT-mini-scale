"""
Workaround for a namespace collision: torch.hub's cached DINO repo
ships its own top-level utils.py (trunc_normal_, etc). This project
also has a `utils` package (src/utils/), and whichever gets imported
into sys.modules first wins for the rest of the process — every
script here imports utils.logging_utils before ever touching DINO,
so DINO's `from utils import trunc_normal_` silently resolves to OUR
utils package and fails.

Fix: temporarily evict `utils` (and submodules) from sys.modules
before the hub load, so DINO's own utils.py is found fresh via
torch.hub's sys.path insertion — then restore our cached copy after,
so the rest of the project is unaffected.
"""
import sys
import torch


def load_dino_hub(repo: str, model_name: str, **kwargs):
    saved = {k: v for k, v in sys.modules.items() if k == "utils" or k.startswith("utils.")}
    for k in saved:
        del sys.modules[k]
    try:
        return torch.hub.load(repo, model_name, **kwargs)
    finally:
        for k in list(sys.modules):
            if k == "utils" or k.startswith("utils."):
                del sys.modules[k]
        sys.modules.update(saved)