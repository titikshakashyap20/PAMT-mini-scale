"""DINO ViT-S/8 patch embedding — inference only. Training loop lives in dino_pretraining.py."""
import numpy as np
import torch
from torchvision import transforms

from utils.logging_utils import get_logger
from wsi_branch.hub_compat import load_dino_hub

log = get_logger(__name__)

_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dino(repo: str = "facebookresearch/dino:main", model_name: str = "dino_vits8",
              checkpoint_path: str | None = None):
    """
    Loads the DINO ViT-S/8 backbone.

    If `checkpoint_path` is given, loads YOUR trained weights (from
    dino_pretraining.py) instead of the public ImageNet-pretrained ones —
    use this once mini-scale training is in place, to stay faithful to the
    paper's per-dataset self-supervised training.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is not None:
        model = load_dino_hub(repo, model_name, pretrained=False, skip_validation=True)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        log.info(f"Loaded DINO weights from local checkpoint: {checkpoint_path}")
    else:
        try:
            model = load_dino_hub(repo, model_name, skip_validation=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download pretrained DINO weights via torch.hub ({repo}). "
                "Check internet access, or pass checkpoint_path= to load local weights."
            ) from exc
        log.info(f"Loaded pretrained {model_name} weights via torch.hub (ImageNet DINO).")

    model.eval().to(device)
    log.info(f"Using device: {device}")
    return model, device


def embed_patches(patches, model, device, batch_size: int = 32) -> np.ndarray:
    """patches: list of (x, y, PIL.Image). Returns (N, 384) embeddings."""
    embeddings = []
    total_batches = (len(patches) + batch_size - 1) // batch_size

    with torch.no_grad():
        for b, i in enumerate(range(0, len(patches), batch_size)):
            log.info(f"Embedding batch {b + 1}/{total_batches}")
            batch = patches[i:i + batch_size]
            imgs = torch.stack([_PREPROCESS(p[2]) for p in batch]).to(device)
            feats = model(imgs)
            embeddings.append(feats.cpu().numpy())

    return np.concatenate(embeddings, axis=0)