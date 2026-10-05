"""
Self-supervised DINO ViT-S/8 pretraining (Section III-B, p.901, Eq. 1),
scaled down for PAMT_MINI (CPU-only): fewer/smaller crops, a small pooled
patch corpus, and 10 epochs instead of 200 — same method, smaller dose.

Produces a checkpoint that dino_embedding.load_dino(checkpoint_path=...)
can load, so the rest of the pipeline is unaffected.
"""

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.logging_utils import get_logger
from wsi_branch.hub_compat import load_dino_hub

log = get_logger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Multi-crop augmentation (standard DINO recipe, fewer/smaller crops)
# --------------------------------------------------------------------------

class MultiCropAugmentation:
    def __init__(self, global_crops, global_crop_size, global_crop_scale,
                 local_crops, local_crop_size, local_crop_scale):
        normalize = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)

        flip_and_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        self.global_crops = global_crops
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size, scale=tuple(global_crop_scale), interpolation=Image.BICUBIC
            ),
            flip_and_jitter,
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize,
        ])

        self.local_crops = local_crops
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size, scale=tuple(local_crop_scale), interpolation=Image.BICUBIC
            ),
            flip_and_jitter,
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, image: Image.Image):
        crops = [self.global_transform(image) for _ in range(self.global_crops)]
        crops += [self.local_transform(image) for _ in range(self.local_crops)]
        return crops


class PatchCorpusDataset(torch.utils.data.Dataset):
    """Wraps a list of already-loaded PIL patch images."""

    def __init__(self, patch_images, augmentation: MultiCropAugmentation):
        self.patch_images = patch_images
        self.augmentation = augmentation

    def __len__(self):
        return len(self.patch_images)

    def __getitem__(self, idx):
        return self.augmentation(self.patch_images[idx])


def multi_crop_collate(batch):
    """Groups crops by view index across the batch: n_views lists of (B, C, H, W)."""
    n_views = len(batch[0])
    return [torch.stack([sample[v] for sample in batch]) for v in range(n_views)]


# --------------------------------------------------------------------------
# DINO head + student/teacher (Eq. 1)
# --------------------------------------------------------------------------

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def _build_backbone(model_name: str, drop_path_rate: float):
    try:
        return load_dino_hub(
            "facebookresearch/dino:main",
            model_name,
            pretrained=False,
            drop_path_rate=drop_path_rate,
        )
    except TypeError:
        log.info("This DINO hubconf does not accept drop_path_rate — using its default.")
        return load_dino_hub(
            "facebookresearch/dino:main",
            model_name,
            pretrained=False,
        )


def train_dino(
    patch_images,
    cfg: dict,
    model_name: str = "dino_vits8",
    embed_dim: int = 384,
    device: str = "cpu",
):
    """
    Trains DINO ViT-S/8 from scratch on `patch_images` (a list of PIL.Image
    tissue patches pooled across your MINI cohort). Returns the path to the
    saved student-backbone checkpoint.
    """
    center = torch.zeros(1, 4096, device=device)  # matches DINOHead out_dim
    center_momentum = cfg.get("center_momentum", 0.9)

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Building DINO training corpus: {len(patch_images)} patches")

    augmentation = MultiCropAugmentation(
        global_crops=cfg["global_crops"], global_crop_size=cfg["global_crop_size"],
        global_crop_scale=cfg["global_crop_scale"],
        local_crops=cfg["local_crops"], local_crop_size=cfg["local_crop_size"],
        local_crop_scale=cfg["local_crop_scale"],
    )
    dataset = PatchCorpusDataset(patch_images, augmentation)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=multi_crop_collate, drop_last=True,
    )

    student_backbone = _build_backbone(model_name, cfg["drop_path_rate"]).to(device)
    teacher_backbone = copy.deepcopy(student_backbone).to(device)
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    student_head = DINOHead(embed_dim).to(device)
    teacher_head = DINOHead(embed_dim).to(device)
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_head.parameters():
        p.requires_grad = False

    params = list(student_backbone.parameters()) + list(student_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    n_global = cfg["global_crops"]
    total_steps = cfg["epochs"] * len(loader)
    step = 0
   
    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0

        for views in loader:
            views = [v.to(device) for v in views]
            global_views, local_views = views[:n_global], views[n_global:]

            momentum = cfg["teacher_momentum_start"] + (
                cfg["teacher_momentum_end"] - cfg["teacher_momentum_start"]
            ) * (step / max(total_steps - 1, 1))

            with torch.no_grad():
                teacher_out = [
                    teacher_head(teacher_backbone(v))
                    for v in global_views
                ]

                teacher_probs = [
                    F.softmax((t - center) / cfg["teacher_temp"], dim=-1)
                    for t in teacher_out
                ]

                batch_center = torch.cat(
                    teacher_out, dim=0
                ).mean(dim=0, keepdim=True)

                center = (
                    center * center_momentum
                    + batch_center * (1 - center_momentum)
                )

            all_views = global_views + local_views
            student_out = [student_head(student_backbone(v)) for v in all_views]
            student_log_probs = [F.log_softmax(s / cfg["student_temp"], dim=-1) for s in student_out]

            loss, n_terms = 0.0, 0
            for i, pt in enumerate(teacher_probs):
                for j, log_ps in enumerate(student_log_probs):
                    if i == j:
                        continue
                    loss = loss + torch.sum(-pt * log_ps, dim=-1).mean()
                    n_terms += 1
            loss = loss / n_terms

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for t_p, s_p in zip(teacher_backbone.parameters(), student_backbone.parameters()):
                    t_p.data.mul_(momentum).add_(s_p.data, alpha=1 - momentum)
                for t_p, s_p in zip(teacher_head.parameters(), student_head.parameters()):
                    t_p.data.mul_(momentum).add_(s_p.data, alpha=1 - momentum)

            epoch_loss += loss.item()
            step += 1

        elapsed = time.time() - epoch_start
        log.info(
            f"Epoch {epoch + 1}/{cfg['epochs']} — loss {epoch_loss / len(loader):.4f} "
            f"— {elapsed:.1f}s ({elapsed / len(loader):.2f}s/step)"
        )

        ckpt_path = checkpoint_dir / f"{model_name}_mini_epoch{epoch + 1}.pt"
        torch.save(student_backbone.state_dict(), ckpt_path)

    latest_path = checkpoint_dir / f"{model_name}_mini_latest.pt"
    torch.save(student_backbone.state_dict(), latest_path)
    log.info(f"Saved final DINO checkpoint: {latest_path}")
    return latest_path