"""
PAMT — full WSI branch: patch embedding (DINO features + positional
embeddings) + Transformer encoder (depth=6 per Table III best config).

Output is x_I, the unimodal WSI feature representation (Sec. III-C),
shape (B, N, 384) — patch tokens after intra-modal (patch-patch)
interaction. No CLS token: the official released fusion model
(vit_model_gene_wsi_concat_label.py) has none on either branch. Feeds
into the same downstream inter-modal alignment (L3) and fusion stages
as x_P from the GENE branch.
"""

import torch
import torch.nn as nn

from .patch_embedding import PatchEmbedding
from gene_branch.transformer_encoder import TransformerEncoder


class WSIBranch(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,      # DINO output dim, per paper
        max_patches: int = 162,    # this dataset's actual ceiling (range: 94-162).
                                    # NOT the paper's 500 — see patch_embedding.py
                                    # docstring for why the class default stays 500
                                    # while this call site overrides it.
        depth: int = 6,            # Table III best config for WSI branch
        num_heads: int = 16,       # official code: same 16 heads as GENE branch
        mlp_ratio: float = 4.0,    # official code default, never overridden
        drop_rate: float = 0.1,    # paper-specified implementation detail
    ):
        super().__init__()
        self.embedding = PatchEmbedding(embed_dim=embed_dim, max_patches=max_patches)
        self.encoder = TransformerEncoder(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=drop_rate,
        )

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        Args:
            patch_embeddings: (B, N, 384) — DINO features per selected patch.
            key_padding_mask: optional (B, N) bool, True = padding.
        Returns:
            x_I: (B, N, 384) — WSI branch unimodal feature representation.
        """
        t0, mask = self.embedding(patch_embeddings, key_padding_mask)
        return self.encoder(t0, key_padding_mask=mask)