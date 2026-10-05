import torch
import torch.nn as nn

from .pathway_embedding import PathwayEmbedding
from .transformer_encoder import TransformerEncoder


class GeneBranch(nn.Module):
    """
    Stage 2 — GENE branch.

    Input:
        (B, 186, 4942)

    Stage 1:
        PathwayEmbedding -> (B, 186, 256)

    Stage 2:
        Add learnable positional embedding -> (B, 186, 256)
        2 Transformer blocks, 16 attention heads

    Output:
        (B, 186, 256)
    """

    def __init__(
        self,
        in_features: int = 4942,
        hidden_features: int = 1000,
        embed_dim: int = 256,
        num_pathways: int = 186,
        depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        # Stage 1 — reuse the already verified PathwayEmbedding
        self.pathway_embed = PathwayEmbedding(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=embed_dim,
            drop=drop_rate,
        )

        # Stage 2 — learnable positional embedding
        self.pos_gene_embed = nn.Parameter(
            torch.zeros(1, num_pathways, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_gene_embed, std=0.02)

        self.pos_drop = nn.Dropout(drop_rate)

        # Stage 2 — GENE Transformer
        self.encoder = TransformerEncoder(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: (B, 186, 4942) -> (B, 186, 256)
        x = self.pathway_embed(x)

        # Add positional information
        x = x + self.pos_gene_embed

        # Positional dropout
        x = self.pos_drop(x)

        # Stage 2: Transformer
        # (B, 186, 256) -> (B, 186, 256)
        x = self.encoder(x)

        return x