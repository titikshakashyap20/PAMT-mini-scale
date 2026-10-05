"""
PAMT — Transformer encoder (intra-modal interaction stage).

Paper reference: Section III-C, Eqs. (3)-(4):
    t'_l = MSA(LN(t_{l-1})) + t_{l-1},   l = 1...L
    t_l  = MLP(LN(t'_l)) + t'_l,         l = 1...L

"The Transformer model used by both branches is a ViT-like structure [55]
... Input sequence length and layer count are the only differences between
the two branches' network architecture, but the overall structure is same."

This module is used for BOTH branches:
    - GENE branch: depth=2  (Table III best config)
    - WSI  branch: depth=6  (Table III best config)

HYPERPARAMETERS NOT SPECIFIED IN THE PAPER (flagged explicitly, kept
configurable rather than silently assumed):
    - num_heads: paper never states K.
        GENE branch default: 4 (head_dim=64), matching the head_dim of the
        DINO ViT-small backbone the paper uses elsewhere (384/6=64).
        WSI branch default: 6 (head_dim=64), matching DINO ViT-small's own
        384-dim/6-head config directly, since WSI branch tokens are 384-dim.
        Neither is stated for PAMT's own Transformer — both are the most
        grounded choices available given what IS stated elsewhere in the
        paper (the DINO backbone config), not values from the PAMT text.
    - mlp_ratio: paper never states this. Default 4.0 is the "vanilla ViT"
      (Dosovitskiy et al. [55], the paper's own cited reference) default.
    - ln_eps: paper never states this. Default 1e-6 is ViT-paper convention.

Values the paper DOES specify and that ARE hardcoded/used correctly:
    - drop_rate = 0.1 (Sec. IV-A2, Implementation details)
    - pre-norm + residual-after-block structure (Eqs. 3-4)

VARIABLE-LENGTH SEQUENCE SUPPORT (engineering necessity, not a paper
deviation): the paper always has a fixed sequence length (M=186 pathways,
N=500 patches by construction: K=50 clusters x S=10 patches/cluster). A
reduced-scale WSI patch selection can yield a different patch count N per
WSI, which requires padding + a key_padding_mask so the Transformer does
not attend to nonexistent patches. This class supports an OPTIONAL
key_padding_mask (default None). When None, behavior is identical to
before (verified via regression test) — this is additive, not a change to
existing GENE-branch behavior.
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Standard multi-head self-attention. No CLS, no positional logic — pure sequence op.
    Optional key_padding_mask: (B, N) bool, True = this key position is padding
    and must be excluded from every query's softmax. When None, behavior is
    identical to the pre-mask version verified in Stage 2."""
    def __init__(self, dim, num_heads=16, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key_padding_mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        if key_padding_mask is not None:
            # (B, N) -> (B, 1, 1, N): broadcasts over heads and query positions.
            # Masks padded KEYS only — a padded query is still allowed to attend
            # to real keys (its own output is discarded downstream anyway), and
            # since every WSI in this dataset has >=94 real patches, no query
            # row is ever fully masked, so no all-(-inf) softmax rows can occur.
            mask = key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Pre-norm + residual, exactly Eqs. 3-4."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio),
                        act_layer=nn.GELU, drop=drop)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)  # Eq. 3
        x = x + self.mlp(self.norm2(x))                                     # Eq. 4
        return x


class TransformerEncoder(nn.Module):
    """Modality-agnostic stack of Blocks. No CLS, no positional embedding —
    those are owned by the calling branch. key_padding_mask is optional and
    passed straight through to every block; GENE branch never supplies one."""
    def __init__(self, dim, depth, num_heads=16, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

    def forward(self, x, key_padding_mask=None):
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x