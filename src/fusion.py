"""
Stage 5A -- Pathway-to-Patch Cross-Modal Fusion (OFFICIAL-CODE-FAITHFUL)

Paper reference:   Section III-C, Eqs. (5)-(8)
Official reference: YANRUI121/PAMT
                     gene_wsi_predict/vit_model_gene_wsi_concat_label.py
                       class Gene_Guided_Transformer_Fusion

============================================================
CHANGE FROM THE PREVIOUS (PAPER-LITERAL) VERSION
============================================================
REMOVED: the `xPI = cat(Qpath, CA)` concatenation, entirely.

  PAPER Eq. (8): xPI = cat(Qpath, CA(Qpath, Kpatch, Vpatch)).
  The previous mini implementation followed this literally, producing
  (B, 186, 512).

  OFFICIAL CODE's Gene_Guided_Transformer_Fusion.forward() returns ONLY
  the cross-attention output (plus attention weights for
  interpretability) -- shape (B, 186, 256). Qpath (the linearly
  projected query) exists only inside the attention computation; it is
  never separately materialized as its own tensor and never
  concatenated onto anything inside this module.

  CONFIRMED PAPER-vs-CODE DISCREPANCY: Eq. (8) is not what the
  officially released model computes. This module now matches the
  official code: output is CA alone, (B, 186, 256).

  Where does the paper's implied "genotype + phenotype" concatenation
  actually happen, then? In the released model, at the risk-head stage
  (Stage 5B), NOT here: gene_features (xP, pre-fusion) and this
  module's CA output are each independently pooled along the FEATURE
  axis (256 -> 1, keeping 186 tokens), then concatenated along the
  TOKEN axis (186 + 186 = 372), then passed through a single linear
  layer. That is a materially different design from "concatenate two
  256-d streams into a 512-d representation per token" -- it is
  Stage 5B's job to implement, not this module's, and it is why Stage
  5A's job here is now simpler (CA only) than it was under the
  paper-literal reading.

============================================================
WHAT IS UNCHANGED
============================================================
  - Q from GENE (pathway) tokens, K/V from WSI (patch) tokens.  [PAPER + OFFICIAL match]
  - 16 attention heads, single fusion layer (no stacking).      [OFFICIAL-CODE match --
      train.py never stacks Gene_Guided_Transformer_Fusion; the paper's
      own Table IV ablation on stacking multiple layers found no
      benefit, consistent with the released model using exactly one.]
  - softmax(QK^T / sqrt(d)) V cross-attention mechanics.        [PAPER + OFFICIAL match]

WHAT IS MINI-SCALE-ONLY (no official-code counterpart)
  - key_padding_mask on the WSI (key/value) side. Official code has a
    fixed N=500 and no padding branch at all. This mask is carried over
    unchanged from the previous Stage 5A: masked WSI positions get
    -inf attention logits BEFORE softmax (not zeroed after), so valid
    patches never attend into padding.
"""
import torch
import torch.nn as nn


class PathwayToPatchFusion(nn.Module):
    """
    OFFICIAL-CODE-FAITHFUL pathway-to-patch cross-attention fusion.

    Q from GENE tokens (xP), K/V from WSI tokens (xI). Output is CA
    alone, (B, M=186, D=256) -- NOT concatenated with Qpath. See module
    docstring for why this is a confirmed departure from Eq. (8).
    """

    def __init__(self, dim: int = 256, num_heads: int = 16,
                 q_bias: bool = False, kv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop_ratio: float = 0.0,
                 proj_drop_ratio: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # OFFICIAL-CODE match: separate Q (from GENE) and combined KV
        # (from WSI) projections, both bias-free by default.
        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, xP: torch.Tensor, xI: torch.Tensor,
                key_padding_mask: torch.Tensor = None):
        """
        Args:
            xP: (B, M, D) GENE/pathway tokens -- query source.
            xI: (B, N, D) WSI/patch tokens -- key/value source.
            key_padding_mask: (B, N) bool, True = padding, False = valid.
                MINI-SCALE ADAPTATION; pass None if N is uniform.

        Returns:
            CA:   (B, M, D) -- fusion output. OFFICIAL-CODE-FAITHFUL:
                  this is the complete Stage 5A output; no cat(Qpath, CA).
            attn: (B, num_heads, M, N) -- attention weights, retained
                  for interpretability (official code also returns attn).
        """
        assert xP.shape[-1] == xI.shape[-1], "GENE and WSI token dims must match"
        B, M, C = xP.shape
        _, N, _ = xI.shape

        q = self.q(xP).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(xI).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, M, N)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, N)
            # MINI-SCALE ADAPTATION: mask BEFORE softmax.
            neg_inf = torch.finfo(attn.dtype).min
            mask = key_padding_mask[:, None, None, :]  # (B,1,1,N)
            attn = attn.masked_fill(mask, neg_inf)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        CA = (attn @ v).transpose(1, 2).reshape(B, M, C)
        CA = self.proj(CA)
        CA = self.proj_drop(CA)

        return CA, attn