"""
Stage 5B -- Survival Risk Head (OFFICIAL-CODE-FAITHFUL, Option A: strict)

Paper reference:   Section III-C, Eq. (9): R = MLP(LN(xPI))
Official reference: YANRUI121/PAMT
                     gene_wsi_predict/vit_model_gene_wsi_concat_label.py
                     VisionTransformer.forward() (self.gap_gene, self.gap_fusion,
                     torch.cat(..., dim=1), self.head)

CONFIRMED PAPER-vs-CODE DISCREPANCY (documented, not silently absorbed):
  Eq. (9) implies a hidden-layer MLP applied to a LayerNorm'd xPI. The
  released model has neither: no LayerNorm at this point (the only LN
  anywhere is norm_gene, already applied inside the Stage 2 gene
  transformer, well before fusion), and self.head is a single
  nn.Linear(372, num_classes) with no hidden layer, activation, or
  dropout. This module implements exactly what the official code does
  (Option A, per your instruction) -- not Eq. (9) literally.

ARCHITECTURE (official-code match, no additions):
  xP  (B,186,256) --AdaptiveAvgPool2d((186,1))--> (B,186,1)
  CA  (B,186,256) --AdaptiveAvgPool2d((186,1))--> (B,186,1)
  cat([pooled_xP, pooled_CA], dim=1) -> (B,372,1) -> squeeze(-1) -> (B,372)
  Linear(372, 1) -> R, (B,1)

Note on pooling axis: AdaptiveAvgPool2d((186,1)) applied to a
(B,186,256) tensor pools the trailing 256-dim FEATURE axis down to 1,
leaving the 186 TOKEN axis untouched. This is the same easy-to-misread
detail flagged in the earlier audit -- it is not pooling over tokens.
"""
import torch
import torch.nn as nn


class SurvivalRiskHead(nn.Module):
    """
    OFFICIAL-CODE-FAITHFUL risk head. No LayerNorm, no MLP hidden layer,
    no activation, no dropout -- exactly self.gap_gene / self.gap_fusion /
    cat / self.head from the released model.
    """

    def __init__(self, num_pathway_tokens: int = 186, num_classes: int = 1):
        super().__init__()
        self.gap_gene = nn.AdaptiveAvgPool2d((num_pathway_tokens, 1))
        self.gap_fusion = nn.AdaptiveAvgPool2d((num_pathway_tokens, 1))
        self.head = nn.Linear(num_pathway_tokens * 2, num_classes)

    def forward(self, xP: torch.Tensor, CA: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xP: (B, 186, 256) -- Stage 2 gene-transformer output (pre-fusion).
            CA: (B, 186, 256) -- Stage 5A fusion output.

        Returns:
            R: (B, num_classes) -- survival risk. (B,1) with default num_classes=1.
        """
        assert xP.shape == CA.shape, "xP and CA must have matching (B,186,256) shapes"

        gene_pooled = self.gap_gene(xP)      # (B,186,1)
        fusion_pooled = self.gap_fusion(CA)  # (B,186,1)

        fused = torch.cat([gene_pooled, fusion_pooled], dim=1)  # (B,372,1)
        fused = fused.squeeze(-1)                                # (B,372)

        R = self.head(fused)  # (B, num_classes)
        return R