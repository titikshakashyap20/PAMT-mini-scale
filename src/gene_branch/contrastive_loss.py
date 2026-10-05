"""
Stage 4 -- L3 Pathway-to-Patch Contrastive Loss (OFFICIAL-CODE-FAITHFUL)

Paper reference:   Section III-D, Eqs. (11)-(14)
Official reference: YANRUI121/PAMT
                     gene_wsi_predict/vit_model_gene_wsi_concat_label.py
                       (raw similarity computed inside model.forward())
                     gene_wsi_predict/utils_cox.py :: train_one_epoch / evaluate
                       (top-h label construction via sort-descending + CrossEntropyLoss)

============================================================
CHANGE FROM THE PREVIOUS (PAPER-LITERAL) VERSION
============================================================
REMOVED: the learnable temperature parameter tau.

  PAPER (Eq. 13) explicitly states tau is "a learnable temperature
  parameter." The previous mini implementation followed this literally.

  OFFICIAL CODE does not use it. `vit_model_gene_wsi_concat.py` defines
  `self.logit_scale = nn.Parameter(...)`, but the line that would apply
  it (`gene2wsi_feature = logit_scale * gene_features_reduction @ ...`)
  is commented out. The model file `train.py` actually imports
  (`vit_model_gene_wsi_concat_label.py`) doesn't even define
  logit_scale/tau -- the similarity matrix is used completely raw, and
  the softmax that follows (implicit inside nn.CrossEntropyLoss in
  utils_cox.py) has no temperature division anywhere.

  This is a CONFIRMED PAPER-vs-CODE DISCREPANCY. Per your instruction,
  this module now matches the official code: no temperature anywhere.

WHAT WE LOSE BY DROPPING TAU (documented, not silently absorbed):
  - No ability for the model to learn how "peaked" vs. "soft" the
    pathway-to-patch softmax distribution should be. With no tau,
    softmax sharpness is fixed entirely by the raw dot-product
    magnitudes, which depend on embedding scale/initialization rather
    than anything this loss's optimizer can directly tune.
  - Literal fidelity to Eq. (13)'s wording is lost. This module is now
    OFFICIAL-CODE-FAITHFUL, not PAPER-LITERAL, for this one detail.
  - This is a known-workable configuration (it's what the released
    model actually trains with), not a guess -- but it is a genuine
    capability loss relative to the paper's stated design.

============================================================
WHAT IS UNCHANGED (kept exactly as before, re-labeled where relevant)
============================================================
  - Raw inner product S = xP @ xI^T                    [PAPER + OFFICIAL match]
  - Top-h=2 positive labels per pathway row             [PAPER + OFFICIAL match]
      Implemented here via topk+scatter rather than the official
      sort-descending + fixed-position-label trick. Both constructions
      select the identical set of positions (whichever patches have the
      top-2 raw similarity for that pathway row) -- OFFICIAL-CODE-
      EQUIVALENT, just an easier-to-follow (and mask-aware) mechanic.
  - Padding exclusion                                   [MINI-SCALE ADAPTATION]
      No official-code counterpart: the released model has a fixed
      N=500 with no padding branch at all. This is ours alone, needed
      because our N varies (94-162) per patient.
  - Reduction: sum of negative log-probs at the 2 positive positions,
    mean over the 186 pathway rows, mean over batch.    [OFFICIAL-CODE-EQUIVALENT]
      nn.CrossEntropyLoss(reduction='mean') applied to a
      (B, N, M)-shaped input against a 0/1 (non-normalized) float
      target of the same shape computes exactly
      -sum(target * log_softmax(input)) per (batch, pathway) row, then
      averages over all B*M rows. Since M=186 is constant across the
      batch, mean-over-M-then-mean-over-B == mean over all B*M values.
      Same result, different nesting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PathwayPatchContrastiveLoss(nn.Module):
    """
    OFFICIAL-CODE-FAITHFUL label-free pathway-to-patch contrastive loss (L3).

    No learnable parameters at all (no tau). Raw inner product, top-h=2
    positive labels per pathway row, cross-entropy against those labels,
    with padded WSI patches excluded from label selection and from the
    softmax (MINI-SCALE ADAPTATION -- official code has no padding).
    """

    def __init__(self, top_h: int = 2):
        super().__init__()
        # PAPER-SPECIFIED: Top_h = 2 (Section IV-A, "Implementation details";
        # also OFFICIAL-CODE-SPECIFIED: train.py's default --topK is 2).
        self.top_h = top_h

    def forward(self, xP: torch.Tensor, xI: torch.Tensor,
                key_padding_mask: torch.Tensor = None):
        """
        Args:
            xP: (B, M=186, D=256) GENE/pathway tokens, Stage-2 output.
            xI: (B, N, D=256)     WSI/patch tokens, Stage-3.5 output.
            key_padding_mask: (B, N) bool, True = padding, False = valid.
                MINI-SCALE ADAPTATION. Pass None if N is uniform (no padding).

        Returns:
            loss: scalar tensor.
            info: dict with S, S_prime, Y_h, topk_idx (for verification/debug).
        """
        B, M, D = xP.shape
        _, N, _ = xI.shape
        assert xI.shape[-1] == D, "GENE and WSI token dims must match"

        # Eq. (11) / OFFICIAL match: raw inner product, no logit_scale.
        S = xP @ xI.transpose(-2, -1)  # (B, M, N)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, N)
            neg_inf = torch.finfo(S.dtype).min
            pad = key_padding_mask.unsqueeze(1)  # (B, 1, N) broadcasts over M
            S_masked = S.masked_fill(pad, neg_inf)
        else:
            S_masked = S

        # Eq. (12): top-h similarities per pathway row.
        top_h = min(self.top_h, N)
        topk_vals, topk_idx = torch.topk(S_masked, k=top_h, dim=-1)  # (B, M, top_h)

        Y_h = torch.zeros_like(S_masked)
        Y_h.scatter_(-1, topk_idx, 1.0)
        if key_padding_mask is not None:
            # Defensive: a padded position must never be a positive label,
            # even in a degenerate all-masked edge case.
            Y_h = Y_h.masked_fill(key_padding_mask.unsqueeze(1), 0.0)

        # Eq. (13), OFFICIAL-CODE-FAITHFUL: plain softmax, NO tau division.
        S_prime = F.softmax(S_masked, dim=-1)  # (B, M, N); padded cols -> 0

        # Eq. (14): CE between S_prime and Y_h, restricted to the top-h
        # labeled positions == sum of negative log-probs at those positions.
        # log_softmax used directly for numerical stability (avoids log(0)
        # at masked-out columns rather than log(S_prime)).
        log_S_prime = F.log_softmax(S_masked, dim=-1)
        per_row_loss = -(Y_h * log_S_prime).sum(dim=-1)  # (B, M)

        # OUR RESOLUTION / OFFICIAL-CODE-EQUIVALENT reduction (see module docstring).
        loss = per_row_loss.mean(dim=1).mean()

        info = {
            "S": S,
            "S_prime": S_prime,
            "Y_h": Y_h,
            "topk_idx": topk_idx,
        }
        return loss, info