"""
PAMT — WSI branch input embedding stage.

Paper reference: Section III-B ("Pathology Branch Input"), Section III-C,
Eq. (2), Fig. 2(a).

UNLIKE the pathway side, the paper applies NO fully-connected projection
here: "The patch embedding is obtained using DINO model. Through DINO, we
convert the N patches selected from each WSI into N feature vectors."
The DINO output IS the patch embedding (384-dim).

NOTE: the official released model (vit_model_gene_wsi_concat_label.py) has
no CLS token on either branch — pos_wsi_embed is (1, 500, 384), no +1 slot.
This module only adds positional embeddings; no CLS anywhere, matching the
same correction already applied to the GENE branch in Stage 1/2.

VARIABLE PATCH COUNT: the paper always selects exactly N=500 patches per
WSI (K=50 clusters x S=10/cluster, fixed by construction), so the official
code never needs padding. A reduced-scale patch selection (as used here)
yields a different N per WSI (this dataset: 94-162). This module supports
variable N via a fixed-size positional embedding table (sized to
`max_patches`) sliced to the actual N at forward time, plus an optional
padding mask for batching WSIs with different patch counts together. This
is an engineering necessity for batching real variable-length data, not a
change to the paper's method, and has no official-code counterpart to
match since the official code never faces this case.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 384, max_patches: int = 500):
        """
        Args:
            embed_dim: D, patch token dimension. Paper: DINO output = 384.
            max_patches: upper bound on patch sequence length used to size
                the positional embedding table. Paper's true value is 500
                (K=50 x S=10). Kept at 500 here by default so the class
                itself stays paper-faithful; mini-scale callers (this
                dataset's actual range is 94-162) must override this
                explicitly at construction time.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_patches = max_patches

        # Learnable 1D positional embeddings, sized for the max possible
        # sequence length; sliced per-batch to actual N. No CLS.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, patch_embeddings: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        Args:
            patch_embeddings: (B, N, D) DINO features, D=384. N may be
                padded to the batch max; use key_padding_mask to mark pads.
            key_padding_mask: optional (B, N) bool, True = this patch
                position is padding (not a real patch). None = no padding
                in this batch (all WSIs have equal N, or batch_size=1).

        Returns:
            t0: (B, N, D) — WSI branch Transformer input, positional
                embeddings added, no CLS.
            full_mask: (B, N) bool or None — key_padding_mask passed
                through unchanged (no CLS row to extend it with), ready
                to pass straight into TransformerEncoder.
        """
        B, N, D = patch_embeddings.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"
        assert N <= self.max_patches, (
            f"N={N} exceeds max_patches={self.max_patches}; "
            f"increase max_patches if you scale up patch selection."
        )

        pos = self.pos_embed[:, :N, :]
        t0 = patch_embeddings + pos

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, N)

        return t0, key_padding_mask