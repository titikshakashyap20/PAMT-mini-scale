"""
Stage 1 — Pathway-based gene embedding.

PAPER SAYS (Sec. III-A, Fig. 2/3):
    - Gene expression is decomposed into M=186 pathway vectors {P1,...,PM}.
    - Each Pm lives in the full gene universe, zero-masked outside its own
      pathway's genes (E = E_indicator (*) e', Fig. 3), NOT a variable-length
      per-pathway vector.
    - "use fully connected layers to reduce it to 256 dimensions" -- prose
      only, no layer sizes, no activation, no dropout specified.
    - Eq. 2 prepends a learnable [CLS] token to this branch.

OFFICIAL CODE SAYS (YANRUI121/PAMT, gene_wsi_predict/vit_model_gene_wsi_concat_label.py):
    - class EmbedReduction: Linear(in,hidden) -> GELU -> Dropout ->
      Linear(hidden,out) -> Dropout  (no activation after the 2nd linear)
    - Instantiated as EmbedReduction(in_features=5245, hidden_features=1000,
      out_features=256, act_layer=nn.GELU)  [line 273]
    - No cls_token anywhere in this file. pos_gene_embed is shaped
      (1, 186, 256) -- confirmed no +1 slot for CLS anywhere in the actual
      fusion model (cls_token only exists in the separate WSI-only baseline
      file, vit_model_one_cls.py, which is a different model entirely).

OUR MINI ADAPTATION:
    - in_features=4942 (our filtered gene universe across the 186 KEGG
      pathways) instead of the paper's 5245 -- a data-coverage difference,
      not an architectural one.
    - drop=0.1, matching the paper's stated "Implementation details" drop
      rate, NOT train.py's hardcoded 0.2 (that mismatch is a separate,
      already-flagged footnote -- not resolved by this change).
    - No CLS token, per Change A above.
"""

import torch
import torch.nn as nn


class PathwayEmbedding(nn.Module):
    """
    Projects each of the 186 pathway-masked gene vectors down to a 256-dim
    pathway token. Pre-transformer, pre-position-embedding -- Stage 2 adds
    those on top of this module's output.

    Input:  (B, 186, 4942)  -- pathway-based gene expression matrix
    Output: (B, 186, 256)   -- pathway tokens
    """

    def __init__(
        self,
        in_features: int = 4942,
        hidden_features: int = 1000,
        out_features: int = 256,
        drop: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 186, 4942)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x  # (B, 186, 256) -- no CLS row, matches official pos_gene_embed shape