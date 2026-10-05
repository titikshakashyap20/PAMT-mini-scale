"""
PAMT — Stage 3.5: WSI post-Transformer reduction.

Paper reference: Eq. 6 ("Kpatch = FC(xI), Vpatch = FC(xI)") — the paper
never specifies this step's internal dimensions; it's silent on the
reduction's architecture.

Official-code reference: EmbedReduction(in_features=384, hidden_features=640,
out_features=256, act_layer=nn.GELU), forward: fc1 -> GELU -> Dropout ->
fc2 -> Dropout, no activation after fc2. Same class/pattern as the GENE
branch's Stage 1 embedding (there: 4942/5245 -> 1000 -> 256), different
in/hidden dims here. Applied ONCE, immediately after the WSI Transformer
encoder (Stage 3) finishes, before either L3 or fusion consume the result.

Dropout rate inside EmbedReduction has no specified numeric value in the
official code (only drop-path elsewhere, a different mechanism). Using
0.1 here for consistency with the same choice already made in Stage 1.

Pointwise transform: Linear layers act independently per patch position,
no attention or cross-token mixing. Padded positions pass through
unchanged in shape and are NOT explicitly zeroed — the key_padding_mask
is threaded through unmodified for downstream stages (L3, fusion) to
apply where it actually matters (inside their own attention/similarity
computations).
"""

import torch
import torch.nn as nn


class WSIReduction(nn.Module):
    def __init__(self, in_features: int = 384, hidden_features: int = 640,
                 out_features: int = 256, drop_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        Args:
            x: (B, N, 384) — WSI branch Transformer output (Stage 3).
            key_padding_mask: optional (B, N) bool, True = padding.
                Passed through unchanged; this module doesn't consult it.
        Returns:
            x: (B, N, 256)
            key_padding_mask: unchanged, same tensor passed in.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x, key_padding_mask