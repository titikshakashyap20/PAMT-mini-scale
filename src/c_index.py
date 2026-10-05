"""
Concordance Index (C-index) -- paper Eq. 15.

Cindex = (1/n) * sum_{i : Si=1} sum_{j : Tj>Ti} Ind[risk_i > risk_j]

where n = number of comparable pairs (not number of patients).
A pair (i, j) is comparable if patient i had an event (Si=1) and
patient j's survival time is strictly longer than patient i's (Tj > Ti).
Ties in predicted risk count as 0.5 (standard convention).
"""

import torch


def concordance_index(risk: torch.Tensor, T: torch.Tensor, S: torch.Tensor) -> float:
    """
    risk: (B,) predicted risk scores (higher = worse prognosis)
    T:    (B,) survival/follow-up time
    S:    (B,) event indicator, 1 = event (death) observed, 0 = censored

    Returns the C-index as a float in [0, 1], or float('nan') if there are
    no comparable pairs (e.g. all patients censored, or all identical times).
    """
    risk = risk.detach()
    T = T.detach()
    S = S.detach()

    n = risk.shape[0]
    comparable_pairs = 0
    concordant = 0.0

    for i in range(n):
        if S[i].item() != 1:
            continue
        for j in range(n):
            if T[j].item() > T[i].item():
                comparable_pairs += 1
                if risk[i].item() > risk[j].item():
                    concordant += 1.0
                elif risk[i].item() == risk[j].item():
                    concordant += 0.5

    if comparable_pairs == 0:
        return float("nan")

    return concordant / comparable_pairs