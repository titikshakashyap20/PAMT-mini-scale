"""
Stage 6 — Training objective: Loss = L1 + alpha*L2 + beta*L3

Approved design (locked in for this file):
  L1 (Cox negative partial log-likelihood):
    - risk set direction: R_mat[i, j] = 1 if T[j] >= T[i]        <-- OFFICIAL-CODE direction
      (paper's printed "Tj <= Ti" in Eq. 10 is treated as a typo; Tj>=Ti is the
       standard Cox convention: the risk set at time Ti is everyone still at
       risk, i.e. surviving at least as long as patient i.)
    - reduction: batch mean over patients with S=1                <-- OUR/OFFICIAL-STYLE
      REDUCTION CHOICE, not something Eq. 10 itself specifies (Eq. 10 is a raw
      sum over {i : Si=1}; we average instead, matching official code's
      batch-size-normalization behavior).

  L2 (paper-literal regularization, Eq. in Sec. III-D, weight_decay=5E-4):
    - p=2 norm, summed over WEIGHT parameters only (bias parameters excluded),
      across every module passed in (GeneBranch, WSIBranch, WSIReduction,
      Fusion, RiskHead).
    - This is paper-literal, NOT the official code's p=1 Regularization class
      (that discrepancy was flagged and explicitly overridden per your Stage 6
      decision #2 — p=2 is used here on purpose).

  L3 (label-free pathway-to-patch contrastive loss):
    - NOT recomputed here. Stage 4's already-verified L3 scalar is passed in
      as an argument and simply scaled by beta. This file has no dependency
      on the Stage 4 module internals.

  Combination:
    Loss = L1 + alpha * weight_decay * L2_raw + beta * L3
    with alpha=1, weight_decay=5e-4, beta=0.8 (paper values, per your
    Stage 6 decisions #2-#4). alpha is applied as a literal multiplier on the
    weight_decay-scaled L2 term for symmetry with L3's beta, even though with
    alpha=1 it has no numeric effect.

No C-index or any other evaluation metric is computed inside this module —
this file only returns the loss and its raw components for inspection.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Survival label loading (OS_STATUS / OS_MONTHS from data_clinical_patient.txt)
# ---------------------------------------------------------------------------

@dataclass
class SurvivalLabels:
    patient_ids: list[str]
    futime: torch.Tensor  # (N,) float, units = MONTHS (OS_MONTHS, not days)
    fustat: torch.Tensor  # (N,) float, 0/1 (1 = DECEASED)


def load_os_labels(clinical_patient_path: str, patient_ids: Sequence[str]) -> SurvivalLabels:
    """
    Parse a cBioPortal-style PanCanAtlas `data_clinical_patient.txt` file and
    return (futime, fustat) for `patient_ids`, in that exact order.

    futime = OS_MONTHS (continuous, units = MONTHS -- documented, not days;
             this is a monotonic rescaling of days so it does not change any
             Tj>=Ti comparison used by the Cox loss below).
    fustat = 1 if OS_STATUS startswith "1:" (DECEASED) else 0.

    This file has no DAYS_TO_DEATH column (DAYS_LAST_FOLLOWUP is only
    populated for living patients in this export), so OS_STATUS/OS_MONTHS is
    the correct, non-fabricated source for these labels -- not a fallback.

    Raises KeyError if any requested patient_id is missing from the file, and
    ValueError if OS_STATUS/OS_MONTHS are missing/unparseable for a requested
    patient -- this function never invents a label.
    """
    with open(clinical_patient_path, "r", newline="") as f:
        raw_lines = f.readlines()

    header_idx = None
    for i, line in enumerate(raw_lines):
        first_field = line.split("\t", 1)[0].strip()
        if first_field == "PATIENT_ID":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find a 'PATIENT_ID' header row in {clinical_patient_path}")

    header = raw_lines[header_idx].rstrip("\n").split("\t")
    col = {name: i for i, name in enumerate(header)}
    for required in ("PATIENT_ID", "OS_STATUS", "OS_MONTHS"):
        if required not in col:
            raise ValueError(f"Column '{required}' not found in {clinical_patient_path}")

    rows: dict[str, list[str]] = {}
    reader = csv.reader(raw_lines[header_idx + 1 :], delimiter="\t")
    for fields in reader:
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) <= max(col.values()):
            continue
        rows[fields[col["PATIENT_ID"]]] = fields

    futime_list: list[float] = []
    fustat_list: list[float] = []
    for pid in patient_ids:
        if pid not in rows:
            raise KeyError(f"Patient id '{pid}' not found in {clinical_patient_path}")
        fields = rows[pid]

        os_status_raw = fields[col["OS_STATUS"]].strip()
        os_months_raw = fields[col["OS_MONTHS"]].strip()
        if not os_status_raw or not os_months_raw:
            raise ValueError(f"Missing OS_STATUS/OS_MONTHS for patient '{pid}'")

        if os_status_raw.startswith("1:"):
            fustat = 1.0
        elif os_status_raw.startswith("0:"):
            fustat = 0.0
        else:
            raise ValueError(f"Unrecognized OS_STATUS '{os_status_raw}' for patient '{pid}'")

        try:
            futime = float(os_months_raw)
        except ValueError as e:
            raise ValueError(f"Unparseable OS_MONTHS '{os_months_raw}' for patient '{pid}'") from e

        futime_list.append(futime)
        fustat_list.append(fustat)

    return SurvivalLabels(
        patient_ids=list(patient_ids),
        futime=torch.tensor(futime_list, dtype=torch.float32),
        fustat=torch.tensor(fustat_list, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# L1 — Cox negative partial log-likelihood (official risk-set direction)
# ---------------------------------------------------------------------------

def cox_neg_log_likelihood(
    theta: torch.Tensor, T: torch.Tensor, S: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    theta: (B,) risk scores (R.squeeze(-1) from Stage 5B's RiskHead)
    T:     (B,) futime
    S:     (B,) fustat (0/1)

    R_mat[i, j] = 1 if T[j] >= T[i] else 0   -- OFFICIAL-CODE direction.
    (paper Eq. 10 prints Tj<=Ti; flagged as a typo, see module docstring)

    L1 = -mean_i( (theta_i - logsumexp_j(theta_j ; masked by R_mat[i,:])) * S_i )

    Uses logsumexp with -inf masking (rather than log(sum(exp(.)*mask))) for
    numerical stability; the diagonal (j=i) always satisfies T[j]>=T[i], so
    every risk set is non-empty and log_risk is always finite -- when S is
    all zero every loss term is exactly zeroed, giving L1==0.0 (not NaN).

    Returns (L1, R_mat) -- R_mat returned for test-time inspection only.
    """
    if theta.dim() != 1:
        raise ValueError(f"theta must be 1-D (B,), got shape {tuple(theta.shape)}")
    B = theta.shape[0]

    T_i = T.view(B, 1)
    T_j = T.view(1, B)
    R_mat = (T_j >= T_i).to(theta.dtype)  # (B, B), R_mat[i, j]

    theta_j = theta.view(1, B).expand(B, B)
    masked_theta = theta_j.masked_fill(R_mat == 0, float("-inf"))
    log_risk = torch.logsumexp(masked_theta, dim=1)  # (B,)

    loss_terms = (theta - log_risk) * S
    L1 = -loss_terms.mean()  # batch-mean reduction -- OUR/OFFICIAL-STYLE CHOICE, see docstring
    return L1, R_mat


# ---------------------------------------------------------------------------
# L2 — paper-literal p=2 weight regularization (bias params excluded)
# ---------------------------------------------------------------------------

def l2_weight_regularization(modules: Iterable[nn.Module]) -> torch.Tensor:
    """
    sum over all params whose name contains 'weight' (i.e. every Linear/
    LayerNorm/embedding weight tensor, no biases), across all given modules,
    of ||w||_2. Paper-literal p=2, matching Sec. III-D's L2 = ||Theta||_2
    restricted to weight-named params (mirrors the official code's naming
    filter, but keeps p=2 per your Stage 6 decision #2).

    Returns a 0-d tensor (differentiable, on the same device/dtype as the
    first weight param found). Raises ValueError if no weight params found
    at all (almost certainly a wiring bug, not a valid zero-reg model).
    """
    total = None
    for module in modules:
        for name, param in module.named_parameters():
            if "weight" not in name:
                continue
            if not param.requires_grad:
                continue
            term = param.norm(p=2)
            total = term if total is None else total + term
    if total is None:
        raise ValueError("l2_weight_regularization found no 'weight'-named parameters")
    return total


# ---------------------------------------------------------------------------
# Combined Stage 6 loss
# ---------------------------------------------------------------------------

class SurvivalLoss(nn.Module):
    """
    Loss = L1 + alpha * weight_decay * L2_raw + beta * L3

    L3 is NOT computed by this module -- pass in the already-computed Stage 4
    scalar each call. This module owns L1 and L2 only.
    """

    def __init__(self, weight_decay: float = 5e-4, alpha: float = 1.0, beta: float = 0.8):
        super().__init__()
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        R: torch.Tensor,
        T: torch.Tensor,
        S: torch.Tensor,
        L3: torch.Tensor,
        modules: Iterable[nn.Module],
    ) -> tuple[torch.Tensor, dict]:
        if R.dim() != 2 or R.shape[1] != 1:
            raise ValueError(f"R must be shape (B,1), got {tuple(R.shape)}")
        theta = R.squeeze(-1)

        L1, R_mat = cox_neg_log_likelihood(theta, T, S)
        L2_raw = l2_weight_regularization(modules)
        L2_weighted = self.alpha * self.weight_decay * L2_raw
        L3_weighted = self.beta * L3

        total = L1 + L2_weighted + L3_weighted

        diagnostics = {
            "L1": L1.detach(),
            "L2_raw": L2_raw.detach(),
            "L2_weighted": L2_weighted.detach(),
            "L3_weighted": L3_weighted.detach() if torch.is_tensor(L3_weighted) else L3_weighted,
            "R_mat": R_mat.detach(),
            "total": total.detach(),
        }
        return total, diagnostics