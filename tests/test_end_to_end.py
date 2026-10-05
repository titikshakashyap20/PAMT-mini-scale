"""
Final end-to-end integration test.

Runs REAL processed data (never random tensors) through the complete
Stage 1-5B forward pass plus Stage 6 loss, then proves the training
mechanics actually work: backward() succeeds, gradients exist and are
finite, optimizer.step() succeeds, and at least one trainable parameter
provably changes value after a single optimizer step.

This test does NOT judge model quality (that's what C-index / training
history are for) -- it only proves the pipeline is wired correctly and
is trainable end to end.
"""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, "src")

from gene_branch.gene_branch import GeneBranch
from wsi_branch.wsi_branch import WSIBranch
from wsi_branch.wsi_reduction import WSIReduction
from fusion import PathwayToPatchFusion
from scripts.risk_head import SurvivalRiskHead
from gene_branch.contrastive_loss import PathwayPatchContrastiveLoss
from scripts.survival_loss import SurvivalLoss, load_os_labels


GENE_DIR = Path("data/processed/gene")
WSI_DIR = Path("data/processed/wsi")
CLINICAL_PATIENT_PATH = (
    "data/raw/blca_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if detail and not condition:
        print(f"[{status}] {name} -- {detail}")
    else:
        print(f"[{status}] {name}")
    return condition


def extract_patch_array(npz_path):
    with np.load(npz_path) as npz:
        keys = list(npz.files)
        if len(keys) == 1:
            return npz[keys[0]]
        candidates = [
            k for k in keys
            if any(word in k.lower() for word in ("patch", "feature", "embed"))
        ]
        if not candidates:
            raise KeyError(f"Could not identify patch array in {npz_path}. Available keys: {keys}")
        return npz[candidates[0]]


def load_matched_patients():
    gene_files = sorted(GENE_DIR.glob("*.npy"))
    if not gene_files:
        raise FileNotFoundError(f"No gene .npy files found under {GENE_DIR}")
    wsi_files = list(WSI_DIR.glob("*_patches.npz"))
    if not wsi_files:
        raise FileNotFoundError(f"No *_patches.npz files found under {WSI_DIR}")

    patient_ids, gene_arrays, wsi_arrays = [], [], []
    for gene_file in gene_files:
        patient_id = gene_file.stem
        matches = [f for f in wsi_files if f.name.startswith(patient_id + "-")]
        if len(matches) == 0:
            raise FileNotFoundError(f"No WSI file found for {patient_id}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple WSI files found for {patient_id}: {[f.name for f in matches]}")

        gene_array = np.load(gene_file)
        wsi_array = extract_patch_array(matches[0])

        if gene_array.shape != (186, 4942):
            raise ValueError(f"{patient_id}: unexpected gene shape {gene_array.shape}")
        if wsi_array.ndim != 2 or wsi_array.shape[1] != 384:
            raise ValueError(f"{patient_id}: unexpected WSI shape {wsi_array.shape}")

        patient_ids.append(patient_id)
        gene_arrays.append(gene_array)
        wsi_arrays.append(wsi_array)

    return patient_ids, gene_arrays, wsi_arrays


def build_batch(gene_arrays, wsi_arrays):
    B = len(gene_arrays)
    gene_input = torch.tensor(np.stack(gene_arrays, axis=0), dtype=torch.float32)
    max_patches = max(a.shape[0] for a in wsi_arrays)
    feature_dim = wsi_arrays[0].shape[1]
    wsi_input = torch.zeros(B, max_patches, feature_dim, dtype=torch.float32)
    patch_mask = torch.ones(B, max_patches, dtype=torch.bool)
    for i, patches in enumerate(wsi_arrays):
        n = patches.shape[0]
        wsi_input[i, :n, :] = torch.tensor(patches, dtype=torch.float32)
        patch_mask[i, :n] = False
    return gene_input, wsi_input, patch_mask


def main():
    all_passed = True

    print("=" * 60)
    print("END-TO-END INTEGRATION TEST (real data)")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Load real data
    # ------------------------------------------------------------
    patient_ids, gene_arrays, wsi_arrays = load_matched_patients()
    gene_input, wsi_input, patch_mask = build_batch(gene_arrays, wsi_arrays)
    labels = load_os_labels(CLINICAL_PATIENT_PATH, patient_ids)
    T, S = labels.futime, labels.fustat

    all_passed &= check("gene input shape (15,186,4942)", tuple(gene_input.shape) == (15, 186, 4942))
    all_passed &= check("WSI input shape (15,162,384)", tuple(wsi_input.shape) == (15, 162, 384))

    # ------------------------------------------------------------
    # 2. Build pipeline
    # ------------------------------------------------------------
    gene_branch = GeneBranch()
    wsi_branch = WSIBranch()
    wsi_reduction = WSIReduction()
    fusion = PathwayToPatchFusion(dim=256, num_heads=16)
    risk_head = SurvivalRiskHead(num_pathway_tokens=186, num_classes=1)
    contrastive_loss_fn = PathwayPatchContrastiveLoss(top_h=2)
    trainable_modules = [gene_branch, wsi_branch, wsi_reduction, fusion, risk_head]
    loss_fn = SurvivalLoss(weight_decay=5e-4, alpha=1.0, beta=0.8)

    # ------------------------------------------------------------
    # 3. Forward pass, checking every intermediate shape
    # ------------------------------------------------------------
    xP = gene_branch(gene_input)
    all_passed &= check("xP shape (15,186,256)", tuple(xP.shape) == (15, 186, 256))

    xI_384 = wsi_branch(wsi_input, patch_mask)
    all_passed &= check("xI_384 shape (15,162,384)", tuple(xI_384.shape) == (15, 162, 384))

    xI, mask_out = wsi_reduction(xI_384, patch_mask)
    all_passed &= check("xI shape (15,162,256)", tuple(xI.shape) == (15, 162, 256))

    CA, _attn = fusion(xP, xI, mask_out)
    all_passed &= check("CA shape (15,186,256)", tuple(CA.shape) == (15, 186, 256))

    R = risk_head(xP, CA)
    all_passed &= check("R shape (15,1)", tuple(R.shape) == (15, 1))

    L3, _info = contrastive_loss_fn(xP, xI, mask_out)
    all_passed &= check("L3 is scalar", L3.dim() == 0)

    total_loss, diagnostics = loss_fn(R, T, S, L3, trainable_modules)
    all_passed &= check("L1 is scalar", diagnostics["L1"].dim() == 0)
    all_passed &= check("L2_raw is scalar", diagnostics["L2_raw"].dim() == 0)
    all_passed &= check("total_loss is scalar", total_loss.dim() == 0)
    all_passed &= check("total_loss is finite", bool(torch.isfinite(total_loss).item()))

    # ------------------------------------------------------------
    # 4. Backward + optimizer step actually train the model
    # ------------------------------------------------------------
    all_params = [p for m in trainable_modules for p in m.parameters()]
    optimizer = optim.AdamW(all_params, lr=1e-3, weight_decay=0.0)

    # Snapshot one real parameter BEFORE the step, to prove it moves.
    snapshot_param = next(p for p in gene_branch.parameters() if p.requires_grad)
    snapshot_before = snapshot_param.detach().clone()

    optimizer.zero_grad()

    backward_ok = True
    try:
        total_loss.backward()
    except Exception as e:
        backward_ok = False
        print(f"backward() raised: {e}")
    all_passed &= check("backward() succeeds", backward_ok)

    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in all_params
    )
    all_passed &= check("all trainable parameters have finite gradients", grads_ok)

    step_ok = True
    try:
        optimizer.step()
    except Exception as e:
        step_ok = False
        print(f"optimizer.step() raised: {e}")
    all_passed &= check("optimizer.step() succeeds", step_ok)

    snapshot_after = snapshot_param.detach().clone()
    param_changed = not torch.equal(snapshot_before, snapshot_after)
    all_passed &= check(
        "at least one trainable parameter changed after optimizer.step()",
        param_changed,
    )

    print("\n" + "=" * 60)
    print("END-TO-END TEST: PASSED" if all_passed else "END-TO-END TEST: FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())