"""
Reproducibility diagnostic for the L3 discrepancy between
training_history.csv (epoch 467, train-mode) and evaluate_best.py's
reload (eval-mode).

This script does NOT modify, retrain, or "fix" anything. It only observes
the already-verified Stage 1-6 modules to confirm or rule out each
hypothesis in the investigation checklist:

  1. train/eval mode + dropout
  2. whether Stage 4 (PathwayPatchContrastiveLoss) has any state missing
     from the checkpoint
  3. whether the checkpoint saves all Stage-4-related trainable params
  4/5/6. whether forward_pipeline()'s behavior / L3's tensors differ
     between train and eval mode due to a random operation

Method:
  - Load best_model.pt (the exact epoch-467 weights).
  - Run forward_pipeline() N times in TRAIN mode, with no optimizer step
    in between (weights never change) -- if L3 varies run to run despite
    identical weights and identical input data, the only possible source
    is a stochastic op inside the forward pass (dropout).
  - Run forward_pipeline() N times in EVAL mode -- if L3 is bit-identical
    across all N runs, eval mode is confirmed deterministic, and the
    train-mode variance above is confirmed to come specifically from
    train/eval-gated stochastic layers (dropout), not from some other
    unaccounted randomness.
  - Enumerate every nn.Dropout layer actually present in each module, to
    show concretely where the stochasticity lives, rather than inferring
    it from documentation.
  - Count contrastive_loss_fn's parameters directly, to confirm (not just
    assume) that Stage 4 truly has zero learnable state to have been left
    out of the checkpoint.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from train import (
    load_matched_patients,
    build_batch,
    build_modules,
    forward_pipeline,
    EXPECTED_PATIENT_IDS,
    BEST_MODEL_PATH,
    CLINICAL_PATIENT_PATH,
)
from scripts.survival_loss import SurvivalLoss, load_os_labels

N_REPEATS = 5


def list_dropout_layers(module: nn.Module, module_name: str):
    found = []
    for name, sub in module.named_modules():
        if isinstance(sub, nn.Dropout):
            found.append((f"{module_name}.{name}" if name else module_name, sub.p))
    return found


def main():
    print("=" * 60)
    print("REPRODUCIBILITY DIAGNOSTIC — L3 discrepancy investigation")
    print("=" * 60)

    # ------------------------------------------------------------
    # 0. Load data + best checkpoint (read-only — no training happens here)
    # ------------------------------------------------------------
    patient_ids, gene_arrays, wsi_arrays = load_matched_patients()
    assert patient_ids == EXPECTED_PATIENT_IDS

    gene_input, wsi_input, patch_mask = build_batch(gene_arrays, wsi_arrays)
    labels = load_os_labels(CLINICAL_PATIENT_PATH, patient_ids)
    T, S = labels.futime, labels.fustat

    modules, trainable_modules = build_modules()
    gene_branch, wsi_branch, wsi_reduction, fusion, risk_head = trainable_modules
    contrastive_loss_fn = modules[-1]

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"No checkpoint found at {BEST_MODEL_PATH}")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu")
    gene_branch.load_state_dict(checkpoint["GeneBranch"])
    wsi_branch.load_state_dict(checkpoint["WSIBranch"])
    wsi_reduction.load_state_dict(checkpoint["WSIReduction"])
    fusion.load_state_dict(checkpoint["Fusion"])
    risk_head.load_state_dict(checkpoint["SurvivalRiskHead"])

    print(f"\nLoaded checkpoint from epoch {checkpoint.get('epoch')} "
          f"(recorded total_loss={checkpoint.get('total_loss')})")

    loss_fn = SurvivalLoss(weight_decay=5e-4, alpha=1.0, beta=0.8)

    # ------------------------------------------------------------
    # Item 2/3: does Stage 4 have any learnable/missing state at all?
    # ------------------------------------------------------------
    n_contrastive_params = sum(1 for _ in contrastive_loss_fn.parameters())
    print(f"\n[Check 2/3] PathwayPatchContrastiveLoss learnable parameters: "
          f"{n_contrastive_params}")
    if n_contrastive_params == 0:
        print("  -> Confirmed zero learnable params. Nothing Stage-4-specific "
              "could have been missing from the checkpoint.")
    else:
        print("  -> WARNING: Stage 4 has learnable params not included in "
              "save_checkpoint()/load in evaluate_best.py. This IS a real bug "
              "if so -- investigate further, do not proceed on the dropout "
              "hypothesis alone.")

    # ------------------------------------------------------------
    # Item 1/4/5: enumerate actual dropout layers present in each module
    # ------------------------------------------------------------
    print("\n[Check 1] Dropout layers found in each module:")
    all_dropout = []
    for name, mod in [
        ("GeneBranch", gene_branch),
        ("WSIBranch", wsi_branch),
        ("WSIReduction", wsi_reduction),
        ("Fusion", fusion),
        ("SurvivalRiskHead", risk_head),
    ]:
        layers = list_dropout_layers(mod, name)
        all_dropout.extend(layers)
        if layers:
            for full_name, p in layers:
                print(f"  {full_name}: Dropout(p={p})")
        else:
            print(f"  {name}: no Dropout layers")

    if not all_dropout:
        print("  -> No dropout layers found anywhere -- train/eval mode is "
              "NOT the explanation; investigate further before concluding.")

    # ------------------------------------------------------------
    # Item 1/5/6: run forward_pipeline() repeatedly in TRAIN mode,
    # with NO weight updates in between -- any variance here can only
    # come from a stochastic op inside the forward pass.
    # ------------------------------------------------------------
    print(f"\n[Check 5/6] Running forward_pipeline() {N_REPEATS}x in TRAIN mode "
          f"(same weights, same input, no optimizer step)...")

    for m in trainable_modules:
        m.train()

    train_mode_results = []
    with torch.no_grad():
        for i in range(N_REPEATS):
            R, L3 = forward_pipeline(modules, gene_input, wsi_input, patch_mask)
            total_loss, diagnostics = loss_fn(R, T, S, L3, trainable_modules)
            train_mode_results.append({
                "L1": diagnostics["L1"].item(),
                "L3": L3.item(),
                "total_loss": total_loss.item(),
            })
            print(f"  run {i+1}: L1={diagnostics['L1'].item():.6f}  "
                  f"L3={L3.item():.6f}  total_loss={total_loss.item():.6f}")

    l3_values = [r["L3"] for r in train_mode_results]
    l1_values = [r["L1"] for r in train_mode_results]
    l3_range = max(l3_values) - min(l3_values)
    l1_range = max(l1_values) - min(l1_values)

    print(f"\n  TRAIN mode L3 range across {N_REPEATS} runs: {l3_range:.6f}")
    print(f"  TRAIN mode L1 range across {N_REPEATS} runs: {l1_range:.6f}")

    # ------------------------------------------------------------
    # Item 1/6: run forward_pipeline() repeatedly in EVAL mode --
    # should be bit-identical every time (dropout disabled).
    # ------------------------------------------------------------
    print(f"\n[Check 1/6] Running forward_pipeline() {N_REPEATS}x in EVAL mode "
          f"(same weights, same input)...")

    for m in trainable_modules:
        m.eval()

    eval_mode_results = []
    with torch.no_grad():
        for i in range(N_REPEATS):
            R, L3 = forward_pipeline(modules, gene_input, wsi_input, patch_mask)
            total_loss, diagnostics = loss_fn(R, T, S, L3, trainable_modules)
            eval_mode_results.append(L3.item())
            print(f"  run {i+1}: L3={L3.item():.6f}")

    eval_l3_range = max(eval_mode_results) - min(eval_mode_results)
    print(f"\n  EVAL mode L3 range across {N_REPEATS} runs: {eval_l3_range:.10f}")

    # ------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    if l3_range > 1e-6 and eval_l3_range < 1e-6 and all_dropout:
        print(
            "CONFIRMED: L3 varies across repeated TRAIN-mode forward passes\n"
            "(identical weights, identical input, no training step between\n"
            "runs) but is bit-identical across repeated EVAL-mode forward\n"
            "passes. The variation is caused by active Dropout layers listed\n"
            "above, which are stochastic in train mode and disabled in eval\n"
            "mode. The training-time L3 recorded in training_history.csv at\n"
            "epoch 467 was ONE random sample of a stochastic quantity, not a\n"
            "deterministic function of the saved weights alone -- so it is\n"
            "not reproducible by reloading the checkpoint and re-running in\n"
            "eval mode. This is expected behavior of dropout, not a bug in\n"
            "checkpoint saving/loading, and not missing Stage 4 state\n"
            "(Stage 4 itself has zero learnable parameters, confirmed above)."
        )
    else:
        print(
            "Result does not clearly match the dropout hypothesis -- do not\n"
            "conclude dropout is the cause. Re-check checkpoint contents and\n"
            "Stage 4 module state manually before proceeding."
        )
    print("=" * 60)


if __name__ == "__main__":
    main()