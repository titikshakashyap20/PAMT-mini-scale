"""
PAMT MINI-SCALE — PIC INFONCE LAMBDA=0 CONTROL

This script is deliberately built on the project's existing train.py API.
It does NOT duplicate patient loading, model construction, or SurvivalLoss.

Protocol:
  - Start from data/processed/model_checkpoints/best_model.pt (epoch 467).
  - 100 epochs.
  - AdamW, lr=1e-4, optimizer weight_decay=0.
  - lambda_PIC = 0.0 (control: PIC is computed but contributes zero to training).
  - tau = 0.0748829767.
  - Fresh random 20% valid-patch removal every epoch.
  - Original best_model.pt is never overwritten.
  - This is the matched lambda=0 control for the completed PIC experiment.

PIC success criteria are not applied to the lambda=0 control.
This run tests whether ordinary fine-tuning alone can explain the PIC gain.

IMPORTANT:
This is a same-15-patient exploratory fine-tune. It does not establish
generalization to unseen patients.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# The project's train.py inserts src/ into sys.path and exposes the exact
# loading/model-building functions used by the verified PAMT implementation.
from train import (
    BEST_MODEL_PATH,
    CLINICAL_PATIENT_PATH,
    build_batch,
    build_modules,
    load_matched_patients,
    forward_pipeline,
)
from scripts.survival_loss import SurvivalLoss, load_os_labels
from c_index import concordance_index


# ---------------------------------------------------------------------
# Fixed pre-registered values
# ---------------------------------------------------------------------
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_LAMBDA_PIC = 0.0
DEFAULT_TAU = 0.0748829767
DEFAULT_REMOVAL = 0.20
DEFAULT_SEED = 42

BASELINE_REID = 0.258351
BASELINE_MAP_JS = 0.0877

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "pic_infonce_control"
CKPT_DIR = PROJECT_ROOT / "data" / "processed" / "pic_infonce_control"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Exact checkpoint loading used by this experiment
# ---------------------------------------------------------------------
def load_verified_checkpoint(trainable_modules, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    required = {
        "GeneBranch": trainable_modules[0],
        "WSIBranch": trainable_modules[1],
        "WSIReduction": trainable_modules[2],
        "Fusion": trainable_modules[3],
        "SurvivalRiskHead": trainable_modules[4],
    }

    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint is missing expected keys: {missing}"
        )

    for key, module in required.items():
        module.load_state_dict(checkpoint[key])

    return checkpoint


# ---------------------------------------------------------------------
# Exact Stage 1-5A forward needed for attention + risk.
#
# train.py's forward_pipeline returns only R and L3. PIC needs the
# attention matrix as well, so this function follows the exact same
# module sequence but retains the fusion attention output.
# ---------------------------------------------------------------------
def forward_with_attention(
    modules,
    gene_input,
    wsi_input,
    patch_mask,
):
    (
        gene_branch,
        wsi_branch,
        wsi_reduction,
        fusion,
        risk_head,
        contrastive_loss_fn,
    ) = modules

    xP = gene_branch(gene_input)

    xI_384 = wsi_branch(
        wsi_input,
        patch_mask,
    )

    xI, mask_out = wsi_reduction(
        xI_384,
        patch_mask,
    )

    CA, attention = fusion(
        xP,
        xI,
        mask_out,
    )

    R = risk_head(
        xP,
        CA,
    )

    L3, _info = contrastive_loss_fn(
        xP,
        xI,
        mask_out,
    )

    return R, L3, attention, mask_out


# ---------------------------------------------------------------------
# Randomly remove 20% of each patient's valid WSI patches.
# Removed positions become padding; tensor shape is unchanged.
# ---------------------------------------------------------------------
def make_perturbed_batch(
    wsi_input,
    patch_mask,
    removal,
    generator,
):
    pert_wsi = wsi_input.clone()
    pert_mask = patch_mask.clone()

    batch_size = pert_wsi.shape[0]

    for b in range(batch_size):
        valid = torch.where(~patch_mask[b])[0]

        if valid.numel() <= 1:
            continue

        n_remove = max(
            1,
            int(round(valid.numel() * removal)),
        )

        perm = torch.randperm(
            valid.numel(),
            generator=generator,
            device=valid.device,
        )

        remove_idx = valid[perm[:n_remove]]

        # Zero the removed patches and mark them as padding.
        pert_wsi[b, remove_idx, :] = 0.0
        pert_mask[b, remove_idx] = True

    return pert_wsi, pert_mask


# ---------------------------------------------------------------------
# Convert (B,H,P,N) attention to pathway distributions (B,P,N).
# ---------------------------------------------------------------------
def pathway_attention_map(
    attention,
    mask,
):
    # Average attention heads.
    pathway_map = attention.mean(dim=1)

    # True = padding, therefore zero padded positions.
    pathway_map = pathway_map.masked_fill(
        mask.unsqueeze(1),
        0.0,
    )

    denom = pathway_map.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)

    return pathway_map / denom


# ---------------------------------------------------------------------
# Pairwise pathway JS distance:
# clean pathway p vs perturbed pathway q.
# Output: (B,P,P)
# ---------------------------------------------------------------------
def pairwise_js(clean_map, perturbed_map):
    p = clean_map.unsqueeze(2).clamp_min(1e-8)
    q = perturbed_map.unsqueeze(1).clamp_min(1e-8)

    m = 0.5 * (p + q)

    return 0.5 * (
        (p * (p.log() - m.log())).sum(dim=-1)
        + (q * (q.log() - m.log())).sum(dim=-1)
    )


# ---------------------------------------------------------------------
# Corrected pathway-identity InfoNCE.
# Positive = same pathway identity.
# Negatives = every other pathway identity.
# ---------------------------------------------------------------------
def pic_infonce_from_maps(
    clean_map,
    perturbed_map,
    tau,
):
    distances = pairwise_js(
        clean_map,
        perturbed_map,
    )

    batch_size, num_pathways, _ = distances.shape

    logits = -distances / tau

    targets = torch.arange(
        num_pathways,
        device=distances.device,
    ).unsqueeze(0).expand(
        batch_size,
        num_pathways,
    )

    loss = F.cross_entropy(
        logits.reshape(batch_size * num_pathways, num_pathways),
        targets.reshape(batch_size * num_pathways),
    )

    return loss, distances


# ---------------------------------------------------------------------
# C-index using the project's exact helper.
# ---------------------------------------------------------------------
def evaluate_survival(
    modules,
    gene_input,
    wsi_input,
    patch_mask,
    T,
    S,
    loss_fn,
):
    for module in modules[:-1]:
        module.eval()

    with torch.no_grad():
        R, L3 = forward_pipeline(
            modules,
            gene_input,
            wsi_input,
            patch_mask,
        )

        total_loss, diagnostics = loss_fn(
            R,
            T,
            S,
            L3,
            modules[:-1],
        )

        c_index = concordance_index(
            R.squeeze(-1),
            T,
            S,
        )

    return {
        "survival_total": float(total_loss.item()),
        "L1": float(diagnostics["L1"].item()),
        "L2_raw": float(diagnostics["L2_raw"].item()),
        "L2_weighted": float(
            diagnostics["L2_weighted"].item()
        ),
        "L3": float(L3.item()),
        "L3_weighted": float(
            diagnostics["L3_weighted"].item()
        ),
        "cindex": float(c_index),
    }


# ---------------------------------------------------------------------
# Fresh PIC evaluation.
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_pic(
    modules,
    gene_input,
    wsi_input,
    patch_mask,
    draws,
    removal,
    tau,
    seed,
):
    for module in modules[:-1]:
        module.eval()

    clean_R, clean_L3, clean_attention, clean_mask = (
        forward_with_attention(
            modules,
            gene_input,
            wsi_input,
            patch_mask,
        )
    )

    clean_map = pathway_attention_map(
        clean_attention,
        clean_mask,
    )

    batch_size, num_pathways, _ = clean_map.shape

    generator = torch.Generator(
        device=wsi_input.device
    )
    generator.manual_seed(seed)

    losses = []
    reid_scores = []
    map_js_scores = []

    for _ in range(draws):
        pert_wsi, pert_mask = make_perturbed_batch(
            wsi_input,
            patch_mask,
            removal,
            generator,
        )

        (
            pert_R,
            pert_L3,
            pert_attention,
            pert_mask_out,
        ) = forward_with_attention(
            modules,
            gene_input,
            pert_wsi,
            pert_mask,
        )

        pert_map = pathway_attention_map(
            pert_attention,
            pert_mask_out,
        )

        pic_loss, distances = pic_infonce_from_maps(
            clean_map,
            pert_map,
            tau,
        )

        nearest = distances.argmin(dim=-1)

        targets = torch.arange(
            num_pathways,
            device=distances.device,
        ).view(
            1,
            num_pathways,
        ).expand(
            batch_size,
            num_pathways,
        )

        reid = (
            nearest == targets
        ).float().mean()

        same_pathway_js = torch.diagonal(
            distances,
            dim1=1,
            dim2=2,
        ).mean()

        losses.append(float(pic_loss.item()))
        reid_scores.append(float(reid.item()))
        map_js_scores.append(
            float(same_pathway_js.item())
        )

    return {
        "pic_infonce": float(np.mean(losses)),
        "reid": float(np.mean(reid_scores)),
        "map_js": float(np.mean(map_js_scores)),
    }


# ---------------------------------------------------------------------
# Save experiment checkpoint WITHOUT touching best_model.pt.
# ---------------------------------------------------------------------
def save_pic_checkpoint(
    path,
    trainable_modules,
    optimizer,
    epoch,
    metrics,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "GeneBranch": trainable_modules[0].state_dict(),
            "WSIBranch": trainable_modules[1].state_dict(),
            "WSIReduction": trainable_modules[2].state_dict(),
            "Fusion": trainable_modules[3].state_dict(),
            "SurvivalRiskHead": trainable_modules[4].state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )
    parser.add_argument(
        "--lambda-pic",
        type=float,
        default=DEFAULT_LAMBDA_PIC,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
    )
    parser.add_argument(
        "--removal",
        type=float,
        default=DEFAULT_REMOVAL,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    print("=" * 78)
    print(
        "PAMT MINI-SCALE — PIC INFONCE LAMBDA=0 CONTROL"
    )
    print("=" * 78)
    print()
    print(f"Epochs:          {args.epochs}")
    print(f"Learning rate:   {args.lr}")
    print(f"lambda_PIC:      {args.lambda_pic}")
    print(f"Removal:         {args.removal:.0%}")
    print(f"InfoNCE tau:     {args.tau}")
    print(f"Seed:            {args.seed}")
    print(f"Start checkpoint: {BEST_MODEL_PATH}")
    print()
    print("Original best_model.pt will NOT be modified.")
    print()

    # -------------------------------------------------------------
    # 1. Exact project data-loading path
    # -------------------------------------------------------------
    patient_ids, gene_arrays, wsi_arrays = (
        load_matched_patients()
    )

    gene_input, wsi_input, patch_mask = build_batch(
        gene_arrays,
        wsi_arrays,
    )

    labels = load_os_labels(
        CLINICAL_PATIENT_PATH,
        patient_ids,
    )

    T = labels.futime
    S = labels.fustat

    print(f"Patients:         {len(patient_ids)}")
    print(
        f"Gene batch shape: {tuple(gene_input.shape)}"
    )
    print(
        f"WSI batch shape:  {tuple(wsi_input.shape)}"
    )
    print(
        f"Mask shape:       {tuple(patch_mask.shape)}"
    )

    # -------------------------------------------------------------
    # 2. Exact project architecture + verified checkpoint
    # -------------------------------------------------------------
    modules, trainable_modules = build_modules()

    checkpoint = load_verified_checkpoint(
        trainable_modules,
        BEST_MODEL_PATH,
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        467,
    )

    print(
        f"Checkpoint epoch: {checkpoint_epoch}"
    )

    loss_fn = SurvivalLoss(
        weight_decay=5e-4,
        alpha=1.0,
        beta=0.8,
    )

    # -------------------------------------------------------------
    # 3. Baseline reproduction
    # -------------------------------------------------------------
    baseline_survival = evaluate_survival(
        modules,
        gene_input,
        wsi_input,
        patch_mask,
        T,
        S,
        loss_fn,
    )

    baseline_pic = evaluate_pic(
        modules,
        gene_input,
        wsi_input,
        patch_mask,
        draws=5,
        removal=args.removal,
        tau=args.tau,
        seed=args.seed,
    )

    print()
    print("BASELINE")
    print(
        f"  Survival total: {baseline_survival['survival_total']:.8f}"
    )
    print(
        f"  L1:              {baseline_survival['L1']:.8f}"
    )
    print(
        f"  L2 raw:          {baseline_survival['L2_raw']:.8f}"
    )
    print(
        f"  L2 weighted:     {baseline_survival['L2_weighted']:.8f}"
    )
    print(
        f"  L3:              {baseline_survival['L3']:.8f}"
    )
    print(
        f"  L3 weighted:     {baseline_survival['L3_weighted']:.8f}"
    )
    print(
        f"  C-index:         {baseline_survival['cindex']:.8f}"
    )
    print(
        f"  PIC InfoNCE:     {baseline_pic['pic_infonce']:.8f}"
    )
    print(
        f"  PIC re-ID:       {baseline_pic['reid']:.8f}"
    )
    print(
        f"  Mean map JS:     {baseline_pic['map_js']:.8f}"
    )

    # -------------------------------------------------------------
    # 4. Optimizer: exactly one pre-registered experiment
    # -------------------------------------------------------------
    optimizer = optim.AdamW(
        [
            parameter
            for module in trainable_modules
            for parameter in module.parameters()
        ],
        lr=args.lr,
        weight_decay=0.0,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0.0,
    )

    history = []

    best_combined = math.inf
    best_combined_epoch = None

    best_survival = math.inf
    best_survival_epoch = None

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CKPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = OUT_DIR / "training_history.csv"

    with open(
        history_path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "learning_rate",
                "survival_total",
                "pic_infonce",
                "combined_total",
            ]
        )

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            for module in trainable_modules:
                module.train()

            optimizer.zero_grad(
                set_to_none=True
            )

            # Fresh random perturbation for this epoch.
            generator = torch.Generator(
                device=wsi_input.device
            )
            generator.manual_seed(
                args.seed + epoch
            )

            pert_wsi, pert_mask = (
                make_perturbed_batch(
                    wsi_input,
                    patch_mask,
                    args.removal,
                    generator,
                )
            )

            # Clean forward: exact PAMT training pipeline.
            (
                R,
                L3,
                clean_attention,
                clean_mask,
            ) = forward_with_attention(
                modules,
                gene_input,
                wsi_input,
                patch_mask,
            )

            # Survival objective: exact project's SurvivalLoss.
            survival_total, survival_diag = (
                loss_fn(
                    R,
                    T,
                    S,
                    L3,
                    trainable_modules,
                )
            )

            # Perturbed forward is only used for PIC.
            (
                pert_R,
                pert_L3,
                pert_attention,
                pert_mask_out,
            ) = forward_with_attention(
                modules,
                gene_input,
                pert_wsi,
                pert_mask,
            )

            clean_map = pathway_attention_map(
                clean_attention,
                clean_mask,
            )

            pert_map = pathway_attention_map(
                pert_attention,
                pert_mask_out,
            )

            pic_loss, _distances = (
                pic_infonce_from_maps(
                    clean_map,
                    pert_map,
                    args.tau,
                )
            )

            combined = (
                survival_total
                + args.lambda_pic * pic_loss
            )

            combined.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            record = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "survival_total": float(
                    survival_total.detach().item()
                ),
                "pic_infonce": float(
                    pic_loss.detach().item()
                ),
                "combined_total": float(
                    combined.detach().item()
                ),
            }

            history.append(record)
            writer.writerow(
                [
                    record["epoch"],
                    record["learning_rate"],
                    record["survival_total"],
                    record["pic_infonce"],
                    record["combined_total"],
                ]
            )
            f.flush()

            if record["combined_total"] < best_combined:
                best_combined = (
                    record["combined_total"]
                )
                best_combined_epoch = epoch

                save_pic_checkpoint(
                    CKPT_DIR
                    / "best_control_combined.pt",
                    trainable_modules,
                    optimizer,
                    epoch,
                    record,
                )

            if record["survival_total"] < best_survival:
                best_survival = (
                    record["survival_total"]
                )
                best_survival_epoch = epoch

                save_pic_checkpoint(
                    CKPT_DIR
                    / "best_control_survival.pt",
                    trainable_modules,
                    optimizer,
                    epoch,
                    record,
                )

            if (
                epoch == 1
                or epoch % 10 == 0
                or epoch == args.epochs
            ):
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"survival {record['survival_total']:.6f} | "
                    f"PIC {record['pic_infonce']:.6f} | "
                    f"combined {record['combined_total']:.6f} | "
                    f"lr {current_lr:.3e}"
                )

    # Save final state.
    save_pic_checkpoint(
        CKPT_DIR / "final_control.pt",
        trainable_modules,
        optimizer,
        args.epochs,
        history[-1],
    )

    # -------------------------------------------------------------
    # 5. Load control + existing PIC checkpoint and evaluate them on
    #    exactly the same fresh perturbation draws.
    # -------------------------------------------------------------
    def load_saved(path):
        fresh_modules, fresh_trainable = build_modules()
        load_verified_checkpoint(fresh_trainable, path)
        return fresh_modules

    control_modules = load_saved(
        CKPT_DIR / "best_control_combined.pt"
    )

    pic_checkpoint_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "pic_infonce_finetune"
        / "best_pic_combined.pt"
    )
    if not pic_checkpoint_path.exists():
        raise FileNotFoundError(
            "Existing PIC checkpoint not found at: "
            f"{pic_checkpoint_path}"
        )
    pic_modules = load_saved(pic_checkpoint_path)

    baseline_modules, baseline_trainable = build_modules()
    load_verified_checkpoint(baseline_trainable, BEST_MODEL_PATH)

    # Same fresh perturbation draws for all three models. This seed is
    # independent of the training seeds (43..142) used above.
    shared_eval_seed = args.seed + 2000

    def shared_eval(modules, name):
        for module in modules[:-1]:
            module.eval()

        survival = evaluate_survival(
            modules, gene_input, wsi_input, patch_mask, T, S, loss_fn
        )

        with torch.no_grad():
            _, _, clean_attention, clean_mask = forward_with_attention(
                modules, gene_input, wsi_input, patch_mask
            )
            clean_map = pathway_attention_map(clean_attention, clean_mask)

            generator = torch.Generator(device=wsi_input.device)
            generator.manual_seed(shared_eval_seed)

            draw_rows = []
            for draw_idx in range(1, 11):
                pert_wsi, pert_mask = make_perturbed_batch(
                    wsi_input, patch_mask, args.removal, generator
                )
                _, _, pert_attention, pert_mask_out = forward_with_attention(
                    modules, gene_input, pert_wsi, pert_mask
                )
                pert_map = pathway_attention_map(
                    pert_attention, pert_mask_out
                )
                pic_loss, distances = pic_infonce_from_maps(
                    clean_map, pert_map, args.tau
                )
                nearest = distances.argmin(dim=-1)
                targets = torch.arange(
                    clean_map.shape[1], device=distances.device
                ).view(1, -1).expand(clean_map.shape[0], -1)
                reid = (nearest == targets).float().mean()
                same_js = torch.diagonal(
                    distances, dim1=1, dim2=2
                ).mean()
                draw_rows.append({
                    "model": name,
                    "draw": draw_idx,
                    "pic_infonce": float(pic_loss.item()),
                    "reid": float(reid.item()),
                    "map_js": float(same_js.item()),
                })

        return survival, draw_rows

    baseline_survival, baseline_draws = shared_eval(
        baseline_modules, "baseline_epoch467"
    )
    control_survival, control_draws = shared_eval(
        control_modules, "control_lambda0"
    )
    pic_survival, pic_draws = shared_eval(
        pic_modules, "pic_lambda_positive"
    )

    all_draws = baseline_draws + control_draws + pic_draws

    def mean_metric(rows, key):
        return float(np.mean([row[key] for row in rows]))

    baseline_reid = mean_metric(baseline_draws, "reid")
    control_reid = mean_metric(control_draws, "reid")
    pic_reid = mean_metric(pic_draws, "reid")

    baseline_js = mean_metric(baseline_draws, "map_js")
    control_js = mean_metric(control_draws, "map_js")
    pic_js = mean_metric(pic_draws, "map_js")

    pic_gain_over_control = pic_reid - control_reid
    control_gain_over_baseline = control_reid - baseline_reid

    # Final control rule from the external review:
    # PASS if PIC - control >= 0.10 and control <= baseline + 0.10.
    control_pass = (
        pic_gain_over_control >= 0.10
        and control_gain_over_baseline <= 0.10
    )
    control_reaches_040 = control_reid >= 0.40

    if control_reaches_040:
        decision = (
            "FAIL — CONTROL ALSO REACHED >= 0.40; "
            "PIC GAIN IS NOT ATTRIBUTABLE TO PIC"
        )
    elif control_pass:
        decision = "PASS — PIC OUTPERFORMS THE LAMBDA=0 CONTROL"
    else:
        decision = "FAIL — PIC DOES NOT BEAT THE LAMBDA=0 CONTROL BY >= 0.10"

    # -------------------------------------------------------------
    # 6. Save paired-control outputs.
    # -------------------------------------------------------------
    survival_rows = [
        {"model": "baseline_epoch467", **baseline_survival},
        {"model": "control_lambda0", **control_survival},
        {"model": "pic_lambda_positive", **pic_survival},
    ]

    survival_csv = OUT_DIR / "shared_survival_results.csv"
    with open(survival_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=survival_rows[0].keys())
        writer.writeheader()
        writer.writerows(survival_rows)

    pic_csv = OUT_DIR / "shared_pic_eval_draws.csv"
    with open(pic_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_draws[0].keys())
        writer.writeheader()
        writer.writerows(all_draws)

    comparison_csv = OUT_DIR / "control_vs_pic_comparison.csv"
    comparison_rows = [
        {
            "model": "baseline_epoch467",
            "reid_mean": baseline_reid,
            "map_js_mean": baseline_js,
            "survival_total": baseline_survival["survival_total"],
            "cindex": baseline_survival["cindex"],
        },
        {
            "model": "control_lambda0",
            "reid_mean": control_reid,
            "map_js_mean": control_js,
            "survival_total": control_survival["survival_total"],
            "cindex": control_survival["cindex"],
        },
        {
            "model": "pic_lambda_positive",
            "reid_mean": pic_reid,
            "map_js_mean": pic_js,
            "survival_total": pic_survival["survival_total"],
            "cindex": pic_survival["cindex"],
        },
    ]
    with open(comparison_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_txt = OUT_DIR / "final_summary.txt"
    summary_txt.write_text(
        "\n".join([
            "PIC INFONCE LAMBDA=0 CONTROL — FINAL SUMMARY",
            "=" * 58,
            "",
            "CONTROL PROTOCOL",
            "----------------",
            f"Start checkpoint: {BEST_MODEL_PATH}",
            f"Start epoch: {checkpoint_epoch}",
            f"Epochs: {args.epochs}",
            f"Learning rate: {args.lr}",
            "Lambda PIC: 0.0",
            f"Removal: {args.removal}",
            f"Tau: {args.tau}",
            f"Training seed: {args.seed}",
            f"Shared evaluation seed: {shared_eval_seed}",
            "Original best_model.pt was never modified.",
            "Existing PIC checkpoint was never modified.",
            "",
            "SHARED FRESH EVALUATION",
            "-----------------------",
            "Baseline, lambda=0 control, and PIC were evaluated on the",
            "same 10 fresh perturbation draws.",
            "",
            f"Baseline re-ID: {baseline_reid:.8f}",
            f"Control re-ID: {control_reid:.8f}",
            f"PIC re-ID: {pic_reid:.8f}",
            "",
            f"PIC - Control re-ID gain: {pic_gain_over_control:.8f}",
            f"Control - Baseline re-ID gain: {control_gain_over_baseline:.8f}",
            "",
            "CONTROL DECISION RULE",
            "---------------------",
            "PASS if PIC - Control >= 0.10 AND Control <= Baseline + 0.10.",
            f"PIC beats control by >= 0.10: {pic_gain_over_control >= 0.10}",
            f"Control <= baseline + 0.10: {control_gain_over_baseline <= 0.10}",
            f"Control re-ID >= 0.40: {control_reaches_040}",
            "",
            f"DECISION: {decision}",
            "",
            "This remains a same-15-patient exploratory experiment.",
            "It does not establish generalization to unseen patients.",
        ]) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PIC INFONCE LAMBDA=0 CONTROL — FINAL RESULT")
    print("=" * 78)
    print(f"Baseline re-ID:       {baseline_reid:.8f}")
    print(f"Control re-ID:        {control_reid:.8f}")
    print(f"PIC re-ID:            {pic_reid:.8f}")
    print()
    print(f"PIC - Control gain:   {pic_gain_over_control:.8f}")
    print(f"Control - Baseline:   {control_gain_over_baseline:.8f}")
    print()
    print(f"Control >= 0.40:      {control_reaches_040}")
    print(f"Decision:             {decision}")
    print()
    print(f"History:              {history_path}")
    print(f"Shared survival CSV:  {survival_csv}")
    print(f"Shared PIC CSV:       {pic_csv}")
    print(f"Comparison CSV:       {comparison_csv}")
    print(f"Summary:              {summary_txt}")
    print(f"Checkpoints:          {CKPT_DIR}")
    print("Original best_model.pt was NOT modified.")
    print("Existing PIC checkpoint was NOT modified.")
    print("=" * 78)



if __name__ == "__main__":
    main()
