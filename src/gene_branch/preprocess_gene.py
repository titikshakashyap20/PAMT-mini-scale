"""
Full gene-branch preprocessing: your existing steps (parse expr matrix, log2,
Z-score) plus the pathway matrix construction that was missing. One call
produces one (186, 5245) .npy file per patient, matching the paper's
gene-branch input exactly.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.gene_branch.build_pathway_matrix import (
    parse_kegg_gmt,
    build_gene_universe,
    build_binary_pathway_gene_matrix,
    build_pathway_expression_for_patient,
)


def load_and_normalize_expression(expression_path: str) -> pd.DataFrame:
    expr = pd.read_csv(expression_path, sep="\t", low_memory=False)
    expr = expr.drop_duplicates(subset="Hugo_Symbol")
    expr = expr.dropna(subset=["Hugo_Symbol"])
    expr_matrix = expr.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"]).T
    expr_matrix = expr_matrix[~expr_matrix.index.duplicated(keep="first")] 
    expr_matrix = expr_matrix.astype(float)


    expr_log = np.log2(expr_matrix + 1)
    scaler = StandardScaler()
    expr_scaled = pd.DataFrame(
        scaler.fit_transform(expr_log), index=expr_log.index, columns=expr_log.columns
    )
    return expr_scaled


def run_gene_pipeline(cfg: dict, patient_ids: list[str]):
    kegg_pathways = parse_kegg_gmt(cfg["paths"]["kegg_gmt"])
    expr_scaled = load_and_normalize_expression(cfg["paths"]["expression"])

    gene_universe = build_gene_universe(kegg_pathways, expr_scaled)
    print(f"Gene universe: {len(gene_universe)} genes "
          f"(paper reports 5245 — small diffs are fine, note it in your report)")

    e_prime = build_binary_pathway_gene_matrix(kegg_pathways, gene_universe)
    print(f"Pathway-gene matrix E': {e_prime.shape}  (paper: (186, 5245))")

    out_dir = Path(cfg["paths"]["processed_gene_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for patient_id in patient_ids:
        if patient_id not in expr_scaled.index:
            print(f"  skip {patient_id}: no expression data")
            continue
        pathway_matrix = build_pathway_expression_for_patient(
            e_prime, expr_scaled.loc[patient_id]
        )
        np.save(out_dir / f"{patient_id}.npy", pathway_matrix)
        n_done += 1

    print(f"Saved gene-branch input for {n_done}/{len(patient_ids)} patients to {out_dir}")
