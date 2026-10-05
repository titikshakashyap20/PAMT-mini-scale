"""
Implements the part of Section III-A your notebook was missing:
building the 186 x 5245 binary pathway-gene matrix E', and the
element-wise multiplication E = E' * e that produces the actual
gene-branch input for each patient.

Paper equations (p.901):
  - normalize: x' = log2(x + 1), then z = (x' - mu) / sigma   [already in your notebook]
  - E' in {0,1}^(186 x 5245): E'[p, g] = 1 if gene g is in pathway p
  - per-sample gene branch input: E = E' * e   (row-wise element-wise multiply,
    broadcasting each sample's 5245-length expression vector across every pathway row)
"""
from collections import defaultdict
import numpy as np
import pandas as pd


def parse_kegg_gmt(gmt_path: str) -> dict:
    """Returns {pathway_name: [gene_symbols]}."""
    pathways = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            pathway_name, genes = parts[0], parts[2:]
            pathways[pathway_name] = genes
    return pathways


def build_binary_pathway_gene_matrix(kegg_pathways: dict, gene_universe: list) -> pd.DataFrame:
    """
    Returns E' as a DataFrame of shape (n_pathways, n_genes), values in {0,1}.
    Rows = pathway names, columns = gene_universe (fixed, ordered gene list).
    """
    gene_index = {g: i for i, g in enumerate(gene_universe)}
    E_prime = np.zeros((len(kegg_pathways), len(gene_universe)), dtype=np.float32)

    pathway_names = list(kegg_pathways.keys())
    for p_idx, pname in enumerate(pathway_names):
        for gene in kegg_pathways[pname]:
            g_idx = gene_index.get(gene)
            if g_idx is not None:
                E_prime[p_idx, g_idx] = 1.0

    return pd.DataFrame(E_prime, index=pathway_names, columns=gene_universe)


def build_gene_universe(kegg_pathways: dict, expr_matrix: pd.DataFrame) -> list:
    """
    5245-gene universe used in the paper: all KEGG genes across the 186 pathways,
    intersected with genes actually present in the RNA-seq matrix.
    """
    all_kegg_genes = set()
    for genes in kegg_pathways.values():
        all_kegg_genes.update(genes)
    present = [g for g in sorted(all_kegg_genes) if g in expr_matrix.columns]
    return present


def build_pathway_expression_for_patient(
    e_prime: pd.DataFrame, expr_scaled_row: pd.Series
) -> np.ndarray:
    """
    e_prime: (186, 5245) binary matrix, columns aligned to gene_universe
    expr_scaled_row: Z-scored expression for ONE patient, indexed by the same gene_universe
    Returns: (186, 5245) pathway-based gene expression matrix for that patient — this
             is one row of the paper's {P1, ..., P186} sequence input.
    """
    e_vec = expr_scaled_row.reindex(e_prime.columns).fillna(0.0).values  # (5245,)
    pathway_expr = e_prime.values * e_vec[None, :]  # broadcast multiply, (186, 5245)
    return pathway_expr.astype(np.float32)
