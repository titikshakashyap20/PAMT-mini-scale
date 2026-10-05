"""
Run from the project root:
    python scripts/run_gene_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # let `src` imports work

import pandas as pd
from src.utils.config import load_config
from src.gene_branch.preprocess_gene import run_gene_pipeline


def main():
    cfg = load_config()
    clinical = pd.read_csv(cfg["paths"]["clinical"], sep="\t", comment="#", low_memory=False)
    patient_ids = (
        clinical["PATIENT_ID"].dropna().unique()[: cfg["sample"]["n_patients"]].tolist()
    )
    print(f"Running gene-branch preprocessing for {len(patient_ids)} patients")
    run_gene_pipeline(cfg, patient_ids)


if __name__ == "__main__":
    main()
