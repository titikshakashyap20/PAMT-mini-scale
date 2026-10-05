"""
Run from project root:
    python show_gene_output.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATIENT_ID = "TCGA-2F-A9KO"

arr = np.load(f"data/processed/gene/{PATIENT_ID}.npy")
print(f"Shape: {arr.shape}")

df = pd.DataFrame(arr)
df.to_csv("demo_gene_output.csv", index=False)
print("Saved demo_gene_output.csv")

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
ax.set_title(f"Gene-Branch Pathway Matrix — {PATIENT_ID}\n"
             f"({arr.shape[0]} pathways x {arr.shape[1]} genes)")
ax.set_xlabel("Genes (gene universe, ordered)")
ax.set_ylabel("KEGG pathways")
fig.colorbar(im, ax=ax, label="Z-scored expression (0 = gene not in that pathway)")
fig.tight_layout()
fig.savefig("demo_gene_output_heatmap.png", dpi=200)
print("Saved demo_gene_output_heatmap.png")