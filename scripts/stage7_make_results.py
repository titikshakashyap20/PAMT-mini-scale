from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("data/processed/training")
df = pd.read_csv(BASE / "training_history.csv")

# 1. Train vs deterministic eval total loss
plt.figure(figsize=(9, 5))
plt.plot(df["epoch"], df["train_total_loss"], label="Train total loss")
plt.plot(df["epoch"], df["eval_total_loss"], label="Eval total loss")
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.title("PAMT Mini-Scale Training and Evaluation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(BASE / "training_vs_eval_loss.png", dpi=200)
plt.close()

# 2. Eval C-index
plt.figure(figsize=(9, 5))
plt.plot(df["epoch"], df["eval_C_index"])
plt.xlabel("Epoch")
plt.ylabel("C-index")
plt.title("PAMT Mini-Scale Evaluation C-index")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(BASE / "eval_c_index.png", dpi=200)
plt.close()

# 3. Best vs final summary
best = df.loc[df["train_total_loss"].idxmin()]
final = df.iloc[-1]

summary = pd.DataFrame([
    {
        "checkpoint": "Best training-loss checkpoint",
        "epoch": int(best["epoch"]),
        "train_total_loss": best["train_total_loss"],
        "eval_total_loss": best["eval_total_loss"],
        "eval_C_index": best["eval_C_index"],
    },
    {
        "checkpoint": "Final checkpoint",
        "epoch": int(final["epoch"]),
        "train_total_loss": final["train_total_loss"],
        "eval_total_loss": final["eval_total_loss"],
        "eval_C_index": final["eval_C_index"],
    },
])
summary.to_csv(BASE / "stage7_checkpoint_summary.csv", index=False)

# 4. Clean risk-score table
risk = pd.read_csv(BASE / "final_risk_scores.csv")
risk["OS_MONTHS"] = risk["OS_MONTHS"].round(2)
risk["risk_score"] = risk["risk_score"].round(4)
risk.to_csv(BASE / "stage7_risk_score_table.csv", index=False)

print("Stage 7 analysis complete.")
print("\nCheckpoint summary:")
print(summary.to_string(index=False))
print("\nFiles created:")
for name in [
    "training_vs_eval_loss.png",
    "eval_c_index.png",
    "stage7_checkpoint_summary.csv",
    "stage7_risk_score_table.csv",
]:
    print(BASE / name)
