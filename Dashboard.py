"""
PAMT Mini-Scale Implementation Dashboard
Run from the project root:
    streamlit run dashboard.py

The dashboard is intentionally file-driven: when the project source files exist,
it reads the real source code and numbers the lines automatically, so the
presentation can point to exact code lines without inventing line numbers.
"""

from pathlib import Path
import re
import textwrap

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

def p(*parts):
    return ROOT.joinpath(*parts)

PATHS = {
    "gene_runner": p("scripts", "run_gene_pipeline.py"),
    "gene_preprocess": p("src", "gene_branch", "preprocess_gene.py"),
    "build_pathway_matrix": p("src", "gene_branch", "build_pathway_matrix.py"),
    "test_gene_branch": p("tests", "test_gene_branch.py"),
    "pathway_embedding": p("src", "gene_branch", "pathway_embedding.py"),
    "transformer_encoder": p("src", "gene_branch", "transformer_encoder.py"),
    "gene_branch": p("src", "gene_branch", "gene_branch.py"),
    "dino_runner": p("scripts", "run_dino_training.py"),
    "wsi_loader": p("src", "wsi_branch", "loader.py"),
    "patch_extraction": p("src", "wsi_branch", "patch_extraction.py"),
    "dino_pretraining": p("src", "wsi_branch", "dino_pretraining.py"),
    "dino_embedding": p("src", "wsi_branch", "dino_embedding.py"),
    "patch_clustering": p("src", "wsi_branch", "patch_clustering.py"),
    "test_wsi_branch_batch": None,
    "test_wsi_branch": p("tests", "test_wsi_branch.py"),
    "patch_embedding": p("src", "wsi_branch", "patch_embedding.py"),
    "wsi_transformer_encoder": None,
    "test_wsi_reduction": p("tests", "test_wsi_reduction.py"),
    "wsi_branch": p("src", "wsi_branch", "wsi_branch.py"),
    "wsi_reduction": p("src", "wsi_branch", "wsi_reduction.py"),
    "contrastive": p("src", "gene_branch", "contrastive_loss.py"),
    "fusion": p("src", "fusion.py"),
    "risk_head": p("scripts", "risk_head.py"),
    "survival_loss": p("scripts", "survival_loss.py"),
    "train": p("scripts", "train.py"),
    "evaluate": p("scripts", "evaluate_best.py"),
    "results": p("scripts", "stage7_make_results.py"),

    "pic_diagnostic": p("outputs", "pic_diagnostic", "pic_summary.csv"),
    "pic_null_summary": p("outputs", "pic_diagnostic", "pic_null_summary.csv"),
    "pic_reidentification": p("outputs", "pic_diagnostic", "patient_reidentification.csv"),
    "pic_finetune_summary": p("outputs", "pic_infonce_finetune", "final_summary.txt"),
    "pic_finetune_history": p("outputs", "pic_infonce_finetune", "training_history.csv"),
    "pic_baseline_vs_pic": p("outputs", "pic_infonce_finetune", "baseline_vs_pic.csv"),
    "pic_control_summary": p("outputs", "pic_infonce_control", "final_summary.txt"),
    "pic_control_comparison": p("outputs", "pic_infonce_control", "control_vs_pic_comparison.csv"),
    "pic_control_history": p("outputs", "pic_infonce_control", "training_history.csv"),
    "pic_faithfulness_summary": p("outputs", "pic_faithfulness_diagnostic", "overall_summary.csv"),
    "pic_faithfulness_patients": p("outputs", "pic_faithfulness_diagnostic", "patient_summary.csv"),
    "pic_faithfulness_pathways": p("outputs", "pic_faithfulness_diagnostic", "pathway_faithfulness.csv"),
    "pic_train": p("scripts", "train_pic_infonce_finetune.py"),
    "pic_control_train": p("scripts", "train_pic_infonce_control.py"),

}

TRAIN_DIR = p("data", "processed", "training")
MODEL_DIR = p("data", "processed", "model_checkpoints")

# Hosted/demo mode: raw TCGA/WSI data and large checkpoints are intentionally
# excluded from the public repository. The dashboard therefore falls back to
# lightweight saved results and source-code evidence when those local artifacts
# are unavailable (e.g. on Streamlit Community Cloud).
DEMO_MODE = not (TRAIN_DIR.exists() and MODEL_DIR.exists())




# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="PAMT Mini-Scale Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.small-muted {color:#777; font-size:0.88rem;}
.stage-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.flow {
    font-family: monospace;
    font-size: 0.92rem;
    line-height: 1.55;
}
.badge {
    display:inline-block;
    padding:3px 8px;
    border-radius:999px;
    border:1px solid rgba(128,128,128,.35);
    font-size:.78rem;
    margin-right:4px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------
def read_source(path):
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="replace")


def numbered_source(path):
    src = read_source(path)
    if src is None:
        return None
    lines = src.splitlines()
    return "\n".join(f"{i:4d} | {line}" for i, line in enumerate(lines, 1))


def source_lines(path, start, end):
    src = read_source(path)
    if src is None:
        return None
    lines = src.splitlines()
    start = max(1, start)
    end = min(len(lines), end)
    return "\n".join(
        f"{i:4d} | {lines[i-1]}" for i in range(start, end + 1)
    )


def find_block(path, patterns, context_before=3, context_after=10):
    """Find the first matching line and return a compact, numbered excerpt."""
    src = read_source(path)
    if src is None:
        return None, None
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if any(re.search(pattern, line) for pattern in patterns):
            start = max(0, i - context_before)
            end = min(len(lines), i + context_after + 1)
            excerpt = "\n".join(
                f"{j+1:4d} | {lines[j]}" for j in range(start, end)
            )
            return excerpt, (start + 1, end)
    return None, None


def show_code(path, patterns=None, title="Exact source code"):
    if path is None:
        return
    if not path.exists():
        st.warning(
            f"Source file not found at `{path.relative_to(ROOT)}`. "
            "The dashboard is ready, but this project file must be present "
            "when you run the dashboard to display its exact source lines."
        )
        return

    if patterns:
        excerpt, rng = find_block(path, patterns)
        if excerpt:
            st.caption(
                f"{path.relative_to(ROOT)} — lines {rng[0]}–{rng[1]} "
                "(automatically numbered from the real file)"
            )
            st.code(excerpt, language="python")
            with st.expander("Show complete file"):
                st.code(numbered_source(path), language="python")
            return

    st.caption(f"{path.relative_to(ROOT)} — complete file, automatically numbered")
    st.code(numbered_source(path), language="python")


def metric_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        # Some dashboard rows also carry a third field describing the source.
        # st.metric only needs the label/value pair, so safely ignore extra fields.
        label, value = item[0], item[1]
        col.metric(label, value)


def file_status(path):
    if path is None:
        return "not included in public demo"
    return "✓ present" if path.exists() else "not included in public demo"


def local_data_notice(label="This local-only artifact"):
    st.info(
        f"{label} is intentionally excluded from the public GitHub repository. "
        "The hosted dashboard uses the lightweight saved results, figures, and source code that are committed to the repository. "
        "The full TCGA/WSI dataset and model checkpoints remain available only in the local research environment."
    )


# ---------------------------------------------------------------------
# Known project facts / verified results
# ---------------------------------------------------------------------
PATIENTS = [
    "TCGA-2F-A9KO", "TCGA-2F-A9KP", "TCGA-2F-A9KQ", "TCGA-2F-A9KR",
    "TCGA-2F-A9KT", "TCGA-2F-A9KW", "TCGA-4Z-AA7M", "TCGA-4Z-AA7N",
    "TCGA-4Z-AA7O", "TCGA-4Z-AA7Q", "TCGA-4Z-AA7R", "TCGA-4Z-AA7S",
    "TCGA-4Z-AA7W", "TCGA-4Z-AA7Y", "TCGA-4Z-AA80",
]
PATCH_COUNTS = [144,148,109,126,137,139,94,147,142,100,118,162,123,159,109]

STAGES = {
    "Overview": None,
    "1 — Gene preprocessing + GeneBranch": "gene_complete",
    "3 — WSI / DINO pipeline + WSIBranch + reduction": "wsi_complete",
    "4 — Contrastive L3": "contrastive",
    "5A — Pathway→patch fusion": "fusion",
    "5B — Survival risk head": "risk_head",
    "6 — SurvivalLoss": "survival_loss",
    "7 — Training": "train",
    "7 — Best checkpoint": "evaluate",
    "7 — Results": "results",
    "8 - Pathway-Identity Consistency (PIC)": "pic",
    "📸 Visual Gallery": "visual_gallery",
}




# ---------------------------------------------------------------------
# Extra visualizations: generated from the real project artifacts
# ---------------------------------------------------------------------
def render_architecture_picture():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    boxes = [
        (0.3, 5.8, 2.1, 1.0, "GENE\n186 × 4942"),
        (3.0, 5.8, 2.1, 1.0, "GeneBranch\n→ 186 × 256"),
        (0.3, 2.9, 2.1, 1.0, "WSI / DINO\nN × 384"),
        (3.0, 2.9, 2.1, 1.0, "WSIBranch\n→ N × 384"),
        (5.8, 2.9, 2.1, 1.0, "Reduction\n384 → 256"),
        (8.6, 4.35, 2.1, 1.0, "Fusion\nCA: 186 × 256"),
        (11.4, 4.35, 2.1, 1.0, "Risk Head\nR: 1 score"),
        (8.6, 1.2, 2.1, 1.0, "Contrastive\nL3"),
        (11.4, 1.2, 2.1, 1.0, "Survival Loss\nL1 + L2 + L3"),
    ]
    for x,y,w,h,t in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,linewidth=2))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=11,fontweight="bold")
    arrows=[
        ((2.4,6.3),(3.0,6.3)), ((2.4,3.4),(3.0,3.4)),
        ((5.1,3.4),(5.8,3.4)), ((7.9,3.4),(8.6,4.75)),
        ((5.1,6.3),(8.6,4.85)), ((10.7,4.85),(11.4,4.85)),
        ((7.9,3.0),(8.6,1.7)), ((10.7,1.7),(11.4,1.7)),
        ((12.45,4.35),(12.45,2.2)),
    ]
    for (x1,y1),(x2,y2) in arrows:
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",linewidth=1.7))
    ax.text(7,7.45,"PAMT MINI-SCALE — COMPLETE DATA FLOW",ha="center",fontsize=17,fontweight="bold")
    ax.text(7,0.35,"Gene + WSI → representations → fusion → patient risk → survival loss",ha="center",fontsize=11)
    st.pyplot(fig,clear_figure=True)


def render_loss_components():
    path = TRAIN_DIR / "training_history.csv"
    if not path.exists():
        st.info("training_history.csv not found yet.")
        return
    df = pd.read_csv(path)
    cols=[c for c in ["eval_L1","eval_L2_weighted","eval_L3_weighted"] if c in df.columns]
    if not cols: return
    fig, ax = plt.subplots(figsize=(10,5))
    for c in cols: ax.plot(df["epoch"],df[c],label=c.replace("eval_",""))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss contribution"); ax.set_title("What makes up the evaluation loss?"); ax.legend(); fig.tight_layout(); st.pyplot(fig,clear_figure=True)


def render_visual_gallery():
    """Stage-linked visual evidence only; no generic file discovery."""
    st.header("📸 Visual Gallery — Stage Evidence")
    st.write(
        "Each image below is tied to a specific implementation stage or result. "
        "Only saved project artifacts with a clear interpretation are shown."
    )

    gallery = [
        (
            "Stage 1 — Gene preprocessing",
            p("outputs", "visuals", "demo_gene_output_heatmap.png"),
            "Example processed gene/pathway representation produced by the gene preprocessing pipeline."
        ),
        (
            "Stage 2 — GeneBranch",
            p("outputs", "visuals", "demo_gene_output_heatmap.png"),
            "The processed gene/pathway matrix entering the GeneBranch; useful as a visual representation of the pathway-level input."
        ),
        (
            "Stage 3 — WSI input / WSIBranch",
            p("outputs", "visuals", "TCGA-2F-A9KO_model_input_panel.png"),
            "Patient-level model-input visualization showing the WSI side of the multimodal pipeline."
        ),
        (
            "Stage 4 — Pathway–patch contrastive alignment",
            p("stage4_5b_figures", "stage4_similarity_heatmap.png"),
            "Similarity structure used to inspect pathway–patch relationships in the contrastive stage."
        ),
        (
            "Stage 4 — Top-2 patch selections",
            p("stage4_5b_figures", "stage4_top2_selections.png"),
            "Top-2 patch selections produced by the pathway–patch contrastive analysis."
        ),
        (
            "Stage 5A — Cross-modal fusion",
            p("stage4_5b_figures", "stage5a_cross_attention_heatmap.png"),
            "Pathway-to-patch cross-attention produced by the Stage 5A fusion module."
        ),
        (
            "Stage 5B — Risk head",
            p("stage4_5b_figures", "stage5b_risk_outputs.png"),
            "Risk outputs from the Stage 5B survival-risk head."
        ),
        (
            "Stage 7 — Training / survival objective",
            None,
            "The full training/checkpoint artifacts are local-only; the dashboard presents the verified Stage 7 metrics and training history description in the dedicated Stage 7 sections."
        ),
        (
            "Stage 7 — Patient risk ranking",
            None,
            "The trained model checkpoint and patient risk-score artifacts are local-only; the verified Stage 7 results are presented in the dedicated results section."
        ),
    ]

    for title, image_path, caption in gallery:
        st.subheader(title)
        if image_path is None:
            st.info(caption)
            continue
        if image_path.exists():
            st.image(str(image_path), caption=caption, width="stretch")
            st.caption(f"Source artifact: `{image_path.relative_to(ROOT)}`")
        else:
            st.warning(
                f"Visual artifact not found: `{image_path.relative_to(ROOT)}`. "
                "The corresponding stage remains available in its dedicated dashboard section."
            )


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("🧬 PAMT Dashboard")
st.sidebar.caption("15-patient mini-scale implementation")

stage_name = st.sidebar.radio("Navigate by stage", list(STAGES.keys()))

st.sidebar.divider()
st.sidebar.caption("Hosted demo uses committed source code, figures, and lightweight results.")


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Pathway-Aware Multi-modal Transformer")
st.subheader("PAMT — Complete Mini-Scale Implementation Trace")

st.markdown(
    "This dashboard traces the implementation **file → function → input → "
    "processing → output → saved artifact → next consumer**, with tensor "
    "shapes, short theory, source code and Stage 7 results."
)

if DEMO_MODE:
    st.info(
        "**Hosted demo mode:** full TCGA/WSI data and model checkpoints are intentionally not included in the public repository. "
        "This online dashboard therefore focuses on the committed source code, lightweight result tables, figures, and verified Stage 7–8 findings. "
        "Run the project locally to inspect the full data-backed pipeline."
    )

metric_row([
    ("Patients", "15"),
    ("Gene matrix", "(15, 186, 4942)"),
    ("WSI batch", "(15, 162, 384)"),
    ("Fusion output", "(15, 186, 256)"),
    ("Risk output", "(15, 1)"),
])


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------

def render_theory_summary():
    """Compact theory + exact input/output map for presentation use."""
    st.header("Theory at a glance — inputs → processing → outputs")

    st.code("""
GENE SIDE
15 patient files
each (186,4942)
      ↓
GeneBranch
      ↓
xP = (15,186,256)
186 = pathway tokens
256 = learned feature size

WSI SIDE
each patient: N patches, N = 94–162
each patch = 384-D DINO vector
      ↓
pad to N=162 + mask
      ↓
WSIBranch
      ↓
xI_384 = (15,162,384)
      ↓
WSIReduction
      ↓
xI = (15,162,256)

MULTIMODAL
xP + xI
   ├── Stage 4: contrastive L3
   │      pathway ↔ patch alignment
   │
   └── Stage 5A: cross-attention fusion
          ↓
        CA = (15,186,256)
          ↓
      Stage 5B RiskHead
          ↓
        R = (15,1)
          ↓
      Stage 6 SurvivalLoss
          ↓
        total loss
    """, language="text")

    st.markdown("### The most important shape idea")
    st.write(
        "**The first dimension is the patient batch (15).** "
        "The second dimension is the token sequence: 186 pathways on the gene side, "
        "and up to 162 WSI patches on the image side. The last dimension is the learned "
        "feature size. Gene features and reduced WSI features are both 256-D so they can interact."
    )

    st.markdown("### What each major output means")
    metric_row([
        ("Gene input", "(15,186,4942)"),
        ("xP", "(15,186,256)"),
        ("WSI input", "(15,162,384)"),
        ("xI", "(15,162,256)"),
        ("CA", "(15,186,256)"),
        ("R", "(15,1)"),
        ("L3", "scalar"),
    ])
    st.caption(
        "**Gene input:** 15 patients × 186 pathways × 4,942 gene values · "
        "**xP:** learned pathway representation · "
        "**WSI input:** padded DINO patch features · "
        "**xI:** learned WSI patch representation · "
        "**CA:** fused pathway representation · "
        "**R:** one risk score per patient · "
        "**L3:** pathway-to-patch alignment loss"
    )

    with st.expander("⭐ Stage 4 — contrastive L3 in very simple words", expanded=True):
        st.write(
            "Think of each of the **186 pathways asking: 'Which WSI patches are most relevant to me?'** "
            "The loss compares every pathway token with every real patch token."
        )
        st.code("""
For one patient:

186 pathway vectors
        ×
N real WSI patch vectors
        ↓
similarity for every pathway–patch pair
        ↓
S: (186,N)

For each pathway:
    choose the top 2 most similar real patches
        ↓
    mark those 2 as positive (Y_h = 1)
    mark the others as negative (Y_h = 0)
        ↓
    softmax over real patches
        ↓
    contrastive loss for that pathway

Average over pathways and patients
        ↓
L3 = one scalar
        """, language="text")

        st.markdown("**Exact mini-project shapes:**")
        st.code("""
xP = (15,186,256)
xI = (15,162,256)
mask = (15,162)      True = padding

raw similarity:
S = xP @ xIᵀ
S = (15,186,162)

top_h = 2
topk_idx = (15,186,2)

binary positive labels:
Y_h = (15,186,162)
exactly 2 positives per pathway

masked softmax:
S' = (15,186,162)

final:
L3 = scalar
        """, language="text")

        st.markdown("### Why is this called contrastive?")
        st.write(
            "The model is pushed to give more probability to the selected positive patches "
            "than to the other patches for the same pathway. So it learns a useful **pathway ↔ WSI "
            "alignment** without needing a manually supplied pathway-to-patch label."
        )

        st.markdown("### What creates the labels?")
        st.write(
            "There is **no separate human label file for Y_h**. The two positives are created inside "
            "the loss itself by taking the top-2 highest raw similarities for each pathway."
        )

        st.markdown("### What happens to padding?")
        st.write(
            "Patients have different numbers of patches (94–162), so the batch is padded to 162. "
            "Padded positions are excluded from top-k selection and softmax. Their `S'` and `Y_h` "
            "values are zero. This is the mini-project adaptation needed for variable-length WSI sequences."
        )

        st.markdown("### One crucial implementation note")
        st.warning(
            "The paper describes a learnable temperature τ, but the released official code used here "
            "has **no learnable temperature**. Therefore `PathwayPatchContrastiveLoss` has **0 learnable parameters**."
        )

    st.markdown("### Who creates what? — the handoffs")
    st.code("""
test_gene_branch.py
    creates GeneBranch input batch
    (15,186,4942)
        ↓
GeneBranch
    creates xP
    (15,186,256)
        ↓
Stage 4 loss consumes xP

test_wsi_branch.py
    loads real .npz embeddings + builds padded batch
    (15,162,384) + mask
        ↓
WSIBranch
    creates xI_384
    (15,162,384)
        ↓
WSIReduction
    creates xI
    (15,162,256)
        ↓
Stage 4 loss consumes xI

Stage 4
    creates L3 + diagnostics
        ↓
Stage 6 SurvivalLoss consumes L3

Stage 5A
    consumes xP + xI
    creates CA
        ↓
Stage 5B
    consumes xP + CA
    creates R
        ↓
Stage 6 SurvivalLoss consumes R + survival labels
    """, language="text")


def render_overview():
    st.header("End-to-end pipeline")

    st.code("""
RAW GENE EXPRESSION
        │
        ▼
preprocess_gene.py
        │
        └── patient .npy: (186, 4942)
                    │
                    ▼
              GeneBranch
                    │
                    └── xP: (15,186,256)
                                      │
RAW WSI DINO PATCHES                     │
        │                                │
        ▼                                │
   WSIBranch                             │
        │                                │
        └── xI_384: (15,162,384)         │
                    │                    │
                    ▼                    │
              WSIReduction               │
                    │                    │
                    └── xI: (15,162,256)
                           │              │
                           ├──────────────┤
                           ▼
                 Stage 5A Fusion
                           │
                           └── CA: (15,186,256)
                                      │
                         ┌────────────┘
                         ▼
                 Stage 5B RiskHead
                         │
                         └── R: (15,1)
                                  │
                 Stage 4 L3 ──────┤
                                  ▼
                         Stage 6 SurvivalLoss
                                  │
                                  ▼
                            Stage 7 Training
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
             best_model.pt                final_model.pt
                   │
                   ▼
           evaluate_best.py
                   │
                   ▼
          best_risk_scores.csv
    """, language="text")

    st.info(
        "Presentation framing: this is a **mini-scale implementation**, "
        "not a full reproduction of the paper's experimental setup."
    )

    render_theory_summary()

    st.header("What is actually different from the paper?")
    differences = pd.DataFrame([
        ["Cohort", "15 real BLCA/TCGA patients", "Paper uses a much larger cohort"],
        ["Gene universe", "4942 genes", "Paper reports 5245 genes"],
        ["WSI patches", "94–162 real patches/patient, padded to 162", "Paper uses larger-scale WSI processing/resampling"],
        ["Training split", "All 15 patients in one batch; no held-out set", "No generalization claim"],
        ["C-index", "Same 15 training patients; exploratory", "Not a validation result"],
        ["Stage 4 temperature", "No learnable τ; official-code-faithful", "Paper Eq. 13 describes learnable τ"],
        ["Stage 5B risk head", "Official-code-faithful pooling + Linear", "Paper Eq. 9 describes MLP(LN(xPI))"],
        ["Stage 6 L2", "p=2 over weight-named parameters", "Chosen paper-literal regularization"],
    ], columns=["Aspect", "This project", "Reference / interpretation"])
    st.dataframe(differences, width="stretch", hide_index=True)

    st.header("Patient / WSI batch")
    metric_row([
        ("Patients", "15"),
        ("Minimum patches", "94"),
        ("Maximum patches", "162"),
        ("Patch feature dim", "384"),
    ])
    st.write("Patch counts:", PATCH_COUNTS)

    st.header("Dashboard presentation pattern")
    st.code("""
1. THEORY       What does this stage mean?
2. INPUT        What enters? Exact file/tensor + shape.
3. CODE         Which file/function creates it?
4. PROCESS      What mathematical/NN operation happens?
5. OUTPUT       Exact resulting tensor/result.
6. SAVED TO     Where is it written, if applicable?
7. NEXT         Which file/function consumes it?
8. NOTES        Paper vs official code vs mini adaptation.
    """, language="text")


# ---------------------------------------------------------------------
# Stage renderers
# ---------------------------------------------------------------------

def render_gene_pipeline_complete():

    st.header("Stage 1 — Gene preprocessing → GeneBranch")
    st.write(
        "This is the complete gene-side execution flow: raw RNA-seq is prepared, "
        "the 15 patient .npy files are created, test_gene_branch.py stacks them, "
        "GeneBranch creates pathway tokens and applies the Transformer, and the "
        "test script verifies the result."
    )
    st.info("preprocess_gene.py → build_pathway_matrix.py → 15 .npy files → test_gene_branch.py → GeneBranch → xP → verification")

    st.markdown("### Complete gene-side flow")
    st.code(
        "RAW RNA-seq expression\n"
        "      ↓\n"
        "preprocess_gene.py\n"
        "      ↓ calls\n"
        "build_pathway_matrix.py\n"
        "      ↓\n"
        "15 patient .npy files\n"
        "each: (186,4942)\n"
        "      ↓\n"
        "test_gene_branch.py\n"
        "      ↓ load + np.stack\n"
        "batch: (15,186,4942)\n"
        "      ↓\n"
        "gene_branch.py\n"
        "      ↓\n"
        "PathwayEmbedding: 4942 → 1000 → 256\n"
        "      ↓\n"
        "positional embedding\n"
        "      ↓\n"
        "TransformerEncoder × 2\n"
        "      ↓\n"
        "xP: (15,186,256)\n"
        "      ↓\n"
        "test_gene_branch.py checks shape + finite values + parameters + gradients",
        language="text",
    )

    with st.expander("1 — preprocess_gene.py: clean and normalize RNA-seq", expanded=True):
        st.markdown("**File:** `src/gene_branch/preprocess_gene.py`")
        st.markdown("### Actual code")
        st.code(
            '''def load_and_normalize_expression(expression_path):
    expr = pd.read_csv(expression_path, sep="\t", comment="#", low_memory=False)
    expr = expr.drop_duplicates(subset="Hugo_Symbol")
    expr = expr.dropna(subset=["Hugo_Symbol"])
    expr = expr.set_index("Hugo_Symbol")
    expr = expr.drop(columns=["Entrez_Gene_Id"], errors="ignore")
    expr = expr.T
    expr = expr[~expr.index.duplicated(keep="first")]
    expr = expr.astype(float)
    expr = np.log2(expr + 1)
    expr_scaled = StandardScaler().fit_transform(expr)
    return pd.DataFrame(expr_scaled, index=expr.index, columns=expr.columns)''',
            language="python",
        )
        st.write("Cleans the expression matrix, changes it to patient × gene orientation, applies log2(x+1), then Z-score standardization.")
        st.markdown("### Actual handoff to pathway construction")
        st.code(
            '''e_prime = build_pathway_expression_for_patient(
    e_prime,
    expr_scaled.loc[patient_id]
)

np.save(
    out_dir / f"{patient_id}.npy",
    e_prime.values.astype(np.float32)
)''',
            language="python",
        )

    with st.expander("2 — build_pathway_matrix.py: construct the pathway representation", expanded=True):
        st.markdown("**File:** `src/gene_branch/build_pathway_matrix.py`")
        st.code(
            "KEGG GMT → parse_kegg_gmt() → build_gene_universe() → "
            "build_binary_pathway_gene_matrix() → E′ → "
            "build_pathway_expression_for_patient() → patient matrix",
            language="text",
        )
        st.write("This creates the pathway-by-gene representation used to make each patient's input. Your verified mini run has 186 pathways and 4,942 genes.")
        st.code("E′ / patient output: (186,4942)\nSaved: data/processed/gene/<PATIENT_ID>.npy", language="text")
        if PATHS["build_pathway_matrix"].exists():
            st.markdown("### Complete source file")
            st.code(numbered_source(PATHS["build_pathway_matrix"]), language="python")
        else:
            st.warning("The dashboard project folder does not currently contain build_pathway_matrix.py, so its full source cannot be rendered here. The flow above reflects the verified function calls.")

    with st.expander("3 — test_gene_branch.py: load the 15 files and stack them", expanded=True):
        st.markdown("**File:** `test_gene_branch.py`")
        st.markdown("### Actual code")
        st.code(
            '''def load_real_batch(batch_size=BATCH_SIZE):
    files = sorted(glob.glob(os.path.join(GENE_DIR, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {GENE_DIR}")
    files = files[:batch_size]

    arrs = []
    for f in files:
        a = np.load(f)
        assert a.shape == (186, 4942)
        arrs.append(a)

    batch = np.stack(arrs, axis=0)
    return torch.tensor(batch, dtype=torch.float32), files''',
            language="python",
        )
        st.code("15 × (186,4942)\n        ↓ np.stack(axis=0)\n(15,186,4942)\n        ↓ torch.tensor(...)\nGeneBranch input", language="text")
        if PATHS["test_gene_branch"].exists():
            st.markdown("### Complete test file")
            st.code(numbered_source(PATHS["test_gene_branch"]), language="python")
        else:
            st.info("The exact test_gene_branch.py source is represented by the verified code block above; if the file is placed in the project root, the dashboard will also show the complete file automatically.")

    with st.expander("4 — gene_branch.py: process the batch", expanded=True):
        st.markdown("**File:** `src/gene_branch/gene_branch.py`")
        st.markdown("### Actual forward code")
        st.code(
            '''def forward(self, x):
    x = self.pathway_embed(x)
    x = x + self.pos_gene_embed
    x = self.pos_drop(x)
    x = self.encoder(x)
    return x''',
            language="python",
        )
        st.code("(15,186,4942) → PathwayEmbedding → (15,186,256) → + position → Transformer → xP (15,186,256)", language="text")

    with st.expander("4A — pathway_embedding.py: 4,942 → 1,000 → 256", expanded=True):
        st.markdown("**File:** `src/gene_branch/pathway_embedding.py`")
        st.markdown("### Actual code")
        st.code(
            '''class PathwayEmbedding(nn.Module):
    def __init__(self, in_features=4942, hidden_features=1000,
                 out_features=256, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x''',
            language="python",
        )
        st.code("(15,186,4942) → (15,186,1000) → (15,186,256)", language="text")
        st.write("The 186 pathway rows stay as 186 tokens; each token becomes a 256-dimensional learned representation.")

    with st.expander("4B — positional embedding", expanded=True):
        st.markdown("**File:** `src/gene_branch/gene_branch.py`")
        st.code("x = x + self.pos_gene_embed\nx = self.pos_drop(x)", language="python")
        st.code("(15,186,256) + (1,186,256) → (15,186,256)", language="text")
        st.write("A learnable position vector is added for each of the 186 pathway positions.")

    with st.expander("4C — transformer_encoder.py: two Transformer blocks", expanded=True):
        st.markdown("**File:** `src/gene_branch/transformer_encoder.py`")
        st.code(
            '''class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads=16, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

    def forward(self, x, key_padding_mask=None):
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x''',
            language="python",
        )
        st.code("(15,186,256) → Block 1 → Block 2 → (15,186,256)", language="text")

    with st.expander("4D — inside a Transformer block: attention + MLP", expanded=False):
        st.markdown("**File:** `src/gene_branch/transformer_encoder.py`")
        st.markdown("### Actual block code")
        st.code(
            '''x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
x = x + self.mlp(self.norm2(x))''',
            language="python",
        )
        st.markdown("### Actual attention core")
        st.code(
            '''qkv = (
    self.qkv(x)
    .reshape(B, N, 3, self.num_heads, C // self.num_heads)
    .permute(2, 0, 3, 1, 4)
)
q, k, v = qkv[0], qkv[1], qkv[2]
attn = (q @ k.transpose(-2, -1)) * self.scale
attn = attn.softmax(dim=-1)
x = (attn @ v).transpose(1, 2).reshape(B, N, C)''',
            language="python",
        )
        st.write("Self-attention allows the pathway tokens to exchange information; the MLP then transforms each token.")

    with st.expander("5 — test_gene_branch.py: final verification", expanded=True):
        st.markdown("**File:** `test_gene_branch.py`")
        st.markdown("### Actual verification code")
        st.code(
            '''model = GeneBranch()
model.train()
out = model(x)

assert out.shape == (x.shape[0], 186, 256)
ok_finite = not torch.isnan(out).any() and not torch.isinf(out).any()

total_params = sum(p.numel() for p in model.parameters())

loss = out.sum()
loss.backward()
n_total = sum(1 for _ in model.parameters())
n_grad = sum(
    1 for p in model.parameters()
    if p.grad is not None and p.grad.abs().sum() > 0
)''',
            language="python",
        )
        metric_row([
            ("Input", "(15,186,4942)"),
            ("Output", "(15,186,256)"),
            ("Params", "6,826,392"),
            ("Gradients", "all parameters"),
        ])
        st.success("GeneBranch verification checks output shape, NaN/Inf safety, parameter count, and nonzero gradients.")

    st.markdown("### Final gene-side result")
    st.code(
        "RNA-seq → preprocess_gene.py → build_pathway_matrix.py → 15 .npy files\n"
        "→ test_gene_branch.py → (15,186,4942)\n"
        "→ GeneBranch → PathwayEmbedding → positional embedding → Transformer ×2\n"
        "→ xP (15,186,256) → test_gene_branch.py verification",
        language="text",
    )

    st.divider()

    st.header("GeneBranch — continued from the preprocessing output")
    st.write("Follow the real execution path: patient selection → pathway-aware gene input → PathwayEmbedding → positional embedding → Transformer.")
    st.info("15 × 186 × 4,942 → 15 × 186 × 256")

    with st.expander("2.1 — Patient selection", expanded=True):
        st.markdown("**File:** `scripts/run_gene_pipeline.py`")
        st.markdown("**Actual code:**")
        st.code("""def main():
    cfg = load_config()
    clinical = pd.read_csv(
        cfg["paths"]["clinical"],
        sep="\t",
        comment="#",
        low_memory=False
    )
    patient_ids = (
        clinical["PATIENT_ID"]
        .dropna()
        .unique()[: cfg["sample"]["n_patients"]]
        .tolist()
    )
    run_gene_pipeline(cfg, patient_ids)""", language="python")
        st.write("The runner reads the clinical table, selects the configured 15 unique patients, and passes their IDs into preprocessing.")

    with st.expander("2.2 — Create the pathway-aware input", expanded=True):
        st.markdown("**File:** `src/gene_branch/preprocess_gene.py`")
        st.markdown("**Actual code:**")
        st.code("""e_prime = build_pathway_expression_for_patient(
    e_prime,
    expr_scaled.loc[patient_id]
)

np.save(
    out_dir / f"{patient_id}.npy",
    e_prime.values.astype(np.float32)
)""", language="python")
        metric_row([("One patient", "(186, 4942)"), ("15-patient batch", "(15, 186, 4942)")])
        st.write("Each pathway is represented in the same 4,942-gene universe. One `.npy` file is saved for each patient.")
        st.markdown("### Theory you should know")
        st.write("**Pathway-aware representation:** the genes are organized according to the 186 KEGG pathways, so each row corresponds to one pathway rather than treating all genes as one undifferentiated list.")
        st.write("**Why 186 rows?** These 186 rows become the 186 pathway tokens processed by GeneBranch. Each row uses the full 4,942-gene universe; genes outside that pathway are effectively zero-masked.")
        st.info("Think of Stage 1 as preparing 186 pathway-specific views of the same gene-expression profile. Stage 2 learns useful features from those views.")

    with st.expander("2.3 — Build GeneBranch", expanded=True):
        st.markdown("**File:** `src/gene_branch/gene_branch.py`")
        st.markdown("**Actual constructor code:**")
        st.code("""self.pathway_embed = PathwayEmbedding(
    in_features=in_features,
    hidden_features=hidden_features,
    out_features=embed_dim,
    drop=drop_rate,
)

self.pos_embed = nn.Parameter(
    torch.zeros(1, num_pathways, embed_dim)
)
trunc_normal_(self.pos_embed, std=0.02)
self.pos_drop = nn.Dropout(p=drop_rate)

self.transformer = TransformerEncoder(
    dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    drop_rate=drop_rate,
    attn_drop_rate=drop_rate,
)""", language="python")
        st.write("GeneBranch contains PathwayEmbedding, a learnable pathway-position embedding, and a 2-block TransformerEncoder.")
        st.markdown("### Important definitions")
        st.write("**GeneBranch:** the neural-network branch that converts the pathway-aware gene tensor into learned pathway representations.")
        st.write("**Embedding:** a learned transformation that converts a high-dimensional input into a smaller feature representation. Here, each pathway's 4,942 values become a 256-dimensional pathway token.")
        st.write("**Token:** one vector representing one item in the sequence. Here, each of the 186 pathways is treated as one token, so the sequence length is 186.")

    with st.expander("2.4 — Actual input → PathwayEmbedding call", expanded=True):
        st.markdown("**File:** `src/gene_branch/gene_branch.py → GeneBranch.forward()`")
        st.markdown("**Actual code:**")
        st.code("""def forward(self, x):
    x = self.pathway_embed(x)
    x = x + self.pos_embed
    x = self.pos_drop(x)
    x = self.transformer(x)
    return x""", language="python")
        st.code("(15,186,4942)\n      ↓ self.pathway_embed(x)\n(15,186,256)\n      ↓ + self.pos_embed\n(15,186,256)\n      ↓ self.transformer(x)\nxP = (15,186,256)", language="text")
        st.markdown("### The key idea")
        st.write("The `forward()` function is the actual execution path: the input is compressed into pathway tokens, pathway identity is added, and then the Transformer lets those pathway tokens interact.")

    with st.expander("2.5 — PathwayEmbedding: complete class", expanded=True):
        st.markdown("**File:** `src/gene_branch/pathway_embedding.py`")
        st.markdown("**Actual code:**")
        st.code("""class PathwayEmbedding(nn.Module):
    def __init__(
        self,
        in_features: int = 4942,
        hidden_features: int = 1000,
        out_features: int = 256,
        drop: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x""", language="python")
        st.code("(B,186,4942) → Linear 4942→1000 → GELU → Dropout → Linear 1000→256 → Dropout → (B,186,256)", language="text")
        st.write("The 186 pathways remain 186 tokens; only each token's feature dimension changes from 4,942 to 256.")
        st.markdown("### Why is PathwayEmbedding important?")
        st.write("A **pathway token** is the learned 256-number representation of one pathway's 4,942-dimensional input. The embedding does not reduce the number of pathways: 186 pathways in → 186 tokens out.")
        st.write("The two linear layers perform a feature projection: `4942 → 1000 → 256`. The first layer creates an intermediate feature space; the second produces the 256-dimensional representation used by the Transformer.")
        st.write("**GELU** is the nonlinear activation between the two linear layers. Nonlinearity allows the network to learn relationships that a purely linear transformation could not represent.")
        st.write("**Dropout (0.1)** randomly drops some activations during training as regularization. It is disabled during evaluation.")
        st.info("Presentation sentence: 'PathwayEmbedding converts every 4,942-dimensional pathway vector into a compact 256-dimensional learned pathway token.'")

    with st.expander("2.6 — Add positional embedding", expanded=True):
        st.markdown("**File:** `src/gene_branch/gene_branch.py → GeneBranch.forward()`")
        st.code("""x = x + self.pos_embed
x = self.pos_drop(x)""", language="python")
        st.code("(15,186,256) pathway tokens\n        +\n(1,186,256) learnable positions\n        ↓\n(15,186,256) position-aware tokens", language="text")
        st.write("The `(1,186,256)` positional tensor broadcasts across the 15 patients.")
        st.markdown("### Why do we need positional embedding?")
        st.write("Self-attention does not inherently encode sequence position. The learnable `pos_embed` gives each of the 186 pathway positions its own trainable positional information.")
        st.write("It is **learnable**, meaning its values are parameters updated during training. It is not another gene-expression measurement.")

    with st.expander("2.7 — Actual TransformerEncoder call", expanded=True):
        st.markdown("**Caller — `src/gene_branch/gene_branch.py`**")
        st.code("x = self.transformer(x)", language="python")
        st.markdown("**Called implementation — `src/gene_branch/transformer_encoder.py`**")
        st.code("""class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads=16, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

    def forward(self, x, key_padding_mask=None):
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x""", language="python")
        st.write("GeneBranch uses depth=2, so two Transformer Blocks run sequentially.")
        st.markdown("### What is a Transformer Encoder?")
        st.write("A **Transformer Encoder** is a stack of Transformer blocks that lets tokens exchange information and build richer contextual representations. Here, the tokens are biological pathways rather than words.")
        st.write("**Depth = 2** means Block 1 runs first and its output becomes the input to Block 2. The tensor shape remains `(15,186,256)` while the representation becomes more refined.")
        st.info("Presentation sentence: 'The Transformer encoder models relationships between the 186 pathway tokens using two self-attention blocks.'")

    with st.expander("2.8 — Inside one Transformer Block", expanded=False):
        st.markdown("**File:** `src/gene_branch/transformer_encoder.py`")
        st.code("""class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            dim, hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU, drop=drop
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(
            self.norm1(x),
            key_padding_mask=key_padding_mask
        )
        x = x + self.mlp(self.norm2(x))
        return x""", language="python")
        st.code("Input → LayerNorm → Self-Attention → residual → LayerNorm → MLP → residual → Output", language="text")
        st.markdown("### Four terms you should know")
        st.write("**LayerNorm:** normalizes the features within each token before attention or the MLP, helping keep the network stable.")
        st.write("**Self-Attention:** lets every pathway token compare itself with other pathway tokens and combine relevant information.")
        st.write("**Residual connection:** adds the original input back to the processed result (`x + ...`), helping information and gradients flow through the network.")
        st.write("**MLP (feed-forward network):** transforms each token after attention. Here it expands `256 → 1024` and then returns `1024 → 256`.")

    with st.expander("2.9 — Inside Multi-Head Self-Attention", expanded=False):
        st.markdown("**File:** `src/gene_branch/transformer_encoder.py`")
        st.code("""qkv = (
    self.qkv(x)
    .reshape(B, N, 3, self.num_heads, C // self.num_heads)
    .permute(2, 0, 3, 1, 4)
)
q, k, v = qkv[0], qkv[1], qkv[2]

attn = (q @ k.transpose(-2, -1)) * self.scale
attn = attn.softmax(dim=-1)

x = (attn @ v).transpose(1, 2).reshape(B, N, C)
x = self.proj(x)
x = self.proj_drop(x)""", language="python")
        st.code("Pathway tokens → Q,K,V → QKᵀ → softmax → attention weights × V → updated pathway tokens", language="text")
        st.write("In simple terms, the pathways can exchange information with one another.")
        st.markdown("### Q, K and V — the idea")
        st.write("For every pathway token, attention creates three learned representations: **Query (Q), Key (K), and Value (V)**.")
        st.write("A query effectively asks which other pathway information is relevant. Keys are compared with that query to produce attention scores, while Values contain the information that is combined.")
        st.write("**Softmax** converts the scores into attention weights, so the model can give more importance to some pathways and less to others.")
        st.write("**Multi-head attention:** the 256 features are split across 16 heads. Each head works with `256 / 16 = 16` features, allowing different heads to learn different interaction patterns before their outputs are combined.")

    with st.expander("2.10 — Inside the Transformer MLP", expanded=False):
        st.markdown("**File:** `src/gene_branch/transformer_encoder.py`")
        st.code("""class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x""", language="python")
        st.code("256 → 1024 → GELU → Dropout → 256", language="text")
        st.write("With mlp_ratio=4, each 256-dimensional token temporarily expands to 1,024 features and then returns to 256.")
        st.markdown("### Why is the MLP there?")
        st.write("Attention mainly mixes information **between pathway tokens**. The MLP then performs a nonlinear feature transformation **within each token**. Together, attention + MLP provide both interaction and feature processing.")
        st.info("Easy distinction: **Attention = pathways talk to each other. MLP = each pathway token is further transformed on its own.**")

    with st.expander("2.11 — Final GeneBranch output: xP", expanded=True):
        st.markdown("**File:** `src/gene_branch/gene_branch.py`")
        st.code("""x = self.transformer(x)
return x""", language="python")
        metric_row([("Input", "(15,186,4942)"), ("PathwayEmbedding", "(15,186,256)"), ("Transformer", "2 blocks"), ("Final xP", "(15,186,256)")])
        st.success("Stage 2 output: xP = (15,186,256). This learned pathway representation is passed to later PAMT stages.")
        st.markdown("### What does `xP` mean?")
        st.write("`xP` is the final **pathway representation** produced by GeneBranch. There are still 186 pathway tokens, but each is now a learned 256-dimensional feature vector containing information processed from the original gene input and relationships with other pathways.")
        st.write("Later PAMT components use this learned representation instead of the original 4,942-dimensional pathway input.")

    st.markdown("### Stage 2 — complete execution chain")
    st.code("patient gene files → batch (15,186,4942) → PathwayEmbedding 4942→1000→256 → + positions → Transformer ×2 → xP (15,186,256)", language="text")


def render_wsi_pipeline_complete():
    st.header("Stage 3 — WSI / DINO pipeline → WSIBranch → WSI reduction")
    st.markdown("### Theory / input / output summary")
    st.code("""
INPUT  → selected WSI tissue patches
         each patch → DINO → 384-D vector
         each patient has N=94–162 patches

PROCESS → pad to N=162 + padding mask
          → WSIBranch: positional embedding + Transformer ×6
          → WSIReduction: 384→640→256

OUTPUT → xI = (15,162,256)
         162 is the padded patch-token limit;
         the mask tells later attention/loss which positions are real.
    """, language="text")
    st.write(
        "This is the complete WSI-side execution flow in one continuous stage. "
        "It follows slide preparation → tissue patch extraction → DINO pretraining → "
        "384-D patch embeddings → representative-patch selection → batching/padding → "
        "WSIBranch → Stage-3 verification → WSI reduction."
    )
    st.info(
        "run_dino_training.py → loader.py → patch_extraction.py → dino_pretraining.py → "
        "dino_embedding.py → patch_clustering.py → test_wsi_branch_batch.py → "
        "wsi_branch.py → patch_embedding.py → transformer_encoder.py × 6 → "
        "test_wsi_branch.py → test_wsi_reduction.py → wsi_reduction.py → xI"
    )

    st.markdown("### Complete WSI-side flow")
    st.code("""RAW .SVS WHOLE-SLIDE IMAGES
        ↓
run_dino_training.py
        ↓
loader.py
        ↓
patch_extraction.py
        ↓
valid tissue patches
        ↓
dino_pretraining.py
        ↓
trained DINO ViT-S/8 checkpoint
        ↓
dino_embedding.py
        ↓
per-patch embeddings: (N,384)
        ↓
patch_clustering.py
        ↓
representative patches / embeddings
        ↓
test_wsi_branch_batch.py
        ↓ padding + mask
WSI batch: (15,162,384)
mask: (15,162), True = padding
        ↓
wsi_branch.py
        ↓
PatchEmbedding
        ↓
TransformerEncoder × 6, 16 heads
        ↓
xI_384: (15,162,384)
        ↓
test_wsi_branch.py
        ↓
test_wsi_reduction.py
        ↓
wsi_reduction.py
        ↓
xI: (15,162,256)
        ↓
Fusion + Contrastive L3""", language="text")

    metric_row([
        ("Patients", "15"),
        ("Patches / patient", "94–162"),
        ("Padded N", "162"),
        ("Patch features", "384"),
        ("Final WSI features", "256"),
    ])
    st.divider()

    with st.expander("1 — run_dino_training.py: create the WSI/DINO training corpus", expanded=True):
        st.write(
            "Reads the WSI manifest, finds `.svs` slides, opens them through `loader.py`, "
            "selects the appropriate 20× pyramid level, calls patch extraction, collects "
            "usable tissue patches across the cohort, and sends the corpus to DINO pretraining."
        )
        st.markdown("### Theory you should know")
        st.write(
            "**WSI (Whole-Slide Image):** a very large digitized pathology slide. It is too large "
            "to process as one normal image, so it is divided into smaller patches."
        )
        st.write(
            "**Pyramid level:** WSIs are stored at multiple resolutions. Selecting a 20× level "
            "provides the resolution used for this patch-extraction pipeline."
        )
        show_code(PATHS["dino_runner"], [r"def main", r"manifest", r"patch", r"dino_pretraining"])

    with st.expander("2 — loader.py: open the `.svs` slide and create a thumbnail", expanded=True):
        st.write(
            "Provides slide-loading utilities. It opens the whole-slide image and creates a "
            "low-resolution thumbnail that can be used for efficient tissue filtering."
        )
        st.markdown("### Theory you should know")
        st.write(
            "A **thumbnail** is a much smaller representation of the slide. It is useful for "
            "quickly deciding where tissue exists without scanning the full-resolution slide."
        )
        show_code(PATHS["wsi_loader"], [r"def", r"thumbnail", r"slide"])

    with st.expander("3 — patch_extraction.py: extract and clean tissue patches", expanded=True):
        st.write(
            "Selects the appropriate pyramid level, extracts non-overlapping 256×256 patches, "
            "performs low-resolution tissue pre-filtering, removes background, removes blurred or "
            "low-sharpness patches, removes ink/debris artifacts, and outputs valid tissue patches."
        )
        st.markdown("### Theory you should know")
        st.write(
            "**Patch extraction** converts a huge WSI into manageable local regions. Filtering is "
            "important because background and artifacts contain little useful histological information "
            "and would waste downstream computation."
        )
        st.write("**Non-overlapping 256×256** patches avoid repeatedly processing the same pixels.")
        show_code(PATHS["patch_extraction"], [r"256", r"patch", r"tissue", r"sharp", r"blur", r"ink"])

    with st.expander("4 — dino_pretraining.py: learn visual features without manual labels", expanded=True):
        st.write(
            "Takes the tissue-patch training corpus, applies multi-crop augmentation, and trains "
            "DINO ViT-S/8 using student–teacher self-supervised learning. The output is a trained DINO checkpoint."
        )
        st.markdown("### Theory you should know")
        st.write(
            "**Self-supervised learning** learns useful representations without requiring a manual "
            "class label for every patch. **Student–teacher learning** trains the student to produce "
            "representations consistent with the teacher across augmented views."
        )
        st.write(
            "**Multi-crop augmentation** gives the model different views of the same patch, encouraging "
            "features that are robust to changes in crop and scale."
        )
        show_code(PATHS["dino_pretraining"], [r"class", r"train", r"crop", r"teacher", r"student"])

    with st.expander("5 — dino_embedding.py: turn each patch into a 384-D vector", expanded=True):
        st.write(
            "Loads the trained DINO ViT-S/8 checkpoint, preprocesses each selected patch, passes it "
            "through DINO, and converts each patch into a 384-dimensional embedding. One WSI therefore "
            "becomes `(N,384)`."
        )
        st.markdown("### Theory you should know")
        st.write(
            "An **embedding** is a learned numerical representation. Instead of passing raw pixels into "
            "the WSI Transformer, the model receives a compact 384-number description of each patch."
        )
        st.code("one patch → DINO → 384-D vector\nN patches → (N,384)", language="text")
        show_code(PATHS["dino_embedding"], [r"checkpoint", r"embedding", r"forward", r"384"])

    with st.expander("6 — patch_clustering.py: select representative patches", expanded=True):
        st.write(
            "Takes the 384-D patch embeddings, applies K-means with 20 clusters, and selects up to 5 "
            "representative patches per cluster, producing up to 100 selected patches for a WSI sequence."
        )
        st.markdown("### Theory you should know")
        st.write(
            "**K-means** groups similar feature vectors into K clusters by assigning points to nearby "
            "cluster centers and updating those centers. Here K=20."
        )
        st.write(
            "The purpose is to reduce the number of patches while retaining representative visual regions. "
            "Clustering changes how many patches continue downstream; it does not change the 384-D feature size."
        )
        st.code("384-D embeddings\n      ↓ K-means, K=20\nup to 5 representatives / cluster\n      ↓\nup to 100 selected patches", language="text")
        show_code(PATHS["patch_clustering"], [r"KMeans", r"20", r"cluster", r"representative"])

    with st.expander("7 — test_wsi_branch_batch.py: make the 15-patient padded batch", expanded=True):
        st.write(
            "Loads the processed WSI embeddings, handles different numbers of patches using padding, "
            "creates the boolean padding mask, produces the current batch `(15,162,384)`, and sends it into WSIBranch."
        )
        st.markdown("### Theory — padding and masking")
        st.write(
            "Your patients have different patch counts, from 94 to 162. A batch needs a common sequence length, "
            "so shorter sequences are padded to 162. The **padding mask** tells attention which positions are artificial."
        )
        st.write("In this project: **True = padding**, **False = valid patch**.")
        st.code("Patient example: 144 × 384\n        ↓ pad\n162 × 384\n\n15 patients → (15,162,384)\nmask → (15,162)", language="text")
        st.caption("The batch-padding behavior is documented and verified by the committed WSI branch tests; the original local helper `test_wsi_branch_batch.py` is not distributed in the public demo repository.")

    with st.expander("8 — wsi_branch.py: process the visual patch-token sequence", expanded=True):
        st.write(
            "WSIBranch receives the padded WSI tensor and calls PatchEmbedding followed by TransformerEncoder. "
            "The feature dimension remains 384 through this branch."
        )
        st.markdown("### Theory you should know")
        st.write(
            "The WSI branch treats **image patches as tokens**. The Transformer lets each patch use information "
            "from other patches in the same patient's WSI, producing contextual visual representations."
        )
        st.code("(15,162,384) + mask\n        ↓\nWSIBranch\n        ↓\nPatchEmbedding\n        ↓\nTransformerEncoder × 6\n        ↓\nxI_384 = (15,162,384)", language="text")
        show_code(PATHS["wsi_branch"], [r"class WSIBranch", r"def forward", r"PatchEmbedding", r"TransformerEncoder"])
        metric_row([("Input", "(15,162,384)"), ("Blocks", "6"), ("Heads", "16"), ("Output", "(15,162,384)")])

    with st.expander("9 — patch_embedding.py: add learnable positional information", expanded=True):
        st.write(
            "Adds learnable positional embeddings while keeping the feature dimension at 384. "
            "The padding mask is carried onward for the Transformer."
        )
        st.markdown("### Theory you should know")
        st.write(
            "A **positional embedding** gives the model information about where a token occurs in the sequence. "
            "Self-attention alone does not inherently encode position."
        )
        st.code("(15,162,384) + positional information\n                ↓\n          (15,162,384)", language="text")
        show_code(PATHS["patch_embedding"], [r"class PatchEmbedding", r"pos", r"forward", r"384"])

    with st.expander("10 — transformer_encoder.py: run 6 Transformer blocks", expanded=True):
        st.write(
            "The WSI TransformerEncoder runs 6 Transformer blocks with 16 attention heads. The tensor shape stays "
            "`(15,162,384)`; the representation is refined rather than resized."
        )
        st.markdown("### Theory you should know")
        st.write(
            "A **Transformer Encoder** is a stack of blocks that repeatedly performs self-attention and nonlinear "
            "feature processing. Here the tokens are tissue patches. **Depth = 6** means six blocks run sequentially."
        )
        st.write(
            "**Multi-head attention:** the feature space is processed through multiple attention heads in parallel, "
            "allowing the model to learn different patch-to-patch relationships."
        )
        st.code("(15,162,384)\n ↓ Block 1 → Block 2 → … → Block 6\n(15,162,384) = xI_384", language="text")
        if PATHS["wsi_transformer_encoder"] is not None:
            show_code(PATHS["wsi_transformer_encoder"], [r"class TransformerEncoder", r"class Block", r"num_heads", r"def forward"])
        else:
            st.caption("The committed mini-scale repository keeps the WSI Transformer implementation inside `src/wsi_branch/wsi_branch.py`; no separate transformer_encoder.py file is required for the hosted demo.")
        st.markdown("### Concepts to remember")
        st.write(
            "**LayerNorm:** stabilizes features. **Self-attention:** lets patches exchange information. "
            "**Residual connection:** adds the original representation back to the transformed result. "
            "**MLP:** performs nonlinear feature processing after attention."
        )
        st.write(
            "In attention, **Q (Query)** represents what a patch is looking for, **K (Key)** represents what a patch "
            "offers for matching, and **V (Value)** is the information that gets combined. **Softmax** converts "
            "attention scores into weights."
        )

    with st.expander("11 — test_wsi_branch.py: verify the Stage-3 WSIBranch", expanded=True):
        st.write(
            "Verifies output shape, numerical validity, gradients, and padding behavior. Your previously verified "
            "Stage-3 result is `(15,162,384)` with 6 blocks, 16 heads, 10,708,992 parameters, and a worst "
            "masked-output difference of about `1.67e-6`."
        )
        st.markdown("### Why the mask matters")
        st.write(
            "Without masking, padded positions could behave like real tissue patches and influence attention. "
            "The mask prevents artificial positions from affecting valid patch representations."
        )
        show_code(PATHS["test_wsi_branch"], [r"assert", r"shape", r"mask", r"grad", r"PASSED"])
        st.success("Verified Stage 3 WSIBranch output: xI_384 = (15,162,384).")

    st.divider()

    with st.expander("12 — test_wsi_reduction.py: hand Stage-3 output to reduction", expanded=True):
        st.write(
            "Takes the verified Stage-3 WSI representation and passes it to WSIReduction. This is the handoff "
            "from the 384-D WSI representation into the common multimodal feature space."
        )
        st.code("xI_384 = (15,162,384)\n        ↓\ntest_wsi_reduction.py\n        ↓\nWSIReduction", language="text")
        show_code(PATHS["test_wsi_reduction"], [r"WSIReduction", r"384", r"256", r"shape"])

    with st.expander("13 — wsi_reduction.py: reduce 384 → 256", expanded=True):
        st.write(
            "WSIReduction aligns the WSI feature dimension with the gene branch. It keeps the patch-token count "
            "at 162 and changes only the feature dimension from 384 to 256."
        )
        st.markdown("### Theory you should know")
        st.write(
            "This is a **feature projection**, not patch pooling. The 162 patch tokens remain 162. "
            "The padding mask remains `(15,162)`. The resulting 256-D space is compatible with `xP (15,186,256)` "
            "for fusion and contrastive learning."
        )
        st.code("(15,162,384)\n      ↓ Linear 384 → 640\n(15,162,640)\n      ↓ GELU\n      ↓ Dropout\n      ↓ Linear 640 → 256\n(15,162,256)\n      ↓ Dropout\nxI = (15,162,256)\nmask: (15,162) → unchanged", language="text")
        st.markdown("### GELU and Dropout")
        st.write("**GELU** adds nonlinearity so the projection can learn richer feature mappings. **Dropout** randomly removes activations during training as regularization and is disabled during evaluation.")
        show_code(PATHS["wsi_reduction"], [r"class WSIReduction", r"def forward", r"self.fc1", r"self.fc2"])
        metric_row([("Input", "(15,162,384)"), ("Hidden", "640"), ("Output", "(15,162,256)"), ("Mask", "unchanged")])

    st.success(
        "Stage 3 complete: the WSI side produces xI = (15,162,256), dimension-compatible with "
        "xP = (15,186,256) for the later multimodal stages."
    )
    st.markdown("### Final WSI-side execution chain")
    st.code(
        "run_dino_training.py → loader.py → patch_extraction.py → dino_pretraining.py → "
        "dino_embedding.py → patch_clustering.py → test_wsi_branch_batch.py → wsi_branch.py → "
        "patch_embedding.py → transformer_encoder.py ×6 → test_wsi_branch.py → "
        "test_wsi_reduction.py → wsi_reduction.py → xI (15,162,256)",
        language="text",
    )

def render_contrastive():
    st.header("Stage 4 — Pathway-to-patch contrastive loss (L3)")
    st.markdown("### Theory / input / output summary")
    st.code("""
INPUT
  xP = (15,186,256)  ← GeneBranch pathway tokens
  xI = (15,162,256)  ← reduced WSI patch tokens
  mask = (15,162)    ← True means padding

PROCESS
  1. Compare every pathway with every patch:
       S = xP @ xIᵀ → (15,186,162)
  2. For each pathway, find the top 2 real patches.
  3. Build Y_h: those 2 are positive, the rest negative.
  4. Softmax over real patches only.
  5. Calculate the contrastive loss for each pathway.
  6. Average → L3.

OUTPUT
  L3 = scalar
  Also returns S, S', Y_h and topk_idx as diagnostics.

KEY IDEA
  "Which WSI patches are most relevant to this pathway?"
  The loss makes the selected top-2 patches the positive targets.
    """, language="text")

    st.markdown("### Purpose")
    st.write(
        "For every pathway token, the loss encourages the model to assign "
        "high probability to the two WSI patches with the highest raw "
        "pathway-to-patch similarity."
    )

    st.code("""
xP (B,186,256) ─────────┐
                        ▼
                 raw inner product
                 S = xP @ xIᵀ
                        ▲
                        │
xI (B,N,256) ───────────┘
                        │
                        ▼
                 S: (B,186,N)
                        │
                top-h = 2
                        │
                        ▼
                   Y_h labels
                        │
              masked softmax
                        │
                        ▼
                       L3
                    scalar
    """, language="text")

    st.markdown("### Important implementation decision")
    st.warning(
        "The paper describes a learnable temperature τ, but this module "
        "intentionally follows the released official code: there is no "
        "temperature parameter. The module therefore has 0 learnable parameters."
    )

    st.markdown("### Code")
    show_code(PATHS["contrastive"], [
        r"class PathwayPatchContrastiveLoss",
        r"def forward",
        r"torch.topk",
        r"F.softmax",
        r"per_row_loss",
    ])

    metric_row([
        ("xP", "(15,186,256)"),
        ("xI", "(15,162,256)"),
        ("Top h", "2"),
        ("L3", "scalar"),
    ])

    st.markdown("### Padding adaptation")
    st.write(
        "The released model assumes a fixed patch count. This mini project "
        "has variable N, so padded columns are excluded from top-k selection "
        "and the softmax. This is a project-specific adaptation."
    )

    st.code(
        "Stage 4 output L3 → passed directly into SurvivalLoss in Stage 6.\n"
        "Stage 4 does NOT create a new patient label file: Y_h is created internally from top-2 similarities.",
        language="text",
    )


def render_fusion():
    st.header("Stage 5A — Pathway-to-patch fusion")

    st.markdown("### 5A.1 — Why do we need fusion?")
    st.write(
        "Before fusion, the two modalities have been processed separately: "
        "the GENE branch has learned pathway representations and the WSI branch "
        "has learned patch representations. Fusion is where they are finally "
        "allowed to interact. The key idea is pathway-guided attention: each "
        "biological pathway asks which image patches are most relevant to it."
    )
    st.info(
        "Easy way to remember it: **GENE asks the question; WSI provides the evidence.** "
        "A pathway is the query, and the WSI patches are the keys/values."
    )

    st.markdown("### 5A.2 — Inputs")
    st.code("""
INPUTS TO FUSION

xP  = (15, 186, 256)
      15 patients × 186 pathway tokens × 256 features
      ↑ comes from GeneBranch

xI  = (15, N, 256)
      15 patients × N WSI patches × 256 features
      N is variable per patient (94–162); padded batch uses N=162
      ↑ comes from WSIReduction

mask = (15, 162)
       True  = padding / fake patch position
       False = real patch
    """, language="text")

    st.markdown("**Why must both inputs have 256 features?**")
    st.write(
        "Cross-attention compares pathway queries with patch keys. The model therefore "
        "uses the same 256-dimensional feature space for xP and xI after WSIReduction. "
        "The patient dimension and token counts stay unchanged."
    )

    st.markdown("### 5A.3 — What is cross-attention? (simple idea)")
    st.write(
        "In ordinary self-attention, tokens from the same modality look at each other. "
        "Here it is **cross-attention**: one modality supplies the queries and the other "
        "supplies the keys and values. This creates a direct pathway → image relationship."
    )
    st.code("""
GENE / pathways                         WSI / patches
xP: (B,186,256)                         xI: (B,N,256)
      │                                      │
      │ Linear                               │ Linear × 2
      ▼                                      ▼
Qpath: (B,186,256)                  Kpatch: (B,N,256)
                                    Vpatch: (B,N,256)
      │                                      │
      └────────────── compare Q with K ─────┘
                         │
                         ▼
                  attention scores
                    (B,16,186,N)
                         │
                   mask padding
                         │
                      softmax
                         │
                  weights × Vpatch
                         │
                         ▼
                    CA: (B,186,256)
    """, language="text")

    st.markdown("### 5A.4 — How the attention is calculated")
    st.write(
        "For every pathway, the model compares its query with every WSI patch key. "
        "A larger similarity means that patch receives more attention for that pathway. "
        "Softmax converts the scores into weights, and those weights are used to combine "
        "the patch values."
    )
    st.code("""
Qpath = Linear(xP)
Kpatch = Linear(xI)
Vpatch = Linear(xI)

scores = Qpath @ Kpatchᵀ / √d
weights = softmax(scores, over patches)
CA = weights @ Vpatch
    """, language="text")

    st.markdown("### 5A.5 — What do the dimensions mean?")
    st.code("""
attention = (15, 16, 186, 162)
              │   │   │    └─ patches each pathway can attend to
              │   │   └────── pathways / queries
              │   └────────── attention heads
              └────────────── patients

CA = (15,186,256)
      │   │    └─ 256 fused features
      │   └────── one fused representation per pathway
      └────────── patients
    """, language="text")
    st.write(
        "There are 16 heads. Each head works on 16 features (256 ÷ 16), so different "
        "heads can learn different pathway-to-patch relationships before their results "
        "are combined back into 256 features."
    )

    st.markdown("### 5A.6 — Why is the padding mask important?")
    st.write(
        "Patients do not all have the same number of real patches. We pad shorter WSI "
        "sequences to 162 so they can be processed as one batch. The mask tells attention "
        "which positions are fake. Padded positions are excluded from the attention "
        "calculation, so the model cannot learn from nonexistent patches."
    )
    st.code("""
Example patient has N=100 real patches:

positions 0–99   → real patches → usable by attention
positions 100–161 → padding     → masked out

mask shape = (15,162)
True  = ignore this patch
False = use this patch
    """, language="text")

    st.markdown("### 5A.7 — What does the output mean?")
    st.write(
        "CA is a **pathway-guided image representation**. For each of the 186 pathways, "
        "the model has combined information from the WSI patches that it considers relevant "
        "to that pathway. This is why the output is still `(15,186,256)`: fusion changes the "
        "features, not the number of pathways."
    )
    st.code("""
xP  (15,186,256) ───────────────┐
                                │ pathway queries
xI  (15,162,256) ───────────────┤ patch keys + values
                                ▼
                         Cross-Attention
                                │
                                ▼
CA  (15,186,256)  ← fused pathway-level multimodal features
                                │
                                ▼
                    Stage 5B — RiskHead
    """, language="text")

    st.markdown("### 5A.8 — Paper equation vs our official-code implementation")
    st.write(
        "The paper describes a final concatenation of the pathway feature with the "
        "cross-attention result. However, the released implementation used in this project "
        "returns **CA directly**. Our dashboard documents and tests the code-faithful version, "
        "so the fusion output is `(15,186,256)`, not `(15,186,512)`."
    )
    st.code("""
Paper Eq. 8 (paper description):
    xPI = cat(xP, CA)  → pathway feature size would become 512

Released-code behavior used here:
    fusion(...) → CA    → (15,186,256)

This project follows the released-code behavior.
    """, language="text")

    st.markdown("### 5A.9 — What `test_fusion.py` verifies")
    st.write(
        "The fusion test uses the real 15-patient gene and WSI data, runs the already-verified "
        "upstream branches, and then checks that fusion behaves correctly."
    )
    st.code("""
Real gene .npy + real WSI .npz
          │
          ▼
GeneBranch → xP (15,186,256)
WSIBranch → (15,162,384)
WSIReduction → xI (15,162,256)
          │
          ▼
Fusion(xP, xI, mask)
          │
          ├─ CA shape = (15,186,256)
          ├─ attention shape = (15,16,186,162)
          ├─ attention rows sum ≈ 1
          ├─ padded attention = 0
          ├─ all values finite
          ├─ gradients reach all trainable modules
          └─ padded-batch result ≈ single-patient result
    """, language="text")

    st.markdown("### 5A.10 — Code: where it happens")
    show_code(PATHS["fusion"], [
        r"class PathwayToPatchFusion",
        r"def forward",
        r"self.q",
        r"self.kv",
        r"attn =",
    ])

    metric_row([
        ("Pathway input xP", "(15,186,256)"),
        ("WSI input xI", "(15,162,256)"),
        ("Attention", "(15,16,186,162)"),
        ("Fusion output CA", "(15,186,256)"),
    ])

    st.success(
        "Stage 5A in one line: **pathway features query WSI patch features → "
        "attention selects relevant patches → weighted patch information becomes CA, "
        "a fused 256-dimensional representation for every pathway.**"
    )

    st.code(
        "CA → Stage 5B SurvivalRiskHead\n"
        "CA is the fused multi-modal pathway representation consumed by the risk head.",
        language="text",
    )


def render_risk_head():
    st.header("Stage 5B — Survival Risk Head")

    st.markdown("## 1. What is the Risk Head?")
    st.write(
        "The Risk Head is the final prediction component of PAMT. Earlier stages have "
        "learned features from gene expression and WSI images and Stage 5A has fused "
        "them. The Risk Head converts those learned features into **one survival-risk "
        "score for each patient**."
    )
    st.info("Simple idea: earlier stages learn useful features; Stage 5B turns them into the final patient-level number.")

    st.markdown("## 2. Inputs — where do they come from?")
    st.code("""
Gene expression → GeneBranch → xP = (15,186,256)

WSI patches → WSIBranch → WSIReduction → xI = (15,162,256)
                                      │
                                      ▼
                         Stage 5A Fusion
                                      │
                                      ▼
                           CA = (15,186,256)

                     xP + CA → Risk Head → R = (15,1)
    """, language="text")

    metric_row([
        ("xP", "(15,186,256)"),
        ("CA", "(15,186,256)"),
        ("Risk R", "(15,1)"),
    ])
    st.write(
        "**xP** is the learned gene/pathway representation. **CA** is the pathway-guided "
        "multimodal representation from Stage 5A. In both tensors: 15 = patients, "
        "186 = pathway tokens, and 256 = learned features per pathway."
    )

    st.markdown("## 3. What happens inside it?")
    st.write("There are three main operations: **pool → concatenate → linear prediction**.")
    st.code("""
xP  (15,186,256) ──► AdaptiveAvgPool2d((186,1)) ──► (15,186,1) ──┐
                                                                    │
CA  (15,186,256) ──► AdaptiveAvgPool2d((186,1)) ──► (15,186,1) ──┤
                                                                    ▼
                                                              concatenate
                                                                    ↓
                                                               (15,372,1)
                                                                    ↓ squeeze
                                                                 (15,372)
                                                                    ↓
                                                               Linear(372,1)
                                                                    ↓
                                                                 R = (15,1)
    """, language="text")

    with st.expander("3.1 — Step 1: pool xP", expanded=True):
        st.code("""xP = (15,186,256)
       ↓
AdaptiveAvgPool2d((186,1))
       ↓
gene_pooled = (15,186,1)""", language="text")
        st.write("For each pathway, the 256 learned feature values are averaged into one value. The 186 pathways remain separate.")
        st.info("256 features per pathway → 1 summary value per pathway. Pooling does NOT combine the 186 pathways.")

    with st.expander("3.2 — Step 2: pool CA", expanded=True):
        st.code("""CA = (15,186,256)
       ↓
AdaptiveAvgPool2d((186,1))
       ↓
fusion_pooled = (15,186,1)""", language="text")
        st.write("The same compression is applied to the fused pathway representation.")

    with st.expander("3.3 — Step 3: concatenate", expanded=True):
        st.code("""gene_pooled   = (15,186,1)
fusion_pooled = (15,186,1)
                 ↓
       torch.cat(..., dim=1)
                 ↓
             (15,372,1)
                 ↓ squeeze(-1)
             (15,372)""", language="text")
        st.write("Each patient now has 372 values: 186 summarized gene/pathway values + 186 summarized fusion values.")
        st.code("186 + 186 = 372", language="text")

    with st.expander("3.4 — Step 4: predict risk", expanded=True):
        st.code("""(15,372)
   ↓
Linear(372 → 1)
   ↓
R = (15,1)""", language="text")
        st.write("The linear layer learns how to combine the 372 values into one risk score for each patient.")

    st.markdown("## 4. What does R mean?")
    st.write(
        "`R = (15,1)` means exactly one model risk score per patient. It is a risk score "
        "used by the Cox survival objective, **not a percentage or probability of survival**."
    )
    st.code("Patient 1 → one risk score\nPatient 2 → one risk score\n...\nPatient 15 → one risk score", language="text")

    st.markdown("## 5. `risk_head.py` — what is actually implemented?")
    show_code(PATHS["risk_head"], [
        r"class SurvivalRiskHead", r"def forward", r"self.gap_gene", r"self.gap_fusion", r"self.head",
    ])
    metric_row([
        ("Pooling", "256 → 1 per pathway"),
        ("Head input", "(15,372)"),
        ("Output", "(15,1)"),
        ("Parameters", "373"),
    ])
    st.code("""Linear(372,1)
372 weights + 1 bias = 373 parameters""", language="text")
    st.write("There is no extra trainable hidden layer in this implementation.")

    st.markdown("## 6. Paper vs released-code-faithful implementation")
    st.warning(
        "The paper's Eq. 9 describes an LN + MLP risk-prediction design. "
        "This project follows the released-code behavior: pooling → concatenation → Linear(372,1)."
    )
    st.code("""OUR IMPLEMENTATION
xP ──► Pool ──┐
              ├──► Concatenate ──► Linear(372,1) ──► R
CA ──► Pool ──┘

No LayerNorm
No hidden MLP
No activation
No dropout
No extra projection""", language="text")

    st.markdown("## 7. What does `test_risk_head.py` verify?")
    st.write(
        "It is a **real-data verification**. It loads the 15 matched patients and rebuilds "
        "Stages 2–5A so that the Risk Head receives real `xP` and real `CA`, rather than random test tensors."
    )
    st.code("""real gene .npy → GeneBranch → xP (15,186,256)
real WSI .npz → WSIBranch → WSIReduction → xI (15,162,256)
                         xP + xI → Fusion → CA (15,186,256)
                         xP + CA → RiskHead → R (15,1)""", language="text")

    with st.expander("7.1 — Check: real inputs and shapes", expanded=False):
        st.code("""assert xP.shape == (15,186,256)
assert CA.shape == (15,186,256)""", language="python")
        st.write("Confirms the Risk Head receives the expected inputs.")

    with st.expander("7.2 — Check: exact parameter count", expanded=False):
        st.code("""n_params = sum(p.numel() for p in risk_head.parameters())
assert n_params == 373""", language="python")
        st.write("Confirms the expected architecture: 372 weights + 1 bias.")

    with st.expander("7.3 — Check: intermediate shapes", expanded=False):
        st.code("""gene_pooled = risk_head.gap_gene(xP)
fusion_pooled = risk_head.gap_fusion(CA)
assert gene_pooled.shape == (15,186,1)
assert fusion_pooled.shape == (15,186,1)

fused = torch.cat([gene_pooled, fusion_pooled], dim=1).squeeze(-1)
assert fused.shape == (15,372)""", language="python")
        st.write("Checks the internal pooling and concatenation steps before prediction.")

    with st.expander("7.4 — Check: final output and finite values", expanded=False):
        st.code("""R = risk_head(xP, CA)
assert R.shape == (15,1)
assert torch.isfinite(R).all().item()""", language="python")
        st.write("Confirms one valid finite risk score per patient.")

    with st.expander("7.5 — Check: gradient flow through the whole network", expanded=True):
        st.code("""loss = R.sum()
loss.backward()""", language="python")
        st.code("""R
↑
RiskHead
↑
├── xP → GeneBranch
│
└── CA → Fusion → WSIReduction → WSIBranch""", language="text")
        st.write(
            "The test checks that every trainable parameter in RiskHead, Fusion, WSIReduction, "
            "WSIBranch and GeneBranch has a finite, non-zero gradient. This confirms the final "
            "prediction is connected to both modalities for learning."
        )

    st.markdown("## 8. Where does R go next?")
    st.code("""R = (15,1)
   ↓
Stage 6 — SurvivalLoss
   ├── L1: Cox survival loss
   ├── L2: weight regularization
   └── L3: contrastive loss from Stage 4
             ↓
        total loss → backward() → model update""", language="text")
    st.write("The Risk Head itself does not write the CSV during its forward pass. Stage 6 consumes R during training.")

    st.markdown("## 9. Where is the risk output saved?")
    st.info("Important: **Stage 5B creates R in memory.** The saved risk-score CSVs are produced later by the training/evaluation workflow.")
    st.code("""After final evaluation:
data/processed/training/final_risk_scores.csv

After best-checkpoint evaluation:
data/processed/training/best_risk_scores.csv""", language="text")
    st.write(
        "The dashboard prefers `best_risk_scores.csv` when it exists and otherwise uses `final_risk_scores.csv`. "
        "The saved table contains patient-level risk scores alongside the survival information used for analysis."
    )

    st.markdown("### Model weights are saved separately")
    st.code("""data/processed/model_checkpoints/best_model.pt
data/processed/model_checkpoints/final_model.pt""", language="text")
    st.write("These checkpoint files contain the learned parameters, including the Risk Head weights. The CSVs contain evaluated patient risk scores.")

    st.markdown("## 10. Complete Stage 5B handoff")
    st.code("""CREATED EARLIER
GeneBranch → xP = (15,186,256)
Stage 5A Fusion → CA = (15,186,256)

CONSUMED BY
SurvivalRiskHead.forward(xP, CA)
        ↓
pool both → concatenate → Linear(372,1)
        ↓
R = (15,1)

NEXT CONSUMER
Stage 6 SurvivalLoss

LATER SAVED / EVALUATED
best_risk_scores.csv
final_risk_scores.csv

MODEL PARAMETERS
best_model.pt
final_model.pt""", language="text")

    st.markdown("## ⭐ Presentation-ready summary")
    st.success(
        "Stage 5B takes xP from the gene branch and CA from pathway-to-patch fusion. "
        "Both are (15,186,256). It averages the 256 features of every pathway down to one value, "
        "giving (15,186,1) for each input. These are concatenated into 372 values per patient, "
        "then Linear(372,1) produces R = (15,1), one risk score per patient. The test verifies "
        "the real-data inputs, every intermediate shape, the exact 373 parameters, finite output, "
        "and gradient flow through the complete upstream network."
    )
    st.code("""MEMORIZE
xP (15,186,256) ──► Pool ──┐
                           ├─► (15,372) ─► Linear(372→1) ─► R (15,1)
CA (15,186,256) ──► Pool ──┘
                           ↓
                    SurvivalLoss""", language="text")

def render_survival_loss():
    st.header("Stage 6 — SurvivalLoss")

    st.markdown("### 1. What is SurvivalLoss?")
    st.write(
        "Stage 6 is the training objective that converts the model's patient-level "
        "risk scores into a single scalar loss. It combines three things: "
        "the Cox survival signal (L1), weight regularization (L2), and the "
        "gene↔WSI contrastive signal from Stage 4 (L3)."
    )
    st.info(
        "Simple idea: L1 teaches the model who should have higher risk; "
        "L2 controls large weights; L3 keeps the two modalities aligned."
    )

    st.markdown("### 2. Complete Stage 6 input map")
    st.code("""
STAGE 5B                         CLINICAL FILE
R = (15,1)                       OS_MONTHS → T = (15,)
one risk score / patient        OS_STATUS → S = (15,)
       │                                │
       ├────────────────┐               │
       │                │               │
STAGE 4                │               │
L3 = scalar ───────────┼───────────────┤
                        ▼               ▼
                 SurvivalLoss.forward()
                        │
                        ▼
             total = L1 + α·wd·L2 + β·L3
    """, language="text")

    metric_row([
        ("Risk R", "(15,1)", "Stage 5B RiskHead"),
        ("T", "(15,)", "OS_MONTHS / survival time"),
        ("S", "(15,)", "OS_STATUS / event indicator"),
        ("L3", "scalar", "Stage 4 contrastive loss"),
    ])

    st.write(
        "The patient order must match across R, T and S. Here T is survival time "
        "in months and S is 1 for deceased/event patients and 0 for living/censored patients."
    )

    st.markdown("### 3. Where the clinical labels come from")
    st.code(
        "data/raw/blca_tcga_pan_can_atlas_2018/data_clinical_patient.txt\n\n"
        "PATIENT_ID   OS_STATUS   OS_MONTHS\n"
        "     │            │          │\n"
        "     └────────────┴──────────┴──► load_os_labels()\n"
        "                                      ↓\n"
        "                              T = OS_MONTHS (15,)\n"
        "                              S = OS_STATUS (15,)",
        language="text",
    )
    st.write(
        "`load_os_labels()` matches the requested TCGA patient IDs in the clinical table "
        "and returns the survival labels used by the loss."
    )

    st.markdown("### 4. L1 — Cox negative partial log-likelihood")
    st.write(
        "L1 is the actual survival-learning part. For an event patient i, the Cox objective "
        "compares that patient's risk with patients whose observed survival time is at least "
        "as large as T[i]. Higher predicted risk is therefore encouraged for patients who "
        "experience the event earlier."
    )
    st.code("""
R = (15,1)
   ↓ squeeze(-1)
θ = (15,)

Risk-set matrix R_mat:
R_mat[i,j] = 1  if T[j] >= T[i]
             0  otherwise

R_mat = (15,15)
row i → risk set for patient i
col j → whether patient j belongs to that set

log_risk[i] = logsumexp(θ[j] for j in risk set i)

L1 = -mean((θ - log_risk) * S)
    """, language="text")

    st.markdown("#### Why is `R_mat` 15 × 15?")
    st.write(
        "There are 15 patients, so every patient gets one risk-set row and every possible "
        "patient is a column. The diagonal is always 1 because T[i] >= T[i]. The patient "
        "with the largest observed T has only itself in its risk set."
    )
    st.code("""
T[j] >= T[i]  → included in row i
T[j] <  T[i]  → excluded from row i

Diagonal: all 1s ✓
Largest-T patient: self-only risk set ✓
    """, language="text")

    st.markdown("#### Why `logsumexp`?")
    st.write(
        "The Cox denominator contains a sum of exponentiated risk scores. The implementation "
        "masks patients outside the risk set with `-inf` and uses `logsumexp`, which computes "
        "the same log-sum in a numerically stable way."
    )

    st.markdown("### 5. L2 — weight regularization")
    st.write(
        "L2 penalizes large trainable weights. This project intentionally uses the paper-literal "
        "p=2 norm and includes only trainable parameters whose names contain `weight`. Biases are excluded."
    )
    st.code("""
L2_raw = Σ ||parameter||₂

Included modules:
  GeneBranch
  WSIBranch
  WSIReduction
  Fusion
  SurvivalRiskHead

weight parameters → included ✓
bias parameters   → excluded ✗

L2_weighted = α × weight_decay × L2_raw
α = 1.0
weight_decay = 5×10⁻⁴
    """, language="text")
    st.warning(
        "Important project decision: Stage 6 uses p=2 weight regularization. "
        "This intentionally differs from the released code's p=1 regularization."
    )

    st.markdown("### 6. L3 — passed from Stage 4, not recomputed")
    st.code("""
Stage 4
contrastive alignment
       ↓
L3 = one scalar
       ↓
Stage 6
L3_weighted = β × L3
β = 0.8
    """, language="text")
    st.write(
        "Stage 6 receives the already-computed Stage 4 L3 scalar. It does not run the "
        "contrastive calculation again. In the verified Stage-4 run, the reference L3 was "
        "approximately 5.2250147."
    )

    st.markdown("### 7. Final objective")
    st.latex(r"L_{total}=L_1+\alpha\cdot wd\cdot L_2+\beta\cdot L_3")
    metric_row([
        ("α", "1.0"),
        ("weight decay", "5×10⁻⁴"),
        ("β", "0.8"),
        ("Batch", "all 15 patients"),
    ])
    st.code("""
TOTAL LOSS
   = L1
   + 1.0 × 0.0005 × L2_raw
   + 0.8 × L3
    """, language="text")

    st.markdown("### 8. `survival_loss.py` — function-by-function")
    st.code("""
SurvivalLabels
   → stores T and S

load_os_labels(...)
   → reads clinical TXT and returns T, S

cox_neg_log_likelihood(risk, time, event)
   → builds R_mat → log-risk → L1

l2_weight_regularization(modules)
   → sums p=2 norms of weight parameters only

SurvivalLoss.forward(...)
   → L1 + weighted L2 + weighted L3
   → returns total_loss + diagnostics
    """, language="text")
    show_code(PATHS["survival_loss"], [
        r"class SurvivalLabels",
        r"def load_os_labels",
        r"def cox_neg_log_likelihood",
        r"def l2_weight_regularization",
        r"class SurvivalLoss",
        r"def forward",
    ])

    st.markdown("### 9. Exact `SurvivalLoss.forward()` contract")
    st.code("""
forward(
    risk,             # R: (15,1)
    time,             # T: (15,)
    event,            # S: (15,)
    l3_loss,          # Stage-4 scalar
    modules_for_l2    # Gene/WSI/Fusion/RiskHead modules
)

RETURNS
  total_loss       → scalar used for backward()
  diagnostics      → L1, L2_raw, L2_weighted,
                      L3, L3_weighted, total
    """, language="text")

    st.markdown("### 10. Real-data test: `test_survival_loss.py`")
    st.write(
        "The test rebuilds the real 15-patient upstream pipeline instead of using fake tensors, "
        "then sends the resulting R, T, S and L3 into SurvivalLoss. It checks both numerical "
        "correctness and gradient flow."
    )
    st.code("""
✓ 15 real patient IDs match gene / WSI / clinical data
✓ T = (15,), S = (15,), R = (15,1)
✓ gene input = (15,186,4942)
✓ WSI input = (15,162,384)
✓ patch counts and padding mask preserved
✓ GeneBranch → WSIBranch → Reduction → Fusion → RiskHead rebuilt
✓ R_mat = (15,15)
✓ R_mat diagonal = all 1
✓ largest-T patient has only itself in risk set
✓ L1, L2_raw, L2_weighted, L3_weighted, total are finite
✓ L3_weighted = 0.8 × L3
✓ L2 excludes bias parameters
✓ total_loss.backward() gives finite, non-zero gradients
  through GeneBranch, WSIBranch, WSIReduction, Fusion and RiskHead
✓ all-censored edge case: L1 = 0
    """, language="text")

    st.markdown("### 11. What is actually flowing through the whole model?")
    st.code("""
GENE (15,186,4942) ─► GeneBranch ─► xP (15,186,256) ─────┐
                                                          │
WSI (15,162,384) ─► WSIBranch ─► Reduction ─► xI ───────┤
                                                          │
                         ┌── Stage 4 ──► L3 ─────────────┤
                         │                                │
                         └── Stage 5A ─► CA ─► RiskHead ─► R (15,1)
                                                           │
Clinical ─► T,S ───────────────────────────────────────────┤
                                                           ▼
                                                    SurvivalLoss
                                                           │
                                                  total_loss scalar
                                                           │
                                                        backward()
                                                           │
                                                       optimizer
    """, language="text")

    st.markdown("### 12. Output and next consumer")
    st.code("""
OUTPUT
  total_loss  → one scalar training objective
  diagnostics → component values for logging / inspection

NEXT CONSUMER
  scripts/train.py
      ↓
  total_loss.backward()
      ↓
  optimizer.step()
      ↓
  scheduler.step()
    """, language="text")
    st.info(
        "Stage 6 itself does not save the final risk CSV. Risk scores are written later "
        "during Stage 7 training/evaluation."
    )

    st.markdown("### 13. Saved artifacts later in the pipeline")
    st.code("""
data/processed/training/training_history.csv
data/processed/training/final_risk_scores.csv
data/processed/training/best_risk_scores.csv

data/processed/model_checkpoints/best_model.pt
data/processed/model_checkpoints/final_model.pt
    """, language="text")

    st.markdown("### 14. Presentation-ready summary")
    st.success(
        "Stage 6 receives R from Stage 5B, T/S from the clinical table, and L3 from Stage 4. "
        "L1 uses Cox risk sets to learn survival-risk ordering; L2 adds p=2 weight regularization "
        "with biases excluded; L3 preserves multimodal alignment. The three are combined as "
        "Ltotal = L1 + α·wd·L2 + β·L3, then the scalar is backpropagated through the entire "
        "GeneBranch → WSI branch → Fusion → RiskHead network."
    )
    st.code("""
MEMORIZE
R (15,1) + T (15,) + S (15,) + L3 (scalar)
                    ↓
               SurvivalLoss
                    ↓
        L1 + 0.0005·L2 + 0.8·L3
                    ↓
              total_loss
                    ↓
               backward()
    """, language="text")

def render_train():
    st.header("Stage 7 — Full-batch training")

    st.markdown("### 7.1 — What is training doing?")
    st.write(
        "Stages 1–5B define the model and Stage 6 defines how its predictions are scored. "
        "Training is the repeated process that changes the model's learnable parameters so "
        "the total SurvivalLoss becomes smaller. Each epoch follows: **forward → loss → "
        "backward → optimizer update → evaluation → logging/checkpointing**."
    )

    st.code("""
ALL 15 PATIENTS
      │
      ├── Gene input       (15,186,4942)
      ├── WSI input        (15,162,384) + padding mask
      └── Survival labels  T (15,), S (15,)
                │
                ▼
        forward_pipeline()
                │
        ┌───────┴──────────────────────────────┐
        │ GeneBranch → xP      (15,186,256)    │
        │ WSIBranch → xI_384   (15,162,384)    │
        │ Reduction → xI       (15,162,256)    │
        │ Fusion → CA          (15,186,256)    │
        │ RiskHead → R         (15,1)           │
        │ Contrastive → L3     scalar          │
        └───────┬──────────────────────────────┘
                │
                ▼
          SurvivalLoss
                │
                ▼
          total loss scalar
                │
          backward()
                │
          parameter gradients
                │
          AdamW optimizer
                │
       CosineAnnealingLR
                │
                ▼
          next epoch
    """, language="text")

    st.markdown("### 7.2 — Where the inputs come from")
    st.write(
        "Training does not create a new kind of biological input. It reuses the outputs "
        "already produced by the earlier stages, plus the clinical survival labels needed "
        "by SurvivalLoss. The implementation uses the same 15 matched patients throughout."
    )
    st.dataframe(pd.DataFrame([
        {"Input": "Gene batch", "Shape": "(15,186,4942)", "Created by": "Gene preprocessing / saved .npy", "Consumed by": "GeneBranch"},
        {"Input": "WSI batch", "Shape": "(15,162,384)", "Created by": "DINO patch embeddings + padding", "Consumed by": "WSIBranch"},
        {"Input": "WSI padding mask", "Shape": "(15,162)", "Created by": "build_batch()", "Consumed by": "WSIBranch / attention"},
        {"Input": "T", "Shape": "(15,)", "Created by": "clinical file: OS_MONTHS", "Consumed by": "SurvivalLoss / C-index"},
        {"Input": "S", "Shape": "(15,)", "Created by": "clinical file: OS_STATUS", "Consumed by": "SurvivalLoss / C-index"},
        {"Input": "L3", "Shape": "scalar", "Created by": "contrastive loss during forward", "Consumed by": "SurvivalLoss"},
    ],), hide_index=True, width="stretch")

    st.markdown("### 7.3 — Why are all 15 patients one batch?")
    st.write(
        "This is intentional. The Cox partial-likelihood loss compares each patient's risk "
        "with the risks of patients in that patient's risk set. If patients were split into "
        "independent mini-batches, those patient-to-patient comparisons would be incomplete. "
        "Therefore this mini-scale implementation uses one full batch containing all 15 patients."
    )
    st.info("Presentation sentence: **‘We train on all 15 patients as one batch because the Cox loss requires the patients to be compared through shared risk sets.’**")

    st.markdown("### 7.4 — `scripts/train.py`: data loading")
    st.write(
        "`main()` first prepares the matched cohort. `load_matched_patients()` keeps the gene "
        "and WSI files aligned by patient ID and checks the expected shapes/order. "
        "`build_batch()` stacks the gene arrays and pads variable-length WSI patch sequences "
        "to the largest patch count (162), creating a boolean mask where `True` means padding. "
        "`load_os_labels()` reads OS_MONTHS and OS_STATUS from the clinical file."
    )
    st.code("""
load_matched_patients()
        ↓
build_batch()
        ↓
Gene  → (15,186,4942)
WSI   → (15,162,384)
Mask  → (15,162)
        ↓
load_os_labels()
        ↓
T → (15,),  S → (15,)
    """, language="text")

    st.markdown("### 7.5 — Model construction")
    st.write(
        "`build_modules()` creates the five trainable parts used in the forward path: "
        "GeneBranch, WSIBranch, WSIReduction, PathwayToPatchFusion, and RiskHead. "
        "The contrastive-loss object is used to compute L3 but has no learnable parameters."
    )
    st.code("""
GeneBranch
WSIBranch
WSIReduction
PathwayToPatchFusion
RiskHead
ContrastiveLoss  ← no trainable parameters
    """, language="text")

    st.markdown("### 7.6 — One forward pass")
    st.write("`forward_pipeline()` runs the complete Stage 1–5B model once and returns only the two quantities needed by Stage 6: the predicted risk scores `R` and the contrastive loss `L3`. Intermediate tensors are passed between modules as follows:")
    st.code("""
Gene:      (15,186,4942) ──► GeneBranch ─────────► xP (15,186,256)
WSI:       (15,162,384)  ──► WSIBranch ──────────► xI_384 (15,162,384)
                                      │
                                      ▼
                               WSIReduction
                                      │
                                      ▼
                                  xI (15,162,256)
                                      │
                         xP + xI ─────┴─────► Fusion
                                              │
                                              ▼
                                         CA (15,186,256)
                                              │
                               xP + CA ───────┴────► RiskHead
                                                       │
                                                       ▼
                                                   R (15,1)

                         xP + xI ───────────────► ContrastiveLoss
                                                       │
                                                       ▼
                                                   L3 (scalar)
    """, language="text")
    st.write("Then Stage 6 receives `R`, `T`, `S`, and the already-computed `L3` and produces the total loss.")

    st.markdown("### 7.7 — Loss + optimizer configuration")
    st.dataframe(pd.DataFrame([
        {"Component": "SurvivalLoss", "Setting": "alpha=1, weight_decay=5e-4, beta=0.8", "Purpose": "Combines Cox L1 + p=2 weight regularization + L3"},
        {"Component": "Optimizer", "Setting": "AdamW", "Purpose": "Updates all trainable model parameters from gradients"},
        {"Component": "Initial learning rate", "Setting": "1e-3", "Purpose": "Step size at the beginning of training"},
        {"Component": "Optimizer weight_decay", "Setting": "0", "Purpose": "Avoids applying a second L2 penalty because Stage 6 already adds it manually"},
        {"Component": "Scheduler", "Setting": "CosineAnnealingLR", "Purpose": "Gradually decreases learning rate toward 0 over 500 epochs"},
        {"Component": "Seed", "Setting": "42", "Purpose": "Reproducibility"},
        {"Component": "Epochs", "Setting": "500", "Purpose": "Number of complete full-batch update cycles"},
    ]), hide_index=True, width="stretch")

    st.markdown("### 7.8 — What happens inside every epoch?")
    st.code("""
for epoch in range(1, 501):

    1. model.train()
       → dropout is active

    2. optimizer.zero_grad()
       → clear gradients from the previous epoch

    3. R, L3 = forward_pipeline(...)
       → compute current predictions and L3

    4. total_loss = SurvivalLoss(R, T, S, L3, ...)
       → compute L1 + weighted L2 + weighted L3

    5. total_loss.backward()
       → backpropagate error through the whole model
       → calculate gradients for trainable parameters

    6. optimizer.step()
       → AdamW updates the parameters using those gradients

    7. scheduler.step()
       → update the learning rate for the next epoch

    8. model.eval() + torch.no_grad()
       → run the same patients with dropout disabled
       → calculate deterministic eval loss + exploratory C-index

    9. append one row to training_history.csv

   10. if training loss is the lowest so far:
       → save best_model.pt
    """, language="text")

    st.markdown("### 7.9 — Train mode vs evaluation mode")
    st.write(
        "During the training forward pass, modules are in `train()` mode, so dropout layers "
        "are active. After the optimizer update, the code switches to `eval()` and wraps the "
        "forward pass in `torch.no_grad()`. This makes the recorded evaluation loss deterministic "
        "for the current weights and avoids storing gradients for that pass."
    )

    st.markdown("### 7.10 — What exactly is being optimized?")
    st.code("""
Total loss
    = L1 (Cox survival loss)
    + alpha × weight_decay × L2_raw
    + beta × L3

              ↓ backward()

GeneBranch ───────────────┐
WSIBranch ────────────────┤
WSIReduction ─────────────┤──► gradients ─► AdamW ─► new parameters
Fusion ───────────────────┤
RiskHead ─────────────────┘

ContrastiveLoss itself has no learnable parameters.
    """, language="text")
    st.write(
        "So the optimizer does not ‘optimize the CSV’ or the risk scores directly. It changes "
        "the learnable weights inside the five model modules; those changed weights produce new "
        "risk scores on the next forward pass."
    )

    st.markdown("### 7.11 — Learning-rate schedule")
    st.write(
        "`CosineAnnealingLR` changes the learning rate over the 500 epochs. The run starts at "
        "`1e-3` and the scheduler decreases it toward 0. This is separate from the loss: the "
        "loss says **how wrong the model is**, while the learning rate controls **how large the "
        "parameter update can be**."
    )

    st.markdown("### 7.12 — Best checkpoint: what does ‘best’ mean?")
    st.warning(
        "In this implementation, the best checkpoint is the epoch with the **lowest training-mode total loss**. "
        "It is NOT selected using a held-out validation set or validation C-index, because this revised mini-scale "
        "run uses all 15 patients for training."
    )
    st.code("""
if total_loss.item() < best_total_loss:
    best_total_loss = total_loss.item()
    save_checkpoint(...)
    """, language="python")
    st.write("`best_model.pt` stores the epoch/loss metadata plus the state dictionaries for GeneBranch, WSIBranch, WSIReduction, Fusion, and RiskHead.")

    st.markdown("### 7.13 — What is recorded in `training_history.csv`?")
    st.write("One row is written per epoch, so the verified 500-epoch run produces 500 training records. The history tracks the learning rate, training/evaluation loss components, total losses, and evaluation C-index so the run can be inspected after training.")
    st.code("""
epoch
lr
train_L1, train_L2_raw, train_L2_weighted, train_L3, train_L3_weighted, train_total_loss

eval_L1, eval_L2_raw, eval_L2_weighted, eval_L3, eval_L3_weighted, eval_total_loss
eval_C_index
    """, language="text")

    st.markdown("### 7.14 — `evaluate_best.py`: what happens after training?")
    st.write(
        "`evaluate_best.py` does **not** retrain the network. It loads `best_model.pt`, reconstructs "
        "the same model components, restores their saved state dictionaries, switches them to `eval()` "
        "mode, and recomputes the risk score and loss for the 15 patients. It then writes the best-checkpoint "
        "risk scores to `best_risk_scores.csv`."
    )
    st.code("""
scripts/evaluate_best.py
        │
        ├── load best_model.pt
        ├── rebuild model modules
        ├── load saved state_dicts
        ├── eval() + no_grad()
        ├── forward_pipeline()
        ├── SurvivalLoss + C-index
        └── save best_risk_scores.csv
    """, language="text")

    st.markdown("### 7.15 — Files involved in Stage 7")
    st.code("""
scripts/train.py
    → main training loop
    → data loading / batch construction
    → model construction
    → forward pass
    → optimizer + scheduler
    → checkpoint saving

scripts/evaluate_best.py
    → loads best checkpoint
    → evaluates it
    → saves best risk scores

survival_loss.py
    → provides Stage 6 total loss used by training

src/c_index.py / C-index implementation used by the project
    → evaluation metric
    """, language="text")

    st.markdown("### 7.16 — Saved artifacts")
    st.code("""
data/processed/training/training_history.csv
    → complete epoch-by-epoch history

data/processed/training/final_risk_scores.csv
    → risk scores from the final epoch-500 model

data/processed/training/best_risk_scores.csv
    → risk scores from the best checkpoint after evaluation

data/processed/model_checkpoints/best_model.pt
    → lowest training-loss checkpoint

data/processed/model_checkpoints/final_model.pt
    → epoch-500 checkpoint
    """, language="text")

    st.markdown("### 7.17 — Verified 500-epoch run")
    metric_row([
        ("Patients", "15"),
        ("Epochs", "500"),
        ("Best epoch", "467"),
        ("Best train total loss", "2.253419"),
        ("Final eval total loss", "7.749714"),
        ("Final eval C-index", "1.000000*"),
    ])
    st.caption(
        "*The C-index is exploratory: the same 15 patients were used for training and evaluation, "
        "so this number is not a held-out validation/test generalization result."
    )

    st.markdown("### 7.18 — Presentation-ready summary")
    st.info(
        "**Input:** all 15 matched patients — gene `(15,186,4942)`, padded WSI `(15,162,384)` + mask, "
        "and clinical `T,S`. **Process:** run the complete Stage 1–5B forward path, compute `R` and `L3`, "
        "calculate Stage 6 total loss, backpropagate, update the five trainable modules with AdamW, and "
        "repeat for 500 epochs while cosine-decaying the learning rate. After each update, evaluate the "
        "same 15 patients deterministically, log the metrics, and save a checkpoint whenever the training "
        "total loss reaches a new minimum. **Output:** trained checkpoints, training history, and risk-score CSVs."
    )

    st.markdown("### 7.19 — Code: `scripts/train.py`")
    show_code(PATHS["train"], [
        r"def load_matched_patients",
        r"def build_batch",
        r"def load_os_labels",
        r"def build_modules",
        r"def forward_pipeline",
        r"def save_checkpoint",
        r"def main",
        r"for epoch in range",
        r"best_total_loss",
    ])

    st.markdown("### Key hyperparameters")
    metric_row([
        ("Epochs", "500"),
        ("Optimizer", "AdamW"),
        ("Initial LR", "1e-3"),
        ("AdamW weight decay", "0"),
        ("Manual L2 weight", "5e-4"),
        ("L3 weight (beta)", "0.8"),
        ("Seed", "42"),
        ("Batch", "15 patients"),
    ])


def render_evaluate():
    st.header("Stage 7 — Best checkpoint evaluation")

    st.markdown("### 7.1 — What is this stage?")
    st.write(
        "`evaluate_best.py` is the **post-training evaluation step**. Training has already learned the model parameters and saved checkpoints. "
        "This script does not train anything again; it takes the checkpoint selected as best, reconstructs the same model, restores its learned weights, "
        "runs the 15-patient cohort through the model in deterministic evaluation mode, and saves the resulting patient-level risk scores."
    )

    st.code("""
TRAINING (train.py)
      │
      ├── 500 epochs
      ├── updates model weights
      └── saves best_model.pt
                │
                ▼
       evaluate_best.py
                │
                ├── load checkpoint
                ├── rebuild same model
                ├── restore learned weights
                ├── eval() + no_grad()
                ├── forward pass
                ├── compute loss + C-index
                └── save risk scores
    """, language="text")

    st.markdown("### 7.2 — What does it take as input?")
    st.code("""
INPUT 1 — BEST CHECKPOINT
best_model.pt
→ learned parameters from the epoch with the lowest training total loss

INPUT 2 — SAME 15-PATIENT DATA
Gene batch  → (15,186,4942)
WSI batch   → (15,162,384) + padding mask
T           → (15,)   survival / follow-up time
S           → (15,)   event indicator

INPUT 3 — SAME MODEL DEFINITION
GeneBranch + WSIBranch + WSIReduction + Fusion + RiskHead
    """, language="text")
    st.write(
        "The checkpoint supplies the **learned weights**; the patient data is loaded separately. "
        "The model architecture must be rebuilt so those saved weights have the correct layers and tensor shapes."
    )

    st.markdown("### 7.3 — Which checkpoint is selected?")
    st.write(
        "During `train.py`, `best_model.pt` is overwritten whenever the current **training-mode total loss** is lower than the previous best. "
        "Therefore, in this project, 'best' means lowest training total loss — it does **not** mean highest validation C-index, because there is no held-out validation cohort."
    )
    st.code("""
best_total_loss = +infinity

for each epoch:
    train_total_loss = ...

    if train_total_loss < best_total_loss:
        best_total_loss = train_total_loss
        save_checkpoint(...)
    """, language="python")
    st.info(
        "Verified run: the best checkpoint came from epoch 467, with training total loss ≈ 2.253419."
    )

    st.markdown("### 7.4 — Step 1: load `best_model.pt`")
    st.write(
        "`torch.load(...)` reads the saved checkpoint from `data/processed/model_checkpoints/best_model.pt`. "
        "The checkpoint contains the saved state dictionaries for the trainable modules."
    )
    st.code("""
checkpoint = torch.load(BEST_CHECKPOINT, map_location=device)
""", language="python")
    st.write(
        "A **state_dict** is essentially a collection of the model's learned parameter tensors. "
        "It lets us restore exactly what the network learned at the selected epoch."
    )

    st.markdown("### 7.5 — Step 2: rebuild the same five trainable modules")
    st.code("""
build_modules()
      │
      ├── GeneBranch
      ├── WSIBranch
      ├── WSIReduction
      ├── Fusion
      └── SurvivalRiskHead
    """, language="text")
    st.write(
        "This recreates the architecture used during training. The modules are initially new objects; their parameters become the trained parameters only after the checkpoint state dictionaries are loaded."
    )

    st.markdown("### 7.6 — Step 3: restore the learned weights")
    st.code("""
model["gene_branch"].load_state_dict(...)
model["wsi_branch"].load_state_dict(...)
model["wsi_reduction"].load_state_dict(...)
model["fusion"].load_state_dict(...)
model["risk_head"].load_state_dict(...)
    """, language="python")
    st.write(
        "After `load_state_dict()`, the rebuilt network now has the parameter values learned during the selected training epoch. "
        "No parameter update happens in this stage."
    )

    st.markdown("### 7.7 — Step 4: switch to evaluation mode")
    st.code("""
for module in modules:
    module.eval()
    """, language="python")
    st.write(
        "`eval()` changes modules such as Dropout to their inference behaviour. Dropout is therefore disabled, making the evaluation forward pass deterministic for the same inputs and weights."
    )

    st.markdown("### 7.8 — Step 5: disable gradient tracking")
    st.code("""
with torch.no_grad():
    R, L3 = forward_pipeline(...)
    """, language="python")
    st.write(
        "Evaluation does not need `backward()`, gradients, an optimizer step, or a learning-rate update. `torch.no_grad()` prevents PyTorch from building the gradient graph, reducing unnecessary memory and computation."
    )

    st.markdown("### 7.9 — Step 6: run the complete forward pipeline")
    st.code("""
Gene input (15,186,4942)
        │
        ▼
   GeneBranch
        │
        ▼
xP (15,186,256)

WSI input (15,162,384) + mask
        │
        ▼
   WSIBranch
        │
        ▼
(15,162,384)
        │
        ▼
 WSIReduction
        │
        ▼
xI (15,162,256)

xP + xI
   │
   ▼
Fusion
   │
   ▼
CA (15,186,256)
   │
   ▼
RiskHead(xP, CA)
   │
   ▼
R (15,1)
    """, language="text")
    st.write(
        "The important output here is `R`: one scalar risk score for each of the 15 patients. The Stage 4 contrastive branch also produces `L3`, which is used when the Stage 6 total loss is recomputed."
    )

    st.markdown("### 7.10 — Step 7: recompute SurvivalLoss")
    st.code("""
R (15,1) + T (15,) + S (15,) + L3
                    │
                    ▼
             SurvivalLoss
                    │
                    ▼
        L1 + L2 + weighted L3
                    │
                    ▼
              total loss
    """, language="text")
    st.write(
        "This is a fresh evaluation of the Stage 6 objective using the fixed best-checkpoint weights. It measures how the selected checkpoint performs on the same 15-patient cohort."
    )
    st.caption(
        "No `backward()` or optimizer update occurs here — the loss is measured, not used to change the model."
    )

    st.markdown("### 7.11 — Step 8: calculate C-index")
    st.write(
        "The C-index evaluates the ordering of predicted risk against observed survival outcomes. In simple terms: if one patient has a shorter observed survival time than another comparable patient, the model should assign the shorter-survival patient a higher risk score."
    )
    st.code("""
Risk scores R + survival times T + event indicators S
                         │
                         ▼
                     C-index
                         │
                         ▼
              how well risk ordering agrees
              with observed survival ordering
    """, language="text")
    st.warning(
        "Your C-index = 1.0 is exploratory, not a held-out validation/test result, because the same 15 patients were used for training and evaluation."
    )

    st.markdown("### 7.12 — Step 9: save one risk score per patient")
    st.code("""
R (15,1)
   │
   ▼
patient_id + OS_MONTHS + OS_STATUS + risk_score
   │
   ▼
best_risk_scores.csv
    """, language="text")
    st.write(
        "This CSV is the main patient-level output of `evaluate_best.py`. It makes the trained model's predictions easy to inspect, rank, plot, and use in the dashboard."
    )

    st.markdown("### 7.13 — What is the difference between `best_model.pt` and `final_model.pt`?")
    st.code("""
best_model.pt
→ checkpoint saved at the epoch with the lowest training total loss
→ selected model for `evaluate_best.py`

final_model.pt
→ parameters after the last training epoch (epoch 500)
→ represents the final state of the training loop
    """, language="text")
    st.write(
        "These can be different. In your run, the best checkpoint was epoch 467, while the training continued through epoch 500. Therefore the best checkpoint and final checkpoint are not the same model state."
    )

    st.markdown("### 7.14 — Code: what should I look at in `scripts/evaluate_best.py`?")
    show_code(PATHS["evaluate"], [
        r"def main",
        r"torch.load",
        r"load_state_dict",
        r"forward_pipeline",
        r"BEST_RISK_CSV",
    ])
    st.write(
        "The key functions/operations to explain in a presentation are: `main()` controls the evaluation workflow; `torch.load()` reads the checkpoint; `load_state_dict()` restores learned parameters; `forward_pipeline()` generates `R` and `L3`; and the risk-score output is written to `BEST_RISK_CSV`."
    )

    st.markdown("### 7.15 — Complete input → process → output map")
    st.code("""
INPUT
├── best_model.pt
├── gene batch (15,186,4942)
├── WSI batch (15,162,384) + mask
└── survival labels T,S

PROCESS
├── rebuild same architecture
├── restore checkpoint weights
├── eval() → dropout off
├── no_grad() → no gradient graph
├── forward → R (15,1) and L3
├── SurvivalLoss → total loss
└── C-index → risk-ranking metric

OUTPUT
├── best_risk_scores.csv
├── evaluated total loss
└── evaluated C-index
    """, language="text")

    st.markdown("### 7.16 — Saved artifact and next consumer")
    st.code("""
scripts/evaluate_best.py
        │
        ▼
data/processed/model_checkpoints/best_model.pt
        │
        ▼
  restored best model
        │
        ▼
R (15,1) + metrics
        │
        ▼
data/processed/training/best_risk_scores.csv
        │
        ├── dashboard risk table
        ├── risk ranking
        └── result visualizations / inspection
    """, language="text")

    st.markdown("### 7.17 — Verified best-checkpoint evaluation")
    metric_row([
        ("Checkpoint", "epoch 467"),
        ("Best train loss", "2.253419"),
        ("Evaluated risk shape", "(15,1)"),
        ("Eval total loss", "7.744978"),
        ("C-index", "1.000000*"),
    ])
    st.caption(
        "*Same 15 patients were used for training and evaluation; therefore the C-index is mini-scale/exploratory and should not be presented as generalization performance."
    )

    st.markdown("### 7.18 — Presentation summary")
    st.success(
        "`evaluate_best.py` takes the checkpoint selected during training, rebuilds the same PAMT modules, restores the learned weights, switches the model to deterministic evaluation mode, and runs the 15 patients through the complete pipeline without gradient updates. It then recomputes SurvivalLoss and C-index and saves one predicted risk score per patient in `best_risk_scores.csv`."
    )

    st.code(
        "data/processed/training/best_risk_scores.csv",
        language="text",
    )


def render_results():
    st.header("Stage 7 — Results & figures")

    history_path = TRAIN_DIR / "training_history.csv"
    risk_path = TRAIN_DIR / "final_risk_scores.csv"
    best_risk_path = TRAIN_DIR / "best_risk_scores.csv"

    st.markdown("### Results generator")
    show_code(PATHS["results"], [
        r"training_history.csv",
        r"training_vs_eval_loss",
        r"eval_c_index",
        r"stage7_checkpoint_summary",
        r"stage7_risk_score_table",
    ])

    if history_path.exists():
        df = pd.read_csv(history_path)
        st.markdown("### Training history")
        st.dataframe(df.tail(20), width="stretch", hide_index=True)

        st.markdown("### Training vs deterministic evaluation loss")
        st.line_chart(
            df.set_index("epoch")[["train_total_loss", "eval_total_loss"]]
        )

        st.markdown("### Evaluation C-index")
        st.line_chart(df.set_index("epoch")[["eval_C_index"]])

        best = df.loc[df["train_total_loss"].idxmin()]
        final = df.iloc[-1]

        st.markdown("### Best vs final")
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
        st.dataframe(summary, width="stretch", hide_index=True)
    else:
        st.info(
            "training_history.csv not found. Run the training stage first, "
            "or copy the verified output into data/processed/training/."
        )

    st.markdown("### Risk scores")
    selected = best_risk_path if best_risk_path.exists() else risk_path
    if selected.exists():
        risk = pd.read_csv(selected)
        risk["OS_MONTHS"] = risk["OS_MONTHS"].round(2)
        risk["risk_score"] = risk["risk_score"].round(4)
        st.dataframe(risk, width="stretch", hide_index=True)

        st.markdown("### Risk ranking")
        ranked = risk.sort_values("risk_score", ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        st.dataframe(
            ranked[["rank", "patient_id", "OS_MONTHS", "OS_STATUS", "risk_score"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No risk-score CSV found yet.")

    st.markdown("### Generated files")
    for filename in [
        "training_vs_eval_loss.png",
        "eval_c_index.png",
        "stage7_checkpoint_summary.csv",
        "stage7_risk_score_table.csv",
        "final_risk_scores.csv",
        "best_risk_scores.csv",
    ]:
        path = TRAIN_DIR / filename
        if path.exists():
            st.write(f"✓ `{path.relative_to(ROOT)}`")
        else:
            st.write(f"○ `{path.relative_to(ROOT)}` not found")



def render_pic():
    st.header("Stage 8 - Pathway-Identity Consistency (PIC)")

    st.write(
        "PIC is an exploratory regularization extension. It encourages "
        "pathway-specific cross-attention maps to remain identifiable when "
        "a subset of WSI patches is removed."
    )
    st.warning(
        "All evaluations use the same 15 patients used during training. "
        "These results are exploratory and do not establish generalization, "
        "biological validity, or causal interpretability."
    )

    def read_csv_if_present(key):
        path = PATHS[key]
        if path.exists():
            return pd.read_csv(path)
        st.warning(f"Saved result not found: {path.relative_to(ROOT)}")
        return None

    def show_summary_csv(key, heading):
        st.subheader(heading)
        frame = read_csv_if_present(key)
        if frame is not None:
            st.dataframe(frame, width="stretch", hide_index=True)
        return frame

    st.markdown("## 8.1 - Initial pathway-identity diagnostic")
    st.write(
        "The diagnostic removes 20% of valid WSI patches and measures "
        "whether each pathway's attention map is re-identified among the "
        "perturbed maps. A permutation null provides a reference comparison."
    )
    show_summary_csv("pic_diagnostic", "Initial diagnostic summary")
    show_summary_csv("pic_null_summary", "Permutation-null comparison")
    show_summary_csv("pic_reidentification", "Patient-level re-identification")

    st.markdown("## 8.2 - PIC fine-tuning")
    st.write(
        "The fine-tuning objective adds an InfoNCE consistency term based "
        "on pairwise Jensen-Shannon distances between clean and perturbed "
        "pathway attention distributions. The PIC checkpoint is separate "
        "from the original PAMT checkpoint."
    )
    st.latex(
        r"\\mathcal{L}_{PIC}=-\\frac{1}{P}\\sum_p "
        r"\\log\\frac{\\exp(-D_{pp}/\\tau)}"
        r"{\\sum_q\\exp(-D_{pq}/\\tau)}"
    )
    if PATHS["pic_finetune_summary"].exists():
        st.code(
            PATHS["pic_finetune_summary"].read_text(
                encoding="utf-8", errors="replace"
            ),
            language="text",
        )
    else:
        st.warning("PIC fine-tuning summary not found.")

    show_summary_csv("pic_baseline_vs_pic", "Baseline versus PIC")
    history = read_csv_if_present("pic_finetune_history")
    if history is not None:
        st.markdown("### Fine-tuning history")
        st.dataframe(history, width="stretch", hide_index=True)

    st.markdown("## 8.3 - Lambda-zero control")
    st.write(
        "The control uses the same fine-tuning setup with the PIC loss "
        "weight set to zero. Baseline, control, and PIC were compared "
        "using the shared evaluation protocol and fresh perturbation draws."
    )
    if PATHS["pic_control_summary"].exists():
        st.code(
            PATHS["pic_control_summary"].read_text(
                encoding="utf-8", errors="replace"
            ),
            language="text",
        )
    else:
        st.warning("PIC control summary not found.")

    show_summary_csv("pic_control_comparison", "Shared evaluation comparison")
    control_history = read_csv_if_present("pic_control_history")
    if control_history is not None:
        with st.expander("Control training history"):
            st.dataframe(control_history, width="stretch", hide_index=True)

    st.markdown("## 8.4 - Attention-faithfulness diagnostic")
    st.write(
        "This analysis compares risk-score changes after attention-guided "
        "patch deletion with changes after random deletion. A larger change "
        "indicates greater sensitivity under this deletion test; it is not "
        "proof that an attended patch is causally or biologically important."
    )
    show_summary_csv("pic_faithfulness_summary", "Overall faithfulness results")
    show_summary_csv("pic_faithfulness_patients", "Patient-level results")

    pathway_results = read_csv_if_present("pic_faithfulness_pathways")
    if pathway_results is not None:
        with st.expander("Pathway-level faithfulness results"):
            st.dataframe(
                pathway_results.head(100),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Showing up to 100 rows. The complete CSV remains available "
                "in the project output folder."
            )

    st.markdown("## 8.5 - Implementation and limitations")
    for key, label in [
        ("pic_train", "PIC fine-tuning source"),
        ("pic_control_train", "Lambda-zero control source"),
    ]:
        source = PATHS[key]
        if source.exists():
            with st.expander(label):
                st.code(
                    source.read_text(encoding="utf-8", errors="replace"),
                    language="python",
                )
        else:
            st.caption(f"Source file not available: {source.relative_to(ROOT)}")

    st.info(
        "Interpretation: the diagnostic and control results support further "
        "investigation of pathway-identity consistency. They do not establish "
        "external validity because the cohort contains only 15 patients and "
        "there is no independent test cohort."
    )


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------
renderer_map = {
    "Overview": render_overview,
    "1 — Gene preprocessing + GeneBranch": render_gene_pipeline_complete,
    "3 — WSI / DINO pipeline + WSIBranch + reduction": render_wsi_pipeline_complete,
    "4 — Contrastive L3": render_contrastive,
    "5A — Pathway→patch fusion": render_fusion,
    "5B — Survival risk head": render_risk_head,
    "6 — SurvivalLoss": render_survival_loss,
    "7 — Training": render_train,
    "7 — Best checkpoint": render_evaluate,
    "7 — Results": render_results,
    "8 - Pathway-Identity Consistency (PIC)": render_pic,
    "📸 Visual Gallery": render_visual_gallery,
}

renderer_map[stage_name]()

st.divider()
st.caption(
    "PAMT mini-scale dashboard • 15 real BLCA/TCGA patients • "
    "C-index values are exploratory because the same patients were used for training."
)
