# PAMT: Mini-Scale Multimodal Survival Analysis

A mini-scale implementation of the **Pathology-Aware Multimodal Transformer (PAMT)** for multimodal survival analysis using transcriptomic and whole-slide image (WSI) data from **TCGA-BLCA**.

This repository implements the major components of the PAMT pipeline, including pathway-aware gene representation, WSI feature extraction and processing, pathway–patch contrastive learning, cross-modal fusion, survival-risk prediction, and Cox-based survival loss.

> **Scope:** This is a mini-scale research implementation using a small cohort of real TCGA-BLCA patients. It is intended for implementation study, experimentation, and reproducibility of the core pipeline rather than a full-scale reproduction of the original PAMT study.

---

## Overview

Cancer survival prediction can benefit from combining information from multiple modalities.

This implementation combines:

- **Transcriptomic data** represented at the pathway level
- **Whole-slide image (WSI) features** represented as image patches
- **Pathway–patch contrastive learning** to align the two modalities
- **Cross-modal attention** for multimodal fusion
- **Survival-risk prediction**
- **Cox-based survival loss**

The implementation follows the major stages of the PAMT architecture while operating on a substantially smaller cohort and computational scale.

---

## Implementation Scope

The current implementation uses:

- **15 TCGA-BLCA patients**
- **186 pathways**
- **4,942 processed gene features**
- WSI DINO embeddings with **384-dimensional** patch features
- Between **94 and 162 WSI patches per patient**
- Padding to a maximum of **162 patches**
- Pathway embeddings of dimension **256**
- WSI feature reduction from **384 → 640 → 256**
- A **16-head** cross-modal attention mechanism
- Full-batch training on the available mini cohort

Because the cohort is small and no independent validation/test cohort is used, the reported performance should be interpreted as **exploratory implementation results**, not as evidence of clinical generalization.

---

# Architecture

The implemented pipeline consists of the following stages:

```text
                    TCGA-BLCA Data
                         │
              ┌──────────┴──────────┐
              │                     │
        Gene / RNA data          WSI data
              │                     │
              ▼                     ▼
     Gene preprocessing       Patch extraction
              │                     │
              ▼                     ▼
      Pathway construction     DINO embeddings
              │                     │
              ▼                     ▼
      Pathway Embedding        WSI Transformer
              │                     │
              ▼                     ▼
       Gene Transformer        WSI Reduction
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
             Pathway–Patch Contrastive
                    Learning (L3)
                         │
                         ▼
               Cross-Modal Fusion
              Pathway → Patch Attention
                         │
                         ▼
                   Risk Head
                         │
                         ▼
                Survival Risk Score
                         │
                         ▼
               Cox Survival Loss
```

---

# Pipeline

## Stage 1 — Gene preprocessing

The gene pipeline performs preprocessing of the transcriptomic data and constructs the pathway-level representation.

Main components:

```text
src/gene_branch/preprocess_gene.py
src/gene_branch/build_pathway_matrix.py
src/gene_branch/pathway_embedding.py
src/gene_branch/transformer_encoder.py
src/gene_branch/gene_branch.py
```

The resulting gene representation is organized as:

```text
(B, 186, 256)
```

where:

- `B` = batch size
- `186` = number of pathways
- `256` = pathway embedding dimension

The mini cohort contains:

```text
(15, 186, 4942)
```

before the pathway embedding stage.

Run the gene pipeline with:

```bash
python scripts/run_gene_pipeline.py
```

---

## Stage 2 — WSI processing

The WSI branch processes whole-slide images into patch-level representations.

Relevant modules include:

```text
src/wsi_branch/patch_extraction.py
src/wsi_branch/patch_embedding.py
src/wsi_branch/dino_embedding.py
src/wsi_branch/dino_pretraining.py
src/wsi_branch/patch_clustering.py
src/wsi_branch/wsi_branch.py
src/wsi_branch/wsi_reduction.py
```

The WSI pipeline produces DINO-based patch embeddings with:

```text
384 features per patch
```

The number of patches varies between patients.

For the current mini cohort:

```text
Minimum patches: 94
Maximum patches: 162
```

To enable batch processing, sequences are padded to:

```text
(B, 162, 384)
```

with a corresponding padding mask.

Run the WSI pipeline with:

```bash
python scripts/run_wsi_pipeline.py
```

or the batch version:

```bash
python scripts/run_wsi_pipeline_batch.py
```

---

# Stage 3 — WSI Transformer and Reduction

The WSI branch processes patch embeddings using a Transformer encoder.

The implementation uses:

- 6 Transformer blocks
- 16 attention heads
- 384-dimensional input features
- Padding masks for variable-length WSI sequences

The resulting representation is reduced from:

```text
384 → 640 → 256
```

giving:

```text
(B, N, 256)
```

where `N ≤ 162` is the padded patch dimension.

The reduction preserves the WSI padding mask.

---

# Stage 4 — Pathway–Patch Contrastive Learning

The implementation includes the pathway–patch contrastive objective used to align the transcriptomic and image modalities.

The implementation uses:

- Top-`h` patch selection
- `top_h = 2`
- Padding-aware selection
- Padding-aware softmax
- Binary pathway–patch relationship targets
- Raw dot-product similarity
- No learnable temperature parameter

The contrastive loss is implemented in:

```text
src/gene_branch/contrastive_loss.py
```

A visualization of the contrastive stage is provided in:

```text
stage4_5b_figures/
├── stage4_similarity_heatmap.png
└── stage4_top2_selections.png
```

---

# Stage 5A — Cross-Modal Fusion

The fusion module combines pathway representations with WSI patch representations.

The implementation uses pathway representations as the **query** and image patch representations as **keys/values**:

```text
Pathway Q
   │
   ├───────────────┐
   │               │
   ▼               ▼
 Pathway        WSI patches
   │               │
   └───────┬───────┘
           ▼
   Cross-Attention
           │
           ▼
    Fused representation
```

The fusion module uses:

- 256-dimensional representations
- 16 attention heads
- Padding-aware attention

The resulting cross-attention representation has shape:

```text
(B, 186, 256)
```

The attention weights have shape:

```text
(B, 16, 186, N)
```

A visualization is provided in:

```text
stage4_5b_figures/stage5a_cross_attention_heatmap.png
```

---

# Stage 5B — Risk Head

The risk head combines:

- Pathway representation
- Cross-modal attention representation

Each representation is pooled over its feature dimension and concatenated before risk prediction.

The resulting representation has:

```text
372 features
```

which is passed through a linear layer:

```text
372 → 1
```

producing a single survival-risk score per patient.

The risk head contains:

```text
373 trainable parameters
```

The implementation is available in:

```text
scripts/risk_head.py
```

A visualization is provided in:

```text
stage4_5b_figures/stage5b_risk_outputs.png
```

---

# Stage 6 — Survival Loss

The survival objective combines three components:

```text
L = L1 + αλL2 + βL3
```

where:

- `L1` = Cox negative partial log-likelihood
- `L2` = weight regularization
- `L3` = pathway–patch contrastive loss
- `α = 1`
- `λ = 5 × 10⁻⁴`
- `β = 0.8`

The Cox loss uses a risk-set formulation based on survival times and event indicators.

The implementation also handles the all-censored edge case, where the Cox event-dependent loss becomes zero.

Relevant files:

```text
scripts/survival_loss.py
src/gene_branch/contrastive_loss.py
```

---

# Stage 7 — Training

Training is performed using:

```text
scripts/train.py
```

The best checkpoint is selected based on training loss.

Best checkpoint:

```text
Epoch: 467
```

The checkpoint can be evaluated using:

```bash
python scripts/evaluate_best.py
```

---

# Results

The current mini-scale experiment produced the following results for the best checkpoint:

| Metric | Value |
|---|---:|
| Best epoch | 467 |
| Best training loss | 2.253419 |
| Evaluated total loss | 7.744978 |
| Cox survival loss (L1) | 0.000995 |
| Raw regularization (L2) | 912.447876 |
| Weighted regularization | 0.456224 |
| Contrastive loss (L3) | 9.109699 |
| Weighted contrastive loss | 7.287759 |
| C-index | 1.0000 |

### Important interpretation

The reported C-index of `1.0000` was obtained on the same small cohort used for training.

Therefore, it should **not** be interpreted as evidence of generalization or clinical performance.

A proper evaluation of predictive performance would require an independent validation/test cohort and substantially larger sample size.

---

# Visual Results

Selected intermediate results are included in:

```text
stage4_5b_figures/
```

These include:

- Pathway–patch similarity heatmap
- Top-2 patch selections
- Cross-attention heatmap
- Risk-output visualization

The repository also contains model-input visualizations under:

```text
outputs/visuals/
```

---

# Interactive Dashboard

The project includes an interactive Streamlit dashboard for exploring the implementation and its intermediate results.

Launch it with:

```bash
streamlit run dashboard.py
```

The dashboard provides sections covering:

- Gene preprocessing
- GeneBranch
- WSI branch
- WSI reduction
- Contrastive learning
- Cross-modal fusion
- Risk head
- Survival loss
- Training
- Best checkpoint
- Results

The dashboard is designed to make the implementation easier to inspect and demonstrate during project presentations.

---

# Repository Structure

```text
pamt_mini/
│
├── dashboard.py
├── config.yaml
├── demo_gene_output.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── scripts/
│   ├── diagnose_reproducibility.py
│   ├── evaluate_best.py
│   ├── generate_model_input_panel.py
│   ├── generate_stage4_5b_figures.py
│   ├── risk_head.py
│   ├── run_dino_training.py
│   ├── run_gene_pipeline.py
│   ├── run_wsi_pipeline.py
│   ├── run_wsi_pipeline_batch.py
│   ├── show_gene_output.py
│   ├── show_wsi_inputslide.py
│   ├── show_wsi_output.py
│   ├── stage7_make_results.py
│   ├── survival_loss.py
│   └── train.py
│
├── src/
│   ├── gene_branch/
│   ├── wsi_branch/
│   ├── fusion.py
│   ├── c_index.py
│   ├── utils/
│   └── visualization/
│
├── tests/
│   ├── test_contrastive_loss.py
│   ├── test_end_to_end.py
│   ├── test_fusion.py
│   ├── test_gene_branch.py
│   ├── test_pathway_embedding.py
│   ├── test_risk_head.py
│   ├── test_survival_loss.py
│   ├── test_wsi_branch.py
│   └── test_wsi_reduction.py
│
├── stage4_5b_figures/
│
└── outputs/
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd pamt_mini
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Data

This project uses real TCGA-BLCA molecular and whole-slide image data.

Due to the size of the raw data and associated medical imaging files, the raw TCGA/GDC data are **not included in this Git repository**.

The repository is intended to contain:

- Source code
- Training and evaluation scripts
- Tests
- Configuration
- Visualizations
- Selected lightweight results
- Dashboard

Large raw datasets, processed arrays, WSI files, and model checkpoints are excluded through `.gitignore`.

If a redistributable processed dataset is made available separately, its download location can be added here.

---

# Tests

The repository contains unit and integration tests for the major components:

```text
tests/
```

Run the test suite with:

```bash
pytest
```

The tests cover components including:

- Gene branch
- Pathway embedding
- WSI branch
- WSI reduction
- Contrastive loss
- Fusion
- Risk head
- Survival loss
- End-to-end execution

---

# Reproducibility

The implementation includes a reproducibility diagnostic script:

```bash
python scripts/diagnose_reproducibility.py
```

Configuration parameters are maintained in:

```text
config.yaml
```

Training and evaluation outputs are generated through the scripts in:

```text
scripts/
```

---

# Differences from the Original PAMT

This repository should not be considered a full reproduction of the original PAMT study.

Important differences include:

- Much smaller patient cohort
- 15 patients in the current implementation
- Reduced computational scale
- 4,942 processed gene features
- Smaller experimental setup
- No independent validation/test cohort
- Mini-scale WSI processing
- Exploratory training rather than clinical-scale evaluation

The goal is to reproduce and study the **core multimodal architecture and computational pipeline**, rather than reproduce the exact experimental scale and reported results of the original paper.

---

# Limitations

The current implementation has several important limitations:

1. **Small sample size**

   Only 15 patients are used.

2. **No independent validation set**

   Training and evaluation are performed on the available mini cohort.

3. **No clinical generalization claim**

   The results should not be interpreted as clinically predictive.

4. **Reduced dataset scale**

   The original study operates on a substantially larger dataset.

5. **Computational constraints**

   The implementation was designed to be executable on a much smaller computational setup.

6. **Exploratory results**

   The reported metrics demonstrate implementation behavior rather than statistically reliable clinical performance.

---

# Citation

If you use or build upon this implementation, please also cite the original PAMT paper.

Replace the citation block below with the exact bibliographic information for the PAMT paper used as the implementation reference:

```bibtex
@article{
  PAMT,
  title = {Pathology-Aware Multimodal Transformer},
  ...
}
```
