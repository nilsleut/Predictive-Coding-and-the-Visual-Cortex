# Predictive Coding and the Visual Cortex
### Does a brain-inspired model represent visual information the way the brain does?

**Nils Leutenegger · 2026**

---

## Overview

This project tests a core prediction of **Predictive Coding (PC) theory**: that the brain organizes visual processing as a hierarchy of prediction and error signals, where each layer tries to predict the activity of the layer below it. If this theory is correct, a PC network should produce representations that mirror the cortical hierarchy — early layers aligning with early visual areas (V1/V2), late layers with higher areas (LOC/IT).

To test this, I built a hierarchical PC network, trained it on image features extracted from ResNet-50, and compared its internal representations to human fMRI data from the THINGS-fMRI dataset using **Representational Similarity Analysis (RSA)**. The comparison was done across 6 cortical ROIs (V1, V2, V3, V4, LOC, IT) in 3 subjects.

The central question: *does a model built according to the brain's own logic model the visual cortex better than purely data-driven CNNs?*

---

## Motivation

Most computational neuroscience work uses CNNs as brain models. ResNet, ViT, and CLIP achieve impressive performance, but they are feedforward models trained on classification — not models of how the brain actually computes. Predictive Coding, originally proposed by Rao & Ballard (1999) and later formalized by Karl Friston, offers a fundamentally different account: the brain is a prediction machine, constantly generating top-down predictions and computing bottom-up prediction errors.

If PC is a correct description of cortical computation, a model built on PC principles should produce a specific signature in its representations: a **crossing gradient** across the visual hierarchy. This project operationalizes and empirically tests that prediction.

---

## Background

### Predictive Coding Theory

In PC, each layer of the cortical hierarchy maintains a *representation* (r) of the expected input and computes a *prediction error* (ε) — the difference between what it predicted and what it received. Top-down connections carry predictions downward; bottom-up connections carry errors upward. Learning minimizes prediction error, which is equivalent to minimizing a quantity called **free energy**.

The key prediction for RSA: if lower PC layers represent more concrete, stimulus-level features (like V1 represents edges and orientations), and higher PC layers represent more abstract predictions (like IT represents object categories), then the RSA signature should show a characteristic crossing pattern across the visual hierarchy.

### Representational Similarity Analysis (RSA)

RSA compares the *geometry* of representations, not their values directly. For each model layer and each cortical ROI, I compute a **Representational Dissimilarity Matrix (RDM)** — a 100×100 matrix where entry (i,j) is the dissimilarity between the representations of stimulus i and stimulus j. Two systems are compared by computing the Spearman correlation between their RDMs. A high correlation means the two systems organize stimuli similarly, not that they produce identical activations.

This makes RSA model-agnostic: I can compare a neural network layer to fMRI voxel patterns without assuming anything about dimensionality or activation values.

---

## Dataset

**THINGS-fMRI** (Hebart et al., 2023) — OpenNeuro ds004192

- 3 subjects (sub-01, sub-02, sub-03), 7T fMRI
- 100 object concepts, 1 image per concept (~12 repetitions per image)
- 6 manually annotated cortical ROIs: V1, V2, V3, V4, LOC, IT
- Voxel-wise GLM responses provided as `.h5` files

**Stimuli used:** 100 images (1 per concept), alphabetically selected and verified identical across all 3 subjects.

---

## Architecture

### PC Network

```
Input: ResNet-50 layer4 features (2048-dim)
       ↓
r3  [2048-dim]  — IT-analog     (initialized from layer4)
r2  [1024-dim]  — LOC-analog    (initialized from layer3)
r1  [ 512-dim]  — V4-analog     (initialized from layer2)
r0  [ 256-dim]  — V1-analog     (initialized from layer1)
```

Each layer maintains a representation rᵢ and computes prediction errors εᵢ = rᵢ - Wᵢrᵢ₊₁. During inference, representations are updated to minimize prediction error (gradient descent on free energy, T=30 steps). Weights are updated after each batch via Hebbian-style learning proportional to prediction errors.

**Key design choice:** The network is initialized from ResNet features rather than random noise. This grounds the representations in visually structured inputs from the start, avoiding the degenerate solutions that arise from random initialization.

### Training

- **Objective:** Minimize total free energy = Σᵢ ||εᵢ||²
- **Optimizer:** Gradient descent on representations (lr_r=0.01), Hebbian weight update (lr_w=5e-4)
- **Early Stopping:** Patience=15 epochs, best weights restored
- **Seed:** Fixed (42) for reproducibility
- **Convergence:** All 3 subjects converge at Epoch ~55, FE≈0.628

---

## Methods

### Pipeline

```
1. Load THINGS-fMRI data (sub-01, sub-02, sub-03)
2. Select 100 stimuli (1 per concept, alphabetically sorted — identical across subjects)
3. Average fMRI responses across repetitions → compute fMRI RDMs per ROI
4. Extract ResNet-50 features (layer1–layer4)
5. Train PC network on ResNet features → extract r0–r3, ε0–ε2
6. Compute model RDMs for all PC layers + ResNet/ViT/CLIP baselines
7. RSA: Spearman ρ between model RDMs and fMRI RDMs
8. Bootstrap 95% CI (1000 samples) for each comparison
9. Noise Ceiling via split-half reliability (Spearman-Brown corrected)
10. Interaction test: Δr0 − Δr3 (early vs. late ROI gradient)
```

### Baselines

| Model | Layers used |
|---|---|
| ResNet-50 | layer1–layer4 (best per ROI) |
| ViT-B/16 | block3, block6, block9, block12 (best per ROI) |
| CLIP ViT-B/32 | block3, block6, block9, block12 (best per ROI) |

All baselines use the best-performing layer per ROI, giving them maximum advantage.

### Statistical Approach

- **RSA metric:** Spearman ρ on upper triangle of RDMs
- **Confidence intervals:** 1000-iteration bootstrap, 95% CI
- **Noise Ceiling:** Split-half reliability with Spearman-Brown correction (100 splits), represents the theoretical maximum correlation achievable by any model given fMRI measurement noise
- **Interaction test:** Quantifies the crossing pattern as Δr0 − Δr3, where Δ = mean(early ROIs) − mean(late ROIs)

---

## Results

### Training Convergence

All three subjects show identical training curves (the PC network trains on ResNet features, which are subject-independent). Free energy decreases monotonically from ~1.68 to 0.628, with early stopping triggered at Epoch 70. The best weights (Epoch 55, FE=0.628) are used for all RSA comparisons.

### Hierarchical Gradient — Sub-01

| Layer | V1 | V2 | V3 | V4 | LOC | IT |
|---|---|---|---|---|---|---|
| r0 (V1-init) | **0.308** | **0.208** | 0.155 | 0.147 | 0.040 | 0.065 |
| r1 (V4-init) | 0.258 | 0.184 | 0.149 | 0.163 | 0.070 | 0.101 |
| r2 (LOC-init) | 0.220 | 0.175 | 0.146 | 0.193 | 0.149 | 0.187 |
| r3 (IT-init) | 0.103 | 0.073 | 0.067 | 0.108 | **0.142** | **0.154** |
| ResNet-50 | 0.083 | 0.050 | 0.049 | 0.090 | 0.123 | 0.145 |
| ViT-B/16 | 0.201 | 0.163 | 0.125 | 0.127 | 0.064 | 0.086 |
| CLIP | 0.160 | 0.131 | 0.109 | 0.124 | 0.083 | 0.121 |
| Noise Ceiling | 0.519 | 0.447 | 0.379 | 0.231 | 0.400 | 0.404 |

The crossing pattern is visible: r0 peaks at V1 (ρ=0.308), r3 peaks at LOC/IT (ρ=0.142/0.154). Crucially, **PC r0 outperforms ViT-B/16 at V1** (0.308 vs. 0.201) despite ViT being trained on orders of magnitude more data.

### Replication Across Subjects

The crossing pattern replicates in all 3 subjects:

| | r0@V1 | r0@LOC/IT | r3@V1 | r3@LOC/IT |
|---|---|---|---|---|
| Sub-01 | 0.308 | 0.052 | 0.103 | 0.148 |
| Sub-02 | 0.355 | 0.048 | 0.102 | 0.171 |
| Sub-03 | 0.252 | 0.050 | 0.092 | 0.137 |

Sub-03 has lower absolute values throughout, consistent with its lower noise ceilings (NC~0.17 vs ~0.50 for sub-01/02), suggesting less reliable fMRI data rather than a model failure.

### Group-Level Interaction Test

```
Layer     Early (V1/V2)    Late (LOC/IT)    Δ
──────────────────────────────────────────────
r0            0.253            0.050       +0.203
r1            0.226            0.085       +0.141
r2            0.192            0.160       +0.032
r3            0.089            0.152       -0.063

Interaction effect (Δr0 − Δr3) = +0.266  ✅
```

The interaction effect of **+0.266** quantifies the crossing pattern: r0 has a strong positive gradient (better at early ROIs), r3 has a negative gradient (better at late ROIs). This is the precise signature predicted by PC theory.

### PC vs. Baselines at V1

| Model | V1 ρ (mean across subjects) |
|---|---|
| **PC r0** | **0.305** |
| ViT-B/16 block3 | 0.189 |
| CLIP block3 | 0.159 |
| ResNet-50 layer2 | 0.079 |

PC r0 consistently outperforms all baselines at V1 across all 3 subjects. This is the project's strongest finding — a brain-inspired model beats state-of-the-art vision models at the earliest visual area.

---

## Interpretation

### What Was Confirmed

**The central prediction is supported.** The crossing gradient replicates in N=3 subjects with a quantified interaction effect of +0.266. Lower PC layers align with early visual cortex, higher PC layers align with later areas — exactly as PC theory predicts.

The finding that PC r0 outperforms ViT and CLIP at V1 is notable. These models have dramatically more parameters and training data. The advantage of PC r0 may reflect that iterative inference (T=30 steps of representation refinement) produces representations that are more aligned with the kind of local, low-level structure that V1 encodes — edges, orientations, spatial frequencies — than a single feedforward pass.

### What Was Not Confirmed

PC does not outperform ResNet at LOC and IT. The model captures early visual processing well but does not improve on feedforward baselines at higher cortical areas. This is theoretically coherent: the current implementation has no genuine top-down connections to semantic or object-category representations. The PC dynamics operate entirely within the ResNet feature space, which limits what higher layers can represent.

### Limitations

**N=100 stimuli** is small for RSA — 100×100 RDMs have limited resolution. Larger stimulus sets (e.g., all 720 THINGS concepts) would yield more reliable RDMs.

**PC on ResNet features** means the representations are ultimately derived from a feedforward model. A truly end-to-end PC model trained on raw images from scratch might show a different pattern.

**Sub-03 data quality** is substantially lower (noise ceilings ~0.17 vs ~0.50). Results for this subject should be interpreted with caution.

**Training instability.** Free energy shows high variance after Epoch 20 even with early stopping. A more stable optimizer (e.g., Adam with learning rate scheduling) might improve results.

---

## Repository Structure

```
predictive_coding_rsa_v7.py   — Main analysis script (all subjects)
RSA_COMPARE_v2.ipynb          — ViT/CLIP RDM computation & export
outputs/
├── pc_hierarchy_group.png        — Group-level crossing pattern (main result)
├── pc_hierarchy_sub-01.png       — Subject-level with 95% bootstrap CIs
├── pc_hierarchy_sub-02.png
├── pc_hierarchy_sub-03.png
├── pc_rsa_comparison_sub-01.png  — Full RSA comparison with all baselines
├── pc_rsa_comparison_sub-02.png
├── pc_rsa_comparison_sub-03.png
├── pc_training_curve_sub-01.png  — Free energy convergence
├── pc_training_curve_sub-02.png
├── pc_training_curve_sub-03.png
├── stim_order_sub-01.txt         — Verified stimulus list (identical across subjects)
├── stim_order_sub-02.txt
└── stim_order_sub-03.txt
```

---

## Reproduction

### Requirements

```bash
pip install torch torchvision numpy scipy pandas matplotlib h5py tqdm
pip install clip-anytorch  # for CLIP
```

### Data

Download THINGS-fMRI from OpenNeuro (ds004192):
```bash
pip install datalad
datalad clone https://github.com/OpenNeuroDatasets/ds004192.git
cd ds004192
datalad get sub-01/ sub-02/ sub-03/
```

You also need the THINGS image set (object_images folder) from the same dataset.

### Run

1. Set paths in the `Config` class at the top of `predictive_coding_rsa_v7.py`
2. Run the notebook `RSA_COMPARE_v2.ipynb` (Kernel → Restart & Run All) to export ViT/CLIP RDMs
3. Run the main script:

```bash
python predictive_coding_rsa_v7.py
```

All outputs are saved to `Predictive Coding/outputs/`.

---

## Key References

- Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2, 79–87.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127–138.
- Hebart, M.N. et al. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. *eLife*, 12, e82580.
- Kriegeskorte, N., Mur, M. & Bandettini, P. (2008). Representational similarity analysis – connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*.

---

*Built with Python 3.11 · PyTorch · THINGS-fMRI · N=3 subjects · 100 concepts · 6 cortical ROIs*
