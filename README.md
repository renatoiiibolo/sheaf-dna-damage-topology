# sheaf-dna-damage-topology

**Cellular sheaf restriction maps formalize radiation-induced DNA damage topology as modality coherence energy**

Renato III Fernan Bolo and Ramon Jose C. Bagunu  
Department of Physical Sciences and Mathematics, University of the Philippines Manila

[![Target: Physical Review E](https://img.shields.io/badge/target-Physical%20Review%20E-blue)](https://journals.aps.org/pre/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

This is **Project [3]** of the Bolo & Bagunu computational radiation biophysics arc. The paper demonstrates that cross-covariance SVD (CCSV) restriction maps on a seven-modality cellular sheaf faithfully encode the inter-modality covariance structure of radiation-induced DNA double-strand break (DSB) damage across 49 LET–oxygen irradiation conditions (Spearman ρ = −0.598, p = 0.004, n = 21; 3/3 supplementary tests). The central result — that CCSV maps uniquely minimize the per-edge coboundary energy over all Stiefel-constrained map pairs — is proven analytically via the von Neumann trace inequality (Proposition 1 in the manuscript).

**Arc context:**

| Project | Title | Status |
|---------|-------|--------|
| [1] VOxA | Voxel-aware oxygen kinetics | Complete — [arXiv:2605.12558](https://arxiv.org/abs/2605.12558) |
| [2] Topology | DSB topology encodes LET–oxygen signatures | Complete — submitted |
| **[3] Sheaf** | **Sheaf restriction maps as modality coherence energy** | **This repository** |
| [3.5] SMLM bridge | Synthetic SMLM forward model | In preparation |
| [4] Digital twin | Hypoxic LET painting with digital twin | Ongoing |

---

## Repository structure

```
sheaf-dna-damage-topology/
│
├── config.py                     # Project-wide constants (MODALITY_ORDER, D_V, edges, etc.)
├── requirements.txt
├── LICENSE
│
├── models/                       # Neural network components
│   ├── __init__.py
│   ├── sheaf_modality.py         # Main ModalitySheafModel (forward pass, diffusion, head)
│   ├── laplacian_hetero.py       # Sheaf Laplacian construction and E_i computation
│   ├── restriction_maps_qr.py    # QR gradient-trained maps            [PRODUCTION]
│   ├── restriction_maps_cca.py   # CCSV analytical maps                [PRODUCTION]
│   ├── restriction_maps_rand.py  # RandEdge Haar-random null           [PRODUCTION]
│   ├── orthogonal_thin.py        # Stiefel manifold via Householder     [UTILITY]
│   └── restriction_maps.py       # Original Householder parametrization [DEPRECATED]
│
├── data/                         # Feature matrix and metadata
│   ├── __init__.py
│   ├── loader.py                 # Dataset loading and preprocessing utilities
│   ├── feature_matrix.csv        # 2450 × 107 per-nucleus feature matrix
│   ├── feature_matrix_summary.csv
│   ├── feature_metadata.json     # Feature names and modality assignments
│   ├── condition_summary_compiled.csv  # Per-condition mean E_i (analysis output)
│   ├── edge_energy_summary.csv         # Per-edge mean E_e/k_e (analysis output)
│   └── README.md
│
├── training/                     # Training entry points
│   ├── train_improved.py         # QR parametrization              [PRODUCTION]
│   ├── train_analytical.py       # CCSV parametrization            [PRODUCTION]
│   ├── train_rand.py             # RandEdge parametrization        [PRODUCTION]
│   └── train.py                  # Original Householder            [DEPRECATED]
│
├── evaluation/                   # Evaluation and verification scripts
│   ├── evaluate.py               # General evaluation (QR)
│   ├── evaluate_analytical.py    # CCSV-specific evaluation
│   ├── freeze_qr_eval.py         # Frozen-QR capacity decomposition
│   └── verify/                   # Batch correctness checks
│       ├── verify_batch1.py
│       ├── verify_batch2.py
│       ├── verify_batch3.py
│       ├── verify_batch4.py
│       ├── verify_batch5.py
│       ├── verify_batch6.py
│       ├── verify_batch7.py
│       ├── verify_cca.py
│       ├── verify_improved.py
│       └── verify_rand.py
│
└── analysis/                     # Post-training analysis scripts
    ├── analyze_ei_landscape.py   # E_i landscape (LET and O2 gradients)
    ├── cca_k_robustness.py       # Multi-k robustness sweep
    ├── extract_condition_summary.py  # Compile condition_summary_compiled.csv
    └── extract_edge_energies.py      # Compile edge_energy_summary.csv
```

> **Trained run outputs** (`outputs/`, `outputs_cca/`, `outputs_improved/`, `outputs_rand/`)
> are archived at **Zenodo DOI: `10.5281/zenodo.XXXXXXX`** and are not committed here.

---

## Installation

```bash
git clone https://github.com/[username]/sheaf-dna-damage-topology.git
cd sheaf-dna-damage-topology
pip install -r requirements.txt
```

Ensure the repository root is on your Python path so that `config.py` and `models/`
are importable:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Reproducing the results

### Step 1 — Confirm data

The feature matrix is already in `data/feature_matrix.csv`.
Download the trained run outputs from Zenodo (DOI above) and place them
under `outputs/`, `outputs_cca/`, `outputs_improved/`, `outputs_rand/`
at the repository root if you want to skip retraining.

### Step 2 — Train all three parametrizations (optional if using Zenodo outputs)

```bash
# QR gradient-trained
for seed in 0 1 2; do
    python training/train_improved.py --seed $seed --k 24
done

# CCSV analytical
for seed in 0 1 2; do
    python training/train_analytical.py --seed $seed --k 24
done

# RandEdge null
for seed in 0 1 2; do
    python training/train_rand.py --seed $seed --k 24
done
```

### Step 3 — Compile analysis tables

```bash
python analysis/extract_condition_summary.py
python analysis/extract_edge_energies.py
```

### Step 4 — Run the reported analyses

```bash
# Figure 1 and 2 landscape data
python analysis/analyze_ei_landscape.py

# Figure 3 per-edge geometric validation
python evaluation/evaluate_analytical.py

# Figure 4 multi-k robustness
python analysis/cca_k_robustness.py

# Table II capacity decomposition
python evaluation/freeze_qr_eval.py
```

---

## Three-parametrization design

| Parametrization | Maps derived from | Trainable params | Role |
|----------------|------------------|-----------------|------|
| **CCSV** | SVD of training cross-covariance | 11,249 | Analytical covariance encoding; central result |
| **QR** | Gradient descent on unconstrained W | 31,169 | Classification-learned geometry |
| **RandEdge** | Haar measure (Stewart 1980, sign-corrected QR) | 11,249 | Null baseline to confirm CCSV result is data-driven |

All maps satisfy the Stiefel constraint F Fᵀ = I_{k_e} by construction
(max residual across all 84 maps: 1.40 × 10⁻⁶).

---

## Key analytical result

**Proposition 1 (CCSV optimality):** Under the approximation of approximately
decorrelated within-modality features (valid for z-scored data), the CCSV maps
uniquely minimize the empirical mean per-edge coboundary energy:

$$Q_\text{min} = 2k_e - 2\sum_{j=1}^{k_e} \sigma_j(\mathbf{C}_{vu})$$

The minimum is monotonically decreasing in the sum of leading cross-covariance
singular values, which is empirically anti-correlated with the cross-modality
Pearson coefficient |r_vu| from the companion study.
The empirical Spearman ρ = −0.598 is the confirmation of this analytical
prediction (proof via the Ky Fan trace inequality; see manuscript Sec. II.C
and Appendix C).

---

## Modality definitions

| Label | Description | d_v | Feature indices |
|-------|-------------|-----|-----------------|
| m1 | Spatial distribution | 33 | 0–32 |
| m2 | Radial track structure | 14 | 33–46 |
| m3 | Local energy heterogeneity | 13 | 47–59 |
| m4 | Dose distribution | 10 | 60–69 |
| m5 | Genomic location | 16 | 70–85 |
| m6 | Damage complexity | 11 | 86–96 |
| m7 | Topological summaries | 10 | 97–106 |
| **Total** | | **107** | |

---

## Citation

```bibtex
@article{BoloBagunu2026sheaf,
  author  = {Bolo, Renato III Fernan and Bagunu, Ramon Jose C.},
  title   = {Cellular sheaf restriction maps formalize radiation-induced
             {DNA} damage topology as modality coherence energy},
  journal = {Physical Review E},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
