# Noise-Ceiling-Aware Hybrid 2D+3D Learning for Experimental Dipole Moment Prediction

This repository accompanies the manuscript:  
**“Noise-Ceiling-Aware Hybrid 2D+3D Learning for Experimental Dipole Moment Prediction”**

It provides a leakage-safe, reproducible pipeline for predicting experimental molecular dipole moments using:
1.  **SMILES-derived 2D fingerprints** (Multi-view).
2.  **Conformer-derived 3D molecular graphs** (Geometry-aware).
3.  **Hybrid Ensemble Learning** that bridges experimentally constrained noise and model expressivity.

---

## 📖 Overview

Experimental dipole-moment labels are inherently noisy and heterogeneous due to protocol differences, solvent effects, and reproducibility limits. Accordingly, this repository implements a robust pipeline featuring:

* **Multi-view 2D learning:** Utilization of MACCS keys + multiple Avalon fingerprint configurations.
* **3D geometry-aware learning:** RDKit-generated conformers converted into 3D graphs, processed by a stabilized EGNN-style GNN regressor.
* **Hybrid fusion:** A strategy combining complementary 2D and 3D signals to improve generalization under experimental noise.
* **Noise-aware evaluation:** Metrics designed to contextualize performance relative to a dataset-specific "experimental noise ceiling" (using twin-pair analysis).

---

## 📂 Repository Structure

The codebase is intentionally modular. File names reflect the sequential stage of the pipeline:

| File / Directory | Description |
| :--- | :--- |
| `1_convert_smiles.py` | Converts chemical names (if present) to SMILES using resolvers (PubChem/Cactus). |
| `2_rdkit_fingerprints_features.py` | Generates 2D fingerprint features (MACCS & Avalon variants) and exports feature tables. |
| `3_catboost_train_and_save.py` | Trains a tuned **CatBoost** regressor on the multi-view 2D fingerprints. |
| `3_ann_multiview_train_and_save.py` | Trains a **Multi-view ANN** (three-tower design) to learn cross-view interactions. |
| `3_3d_gnn_train_and_save.py` | Builds conformers, creates 3D graphs, and trains a stabilized **EGNN-style GNN**. |
| `4_Hybrid_3d_gnn+...py` | Implements the **Hybrid Ensemble Fusion** across the three model families. |
| `train_ready.csv` | Fixed training split used for standardized evaluation. |
| `test_ready.csv` | Fixed testing split used for standardized evaluation. |
| `dipole_moments.csv` | Source dataset file. |

---

## 📋 Requirements

* **Python Version:** 3.9+ (3.10/3.11 recommended)

### Core Packages
* **Data handling:** `numpy`, `pandas`
* **ML utilities:** `scikit-learn`
* **2D fingerprints:** `rdkit`
* **Boosting model:** `catboost`
* **3D GNN training:** `torch` (PyTorch)
* **Resolvers (Optional):** `pubchempy`, `requests`

---

## ⚙️ Installation

### Option A: Conda (Recommended)
Conda is strictly recommended for easier RDKit installation.

```bash
conda create -n dipole_mu python=3.10 -y
conda activate dipole_mu

conda install -c conda-forge rdkit -y
pip install numpy pandas scikit-learn catboost torch pubchempy requests
