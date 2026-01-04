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




````
## Option B — Pip-only Installation

If you cannot use **Conda**, you can install all required dependencies via `pip`:

```bash
pip install numpy pandas scikit-learn catboost torch pubchempy requests rdkit-pypi
````

> **Note:** `rdkit-pypi` may be less robust than the `conda-forge` RDKit distribution on some operating systems. Conda is recommended when possible.

---

## 📊 Data Format (Input Schema)

To run the pipeline on your own data, ensure that your input CSV files
`train_ready.csv` and `test_ready.csv` follow the schema below:

| Column Name     | Type   | Description                                     |
| --------------- | ------ | ----------------------------------------------- |
| `SMILES`        | String | Canonical SMILES representation of the molecule |
| `dipole_moment` | Float  | Experimental dipole moment value (**target**)   |
| `Name`          | String | *(Optional)* Common molecule name or identifier |

---

## 🚀 Quick Start (End-to-End Pipeline)

### 1. *(Optional)* Convert Chemical Names to SMILES

If your dataset contains only molecule names:

```bash
python 1_convert_smiles.py
```

---

### 2. Generate 2D Fingerprint Features

This step extracts **MACCS keys** and **Avalon fingerprints**:

```bash
python 2_rdkit_fingerprints_features.py
```

---

### 3. Train the 2D CatBoost Model

```bash
python 3_catboost_train_and_save.py \
  --train_csv train_ready.csv \
  --test_csv test_ready.csv
```

---

### 4. Train the Multi-view ANN Model

```bash
python 3_ann_multiview_train_and_save.py \
  --train_csv train_ready.csv \
  --test_csv test_ready.csv
```

---

### 5. Train the Stabilized 3D GNN Model

This step generates molecular conformers on-the-fly (or loads them from cache) and trains a geometry-aware graph neural network.

**CPU execution:**

```bash
python 3_3d_gnn_train_and_save.py \
  --train_csv train_ready.csv \
  --test_csv test_ready.csv \
  --device cpu
```

**GPU execution:**

```bash
python 3_3d_gnn_train_and_save.py \
  --train_csv train_ready.csv \
  --test_csv test_ready.csv \
  --device cuda
```

---

### 6. Hybrid Ensemble Fusion

This step combines predictions from:

* 2D CatBoost
* Multi-view ANN
* 3D GNN

```bash
python 4_Hybrid_3d_gnn+catboosting+multi_view_*.py
```

---

## 📈 Outputs & Reproducibility

Each training script generates the following artifacts to ensure full reproducibility of the manuscript results:

### Generated Outputs

* **Metrics:**
  JSON and CSV summaries containing `R²`, `MAE`, and `RMSE` for each run

* **Predictions:**
  CSV files with `y_true` vs. `y_pred` for detailed error analysis

* **Models:**
  Trained model checkpoints saved under the `models/` directory

---

### Reproducing Manuscript Results

* **2D baseline results:**
  Use outputs from **Step 3** (CatBoost) and **Step 4** (Multi-view ANN)

* **3D geometry-aware results:**
  Use outputs from **Step 5** (3D GNN)

* **Final hybrid performance (Table X in manuscript):**
  Refer to the console output and log files generated in **Step 6**

> **Note:** Deterministic random seeds are used where possible.
> However, **3D conformer generation (ETKDG)** and **GPU-level non-determinism** may result in minor variations in decimal values.

---

## 📄 Citation

If you use this repository or its methodology, please cite the associated manuscript:

> **Noise-Ceiling-Aware Hybrid 2D+3D Learning for Experimental Dipole Moment Prediction**
> *(Author list and journal information will be updated upon publication.)*

---

## ⚖️ License & Contact

**License:**
This code is provided for **academic use only**.

**Commercial usage:**
Please contact the author for licensing inquiries.

**Contact:**
**Sadettin Y. Ugurlu**
📧 *[s.yavuz.ugurlu@gmail.com](mailto:s.yavuz.ugurlu@gmail.com)*

```
```



