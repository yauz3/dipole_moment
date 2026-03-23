#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Ensemble Evaluator (Full-train + CV) for saved models:
  - 3D GNN (PyTorch / EGNNRegressor) saved as .pt (state_dict)
  - ANN Multi-view (Keras) saved as .keras
  - CatBoost saved as .cbm

What this script does
---------------------
Given:
  1) a list of model names (e.g., ["catboost","ann","3d_gnn"])
  2) a list of weights      (e.g., [0.4, 0.4, 0.2])

It loads:
  - full-train models for each model name
  - fold models for each model name (fold_1..K)

Then it computes:
  A) FULL TRAIN -> TEST ensemble metrics
  B) CV (per fold):
     - ensemble metrics on HOLDOUT (train fold's validation split)
     - ensemble metrics on TEST

And it saves exactly 3 CSVs:
  1) metrics_full_train_test.csv
  2) metrics_cv_holdout.csv          (fold rows + mean + std)
  3) metrics_cv_test.csv             (fold rows + mean + std)

It also saves "ensemble fold models" (not trainable) as JSON configs:
  - models_ensemble/ensemble_full.json
  - models_ensemble/ensemble_fold_{i}.json

IMPORTANT ASSUMPTION
--------------------
Your fold models were trained using:
  KFold(n_splits=cv, shuffle=True, random_state=seed)

So this script reproduces the SAME splits to evaluate fold models.

Expected artifact layout (defaults; you can override via CLI)
------------------------------------------------------------
CatBoost:
  <catboost_outdir>/models/catboost_model_full.cbm
  <catboost_outdir>/models/catboost_model_fold_{i}.cbm

ANN:
  <ann_outdir>/models/<ann_prefix>_full_train.keras
  <ann_outdir>/models/<ann_prefix>_fold_{i}.keras
  <ann_outdir>/features_avalon.csv
  <ann_outdir>/features_maccs.csv
  <ann_outdir>/scalers_full_train.npz          (optional; if missing => no scaling)

3D GNN:
  <gnn_outdir>/models/gnn3d_full_train.pt
  <gnn_outdir>/models/gnn3d_fold_{i}.pt
  <gnn_outdir>/full_train_test_metrics.json     (to get y_mean / y_std for inverse scaling)
  optional per-fold:
    <gnn_outdir>/fold_{i}_summary.json or <gnn_outdir>/models/gnn3d_fold_{i}_summary.json
    (if missing => falls back to full y_mean/y_std)

Dependencies
------------
pip install numpy pandas scikit-learn catboost lightgbm tensorflow torch rdkit-pypi

Example
-------
python weighted_ensemble_eval.py \
  --train_csv train_ready.csv --test_csv test_ready.csv \
  --model_names catboost,ann,3d_gnn \
  --model_weights 0.4,0.4,0.2 \
  --seed 42 --cv 5 \
  --catboost_outdir run_trainready_testready_full_then_cv \
  --ann_outdir run_multiview_ann_avalon_maccs_full_then_cv \
  --gnn_outdir run_gnn3d_fixed_nan \
  --outdir run_weighted_ensemble
"""

import os
import re
import json
import glob
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ---------- CatBoost ----------
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

# ---------- TensorFlow / Keras ----------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

# ---------- RDKit + Torch for GNN ----------
try:
    import torch
    import torch.nn as nn
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
except Exception:
    torch = None
    nn = None
    Chem = None
    AllChem = None


# =============================
# Common utilities
# =============================
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_csv_list(s: str) -> List[str]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def coerce_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    for c in X2.columns:
        if X2[c].dtype == "object":
            X2[c] = pd.to_numeric(X2[c], errors="coerce")
    X2 = X2.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X2


def metrics_reg(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "n": int(len(y_true)),
    }


def write_metrics_csv_with_mean_std(path: str, fold_metrics: list, tag: str):
    rows = []
    for i, m in enumerate(fold_metrics, start=1):
        r = {"tag": tag, "row_type": "fold", "fold": i}
        r.update(m)
        rows.append(r)

    keys = list(fold_metrics[0].keys())
    ddof = 1 if len(fold_metrics) > 1 else 0

    mean_row = {"tag": tag, "row_type": "mean", "fold": -1}
    std_row = {"tag": tag, "row_type": "std", "fold": -1}

    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics], dtype=float)
        mean_row[k] = float(np.mean(vals))
        std_row[k] = float(np.std(vals, ddof=ddof)) if len(vals) > 1 else 0.0

    rows.append(mean_row)
    rows.append(std_row)

    pd.DataFrame(rows).to_csv(path, index=False)


def discover_one(patterns: List[str]) -> Optional[str]:
    """Return first existing file from a list of glob patterns."""
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


# =============================
# Model wrappers
# =============================
class BaseWrapper:
    name: str

    def predict_on_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          y_pred: (N,) float array (undefined values ignored by mask)
          mask:   (N,) bool array where prediction is valid
        """
        raise NotImplementedError


# ---------- CatBoost wrapper ----------
class CatBoostWrapper(BaseWrapper):
    def __init__(self, model_path: str, feat_cols: List[str], name: str = "catboost"):
        if CatBoostRegressor is None:
            raise ImportError("catboost is required. Install with: pip install catboost")
        self.name = name
        self.model_path = model_path
        self.feat_cols = feat_cols
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)

    def predict_on_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = coerce_numeric_df(df[self.feat_cols])
        pred = self.model.predict(X)
        pred = np.asarray(pred, dtype=float).reshape(-1)
        mask = np.isfinite(pred)
        return pred, mask


# ---------- ANN wrapper ----------
class ANNWrapper(BaseWrapper):
    def __init__(
        self,
        model_path: str,
        avalon_cols: List[str],
        maccs_cols: List[str],
        scalers_npz: Optional[str],
        name: str = "ann"
    ):
        self.name = name
        self.model_path = model_path
        self.avalon_cols = avalon_cols
        self.maccs_cols = maccs_cols
        self.model = tf.keras.models.load_model(model_path)

        self.scalers_npz = scalers_npz
        self.use_scaler = False
        self.a_mean = None
        self.a_scale = None
        self.m_mean = None
        self.m_scale = None

        if scalers_npz is not None and os.path.exists(scalers_npz):
            d = np.load(scalers_npz)
            self.a_mean = d["avalon_mean"].astype(np.float64)
            self.a_scale = d["avalon_scale"].astype(np.float64)
            self.m_mean = d["maccs_mean"].astype(np.float64)
            self.m_scale = d["maccs_scale"].astype(np.float64)
            self.use_scaler = True

    def _scale(self, X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (X - mean) / np.where(scale == 0, 1.0, scale)

    def predict_on_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        Xa = coerce_numeric_df(df[self.avalon_cols]).values.astype(np.float32)
        Xm = coerce_numeric_df(df[self.maccs_cols]).values.astype(np.float32)

        if self.use_scaler:
            Xa = self._scale(Xa, self.a_mean, self.a_scale).astype(np.float32)
            Xm = self._scale(Xm, self.m_mean, self.m_scale).astype(np.float32)

        pred = self.model.predict(
            {"avalon_in": Xa, "maccs_in": Xm},
            batch_size=256,
            verbose=0
        ).reshape(-1)
        pred = np.asarray(pred, dtype=float)
        mask = np.isfinite(pred)
        return pred, mask


# ---------- 3D GNN wrapper (EGNNRegressor) ----------
DEFAULT_DISALLOWED_ATOMIC_NUMS = {
    2, 10, 18,
    3, 4, 11, 12, 13, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52,
    55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86,
}


def is_problematic_mol(mol, allow_metals: bool, allow_only_organic: bool, disallow_atomic_nums: set) -> Tuple[bool, str]:
    nums = [a.GetAtomicNum() for a in mol.GetAtoms()]

    if allow_only_organic:
        allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 34}
        bad = sorted(set([z for z in nums if z not in allowed]))
        if bad:
            return True, f"non_organic_atomic_nums={bad}"

    if not allow_metals:
        bads = sorted(set([z for z in nums if z in disallow_atomic_nums]))
        if bads:
            return True, f"disallowed_atomic_nums={bads}"

    return False, "ok"


def smiles_to_best_3d_mol(
    smiles: str,
    seed: int = 42,
    num_confs: int = 20,
    max_attempts: int = 30,
    add_hs: bool = True,
    optimize: bool = True,
    mmff_max_its: int = 300,
    allow_uff_fallback: bool = False,
):
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None

    try:
        mol = Chem.MolFromSmiles(s)
    except Exception:
        return None
    if mol is None:
        return None

    if add_hs:
        try:
            mol = Chem.AddHs(mol)
        except Exception:
            return None

    try:
        params = AllChem.ETKDGv3()
    except Exception:
        params = AllChem.ETKDG()
    params.randomSeed = int(seed)
    params.maxAttempts = int(max_attempts)

    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(num_confs), params=params))
    except Exception:
        conf_ids = []

    if not conf_ids:
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=int(seed))
        except Exception:
            res = -1
        if res == -1:
            try:
                res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=int(seed))
            except Exception:
                res = -1
            if res == -1:
                return None
        conf_ids = [mol.GetConformer().GetId()]

    best_id = conf_ids[0]
    best_e = None

    if optimize:
        props = None
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        except Exception:
            props = None

        if props is None and not allow_uff_fallback:
            optimize = False

        if optimize:
            for cid in conf_ids:
                e = None
                if props is not None:
                    try:
                        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(cid))
                        ff.Minimize(maxIts=int(mmff_max_its))
                        e = float(ff.CalcEnergy())
                    except Exception:
                        e = None
                if (e is None) and allow_uff_fallback:
                    try:
                        uff = AllChem.UFFGetMoleculeForceField(mol, confId=int(cid))
                        uff.Minimize(maxIts=int(mmff_max_its))
                        e = float(uff.CalcEnergy())
                    except Exception:
                        e = None
                if e is None:
                    continue
                if best_e is None or e < best_e:
                    best_e = e
                    best_id = cid

    try:
        conf = mol.GetConformer(int(best_id))
    except Exception:
        return None

    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_conf = Chem.Conformer(new_mol.GetNumAtoms())
    for i in range(new_mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        new_conf.SetAtomPosition(i, p)
    new_mol.AddConformer(new_conf, assignId=True)
    return new_mol


def bond_type_code(bt) -> float:
    if bt == Chem.rdchem.BondType.SINGLE:
        return 1.0
    if bt == Chem.rdchem.BondType.DOUBLE:
        return 2.0
    if bt == Chem.rdchem.BondType.TRIPLE:
        return 3.0
    if bt == Chem.rdchem.BondType.AROMATIC:
        return 1.5
    return 0.0


def mol_to_graph_3d(mol) -> Dict[str, "torch.Tensor"]:
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()

    z = torch.tensor([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)], dtype=torch.long)

    feats = []
    for i in range(n):
        a = mol.GetAtomWithIdx(i)
        feats.append([
            float(a.GetTotalDegree()),
            float(a.GetFormalCharge()),
            float(a.GetIsAromatic()),
            float(a.IsInRing()),
            float(a.GetTotalValence()),
            float(a.GetTotalNumHs()),
        ])
    x = torch.tensor(feats, dtype=torch.float32)

    pos = []
    for i in range(n):
        p = conf.GetAtomPosition(i)
        pos.append([p.x, p.y, p.z])
    pos = torch.tensor(pos, dtype=torch.float32)

    rows, cols, btypes = [], [], []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        code = bond_type_code(b.GetBondType())
        rows += [i, j]
        cols += [j, i]
        btypes += [code, code]

    if len(rows) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(btypes, dtype=torch.float32).view(-1, 1)

    return {"z": z, "x": x, "pos": pos, "edge_index": edge_index, "edge_attr": edge_attr}


@dataclass
class Batch:
    z: "torch.Tensor"
    x: "torch.Tensor"
    pos: "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_attr: "torch.Tensor"
    batch: "torch.Tensor"


def collate_graphs(graphs: List[Dict[str, "torch.Tensor"]]) -> Batch:
    zs, xs, poss, eis, eas, bs = [], [], [], [], [], []
    node_offset = 0
    for gi, g in enumerate(graphs):
        n = g["z"].shape[0]
        zs.append(g["z"])
        xs.append(g["x"])
        poss.append(g["pos"])

        ei = g["edge_index"]
        if ei.numel() > 0:
            eis.append(ei + node_offset)
            eas.append(g["edge_attr"])
        else:
            eis.append(ei)
            eas.append(g["edge_attr"])

        bs.append(torch.full((n,), fill_value=gi, dtype=torch.long))
        node_offset += n

    z = torch.cat(zs, dim=0)
    x = torch.cat(xs, dim=0)
    pos = torch.cat(poss, dim=0)

    edge_index = torch.cat(eis, dim=1) if len(eis) else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.cat(eas, dim=0) if len(eas) else torch.zeros((0, 1), dtype=torch.float32)
    batch = torch.cat(bs, dim=0)

    return Batch(z=z, x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def scatter_mean(src: "torch.Tensor", index: "torch.Tensor", dim_size: int) -> "torch.Tensor":
    out = torch.zeros((dim_size, src.size(1)), dtype=src.dtype, device=src.device)
    count = torch.zeros((dim_size, 1), dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.view(-1, 1).expand(-1, src.size(1)), src)
    ones = torch.ones((src.size(0), 1), dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index.view(-1, 1), ones)
    return out / count.clamp(min=1.0)


class EGNNLayer(nn.Module):
    def __init__(self, h_dim: int, edge_dim: int = 1):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * h_dim + 1 + edge_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1),
        )
        self.ln = nn.LayerNorm(h_dim)

    def forward(self, h, pos, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return self.ln(h), pos

        src, dst = edge_index[0], edge_index[1]

        h_i = h[dst]
        h_j = h[src]

        diff = pos[dst] - pos[src]
        r2 = (diff ** 2).sum(dim=1, keepdim=True)

        m_in = torch.cat([h_i, h_j, r2, edge_attr], dim=1)
        m_ij = self.phi_e(m_in)

        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.view(-1, 1).expand(-1, h.size(1)), m_ij)

        dh = self.phi_h(torch.cat([h, agg], dim=1))
        h = self.ln(h + dh)

        gate = torch.tanh(self.phi_x(m_ij))
        coord_update = diff * gate / (r2 + 1.0)

        dpos = torch.zeros_like(pos)
        dpos.scatter_add_(0, dst.view(-1, 1).expand(-1, 3), coord_update)
        pos = pos + dpos

        return h, pos


class EGNNRegressor(nn.Module):
    def __init__(self, h_dim: int = 128, n_layers: int = 4, z_emb_dim: int = 64, in_scalar_dim: int = 6):
        super().__init__()
        self.z_emb = nn.Embedding(120, z_emb_dim)

        self.in_proj = nn.Sequential(
            nn.Linear(z_emb_dim + in_scalar_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

        self.layers = nn.ModuleList([EGNNLayer(h_dim=h_dim, edge_dim=1) for _ in range(n_layers)])

        self.readout = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, batch: Batch):
        z, x, pos, edge_index, edge_attr, batch_index = (
            batch.z, batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch
        )

        # infer n_graphs from max(batch)+1
        n_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 1

        # center per graph
        mean_pos = scatter_mean(pos, batch_index, dim_size=n_graphs)  # (G,3)
        pos = pos - mean_pos[batch_index]

        h = torch.cat([self.z_emb(z), x], dim=1)
        h = self.in_proj(h)

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)

        hg = scatter_mean(h, batch_index, dim_size=n_graphs)
        out = self.readout(hg).view(-1)
        return out


class GNN3DWrapper(BaseWrapper):
    def __init__(
        self,
        model_path: str,
        y_mean: float,
        y_std: float,
        smiles_col: str = "SMILES",
        device: str = "cpu",
        name: str = "3d_gnn",
        seed_3d: int = 42,
        num_confs: int = 20,
        add_hs: bool = True,
        allow_metals: bool = False,
        allow_only_organic: bool = True,
        disallow_atomic_nums: Optional[set] = None,
        allow_uff_fallback: bool = False,
        batch_size: int = 32,
    ):
        if torch is None or Chem is None:
            raise ImportError("3d_gnn requires torch and rdkit. Install: pip install torch rdkit-pypi")

        self.name = name
        self.model_path = model_path
        self.smiles_col = smiles_col
        self.device = device
        self.seed_3d = int(seed_3d)
        self.num_confs = int(num_confs)
        self.add_hs = bool(add_hs)
        self.allow_metals = bool(allow_metals)
        self.allow_only_organic = bool(allow_only_organic)
        self.disallow_atomic_nums = disallow_atomic_nums if disallow_atomic_nums is not None else set(DEFAULT_DISALLOWED_ATOMIC_NUMS)
        self.allow_uff_fallback = bool(allow_uff_fallback)
        self.batch_size = int(batch_size)

        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

        self.model = EGNNRegressor(h_dim=128, n_layers=4).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        # allow either plain state_dict or {"state_dict": ...}
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _build_graphs(self, df: pd.DataFrame) -> Tuple[List[Optional[Dict[str, "torch.Tensor"]]], np.ndarray]:
        graphs = [None] * len(df)
        valid = np.zeros(len(df), dtype=bool)

        for i in range(len(df)):
            smi = df.iloc[i][self.smiles_col]
            if pd.isna(smi):
                continue

            try:
                mol0 = Chem.MolFromSmiles(str(smi))
            except Exception:
                mol0 = None
            if mol0 is None:
                continue

            bad, _ = is_problematic_mol(
                mol0,
                allow_metals=self.allow_metals,
                allow_only_organic=self.allow_only_organic,
                disallow_atomic_nums=self.disallow_atomic_nums
            )
            if bad:
                continue

            mol = smiles_to_best_3d_mol(
                smiles=str(smi),
                seed=self.seed_3d,
                num_confs=self.num_confs,
                add_hs=self.add_hs,
                optimize=True,
                allow_uff_fallback=self.allow_uff_fallback
            )
            if mol is None:
                continue

            try:
                g = mol_to_graph_3d(mol)
            except Exception:
                continue

            graphs[i] = g
            valid[i] = True

        return graphs, valid

    def predict_on_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        graphs, valid = self._build_graphs(df)

        pred = np.full((len(df),), np.nan, dtype=float)
        idxs = np.where(valid)[0]
        if len(idxs) == 0:
            return pred, np.zeros(len(df), dtype=bool)

        with torch.no_grad():
            for start in range(0, len(idxs), self.batch_size):
                batch_ids = idxs[start:start + self.batch_size]
                gs = [graphs[int(j)] for j in batch_ids]
                # safety
                gs = [g for g in gs if g is not None]
                if not gs:
                    continue

                b = collate_graphs(gs)
                b = Batch(
                    z=b.z.to(self.device),
                    x=b.x.to(self.device),
                    pos=b.pos.to(self.device),
                    edge_index=b.edge_index.to(self.device),
                    edge_attr=b.edge_attr.to(self.device),
                    batch=b.batch.to(self.device),
                )
                out_std = self.model(b).detach().cpu().numpy().reshape(-1)
                out = out_std * self.y_std + self.y_mean

                # assign in order (gs aligns to batch_ids after filtering None; but we filtered none => safe)
                for k, row_idx in enumerate(batch_ids):
                    pred[int(row_idx)] = float(out[k])

        mask = np.isfinite(pred)
        return pred, mask


# =============================
# Ensemble logic
# =============================
def weighted_ensemble(preds: List[np.ndarray], masks: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted sum on intersection where all masks are True.
    Returns:
      y_pred_ens: (N,) with NaN where invalid
      mask_ens:   (N,) True where valid (intersection)
    """
    if not preds:
        raise ValueError("No predictions provided to ensemble.")

    N = len(preds[0])
    for p in preds:
        if len(p) != N:
            raise ValueError("Prediction arrays have mismatched lengths.")

    mask_all = np.ones(N, dtype=bool)
    for m in masks:
        mask_all &= m

    out = np.full((N,), np.nan, dtype=float)
    if mask_all.sum() == 0:
        return out, mask_all

    w = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(w)) or np.allclose(w.sum(), 0.0):
        raise ValueError("Invalid weights (non-finite or sum=0).")

    # fixed weights; no renormalization (we use intersection mask)
    num = np.zeros(mask_all.sum(), dtype=float)
    for p, wi in zip(preds, w):
        num += wi * p[mask_all]

    out[mask_all] = num
    return out, mask_all


# =============================
# Loading helpers per model type
# =============================
def load_catboost_wrapper(catboost_outdir: str, fold: Optional[int], feat_cols: List[str]) -> CatBoostWrapper:
    models_dir = os.path.join(catboost_outdir, "models")
    if fold is None:
        path = discover_one([
            os.path.join(models_dir, "catboost_model_full*.cbm"),
            os.path.join(models_dir, "*full*.cbm"),
            os.path.join(catboost_outdir, "catboost_model_full*.cbm"),
        ])
    else:
        path = discover_one([
            os.path.join(models_dir, f"catboost_model_fold_{fold}.cbm"),
            os.path.join(models_dir, f"*fold_{fold}*.cbm"),
        ])
    if path is None:
        raise FileNotFoundError(f"[catboost] model not found (fold={fold}). Check {models_dir}")
    return CatBoostWrapper(model_path=path, feat_cols=feat_cols, name="catboost")


def load_ann_wrapper(ann_outdir: str, fold: Optional[int], ann_prefix: str) -> ANNWrapper:
    models_dir = os.path.join(ann_outdir, "models")

    # feature lists
    avalon_f = os.path.join(ann_outdir, "features_avalon.csv")
    maccs_f = os.path.join(ann_outdir, "features_maccs.csv")
    if not os.path.exists(avalon_f) or not os.path.exists(maccs_f):
        raise FileNotFoundError(f"[ann] features_avalon.csv / features_maccs.csv not found in {ann_outdir}")

    avalon_cols = pd.read_csv(avalon_f)["feature"].tolist()
    maccs_cols = pd.read_csv(maccs_f)["feature"].tolist()

    scalers_npz = discover_one([
        os.path.join(ann_outdir, "scalers_full_train.npz"),
        os.path.join(ann_outdir, "scalers_full_train*.npz"),
    ])

    if fold is None:
        path = discover_one([
            os.path.join(models_dir, f"{ann_prefix}_full_train.keras"),
            os.path.join(models_dir, "*full_train*.keras"),
            os.path.join(models_dir, "*full*.keras"),
        ])
    else:
        path = discover_one([
            os.path.join(models_dir, f"{ann_prefix}_fold_{fold}.keras"),
            os.path.join(models_dir, f"*fold_{fold}.keras"),
        ])
    if path is None:
        raise FileNotFoundError(f"[ann] model not found (fold={fold}). Check {models_dir}")

    return ANNWrapper(model_path=path, avalon_cols=avalon_cols, maccs_cols=maccs_cols, scalers_npz=scalers_npz, name="ann")


def _load_gnn_y_stats(gnn_outdir: str, fold: Optional[int]) -> Tuple[float, float]:
    """
    Prefer fold-specific summary if present; else fallback to full_train_test_metrics.json.
    """
    # fold-specific candidates
    cand = []
    if fold is not None:
        cand += [
            os.path.join(gnn_outdir, f"fold_{fold}_summary.json"),
            os.path.join(gnn_outdir, "models", f"gnn3d_fold_{fold}_summary.json"),
            os.path.join(gnn_outdir, f"fold_{fold}.json"),
        ]
    cand += [
        os.path.join(gnn_outdir, "full_train_test_metrics.json"),
        os.path.join(gnn_outdir, "full_train_test_metrics", "metrics.json"),
    ]

    for p in cand:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            # accept either {"train_summary":{"y_mean":..,"y_std":..}} or {"y_mean":..,"y_std":..}
            if "train_summary" in d and isinstance(d["train_summary"], dict):
                ts = d["train_summary"]
                if "y_mean" in ts and "y_std" in ts:
                    return float(ts["y_mean"]), float(ts["y_std"])
            if "y_mean" in d and "y_std" in d:
                return float(d["y_mean"]), float(d["y_std"])
            if "metrics" in d and "train_summary" in d:
                ts = d["train_summary"]
                if "y_mean" in ts and "y_std" in ts:
                    return float(ts["y_mean"]), float(ts["y_std"])

    # final fallback
    return 0.0, 1.0


def load_gnn_wrapper(gnn_outdir: str, fold: Optional[int], smiles_col: str, device: str) -> GNN3DWrapper:
    models_dir = os.path.join(gnn_outdir, "models")
    if fold is None:
        path = discover_one([
            os.path.join(models_dir, "gnn3d_full_train*.pt"),
            os.path.join(models_dir, "*full*.pt"),
            os.path.join(gnn_outdir, "gnn3d_full_train*.pt"),
        ])
    else:
        path = discover_one([
            os.path.join(models_dir, f"gnn3d_fold_{fold}.pt"),
            os.path.join(models_dir, f"*fold_{fold}*.pt"),
        ])
    if path is None:
        raise FileNotFoundError(f"[3d_gnn] model not found (fold={fold}). Check {models_dir}")

    y_mean, y_std = _load_gnn_y_stats(gnn_outdir, fold=fold)

    return GNN3DWrapper(
        model_path=path,
        y_mean=y_mean,
        y_std=y_std,
        smiles_col=smiles_col,
        device=device,
        name="3d_gnn",
        seed_3d=42,
        num_confs=20,
        add_hs=True,
        allow_metals=False,
        allow_only_organic=True,
        disallow_atomic_nums=set(DEFAULT_DISALLOWED_ATOMIC_NUMS),
        allow_uff_fallback=False,
        batch_size=32,
    )


def build_wrappers_for_fold(
    model_names: List[str],
    weights: List[float],
    fold: Optional[int],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    exclude: List[str],
    catboost_outdir: str,
    ann_outdir: str,
    gnn_outdir: str,
    ann_prefix: str,
    smiles_col_gnn: str,
    device_gnn: str
) -> Tuple[List[BaseWrapper], List[float], Dict]:
    """
    Loads wrappers in the provided model order.
    Returns wrappers, weights (same order), and a metadata dict for saving config.
    """
    wrappers: List[BaseWrapper] = []
    meta = {"fold": fold, "models": []}

    exclude_set = set(exclude)

    # CatBoost feature columns (from TRAIN schema, enforced on TEST)
    feat_cols_train = [c for c in train_df.columns if c not in exclude_set]
    feat_cols_test = [c for c in test_df.columns if c not in exclude_set]
    if set(feat_cols_train) != set(feat_cols_test):
        missing_in_test = sorted(list(set(feat_cols_train) - set(feat_cols_test)))
        missing_in_train = sorted(list(set(feat_cols_test) - set(feat_cols_train)))
        raise ValueError(
            "Train/Test feature columns mismatch for tabular models.\n"
            f"Missing in test: {missing_in_test[:25]}{'...' if len(missing_in_test)>25 else ''}\n"
            f"Missing in train: {missing_in_train[:25]}{'...' if len(missing_in_train)>25 else ''}"
        )
    feat_cols = sorted(feat_cols_train)

    for nm in model_names:
        nm_l = nm.strip().lower()
        if nm_l in ["catboost", "cb"]:
            w = load_catboost_wrapper(catboost_outdir, fold=fold, feat_cols=feat_cols)
            wrappers.append(w)
            meta["models"].append({"name": "catboost", "type": "catboost"})
        elif nm_l in ["ann", "mlp", "multiview_ann"]:
            w = load_ann_wrapper(ann_outdir, fold=fold, ann_prefix=ann_prefix)
            wrappers.append(w)
            meta["models"].append({"name": "ann", "type": "keras"})
        elif nm_l in ["3d_gnn", "gnn", "egnn", "3d"]:
            w = load_gnn_wrapper(gnn_outdir, fold=fold, smiles_col=smiles_col_gnn, device=device_gnn)
            wrappers.append(w)
            meta["models"].append({"name": "3d_gnn", "type": "torch"})
        else:
            raise ValueError(f"Unknown model name '{nm}'. Use: catboost, ann, 3d_gnn")

    # validate weights length
    if len(weights) != len(wrappers):
        raise ValueError(f"weights length ({len(weights)}) != models length ({len(wrappers)})")

    # store weights in meta
    for i, wi in enumerate(weights):
        meta["models"][i]["weight"] = float(wi)

    return wrappers, weights, meta


# =============================
# MAIN
# =============================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", default="train_ready.csv", type=str)
    ap.add_argument("--test_csv", default="test_ready.csv", type=str)
    ap.add_argument("--outdir", default="run_weighted_ensemble", type=str)

    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--cv", default=5, type=int)

    ap.add_argument("--target", default="Dipole momentμ debye", type=str)
    ap.add_argument("--exclude", nargs="*", default=["Name", "Formula", "SMILES", "Dipole momentμ debye"])

    ap.add_argument("--model_names", default="catboost,3d_gnn,ann", type=str,
                    help="Comma-separated list: catboost,ann,") # "catboost,ann,3d_gnn"  "catboost,3d_gnn,ann", 0.4,0.4,0.2 -- 0.821 0.42
    ap.add_argument("--model_weights", default="0.33333333333333333333,0.33333333333333333333,0.33333333333333333333", type=str, # 0.75,0.25, 0.5     0.33333333,0.333333,0.33333333
                    help="Comma-separated weights aligned with model_names")
   
    # model roots (you can point to your real saved folders)
    ap.add_argument("--catboost_outdir", default="run_trainready_testready_full_then_cv", type=str)
    ap.add_argument("--ann_outdir", default="run_multiview_ann_avalon_maccs_full_then_cv", type=str)
    ap.add_argument("--gnn_outdir", default="run_gnn3d_fixed_nan", type=str)

    # ANN naming
    ap.add_argument("--ann_prefix", default="multiview_ann_model", type=str)

    # GNN I/O
    ap.add_argument("--smiles_col_gnn", default="SMILES", type=str)
    ap.add_argument("--device_gnn", default="cpu", choices=["cpu", "cuda"])

    args = ap.parse_args()
    safe_mkdir(args.outdir)
    ens_models_dir = os.path.join(args.outdir, "models_ensemble")
    safe_mkdir(ens_models_dir)

    model_names = parse_csv_list(args.model_names)
    weights = parse_float_list(args.model_weights)
    if len(model_names) == 0:
        raise ValueError("model_names is empty.")
    if len(weights) == 0:
        raise ValueError("model_weights is empty.")
    if len(model_names) != len(weights):
        raise ValueError("model_names and model_weights must have same length.")
    if not np.isfinite(np.asarray(weights)).all():
        raise ValueError("Non-finite weights found.")
    if np.isclose(np.sum(weights), 0.0):
        raise ValueError("Sum of weights is 0.")

    SEED = int(args.seed)

    # -----------------------------
    # Load data
    # -----------------------------
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    test_df = pd.read_csv(args.test_csv, low_memory=False)

    if args.target not in train_df.columns:
        raise ValueError(f"Target column not found in train: {args.target}")
    if args.target not in test_df.columns:
        raise ValueError(f"Target column not found in test: {args.target}")

    ytr = pd.to_numeric(train_df[args.target], errors="coerce")
    yte = pd.to_numeric(test_df[args.target], errors="coerce")

    # drop NaN targets (we evaluate on those with labels)
    mtr = ytr.notna().values
    mte = yte.notna().values

    train_df_eval = train_df.loc[mtr].reset_index(drop=True)
    test_df_eval = test_df.loc[mte].reset_index(drop=True)
    ytr_eval = ytr.loc[mtr].reset_index(drop=True).values.astype(float)
    yte_eval = yte.loc[mte].reset_index(drop=True).values.astype(float)

    print(f"[Train eval] rows={len(train_df_eval)}")
    print(f"[Test  eval] rows={len(test_df_eval)}")
    print(f"[Models] {model_names}")
    print(f"[Weights] {weights}")
    print(f"[CV] n_splits={args.cv} seed={SEED}")

    # -----------------------------
    # (A) FULL TRAIN -> TEST ensemble
    # -----------------------------
    wrappers_full, w_full, meta_full = build_wrappers_for_fold(
        model_names=model_names,
        weights=weights,
        fold=None,
        train_df=train_df_eval,
        test_df=test_df_eval,
        target=args.target,
        exclude=args.exclude,
        catboost_outdir=args.catboost_outdir,
        ann_outdir=args.ann_outdir,
        gnn_outdir=args.gnn_outdir,
        ann_prefix=args.ann_prefix,
        smiles_col_gnn=args.smiles_col_gnn,
        device_gnn=args.device_gnn
    )

    preds = []
    masks = []
    for w in wrappers_full:
        p, m = w.predict_on_df(test_df_eval)
        preds.append(p)
        masks.append(m)

    p_ens, m_ens = weighted_ensemble(preds, masks, w_full)
    if m_ens.sum() == 0:
        raise RuntimeError("FULL ensemble has zero valid rows (intersection is empty).")

    met_full = metrics_reg(yte_eval[m_ens], p_ens[m_ens])
    print("\n==================== ENSEMBLE: FULL TRAIN -> TEST ====================")
    print(json.dumps(met_full, indent=2))
    print(f"[FULL] n_used={int(m_ens.sum())} / {len(test_df_eval)}")

    # Save FULL metrics CSV (single row)
    full_metrics_csv = os.path.join(args.outdir, "metrics_full_train_test.csv")
    full_row = {"tag": "full_train_test", "row_type": "single", "fold": -1, "seed": SEED}
    full_row.update(met_full)
    pd.DataFrame([full_row]).to_csv(full_metrics_csv, index=False)

    # Save FULL ensemble predictions (useful)
    pd.DataFrame({
        "y_true": yte_eval.astype(float),
        "y_pred_ensemble": p_ens.astype(float),
        "valid_mask": m_ens.astype(int),
    }).to_csv(os.path.join(args.outdir, "predictions_full_train_test_ensemble.csv"), index=False)

    # Save FULL ensemble "model" config
    meta_full_out = dict(meta_full)
    meta_full_out.update({
        "seed": SEED,
        "cv": int(args.cv),
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "target": args.target,
        "n_test_total": int(len(test_df_eval)),
        "n_test_used": int(m_ens.sum()),
        "metrics": met_full,
    })
    with open(os.path.join(ens_models_dir, "ensemble_full.json"), "w", encoding="utf-8") as f:
        json.dump(meta_full_out, f, indent=2)

    # -----------------------------
    # (B) CV evaluation using saved fold models
    # -----------------------------
    kf = KFold(n_splits=int(args.cv), shuffle=True, random_state=SEED)

    fold_hold_metrics = []
    fold_test_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df_eval), start=1):
        hold_df = train_df_eval.iloc[va_idx].reset_index(drop=True)
        y_hold = ytr_eval[va_idx].astype(float)

        wrappers_fold, w_fold, meta_fold = build_wrappers_for_fold(
            model_names=model_names,
            weights=weights,
            fold=fold,
            train_df=train_df_eval,
            test_df=test_df_eval,
            target=args.target,
            exclude=args.exclude,
            catboost_outdir=args.catboost_outdir,
            ann_outdir=args.ann_outdir,
            gnn_outdir=args.gnn_outdir,
            ann_prefix=args.ann_prefix,
            smiles_col_gnn=args.smiles_col_gnn,
            device_gnn=args.device_gnn
        )

        # HOLDOUT predictions
        preds_h = []
        masks_h = []
        for w in wrappers_fold:
            p, m = w.predict_on_df(hold_df)
            preds_h.append(p)
            masks_h.append(m)
        p_ens_h, m_ens_h = weighted_ensemble(preds_h, masks_h, w_fold)

        if m_ens_h.sum() == 0:
            met_h = {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "n": 0}
        else:
            met_h = metrics_reg(y_hold[m_ens_h], p_ens_h[m_ens_h])

        # TEST predictions
        preds_t = []
        masks_t = []
        for w in wrappers_fold:
            p, m = w.predict_on_df(test_df_eval)
            preds_t.append(p)
            masks_t.append(m)
        p_ens_t, m_ens_t = weighted_ensemble(preds_t, masks_t, w_fold)

        if m_ens_t.sum() == 0:
            met_t = {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "n": 0}
        else:
            met_t = metrics_reg(yte_eval[m_ens_t], p_ens_t[m_ens_t])

        fold_hold_metrics.append(met_h)
        fold_test_metrics.append(met_t)

        # Save fold ensemble config (the "fold model")
        meta_fold_out = dict(meta_fold)
        meta_fold_out.update({
            "seed": SEED,
            "cv": int(args.cv),
            "fold": int(fold),
            "holdout_n_total": int(len(hold_df)),
            "holdout_n_used": int(m_ens_h.sum()),
            "test_n_total": int(len(test_df_eval)),
            "test_n_used": int(m_ens_t.sum()),
            "metrics_holdout": met_h,
            "metrics_test": met_t,
        })
        with open(os.path.join(ens_models_dir, f"ensemble_fold_{fold}.json"), "w", encoding="utf-8") as f:
            json.dump(meta_fold_out, f, indent=2)

        print(
            f"[fold {fold}] "
            f"holdout: R2={met_h['R2']:.4f} MAE={met_h['MAE']:.4f} RMSE={met_h['RMSE']:.4f} (n={met_h['n']}) | "
            f"test: R2={met_t['R2']:.4f} MAE={met_t['MAE']:.4f} RMSE={met_t['RMSE']:.4f} (n={met_t['n']})"
        )

    # Save the 2 CV metric CSVs (fold rows + mean + std)
    holdout_csv = os.path.join(args.outdir, "metrics_cv_holdout.csv")
    test_csv = os.path.join(args.outdir, "metrics_cv_test.csv")

    # If some folds are NaN (e.g., zero n), writing mean/std still OK; keep as float
    # For mean/std, we only compute over finite values per column? -> keep simple numeric mean; user can filter.
    write_metrics_csv_with_mean_std(holdout_csv, fold_hold_metrics, tag="cv_holdout")
    write_metrics_csv_with_mean_std(test_csv, fold_test_metrics, tag="cv_test")

    # Save summary JSON too (optional)
    cv_summary = {
        "seed": SEED,
        "cv_n_splits": int(args.cv),
        "model_names": model_names,
        "model_weights": weights,
        "metrics_full_train_test": met_full,
        "csv_full_train_test": full_metrics_csv,
        "csv_cv_holdout": holdout_csv,
        "csv_cv_test": test_csv,
        "ensemble_models_dir": ens_models_dir,
    }
    with open(os.path.join(args.outdir, "cv_summary_ensemble.json"), "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    print("\n[Saved]")
    print(full_metrics_csv)
    print(holdout_csv)
    print(test_csv)
    print(os.path.join(args.outdir, "predictions_full_train_test_ensemble.csv"))
    print(os.path.join(args.outdir, "cv_summary_ensemble.json"))
    print(ens_models_dir)


if __name__ == "__main__":
    main()
