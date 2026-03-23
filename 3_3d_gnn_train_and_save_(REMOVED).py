#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/12/2025
# Author: Sadettin Y. Ugurlu

"""
3D Conformer + Stabilized EGNN-style GNN Regressor (RDKit -> 3D -> Graph -> PyTorch)

Includes:
- NaN/Inf-safe metrics + training skips for unstable batches
- Stabilized EGNN coordinate updates (tanh gate + distance damping + LayerNorm)
- Per-graph coordinate centering
- FULL-TRAIN -> TEST evaluation + prediction CSV
- KFold CV
- NEW: CV reporting outputs (3 CSVs) + model checkpoint saving:
    1) cv_test_metrics_per_fold.csv
    2) cv_test_metrics_summary_mean_std.csv
    3) cv_test_predictions_per_fold.csv
  and checkpoints:
    - models/model_full_train.pt
    - models/fold_1.pt ... fold_k.pt

Run:
  pip install pandas numpy scikit-learn torch rdkit-pypi
  python 3d_gnn_fixed.py --train_csv train_ready.csv --test_csv test_ready.csv --device cpu

Columns:
  smiles_col = "SMILES"
  target_col = "Dipole momentμ debye"
"""

import os
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# =========================================================
# RDKit logging: disable noisy warnings/errors (UFFTYPER etc.)
# =========================================================
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


# -----------------------------
# Utils
# -----------------------------
def safe_mkdir(p: str):
    if p is None or str(p).strip() == "":
        return
    os.makedirs(p, exist_ok=True)


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics_reg(y_true, y_pred):
    """
    NaN/Inf-safe regression metrics.
    Filters non-finite values instead of crashing.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(m.sum())

    if n == 0:
        return {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "n": 0}

    yt = y_true[m]
    yp = y_pred[m]

    mse = mean_squared_error(yt, yp)
    rmse = float(np.sqrt(mse))
    return {
        "R2": float(r2_score(yt, yp)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": rmse,
        "n": n,
    }


def mean_of_metrics(metric_dicts):
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in metric_dicts if np.isfinite(d[k])]
        out[k] = float(np.mean(vals)) if len(vals) else float("nan")
    return out


def mean_std_of_metrics(metric_dicts, ddof: int = 1):
    """
    Compute mean and std across folds for each metric key.
    ddof=1 -> sample std (recommended for CV); ddof=0 -> population std.
    """
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in metric_dicts if np.isfinite(d[k])]
        if len(vals) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=ddof)) if len(vals) > 1 else 0.0,
            }
    return out


def save_torch_model(path: str, model: nn.Module, extra: dict):
    """
    Save model state + metadata. This is a lightweight checkpoint:
    - state_dict
    - y_mean/y_std for de-standardization
    - hyperparams and metrics, etc.
    """
    safe_mkdir(os.path.dirname(path))
    payload = {
        "state_dict": model.state_dict(),
        **(extra or {}),
    }
    torch.save(payload, path)


# =========================================================
# Chemistry filters (recommended for "organic molecules" tasks)
# =========================================================
DEFAULT_DISALLOWED_ATOMIC_NUMS = {
    # metals / noble gases that often break FF typing
    2, 10, 18,  # He, Ne, Ar
    3, 4, 11, 12, 13, 19, 20,  # Li Be Na Mg Al K Ca
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # Sc..Zn
    31, 32, 33,  # Ga Ge As
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52,
    55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86,
}


def is_problematic_mol(mol: Chem.Mol,
                       allow_metals: bool,
                       allow_only_organic: bool,
                       disallow_atomic_nums: set) -> Tuple[bool, str]:
    nums = [a.GetAtomicNum() for a in mol.GetAtoms()]

    if allow_only_organic:
        # typical organic QSAR set
        allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 34}  # H,B,C,N,O,F,Si,P,S,Cl,Br,I,Se
        bad = sorted(set([z for z in nums if z not in allowed]))
        if bad:
            return True, f"non_organic_atomic_nums={bad}"

    if not allow_metals:
        bads = sorted(set([z for z in nums if z in disallow_atomic_nums]))
        if bads:
            return True, f"disallowed_atomic_nums={bads}"

    return False, "ok"


# -----------------------------
# RDKit 3D: robust conformer selection
# -----------------------------
def smiles_to_best_3d_mol(
    smiles: str,
    seed: int = 42,
    num_confs: int = 20,
    max_attempts: int = 30,
    add_hs: bool = True,
    optimize: bool = True,
    mmff_max_its: int = 300,
    allow_uff_fallback: bool = False,
) -> Optional[Chem.Mol]:
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


def mol_to_graph_3d(mol: Chem.Mol) -> Dict[str, torch.Tensor]:
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


# -----------------------------
# Batch (PyG-like without torch_geometric)
# -----------------------------
@dataclass
class Batch:
    z: torch.Tensor
    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    y: torch.Tensor


def collate_graphs(items: List[Tuple[Dict[str, torch.Tensor], float]]) -> Batch:
    zs, xs, poss, eis, eas, bs, ys = [], [], [], [], [], [], []
    node_offset = 0
    for g, y in items:
        n = g["z"].shape[0]
        zs.append(g["z"])
        xs.append(g["x"])
        poss.append(g["pos"])
        ys.append(torch.tensor([float(y)], dtype=torch.float32))

        ei = g["edge_index"]
        if ei.numel() > 0:
            eis.append(ei + node_offset)
            eas.append(g["edge_attr"])
        else:
            eis.append(ei)
            eas.append(g["edge_attr"])

        bs.append(torch.full((n,), fill_value=len(ys) - 1, dtype=torch.long))
        node_offset += n

    z = torch.cat(zs, dim=0)
    x = torch.cat(xs, dim=0)
    pos = torch.cat(poss, dim=0)

    edge_index = torch.cat(eis, dim=1) if len(eis) else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.cat(eas, dim=0) if len(eas) else torch.zeros((0, 1), dtype=torch.float32)

    batch = torch.cat(bs, dim=0)
    y = torch.cat(ys, dim=0).view(-1)

    return Batch(z=z, x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=y)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, src.size(1)), dtype=src.dtype, device=src.device)
    count = torch.zeros((dim_size, 1), dtype=src.dtype, device=src.device)

    out.scatter_add_(0, index.view(-1, 1).expand(-1, src.size(1)), src)
    ones = torch.ones((src.size(0), 1), dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index.view(-1, 1), ones)

    return out / count.clamp(min=1.0)


# -----------------------------
# Stabilized EGNN-style layers
# -----------------------------
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

        diff = pos[dst] - pos[src]                   # (E,3)
        r2 = (diff ** 2).sum(dim=1, keepdim=True)    # (E,1)

        m_in = torch.cat([h_i, h_j, r2, edge_attr], dim=1)
        m_ij = self.phi_e(m_in)                      # (E,H)

        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.view(-1, 1).expand(-1, h.size(1)), m_ij)

        dh = self.phi_h(torch.cat([h, agg], dim=1))
        h = self.ln(h + dh)

        # Stabilized coordinate updates:
        gate = torch.tanh(self.phi_x(m_ij))          # (E,1) bounded
        coord_update = diff * gate / (r2 + 1.0)      # damp far updates

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

        n_graphs = int(batch.y.shape[0])

        # Center coordinates per graph (very important for stability)
        mean_pos = scatter_mean(pos, batch_index, dim_size=n_graphs)  # (G,3)
        pos = pos - mean_pos[batch_index]

        h = torch.cat([self.z_emb(z), x], dim=1)
        h = self.in_proj(h)

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)

        hg = scatter_mean(h, batch_index, dim_size=n_graphs)
        out = self.readout(hg).view(-1)
        return out


# -----------------------------
# Data preparation (cache aligned to row index)
# -----------------------------
def build_graph_cache(
    df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    seed: int,
    num_confs: int,
    add_hs: bool,
    allow_metals: bool,
    allow_only_organic: bool,
    disallow_atomic_nums: set,
    allow_uff_fallback: bool,
) -> Tuple[List[Optional[Dict[str, torch.Tensor]]], np.ndarray, Dict[str, int]]:
    graphs = [None] * len(df)
    valid = np.zeros(len(df), dtype=bool)

    stats = {
        "ok": 0,
        "nan_smiles_or_target": 0,
        "mol_parse_fail": 0,
        "filtered_problematic": 0,
        "embed_fail": 0,
        "graph_fail": 0,
    }
    reason_counts = {}

    for i in range(len(df)):
        smi = df.iloc[i][smiles_col]
        y = df.iloc[i][target_col]

        if pd.isna(smi) or pd.isna(y):
            stats["nan_smiles_or_target"] += 1
            continue

        try:
            mol0 = Chem.MolFromSmiles(str(smi))
        except Exception:
            mol0 = None

        if mol0 is None:
            stats["mol_parse_fail"] += 1
            continue

        bad, reason = is_problematic_mol(
            mol0,
            allow_metals=allow_metals,
            allow_only_organic=allow_only_organic,
            disallow_atomic_nums=disallow_atomic_nums
        )
        if bad:
            stats["filtered_problematic"] += 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue

        mol = smiles_to_best_3d_mol(
            smiles=str(smi),
            seed=seed,
            num_confs=num_confs,
            add_hs=add_hs,
            optimize=True,
            allow_uff_fallback=allow_uff_fallback,
        )
        if mol is None:
            stats["embed_fail"] += 1
            continue

        try:
            g = mol_to_graph_3d(mol)
        except Exception:
            stats["graph_fail"] += 1
            continue

        graphs[i] = g
        valid[i] = True
        stats["ok"] += 1

    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
        stats[f"reason::{k}"] = int(v)

    return graphs, valid, stats


def index_to_items(
    df: pd.DataFrame,
    graphs_aligned: List[Optional[Dict[str, torch.Tensor]]],
    idxs: np.ndarray,
    target_col: str,
) -> List[Tuple[Dict[str, torch.Tensor], float]]:
    items = []
    for i in idxs:
        g = graphs_aligned[int(i)]
        if g is None:
            continue
        y = df.iloc[int(i)][target_col]
        if pd.isna(y):
            continue
        items.append((g, float(y)))
    return items


# -----------------------------
# Training loop with early stopping + NaN-safe batches
# -----------------------------
def train_one_model(
    train_items: List[Tuple[Dict[str, torch.Tensor], float]],
    val_items: Optional[List[Tuple[Dict[str, torch.Tensor], float]]],
    device: str,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    h_dim: int,
    n_layers: int,
    clip_grad: float,
    standardize_y: bool,
) -> Tuple[EGNNRegressor, Dict]:
    seed_everything(seed)
    model = EGNNRegressor(h_dim=h_dim, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    if standardize_y:
        y_tr = np.array([y for _, y in train_items], dtype=float)
        y_mean = float(np.mean(y_tr))
        y_std = float(np.std(y_tr) + 1e-8)
    else:
        y_mean, y_std = 0.0, 1.0

    def transform_y(y: torch.Tensor) -> torch.Tensor:
        return (y - y_mean) / y_std

    def inverse_y(y_hat: np.ndarray) -> np.ndarray:
        return y_hat * y_std + y_mean

    nan_skips = {"train": 0, "val": 0}

    def iterate(items, training: bool):
        if training:
            model.train()
        else:
            model.eval()

        idxs = np.arange(len(items))
        if training:
            np.random.shuffle(idxs)

        preds_all = []
        y_all = []
        losses = []

        for start in range(0, len(items), batch_size):
            b_idx = idxs[start:start + batch_size]
            chunk = [items[i] for i in b_idx]
            b = collate_graphs(chunk)

            b = Batch(
                z=b.z.to(device),
                x=b.x.to(device),
                pos=b.pos.to(device),
                edge_index=b.edge_index.to(device),
                edge_attr=b.edge_attr.to(device),
                batch=b.batch.to(device),
                y=b.y.to(device),
            )

            y_t = transform_y(b.y)

            with torch.set_grad_enabled(training):
                pred_t = model(b)

                # Skip non-finite predictions (prevents NaN metrics + exploding training)
                if (not torch.isfinite(pred_t).all()) or (not torch.isfinite(y_t).all()):
                    nan_skips["train" if training else "val"] += 1
                    continue

                loss = loss_fn(pred_t, y_t)

                # Skip non-finite loss
                if not torch.isfinite(loss).all():
                    nan_skips["train" if training else "val"] += 1
                    continue

                if training:
                    opt.zero_grad()
                    loss.backward()
                    if clip_grad is not None and clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    opt.step()

            losses.append(float(loss.detach().cpu().item()))

            pred_np = pred_t.detach().cpu().numpy()
            y_np = b.y.detach().cpu().numpy()

            preds_all.append(inverse_y(pred_np))
            y_all.append(y_np)

        y_all = np.concatenate(y_all, axis=0) if y_all else np.array([])
        preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.array([])
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        return mean_loss, y_all, preds_all

    best_state = None
    best_val_r2 = -1e18
    best_epoch = 0
    bad_epochs = 0
    history = []

    for ep in range(1, epochs + 1):
        tr_loss, y_tr_np, p_tr_np = iterate(train_items, training=True)
        tr_met = metrics_reg(y_tr_np, p_tr_np)

        if val_items is not None and len(val_items) > 0:
            va_loss, y_va_np, p_va_np = iterate(val_items, training=False)
            va_met = metrics_reg(y_va_np, p_va_np)
            val_r2 = va_met["R2"]
        else:
            va_loss, va_met, val_r2 = float("nan"), None, tr_met["R2"]

        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_metrics": tr_met,
            "val_loss": va_loss,
            "val_metrics": va_met,
        })

        if va_met is not None:
            print(f"[Epoch {ep:03d}] train R2={tr_met['R2']:.4f} MAE={tr_met['MAE']:.4f} (n={tr_met['n']}) | "
                  f"val R2={va_met['R2']:.4f} MAE={va_met['MAE']:.4f} (n={va_met['n']}) | "
                  f"nan_skips(train/val)={nan_skips['train']}/{nan_skips['val']}")
        else:
            print(f"[Epoch {ep:03d}] train R2={tr_met['R2']:.4f} MAE={tr_met['MAE']:.4f} (n={tr_met['n']}) | "
                  f"nan_skips(train)={nan_skips['train']}")

        improved = np.isfinite(val_r2) and (val_r2 > best_val_r2 + 1e-6)
        if improved:
            best_val_r2 = float(val_r2)
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if val_items is not None and patience > 0 and bad_epochs >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs. best_epoch={best_epoch}, best_val_R2={best_val_r2:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_r2": float(best_val_r2),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "nan_skips": nan_skips,
        "history_tail": history[-5:],
    }
    return model, summary


def predict_items(model: EGNNRegressor, items: List[Tuple[Dict[str, torch.Tensor], float]], device: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            b = collate_graphs(chunk)
            b = Batch(
                z=b.z.to(device),
                x=b.x.to(device),
                pos=b.pos.to(device),
                edge_index=b.edge_index.to(device),
                edge_attr=b.edge_attr.to(device),
                batch=b.batch.to(device),
                y=b.y.to(device),
            )
            pred = model(b).detach().cpu().numpy()
            y = b.y.detach().cpu().numpy()
            preds_all.append(pred)
            y_all.append(y)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.array([])
    preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.array([])
    return y_all, preds_all


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", default="train_ready.csv", type=str)
    ap.add_argument("--test_csv", default="test_ready.csv", type=str)
    ap.add_argument("--outdir", default="run_gnn3d_fixed_nan", type=str)

    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--cv", default=5, type=int)

    ap.add_argument("--smiles_col", default="SMILES", type=str)
    ap.add_argument("--target", default="Dipole momentμ debye", type=str)

    # 3D conformer
    ap.add_argument("--num_confs", default=20, type=int)
    ap.add_argument("--add_hs", default=True, type=lambda x: str(x).lower() in ["1", "true", "yes", "y"])
    ap.add_argument("--allow_uff_fallback", default=False, type=lambda x: str(x).lower() in ["1", "true", "yes", "y"])

    # chemistry filters
    ap.add_argument("--allow_metals", default=False, type=lambda x: str(x).lower() in ["1", "true", "yes", "y"])
    ap.add_argument("--allow_only_organic", default=True, type=lambda x: str(x).lower() in ["1", "true", "yes", "y"])
    ap.add_argument("--disallow_atomic_nums", default="", type=str)

    # model/training (safer defaults)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", default=250, type=int)
    ap.add_argument("--patience", default=40, type=int)
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--weight_decay", default=1e-5, type=float)
    ap.add_argument("--h_dim", default=128, type=int)
    ap.add_argument("--n_layers", default=4, type=int)
    ap.add_argument("--clip_grad", default=1.0, type=float)
    ap.add_argument("--standardize_y", default=True, type=lambda x: str(x).lower() in ["1", "true", "yes", "y"])

    args = ap.parse_args()
    safe_mkdir(args.outdir)

    SEED = args.seed
    seed_everything(SEED)

    if args.disallow_atomic_nums.strip():
        disallow_set = set(int(x.strip()) for x in args.disallow_atomic_nums.split(",") if x.strip())
    else:
        disallow_set = set(DEFAULT_DISALLOWED_ATOMIC_NUMS)

    train_df = pd.read_csv(args.train_csv, low_memory=False)
    test_df = pd.read_csv(args.test_csv, low_memory=False)

    if args.smiles_col not in train_df.columns:
        raise ValueError(f"SMILES column not found in train: {args.smiles_col}")
    if args.smiles_col not in test_df.columns:
        raise ValueError(f"SMILES column not found in test: {args.smiles_col}")

    if args.target not in train_df.columns:
        raise ValueError(f"Target column not found in train: {args.target}")
    if args.target not in test_df.columns:
        raise ValueError(f"Target column not found in test: {args.target}")

    train_df[args.target] = pd.to_numeric(train_df[args.target], errors="coerce")
    test_df[args.target] = pd.to_numeric(test_df[args.target], errors="coerce")

    print(f"[Train] {args.train_csv} shape={train_df.shape}")
    print(f"[Test ] {args.test_csv}  shape={test_df.shape}")
    print(f"[Cols ] smiles='{args.smiles_col}' target='{args.target}'")
    print(f"[3D   ] num_confs={args.num_confs} add_hs={args.add_hs} seed={SEED} allow_uff_fallback={args.allow_uff_fallback}")
    print(f"[Chem ] allow_only_organic={args.allow_only_organic} allow_metals={args.allow_metals} disallow_set_size={len(disallow_set)}")
    print(f"[GNN  ] h_dim={args.h_dim} n_layers={args.n_layers} epochs={args.epochs} patience={args.patience} bs={args.batch_size} lr={args.lr}")
    print(f"[CV   ] running KFold CV (n_splits={args.cv})")
    print(f"[Dev  ] device={args.device}")

    print("\n[STEP] Building 3D graphs for TRAIN ...")
    train_graphs, train_valid, train_stats = build_graph_cache(
        train_df, args.smiles_col, args.target,
        seed=SEED, num_confs=args.num_confs, add_hs=args.add_hs,
        allow_metals=args.allow_metals,
        allow_only_organic=args.allow_only_organic,
        disallow_atomic_nums=disallow_set,
        allow_uff_fallback=args.allow_uff_fallback
    )
    print(f"[TRAIN] valid graphs: {int(train_valid.sum())} / {len(train_df)}")
    print("[TRAIN] build stats:")
    for k, v in train_stats.items():
        print(f"  - {k}: {v}")

    print("\n[STEP] Building 3D graphs for TEST ...")
    test_graphs, test_valid, test_stats = build_graph_cache(
        test_df, args.smiles_col, args.target,
        seed=SEED, num_confs=args.num_confs, add_hs=args.add_hs,
        allow_metals=args.allow_metals,
        allow_only_organic=args.allow_only_organic,
        disallow_atomic_nums=disallow_set,
        allow_uff_fallback=args.allow_uff_fallback
    )
    print(f"[TEST ] valid graphs: {int(test_valid.sum())} / {len(test_df)}")
    print("[TEST ] build stats:")
    for k, v in test_stats.items():
        print(f"  - {k}: {v}")

    with open(os.path.join(args.outdir, "graph_build_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"train": train_stats, "test": test_stats}, f, indent=2)

    train_idx_all = np.where(train_valid)[0]
    test_idx_all = np.where(test_valid)[0]

    if len(train_idx_all) < 50:
        print("[WARN] Train valid graphs < 50. Training may be unstable.")
    if len(test_idx_all) < 10:
        print("[WARN] Test valid graphs < 10. Metrics will be unreliable.")

    # -----------------------------
    # 1) Full training on TRAIN -> evaluate on TEST
    # -----------------------------
    print("\n==================== FULL TRAIN -> TEST ====================")

    test_items_full = index_to_items(test_df, test_graphs, test_idx_all, args.target)

    # small val split for early stopping
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(train_idx_all)
    n_val = max(1, int(0.1 * len(perm)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    tr_items = index_to_items(train_df, train_graphs, tr_idx, args.target)
    va_items = index_to_items(train_df, train_graphs, val_idx, args.target)

    model, train_summary = train_one_model(
        train_items=tr_items,
        val_items=va_items,
        device=args.device,
        seed=SEED,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        h_dim=args.h_dim,
        n_layers=args.n_layers,
        clip_grad=args.clip_grad,
        standardize_y=args.standardize_y,
    )

    y_mean = train_summary["y_mean"]
    y_std = train_summary["y_std"]

    y_te, p_te_std = predict_items(model, test_items_full, device=args.device, batch_size=args.batch_size)
    p_te = p_te_std * y_std + y_mean
    met_full = metrics_reg(y_te, p_te)

    print(json.dumps(met_full, indent=2))

    with open(os.path.join(args.outdir, "full_train_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": met_full, "train_summary": train_summary}, f, indent=2)

    pred_full_path = os.path.join(args.outdir, "predictions_full_train_test.csv")
    pd.DataFrame({
        "row_index_in_test_csv": test_idx_all.astype(int),
        "y_true": y_te.astype(float),
        "y_pred": p_te.astype(float),
    }).to_csv(pred_full_path, index=False)

    # Save FULL model
    models_dir = os.path.join(args.outdir, "models")
    safe_mkdir(models_dir)

    full_model_path = os.path.join(models_dir, "model_full_train.pt")
    save_torch_model(
        full_model_path,
        model,
        extra={
            "tag": "full_train_model",
            "h_dim": int(args.h_dim),
            "n_layers": int(args.n_layers),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "seed": int(SEED),
            "args": vars(args),
            "train_summary": train_summary,
            "full_train_test_metrics": met_full,
        }
    )

    # -----------------------------
    # 2) CV
    # -----------------------------
    print("\n==================== CV RUN ====================")

    kf = KFold(n_splits=args.cv, shuffle=True, random_state=SEED)

    fold_hold_metrics = []
    fold_test_metrics = []
    fold_test_preds = []

    for fold, (tr_pos, va_pos) in enumerate(kf.split(train_idx_all), start=1):
        tr_idx_fold = train_idx_all[tr_pos]
        va_idx_fold = train_idx_all[va_pos]

        train_items = index_to_items(train_df, train_graphs, tr_idx_fold, args.target)
        hold_items = index_to_items(train_df, train_graphs, va_idx_fold, args.target)
        test_items = test_items_full

        model_fold, summ_fold = train_one_model(
            train_items=train_items,
            val_items=hold_items,
            device=args.device,
            seed=SEED + fold,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            h_dim=args.h_dim,
            n_layers=args.n_layers,
            clip_grad=args.clip_grad,
            standardize_y=args.standardize_y,
        )

        y_mean_f = summ_fold["y_mean"]
        y_std_f = summ_fold["y_std"]

        y_ho, p_ho_std = predict_items(model_fold, hold_items, device=args.device, batch_size=args.batch_size)
        p_ho = p_ho_std * y_std_f + y_mean_f
        met_hold = metrics_reg(y_ho, p_ho)
        fold_hold_metrics.append(met_hold)

        y_te2, p_te2_std = predict_items(model_fold, test_items, device=args.device, batch_size=args.batch_size)
        p_te2 = p_te2_std * y_std_f + y_mean_f
        met_te = metrics_reg(y_te2, p_te2)
        fold_test_metrics.append(met_te)

        fold_test_preds.append(p_te2.astype(float))

        # Save fold model
        fold_model_path = os.path.join(models_dir, f"fold_{fold}.pt")
        save_torch_model(
            fold_model_path,
            model_fold,
            extra={
                "tag": f"cv_fold_{fold}",
                "fold": int(fold),
                "h_dim": int(args.h_dim),
                "n_layers": int(args.n_layers),
                "y_mean": float(y_mean_f),
                "y_std": float(y_std_f),
                "seed": int(SEED + fold),
                "args": vars(args),
                "train_summary": summ_fold,
                "holdout_metrics": met_hold,
                "test_metrics": met_te,
            }
        )

        print(
            f"[fold {fold}] "
            f"holdout: R2={met_hold['R2']:.4f} MAE={met_hold['MAE']:.4f} RMSE={met_hold['RMSE']:.4f} | "
            f"test: R2={met_te['R2']:.4f} MAE={met_te['MAE']:.4f} RMSE={met_te['RMSE']:.4f}"
        )

    mean_hold = mean_of_metrics(fold_hold_metrics)
    mean_test_mean_metrics = mean_of_metrics(fold_test_metrics)

    mean_pred_test = np.mean(np.vstack(fold_test_preds), axis=0)
    mean_test_metric_of_meanpred = metrics_reg(y_te, mean_pred_test)

    cv_summary = {
        "mean_holdout_cv": mean_hold,
        "mean_test_cv__mean_of_metrics": mean_test_mean_metrics,
        "mean_test_cv__metric_of_meanpred": mean_test_metric_of_meanpred,
        "gate_full_train_test_metrics": met_full,
        "cv_n_splits": int(args.cv),
        "seed": int(SEED),
        "test_rows_used": int(len(test_idx_all)),
        "train_rows_used": int(len(train_idx_all)),
    }

    print("\n==================== CV SUMMARY ====================")
    print(json.dumps(cv_summary, indent=2))

    with open(os.path.join(args.outdir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    pred_cv_path = os.path.join(args.outdir, "predictions_test_mean_cv_models.csv")
    pd.DataFrame({
        "row_index_in_test_csv": test_idx_all.astype(int),
        "y_true": y_te.astype(float),
        "y_pred_mean_of_cv_models": mean_pred_test.astype(float),
    }).to_csv(pred_cv_path, index=False)

    # -----------------------------
    # NEW: SAVE CV REPORTS (3 CSVs)
    # -----------------------------
    # 1) TEST metrics per fold
    test_metrics_rows = []
    for i, m in enumerate(fold_test_metrics, start=1):
        row = {"fold": int(i)}
        row.update(m)
        test_metrics_rows.append(row)

    df_test_metrics = pd.DataFrame(test_metrics_rows)
    cv_test_metrics_path = os.path.join(args.outdir, "cv_test_metrics_per_fold.csv")
    df_test_metrics.to_csv(cv_test_metrics_path, index=False)

    # 2) TEST metrics summary (mean ± std)
    ddof = 1 if int(args.cv) > 1 else 0
    test_mean_std = mean_std_of_metrics(fold_test_metrics, ddof=ddof)

    summary_row = {
        "cv_n_splits": int(args.cv),
        "std_ddof": int(ddof),
        "R2_mean": test_mean_std["R2"]["mean"],
        "R2_std":  test_mean_std["R2"]["std"],
        "MAE_mean": test_mean_std["MAE"]["mean"],
        "MAE_std":  test_mean_std["MAE"]["std"],
        "RMSE_mean": test_mean_std["RMSE"]["mean"],
        "RMSE_std":  test_mean_std["RMSE"]["std"],
        "n_mean": test_mean_std["n"]["mean"],
        "n_std":  test_mean_std["n"]["std"],
    }
    df_test_summary = pd.DataFrame([summary_row])
    cv_test_summary_path = os.path.join(args.outdir, "cv_test_metrics_summary_mean_std.csv")
    df_test_summary.to_csv(cv_test_summary_path, index=False)

    # 3) TEST predictions per fold (+ mean, std)
    # fold_test_preds: list of arrays aligned to test_items_full order
    P = np.vstack([p.reshape(1, -1) for p in fold_test_preds])  # (K, N)
    pred_mean = P.mean(axis=0)
    pred_std = P.std(axis=0, ddof=ddof)

    pred_df = pd.DataFrame({
        "row_index_in_test_csv": test_idx_all.astype(int),
        "y_true": y_te.astype(float),
    })
    for i in range(P.shape[0]):
        pred_df[f"y_pred_fold_{i+1}"] = P[i].astype(float)

    pred_df["y_pred_mean_cv"] = pred_mean.astype(float)
    pred_df["y_pred_std_cv"] = pred_std.astype(float)

    cv_pred_path = os.path.join(args.outdir, "cv_test_predictions_per_fold.csv")
    pred_df.to_csv(cv_pred_path, index=False)

    print("\n[Saved]")
    print(os.path.join(args.outdir, "graph_build_stats.json"))
    print(os.path.join(args.outdir, "full_train_test_metrics.json"))
    print(pred_full_path)
    print(full_model_path)
    print(os.path.join(args.outdir, "cv_summary.json"))
    print(pred_cv_path)
    print(cv_test_metrics_path)
    print(cv_test_summary_path)
    print(cv_pred_path)
    print(models_dir)


if __name__ == "__main__":
    main()

