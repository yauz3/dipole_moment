#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
catboost_models.py
==================

İki CatBoost modeli eğitir, tahmin üretir ve kaydeder.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODELLER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Model-1 : CatBoost — Fingerprint
      Girdi : Tüm sayısal sütunlar (Avalon_* + MACCS_* + diğerleri)
              → CSV'deki her sayısal kolon kullanılır (exclude listesi hariç)

  Model-2 : CatBoost — Fingerprint + SVR Features
      Girdi : Model-1 girdisi + SVR fiziksel/geometrik özellikler
              → 9 temel 2D descriptor (MolWt, LogP, TPSA, ...)
              → 7 dipol proxy (Gasteiger/MMFF multikonformer)
              → 10 3D şekil descriptor (PMI, NPR, RadiusOfGyration, ...)
              → MACCS fingerprint (SVR feature setinden)
              NOT: SVR feature build konformer üretimi gerektirir (yavaş).
                   cache_svr_features/ altına kaydedilir; varsa yüklenir.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KAYIT / YÜKLEME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model ve tahminler <outdir>/saved_models/ altına kaydedilir.
Sonraki çalışmada kayıtlı dosyalar varsa eğitim atlanır.

  saved_models/
    cb_fp_full.cbm              ← Model-1 (fingerprint)
    cb_fp_full_pred.npy
    cb_fp_full_feat_cols.json

    cb_fp_svr_full.cbm          ← Model-2 (fingerprint + SVR features)
    cb_fp_svr_full_pred.npy
    cb_fp_svr_full_feat_cols.json

    (RUN_CV = True ise:)
    cb_fp_fold_1.cbm ... cb_fp_fold_K.cbm
    cb_fp_fold_1_pred.npy ...
    cb_fp_svr_fold_1.cbm ...
    cb_fp_svr_fold_1_pred.npy ...

  cache_svr_features/
    features_train_svr.csv      ← SVR feature cache (varsa yüklenir)
    features_test_svr.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kullanım
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python catboost_models.py \\
      --train_csv train_ready.csv \\
      --test_csv  test_ready.csv  \\
      --outdir    run_catboost    \\
      --smiles_col SMILES
"""

# ╔══════════════════════════════════════════════════════╗
# ║          KULLANICI DEĞİŞKENLERİ                      ║
# ╚══════════════════════════════════════════════════════╝

RUN_CV   = False   # True → 5-fold CV çalışır
CV_FOLDS = 5

# ╔══════════════════════════════════════════════════════╗
# ║                   IMPORTS                            ║
# ╚══════════════════════════════════════════════════════╝

import os, json, argparse, warnings, random
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, MACCSkeys
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    RDKIT_OK = True
except Exception:
    Chem = None; AllChem = None
    RDKIT_OK = False

DEBYE_PER_E_ANGSTROM     = 4.80320427
BOLTZMANN_KCAL_PER_MOL_K = 0.0019872041


# ╔══════════════════════════════════════════════════════╗
# ║                   UTILITIES                          ║
# ╚══════════════════════════════════════════════════════╝

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)

def coerce_numeric_df(X):
    X2 = X.copy()
    for c in X2.columns:
        if X2[c].dtype == "object":
            X2[c] = pd.to_numeric(X2[c], errors="coerce")
    return X2.replace([np.inf, -np.inf], np.nan).fillna(0)

def metrics_reg(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "n": 0}
    return {
        "R2":   float(r2_score(yt, yp)),
        "MAE":  float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "n":    int(len(yt)),
    }

def write_cv_csv(path, fold_metrics, tag):
    rows = [{"tag": tag, "row_type": "fold", "fold": i+1, **m}
            for i, m in enumerate(fold_metrics)]
    keys = list(fold_metrics[0].keys())
    ddof = 1 if len(fold_metrics) > 1 else 0
    mr = {"tag": tag, "row_type": "mean", "fold": -1}
    sr = {"tag": tag, "row_type": "std",  "fold": -1}
    for k in keys:
        v = np.array([fm[k] for fm in fold_metrics], dtype=float)
        mr[k] = float(np.mean(v))
        sr[k] = float(np.std(v, ddof=ddof)) if len(v) > 1 else 0.
    pd.DataFrame(rows + [mr, sr]).to_csv(path, index=False)

def mol_from_smiles(smiles):
    if smiles is None or pd.isna(smiles): return None
    s = str(smiles).strip()
    if not s: return None
    try: return Chem.MolFromSmiles(s)
    except: return None


# ╔══════════════════════════════════════════════════════╗
# ║   SVR FEATURE BUILD                                  ║
# ╚══════════════════════════════════════════════════════╝
# Model-2 (CatBoost Fingerprint + SVR Features) için kullanılır.
# Üretilen sütunlar:
#   y_true
#   9 temel 2D  : MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors,
#                 NumRotatableBonds, RingCount, FractionCSP3, HeavyAtomCount
#   167 MACCS   : MACCS_0 … MACCS_166
#   7 dipol     : gasteiger_single, gasteiger_multiconf_mean/boltzmann,
#                 mmff_single, mmff_multiconf_mean/boltzmann, n_confs_used
#   10 3D şekil : PMI1-3, NPR1-2, RadiusOfGyration, InertialShapeFactor,
#                 Asphericity, Eccentricity, SpherocityIndex

def _softmax_boltzmann(energies, T=298.15):
    e = np.asarray(energies, dtype=float)
    if len(e) == 0: return np.array([], dtype=float)
    e -= np.nanmin(e)
    x = -(1. / (BOLTZMANN_KCAL_PER_MOL_K * T)) * e
    x -= np.max(x); w = np.exp(x); s = w.sum()
    return w / s if (np.isfinite(s) and s > 0) else np.ones(len(e)) / len(e)

def _multi3d(smiles, seed=42, num_confs=20, mmff_its=300):
    m0 = mol_from_smiles(smiles)
    if m0 is None: return None, None, None, [], [], None
    m2d = Chem.Mol(m0)
    try: m3d = Chem.AddHs(m0)
    except: return m2d, None, None, [], [], None
    try: params = AllChem.ETKDGv3()
    except: params = AllChem.ETKDG()
    params.randomSeed = seed
    try: cids = list(AllChem.EmbedMultipleConfs(m3d, numConfs=num_confs, params=params))
    except: cids = []
    if not cids:
        try:
            res = AllChem.EmbedMolecule(m3d, randomSeed=seed)
            cids = [m3d.GetConformer().GetId()] if res != -1 else []
        except: cids = []
    if not cids: return m2d, None, None, [], [], None
    props = None
    try: props = AllChem.MMFFGetMoleculeProperties(m3d, mmffVariant="MMFF94")
    except: pass
    vcids, engs = [], []
    for cid in cids:
        e = None
        if props:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(m3d, props, confId=int(cid))
                ff.Minimize(maxIts=mmff_its); e = float(ff.CalcEnergy())
            except: pass
        if e is None:
            try:
                uff = AllChem.UFFGetMoleculeForceField(m3d, confId=int(cid))
                uff.Minimize(maxIts=mmff_its); e = float(uff.CalcEnergy())
            except: pass
        if e is None or not np.isfinite(e): continue
        vcids.append(int(cid)); engs.append(float(e))
    if not vcids: return m2d, None, None, [], [], None
    bi = int(np.argmin(engs)); best = int(vcids[bi])
    try:
        conf = m3d.GetConformer(best); mb = Chem.Mol(m3d); mb.RemoveAllConformers()
        nc = Chem.Conformer(mb.GetNumAtoms())
        for i in range(mb.GetNumAtoms()):
            p = conf.GetAtomPosition(i); nc.SetAtomPosition(i, p)
        mb.AddConformer(nc, assignId=True)
    except: mb = None
    return m2d, mb, m3d, vcids, engs, props

def _get_pos(mol, cid):
    try:
        c = mol.GetConformer(int(cid))
        arr = np.array([[c.GetAtomPosition(i).x,
                         c.GetAtomPosition(i).y,
                         c.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
        return arr if arr.shape == (mol.GetNumAtoms(), 3) else None
    except: return None

def _gasteiger(mol):
    try:
        m2 = Chem.Mol(mol); AllChem.ComputeGasteigerCharges(m2)
        qs = [float(a.GetProp("_GasteigerCharge")) for a in m2.GetAtoms()]
        return np.array(qs) if all(np.isfinite(qs)) else None
    except: return None

def _mmff_charges(mol, cid, props):
    if props is None: return None
    try:
        qs = np.array([float(props.getMMFFPartialCharge(i))
                       for i in range(mol.GetNumAtoms())])
        return qs if np.all(np.isfinite(qs)) else None
    except: return None

def _dipole(charges, pos):
    if charges is None or pos is None: return None
    try:
        q = np.asarray(charges, dtype=float); p = np.asarray(pos, dtype=float)
        if q.shape[0] != p.shape[0]: return None
        return float(np.linalg.norm((q[:, None] * p).sum(0) * DEBYE_PER_E_ANGSTROM))
    except: return None

def _3d_shape(mol):
    nans = {k: np.nan for k in ["PMI1","PMI2","PMI3","NPR1","NPR2",
                                  "RadiusOfGyration","InertialShapeFactor",
                                  "Asphericity","Eccentricity","SpherocityIndex"]}
    try:
        return {
            "PMI1": float(rdMolDescriptors.CalcPMI1(mol)),
            "PMI2": float(rdMolDescriptors.CalcPMI2(mol)),
            "PMI3": float(rdMolDescriptors.CalcPMI3(mol)),
            "NPR1": float(rdMolDescriptors.CalcNPR1(mol)),
            "NPR2": float(rdMolDescriptors.CalcNPR2(mol)),
            "RadiusOfGyration":    float(rdMolDescriptors.CalcRadiusOfGyration(mol)),
            "InertialShapeFactor": float(rdMolDescriptors.CalcInertialShapeFactor(mol)),
            "Asphericity":         float(rdMolDescriptors.CalcAsphericity(mol)),
            "Eccentricity":        float(rdMolDescriptors.CalcEccentricity(mol)),
            "SpherocityIndex":     float(rdMolDescriptors.CalcSpherocityIndex(mol)),
        }
    except: return nans

def _direct_dipole_feats(m3d_multi, vcids, engs, props):
    base = {k: np.nan for k in ["gasteiger_single","gasteiger_multiconf_mean",
                                  "gasteiger_multiconf_boltzmann","mmff_single",
                                  "mmff_multiconf_mean","mmff_multiconf_boltzmann",
                                  "n_confs_used"]}
    if m3d_multi is None or not vcids: return base
    best_cid = vcids[int(np.argmin(engs))]; pos_best = _get_pos(m3d_multi, best_cid)
    gc = _gasteiger(m3d_multi)
    if gc is not None and pos_best is not None:
        d = _dipole(gc, pos_best)
        if d is not None: base["gasteiger_single"] = d
    mc = _mmff_charges(m3d_multi, best_cid, props)
    if mc is not None and pos_best is not None:
        d = _dipole(mc, pos_best)
        if d is not None: base["mmff_single"] = d
    gv, mv, ev2 = [], [], []
    for cid in vcids:
        pos = _get_pos(m3d_multi, cid)
        if pos is None: continue
        if gc is not None:
            d = _dipole(gc, pos)
            if d is not None: gv.append(d)
        mc2 = _mmff_charges(m3d_multi, cid, props)
        if mc2 is not None:
            d = _dipole(mc2, pos)
            if d is not None: mv.append(d)
        ev2.append(engs[vcids.index(cid)])
    base["n_confs_used"] = len(ev2)
    if gv:
        base["gasteiger_multiconf_mean"] = float(np.mean(gv))
        if len(ev2) == len(gv):
            base["gasteiger_multiconf_boltzmann"] = float(
                np.dot(_softmax_boltzmann(ev2), gv))
    if mv:
        base["mmff_multiconf_mean"] = float(np.mean(mv))
        if len(ev2) == len(mv):
            base["mmff_multiconf_boltzmann"] = float(
                np.dot(_softmax_boltzmann(ev2), mv))
    return base

def build_svr_feature_df(df, smiles_col, target_col):
    """SVR fiziksel/geometrik özellik tablosu üretir."""
    rows = []
    for i in range(len(df)):
        smi = df.iloc[i][smiles_col]; yi = df.iloc[i][target_col]
        row = {"y_true": float(yi) if pd.notna(yi) else np.nan}
        m0  = mol_from_smiles(smi) if pd.notna(smi) else None
        if m0 is None:
            for k in ["MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors",
                      "NumRotatableBonds","RingCount","FractionCSP3","HeavyAtomCount"]:
                row[k] = np.nan
            for k in range(167): row[f"MACCS_{k}"] = 0
            rows.append(row); continue
        row.update({
            "MolWt":             float(Descriptors.MolWt(m0)),
            "MolLogP":           float(Descriptors.MolLogP(m0)),
            "TPSA":              float(Descriptors.TPSA(m0)),
            "NumHDonors":        float(rdMolDescriptors.CalcNumHBD(m0)),
            "NumHAcceptors":     float(rdMolDescriptors.CalcNumHBA(m0)),
            "NumRotatableBonds": float(rdMolDescriptors.CalcNumRotatableBonds(m0)),
            "RingCount":         float(rdMolDescriptors.CalcNumRings(m0)),
            "FractionCSP3":      float(rdMolDescriptors.CalcFractionCSP3(m0)),
            "HeavyAtomCount":    float(m0.GetNumHeavyAtoms()),
        })
        try:
            mk = MACCSkeys.GenMACCSKeys(m0); bits = set(mk.GetOnBits())
            for k in range(167): row[f"MACCS_{k}"] = 1 if k in bits else 0
        except:
            for k in range(167): row[f"MACCS_{k}"] = 0
        _, mb, mm, vcids, engs, props = _multi3d(str(smi))
        row.update(_3d_shape(mb) if mb is not None
                   else {k: np.nan for k in ["PMI1","PMI2","PMI3","NPR1","NPR2",
                         "RadiusOfGyration","InertialShapeFactor",
                         "Asphericity","Eccentricity","SpherocityIndex"]})
        row.update(_direct_dipole_feats(mm, vcids, engs, props))
        rows.append(row)
        if (i + 1) % 250 == 0:
            print(f"  [SVR feature build] {i+1}/{len(df)}")
    return pd.DataFrame(rows)

def _svr_feature_cols(feat_df):
    """SVR feature DataFrame'inden kolon isimlerini döndürür (y_true hariç)."""
    basic = ["MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors",
             "NumRotatableBonds","RingCount","FractionCSP3","HeavyAtomCount"]
    dipc  = ["gasteiger_single","gasteiger_multiconf_mean","gasteiger_multiconf_boltzmann",
             "mmff_single","mmff_multiconf_mean","mmff_multiconf_boltzmann","n_confs_used"]
    shp   = ["PMI1","PMI2","PMI3","NPR1","NPR2","RadiusOfGyration",
             "InertialShapeFactor","Asphericity","Eccentricity","SpherocityIndex"]
    maccs = [c for c in feat_df.columns if c.startswith("MACCS_")]
    return basic + dipc + shp + maccs


# ╔══════════════════════════════════════════════════════╗
# ║   CatBoost — ortak eğitim fonksiyonu                 ║
# ╚══════════════════════════════════════════════════════╝

def _run_catboost(Xtr, ytr, Xte, seed):
    """CatBoost eğitir, test seti tahminini döndürür."""
    model = CatBoostRegressor(
        iterations=4000, depth=7, loss_function="RMSE",
        random_seed=seed, verbose=False)
    model.fit(Xtr, ytr)
    pred = np.asarray(model.predict(Xte), dtype=float).reshape(-1)
    return model, pred


def train_one_catboost(
    name,        # "cb_fp" veya "cb_fp_svr"
    Xtr, ytr,    # train features + labels (numpy)
    Xte, yte,    # test features + labels (numpy)
    feat_cols,   # kolon isimleri (kayıt için)
    seed,
    models_dir,
    outdir,
):
    """
    Tek CatBoost modeli eğitir veya yükler.
    name: dosya prefix'i (cb_fp veya cb_fp_svr)
    """
    model_path = os.path.join(models_dir, f"{name}_full.cbm")
    pred_path  = os.path.join(models_dir, f"{name}_full_pred.npy")
    cols_path  = os.path.join(models_dir, f"{name}_full_feat_cols.json")

    if os.path.exists(model_path) and os.path.exists(pred_path):
        print(f"  [{name}] Kayıtlı model yükleniyor: {model_path}")
        model = CatBoostRegressor()
        model.load_model(model_path)
        pred = np.load(pred_path)
    else:
        print(f"  [{name}] Eğitiliyor...")
        model, pred = _run_catboost(Xtr, ytr, Xte, seed)
        model.save_model(model_path)
        np.save(pred_path, pred)
        with open(cols_path, "w") as f:
            json.dump(feat_cols, f)
        print(f"  [{name}] Model kaydedildi: {model_path}")

    mask = np.isfinite(pred)
    met  = metrics_reg(yte[mask], pred[mask])
    print(f"  [{name}]  R2={met['R2']:.4f}  MAE={met['MAE']:.4f}  "
          f"RMSE={met['RMSE']:.4f}  n={met['n']}")

    # Tahmin CSV'si
    pred_csv = os.path.join(outdir, f"predictions_test_{name}_full.csv")
    pd.DataFrame({"y_true": yte, "y_pred": pred,
                  "valid_mask": mask.astype(int)}).to_csv(pred_csv, index=False)

    return {"model": model, "feat_cols": feat_cols,
            "pred_test": pred, "mask_test": mask, "metrics_test": met}


def cv_one_catboost(
    name,
    Xall, yall,  # tüm train
    Xte, yte,
    feat_cols,
    seed, n_splits,
    outdir, models_dir,
):
    """Tek CatBoost modeli için KFold CV."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    hm, tm = [], []
    all_fold_preds = np.full((n_splits, len(yte)), np.nan)

    for fold, (tri, vai) in enumerate(kf.split(Xall), 1):
        mp = os.path.join(models_dir, f"{name}_fold_{fold}.cbm")
        pp = os.path.join(models_dir, f"{name}_fold_{fold}_pred.npy")

        if os.path.exists(mp) and os.path.exists(pp):
            print(f"  [{name} fold {fold}] Yükleniyor: {mp}")
            m = CatBoostRegressor(); m.load_model(mp)
            pt = np.load(pp)
            pv = np.asarray(m.predict(Xall[vai]), dtype=float)
        else:
            m, pv = _run_catboost(Xall[tri], yall[tri], Xall[vai], seed + fold)
            _, pt = _run_catboost(Xall[tri], yall[tri], Xte, seed + fold)
            m.save_model(mp); np.save(pp, pt)

        mv = np.isfinite(pv); mt = np.isfinite(pt)
        hm.append(metrics_reg(yall[vai][mv], pv[mv]))
        tm.append(metrics_reg(yte[mt], pt[mt]))
        all_fold_preds[fold-1] = pt
        print(f"  [{name} fold {fold}]  "
              f"hold R2={hm[-1]['R2']:.4f} MAE={hm[-1]['MAE']:.4f}  |  "
              f"test R2={tm[-1]['R2']:.4f} MAE={tm[-1]['MAE']:.4f}")

    write_cv_csv(os.path.join(outdir, f"cv_holdout_{name}.csv"), hm, f"{name}_holdout")
    write_cv_csv(os.path.join(outdir, f"cv_test_{name}.csv"),    tm, f"{name}_test")

    # Fold tahminlerini tek CSV'ye yaz
    pred_df = pd.DataFrame({"y_true": yte})
    for i in range(n_splits):
        pred_df[f"y_pred_fold_{i+1}"] = all_fold_preds[i]
    pred_df["y_pred_mean_cv"] = np.nanmean(all_fold_preds, axis=0)
    pred_df.to_csv(os.path.join(outdir, f"cv_predictions_{name}.csv"), index=False)

    return hm, tm


# ╔══════════════════════════════════════════════════════╗
# ║                      MAIN                          ║
# ╚══════════════════════════════════════════════════════╝

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv",   default="train_ready.csv")
    ap.add_argument("--test_csv",    default="test_ready.csv")
    ap.add_argument("--outdir",      default="run_catboost")
    ap.add_argument("--target",      default="Dipole momentμ debye")
    ap.add_argument("--smiles_col",  default="SMILES")
    ap.add_argument("--exclude",     nargs="*",
                    default=["Name","Formula","SMILES","Dipole momentμ debye"])
    ap.add_argument("--seed",        default=42, type=int)
    args = ap.parse_args()

    if CatBoostRegressor is None:
        raise ImportError("catboost gerekli: pip install catboost")

    safe_mkdir(args.outdir)
    models_dir = os.path.join(args.outdir, "saved_models")
    cache_dir  = os.path.join(args.outdir, "cache_svr_features")
    safe_mkdir(models_dir)
    safe_mkdir(cache_dir)
    seed_everything(args.seed)

    # ── Veri yükle ────────────────────────────────────
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    test_df  = pd.read_csv(args.test_csv,  low_memory=False)
    train_df = train_df[pd.to_numeric(train_df[args.target],
                        errors="coerce").notna()].reset_index(drop=True)
    test_df  = test_df[pd.to_numeric(test_df[args.target],
                       errors="coerce").notna()].reset_index(drop=True)

    y_train = pd.to_numeric(train_df[args.target], errors="coerce").values.astype(float)
    y_test  = pd.to_numeric(test_df[args.target],  errors="coerce").values.astype(float)

    print(f"[Train] rows={len(train_df)}  [Test] rows={len(test_df)}")
    print(f"[RUN_CV] {RUN_CV}  CV_FOLDS={CV_FOLDS}\n")

    # ══════════════════════════════════════════════════
    # MODEL-1: CatBoost — Fingerprint
    # Girdi: tüm sayısal CSV sütunları (Avalon_* + MACCS_* + diğerleri)
    # ══════════════════════════════════════════════════
    print("=" * 60)
    print("MODEL-1: CatBoost — Fingerprint")
    print("  Girdi: Tüm sayısal sütunlar (Avalon_* + MACCS_* + ...)")
    print("=" * 60)

    excl   = set(args.exclude)
    fp_cols = sorted([c for c in train_df.columns if c not in excl])

    Xtr_fp = coerce_numeric_df(train_df[fp_cols]).values
    Xte_fp = coerce_numeric_df(test_df[fp_cols]).values

    res1 = train_one_catboost(
        name="cb_fp",
        Xtr=Xtr_fp, ytr=y_train,
        Xte=Xte_fp, yte=y_test,
        feat_cols=fp_cols,
        seed=args.seed,
        models_dir=models_dir,
        outdir=args.outdir,
    )
    print(json.dumps(res1["metrics_test"], indent=2))

    if RUN_CV:
        print(f"\n  >> CV ({CV_FOLDS}-fold) — CatBoost Fingerprint")
        cv_one_catboost(
            name="cb_fp",
            Xall=Xtr_fp, yall=y_train,
            Xte=Xte_fp,  yte=y_test,
            feat_cols=fp_cols,
            seed=args.seed, n_splits=CV_FOLDS,
            outdir=args.outdir, models_dir=models_dir,
        )

    # ══════════════════════════════════════════════════
    # MODEL-2: CatBoost — Fingerprint + SVR Features
    # SVR feature cache: varsa yükle, yoksa üret + kaydet
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MODEL-2: CatBoost — Fingerprint + SVR Features")
    print("  SVR features: 9 temel 2D + 7 dipol proxy + 10 3D şekil + MACCS")
    print("  Cache: cache_svr_features/ (varsa yüklenir)")
    print("=" * 60)

    tr_feat_path = os.path.join(cache_dir, "features_train_svr.csv")
    te_feat_path = os.path.join(cache_dir, "features_test_svr.csv")

    if os.path.exists(tr_feat_path) and os.path.exists(te_feat_path):
        print("  [Cache HIT] SVR feature'ları yükleniyor...")
        tr_feat = pd.read_csv(tr_feat_path, low_memory=False)
        te_feat = pd.read_csv(te_feat_path, low_memory=False)
    else:
        if not RDKIT_OK:
            raise ImportError("Model-2 için RDKit gerekli.")
        print("  [Cache MISS] SVR feature'ları üretiliyor (yavaş adım)...")
        tr_feat = build_svr_feature_df(train_df, args.smiles_col, args.target)
        te_feat = build_svr_feature_df(test_df,  args.smiles_col, args.target)
        tr_feat.to_csv(tr_feat_path, index=False)
        te_feat.to_csv(te_feat_path, index=False)
        print(f"  [Cache] Kaydedildi: {tr_feat_path}")

    svr_cols = _svr_feature_cols(tr_feat)

    # SVR feature'larını impute et (NaN → median), sonra fingerprint ile birleştir
    imp = SimpleImputer(strategy="median")
    Xtr_svr_raw = tr_feat[svr_cols].values.astype(float)
    Xte_svr_raw = te_feat[svr_cols].values.astype(float)
    Xtr_svr = imp.fit_transform(Xtr_svr_raw)
    Xte_svr = imp.transform(Xte_svr_raw)

    # Birleştir: [fingerprint | SVR features]
    Xtr_combined = np.hstack([Xtr_fp, Xtr_svr])
    Xte_combined = np.hstack([Xte_fp, Xte_svr])
    combined_cols = fp_cols + svr_cols

    print(f"  Feature boyutları: fingerprint={Xtr_fp.shape[1]}, "
          f"svr={Xtr_svr.shape[1]}, toplam={Xtr_combined.shape[1]}")

    res2 = train_one_catboost(
        name="cb_fp_svr",
        Xtr=Xtr_combined, ytr=y_train,
        Xte=Xte_combined, yte=y_test,
        feat_cols=combined_cols,
        seed=args.seed,
        models_dir=models_dir,
        outdir=args.outdir,
    )
    print(json.dumps(res2["metrics_test"], indent=2))

    if RUN_CV:
        print(f"\n  >> CV ({CV_FOLDS}-fold) — CatBoost Fingerprint + SVR Features")
        cv_one_catboost(
            name="cb_fp_svr",
            Xall=Xtr_combined, yall=y_train,
            Xte=Xte_combined,  yte=y_test,
            feat_cols=combined_cols,
            seed=args.seed, n_splits=CV_FOLDS,
            outdir=args.outdir, models_dir=models_dir,
        )

    # ── Özet ──────────────────────────────────────────
    metric_rows = [
        {"model": "cb_fp",     **res1["metrics_test"]},
        {"model": "cb_fp_svr", **res2["metrics_test"]},
    ]
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = os.path.join(args.outdir, "metrics_test_catboost_models.csv")
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "train_csv":  args.train_csv,
        "test_csv":   args.test_csv,
        "target":     args.target,
        "seed":       int(args.seed),
        "run_cv":     RUN_CV,
        "cv_folds":   CV_FOLDS if RUN_CV else None,
        "models": {
            "cb_fp": {
                "description": "CatBoost — Fingerprint only",
                "n_features":  len(fp_cols),
                "metrics":     res1["metrics_test"],
            },
            "cb_fp_svr": {
                "description": "CatBoost — Fingerprint + SVR Features",
                "n_features":  len(combined_cols),
                "n_fp":        len(fp_cols),
                "n_svr":       len(svr_cols),
                "metrics":     res2["metrics_test"],
            },
        },
        "saved_models_dir": models_dir,
        "svr_feature_cache": {
            "train": tr_feat_path,
            "test":  te_feat_path,
        },
    }
    summary_path = os.path.join(args.outdir, "summary_catboost_models.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL TEST METRICS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\n[Saved] {metrics_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Models] {models_dir}/")
    print(f"[Cache]  {tr_feat_path}")


if __name__ == "__main__":
    main()
