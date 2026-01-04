#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Model
try:
    from lightgbm import LGBMRegressor
except Exception as e:
    raise ImportError("This script requires lightgbm. Install with: pip install lightgbm") from e

from catboost import CatBoostRegressor


warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning
)


# -----------------------------
# Utils
# -----------------------------
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


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


def mean_std_of_metrics(metric_dicts, ddof: int = 1):
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in metric_dicts]
        if len(vals) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=ddof)) if len(vals) > 1 else 0.0
            }
    return out


def write_metrics_summary_csv(path: str, metric_dicts: list, tag: str, cv: int, seed: int):
    """
    Writes a 1-row CSV with mean±std for each metric key in metric_dicts.
    """
    ddof = 1 if (len(metric_dicts) > 1) else 0
    ms = mean_std_of_metrics(metric_dicts, ddof=ddof)

    row = {
        "tag": tag,
        "cv_n_splits": int(cv),
        "seed": int(seed),
        "std_ddof": int(ddof),
    }

    # Flatten
    for k in ms.keys():
        row[f"{k}_mean"] = ms[k]["mean"]
        row[f"{k}_std"] = ms[k]["std"]

    pd.DataFrame([row]).to_csv(path, index=False)


def build_model(SEED: int):
    # Your CatBoost settings (as used above)
    return CatBoostRegressor(
        iterations=4000,
        depth=7,
        loss_function="RMSE",
        random_state=SEED,
        verbose=False,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", default="train_ready.csv", type=str)
    ap.add_argument("--test_csv", default="test_ready.csv", type=str)
    ap.add_argument("--outdir", default="run_trainready_testready_full_then_cv", type=str)

    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--cv", default=5, type=int)

    ap.add_argument("--target", default="Dipole momentμ debye", type=str)
    ap.add_argument("--exclude", nargs="*", default=["Name", "Formula", "SMILES", "Dipole momentμ debye"])

    # Model save name prefix
    ap.add_argument("--model_prefix", default="catboost_model", type=str)

    args = ap.parse_args()
    safe_mkdir(args.outdir)

    SEED = int(args.seed)

    # -----------------------------
    # Load train/test
    # -----------------------------
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    test_df = pd.read_csv(args.test_csv, low_memory=False)

    if args.target not in train_df.columns:
        raise ValueError(f"Target column not found in train: {args.target}")
    if args.target not in test_df.columns:
        raise ValueError(f"Target column not found in test: {args.target}")

    exclude_set = set(args.exclude)

    # Feature columns from train schema
    feat_cols_train = [c for c in train_df.columns if c not in exclude_set]
    feat_cols_test = [c for c in test_df.columns if c not in exclude_set]

    # Ensure same features in train and test
    if set(feat_cols_train) != set(feat_cols_test):
        missing_in_test = sorted(list(set(feat_cols_train) - set(feat_cols_test)))
        missing_in_train = sorted(list(set(feat_cols_test) - set(feat_cols_train)))
        raise ValueError(
            "Train/Test feature columns mismatch.\n"
            f"Missing in test: {missing_in_test[:25]}{'...' if len(missing_in_test)>25 else ''}\n"
            f"Missing in train: {missing_in_train[:25]}{'...' if len(missing_in_train)>25 else ''}"
        )

    feat_cols = sorted(feat_cols_train)
    if len(feat_cols) == 0:
        raise ValueError("No feature columns left after excluding.")

    # X/y
    X_train = coerce_numeric_df(train_df[feat_cols])
    y_train = pd.to_numeric(train_df[args.target], errors="coerce")

    X_test = coerce_numeric_df(test_df[feat_cols])
    y_test = pd.to_numeric(test_df[args.target], errors="coerce")

    # Drop NaN targets (train/test separately)
    mtr = y_train.notna()
    X_train = X_train.loc[mtr].reset_index(drop=True)
    y_train = y_train.loc[mtr].reset_index(drop=True)

    mte = y_test.notna()
    X_test = X_test.loc[mte].reset_index(drop=True)
    y_test = y_test.loc[mte].reset_index(drop=True)

    print(f"[Train] {args.train_csv} X={X_train.shape} y={y_train.shape}")
    print(f"[Test ] {args.test_csv}  X={X_test.shape} y={y_test.shape}")
    print(f"[Features] n={len(feat_cols)}")
    print(f"[Model] CatBoostRegressor (fixed params)")

    # Output dirs
    models_dir = os.path.join(args.outdir, "models")
    safe_mkdir(models_dir)

    # -----------------------------
    # 1) Full training on TRAIN -> evaluate on TEST
    # -----------------------------
    base = build_model(SEED)
    base.fit(X_train, y_train)
    pred_test = base.predict(X_test)

    met_full = metrics_reg(y_test, pred_test)

    print("\n==================== FULL TRAIN -> TEST ====================")
    print(json.dumps(met_full, indent=2))

    # Save full eval json
    with open(os.path.join(args.outdir, "full_train_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(met_full, f, indent=2)

    # Save full predictions
    pd.DataFrame({"y_true": y_test.values, "y_pred": np.asarray(pred_test, dtype=float)}).to_csv(
        os.path.join(args.outdir, "predictions_full_train_test.csv"),
        index=False
    )

    # NEW: Save full-train test metrics as CSV (single-row; std=0)
    full_metrics_csv = os.path.join(args.outdir, "metrics_full_train_test.csv")
    full_row = {"tag": "full_train_test", "seed": SEED}
    full_row.update(met_full)
    pd.DataFrame([full_row]).to_csv(full_metrics_csv, index=False)

    # NEW: Save full trained model
    full_model_path = os.path.join(models_dir, f"{args.model_prefix}_full_train.cbm")
    base.save_model(full_model_path)

    # -----------------------------
    # 2) CV
    # -----------------------------
    print("\n==================== CV RUN ====================")

    kf = KFold(n_splits=int(args.cv), shuffle=True, random_state=SEED)

    fold_hold_metrics = []
    fold_test_metrics = []

    fold_hold_preds = []  # optional (not written by default)
    fold_test_preds = []  # used for mean prediction CSV

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        model = clone(base)
        model.fit(X_tr, y_tr)

        # Holdout metrics
        pred_va = model.predict(X_va)
        met_hold = metrics_reg(y_va, pred_va)
        fold_hold_metrics.append(met_hold)

        # Test (unseen) metrics
        pred_te = model.predict(X_test)
        met_te = metrics_reg(y_test, pred_te)
        fold_test_metrics.append(met_te)

        fold_test_preds.append(np.asarray(pred_te, dtype=float))

        print(
            f"[fold {fold}] "
            f"holdout: R2={met_hold['R2']:.4f} MAE={met_hold['MAE']:.4f} RMSE={met_hold['RMSE']:.4f} | "
            f"test: R2={met_te['R2']:.4f} MAE={met_te['MAE']:.4f} RMSE={met_te['RMSE']:.4f}"
        )

        # NEW: save each fold model (optional but usually helpful)
        fold_model_path = os.path.join(models_dir, f"{args.model_prefix}_fold_{fold}.cbm")
        model.save_model(fold_model_path)

    # -----------------------------
    # NEW: SAVE CV REPORTS (holdout, test, full-train test)
    # -----------------------------
    # A) Holdout (per-fold) + summary
    holdout_per_fold_csv = os.path.join(args.outdir, "metrics_cv_holdout_per_fold.csv")
    pd.DataFrame([{"fold": i + 1, **m} for i, m in enumerate(fold_hold_metrics)]).to_csv(holdout_per_fold_csv, index=False)

    holdout_summary_csv = os.path.join(args.outdir, "metrics_cv_holdout_summary_mean_std.csv")
    write_metrics_summary_csv(
        holdout_summary_csv,
        fold_hold_metrics,
        tag="cv_holdout",
        cv=int(args.cv),
        seed=SEED
    )

    # B) Test (per-fold) + summary
    test_per_fold_csv = os.path.join(args.outdir, "metrics_cv_test_per_fold.csv")
    pd.DataFrame([{"fold": i + 1, **m} for i, m in enumerate(fold_test_metrics)]).to_csv(test_per_fold_csv, index=False)

    test_summary_csv = os.path.join(args.outdir, "metrics_cv_test_summary_mean_std.csv")
    write_metrics_summary_csv(
        test_summary_csv,
        fold_test_metrics,
        tag="cv_test",
        cv=int(args.cv),
        seed=SEED
    )

    # C) Full-train test metrics CSV is already written: metrics_full_train_test.csv

    # -----------------------------
    # Existing: CV summary json + mean-of-fold predictions
    # -----------------------------
    mean_pred_test = np.mean(np.vstack(fold_test_preds), axis=0)
    mean_test_metric_of_meanpred = metrics_reg(y_test, mean_pred_test)

    # mean of metrics (not used for CSV; kept for json summary)
    mean_hold = {k: float(np.mean([d[k] for d in fold_hold_metrics])) for k in fold_hold_metrics[0].keys()}
    mean_test_mean_metrics = {k: float(np.mean([d[k] for d in fold_test_metrics])) for k in fold_test_metrics[0].keys()}

    cv_summary = {
        "mean_holdout_cv": mean_hold,
        "mean_test_cv__mean_of_metrics": mean_test_mean_metrics,
        "mean_test_cv__metric_of_meanpred": mean_test_metric_of_meanpred,
        "gate_full_train_test_metrics": met_full,
        "cv_n_splits": int(args.cv),
        "seed": int(SEED),
        "saved_models_dir": models_dir,
        "full_model_path": full_model_path,
    }

    print("\n==================== CV SUMMARY ====================")
    print(json.dumps(cv_summary, indent=2))

    with open(os.path.join(args.outdir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    # Optional: save mean-pred test predictions from CV models
    pd.DataFrame({
        "y_true": y_test.values,
        "y_pred_mean_of_cv_models": mean_pred_test.astype(float),
    }).to_csv(os.path.join(args.outdir, "predictions_test_mean_cv_models.csv"), index=False)

    print("\n[Saved]")
    print(os.path.join(args.outdir, "full_train_test_metrics.json"))
    print(os.path.join(args.outdir, "predictions_full_train_test.csv"))
    print(full_metrics_csv)
    print(full_model_path)
    print(holdout_per_fold_csv)
    print(holdout_summary_csv)
    print(test_per_fold_csv)
    print(test_summary_csv)
    print(os.path.join(args.outdir, "cv_summary.json"))
    print(os.path.join(args.outdir, "predictions_test_mean_cv_models.csv"))
    print(models_dir)


if __name__ == "__main__":
    main()

