#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/12/2025
# Author: Sadettin Y. Ugurlu

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def coerce_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    for c in X2.columns:
        if X2[c].dtype == "object":
            X2[c] = pd.to_numeric(X2[c], errors="coerce")
    X2 = X2.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X2


def metrics_reg(y_true, y_pred):
    # Not: Orijinal fonksiyon davranışı korunuyor (RMSE alanı MSE döndürüyor).
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred)),
    }


def build_multiview_model(
    input_dim_avalon: int,
    input_dim_maccs: int,
    lr: float = 1e-3,
    dropout: float = 0.15,
    alpha: float = 0.10
):
    in_avalon = Input(shape=(input_dim_avalon,), name="avalon_in")
    in_maccs  = Input(shape=(input_dim_maccs,),  name="maccs_in")

    # View-1: Avalon tower
    v1 = Dense(64)(in_avalon)
    v1 = LeakyReLU(alpha=alpha)(v1)
    v1 = Dropout(dropout)(v1)

    v1 = Dense(32)(v1)
    v1 = LeakyReLU(alpha=alpha)(v1)
    v1 = Dropout(dropout)(v1)

    v1 = Dense(32)(v1)
    v1 = LeakyReLU(alpha=alpha)(v1)

    # View-2: MACCS tower
    v2 = Dense(128)(in_maccs)
    v2 = LeakyReLU(alpha=alpha)(v2)
    v2 = Dropout(dropout)(v2)

    v2 = Dense(64)(v2)
    v2 = LeakyReLU(alpha=alpha)(v2)

    # View-3: combined tower
    in_all = Concatenate(name="input_concat")([in_avalon, in_maccs])

    v3 = Dense(128)(in_all)
    v3 = LeakyReLU(alpha=alpha)(v3)
    v3 = Dropout(dropout)(v3)

    v3 = Dense(64)(v3)
    v3 = LeakyReLU(alpha=alpha)(v3)
    v3 = Dropout(dropout)(v3)

    v3 = Dense(32)(v3)
    v3 = LeakyReLU(alpha=alpha)(v3)

    # Merge
    z = Concatenate(name="views_concat")([v1, v2, v3])

    # Shared head  (kritik fark burası)
    z = Dense(64)(z)
    z = LeakyReLU(alpha=alpha)(z)
    z = Dropout(dropout)(z)

    z = Dense(16)(z)
    z = LeakyReLU(alpha=alpha)(z)

    z = Dense(8)(z)
    z = LeakyReLU(alpha=alpha)(z)

    z = Dense(4)(z)
    z = LeakyReLU(alpha=alpha)(z)

    out = Dense(1, activation="linear", name="y")(z)

    model = Model(inputs=[in_avalon, in_maccs], outputs=out, name="multiview_ann_3views")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def _summarize_with_mean_std(df: pd.DataFrame, metric_cols):
    df_out = df.copy()
    mean_row = {"fold": "MEAN"}
    std_row  = {"fold": "STD"}
    for c in metric_cols:
        mean_row[c] = float(df_out[c].mean())
        std_row[c]  = float(df_out[c].std(ddof=1)) if len(df_out) > 1 else 0.0

    for c in df_out.columns:
        if c not in metric_cols and c != "fold":
            mean_row.setdefault(c, "")
            std_row.setdefault(c, "")

    df_out = pd.concat([df_out, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    return df_out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", default="train_ready.csv", type=str)
    ap.add_argument("--test_csv", default="test_ready.csv", type=str)
    ap.add_argument("--outdir", default="run_multiview_ann_avalon_maccs", type=str)

    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--target", default="Dipole momentμ debye", type=str)
    ap.add_argument("--exclude", nargs="*", default=["Name", "Formula", "SMILES", "Dipole momentμ debye"])

    ap.add_argument("--epochs", default=300, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--val_split", default=0.15, type=float)
    ap.add_argument("--patience", default=50, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)

    ap.add_argument("--no_scaler", action="store_true")

    # CV
    ap.add_argument("--n_splits", default=5, type=int)

    args = ap.parse_args()

    safe_mkdir(args.outdir)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # -----------------------------
    # Load
    # -----------------------------
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    if args.target not in train_df.columns:
        raise ValueError(f"Target column not found in train: {args.target}")
    if args.target not in test_df.columns:
        raise ValueError(f"Target column not found in test: {args.target}")

    # -----------------------------
    # Define multiview columns by prefix
    # -----------------------------
    exclude_set = set(args.exclude)
    feat_train_cols = [c for c in train_df.columns if c not in exclude_set]

    avalon_cols = sorted([c for c in feat_train_cols if c.startswith("Avalon_")])
    maccs_cols  = sorted([c for c in feat_train_cols if c.startswith("MACCS_")])

    if len(avalon_cols) == 0:
        raise ValueError("No Avalon_ columns found in train (after exclude).")
    if len(maccs_cols) == 0:
        raise ValueError("No MACCS_ columns found in train (after exclude).")

    for c in avalon_cols + maccs_cols + [args.target]:
        if c not in test_df.columns:
            raise ValueError(f"Column missing in test: {c}")

    # -----------------------------
    # Prepare X/y (ALL TRAIN, then CV indices)
    # -----------------------------
    Xtr_avalon_df = coerce_numeric_df(train_df[avalon_cols])
    Xtr_maccs_df  = coerce_numeric_df(train_df[maccs_cols])
    ytr = pd.to_numeric(train_df[args.target], errors="coerce")

    Xte_avalon_df = coerce_numeric_df(test_df[avalon_cols])
    Xte_maccs_df  = coerce_numeric_df(test_df[maccs_cols])
    yte = pd.to_numeric(test_df[args.target], errors="coerce")

    # Drop NaN targets + align
    mtr = ytr.notna()
    Xtr_avalon_df = Xtr_avalon_df.loc[mtr].reset_index(drop=True)
    Xtr_maccs_df  = Xtr_maccs_df.loc[mtr].reset_index(drop=True)
    ytr = ytr.loc[mtr].reset_index(drop=True)

    mte = yte.notna()
    Xte_avalon_df = Xte_avalon_df.loc[mte].reset_index(drop=True)
    Xte_maccs_df  = Xte_maccs_df.loc[mte].reset_index(drop=True)
    yte = yte.loc[mte].reset_index(drop=True)

    print(f"[Train] {args.train_csv} avalon={Xtr_avalon_df.shape} maccs={Xtr_maccs_df.shape} y={ytr.shape}")
    print(f"[Test ] {args.test_csv}  avalon={Xte_avalon_df.shape} maccs={Xte_maccs_df.shape} y={yte.shape}")

    # Save feature lists once
    pd.DataFrame({"feature": avalon_cols}).to_csv(os.path.join(args.outdir, "features_avalon.csv"), index=False)
    pd.DataFrame({"feature": maccs_cols}).to_csv(os.path.join(args.outdir, "features_maccs.csv"), index=False)

    use_scaler = (not args.no_scaler)
    print("[Scaler] ON (separate StandardScaler per view)" if use_scaler else "[Scaler] OFF")

    # =========================================================
    # (C) FULL TRAIN FIRST  (as requested)
    # =========================================================
    full_dir = os.path.join(args.outdir, "full_train")
    safe_mkdir(full_dir)

    if use_scaler:
        scaler_a_full = StandardScaler()
        scaler_m_full = StandardScaler()
        Xtr_a_full = scaler_a_full.fit_transform(Xtr_avalon_df.values)
        Xtr_m_full = scaler_m_full.fit_transform(Xtr_maccs_df.values)
        Xte_a_full = scaler_a_full.transform(Xte_avalon_df.values)
        Xte_m_full = scaler_m_full.transform(Xte_maccs_df.values)
    else:
        scaler_a_full = None
        scaler_m_full = None
        Xtr_a_full = Xtr_avalon_df.values.astype(np.float32)
        Xtr_m_full = Xtr_maccs_df.values.astype(np.float32)
        Xte_a_full = Xte_avalon_df.values.astype(np.float32)
        Xte_m_full = Xte_maccs_df.values.astype(np.float32)

    ytr_full = ytr.values.astype(np.float32)
    yte_full = yte.values.astype(np.float32)

    tf.keras.backend.clear_session()
    model_full = build_multiview_model(
        input_dim_avalon=Xtr_a_full.shape[1],
        input_dim_maccs=Xtr_m_full.shape[1],
        lr=args.lr
    )

    ckpt_path_full = os.path.join(full_dir, "best_model.keras")
    callbacks_full = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path_full, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(5, args.patience // 3), min_lr=1e-6, verbose=1),
    ]

    print("\n==================== FULL TRAIN (START) ====================")
    history_full = model_full.fit(
        {"avalon_in": Xtr_a_full, "maccs_in": Xtr_m_full},
        ytr_full,
        validation_split=args.val_split,   # mevcut davranış korunuyor
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        shuffle=True,
        callbacks=callbacks_full
    )

    y_pred_full = model_full.predict(
        {"avalon_in": Xte_a_full, "maccs_in": Xte_m_full},
        batch_size=args.batch_size
    ).reshape(-1)
    met_full = metrics_reg(yte_full, y_pred_full)

    print("\n==================== FULL TRAIN -> TEST METRICS ====================")
    print(json.dumps(met_full, indent=2))

    # Save full train artifacts
    with open(os.path.join(full_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(met_full, f, indent=2)

    pd.DataFrame(history_full.history).to_csv(os.path.join(full_dir, "history.csv"), index=False)

    pd.DataFrame({"y_true": yte.values, "y_pred": y_pred_full.astype(float)}).to_csv(
        os.path.join(full_dir, "predictions_test.csv"), index=False
    )

    if scaler_a_full is not None and scaler_m_full is not None:
        scaler_info_full = {
            "avalon": {"mean_": scaler_a_full.mean_.tolist(), "scale_": scaler_a_full.scale_.tolist(), "feature_names": avalon_cols},
            "maccs":  {"mean_": scaler_m_full.mean_.tolist(), "scale_": scaler_m_full.scale_.tolist(), "feature_names": maccs_cols},
        }
        with open(os.path.join(full_dir, "scalers.json"), "w", encoding="utf-8") as f:
            json.dump(scaler_info_full, f)

    with open(os.path.join(full_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        model_full.summary(print_fn=lambda s: f.write(s + "\n"))

    model_full.save(os.path.join(full_dir, "final_model.keras"))

    # Save requested "full_train.cv" as CSV
    full_cv_csv = os.path.join(args.outdir, "full_train.cv.csv")
    pd.DataFrame([{
        "scope": "full_train",
        "eval_set": "test",
        "n_train": int(len(ytr_full)),
        "n_test": int(len(yte_full)),
        **met_full
    }]).to_csv(full_cv_csv, index=False)

    print("\n[Saved FULL TRAIN]")
    print(os.path.join(full_dir, "best_model.keras"))
    print(os.path.join(full_dir, "final_model.keras"))
    print(full_cv_csv)

    # =========================================================
    # (A,B) THEN 5-FOLD CV (as requested)
    # =========================================================
    print("\n==================== 5-FOLD CV (START) ====================")

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    holdout_rows = []
    test_rows = []

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(Xtr_avalon_df), start=1):
        fold_dir = os.path.join(args.outdir, f"fold_{fold_i}")
        safe_mkdir(fold_dir)

        # Split
        X_a_tr = Xtr_avalon_df.iloc[tr_idx]
        X_m_tr = Xtr_maccs_df.iloc[tr_idx]
        y_tr   = ytr.iloc[tr_idx].values.astype(np.float32)

        X_a_va = Xtr_avalon_df.iloc[va_idx]
        X_m_va = Xtr_maccs_df.iloc[va_idx]
        y_va   = ytr.iloc[va_idx].values.astype(np.float32)

        # Scale (fit on fold-train)
        if use_scaler:
            scaler_a = StandardScaler()
            scaler_m = StandardScaler()

            Xtr_a = scaler_a.fit_transform(X_a_tr.values)
            Xtr_m = scaler_m.fit_transform(X_m_tr.values)

            Xva_a = scaler_a.transform(X_a_va.values)
            Xva_m = scaler_m.transform(X_m_va.values)

            Xte_a = scaler_a.transform(Xte_avalon_df.values)
            Xte_m = scaler_m.transform(Xte_maccs_df.values)
        else:
            scaler_a = None
            scaler_m = None
            Xtr_a = X_a_tr.values.astype(np.float32)
            Xtr_m = X_m_tr.values.astype(np.float32)
            Xva_a = X_a_va.values.astype(np.float32)
            Xva_m = X_m_va.values.astype(np.float32)
            Xte_a = Xte_avalon_df.values.astype(np.float32)
            Xte_m = Xte_maccs_df.values.astype(np.float32)

        yte_np = yte.values.astype(np.float32)

        # Build
        tf.keras.backend.clear_session()
        model = build_multiview_model(
            input_dim_avalon=Xtr_a.shape[1],
            input_dim_maccs=Xtr_m.shape[1],
            lr=args.lr
        )

        ckpt_path = os.path.join(fold_dir, "best_model.keras")
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(5, args.patience // 3), min_lr=1e-6, verbose=1),
        ]

        print(f"\n==================== FOLD {fold_i}/{args.n_splits} ====================")
        print(f"[Fold {fold_i}] train={len(tr_idx)} holdout={len(va_idx)} test={len(yte_np)}")

        history = model.fit(
            {"avalon_in": Xtr_a, "maccs_in": Xtr_m},
            y_tr,
            validation_data=({"avalon_in": Xva_a, "maccs_in": Xva_m}, y_va),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
            shuffle=True,
            callbacks=callbacks
        )

        # Hold-out metrics
        y_va_pred = model.predict({"avalon_in": Xva_a, "maccs_in": Xva_m}, batch_size=args.batch_size).reshape(-1)
        met_hold = metrics_reg(y_va, y_va_pred)

        # Test metrics
        y_te_pred = model.predict({"avalon_in": Xte_a, "maccs_in": Xte_m}, batch_size=args.batch_size).reshape(-1)
        met_test = metrics_reg(yte_np, y_te_pred)

        print("[Fold Hold-out]", json.dumps(met_hold, indent=2))
        print("[Fold Test   ]", json.dumps(met_test, indent=2))

        # Save fold artifacts
        with open(os.path.join(fold_dir, "holdout_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(met_hold, f, indent=2)
        with open(os.path.join(fold_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(met_test, f, indent=2)

        pd.DataFrame(history.history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)

        if scaler_a is not None and scaler_m is not None:
            scaler_info = {
                "avalon": {"mean_": scaler_a.mean_.tolist(), "scale_": scaler_a.scale_.tolist(), "feature_names": avalon_cols},
                "maccs":  {"mean_": scaler_m.mean_.tolist(), "scale_": scaler_m.scale_.tolist(), "feature_names": maccs_cols},
            }
            with open(os.path.join(fold_dir, "scalers.json"), "w", encoding="utf-8") as f:
                json.dump(scaler_info, f)

        with open(os.path.join(fold_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
            model.summary(print_fn=lambda s: f.write(s + "\n"))

        model.save(os.path.join(fold_dir, "final_model.keras"))

        holdout_rows.append({
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_holdout": int(len(va_idx)),
            **met_hold
        })
        test_rows.append({
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(yte_np)),
            **met_test
        })

    # Save summary CSVs with MEAN/STD rows
    holdout_df = pd.DataFrame(holdout_rows)
    test_df_m  = pd.DataFrame(test_rows)

    metric_cols = ["R2", "MAE", "RMSE"]
    holdout_df2 = _summarize_with_mean_std(holdout_df, metric_cols)
    test_df2    = _summarize_with_mean_std(test_df_m, metric_cols)

    holdout_csv = os.path.join(args.outdir, "fold-holdout.csv")
    test_csv    = os.path.join(args.outdir, "fold-test.csv")

    holdout_df2.to_csv(holdout_csv, index=False)
    test_df2.to_csv(test_csv, index=False)

    print("\n==================== CV SUMMARY SAVED ====================")
    print(holdout_csv)
    print(test_csv)

    print("\n==================== FINAL OUTPUTS ====================")
    print("[Full Train] ", full_cv_csv)
    print("[CV Hold-out] ", holdout_csv)
    print("[CV Test   ] ", test_csv)
    print("\n[Models Saved]")
    print(f"- Full model : {os.path.join(full_dir, 'best_model.keras')} and full_train/final_model.keras")
    print(f"- Fold models: {os.path.join(args.outdir, 'fold_*/best_model.keras')} and fold_*/final_model.keras")


if __name__ == "__main__":
    main()
