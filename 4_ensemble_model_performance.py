import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# USER SETTINGS
# ============================================================

input_models = ["gnn", "catboost", "svr", "ann",]
input_models = ["gnn", "catboost",]

MODEL_DIRS = {
    "catboost": "run_catboost",
    "svr": "run_svr",
    "ann": "run_ann",
    "gnn": "run_gnn
    ",
    "lgbm": "run_lgbm"
}

MODEL_FILE_PATTERNS = {
    "catboost": [
        "predictions_test_cb_fp_svr_full.csv",
    ],
    "svr": [
        "predictions_test_svr_fp_svr_full.csv",
    ],
    "ann": [
        "predictions_test_ann_full.csv",
    ],
    "gnn": [
        "predictions_full_train_test.csv",
    ],
    "lgbm": [
        "predictions_test_lgbm_full.csv",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def detect_col(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Column not found. Candidates={candidates} | columns={list(df.columns)}")
    return None


def read_prediction_csv(path):
    df = pd.read_csv(path)

    # row index (GNN'deki row_index_in_test_csv kolonunu yakalaması için güncellendi)
    row_idx_col = detect_col(df, ["row_index_in_test_csv", "row_idx", "index", "idx", "unnamed: 0"], required=False)

    if row_idx_col is None:
        # Eğer indeks kolonu yoksa (CatBoost, SVR, ANN gibi modeller tüm veriyi içeriyorsa)
        # doğal sıralamasını indeks kabul ediyoruz.
        df["row_idx"] = np.arange(len(df))
        row_idx_col = "row_idx"

    # true / pred
    y_true_col = detect_col(df, ["y_true", "true", "target", "y"], required=False)
    y_pred_col = detect_col(df, ["y_pred", "pred", "prediction", "predictions"], required=True)

    out = pd.DataFrame({
        "row_idx": df[row_idx_col].astype(int),
        "y_pred": pd.to_numeric(df[y_pred_col], errors="coerce")
    })

    if y_true_col is not None:
        out["y_true"] = pd.to_numeric(df[y_true_col], errors="coerce")

    return out


def find_prediction_files(selected_models, model_dirs=None):
    if model_dirs is None:
        model_dirs = MODEL_DIRS

    found = []

    for m in selected_models:
        if m not in MODEL_FILE_PATTERNS:
            raise ValueError(f"Unknown model key: {m}")

        base_dir = model_dirs[m]
        patterns = MODEL_FILE_PATTERNS[m]

        for patt in patterns:
            path = os.path.join(base_dir, patt)
            if os.path.exists(path):
                found.append((m, os.path.basename(path).replace(".csv", ""), path))
            else:
                print(f"[WARN] file not found: {path}")

    return found


def build_global_truth(pred_infos):
    """
    Tüm dosyalardaki y_true bilgilerini row_idx üzerinden birleştir.
    """
    truth_parts = []

    for _, _, path in pred_infos:
        df = read_prediction_csv(path)
        if "y_true" in df.columns:
            truth_parts.append(df[["row_idx", "y_true"]].dropna())

    if not truth_parts:
        raise ValueError("Hiçbir prediction CSV içinde y_true bulunamadı.")

    truth_df = pd.concat(truth_parts, axis=0, ignore_index=True)
    truth_df = truth_df.drop_duplicates(subset=["row_idx"], keep="first")
    truth_df = truth_df.sort_values("row_idx").reset_index(drop=True)

    return truth_df


def align_predictions(pred_infos, truth_df):
    """
    truth_df üzerinden tüm prediction'ları hizalar.
    Döndürür:
      names, pred_matrix, y_true, aligned_frames
    """
    names = []
    pred_cols = []
    aligned_frames = []

    base = truth_df.copy()

    for model_group, name, path in pred_infos:
        df = read_prediction_csv(path)
        merged = base.merge(df[["row_idx", "y_pred"]], on="row_idx", how="left")
        names.append(name)
        pred_cols.append(merged["y_pred"].values.astype(float))
        aligned_frames.append((model_group, name, path, merged))

    pred_matrix = np.vstack(pred_cols)  # (n_models, N)
    y_true = base["y_true"].values.astype(float)

    return names, pred_matrix, y_true, aligned_frames


def metric_row(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(valid.sum())
    n_nan = int((~np.isfinite(y_pred)).sum())

    if n == 0:
        return {
            "R2": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "n": 0,
            "NaN": n_nan
        }

    yt = y_true[valid]
    yp = y_pred[valid]

    return {
        "R2": r2_score(yt, yp),
        "MAE": mean_absolute_error(yt, yp),
        "RMSE": rmse_score(yt, yp),
        "n": n,
        "NaN": n_nan
    }


def print_metric_line(name, metrics):
    print(
        f"  {name:<30} "
        f"R2={metrics['R2']:>7.4f}  "
        f"MAE={metrics['MAE']:.4f}  "
        f"RMSE={metrics['RMSE']:.4f}  "
        f"n={metrics['n']:>4}  "
        f"NaN={metrics['NaN']}"
    )


def adaptive_ensemble(pred_matrix, weights=None, model_names=None):
    """
    Her örnek için mevcut (non-NaN) tahminlerin ağırlıklı ortalamasını alır.
    GNN bazı satırlarda eksikse, kalan modellerin normalize edilmiş ağırlıklarıyla devam eder.
    """
    n_models, N = pred_matrix.shape

    if weights is None:
        weights = np.ones(n_models, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(weights) != n_models:
        raise ValueError(f"weights length mismatch: {len(weights)} != {n_models}")

    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_models)]

    ensemble = np.full(N, np.nan, dtype=float)
    models_used = []

    for j in range(N):
        col = pred_matrix[:, j]
        valid = np.isfinite(col)

        if valid.sum() == 0:
            models_used.append("")
            continue

        p_valid = col[valid]
        w_valid = weights[valid]

        if w_valid.sum() == 0:
            w_valid = np.ones_like(w_valid, dtype=float)

        ensemble[j] = np.dot(p_valid, w_valid) / w_valid.sum()

        used_names = [name for name, ok in zip(model_names, valid) if ok]
        models_used.append("+".join(used_names))

    return ensemble, models_used


def build_equal_weights(n_models):
    return [1.0 / n_models] * n_models


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="run_ensemble")
    parser.add_argument("--models", nargs="+", default=input_models,
                        help="Seçilecek model grupları: gnn svr catboost ann")
    parser.add_argument("--catboost_dir", type=str, default=MODEL_DIRS["catboost"])
    parser.add_argument("--svr_dir", type=str, default=MODEL_DIRS["svr"])
    parser.add_argument("--ann_dir", type=str, default=MODEL_DIRS["ann"])
    parser.add_argument("--gnn_dir", type=str, default=MODEL_DIRS["gnn"])
    parser.add_argument("--lgbm_dir", type=str, default=MODEL_DIRS["lgbm"])

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    selected_models = args.models
    model_dirs = {
        "catboost": args.catboost_dir,
        "svr": args.svr_dir,
        "ann": args.ann_dir,
        "gnn": args.gnn_dir,
        "lgbm": args.lgbm_dir,
    }

    print("=" * 60)
    print("SEÇİLEN MODEL GRUPLARI")
    print("=" * 60)
    print(selected_models)

    pred_infos = find_prediction_files(selected_models, model_dirs=model_dirs)

    if len(pred_infos) == 0:
        raise RuntimeError("Hiç prediction dosyası bulunamadı.")

    print("\n" + "=" * 60)
    print(f"[Scan] {len(pred_infos)} dosya bulundu:")
    print("=" * 60)
    for model_group, name, path in pred_infos:
        print(f"  {name:<30} ← {path}")

    truth_df = build_global_truth(pred_infos)
    names, pred_matrix, y_true, aligned_frames = align_predictions(pred_infos, truth_df)

    print("\n" + "=" * 60)
    print("PREDICTION CSV'LERİ YÜKLENİYOR")
    print("=" * 60)
    for (_, name, path, merged) in aligned_frames:
        n_rows = len(merged)
        n_pred = merged["y_pred"].notna().sum()
        print(f"  {name:<30} satır={n_pred}/{n_rows:<6} {path}")

    print(f"\n[Align] Test seti: {len(y_true)} satır  |  {len(names)} model")

    print("\n" + "=" * 60)
    print("BİREYSEL MODEL METRİKLERİ")
    print("=" * 60)

    individual_rows = []
    for i, name in enumerate(names):
        metrics = metric_row(y_true, pred_matrix[i])
        print_metric_line(name, metrics)

        individual_rows.append({
            "model_name": name,
            "R2": metrics["R2"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "n": metrics["n"],
            "NaN": metrics["NaN"],
        })

    ind_df = pd.DataFrame(individual_rows)
    ind_path = os.path.join(args.outdir, "individual_model_metrics.csv")
    ind_df.to_csv(ind_path, index=False)

    weights = build_equal_weights(len(names))

    print("\n" + "=" * 60)
    print("ENSEMBLE")
    print("=" * 60)
    print(f"Model listesi   : {names}")
    print(f"Eşit ağırlıklar : {weights}")

    p_ens, models_used = adaptive_ensemble(pred_matrix, weights=weights, model_names=names)
    ens_metrics = metric_row(y_true, p_ens)
    print_metric_line("ensemble_equal", ens_metrics)

    ensemble_df = truth_df.copy()
    for i, name in enumerate(names):
        ensemble_df[name] = pred_matrix[i]
    ensemble_df["ensemble_pred"] = p_ens
    ensemble_df["models_used"] = models_used

    ens_pred_path = os.path.join(args.outdir, "ensemble_predictions.csv")
    ensemble_df.to_csv(ens_pred_path, index=False)

    summary = {
        "selected_model_groups": selected_models,
        "prediction_files": [
            {"model_group": g, "name": n, "path": p}
            for g, n, p in pred_infos
        ],
        "used_model_names": names,
        "weights": weights,
        "ensemble_metrics": ens_metrics,
        "n_test_rows": int(len(y_true)),
    }

    summary_path = os.path.join(args.outdir, "ensemble_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("KAYDEDİLEN DOSYALAR")
    print("=" * 60)
    print(f"  {ind_path}")
    print(f"  {ens_pred_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
