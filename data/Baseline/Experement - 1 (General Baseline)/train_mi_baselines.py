# -*- coding: utf-8 -*-
"""
train_mi_baselines_v4.py
Baseline with robust evaluation:
- MMASD (IID): meta-split train∪val vs test (holdout)
- Engagnition (IID): CV-only (GroupKFold by participant), no tiny holdout
- LODO: MMASD->Engagnition (all Eng as test); Engagnition->MMASD (MMASD meta test)

Outputs:
  outputs/tables/metrics_all.csv
  outputs/tables/metrics_iid.csv
  outputs/tables/metrics_lodo.csv
  outputs/tables/metrics_fairness.csv (if any)
  outputs/figures/reliability_*.png, fairness_auc_*.png
"""

import os, re, warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer

# progress bars
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs): return x  # no-op

# xgboost (optional)
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# CONFIG

CONFIG = {
    # paths
    "PATH_MMASD": r"MMASD_merged.xlsx",
    "PATH_ENG": r"Engagnition_features.xlsx",
    "PATH_META": r"metadata_ml_ready_splits_withGlobalID.xlsx",

    # fallback search dirs
    "FALLBACKS": [
        r"C:\Users\rusla\Desktop\Ruslan\AutDB-Video\data\frozen\v1_2025-09-13",
        r"C:\Users\rusla\Desktop\Ruslan\AutDB-Video\data"
    ],

    "OUT_DIR": "outputs",

    # label building
    "LABEL_VARIANT": "A",           # "A": robust-z per (dataset, participant) ; "B": per-dataset
    "LABEL_USE_EXISTING_BIN": True,
    "LABEL_B_QUANTILE": 0.5,

    # feature exclusions (regex)
    "EXCLUDE_PATTERNS": [
        r"^sample_id$", r"^participant_id_global$", r"^dataset$",
        r"^source_file$", r"^path_", r"_path$", r"^split_.*", r"^fold_.*",
        r"^group_kfold$", r"^global_id$",
        r"^movement_intensity_.*",
        r"^engagement_level$", r"^activity_class$",
        r"^duration_s$", r"^condition$", r"^has_.*",
        r"^sex$", r"^age_", r"^ados_.*"
    ],
    "ALLOW_SHORTCUTS": False,

    # CV / bootstrap
    "N_OUTER_SPLITS": 5,
    "TUNING_CV_SPLITS": 3,
    "N_TUNING_ITERS": 30,
    "RANDOM_STATE": 42,
    "N_BOOTSTRAP": 300,
    "BOOTSTRAP_SEED": 123,

    # model grids
    "LOGREG_PARAM_GRID": {"clf__C": np.logspace(-2, 2, 10)},
    "XGB_PARAM_GRID": {
        "max_depth": [3,4,5],
        "learning_rate": [0.03,0.05,0.1],
        "subsample": [0.7,0.8,0.9,1.0],
        "colsample_bytree": [0.6,0.7,0.8,0.9],
        "min_child_weight": [5,10,20],
        "reg_lambda": [1.0,5.0,10.0],
        "n_estimators": [300,500,800],
        "gamma": [0,0.5,1.0]
    },
}

# IO utils

def ensure_outdirs():
    for sub in ["tables", "figures"]:
        os.makedirs(os.path.join(CONFIG["OUT_DIR"], sub), exist_ok=True)

def _try_paths(path_rel: str) -> List[str]:
    cands = [path_rel]
    for fb in CONFIG["FALLBACKS"]:
        cands += [os.path.join(fb, os.path.basename(path_rel)),
                  os.path.join(fb, path_rel)]
    return cands

def try_read_excel(path_rel: str) -> pd.DataFrame:
    for p in _try_paths(path_rel):
        if os.path.isfile(p):
            return pd.read_excel(p)
    raise FileNotFoundError(f"Not found: {path_rel}\nTried:\n  " + "\n  ".join(_try_paths(path_rel)))

# ID helpers

def coerce_meta_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sample_id" not in df.columns:
        alts = [c for c in df.columns if c.lower() == "sample_id"]
        if alts: df["sample_id"] = df[alts[0]]
        else: raise ValueError("Meta: 'sample_id' column is required.")
    if "participant_id_global" not in df.columns:
        alts = [c for c in df.columns if c.lower() == "participant_id_global"]
        if alts: df["participant_id_global"] = df[alts[0]]
        else: raise ValueError("Meta: 'participant_id_global' column is required.")
    df["sample_id"] = df["sample_id"].astype(str)
    df["participant_id_global"] = df["participant_id_global"].astype(str)
    return df

def coerce_feature_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sample_id" not in df.columns:
        alts = [c for c in df.columns if c.lower() == "sample_id"]
        if alts: df["sample_id"] = df[alts[0]]
        else: raise ValueError("Feature table: 'sample_id' column is required.")
    df["sample_id"] = df["sample_id"].astype(str)
    return df

def infer_dataset_column(df: pd.DataFrame, default: str) -> pd.DataFrame:
    if "dataset" not in df.columns:
        df = df.copy(); df["dataset"] = default
    return df

# Merge + labels

def merge_all(df_meta: pd.DataFrame, df_mm: pd.DataFrame, df_eng: pd.DataFrame) -> pd.DataFrame:
    for name, d in [("meta", df_meta), ("MMASD", df_mm), ("Engagnition", df_eng)]:
        if d["sample_id"].duplicated().any():
            dups = d[d["sample_id"].duplicated()]["sample_id"].unique()
            raise ValueError(f"Duplicate sample_id in {name}: {dups[:10]} ...")
    merged = df_meta.merge(df_mm.drop(columns=["dataset"], errors="ignore"),
                           on="sample_id", how="left", suffixes=("", "_mm"))
    merged = merged.merge(df_eng.drop(columns=["dataset"], errors="ignore"),
                          on="sample_id", how="left", suffixes=("", "_eng"))
    if "dataset" not in merged.columns:
        sid = merged["sample_id"].astype(str)
        merged["dataset"] = np.where(sid.str.startswith(("ENG_","EN_")), "Engagnition",
                              np.where(sid.str.startswith(("MM_","as_","MMASD")), "MMASD", "UNKNOWN"))
    return merged

def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    if iqr == 0:
        iqr = np.nanstd(x) + 1e-6
    return (x - med) / (iqr if iqr != 0 else 1.0)

def build_labels(df: pd.DataFrame, variant: str = "A") -> pd.DataFrame:
    df = df.copy()
    raw_col = next((c for c in df.columns if c.lower() == "movement_intensity_raw"), None)
    if raw_col is not None:
        z_vals = np.zeros(len(df), float)
        if variant.upper() == "A":
            for (dset, pid), idx in df.groupby(["dataset", "participant_id_global"]).groups.items():
                z_vals[list(idx)] = robust_z(df.loc[idx, raw_col].values)
        else:
            for dset, idx in df.groupby("dataset").groups.items():
                z_vals[list(idx)] = robust_z(df.loc[idx, raw_col].values)
        df["movement_intensity_z_auto"] = z_vals
        if not CONFIG["LABEL_USE_EXISTING_BIN"] or "movement_intensity_bin" not in df.columns:
            if variant.upper() == "A":
                df["movement_intensity_bin"] = (df["movement_intensity_z_auto"] >= 0.0).astype(int)
            else:
                q = CONFIG["LABEL_B_QUANTILE"]
                bins = np.zeros(len(df), int)
                for dset, idx in df.groupby("dataset").groups.items():
                    thr = np.nanquantile(df.loc[idx, "movement_intensity_z_auto"].values, q)
                    bins[list(idx)] = (df.loc[idx, "movement_intensity_z_auto"].values >= thr).astype(int)
                df["movement_intensity_bin"] = bins
    else:
        if "movement_intensity_bin" not in df.columns:
            raise ValueError("No movement_intensity_raw and no movement_intensity_bin present.")
        df["movement_intensity_z_auto"] = np.nan

    df["movement_intensity_bin"] = df["movement_intensity_bin"].astype(int)
    return df


# Feature selection & metrics
def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    pats = [re.compile(p) for p in CONFIG["EXCLUDE_PATTERNS"]]
    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if df[c].notna().sum() == 0:
            continue
        if any(p.search(c) for p in pats):
            continue
        cols.append(c)
    if not CONFIG["ALLOW_SHORTCUTS"]:
        cols = [c for c in cols if c not in ("duration_s", "condition")]
    return sorted(cols)

class CorrFilter(BaseEstimator, TransformerMixin):
    def __init__(self, thr: float = 0.98):
        self.thr = thr
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        corr = np.corrcoef(X, rowvar=False)
        n = corr.shape[0]
        keep = np.ones(n, bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if keep[j] and abs(corr[i, j]) > self.thr:
                    keep[j] = False
        self.keep_ = keep
        return self
    def transform(self, X):
        check_is_fitted(self, "keep_")
        X = np.asarray(X, float)
        return X[:, self.keep_]

def weights_by_participant(pids: pd.Series) -> np.ndarray:
    counts = pids.value_counts()
    return pids.map(lambda pid: 1.0 / counts[pid]).values

def compute_metrics(y_true, y_proba, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "bacc":  balanced_accuracy_score(y_true, y_pred),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
        "acc":   accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_proba),
    }

def best_threshold_for_f1(y_true, y_proba) -> float:
    qs = np.linspace(0.01, 0.99, 199)
    thr_cands = np.quantile(y_proba, qs)
    best_thr, best_f1 = 0.5, -1.0
    for t in np.unique(thr_cands):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), pos_label=1)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr)

def cluster_bootstrap_metrics(y_true, y_proba, groups, metric_fn, n_boot=300, seed=123):
    rng = check_random_state(seed)
    uniq = np.array(sorted(pd.Series(groups).unique()))
    vals = []
    for _ in range(n_boot):
        sel = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, sel)
        if mask.sum() < 2 or len(np.unique(np.asarray(y_true)[mask])) < 2:
            continue
        vals.append(metric_fn(np.asarray(y_true)[mask], np.asarray(y_proba)[mask]))
    point = metric_fn(y_true, y_proba)
    if not vals:
        return point, (np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, (float(lo), float(hi))

def plot_reliability(y_true, y_proba, fig_path: str, title: str):
    try:
        import matplotlib.pyplot as plt
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='quantile')
        plt.figure(figsize=(4.8, 4.8))
        plt.plot([0,1],[0,1],'--', lw=1)
        plt.plot(mean_pred, frac_pos, marker='o')
        plt.xlabel("Predicted probability"); plt.ylabel("Fraction positive"); plt.title(title)
        plt.tight_layout(); plt.savefig(fig_path, dpi=160); plt.close()
    except Exception as e:
        print(f"[warn] reliability plot failed: {e}")

# Models & calibration

def build_logreg_pipeline(n_features: int) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("corr", CorrFilter(thr=0.98))
    ]
    clf = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500
    )
    return Pipeline(steps + [("clf", clf)])

def build_xgb_classifier(scale_pos_weight: float = 1.0) -> "XGBClassifier":
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=1, verbosity=0,
        scale_pos_weight=scale_pos_weight
    )

class IsoCalibratedModel:
    def __init__(self, base, iso: IsotonicRegression):
        self.base = base
        self.iso  = iso
    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        p_cal = self.iso.predict(p)
        p_cal = np.clip(p_cal, 1e-6, 1-1e-6)
        return np.column_stack([1 - p_cal, p_cal])

def grouped_isotonic_calibration(best_estimator, X, y, groups) -> IsoCalibratedModel:
    gkf = GroupKFold(n_splits=CONFIG["TUNING_CV_SPLITS"])
    oof_p, oof_y = [], []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        est = clone(best_estimator).fit(X[tr_idx], y[tr_idx])
        oof_p.append(est.predict_proba(X[va_idx])[:, 1])
        oof_y.append(y[va_idx])
    oof_p = np.concatenate(oof_p); oof_y = np.concatenate(oof_y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_p, oof_y)
    final_est = clone(best_estimator).fit(X, y)
    return IsoCalibratedModel(final_est, iso)

# Train / Eval

def fit_with_nested_cv(X, y, groups, model_name: str, rs: int = 42) -> Tuple[IsoCalibratedModel, Dict[str,float], float, np.ndarray]:
    """Return calibrated model, CV metrics, optimal threshold, and OOF probabilities."""
    rng = check_random_state(rs)
    outer_cv = GroupKFold(n_splits=CONFIG["N_OUTER_SPLITS"])
    oof_proba = np.zeros_like(y, float)
    thr_parts = []

    for tr_idx, va_idx in tqdm(list(outer_cv.split(X, y, groups)), desc=f"NestedCV[{model_name}]", leave=False):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        gtr      = groups[tr_idx]

        inner_cv = GroupKFold(n_splits=CONFIG["TUNING_CV_SPLITS"])

        if model_name == "logreg":
            base_est = build_logreg_pipeline(Xtr.shape[1])
            param_grid = CONFIG["LOGREG_PARAM_GRID"]
        elif model_name == "xgb":
            if not XGB_OK:
                raise RuntimeError("xgboost not installed. Install with: py -m pip install xgboost")
            scale_pos = (ytr == 0).sum() / max(1, (ytr == 1).sum())
            base_est = build_xgb_classifier(scale_pos_weight=scale_pos)
            param_grid = CONFIG["XGB_PARAM_GRID"]
        else:
            raise ValueError("Unknown model name")

        tuner = RandomizedSearchCV(
            estimator=base_est,
            param_distributions=param_grid,
            n_iter=CONFIG["N_TUNING_ITERS"],
            scoring="roc_auc",
            cv=inner_cv.split(Xtr, ytr, gtr),
            n_jobs=1, random_state=rng
        )
        tuner.fit(Xtr, ytr)
        best_est = tuner.best_estimator_

        cal_model = grouped_isotonic_calibration(best_est, Xtr, ytr, gtr)
        proba_va = cal_model.predict_proba(Xva)[:, 1]
        oof_proba[va_idx] = proba_va

        thr_fold = best_threshold_for_f1(yva, proba_va)
        thr_parts.append((len(va_idx), thr_fold))

    thr = np.average([t for n, t in thr_parts], weights=[n for n, t in thr_parts]) if thr_parts else 0.5
    cv_metrics = compute_metrics(y, oof_proba, threshold=thr)

    # final model on all data
    if model_name == "logreg":
        base_est = build_logreg_pipeline(X.shape[1])
        param_grid = CONFIG["LOGREG_PARAM_GRID"]
    else:
        scale_pos = (y == 0).sum() / max(1, (y == 1).sum())
        base_est = build_xgb_classifier(scale_pos_weight=scale_pos)
        param_grid = CONFIG["XGB_PARAM_GRID"]

    tuner_full = RandomizedSearchCV(
        estimator=base_est, param_distributions=param_grid,
        n_iter=min(CONFIG["N_TUNING_ITERS"], 20), scoring="roc_auc",
        cv=GroupKFold(n_splits=CONFIG["TUNING_CV_SPLITS"]).split(X, y, groups),
        n_jobs=1, random_state=rs
    )
    tuner_full.fit(X, y)
    best_est_full = tuner_full.best_estimator_
    calibrated_full = grouped_isotonic_calibration(best_est_full, X, y, groups)

    return calibrated_full, cv_metrics, float(thr), oof_proba

def eval_on_holdout(model, X_te, y_te, thr, title_prefix, fig_stub, out_dir, pid_holdout):
    proba = model.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_te, proba, threshold=thr)
    plot_reliability(y_te, proba,
                     fig_path=os.path.join(out_dir, "figures", f"reliability_{fig_stub}.png"),
                     title=f"{title_prefix} reliability")
    # bootstrap CI for AUROC using participant clusters
    def auc_fn(a, b):
        return roc_auc_score(a, b) if len(np.unique(a)) > 1 else np.nan
    auc_point, (auc_lo, auc_hi) = cluster_bootstrap_metrics(
        y_te, proba, pid_holdout, metric_fn=auc_fn,
        n_boot=CONFIG["N_BOOTSTRAP"], seed=CONFIG["BOOTSTRAP_SEED"]
    )
    m["auroc_ci_low"] = auc_lo; m["auroc_ci_high"] = auc_hi
    return m, proba

def fairness_slices(y, proba, pid_groups: pd.Series, groups_dict: Dict[str, pd.Series],
                    fig_stub: str, out_dir: str) -> pd.DataFrame:
    rows = []
    for gname, gser in groups_dict.items():
        if gser is None or gser.isna().all():
            continue
        ser = gser.astype(str)
        for val in sorted(ser.dropna().unique()):
            mask = (ser == val).values
            if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
                continue
            y_sub, p_sub = y[mask], proba[mask]
            import numpy as _np
            pid_arr = _np.asarray(pid_groups) 
            pid_sub = pid_arr[mask]
            def auc_fn(a, b):
                return roc_auc_score(a, b) if len(np.unique(a)) > 1 else np.nan
            auc_point, (auc_lo, auc_hi) = cluster_bootstrap_metrics(
                y_sub, p_sub, pid_sub, metric_fn=auc_fn,
                n_boot=CONFIG["N_BOOTSTRAP"], seed=CONFIG["BOOTSTRAP_SEED"]
            )
            thr_sub = best_threshold_for_f1(y_sub, p_sub)
            met = compute_metrics(y_sub, p_sub, thr_sub)
            rows.append({
                "group": gname, "value": val, "n": int(mask.sum()),
                "auroc": auc_point, "auroc_lo": auc_lo, "auroc_hi": auc_hi,
                "bacc": met["bacc"], "f1_pos": met["f1_pos"], "acc": met["acc"], "brier": met["brier"]
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # quick bars
    try:
        import matplotlib.pyplot as plt
        for gname, gdf in df.groupby("group"):
            xs = gdf["value"].astype(str).tolist()
            ys = gdf["auroc"].values
            ylo = gdf["auroc_lo"].values; yhi = gdf["auroc_hi"].values
            plt.figure(figsize=(6.0, 3.6))
            plt.bar(xs, ys)
            eb_lo = ys - ylo; eb_hi = yhi - ys
            plt.errorbar(xs, ys, yerr=[eb_lo, eb_hi], fmt='none', ecolor='k', capsize=4)
            plt.ylabel("AUC"); plt.title(f"Fairness AUC by {gname}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "figures", f"fairness_auc_{fig_stub}_{gname}.png"), dpi=160)
            plt.close()
    except Exception as e:
        print(f"[warn] fairness plot failed: {e}")
    return df

# MAIN

def main():
    ensure_outdirs()
    print("[i] Loading tables...")
    df_mm   = try_read_excel(CONFIG["PATH_MMASD"])
    df_eng  = try_read_excel(CONFIG["PATH_ENG"])
    df_meta = try_read_excel(CONFIG["PATH_META"])

    df_mm   = infer_dataset_column(df_mm, "MMASD")
    df_eng  = infer_dataset_column(df_eng, "Engagnition")

    df_meta = coerce_meta_ids(df_meta)
    df_mm   = coerce_feature_ids(df_mm)
    df_eng  = coerce_feature_ids(df_eng)

    print("[i] Merging...")
    df = merge_all(df_meta, df_mm, df_eng)

    mask_unk = df["dataset"].eq("UNKNOWN")
    if mask_unk.any():
        sid = df.loc[mask_unk, "sample_id"].astype(str)
        df.loc[mask_unk & sid.str.startswith(("ENG_","EN_")), "dataset"] = "Engagnition"
        df.loc[mask_unk & sid.str.startswith(("MM_","as_","MMASD")), "dataset"] = "MMASD"

    print(f"[i] Building labels variant={CONFIG['LABEL_VARIANT']}, use_existing_bin={CONFIG['LABEL_USE_EXISTING_BIN']}")
    df = build_labels(df, variant=CONFIG["LABEL_VARIANT"])
    df = df[~df["movement_intensity_bin"].isna()].copy()
    df["movement_intensity_bin"] = df["movement_intensity_bin"].astype(int)

    feat_cols = select_feature_columns(df)
    if len(feat_cols) < 3:
        raise ValueError(f"Too few features selected ({len(feat_cols)}). Check EXCLUDE_PATTERNS or your tables.")
    print(f"[i] Selected {len(feat_cols)} numeric features for modeling.")

    df["w_pid"] = weights_by_participant(df["participant_id_global"])
    split_iid_col = next((c for c in df.columns if c.lower() == "split_iid"), None)

    metrics_rows = []
    fairness_rows = []
    rng = check_random_state(CONFIG["RANDOM_STATE"])

    # IID: MMASD (holdout)
    dataset_name = "MMASD"
    dset = df[df["dataset"] == dataset_name].copy()
    if not dset.empty and dset["movement_intensity_bin"].nunique() > 1:
        X = dset[feat_cols].values.astype(float)
        y = dset["movement_intensity_bin"].values.astype(int)
        pid = dset["participant_id_global"].values

        # meta-split train∪val vs test ; fallback to group 80/20
        if split_iid_col and set(dset[split_iid_col].astype(str).str.lower().unique()) & {"train","val","test"}:
            sv = dset[split_iid_col].astype(str).str.lower()
            tr_mask = sv.isin(["train","val"])
            te_mask = sv.eq("test")
        else:
            uniq = np.array(sorted(pd.Series(pid).unique()))
            rng.shuffle(uniq); te_g = set(uniq[:max(1, len(uniq)//5)])
            te_mask = np.array([g in te_g for g in pid]); tr_mask = ~te_mask

        n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
        print(f"[i] IID MMASD: train n={n_tr} (p={dset.loc[tr_mask,'participant_id_global'].nunique()}), "
              f"test n={n_te} (p={dset.loc[te_mask,'participant_id_global'].nunique()})")

        Xtr, ytr, gtr = X[tr_mask], y[tr_mask], pid[tr_mask]
        for model_name in tqdm(["logreg", "xgb"], desc="Models[MMASD]", leave=False):
            mdl, cvm, thr, _ = fit_with_nested_cv(Xtr, ytr, gtr, model_name=model_name, rs=CONFIG["RANDOM_STATE"])
            m_hold, proba_hold = eval_on_holdout(
                mdl, X[te_mask], y[te_mask], thr,
                title_prefix=f"IID MMASD {model_name}",
                fig_stub=f"IID_MMASD_{model_name}",
                out_dir=CONFIG["OUT_DIR"],
                pid_holdout=pid[te_mask]
            )
            row = {"task":"IID","dataset":"MMASD","model":model_name,"split":"holdout",
                   "n_train": n_tr, "n_test": n_te, "thr": thr}
            for k, v in cvm.items(): row[f"cv_{k}"] = v
            for k, v in m_hold.items(): row[f"test_{k}"] = v
            metrics_rows.append(row)

            fair_groups = {
                "sex": dset.loc[te_mask, "sex"] if "sex" in dset.columns else None,
                "age_group": dset.loc[te_mask, "age_group"] if "age_group" in dset.columns else None
            }
            df_fair = fairness_slices(
                y=y[te_mask], proba=proba_hold, pid_groups=dset.loc[te_mask, "participant_id_global"],
                groups_dict=fair_groups, fig_stub=f"IID_MMASD_{model_name}",
                out_dir=CONFIG["OUT_DIR"]
            )
            if not df_fair.empty:
                df_fair.insert(0, "task", "IID"); df_fair.insert(1, "dataset", "MMASD"); df_fair.insert(2, "model", model_name)
                fairness_rows.append(df_fair)
    else:
        print("[warn] IID skipped for MMASD (empty or single-class).")

    # IID: Engagnition (CV-only)
    dataset_name = "Engagnition"
    dset = df[df["dataset"] == dataset_name].copy()
    if not dset.empty and dset["movement_intensity_bin"].nunique() > 1:
        X = dset[feat_cols].values.astype(float)
        y = dset["movement_intensity_bin"].values.astype(int)
        pid = dset["participant_id_global"].values

        print(f"[i] IID Engagnition: CV-only on all data (n={len(y)}, p={dset['participant_id_global'].nunique()})")
        for model_name in tqdm(["logreg", "xgb"], desc="Models[Engagnition-CV]", leave=False):
            mdl, cvm, thr, oof = fit_with_nested_cv(X, y, pid, model_name=model_name, rs=CONFIG["RANDOM_STATE"])
            # OOF reliability & fairness on CV
            plot_reliability(y, oof,
                fig_path=os.path.join(CONFIG["OUT_DIR"], "figures", f"reliability_IID_Engagnition_{model_name}_OOF.png"),
                title=f"IID Engagnition {model_name} (OOF)")
            fair_groups = {
                "sex": dset["sex"] if "sex" in dset.columns else None,
                "age_group": dset["age_group"] if "age_group" in dset.columns else None
            }
            df_fair = fairness_slices(
                y=y, proba=oof, pid_groups=dset["participant_id_global"],
                groups_dict=fair_groups, fig_stub=f"IID_Engagnition_{model_name}_OOF",
                out_dir=CONFIG["OUT_DIR"]
            )
            if not df_fair.empty:
                df_fair.insert(0, "task", "IID"); df_fair.insert(1, "dataset", "Engagnition"); df_fair.insert(2, "model", model_name)
                fairness_rows.append(df_fair)

            row = {"task":"IID","dataset":"Engagnition","model":model_name,"split":"cv",
                   "n_train": int(len(y)), "n_test": 0, "thr": thr}
            for k, v in cvm.items(): row[f"cv_{k}"] = v
            # no holdout test here
            metrics_rows.append(row)
    else:
        print("[warn] IID skipped for Engagnition (empty or single-class).")

    # LODO
    # MMASD -> Engagnition (test = ALL Engagnition)
    dtr = df[df["dataset"] == "MMASD"].copy()
    dte = df[df["dataset"] == "Engagnition"].copy()
    if not dtr.empty and not dte.empty and dtr["movement_intensity_bin"].nunique() > 1 and dte["movement_intensity_bin"].nunique() > 1:
        Xtr = dtr[feat_cols].values.astype(float); ytr = dtr["movement_intensity_bin"].values.astype(int); gtr = dtr["participant_id_global"].values
        Xte = dte[feat_cols].values.astype(float); yte = dte["movement_intensity_bin"].values.astype(int); gte = dte["participant_id_global"].values
        for model_name in tqdm(["logreg", "xgb"], desc="Models[MMASD->Engagnition]", leave=False):
            mdl, cvm, thr, _ = fit_with_nested_cv(Xtr, ytr, gtr, model_name=model_name, rs=CONFIG["RANDOM_STATE"])
            m_hold, proba_hold = eval_on_holdout(
                mdl, Xte, yte, thr,
                title_prefix=f"LODO MMASD->Engagnition {model_name}",
                fig_stub=f"LODO_MMASD_Engagnition_{model_name}",
                out_dir=CONFIG["OUT_DIR"],
                pid_holdout=gte
            )
            row = {"task":"LODO","dataset":"MMASD->Engagnition","model":model_name,"split":"transfer",
                   "n_train": int(len(ytr)), "n_test": int(len(yte)), "thr": thr}
            for k, v in cvm.items(): row[f"cv_{k}"] = v
            for k, v in m_hold.items(): row[f"test_{k}"] = v
            metrics_rows.append(row)

            fair_groups = {"sex": dte["sex"] if "sex" in dte.columns else None,
                           "age_group": dte["age_group"] if "age_group" in dte.columns else None}
            df_fair = fairness_slices(y=yte, proba=proba_hold, pid_groups=gte,
                                      groups_dict=fair_groups, fig_stub=f"LODO_MMASD_Engagnition_{model_name}",
                                      out_dir=CONFIG["OUT_DIR"])
            if not df_fair.empty:
                df_fair.insert(0, "task", "LODO"); df_fair.insert(1, "dataset", "MMASD->Engagnition"); df_fair.insert(2, "model", model_name)
                fairness_rows.append(df_fair)
    else:
        print("[warn] LODO skipped MMASD->Engagnition (empty or single-class).")

    # Engagnition -> MMASD (test = MMASD meta test if available)
    dtr = df[df["dataset"] == "Engagnition"].copy()
    dte_full = df[df["dataset"] == "MMASD"].copy()
    if not dtr.empty and not dte_full.empty and dtr["movement_intensity_bin"].nunique() > 1 and dte_full["movement_intensity_bin"].nunique() > 1:
        if split_iid_col and set(dte_full[split_iid_col].astype(str).str.lower().unique()) & {"train","val","test"}:
            mask = dte_full[split_iid_col].astype(str).str.lower().eq("test")
            dte = dte_full[mask].copy()
            print(f"[i] LODO Engagnition->MMASD: using MMASD meta test (n={len(dte)})")
        else:
            dte = dte_full.copy()
            print(f"[i] LODO Engagnition->MMASD: meta test not found -> using ALL MMASD (n={len(dte)})")

        Xtr = dtr[feat_cols].values.astype(float); ytr = dtr["movement_intensity_bin"].values.astype(int); gtr = dtr["participant_id_global"].values
        Xte = dte[feat_cols].values.astype(float); yte = dte["movement_intensity_bin"].values.astype(int); gte = dte["participant_id_global"].values

        for model_name in tqdm(["logreg", "xgb"], desc="Models[Engagnition->MMASD]", leave=False):
            mdl, cvm, thr, _ = fit_with_nested_cv(Xtr, ytr, gtr, model_name=model_name, rs=CONFIG["RANDOM_STATE"])
            m_hold, proba_hold = eval_on_holdout(
                mdl, Xte, yte, thr,
                title_prefix=f"LODO Engagnition->MMASD {model_name}",
                fig_stub=f"LODO_Engagnition_MMASD_{model_name}",
                out_dir=CONFIG["OUT_DIR"],
                pid_holdout=gte
            )
            row = {"task":"LODO","dataset":"Engagnition->MMASD","model":model_name,"split":"transfer",
                   "n_train": int(len(ytr)), "n_test": int(len(yte)), "thr": thr}
            for k, v in cvm.items(): row[f"cv_{k}"] = v
            for k, v in m_hold.items(): row[f"test_{k}"] = v
            metrics_rows.append(row)

            fair_groups = {"sex": dte["sex"] if "sex" in dte.columns else None,
                           "age_group": dte["age_group"] if "age_group" in dte.columns else None}
            df_fair = fairness_slices(y=yte, proba=proba_hold, pid_groups=gte,
                                      groups_dict=fair_groups, fig_stub=f"LODO_Engagnition_MMASD_{model_name}",
                                      out_dir=CONFIG["OUT_DIR"])
            if not df_fair.empty:
                df_fair.insert(0, "task", "LODO"); df_fair.insert(1, "dataset", "Engagnition->MMASD"); df_fair.insert(2, "model", model_name)
                fairness_rows.append(df_fair)
    else:
        print("[warn] LODO skipped Engagnition->MMASD (empty or single-class).")

    # save tables
    out_tab = os.path.join(CONFIG["OUT_DIR"], "tables")
    os.makedirs(out_tab, exist_ok=True)
    dfm = pd.DataFrame(metrics_rows)
    dfm.to_csv(os.path.join(out_tab, "metrics_all.csv"), index=False)
    if not dfm.empty:
        dfm[dfm["task"] == "IID"].to_csv(os.path.join(out_tab, "metrics_iid.csv"), index=False)
        dfm[dfm["task"] == "LODO"].to_csv(os.path.join(out_tab, "metrics_lodo.csv"), index=False)
    if fairness_rows:
        pd.concat(fairness_rows, ignore_index=True).to_csv(os.path.join(out_tab, "metrics_fairness.csv"), index=False)
    print("[✓] Done. See outputs/tables and outputs/figures")

if __name__ == "__main__":
    main()
