# train_from_cleaned.py
# -*- coding: utf-8 -*-
"""
Training on mmasd_cleaned.csv (no raw data).
Two tasks:
  A) activity_grouped (3 classes)
  B) movement_intensity_bin (binary)

Models: LogisticRegression (with interactions) and XGBoost.
Validation: GroupKFold by group column from the file.

Input:   ./mmasd_cleaned.csv  (same folder as the script)
Output:  ./outputs/mmasd_metrics_activity.csv
         ./outputs/mmasd_metrics_intensity.csv
         ./outputs/features_used.txt
"""

from __future__ import annotations
import os, re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
)

from xgboost import XGBClassifier

# config
RANDOM_STATE = 42
N_SPLITS = 5

BASE_DIR = Path(__file__).resolve().parent
IN_CSV   = BASE_DIR / "mmasd_cleaned.csv"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_METRICS_ACT = OUT_DIR / "mmasd_metrics_activity.csv"
OUT_METRICS_BIN = OUT_DIR / "mmasd_metrics_intensity.csv"
OUT_FEATS_TXT   = OUT_DIR / "features_used.txt"


# helpers
def detect_group_col(df: pd.DataFrame) -> str:
    """Find a column for grouping (participant/group) for GroupKFold."""
    candidates = [
        "pid_group", "participant_id_global", "participant_id",
        "group", "group_id", "subject_id", "user_id"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "No group column found for GroupKFold. "
        "Add one of: " + ", ".join(candidates)
    )

def build_feature_list(df: pd.DataFrame, group_col: str) -> List[str]:
    """Select numeric features; exclude service columns and targets."""
    exclude_patterns = [
        rf"^{re.escape(group_col)}$",
        r"^sample_id$", r"^dataset$", r"^source_file$",
        r"^path_", r"_path$", r"^split_.*", r"^fold_.*",
        r"^group_kfold$", r"^global_id$",
        r"^movement_intensity_.*",             # exclude all variants of intensity target
        r"^activity.*",                        # exclude all variants of activity
        r"^engagement_.*",
        r"^duration_.*", r"^condition$", r"^has_.*",
        r"^sex$", r"^age_.*", r"^ados_.*",
    ]
    pats = [re.compile(p) for p in exclude_patterns]

    feats = []
    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if df[c].notna().sum() == 0:
            continue
        if any(p.search(c) for p in pats):
            continue
        # exclude constants
        if df[c].nunique(dropna=True) <= 1:
            continue
        feats.append(c)
    return sorted(feats)

def add_derived_features(df: pd.DataFrame, base_feats: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Simple derived features from existing ones (no raw data)."""
    new = df.copy()
    cols = set(base_feats)

    # Common fields from your table
    for a, b, name in [
        ("skel_max", "skel_median", "skel_range"),
        ("skel_std", "skel_median", "skel_cv"),
        ("skel_mad", "skel_iqr",   "skel_mad_ratio"),
        ("skel_std", "skel_duration_s", "skel_std_per_s"),
        ("skel_max", "skel_duration_s", "skel_max_per_s"),
    ]:
        if a in new.columns and b in new.columns:
            if name == "skel_range":
                new[name] = new[a] - new[b]
            elif name == "skel_cv":
                new[name] = new[a] / (new[b].abs() + 1e-8)
            elif name == "skel_mad_ratio":
                new[name] = new[a] / (new[b].abs() + 1e-8)
            elif name.endswith("_per_s"):
                new[name] = new[a] / (new[b].abs() + 1e-8)
            cols.add(name)

    # Log-transform for non-negative values
    for c in list(cols):
        s = new[c]
        if s.min(skipna=True) >= 0:
            new[f"{c}_log1p"] = np.log1p(s)
            cols.add(f"{c}_log1p")

    return new, sorted(cols)

def map_activity_to_3(name: str) -> str:
    """Fallback: if activity_grouped is missing — group basic class into 3 categories."""
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    if ("drum" in s) or ("marac" in s) or ("sing" in s and "clap" in s):
        return "music"
    if ("frog" in s) or ("tree" in s) or ("twist" in s):
        return "yoga"
    if ("arm" in s and "swing" in s) or ("body" in s and "swing" in s) or ("chest" in s) or ("squat" in s) or ("expansion" in s):
        return "movements"
    return np.nan

def macro_f1_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Safe macro-F1, no warnings if some classes are absent in validation."""
    labs = np.unique(y_true)
    try:
        return f1_score(y_true, y_pred, labels=labs, average="macro")
    except Exception:
        return f1_score(y_true, y_pred, average="macro")

def safe_auroc(y_true: np.ndarray, proba_pos: np.ndarray) -> float:
    labs = np.unique(y_true)
    if len(labs) < 2:
        return np.nan
    return roc_auc_score(y_true, proba_pos)

def best_f1_threshold(y_true: np.ndarray, proba_pos: np.ndarray) -> float:
    qs = np.linspace(0.05, 0.95, 19)
    thr_cands = np.quantile(proba_pos, qs)
    best_thr, best_f1 = 0.5, -1.0
    for t in np.unique(thr_cands):
        y_pred = (proba_pos >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return float(best_thr)


# CV runners
def cv_multiclass(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: str):
    gkf = GroupKFold(n_splits=N_SPLITS)
    n_classes = int(np.max(y)) + 1
    oof = np.zeros((len(y), n_classes), dtype=float)
    rows = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        # class weights by train set
        _, counts = np.unique(y[tr], return_counts=True)
        class_weight = {cls: (len(tr) / (n_classes * cnt)) for cls, cnt in zip(range(n_classes), counts)}
        sw_tr = np.array([class_weight[c] for c in y[tr]])

        if model == "logreg":
            clf = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ("sc", RobustScaler()),
                ("lg", LogisticRegression(
                    solver="lbfgs", max_iter=3000, random_state=RANDOM_STATE,
                    class_weight="balanced"
                ))
            ])
            clf.fit(X[tr], y[tr])
            proba = clf.predict_proba(X[va])

        elif model == "xgb":
            xgb = XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                n_estimators=1200,
                learning_rate=0.04,
                max_depth=5,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=1
            )
            xgb.fit(X[tr], y[tr], sample_weight=sw_tr)
            proba = xgb.predict_proba(X[va])

        else:
            raise ValueError("model must be 'logreg' or 'xgb'")

        oof[va] = proba
        y_hat = proba.argmax(axis=1)
        rows.append({
            "fold": fold,
            "acc": accuracy_score(y[va], y_hat),
            "macro_f1": macro_f1_safe(y[va], y_hat),
            "bal_acc": balanced_accuracy_score(y[va], y_hat),
        })

    return pd.DataFrame(rows), oof


def cv_binary(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: str):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y), dtype=float)
    rows = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        if model == "logreg":
            clf = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", RobustScaler()),
                ("lg", LogisticRegression(
                    solver="lbfgs", max_iter=3000, random_state=RANDOM_STATE,
                    class_weight="balanced"
                ))
            ])
            clf.fit(X[tr], y[tr])
            proba = clf.predict_proba(X[va])[:, 1]

        elif model == "xgb":
            pos = max(1, (y[tr] == 1).sum())
            neg = max(1, (y[tr] == 0).sum())
            spw = float(neg) / float(pos)

            xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_estimators=1200,
                learning_rate=0.04,
                max_depth=4,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                scale_pos_weight=spw,
                random_state=RANDOM_STATE,
                n_jobs=1
            )
            xgb.fit(X[tr], y[tr])
            proba = xgb.predict_proba(X[va])[:, 1]

        else:
            raise ValueError("model must be 'logreg' or 'xgb'")

        oof[va] = proba
        thr = best_f1_threshold(y[va], proba)
        y_hat = (proba >= thr).astype(int)
        rows.append({
            "fold": fold,
            "auroc": safe_auroc(y[va], proba),
            "bal_acc": balanced_accuracy_score(y[va], y_hat),
            "f1_pos": f1_score(y[va], y_hat, pos_label=1),
            "acc": accuracy_score(y[va], y_hat),
            "thr": thr
        })

    return pd.DataFrame(rows), oof


# main
def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"File not found: {IN_CSV}")
    df = pd.read_csv(IN_CSV)

    group_col = detect_group_col(df)

    # targets
    # activity_grouped (may already exist; if not — build from activity_class_basic)
    if "activity_grouped" not in df.columns:
        if "activity_class_basic" not in df.columns:
            raise ValueError("No 'activity_grouped' and no 'activity_class_basic'. Nothing to predict for task A.")
        df["activity_grouped"] = df["activity_class_basic"].map(map_activity_to_3)

    if "movement_intensity_bin" not in df.columns:
        raise ValueError("No 'movement_intensity_bin' for task B.")

    # features
    base_feats = build_feature_list(df, group_col)
    df2, feat_cols = add_derived_features(df, base_feats)

    # drop rows where all features are NaN
    mask_all_na = df2[feat_cols].isna().all(axis=1)
    df2 = df2.loc[~mask_all_na].copy()

    # Task A (multiclass)
    df_act = df2.dropna(subset=["activity_grouped", group_col]).copy()
    X_act = df_act[feat_cols].to_numpy(float)
    groups_act = df_act[group_col].astype(str).to_numpy()
    y_act_lab = df_act["activity_grouped"].astype(str).to_numpy()
    classes = sorted(pd.unique(y_act_lab))
    lab2idx = {lab: i for i, lab in enumerate(classes)}
    y_act = np.array([lab2idx[v] for v in y_act_lab], dtype=int)

    # Task B (binary)
    df_bin = df2.dropna(subset=["movement_intensity_bin", group_col]).copy()
    X_bin = df_bin[feat_cols].to_numpy(float)
    groups_bin = df_bin[group_col].astype(str).to_numpy()
    y_bin = df_bin["movement_intensity_bin"].astype(int).to_numpy()

    # info
    print(f"[i] Group column: {group_col}")
    print(f"[i] Features used: {len(feat_cols)} (base {len(base_feats)})")
    with open(OUT_FEATS_TXT, "w", encoding="utf-8") as f:
        for c in feat_cols:
            f.write(f"{c}\n")

    # CV RUNS
    # A) activity_grouped
    rows_act = []
    for mdl in ["logreg", "xgb"]:
        mdf, oof = cv_multiclass(X_act, y_act, groups_act, mdl)
        y_hat = oof.argmax(axis=1)
        rows_act.append({
            "task": "activity_grouped",
            "model": mdl,
            "n": int(len(y_hat)),
            "n_participants": int(pd.Series(groups_act).nunique()),
            "acc_mean_cv": float(mdf["acc"].mean()),
            "macro_f1_mean_cv": float(mdf["macro_f1"].mean()),
            "bal_acc_mean_cv": float(mdf["bal_acc"].mean()),
            "acc_oof": float(accuracy_score(y_act, y_hat)),
            "macro_f1_oof": float(macro_f1_safe(y_act, y_hat)),
            "bal_acc_oof": float(balanced_accuracy_score(y_act, y_hat)),
            "classes": "|".join(classes)
        })
    pd.DataFrame(rows_act).to_csv(OUT_METRICS_ACT, index=False)
    print(f"[✓] Saved: {OUT_METRICS_ACT}")

    # B) movement_intensity_bin
    rows_bin = []
    for mdl in ["logreg", "xgb"]:
        mdf, oofp = cv_binary(X_bin, y_bin, groups_bin, mdl)
        y_pred_05 = (oofp >= 0.5).astype(int)
        rows_bin.append({
            "task": "movement_intensity_bin",
            "model": mdl,
            "n": int(len(y_bin)),
            "n_participants": int(pd.Series(groups_bin).nunique()),
            "auroc_mean_cv": float(mdf["auroc"].mean()),
            "bal_acc_mean_cv": float(mdf["bal_acc"].mean()),
            "f1_pos_mean_cv": float(mdf["f1_pos"].mean()),
            "acc_mean_cv": float(mdf["acc"].mean()),
            "auroc_oof": float(roc_auc_score(y_bin, oofp) if len(np.unique(y_bin)) > 1 else np.nan),
            "bal_acc_oof_thr05": float(balanced_accuracy_score(y_bin, y_pred_05)),
            "f1_pos_oof_thr05": float(f1_score(y_bin, y_pred_05, pos_label=1)),
            "acc_oof_thr05": float(accuracy_score(y_bin, y_pred_05))
        })
    pd.DataFrame(rows_bin).to_csv(OUT_METRICS_BIN, index=False)
    print(f"[✓] Saved: {OUT_METRICS_BIN}")

    print("\nDone.")
    print(f"Participants (A): {pd.Series(groups_act).nunique()} | (B): {pd.Series(groups_bin).nunique()}")


if __name__ == "__main__":
    main()
