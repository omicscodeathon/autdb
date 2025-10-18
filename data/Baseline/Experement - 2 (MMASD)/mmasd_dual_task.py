# -*- coding: utf-8 -*-
"""
MMASD-only, two tasks, local-folder version
Models: Logistic Regression (sklearn) + XGBoost
CV: GroupKFold(n_splits=5) by participant_id_global (from metadata)

Place in ONE folder:
  - mmasd_dual_task.py (this file)
  - MMASD_merged.xlsx
  - metadata_ml_ready_splits_withGlobalID.xlsx

Results will appear in ./outputs/
"""

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
)

from xgboost import XGBClassifier

# CONFIG
RANDOM_STATE = 42
N_SPLITS = 5

#
BASE_DIR = Path(__file__).resolve().parent
PATH_MM   = BASE_DIR / "MMASD_merged.xlsx"
PATH_META = BASE_DIR / "metadata_ml_ready_splits_withGlobalID.xlsx"

OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_CLEAN = OUT_DIR / "mmasd_cleaned.csv"
OUT_METRICS_ACTIVITY = OUT_DIR / "mmasd_metrics_activity.csv"
OUT_METRICS_INTENSITY = OUT_DIR / "mmasd_metrics_intensity.csv"


# UTILS
def _safe_read_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_excel(path)

def build_feature_list(df: pd.DataFrame) -> List[str]:
    """Select only numeric features; drop service/target columns."""
    exclude_patterns = [
        r"^sample_id$", r"^participant_id$", r"^participant_id_global$", r"^pid_group$",
        r"^dataset$", r"^source_file$", r"^path_", r"_path$", r"^split_.*", r"^fold_.*",
        r"^group_kfold$", r"^global_id$",
        r"^movement_intensity_.*",          # exclude intensity targets
        r"^engagement_level$", r"^activity.*",  # exclude activity/engagement as features
        r"^duration_s$", r"^condition$", r"^has_.*",
        r"^sex$", r"^age_", r"^ados_.*"
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
        feats.append(c)
    # remove constants
    feats = [c for c in feats if df[c].nunique(dropna=True) > 1]
    return sorted(feats)

def map_activity_to_3(name: str) -> str:
    """Group 11 activity classes into 3 broad categories."""
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    # music
    if ("drum" in s) or ("marac" in s) or ("sing" in s and "clap" in s):
        return "music"
    # yoga
    if ("frog" in s) or ("tree" in s) or ("twist" in s):
        return "yoga"
    # general movements
    if ("arm" in s and "swing" in s) or ("body" in s and "swing" in s) or ("chest" in s) or ("squat" in s) or ("expansion" in s):
        return "movements"
    return np.nan

def safe_auroc(y_true: np.ndarray, proba_pos: np.ndarray) -> float:
    uniq = np.unique(y_true)
    if len(uniq) < 2:
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


# CV RUNNERS
def cv_multiclass(
    X: np.ndarray, y_idx: np.ndarray, groups: np.ndarray, model: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    gkf = GroupKFold(n_splits=N_SPLITS)
    n_classes = int(np.max(y_idx)) + 1
    oof_pred = np.zeros((len(y_idx), n_classes), dtype=float)
    rows = []

    for fold, (tr, va) in enumerate(gkf.split(X, y_idx, groups)):
        if model == "logreg":
            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(
                    multi_class="multinomial",
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=500,
                    random_state=RANDOM_STATE
                ))
            ])
        elif model == "xgb":
            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=5,
                n_jobs=1,
                random_state=RANDOM_STATE
            )
        else:
            raise ValueError("model must be 'logreg' or 'xgb'")

        clf.fit(X[tr], y_idx[tr])
        proba = clf.predict_proba(X[va])
        oof_pred[va] = proba
        y_hat = proba.argmax(axis=1)

        rows.append({
            "fold": fold,
            "acc": accuracy_score(y_idx[va], y_hat),
            "macro_f1": f1_score(y_idx[va], y_hat, average="macro"),
            "bal_acc": balanced_accuracy_score(y_idx[va], y_hat),
        })
    return pd.DataFrame(rows), oof_pred


def cv_binary(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_proba = np.zeros(len(y), dtype=float)
    rows = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        if model == "logreg":
            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=500,
                    random_state=RANDOM_STATE
                ))
            ])
        elif model == "xgb":
            pos = max(1, (y[tr] == 1).sum())
            neg = max(1, (y[tr] == 0).sum())
            spw = float(neg) / float(pos)
            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_estimators=600,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=5,
                scale_pos_weight=spw,
                n_jobs=1,
                random_state=RANDOM_STATE
            )
        else:
            raise ValueError("model must be 'logreg' or 'xgb'")

        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[va])[:, 1]
        oof_proba[va] = proba

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
    return pd.DataFrame(rows), oof_proba


#MAIN
def main():
    print("==> Base folder:", BASE_DIR)
    print("==> Reading:", PATH_MM.name, "|", PATH_META.name)

    mm = _safe_read_excel(PATH_MM)
    meta = _safe_read_excel(PATH_META)

    # Add participant_id_global from meta table
    if "sample_id" not in mm.columns or "participant_id_global" not in meta.columns:
        raise ValueError("Check Excel columns: expected 'sample_id' (MMASD) and 'participant_id_global' (meta).")

    mm = mm.merge(meta[["sample_id", "participant_id_global"]], on="sample_id", how="left")

    # fallback: if global is missing, use local participant_id (if available)
    if "participant_id" in mm.columns:
        mm["pid_group"] = mm["participant_id_global"].fillna(mm["participant_id"]).astype(str)
    else:
        mm["pid_group"] = mm["participant_id_global"].astype(str)

    # Feature list
    feat_cols = build_feature_list(mm)
    if len(feat_cols) < 3:
        raise ValueError(f"Too few numeric features ({len(feat_cols)}). Check the table.")

    # Drop rows where ALL features are NaN
    feat_all_na = mm[feat_cols].isna().all(axis=1)
    df = mm.loc[~feat_all_na].copy()

    # --- Targets
    # A: activity -> 3 categories
    df["activity_grouped"] = df["activity_class_basic"].apply(map_activity_to_3)
    df_act = df.dropna(subset=["activity_grouped", "pid_group"]).copy()

    # B: intensity (bin)
    if "movement_intensity_bin" not in df.columns:
        raise ValueError("MMASD_merged.xlsx is missing 'movement_intensity_bin'. Add the column as per methodology.")
    df_bin = df.dropna(subset=["movement_intensity_bin", "pid_group"]).copy()

    # Save cleaned table
    df.to_csv(OUT_CLEAN, index=False)
    print(f"[✓] Saved cleaned table: {OUT_CLEAN}")

    # Task A: Activity (3 classes)
    X_act = df_act[feat_cols].to_numpy(float)
    groups_act = df_act["pid_group"].astype(str).to_numpy()
    y_act_labels = df_act["activity_grouped"].astype(str).to_numpy()
    classes_act = sorted(pd.unique(y_act_labels))
    lab2idx = {lab: i for i, lab in enumerate(classes_act)}
    y_act_idx = np.array([lab2idx[v] for v in y_act_labels], dtype=int)

    # Task B: Movement intensity (bin)
    X_bin = df_bin[feat_cols].to_numpy(float)
    groups_bin = df_bin["pid_group"].astype(str).to_numpy()
    y_bin = df_bin["movement_intensity_bin"].astype(int).to_numpy()

    # CV runs
    # Activity
    metrics_act_rows = []
    for mdl in ["logreg", "xgb"]:
        mdf, oof_pred = cv_multiclass(X_act, y_act_idx, groups_act, mdl)
        y_hat = oof_pred.argmax(axis=1)
        metrics_act_rows.append({
            "task": "activity_grouped",
            "model": mdl,
            "n": int(len(y_hat)),
            "n_participants": int(pd.Series(groups_act).nunique()),
            "acc_mean_cv": float(mdf["acc"].mean()),
            "macro_f1_mean_cv": float(mdf["macro_f1"].mean()),
            "bal_acc_mean_cv": float(mdf["bal_acc"].mean()),
            "acc_oof": float(accuracy_score(y_act_idx, y_hat)),
            "macro_f1_oof": float(f1_score(y_act_idx, y_hat, average="macro")),
            "bal_acc_oof": float(balanced_accuracy_score(y_act_idx, y_hat)),
            "classes": "|".join(classes_act)
        })
    pd.DataFrame(metrics_act_rows).to_csv(OUT_METRICS_ACTIVITY, index=False)
    print(f"[✓] Saved metrics: {OUT_METRICS_ACTIVITY}")

    # Intensity
    metrics_bin_rows = []
    for mdl in ["logreg", "xgb"]:
        mdf, oof_proba = cv_binary(X_bin, y_bin, groups_bin, mdl)
        y_pred_05 = (oof_proba >= 0.5).astype(int)
        metrics_bin_rows.append({
            "task": "movement_intensity_bin",
            "model": mdl,
            "n": int(len(y_bin)),
            "n_participants": int(pd.Series(groups_bin).nunique()),
            "auroc_mean_cv": float(mdf["auroc"].mean()),
            "bal_acc_mean_cv": float(mdf["bal_acc"].mean()),
            "f1_pos_mean_cv": float(mdf["f1_pos"].mean()),
            "acc_mean_cv": float(mdf["acc"].mean()),
            "auroc_oof": float(safe_auroc(y_bin, oof_proba)),
            "bal_acc_oof_thr05": float(balanced_accuracy_score(y_bin, y_pred_05)),
            "f1_pos_oof_thr05": float(f1_score(y_bin, y_pred_05, pos_label=1)),
            "acc_oof_thr05": float(accuracy_score(y_bin, y_pred_05))
        })
    pd.DataFrame(metrics_bin_rows).to_csv(OUT_METRICS_INTENSITY, index=False)
    print(f"[✓] Saved metrics: {OUT_METRICS_INTENSITY}")

    print("\nDone.")
    print(f"Features used: {len(feat_cols)}")
    print(f"Participants (activity): {pd.Series(groups_act).nunique()} | (intensity): {pd.Series(groups_bin).nunique()}")


if __name__ == "__main__":
    main()
