# train_engagnition.py

# Intra-dataset CV for Engagnition with two targets:
#  A) engagement_level (multiclass, 0/1/2)
#  B) movement_intensity_bin (binary, 0/1)
#
# Features: ONLY ACC-derived columns (acc_*)
# Splits:   GroupKFold by participant_id (no subject leakage)
# Models:   Logistic Regression, XGBoost (+ hyperparameter tuning)
# Outputs:  outputs/tables/*.csv, outputs/figures/*.png


import os
import re
import argparse
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, clone
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Utilities / config

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_TUNING_ITERS = 30
N_JOBS = 1  # keep 1 for reproducibility (increase if you like)
OUT_DIR = "outputs"

# XGBoost import
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception as e:
    XGB_OK = False
    _XGB_IMPORT_ERROR = e


def ensure_outdirs():
    os.makedirs(os.path.join(OUT_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)


def _safe_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[num_cols]


def _select_acc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only ACC features (columns starting with 'acc_')."""
    cols = [c for c in df.columns if c.lower().startswith("acc_")]
    if not cols:
        raise ValueError("No ACC features found (columns starting with 'acc_').")
    return df[cols].copy()


def _filter_acc_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows corresponding to ACC modality to avoid duplicates from GSR/TMP/GAZE etc.
    Heuristic: sample_id contains '_ACC' (case-insensitive). If nothing matches, return as-is.
    """
    if "sample_id" in df.columns:
        mask = df["sample_id"].astype(str).str.contains("_ACC", case=False, na=False)
        if mask.any():
            return df.loc[mask].copy()
    return df.copy()


# Correlation filter

class CorrFilter(BaseEstimator, TransformerMixin):
    def __init__(self, thr: float = 0.98):
        self.thr = thr

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] <= 1:
            self.keep_ = np.ones(X.shape[1], dtype=bool)
            return self
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
        X = np.asarray(X, float)
        return X[:, self.keep_]


# Model builders

def build_logreg_pipeline(multiclass: bool) -> Pipeline:
    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=800,
        class_weight="balanced",
        multi_class="multinomial" if multiclass else "auto",
        random_state=RANDOM_STATE
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("corr", CorrFilter(thr=0.98)),
        ("clf", clf)
    ])
    return pipe


def build_xgb_classifier(num_class: int = None, pos_weight: float = 1.0) -> XGBClassifier:
    if not XGB_OK:
        raise RuntimeError(
            f"XGBoost import failed ({_XGB_IMPORT_ERROR}). "
            "Install it with:  py -m pip install xgboost"
        )
    if num_class is None:
        # binary
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=N_JOBS,
            verbosity=0,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE
        )
    else:
        # multiclass
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_class,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=N_JOBS,
            verbosity=0,
            random_state=RANDOM_STATE
        )


# Metrics helpers

def compute_binary_metrics(y_true: np.ndarray, y_proba: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_proba >= thr).astype(int)
    out = {
        "auroc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "bacc":  balanced_accuracy_score(y_true, y_pred),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
        "acc":   accuracy_score(y_true, y_pred),
    }
    return out


def best_threshold_for_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    # search thresholds on quantiles of predicted proba
    qs = np.linspace(0.01, 0.99, 199)
    thr_cands = np.unique(np.quantile(y_proba, qs))
    best_thr, best_f1 = 0.5, -1.0
    for t in thr_cands:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), pos_label=1)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr)


def compute_multiclass_metrics(y_true: np.ndarray, proba: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    y_pred = proba.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc_ovr = roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=labels
        )
    except Exception:
        auc_ovr = np.nan
    return {"macro_f1": macro_f1, "bal_acc": bacc, "auc_ovr_macro": auc_ovr}


# CV runners

def run_binary_groupcv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str
) -> Tuple[Dict[str, float], np.ndarray, float]:
    outer = GroupKFold(n_splits=N_OUTER_SPLITS)
    oof_proba = np.zeros_like(y, dtype=float)

    # Choose base estimator + param distributions
    if model_name == "logreg":
        base = build_logreg_pipeline(multiclass=False)
        param_distributions = {"clf__C": np.logspace(-2, 2, 12)}
        scoring = "roc_auc"
    elif model_name == "xgb":
        # scale_pos_weight per inner fold; start with rough global value
        pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
        base = build_xgb_classifier(num_class=None, pos_weight=pos_weight)
        param_distributions = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.75, 0.9],
            "min_child_weight": [5, 10, 20],
            "reg_lambda": [1.0, 5.0, 10.0],
            "n_estimators": [300, 500, 800],
            "gamma": [0, 0.5, 1.0]
        }
        scoring = "roc_auc"
    else:
        raise ValueError("model_name must be 'logreg' or 'xgb'.")

    # Outer CV with inner tuning
    for tr_idx, va_idx in outer.split(X, y, groups):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        gtr = groups[tr_idx]

        inner = GroupKFold(n_splits=3)
        tuner = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_distributions,
            n_iter=N_TUNING_ITERS,
            scoring=scoring,
            cv=inner.split(Xtr, ytr, gtr),
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0
        )
        tuner.fit(Xtr, ytr)
        best_est = tuner.best_estimator_
        proba_va = best_est.predict_proba(Xva)[:, 1]
        oof_proba[va_idx] = proba_va

    # Threshold tuned on OOF for F1
    thr = best_threshold_for_f1(y, oof_proba)
    metrics = compute_binary_metrics(y, oof_proba, thr)
    return metrics, oof_proba, thr


def run_multiclass_groupcv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str
) -> Tuple[Dict[str, float], np.ndarray]:
    classes = np.sort(np.unique(y))
    n_classes = len(classes)

    outer = GroupKFold(n_splits=N_OUTER_SPLITS)

    if model_name == "logreg":
        base = build_logreg_pipeline(multiclass=True)
        param_distributions = {"clf__C": np.logspace(-2, 2, 12)}
        scoring = "f1_macro"
    elif model_name == "xgb":
        base = build_xgb_classifier(num_class=n_classes)
        param_distributions = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.75, 0.9],
            "min_child_weight": [5, 10, 20],
            "reg_lambda": [1.0, 5.0, 10.0],
            "n_estimators": [300, 500, 800],
            "gamma": [0, 0.5, 1.0]
        }
        scoring = "f1_macro"
    else:
        raise ValueError("model_name must be 'logreg' or 'xgb'.")

    oof_proba = np.zeros((len(y), n_classes), dtype=float)

    for tr_idx, va_idx in outer.split(X, y, groups):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        gtr = groups[tr_idx]

        inner = GroupKFold(n_splits=3)
        tuner = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_distributions,
            n_iter=N_TUNING_ITERS,
            scoring=scoring,
            cv=inner.split(Xtr, ytr, gtr),
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0
        )
        tuner.fit(Xtr, ytr)
        best_est = tuner.best_estimator_
        proba_va = best_est.predict_proba(Xva)
        oof_proba[va_idx, :] = proba_va

    metrics = compute_multiclass_metrics(y, oof_proba, classes)
    return metrics, oof_proba

# Plot helpers

def plot_roc_binary(y: np.ndarray, proba: np.ndarray, title: str, path_png: str):
    if len(np.unique(y)) < 2:
        return
    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure(figsize=(4.8, 4.0))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()


def plot_confusion_multiclass(y_true: np.ndarray, proba: np.ndarray, title: str, path_png: str):
    y_pred = proba.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.sort(np.unique(y_true)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(4.8, 4.0))
    disp.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()

# Main

def main():
    parser = argparse.ArgumentParser(description="Engagnition-only intra-dataset CV")
    parser.add_argument("--data", type=str, default="Engagnition_features_with_engagement.xlsx",
                        help="Path to Engagnition Excel file")
    args = parser.parse_args()

    ensure_outdirs()

    # Load data
    try:
        df = pd.read_excel(args.data, sheet_name=0, engine="openpyxl")
    except Exception as e:
        raise SystemExit(f"Failed to read {args.data}: {e}")

    # Basic checks
    required_cols = {"sample_id", "participant_id", "engagement_level"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in Excel: {missing}")

    # Keep Engagnition only (should already be)
    if "dataset" in df.columns:
        df = df.loc[df["dataset"].astype(str).str.lower() == "engagnition"].copy()

    # Use only ACC rows to avoid duplicated sessions from other modalities
    df = _filter_acc_rows(df)

    # Drop demographics entirely (per requirement)
    for col in ["sex", "age_years", "age_group"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Features (only acc_*)

    num_df = _safe_numeric_cols(df)  # numeric only
    X_acc = _select_acc_features(num_df)  # only acc_*

    # Groups
    groups = df["participant_id"].astype(str).values

    # Task A: engagement_level (multiclass)
    y_eng = df["engagement_level"].astype(int).values
    # Ensure we do NOT use intensity labels as features in this task
    drop_cols_eng = [c for c in X_acc.columns if re.match(r"^movement_intensity_", c, flags=re.I)]
    X_task_eng = X_acc.drop(columns=drop_cols_eng, errors="ignore").values.astype(float)

    results_eng = []
    for model in ["logreg", "xgb"]:
        print(f"[Task A] Running multiclass CV ({model}) on engagement_level...")
        met, oof_proba = run_multiclass_groupcv(X_task_eng, y_eng, groups, model_name=model)
        results_eng.append({"model": model, **met})
        # plots
        plot_confusion_multiclass(
            y_true=y_eng, proba=oof_proba,
            title=f"Engagement (OOF) — {model}",
            path_png=os.path.join(OUT_DIR, "figures", f"engagement_confmat_{model}.png")
        )

    pd.DataFrame(results_eng).to_csv(
        os.path.join(OUT_DIR, "tables", "metrics_engagement.csv"), index=False
    )

    # Task B: movement_intensity_bin
    if "movement_intensity_bin" not in df.columns:
        print("[Task B] movement_intensity_bin not found — skipping binary task.")
        return

    mask_bin = df["movement_intensity_bin"].notna()
    if mask_bin.sum() < 10:
        print(f"[Task B] Too few rows with movement_intensity_bin (n={mask_bin.sum()}) — skipping binary task.")
        return

    df_bin = df.loc[mask_bin].copy()
    X_bin_all = X_acc.loc[mask_bin].copy()

    # Ensure we do NOT use engagement label as feature in this task
    if "engagement_level" in X_bin_all.columns:
        X_bin_all = X_bin_all.drop(columns=["engagement_level"], errors="ignore")

    # Also exclude movement_intensity_* as features except the bin target itself
    feat_drop = [c for c in X_bin_all.columns if re.match(r"^movement_intensity_", c, flags=re.I)]
    X_bin_all = X_bin_all.drop(columns=feat_drop, errors="ignore")

    y_bin = df_bin["movement_intensity_bin"].astype(int).values
    groups_bin = df_bin["participant_id"].astype(str).values
    X_task_bin = X_bin_all.values.astype(float)

    results_bin = []
    for model in ["logreg", "xgb"]:
        print(f"[Task B] Running binary CV ({model}) on movement_intensity_bin...")
        met, oof_proba, thr = run_binary_groupcv(X_task_bin, y_bin, groups_bin, model_name=model)
        results_bin.append({"model": model, "thr_f1": thr, **met})
        # plots
        plot_roc_binary(
            y=y_bin, proba=oof_proba,
            title=f"Intensity ROC (OOF) — {model}",
            path_png=os.path.join(OUT_DIR, "figures", f"intensity_roc_{model}.png")
        )

    pd.DataFrame(results_bin).to_csv(
        os.path.join(OUT_DIR, "tables", "metrics_intensity.csv"), index=False
    )

    print("\n[✓] Done. See outputs/tables/*.csv and outputs/figures/*.png")


if __name__ == "__main__":
    main()
