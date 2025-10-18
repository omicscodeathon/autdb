# train_engagnition_enriched.py
# Engagnition-only, enriched features (acc_*, gsr_*, tmp_*)
# Tasks:
#   A) engagement_level (multiclass 0/1/2)
#   B) movement_intensity_bin (binary 0/1)
#
# Splits: GroupKFold by participant_id  -> no subject leakage
# Models: Logistic Regression, XGBoost  -> hyperparam tuning
# Outputs: outputs/tables/*.csv, outputs/figures/*.png

import os
import re
import argparse
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_TUNING_ITERS = 30
N_JOBS = 1
OUT_DIR = "outputs"

# xgboost
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception as e:
    XGB_OK = False
    _XGB_ERR = e


# Utils

def ensure_outdirs():
    os.makedirs(os.path.join(OUT_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)


def _filter_acc_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only ACC rows to avoid duplicate sessions by modality.
    If no '_ACC' rows exist, returns df as-is.
    """
    if "sample_id" in df.columns:
        m = df["sample_id"].astype(str).str.contains("_ACC", case=False, na=False)
        if m.any():
            return df.loc[m].copy()
    return df.copy()


def _select_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select ONLY real sensor features:
      - acc_*   (accelerometry)
      - gsr_*   (electrodermal activity)
      - tmp_*   (skin temperature)
    """
    feat_cols = [c for c in df.columns if re.match(r"^(acc_|gsr_|tmp_)", c, flags=re.I)]
    if not feat_cols:
        raise ValueError("No feature columns found matching acc_*, gsr_*, tmp_*")
    X = df[feat_cols].copy()
    # drop any movement_intensity_* leakage if present
    leak_cols = [c for c in X.columns if re.match(r"^movement_intensity_", c, flags=re.I)]
    if leak_cols:
        X = X.drop(columns=leak_cols, errors="ignore")
    return X


class CorrFilter(BaseEstimator, TransformerMixin):
    """Remove highly collinear features (simple greedy)."""
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


def build_logreg_pipeline(multiclass: bool) -> Pipeline:
    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=800,
        class_weight="balanced",
        multi_class="multinomial" if multiclass else "auto",
        random_state=RANDOM_STATE
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("corr", CorrFilter(thr=0.98)),
        ("clf", clf)
    ])


def build_xgb_classifier(num_class: int = None, pos_weight: float = 1.0) -> "XGBClassifier":
    if not XGB_OK:
        raise RuntimeError(f"xgboost not available: {_XGB_ERR}\nInstall: py -m pip install xgboost")
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

# Metrics

def compute_binary_metrics(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_proba >= thr).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "bacc":  balanced_accuracy_score(y_true, y_pred),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
        "acc":   accuracy_score(y_true, y_pred),
    }


def best_threshold_for_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
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
        auc_ovr = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
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
    """
    Returns: metrics (OOF), oof_proba, best_threshold_for_f1
    """
    outer = GroupKFold(n_splits=N_OUTER_SPLITS)
    oof_proba = np.zeros_like(y, dtype=float)

    if model_name == "logreg":
        base = build_logreg_pipeline(multiclass=False)
        param_distributions = {"clf__C": np.logspace(-2, 2, 12)}
        scoring = "roc_auc"
    elif model_name == "xgb":
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

    for tr_idx, va_idx in outer.split(X, y, groups):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        gtr      = groups[tr_idx]

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

    thr = best_threshold_for_f1(y, oof_proba)
    metrics = compute_binary_metrics(y, oof_proba, thr)
    return metrics, oof_proba, thr


def run_multiclass_groupcv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Returns: metrics (OOF), oof_proba shape (n_samples, n_classes)
    """
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
        gtr      = groups[tr_idx]

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

# Plots

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
    parser = argparse.ArgumentParser(description="Engagnition enriched features — intra-dataset CV")
    parser.add_argument("--data", type=str, default="Engagnition_features_enriched.xlsx",
                        help="Path to Engagnition enriched Excel file")
    args = parser.parse_args()

    ensure_outdirs()

    try:
        df = pd.read_excel(args.data, sheet_name=0, engine="openpyxl")
    except Exception as e:
        raise SystemExit(f"Failed to read {args.data}: {e}")

    # Basic checks
    needed = {"sample_id", "participant_id", "engagement_level"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Keep Engagnition only (if present)
    if "dataset" in df.columns:
        df = df.loc[df["dataset"].astype(str).str.lower() == "engagnition"].copy()

    # Use only ACC rows to avoid duplicate modalities per session
    df = _filter_acc_rows(df)

    # Drop demographics (strictly excluded)
    for col in ["sex", "age_years", "age_group"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Build feature matrix (acc_*, gsr_*, tmp_*)
    X_all = _select_feature_matrix(df)
    # Groups
    groups_all = df["participant_id"].astype(str).values

    # Task A: engagement_level (multiclass)
    y_eng = df["engagement_level"].astype(int).values
    # No leakage: make sure movement_intensity* columns are not in features (already dropped in selector)

    results_eng = []
    for model in ["logreg", "xgb"]:
        print(f"[Task A] Multiclass CV ({model}) on engagement_level...")
        met, oof_proba = run_multiclass_groupcv(
            X_all.values.astype(float), y_eng, groups_all, model_name=model
        )
        results_eng.append({"model": model, **met})
        plot_confusion_multiclass(
            y_true=y_eng, proba=oof_proba,
            title=f"Engagement (OOF) — {model}",
            path_png=os.path.join(OUT_DIR, "figures", f"engagement_confmat_{model}.png")
        )

    pd.DataFrame(results_eng).to_csv(
        os.path.join(OUT_DIR, "tables", "metrics_engagement_enriched.csv"), index=False
    )

    # Task B: movement_intensity_bin (binary)
    if "movement_intensity_bin" not in df.columns:
        print("[Task B] movement_intensity_bin not found — skipping binary task.")
        print("[✓] Done.")
        return

    mask_bin = df["movement_intensity_bin"].notna()
    if mask_bin.sum() < 10:
        print(f"[Task B] Too few rows with movement_intensity_bin (n={mask_bin.sum()}) — skipping.")
        print("[✓] Done.")
        return

    df_bin = df.loc[mask_bin].copy()
    X_bin = X_all.loc[mask_bin].copy()

    # Extra safety: do not include engagement label in features
    for c in ["engagement_level"]:
        if c in X_bin.columns:
            X_bin = X_bin.drop(columns=[c])

    y_bin = df_bin["movement_intensity_bin"].astype(int).values
    groups_bin = df_bin["participant_id"].astype(str).values

    results_bin = []
    for model in ["logreg", "xgb"]:
        print(f"[Task B] Binary CV ({model}) on movement_intensity_bin...")
        met, oof_proba, thr = run_binary_groupcv(
            X_bin.values.astype(float), y_bin, groups_bin, model_name=model
        )
        results_bin.append({"model": model, "thr_f1": thr, **met})
        plot_roc_binary(
            y=y_bin, proba=oof_proba,
            title=f"Intensity ROC (OOF) — {model}",
            path_png=os.path.join(OUT_DIR, "figures", f"intensity_roc_{model}.png")
        )

    pd.DataFrame(results_bin).to_csv(
        os.path.join(OUT_DIR, "tables", "metrics_intensity_enriched.csv"), index=False
    )

    print("\n[✓] Done. See outputs/tables/*.csv and outputs/figures/*.png")


if __name__ == "__main__":
    main()
