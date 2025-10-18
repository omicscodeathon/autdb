# train_lodo_intensity_scaled.py
# LODO for movement_intensity_bin with configurable Robust scaling
# — готовые фолды из Excel (participant_id_global; cv_fold_lodo/iid), корректный CV и метрики

import os, argparse, warnings, json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, accuracy_score, roc_curve, confusion_matrix
)
from sklearn.base import clone

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_JOBS = 1
N_INNER_SPLITS = 5
N_TUNING_ITERS = 30
OUT_DIR = "outputs"

try:
    from xgboost import XGBClassifier
    XGB_AVAIL = True
except Exception as e:
    XGB_AVAIL = False
    _XGB_ERR = e


# ----------------------------- IO ---------------------------------

def ensure_outdirs():
    os.makedirs(os.path.join(OUT_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)


def detect_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def load_unified_table(path_csv: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Требуемые поля: dataset, sample_id, movement_intensity_bin, + локальный ID участника
    (любой из: group_id / participant_id / subject_id / user_id / pid_group).
    Фича-колонки должны начинаться с 'feat_'.
    """
    df = pd.read_csv(path_csv)
    gcol = detect_col(df, ["group_id","participant_id","subject_id","user_id","pid_group"])
    need = {"dataset", "sample_id", "movement_intensity_bin"}
    miss = need - set(df.columns)
    if gcol is None:
        raise ValueError("CSV must contain one of: group_id/participant_id/subject_id/user_id/pid_group")
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")

    if gcol != "group_id":
        df = df.rename(columns={gcol: "group_id"})

    feats = [c for c in df.columns if c.startswith("feat_")]
    if not feats:
        raise ValueError("No feat_* columns found in CSV.")

    df = df.copy()
    for c in ["dataset", "group_id", "sample_id"]:
        df[c] = df[c].astype(str)

    # строго двоичная метка 0/1
    y = pd.to_numeric(df["movement_intensity_bin"], errors="coerce")
    df["movement_intensity_bin"] = (y > 0).astype(int)
    df = df.dropna(subset=["movement_intensity_bin"]).reset_index(drop=True)

    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    mask_all_nan = df[feats].isna().all(axis=1)
    df = df.loc[~mask_all_nan].reset_index(drop=True)
    return df, feats


def load_meta(path: str) -> pd.DataFrame:
    """
    Excel с маппингом локального участника -> global_id.
    Ожидаемые варианты: dataset; (participant_id|group_id|...); (participant_id_global|global_id|...).
    """
    m = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    dcol = detect_col(m, ["dataset", "Dataset", "DATASET"])
    gcol = detect_col(m, ["participant_id_global", "global_id", "GlobalID", "pid_global"])
    lcol = detect_col(m, ["participant_id","group_id","participant","pid_group","subject_id","user_id","group"])
    if any(c is None for c in [dcol, gcol, lcol]):
        raise ValueError("Meta Excel must contain dataset + local participant id + participant_id_global.")
    m = m[[dcol, lcol, gcol]].copy()
    m.columns = ["dataset", "local_id", "global_id"]
    for c in ["dataset", "local_id", "global_id"]:
        m[c] = m[c].astype(str)
    m = m.dropna().drop_duplicates(["dataset", "local_id"])
    return m


# ------------------------ Robust scaling --------------------------

def robust_params(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    med = df.median(skipna=True)
    iqr = df.quantile(0.75) - df.quantile(0.25)
    iqr = iqr.replace(0, 1e-8).fillna(1.0)
    return med, iqr


def robust_scale(df: pd.DataFrame, med: pd.Series, iqr: pd.Series) -> pd.DataFrame:
    return (df - med) / iqr


def prepare_scaled(df_all: pd.DataFrame, feats: List[str],
                   train_ds: str, test_ds: str, mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    tr = df_all.loc[df_all["dataset"].str.lower() == train_ds.lower()].copy()
    te = df_all.loc[df_all["dataset"].str.lower() == test_ds.lower()].copy()
    if tr.empty or te.empty:
        raise ValueError("Empty train/test split by dataset.")
    Xtr = tr[feats].copy()
    Xte = te[feats].copy()
    info = {"scaling_mode": mode}

    if mode == "train_only":
        med, iqr = robust_params(Xtr)
        Xtr = robust_scale(Xtr, med, iqr)
        Xte = robust_scale(Xte, med, iqr)
        info.update({"stats": "train_only"})
    elif mode == "per_dataset":
        med_tr, iqr_tr = robust_params(Xtr)
        med_te, iqr_te = robust_params(Xte)
        Xtr = robust_scale(Xtr, med_tr, iqr_tr)
        Xte = robust_scale(Xte, med_te, iqr_te)  # DIAGNOSTIC — использует target
        info.update({"stats": "per_dataset"})
        warnings.warn("per_dataset scaling uses target/test statistics — diagnostic only, not deployable.")
    elif mode == "global":
        both = pd.concat([Xtr, Xte], axis=0)
        med, iqr = robust_params(both)           # DIAGNOSTIC — использует target
        Xtr = robust_scale(Xtr, med, iqr)
        Xte = robust_scale(Xte, med, iqr)
        info.update({"stats": "global"})
        warnings.warn("global scaling uses target/test statistics — diagnostic only, not deployable.")
    else:
        raise ValueError("scaling_mode must be one of: train_only, per_dataset, global")

    tr[feats] = Xtr
    te[feats] = Xte
    return tr, te, info


# ------------------------ Models & tuning -------------------------

def build_logreg_pipeline() -> Pipeline:
    clf = LogisticRegression(
        solver="lbfgs", penalty="l2", max_iter=2000,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", clf)])


def build_xgb(pos_weight: float) -> "XGBClassifier":
    if not XGB_AVAIL:
        raise RuntimeError(f"xgboost not available: {_XGB_ERR}")
    return XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        n_estimators=600, learning_rate=0.05, max_depth=4, min_child_weight=3,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        scale_pos_weight=pos_weight, n_jobs=N_JOBS, random_state=RANDOM_STATE, verbosity=0
    )


def proba_pos(est, X: np.ndarray) -> np.ndarray:
    """Гарантированно возвращает вероятность класса 1 (учитывает порядок classes_)."""
    classes = np.asarray(getattr(est, "classes_", [0, 1]))
    try:
        idx = int(np.where(classes == 1)[0][0])
    except Exception:
        idx = 1
    return est.predict_proba(X)[:, idx]


def best_thr_f1(y: np.ndarray, proba: np.ndarray) -> float:
    qs = np.linspace(0.01, 0.99, 199)
    thrs = np.unique(np.quantile(proba, qs))
    if thrs.size == 0:
        return 0.5
    best_t = 0.5
    best_f = -1.0
    for t in thrs:
        f = f1_score(y, (proba >= t).astype(int), pos_label=1)
        if f > best_f:
            best_f = float(f)
            best_t = float(t)
    return best_t


def metrics_bin(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    yhat = (proba >= thr).astype(int)
    auc  = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan
    ap   = average_precision_score(y, proba) if len(np.unique(y)) > 1 else np.nan
    cm   = confusion_matrix(y, yhat, labels=[0,1])
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    return {
        "auroc": auc, "prauc": ap,
        "bacc": balanced_accuracy_score(y, yhat),
        "f1_pos": f1_score(y, yhat, pos_label=1),
        "acc": accuracy_score(y, yhat),
        "thr": float(thr), "prev": float(np.mean(y == 1)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def load_predefined_splits(path_excel: str, train_dataset: str, tr_df: pd.DataFrame):
    """
    Ждём в Excel: dataset, participant_id_global и cv_fold_lodo (или cv_fold_iid).
    """
    meta = pd.read_excel(path_excel, sheet_name=0, engine="openpyxl")

    dcol = detect_col(meta, ["dataset", "Dataset", "DATASET"])
    gcol = detect_col(meta, ["participant_id_global","global_id","GlobalID"])
    scol = detect_col(meta, ["cv_fold_lodo","cv_fold_iid","split","fold","cv"])

    if any(c is None for c in [dcol, gcol, scol]):
        raise ValueError(f"Splits Excel must have dataset + participant_id_global + cv_fold_*; found: {list(meta.columns)}")

    meta = meta[[dcol, gcol, scol]].copy()
    meta.columns = ["dataset", "global_id", "split"]
    meta["dataset"] = meta["dataset"].astype(str).str.lower()
    meta["global_id"] = meta["global_id"].astype(str)

    ds = str(train_dataset).lower()
    meta = meta.loc[meta["dataset"] == ds].copy()
    if meta.empty:
        raise ValueError(f"No splits for dataset='{train_dataset}' in {path_excel}")

    if "global_id" not in tr_df.columns:
        raise ValueError("Train DataFrame must contain 'global_id' (after merge).")
    tr_gids = tr_df["global_id"].astype(str).to_numpy()

    folds = []
    for s in sorted(meta["split"].dropna().unique()):
        val_ids = set(meta.loc[meta["split"] == s, "global_id"])
        val_idx = np.array([i for i, g in enumerate(tr_gids) if g in val_ids], dtype=int)
        train_idx = np.array([i for i in range(len(tr_gids)) if i not in val_idx], dtype=int)
        if len(val_idx) == 0 or len(train_idx) == 0:
            warnings.warn(f"Empty fold '{s}' — skipping.")
            continue
        folds.append((train_idx, val_idx))
    if not folds:
        raise ValueError("All folds empty after mapping by global_id.")
    return folds


def tune_fit_with_cv(X_tr: np.ndarray, y_tr: np.ndarray,
                     model_name: str,
                     cv_splits: List[Tuple[np.ndarray, np.ndarray]],
                     pos_weight_from_y: bool = True):
    if model_name == "logreg":
        base = build_logreg_pipeline()
        space = {"clf__C": np.logspace(-2, 2, 12)}
    elif model_name == "xgb":
        if not XGB_AVAIL:
            raise RuntimeError(f"xgboost not available: {_XGB_ERR}")
        if pos_weight_from_y:
            pos = max(1, int((y_tr == 1).sum()))
            neg = max(1, int((y_tr == 0).sum()))
            base = build_xgb(pos_weight=float(neg) / float(pos))
        else:
            base = build_xgb(pos_weight=1.0)
        space = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.75, 0.9],
            "min_child_weight": [3, 5, 10],
            "reg_lambda": [1.0, 5.0, 10.0],
            "n_estimators": [300, 600, 900],
        }
    else:
        raise ValueError("model_name must be 'logreg' or 'xgb'")

    tuner = RandomizedSearchCV(
        base, space, n_iter=N_TUNING_ITERS, scoring="roc_auc",
        cv=cv_splits, n_jobs=N_JOBS, random_state=RANDOM_STATE, refit=True, verbose=0
    )
    tuner.fit(X_tr, y_tr)
    best = tuner.best_estimator_

    # OOF на тех же сплитах (клоны — без утечки)
    oof = np.zeros_like(y_tr, dtype=float)
    for tr_idx, va_idx in cv_splits:
        est = clone(best)
        est.fit(X_tr[tr_idx], y_tr[tr_idx])
        proba_v = proba_pos(est, X_tr[va_idx])
        oof[va_idx] = proba_v

    thr = best_thr_f1(y_tr, oof)
    return best, oof, thr, tuner.best_params_


def plot_roc(y: np.ndarray, proba: np.ndarray, title: str, path: str):
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
    plt.savefig(path, dpi=160)
    plt.close()


# ----------------------------- RUN --------------------------------

def run_once(df, feats, meta_map, splits_excel_path: Optional[str],
             train_ds, test_ds, model, scaling_mode):
    # масштабирование по режиму
    tr, te, scale_info = prepare_scaled(df, feats, train_ds, test_ds, scaling_mode)

    # добавляем global_id на TRAIN (для маппинга фолдов)
    tr = tr.merge(meta_map, how="left", left_on=["dataset", "group_id"], right_on=["dataset", "local_id"])
    if "global_id" not in tr.columns:
        raise RuntimeError("merge with meta failed: no global_id")
    missing_gid = tr["global_id"].isna().sum()
    if missing_gid > 0:
        warnings.warn(f"{missing_gid} rows in TRAIN have no global_id; fallback to local group_id for them.")
    tr["global_id"] = tr["global_id"].fillna(tr["group_id"]).astype(str)

    # numpy массивы
    Xtr = tr[feats].to_numpy(float)
    ytr = tr["movement_intensity_bin"].astype(int).to_numpy()
    Xte = te[feats].to_numpy(float)
    yte = te["movement_intensity_bin"].astype(int).to_numpy()

    # готовые фолды или GroupKFold (fallback)
    if splits_excel_path:
        cv_splits = load_predefined_splits(splits_excel_path, train_ds, tr)
    else:
        gkf = GroupKFold(n_splits=N_INNER_SPLITS)
        cv_splits = list(gkf.split(Xtr, ytr, groups=tr["global_id"].to_numpy()))
        warnings.warn("Using GroupKFold fallback (no --splits_excel provided).")

    # тюнинг и OOF
    est, oof, thr, best_params = tune_fit_with_cv(Xtr, ytr, model, cv_splits)

    # тест
    pte = proba_pos(est, Xte)

    # диагностика инверсии
    if len(np.unique(yte)) > 1:
        auc_pos = roc_auc_score(yte, pte)
        auc_neg = roc_auc_score(1 - yte, pte)
        if auc_pos < 0.5 and auc_neg > 0.5:
            warnings.warn(f"Test AUC seems inverted ({auc_pos:.3f}); check label orientation in data prep.")

    m = metrics_bin(yte, pte, thr)
    m.update({
        "model": model,
        "train_dataset": train_ds,
        "test_dataset": test_ds,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "scaling_mode": scaling_mode,
        "best_params": json.dumps(best_params),
    })

    # фигура ROC
    plot_roc(
        yte, pte,
        f"LODO ROC — {train_ds}→{test_ds} — {model} [{scaling_mode}]",
        os.path.join(OUT_DIR, "figures", f"roc_{train_ds}2{test_ds}_{model}_{scaling_mode}.png")
    )

    # лог
    with open(os.path.join(OUT_DIR, "logs",
                           f"log_{train_ds}2{test_ds}_{model}_{scaling_mode}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_dataset": train_ds, "test_dataset": test_ds, "model": model,
            "scaling": scale_info, "best_params": best_params,
            "thr_from_train": float(m["thr"]), "train_prevalence": float(np.mean(ytr == 1)),
            "oof_f1_at_thr": float(f1_score(ytr, (oof >= m["thr"]).astype(int))),
        }, f, ensure_ascii=False, indent=2)

    return m


def main():
    parser = argparse.ArgumentParser(description="LODO movement_intensity_bin with Robust scaling modes")
    parser.add_argument("--table", required=True, help="CSV unified table (features start with 'feat_')")
    parser.add_argument("--meta", required=True, help="Excel mapping dataset/local_id -> participant_id_global")
    parser.add_argument("--scaling_mode", choices=["train_only", "per_dataset", "global"], default="train_only")
    parser.add_argument("--splits_excel", default=None,
                        help="Excel with predefined CV folds by dataset/participant_id_global (cv_fold_lodo/iid)")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    ensure_outdirs()

    df, feats = load_unified_table(args.table)
    meta = load_meta(args.meta)

    results = []
    for mdl in ["logreg", "xgb"] if XGB_AVAIL else ["logreg"]:
        results.append(run_once(df, feats, meta, args.splits_excel, "MMASD", "Engagnition", mdl, args.scaling_mode))
        results.append(run_once(df, feats, meta, args.splits_excel, "Engagnition", "MMASD", mdl, args.scaling_mode))

    out = os.path.join(OUT_DIR, "tables", f"metrics_lodo_intensity_{args.scaling_mode}.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"[✓] Saved: {out}")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
