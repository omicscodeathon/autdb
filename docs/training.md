# Training & Evaluation

This document specifies how to train and evaluate baseline models under **IID** and **LODO** setups using the frozen, split-ready tables.

> Scope: **Python-only**. Tested on Python **3.10**.  
> Inputs: unified tables and splits from a versioned snapshot under `data/frozen/vX_YYYY-MM-DD/`.
>
> See also: `docs/preparation.md` (how to prepare features and the frozen snapshot).

---

## 1) Inputs & splits

- **Frozen snapshot** (example): `data/frozen/v1_2025-09-13/` containing:
  - `metadata_ml_ready_splits_withGlobalID.xlsx` — unified table with per-sample rows and `participant_id_global`.
  - `splits_manifest.json` — canonical participant-level splits for **IID** and **LODO** (`train/val/test` lists).
  - `schema_withGlobalID.yaml` — column definitions and dtypes.

- **Grouping:** All subject-aware CV uses `GroupKFold(groups=participant_id_global)` to avoid leakage across the same child.

---

## 2) Experimental setups

### 2.1 IID (within-dataset)
- Train/validate/test **within a dataset** using `GroupKFold` (groups = `participant_id_global`).
- Report mean ± std across folds (and repeats if used).

### 2.2 LODO (leave-one-dataset-out)
- Train on one dataset and **evaluate on the other**, using only columns shared between datasets.
- Use the **frozen splits** for the held-out dataset when available.
- **Scaling:** use `scaling_mode=train_only` (fit scalers on *training* folds only).  
  Diagnostic modes `per_dataset` and `global` are for analysis only (they use test statistics → not deployable).

---

## 3) Targets & features (overview)

- **Binary movement intensity**: computed from robust, per-participant normalized intensity scores (`z`-score); label = `1 if z >= 0 else 0` (balanced per participant).  
- **Multiclass tasks** (dataset-dependent): e.g., engagement levels or activity classes where available.
- Feature tables come from the preparation stage (MMASD 2D-skeleton, Engagnition ACC, optional enrichment with GSR/TMP).

> Exact column names and dtypes are defined in the frozen `schema_withGlobalID.yaml`.

---

## 4) Models & preprocessing

### 4.1 Baseline pipelines
- **Logistic Regression** with robust scaling:
  - `RobustScaler(with_centering=True, quantile_range=(25.0, 75.0))`
  - `LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", n_jobs=-1)`
- **XGBoost** (optional):
  - Typical flags: `n_estimators=400`, `max_depth=3..6`, `learning_rate=0.05..0.2`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`
  - Handle imbalance via `scale_pos_weight` (≈ `n_negative / n_positive` in the *training* data).

> Use **probability calibration** (e.g., `CalibratedClassifierCV` with `cv=GroupKFold(...)`) if well-calibrated probabilities are required.

### 4.2 Hyperparameters (minimal grid suggestion)
- Logistic Regression: `C ∈ {0.1, 1.0, 10.0}`.
- XGBoost: `n_estimators ∈ {200, 400}`, `max_depth ∈ {3, 5}`, `learning_rate ∈ {0.05, 0.1}`.  
Tune on **train/val folds only** (respecting groups).

### 4.3 Reproducibility
- Fix `random_state`/`seed` across CV, models and any sampling steps.
- Save the chosen params and seed values alongside metrics.

---

## 5) Metrics & reporting

- **Primary:** AUROC.  
- **Secondary:** Balanced Accuracy, macro-F1, Accuracy; optionally Precision/Recall and confusion matrices.  
- Always report mean ± std across folds (and repeats if used).
- For IID, ensure **grouped** evaluation per fold. For LODO, report metrics on the **held-out dataset**; if multiple target classes, report per-class metrics and macro-averages.

---

## 6) CLI recipes (baseline scripts)

Run in an environment created from `requirements.txt`. Examples:

### 6.1 General baselines (IID + cross-dataset)
```bash
py -3.10 "data/Baseline/Experiment - 1 (Baseline)/train_mi_baselines.py"
```
- Uses Logistic Regression (always) and **XGBoost if available**.  
- Saves tables under `outputs/tables/` and plots under `outputs/figures/`.

### 6.2 MMASD-only dual tasks
```bash
py -3.10 "data/Baseline/Experiment - 2 (MMASD)/mmasd_dual_task.py"
```

### 6.3 Engagnition (ACC-only) & enriched features
```bash
py -3.10 "data/Baseline/Experiment - 3 (Engagnition)/train_engagnition.py" ^
  --data "data\Prepared data with features\Engagnition_features_with_engagement.xlsx"

# optional enrichment and training on enriched table
py -3.10 "data/Baseline/Experiment - 3 (Engagnition)/make_eng_feature_enrichment.py" ^
  --input  "data\Prepared data with features\Engagnition_features_with_engagement.xlsx" ^
  --output "data\Baseline\Experiment - 3 (Engagnition)\Engagnition_features_enriched.xlsx"

py -3.10 "data/Baseline/Experiment - 3 (Engagnition)/train_engagnition_enriched.py" ^
  --data "data\Baseline\Experiment - 3 (Engagnition)\Engagnition_features_enriched.xlsx"
```

### 6.4 LODO intensity (scaling modes)
```bash
cd "data/Baseline/Experiment - 4 (LODO)"

# Build unified feature table for LODO
py -3.10 build_lodo_intensity_table.py ^
  --mmasd "..\..\..\data\Prepared data with IDglobal\mmasd_cleaned.csv" ^
  --eng   "..\Experiment - 3 (Engagnition)\Engagnition_features_enriched.xlsx" ^
  --out   "lodo_intensity_features.csv"

# Train with leakage-safe scaling (recommended)
py -3.10 train_lodo_intensity_scaled.py ^
  --table "lodo_intensity_features.csv" ^
  --meta  "..\..\..\data\frozen\v1_2025-09-13\metadata_ml_ready_splits_withGlobalID.xlsx" ^
  --scaling_mode train_only

---

## 7) Outputs & logging

- Metrics (CSV): `outputs/tables/metrics_*.csv` (includes seeds/params).  
- Figures (PNG): `outputs/figures/roc_*.png`, `confusion_matrix_*.png`.  
- Optional: save per-fold predictions (`outputs/preds/`) for error analysis.

Keep these artifacts under version control **only if** they are small; otherwise ignore via `.gitignore` and archive them externally.

---

## 8) Troubleshooting

- **XGBoost not installed:** scripts fall back to Logistic Regression only.  
- **Missing columns/paths:** verify `schema_withGlobalID.yaml` and ensure relative paths are valid.  
- **Unexpected high scores:** confirm that `scaling_mode=train_only` was used and that grouping by `participant_id_global` is active.  
- **Class imbalance instability:** check class prevalence per fold; adjust `class_weight` (LR) or `scale_pos_weight` (XGB).

---

## 9) Repro checklist

- [ ] Python 3.10, env from `requirements.txt`  
- [ ] Frozen snapshot chosen (e.g., `v1_2025-09-13`)  
- [ ] GroupKFold by `participant_id_global`  
- [ ] `scaling_mode=train_only` for deployable numbers  
- [ ] Seeds fixed and saved
