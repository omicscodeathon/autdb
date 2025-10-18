Step  - LODO transfer on movement_intensity_bin (MMASD ⇄ Engagnition)

This experiment evaluates cross-dataset transfer (Leave-One-Dataset-Out) for the binary target movement_intensity_bin using a unified feature table and three scaling modes:

train_only — robust scaling fit only on the source (train) dataset → deployable, no leakage.

per_dataset — robust scaling fit separately on each dataset (train and test) → diagnostic; uses test stats.

global — robust scaling fit on train∪test pooled → diagnostic; uses test stats.

Scaling behavior is implemented in train_lodo_intensity_scaled.py (see functions prepare_scaled, robust_params). The script warns that per_dataset/global use target/test statistics and are not deployable.

Folder structure (as used here)
AutDB-Video/
└─ data/
   └─ Baseline/
      └─ Experement - 4 (LODO)/
         ├─ build_lodo_intensity_table.py
         ├─ train_lodo_intensity_scaled.py
         ├─ Lodo.ipynb
         ├─ mmasd_cleaned.csv                      # from Experiment 2
         ├─ Engagnition_features_enriched.xlsx     # from Experiment 3
         ├─ metadata_ml_ready_splits_withGlobalID.xlsx
         ├─ lodo_intensity_features.csv            # created in Step 1
         ├─ outputs global/                        # you will fill by renaming "outputs"
         ├─ outputs per_dataset/
         └─ outputs train_only/

1) build_lodo_intensity_table.py — create unified transfer table

Inputs:

mmasd_cleaned.csv (MMASD skeleton-derived features & label)

Engagnition_features_enriched.xlsx (Engagnition ACC-derived features & label)

Processing:

Detects group (participant) column from candidates like participant_id, pid_group, group_id, etc., and a row ID (e.g., sample_id, clip_id). Drops rows without label or group.

Maps MMASD skel_* and Engagnition acc_* features into a shared set named feat_* (e.g., skel_median→feat_median, acc_std→feat_std), and derives universal fields like feat_range, feat_max_per_s, feat_std_per_s when absent.

Writes a single CSV with columns:
dataset, group_id, sample_id, movement_intensity_bin, feat_median, feat_max, feat_p75, feat_iqr, feat_std, feat_var, feat_mad, feat_cv, feat_range, feat_high_fraction, feat_duration, feat_max_per_s, feat_std_per_s.

Output: lodo_intensity_features.csv. The script prints per-dataset counts and the final column list.

2) train_lodo_intensity_scaled.py — run LODO with chosen scaling

Inputs:

lodo_intensity_features.csv (from Step 1)

metadata_ml_ready_splits_withGlobalID.xlsx — used to map local participant IDs to a global ID and to load predefined CV folds (cv_fold_lodo or cv_fold_iid) per dataset; if missing, the script falls back to GroupKFold by participant on the train set.

Core pipeline:

Selects features (feat_*), converts the label to strict 0/1, removes rows with all-NaN features.

Splits by dataset: run MMASD→Engagnition and Engagnition→MMASD.

Applies robust scaling according to --scaling_mode (see below).

Tunes LogisticRegression (C grid) and, if available, XGBoost (hist, randomized search) using participant-aware CV on the train set; computes OOF probabilities and chooses the F1-optimal threshold on train.

Evaluates on target (held-out) dataset, writes metrics, ROC figure, and a JSON log with scaling metadata and best params.

Outputs (created inside outputs/)

tables/metrics_lodo_intensity_<scaling_mode>.csv — one row per direction × model (AUROC, PRAUC, balanced ACC, F1, ACC, threshold, prevalence, TP/FP/TN/FN, sizes, best params).

figures/roc_<Train>2<Test>_<model>_<scaling_mode>.png — ROC on the target dataset.

logs/log_<Train>2<Test>_<model>_<scaling_mode>.json — scaling stats, OOF thresholding diagnostics.

Scaling modes (the key idea)

All modes use median/IQR robust scaling; the difference is which data supply the statistics:

train_only — production-safe

Fit median/IQR only on the training dataset; apply to both train and test.

No look-ahead; recommended for real deployment and honest transfer evaluation.

per_dataset — diagnostic

Fit median/IQR separately on train and on test; each side uses its own stats.

Simulates “each site normalizes internally”, but uses test information → not deployable; useful to bound upper-limit performance when distribution shifts are neutralized by site-wise scaling. The script emits a warning.

global — diagnostic

Fit median/IQR on pooled train∪test, then apply to both.

Strongest form of leakage; for sanity checks only (e.g., “what if features were globally harmonized?”). The script emits a warning.

Commands (Windows CMD)

Run all commands from inside:

AutDB-Video\data\Baseline\Experement - 4 (LODO)\

Step 1 — Build the unified table
py -3.10 build_lodo_intensity_table.py ^
  --mmasd "mmasd_cleaned.csv" ^
  --eng  "Engagnition_features_enriched.xlsx" ^
  --out  "lodo_intensity_features.csv"


(Requires mmasd_cleaned.csv from Experiment-2 and Engagnition_features_enriched.xlsx from Experiment-3.)

Step 2 — Run LODO in train_only mode (deployable)
py -3.10 train_lodo_intensity_scaled.py ^
  --table "lodo_intensity_features.csv" ^
  --meta  "metadata_ml_ready_splits_withGlobalID.xlsx" ^
  --scaling_mode train_only


Rename the output folder to keep results separated:

ren "outputs" "outputs train_only"


Outputs are now in:

outputs train_only\tables\metrics_lodo_intensity_train_only.csv
outputs train_only\figures\roc_*.png
outputs train_only\logs\log_*.json


Step 3 — Run LODO in per_dataset mode (diagnostic)
py -3.10 train_lodo_intensity_scaled.py ^
  --table "lodo_intensity_features.csv" ^
  --meta  "metadata_ml_ready_splits_withGlobalID.xlsx" ^
  --scaling_mode per_dataset


Then:

ren "outputs" "outputs per_dataset"


(Results saved under outputs per_dataset\...).

Step 4 — Run LODO in global mode (diagnostic)
py -3.10 train_lodo_intensity_scaled.py ^
  --table "lodo_intensity_features.csv" ^
  --meta  "metadata_ml_ready_splits_withGlobalID.xlsx" ^
  --scaling_mode global


Then:

ren "outputs" "outputs global"


(Results saved under outputs global\...).

Optional: if you have a splits Excel with predefined participant-wise folds per dataset (cv_fold_lodo / cv_fold_iid), add:
--splits_excel "metadata_ml_ready_splits_withGlobalID.xlsx" — the script will use those exact folds; otherwise it falls back to GroupKFold on the train dataset.

Outputs overview (after all three runs)
outputs train_only/
   ├─ tables/metrics_lodo_intensity_train_only.csv
   ├─ figures/roc_MM...png, roc_En...png
   └─ logs/log_MM...json, log_En...json

outputs per_dataset/
   └─ (same structure; scaling uses per-dataset stats)

outputs global/
   └─ (same structure; scaling uses pooled stats)


Use Lodo.ipynb to aggregate the three CSVs, compare AUROC/F1/Δ vs. mode, and select figures for the paper.

Dependencies
py -3.10 -m pip install pandas numpy scikit-learn xgboost matplotlib openpyxl


Reproducibility tip. Always keep the three output folders (outputs train_only, outputs per_dataset, outputs global) side-by-side under this experiment directory. They represent comparable LODO evaluations that differ only by scaling strategy; all other code paths (features, splits, models, tuning) remain identical.
