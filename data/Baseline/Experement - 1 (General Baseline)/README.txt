Run baseline experiment on movement_intensity (IID & LODO)

This script trains and evaluates baseline classifiers (Logistic Regression and optional XGBoost) on the movement_intensity_bin target with robust, participant-aware CV and transfer (LODO) tests across MMASD and Engagnition. It also saves calibration plots and fairness slices.

What the script does

Loads inputs (auto-locates via local folder or configured fallbacks):

MMASD_merged.xlsx — per-clip skeleton features merged with MMASD basic metadata.

Engagnition_features.xlsx — per-session ACC features (E4 accelerometer).

metadata_ml_ready_splits_withGlobalID.xlsx — split/ID map with participant_id_global.

Merges meta + feature tables by sample_id, sets/infers dataset for each row.

Builds labels (configurable):

Variant A (default): computes movement_intensity_z_auto via robust z-score per (dataset, participant) and sets movement_intensity_bin by median (if not already present).

Variant B: per-dataset quantile split (LABEL_B_QUANTILE).

Selects features: numeric columns only; excludes IDs/splits/leaks by regex (e.g., ^sample_id$, ^participant_id_global$, ^split_.*, ^movement_intensity_.*, etc.). Optionally drops shortcuts like duration_s, condition.

Modeling pipeline:

Nested GroupKFold by participant_id_global (outer CV for performance, inner CV for tuning).

Preprocessing: median imputation → RobustScaler → correlation filter (drop >0.98 abs corr).

Models: LogisticRegression (with balanced class weights) and optional XGBClassifier (if xgboost installed).

Isotonic calibration fit in a grouped OOF manner; threshold chosen to maximize F1 on validation.

Evaluation tasks:

IID (MMASD): train∪val vs test holdout from meta split (split_iid if available; else participant-wise 80/20). Saves reliability plots and cluster-bootstrap AUC CIs by participant.

IID (Engagnition): CV-only (GroupKFold), with OOF reliability plot.

LODO: MMASD→Engagnition and Engagnition→MMASD (uses MMASD meta test if present, else all MMASD).
For each: train on source, evaluate on target with calibrated probs & F1-optimal threshold from training.

Fairness slices (if columns exist): computes AUC/BACC/F1/ACC/Brier per sex and age_group, with bar plots and AUC CIs by participant cluster.

Outputs

Tables → outputs/tables/metrics_all.csv, plus metrics_iid.csv, metrics_lodo.csv, and metrics_fairness.csv (if any).

Figures → outputs/figures/reliability_*.png, fairness_auc_*.png.

Folder structure (expected)
autdb/
├─ data/
│  ├─ Code for preparing tables/
│  │   ├─ train_mi_baselines.py
│  │   ├─ MMASD_merged.xlsx
│  │   ├─ Engagnition_features.xlsx
│  │   └─ metadata_ml_ready_splits_withGlobalID.xlsx
│  └─ (optional) frozen snapshot(s) used as fallbacks
└─ outputs/
   ├─ tables/
   └─ figures/


The script also searches configured fallback locations (e.g., your data/frozen/v1_2025-09-13) if the three inputs aren’t found in the current folder.

Commands (Windows CMD example)

Run from inside:

autdb\data\Code for preparing tables\


Default run (LogReg + XGB if installed; creates outputs/tables & outputs/figures):

py -3.10 train_mi_baselines.py


If you don’t have XGBoost, install it or ignore; the script automatically skips XGB when the package is missing.
To change label variant, exclusion patterns, CV folds, or random seeds, edit the CONFIG dict at the top of the script.

Outputs (after successful execution)
outputs/
├─ tables/
│  ├─ metrics_all.csv
│  ├─ metrics_iid.csv
│  ├─ metrics_lodo.csv
│  └─ metrics_fairness.csv           # if fairness groups available
└─ figures/
   ├─ reliability_IID_MMASD_*.png
   ├─ reliability_IID_Engagnition_*_OOF.png
   ├─ reliability_LODO_*.png
   └─ fairness_auc_*.png             # grouped by sex / age_group when present

Dependencies

pandas, numpy, scikit-learn, tqdm, matplotlib, openpyxl (Excel I/O), and optional xgboost. Install in your env:

py -3.10 -m pip install pandas numpy scikit-learn tqdm matplotlib openpyxl xgboost


Reproducibility note. With the three inputs present and the folder structure above, running the one-liner CMD will regenerate the exact metric tables and plots reported by this baseline, using GroupKFold by participant and isotonic calibration as described.
