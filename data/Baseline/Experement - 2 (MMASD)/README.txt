Step — MMASD-only dual-task experiment (Activity 3-class & movement_intensity_bin)

This experiment evaluates two tasks on MMASD only:

A) activity_grouped (3 classes) — maps the 11 activity classes into {music, yoga, movements}.

B) movement_intensity_bin (binary) — the proxy movement target used in the baseline.

It performs GroupKFold(n_splits=5) by participant, runs Logistic Regression and XGBoost, and saves all results to ./outputs/ in the same folder as the scripts.

Folder structure
AutDB-Video/
└─ data/
   └─ Baseline/
      └─ Experement - 2 (MMASD)/
         ├─ mmasd_dual_task.py
         ├─ train_from_cleaned.py
         ├─ MMASD_merged.xlsx
         ├─ metadata_ml_ready_splits_withGlobalID.xlsx
         └─ outputs/
            ├─ mmasd_metrics_activity.csv
            ├─ mmasd_metrics_intensity.csv
            ├─ features_used.txt
            ├─ Publication-ready plots.ipynb
            └─ figs_pub_ready/
               ├─ mmasd_activity_(3-class)_barplot.png
               ├─ mmasd_activity_(3-class)_corr_acc_mean_cv_macro_f1_mean_cv.png
               ├─ mmasd_activity_(3-class)_delta.png
               ├─ mmasd_intensity_(binary)_barplot.png
               ├─ mmasd_intensity_(binary)_corr_auroc_mean_cv_f1_pos_mean_cv.png
               └─ mmasd_intensity_(binary)_delta.png

1) mmasd_dual_task.py (primary run)

Inputs:
MMASD_merged.xlsx (features + basic metadata) and
metadata_ml_ready_splits_withGlobalID.xlsx (to attach participant_id_global).
Both must be in the same folder as the script.

Processing:

Adds participant_id_global; falls back to local participant_id if needed → pid_group.

Selects numeric features (excludes IDs, splits, demographics, targets).

Builds two targets:
A) activity_grouped via mapping from activity_class_basic → {music|yoga|movements}.
B) Uses existing movement_intensity_bin (required).

CV & models: GroupKFold(5) by pid_group; LogisticRegression (median impute + RobustScaler) and XGBClassifier; for binary, picks the F1-optimal threshold per fold and reports AUROC, balanced ACC, F1, ACC.

Outputs (auto-created):

outputs/mmasd_cleaned.csv — cleaned training table actually used by the run.

outputs/mmasd_metrics_activity.csv — metrics for task A.

outputs/mmasd_metrics_intensity.csv — metrics for task B.



2) train_from_cleaned.py (optional re-run from CSV)

Input: ./mmasd_cleaned.csv (created by the step above).

Processing: detects the group column (tries pid_group, participant_id_global, etc.), re-selects numeric features, adds a few derived features (e.g., skel_range, skel_cv, _log1p variants), and re-trains both tasks with GroupKFold(5) (LogReg with polynomial interaction-only features and XGBoost).

Outputs:

outputs/mmasd_metrics_activity.csv

outputs/mmasd_metrics_intensity.csv

outputs/features_used.txt (final feature list)

Use your plotting script to read the CSVs from ./outputs/ (e.g., mmasd_metrics_activity.csv, mmasd_metrics_intensity.csv) and generate the figures alongside them.

Commands (Windows CMD example)

Run these from inside:

autdb\data\Code for preparing tables\

Step 1 — Run the primary MMASD experiment

py -3.10 mmasd_dual_task.py

This creates outputs\mmasd_cleaned.csv and both metrics CSVs.


Step 2 — (Optional) Re-train from the cleaned CSV

py -3.10 train_from_cleaned.py

This regenerates metrics from mmasd_cleaned.csv and writes features_used.txt for auditing/plots.

Output summary

After successful execution, you will have:

.\outputs\
├─ mmasd_cleaned.csv
├─ mmasd_metrics_activity.csv        # Task A: acc, macro-F1, bal_acc (CV & OOF)
├─ mmasd_metrics_intensity.csv       # Task B: AUROC, F1, bal_acc, acc (CV & OOF)
└─ features_used.txt                 # only from train_from_cleaned.py

Notes

Dependencies: pandas, numpy, scikit-learn, xgboost, plus openpyxl for Excel I/O if needed.

Both scripts enforce participant-aware CV (GroupKFold by pid_group) to avoid leakage across clips from the same participant.
