Step — Engagnition (ACC-only vs Enriched ACC/GSR/TMP)

This experiment runs intra-dataset CV on Engagnition for two targets:

A) engagement_level (multiclass 0/1/2)

B) movement_intensity_bin (binary 0/1)

You can train on ACC-only features or on an enriched table (ACC + derived ratios + optional GSR/TMP features if raw CSVs are available). Results are written to the local outputs/ folder.

Folder structure (as used here)
AutDB-Video/
└─ data/
   └─ Baseline/
      └─ Experement - 3 (Engagnition)/
         ├─ make_eng_feature_enrichment.py
         ├─ train_engagnition.py
         ├─ train_engagnition_enriched.py
         ├─ Engagnition_features.xlsx
         ├─ Engagnition_features_with_engagement.xlsx
         ├─ Engagnition_features_enriched.xlsx        # created in Step 1
         └─ outputs/
            ├─ figures/
            └─ tables/
               ├─ metrics_engagement_enriched.csv
               ├─ metrics_intensity_enriched.csv
               └─ Publication plots for Engagement.ipynb

What each script does

make_eng_feature_enrichment.py — takes Engagnition_features_with_engagement.xlsx and adds derived ACC features (e.g., log-energy, energy per second, CV, ratios to median, high-activity flag). If you pass --data-root and your table has source_dir, it will also read raw E4 CSVs (E4AccData.csv, E4GsrData.csv, E4TmpData.csv) and append extra ACC/GSR/TMP features (spectral entropy, dominant freq, slopes, peaks/min, etc.). Saves to Engagnition_features_enriched.xlsx.

train_engagnition.py — runs 5-fold GroupKFold by participant_id on ACC-only features (columns acc_*) for both tasks, with Logistic Regression and XGBoost (+ inner tuning). Outputs OOF metrics to outputs/tables/*.csv and plots (ROC for binary, confusion matrix for multiclass) to outputs/figures/*.png.

train_engagnition_enriched.py — same protocol as above, but uses enriched features: acc_*, gsr_*, tmp_* (leakage-safe; demographics removed). Writes metrics_*_enriched.csv and corresponding figures.

Commands (Windows CMD)

Run all commands from inside:

AutDB-Video\data\Baseline\Experement - 3 (Engagnition)\

1) (Optional) Build the enriched feature table

Use this if you want ACC-derived ratios and, optionally, extra GSR/TMP features from raw CSVs.

py -3.10 make_eng_feature_enrichment.py ^
  --input  "Engagnition_features_with_engagement.xlsx" ^
  --output "Engagnition_features_enriched.xlsx" ^
  --data-root "..\..\Engagnition"

Output: Engagnition_features_enriched.xlsx.

2) Train on ACC-only features
py -3.10 train_engagnition.py ^
  --data "Engagnition_features_with_engagement.xlsx"


Outputs:

outputs\tables\metrics_engagement.csv, outputs\tables\metrics_intensity.csv

outputs\figures\engagement_confmat_*.png, outputs\figures\intensity_roc_*.png

3) Train on enriched features (ACC + GSR + TMP)
py -3.10 train_engagnition_enriched.py ^
  --data "Engagnition_features_enriched.xlsx"


Outputs:

outputs\tables\metrics_engagement_enriched.csv, outputs\tables\metrics_intensity_enriched.csv

outputs\figures\engagement_confmat_*.png, outputs\figures\intensity_roc_*.png

Notes & dependencies

All training scripts use GroupKFold by participant to prevent subject leakage; features are median-imputed, RobustScaled, and de-correlated via a simple correlation filter; XGBoost is tuned with randomized search.

Install deps in your env:

py -3.10 -m pip install pandas numpy scikit-learn xgboost matplotlib openpyxl
