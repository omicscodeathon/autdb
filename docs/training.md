# Training & Evaluation

## Experimental setups
- **IID:** within each dataset using **GroupKFold** (groups = `participant_id_global`) to prevent leakage across a child.
- **LODO:** train on one dataset, test on the other, using only columns available in both.

## Models
- Start with **Logistic Regression** (with `RobustScaler`) and optional **XGBoost**.
- Consider **CalibratedClassifierCV** for probability calibration.

## Metrics
- Primary: **AUROC**
- Secondary: **Balanced Accuracy**, **F1**
- Report mean ± std over repeats/folds.

## Implementation sketch
1. Load `metadata_master.*` → select features (exclude demographics if you want fairness-only slicing, not leakage).
2. Split:
   - IID: `GroupKFold(n_splits=5)`; repeat with different seeds if needed.
   - LODO: train=MMASD, test=Engagnition and vice versa.
3. Train, predict, compute metrics; save CSV to `outputs/tables/`.
4. Plot ROC/fairness deltas via `Baseline/figures.ipynb`.
