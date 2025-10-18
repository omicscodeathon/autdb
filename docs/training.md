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
