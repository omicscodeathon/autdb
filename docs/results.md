# Results Files

## Metrics CSV (example columns)
- `setup` (iid / lodo)
- `dataset_train`, `dataset_test`
- `model` (logreg / xgb / ensemble)
- `seed`, `fold`
- `auroc`, `bacc`, `f1`

Optionally add per-slice columns: `group=sex:M`, `auc_group`, etc.

## Figures
- `figs/roc_lodo.png` — ROC curves for cross-dataset tests
- `figs/fairness_deltas.png` — ΔAUC/ΔF1 with 95% CIs
