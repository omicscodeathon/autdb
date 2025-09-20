# Metadata Schema (high-level)

One row = one MMASD clip or one Engagnition session.

| Field | Type | Example | Notes |
|---|---|---|---|
| `sample_id` | str | `ENG_P21_LPE` | Unique sample key |
| `participant_id_global` | str | `EN_P21` | Participant key, used for grouping |
| `dataset` | cat | `Engagnition` / `MMASD` | Dataset source |
| `sex` | cat | `M` / `F` / `NA` | Optional for fairness slices |
| `age_years` | float | `6.8` | Optional; used to form `age_group` |
| `age_group` | cat | `<=6`, `>6` | Example binning |
| `movement_intensity_raw` | float | `0.42` | From skeleton/ACC aggregates |
| `movement_intensity_z` | float | `-0.13` | Robust z-score **within participant** |
| `movement_intensity_bin` | int {0,1} | `1` | `1 if z ≥ 0 else 0` |
| `source_file` | str | path-like | Provenance identifier |
| `path_openpose` | str | `data/MMASD/...json` | Optional modality path |
| `path_acc` | str | `data/Engagnition/...E4AccData.csv` | Optional modality path |
| `split_seed` | int | `42` | Reproducibility |
| `split_iid` | str | `train/val/test` | If pre-generated |
| `split_lodo` | str | `train/test` | Cross-dataset split label |
| `group_kfold` | int | `0..K-1` | Fold index if precomputed |

### Example (JSON)

```json
{
  "sample_id": "ENG_P21_LPE",
  "participant_id_global": "EN_P21",
  "dataset": "Engagnition",
  "sex": "M",
  "age_years": 7.2,
  "movement_intensity_raw": 0.41,
  "movement_intensity_z": 0.55,
  "movement_intensity_bin": 1,
  "source_file": "LPE condition/P21/E4AccData.csv",
  "path_acc": "data/Engagnition/LPE condition/P21/E4AccData.csv",
  "split_seed": 42,
  "split_lodo": "train"
}
```
