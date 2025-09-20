# Preparation Pipeline

## 1) Harmonize IDs & rows
- Build a unified table where **one row = sample** (MMASD clip or Engagnition session).
- Columns: keys, demographics, paths, and placeholders for targets.

## 2) Compute movement intensity

### MMASD (skeleton/flow)
- Derive per-frame joint velocities or optical-flow magnitude statistics.
- Aggregate per clip (e.g., median, 75th percentile) → `movement_intensity_raw`.

### Engagnition (ACC)
- Compute Signal Vector Magnitude (SVM = sqrt(ax^2+ay^2+az^2)) per timestamp.
- Aggregate per session (e.g., median, 75th percentile) → `movement_intensity_raw`.

## 3) Within-participant robust z-score
For each participant independently:
```
z = (raw - median(raw_participant)) / IQR(raw_participant)
```
Store as `movement_intensity_z`.

## 4) Binary target
```
movement_intensity_bin = 1 if movement_intensity_z >= 0 else 0
```

## 5) Save metadata
- Write `metadata_master.*` (CSV/Parquet).
- Keep a frozen snapshot in `Frozen Basic Data/vX_YYYY-MM-DD`.
