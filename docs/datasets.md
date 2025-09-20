# Datasets & Placement

This repository does **not** distribute original data. Obtain datasets from the authors and store them locally as outlined below.

## MMASD (video-derived features only)

```
data/MMASD/2D skeleton/2D_openpose_output/<activity>/<clip_id>/*.json

```

Notes:
- We rely on **derived, non-identifiable** representations (2D/3D skeletons, optical flow). Raw videos are excluded.
- `sample_id` examples: `MM_as_20583_D1_000_y`

## Engagnition (Empatica E4 + annotations)

```
data/Engagnition/
  Baseline condition/Pxx/{E4AccData.csv,E4GsrData.csv,E4TmpData.csv}
  LPE condition/Pxx/{... + EngagementData.csv, GazeData.csv, PerformanceData.csv}
  HPE condition/Pxx/{... + EngagementData.csv, GazeData.csv, PerformanceData.csv}
Subjective questionnaire.xlsx
InterventionData.xlsx
Session Elapsed Time.xlsx
```

Notes:
- Use participant folder `Pxx` for each condition (Baseline/LPE/HPE).
- We aggregate ACC into movement features; GSR/Temp may be used as optional covariates.
- `sample_id` examples: `ENG_P21_LPE`

## General Tips

- Use **relative paths** from repo root for all `path_*` fields in metadata.
- Keep a `Frozen Basic Data/vX_YYYY-MM-DD` snapshot (manifests, split files) for exact reproducibility.
