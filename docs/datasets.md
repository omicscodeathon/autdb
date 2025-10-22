# Datasets & Local Placement

This repository **does not** distribute original data. Obtain MMASD and Engagnition from the authors and place them locally as described below. All paths are **relative to the repo root**.

> **Privacy note.** We work only with **derived, non-identifiable** representations (e.g., 2D skeletons from videos; Empatica E4 time-series). Raw videos are excluded from the repo and experiments.

---

## 1) MMASD (video-derived features)

Expected layout (OpenPose JSONs per activity/clip):  
```text
data/MMASD/2D skeleton/2D_openpose_output/<activity>/<clip_id>/*.json
```
- Example `clip_id`: `MM_as_20583_D1_000_y`
- The preparation scripts read these JSON files and build tabular features stored under `data/Prepared data with features/` and `data/Prepared data with IDglobal/`.

---

## 2) Engagnition (Empatica E4 + annotations)

Folder-per-participant per condition (Baseline / LPE / HPE). Expected layout:  
```text
data/Engagnition/
  Baseline condition/Pxx/{E4AccData.csv, E4GsrData.csv, E4TmpData.csv}
  LPE condition/Pxx/{E4AccData.csv, E4GsrData.csv, E4TmpData.csv, EngagementData.csv, GazeData.csv, PerformanceData.csv}
  HPE condition/Pxx/{E4AccData.csv, E4GsrData.csv, E4TmpData.csv, EngagementData.csv, GazeData.csv, PerformanceData.csv}

# study-level spreadsheets shipped by the dataset
data/Engagnition/Subjective questionnaire.xlsx
data/Engagnition/InterventionData.xlsx
data/Engagnition/Session Elapsed Time.xlsx
```
- Participant folder format: `Pxx` (e.g., `P21`); sample ID examples: `ENG_P21_LPE`.
- ACC is the primary source for movement features; GSR/Temp can be used as optional covariates or for enrichment.
- CSV headers must match Empatica E4 export (timestamps in seconds; acceleration axes x/y/z).

---

## 3) Placement rules & tips

- Use **relative paths** in all metadata fields (e.g., `path_openpose_json`, `path_e4_acc`).  
- Do **not** modify raw exports; keep any cleaning in separate derived files.  
- Keep a versioned **frozen snapshot** under `data/frozen/vX_YYYY-MM-DD/` (tables, schema, and split manifests) and reference it in experiments.  
- Windows paths with spaces should be quoted in the command line (see README examples).

---

```
If any assertion fails, re-check the folder names and relative paths.
