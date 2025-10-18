# Preparation Pipeline
Preparation Pipeline — Detailed
0) Data sources & working files

MMASD: OpenPose 2D skeleton JSONs (per frame) + subject/activity metadata → MMASD_merged.xlsx.

Engagnition: Empatica E4 signals (ACC/GSR/TMP) summarized per session → Engagnition_features.xlsx (and enriched variants).

Splits & IDs: metadata_ml_ready_splits.xlsx → metadata_ml_ready_splits_withGlobalID.xlsx (adds a cross-dataset participant key).

1) Harmonize IDs & rows (one row = one sample)

Goal: a unified, sample-level view that works across datasets.

Row granularity

MMASD: 1 row = 1 clip.

Engagnition: 1 row = 1 session.

Keys we standardize

dataset ∈ {MMASD, Engagnition}

sample_id (clip/session ID used for merges)

participant_id (dataset-local)

participant_id_global (cross-dataset key; derived in add_global_id.py)

Optional: condition (Engagnition: Baseline|LPE|HPE), activity_class_basic (MMASD), paths, etc.

Cleaning & QC

Drop duplicate rows by (dataset,sample_id).

Ensure each row has a valid participant_id.

Keep simple, human-readable columns; avoid any PII.

Outputs:
metadata_master.* (working merge table) and a frozen version:
data/frozen/v1_2025-09-13/{metadata_ml_ready_splits.xlsx, schema.yaml, splits_manifest.json, *withGlobalID.*}

2) Compute movement intensity per sample
2a) MMASD (skeleton / optical flow)

Input: per-frame keypoints (x,y,confidence) from OpenPose.

Per-frame speed series (one value per frame):

For each frame, choose the person with highest total confidence.

For each joint  
<img width="401" height="44" alt="image" src="https://github.com/user-attachments/assets/07b28713-cd75-4468-8fe1-f22681af2efe" />
rame speed = mean of <img width="52" height="30" alt="image" src="https://github.com/user-attachments/assets/e8c5e492-6226-4117-8b55-9147ed6671af" /> across all joints present.

Aggregate per clip (robust summaries):
median, 75th percentile (p75), IQR, std, MAD, max, variance, fraction above median, duration =  <img width="168" height="33" alt="image" src="https://github.com/user-attachments/assets/817a6074-78c4-4a96-9d16-3940c1027ce6" />

Base scalar for intensity:
movement_intensity_raw = median(frame_speed) (robust to outliers).

If optical-flow magnitude is available, the same aggregation is applied to the flow series; results are equivalent in spirit.

2b) Engagnition (ACC)

Input: accelerometer axes ax, ay, az (~32 Hz).

Signal Vector Magnitude (SVM): <img width="269" height="50" alt="image" src="https://github.com/user-attachments/assets/3a71b5e6-3c5a-49ac-9476-d0e1d6193b82" />

Aggregate per session (same robust set):
median, p75, IQR, std, MAD, max, var, fraction above median, duration = <img width="62" height="34" alt="image" src="https://github.com/user-attachments/assets/2d9bfcf1-89dd-45c5-b255-b09eaec8dbeb" />

Base scalar:
movement_intensity_raw = median(SVM).

3) Within-participant robust z-score

To make intensity comparable within the same person (removing per-subject scale):
For each participant p separately (and within each dataset to avoid cross-site leakage):
<img width="322" height="80" alt="image" src="https://github.com/user-attachments/assets/1504339c-c35c-4dd9-b623-41e94408cd39" />
Store as movement_intensity_z.

4) Binary target (proxy outcome)

We define a balanced, participant-relative label:

movement_intensity_bin = 1  if movement_intensity_z >= 0
                         0  otherwise

Threshold 0 equals the participant median → halves the person’s samples into higher- vs lower-movement intervals.

Rationale: keeps class balance and avoids cross-subject scale bias.

5) Save metadata & freeze

Working master: metadata_master.csv/.xlsx — convenient for inspection.

Frozen, split-ready (used by all experiments):

metadata_ml_ready_splits.xlsx

schema.yaml (column types/descriptions)

splits_manifest.json (canonical CV/test splits)

*_withGlobalID.* (adds participant_id_global for group-aware CV and clean merges)

Each freeze is stored under a versioned folder (e.g., data/frozen/v1_2025-09-13/) and kept immutable.

Practical notes & safeguards

Group awareness: all downstream CV uses participant_id_global (or a safe fallback) to prevent subject leakage across folds.

Missingness: rows with missing label or all-NaN features are excluded; partial NaNs are median-imputed per feature (fit on train only).

Reproducibility: every transformation is deterministic and tied to a specific frozen snapshot and script version; new changes → new version folder.

Extensibility: Engagnition can be enriched with GSR/TMP-derived features; MMASD can include optical-flow features — the preparation logic (robust summaries → z-score → binarization) stays the same.
