End-to-End Process
1) Datasets & privacy model

MMASD (Li et al., 2023) — group-therapy video clips of children; privacy-preserving because only derived representations are exposed to us (e.g., 2D skeleton keypoints per frame).

Engagnition (Kim et al., 2024) — serious-game sessions with Empatica E4 signals (ACC 32 Hz, GSR 4 Hz, skin temperature 4 Hz) plus engagement and intervention annotations.

Both datasets are handled without PII; we work with de-identified IDs and derived time-series/features only.

2) Feature extraction (per dataset)
2.1 MMASD — skeleton-based movement

Inputs: OpenPose 2D keypoint JSONs (one file per frame) and a basic table with clip/activity info.
Processing (see build_mmasd_features_fast.py + merge scripts):

For each frame, select the person with max confidence.

Compute joint displacement per frame (L2 distance between successive frames for each joint), then average across joints → a frame-level speed series.

Aggregate per clip with robust statistics: median, p75, IQR, std, MAD, max, var, fraction above the median, duration (frames/fps).

Save one row per clip to Excel and merge with the basic metadata (MMASD_merged.xlsx).

Intuition: the median of the per-frame speed series is a stable proxy for “how much movement happened” in the clip.

2.2 Engagnition — accelerometer-based movement (+ optional enrichment)

Inputs: E4AccData.csv (and optionally E4GsrData.csv, E4TmpData.csv) located under Engagnition/<condition>/<participant>/. A basic table lists sessions and conditions.
Processing (see build_eng_features.py, merge_eng_features_with_engagement.py, make_eng_feature_enrichment.py):

Compute the Signal Vector Magnitude (SVM) per sample:
<img width="282" height="38" alt="image" src="https://github.com/user-attachments/assets/33651f88-f6de-4884-88ea-260dc5d39df2" />

Aggregate per session: median, p75, IQR, std, MAD, max, var, high-fraction (share above session median), duration (N/32).

(Optional) Enrich with additional ACC-derived ratios and, when available, GSR/TMP features (spectral/temporal descriptors).

Left-join the session-level engagement labels (0/1/2) onto the feature table.

Intuition: the median SVM summarizes overall movement intensity in a way that is robust to spikes.

3) Build base tables & harmonize IDs

We keep each dataset’s base table separate and then harmonize:

Row unit: one row = MMASD clip or Engagnition session.

Common keys:

dataset ∈ {MMASD, Engagnition}

sample_id (clip/session ID used for joins)

participant_id (dataset-local)

participant_id_global (cross-dataset key added by add_global_id.py)

Why global ID: enables GroupKFold by participant across all experiments, prevents leakage, and supports subject-level reporting.

4) Form the movement_intensity target

We create a robust, privacy-preserving proxy outcome from the movement summaries.

4.1 Continuous score

For each sample we compute movement_intensity_raw:

MMASD: median per-frame skeleton speed (or flow magnitude if used).

Engagnition: median SVM.

4.2 Normalization (two views, used in different experiments)

Participant-relative z (primary for IID/MMASD)
For each participant (and within each dataset):
<img width="286" height="35" alt="image" src="https://github.com/user-attachments/assets/ceab0287-2282-411b-bcf0-6af94db1fc1b" />

Store as movement_intensity_z.
This cancels person-specific scale and preserves intra-person contrast.

Train-derived global threshold (for transfer probes)
Compute a threshold on the training set (e.g., train median of raw or of z) and apply it everywhere. This mimics “what a deployed model would see” when the threshold was set without peeking at target data.

4.3 Binary label

Define a balanced binary proxy:
movement_intensity_bin = 1 for “high” movement, 0 otherwise.

In MMASD/IID we typically use z ≥ 0 (per-participant median split).

In the quick LODO probes we also evaluate a global threshold learned on train only.

5) Freeze metadata and define splits

We store a frozen snapshot for reproducibility:

metadata_ml_ready_splits.xlsx, schema.yaml, splits_manifest.json (+ *_withGlobalID.* versions).

Participant-wise splitting is enforced everywhere: GroupKFold (CV) and predefined test splits where applicable.

All experiments read from these frozen files to avoid accidental leakage or drift.

6) Experiments
6.1 Experiment 1 — Baseline (sanity checks)

Goal: establish a minimal, reproducible baseline for movement_intensity_bin:

Models: Logistic Regression (balanced) and optional XGBoost.

Preprocess: median impute → RobustScaler → correlation filter.

Evaluation:

IID on each dataset (participant-aware CV / holdout).

Quick LODO probe (MMASD→Engagnition and back) with train-only calibration & thresholding.

Artifacts: outputs/tables/*.csv, outputs/figures/reliability_*.png and fairness slices if demographics available.

Takeaway: a transparent reference point before dataset-specific tuning.

6.2 Experiment 2 — MMASD-only (dual-task)

Tasks:
A) activity_grouped (3-class): map 11 therapy activities into {music, yoga, movements}.
B) movement_intensity_bin (binary).

Protocol: 5-fold GroupKFold by participant, Logistic Regression and XGBoost.
Outputs:

outputs/mmasd_cleaned.csv (auditable training table),

outputs/mmasd_metrics_activity.csv, outputs/mmasd_metrics_intensity.csv,

plotting notebook + publication-ready barplots/correlation/delta.

Purpose: show an intra-dataset ceiling and confirm stability of the proxy within MMASD.

6.3 Experiment 3 — Engagnition (ACC-only vs enriched)

Targets: engagement_level (0/1/2) and movement_intensity_bin.
Feature settings:

ACC-only (baseline)

Enriched (ACC + derived ratios + optional GSR/TMP features from raw E4 files).

Protocol: 5-fold GroupKFold by participant; Logistic Regression / XGBoost with identical preprocessing; export OOF confusion matrices (multiclass) and ROC curves (binary).
Outputs: outputs/tables/metrics_*.csv, outputs/figures/*.png.

Purpose: quantify within-dataset performance and the value of enrichment.

6.4 Experiment 4 — LODO transfer (scaling modes)

Target: movement_intensity_bin, source↔target = MMASD and Engagnition.
Unified table: lodo_intensity_features.csv (harmonized features feat_*, common IDs).
Core idea: performance depends heavily on how you scale features under distribution shift, so we run the same model under three scaling regimes:

train_only — production-safe

Fit RobustScaler (median/IQR) on source (train) only, apply to both source and target.

No leakage; reflects deployable reality.

per_dataset — diagnostic

Fit scaling separately on source and on target; each side uses its own med/IQR.

Simulates “each site normalizes internally”.

Uses target stats → not deployable, but shows an upper bound if local normalization is allowed.

global — diagnostic

Fit scaling on pooled source∪target.

Strong leakage; sanity check only (what if the distributions were perfectly harmonized).

Evaluation: participant-aware CV on the train dataset for tuning, isotonic calibration and F1-optimal threshold learned on train, then test on the held-out dataset.
Outputs: per-mode CSVs (metrics_lodo_intensity_*.csv), ROC plots, and JSON logs with scaling stats.
Purpose: make the scaling assumption explicit, quantify its effect, and keep a deployable reference (train_only) next to upper-bound diagnostics.

7) Reporting & compliance

We document methods and results to meet TRIPOD-AI / PROBAST-AI transparency, and keep alignment with EU AI Act principles (clarity about data provenance, preprocessing, model choices, calibration, and known limitations). Every experiment writes machine-readable tables/figures and references a frozen snapshot to ensure full reproducibility.

8) What this buys us (why this design)

Privacy first: only derived representations (skeletons, SVM/GSR/TMP summaries), no PII.

Leakage control: participant-wise splits, train-only preprocessing/calibration for deployable settings.

Comparability: a single, harmonized target (movement_intensity_bin) exists across both datasets.

Portability insight: the LODO + scaling modes isolate how much of the transfer gap is due to distribution shift vs. modeling.

Reproducibility: everything is versioned (frozen folders) and re-runnable with one-liner commands in each experiment directory.

9) Known limitations (explicit)

The proxy target is behavioral movement, not a clinical ASD diagnosis.

Demographics are limited; fairness slices are exploratory.

Enrichment (GSR/TMP) depends on availability/quality of raw E4 files.

Global/per-dataset scaling are diagnostic only (not deployable); final claims should use train-only results.
