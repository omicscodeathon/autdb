# AutDB
# Privacy-First Benchmark for Video-Based ASD Screening

**One-liner.** Reproducible, privacy-preserving benchmark that harmonizes two local datasets (**MMASD** and **Engagnition**) into a unified metadata table and evaluates **transportability (LODO)** and **fairness** on a shared proxy target `movement_intensity_bin`.

> ⚠️ Raw videos are **not** stored here. We use only **derived, non-identifiable** features and local file paths. Both datasets must be obtained from the original authors and stored locally.

---

## What’s inside

- **Unified metadata** across two datasets (one row = MMASD clip or Engagnition session)
- **Shared target** `movement_intensity_bin` (robust-z within participant, binarized at `z ≥ 0`)
- **Setups:**  
  - **IID:** GroupKFold within each dataset (groups = `participant_id_global`)  
  - **LODO:** train on one dataset, test on the other
- **Metrics:** AUROC (primary), Balanced Accuracy, F1
- **Saved outputs:** CSV metrics in `outputs/tables/` and figures in `figs/`

---

## Repository layout

```
data/
  MMASD/                       # local-only (2D/3D skeletons, optical flow, etc.)
  Engagnition/                 # local-only (ACC/GSR/Temp + annotations)
  Frozen Basic Data/...        # frozen manifests / snapshots
  Prepared data with IDglobal/
  Prepared data with features/
Baseline/
  figures.ipynb                # render ROC & fairness plots from saved CSVs
Code for preparing tables/     # harmonization & label preparation notebooks/utils
outputs/
  tables/                      # metrics_*.csv (IID/LODO, fairness)
figs/                          # roc_lodo.png, fairness_deltas.png
```

---

## Quick start

```bash
# 1) Create environment (Python 3.10+)
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt

# 2) Place data under ./data (see "Data placement" below)

# 3) Build unified metadata & labels
#    Open notebooks in "Code for preparing tables/" and run the pipeline.
#    This produces metadata_master.* and training-ready tables.

# 4) Train & evaluate (IID + LODO)
#    Run the training notebook/script provided in the repo.
#    Metrics will be saved to outputs/tables/metrics_{iid,lodo}.csv

# 5) Render figures
#    Open Baseline/figures.ipynb to generate figs/roc_lodo.png and figs/fairness_deltas.png
```

---

## Data placement (local)

### MMASD (example)
```
data/MMASD/2D skeleton/2D_openpose_output/<activity>/<clip_id>/*.json
# optional
data/MMASD/ROMP/<activity>/<clip_id>/*.npz
data/MMASD/optflow/<activity>/<clip_id>/*.npy
```

### Engagnition (example)
```
data/Engagnition/
  Baseline condition/Pxx/{E4AccData.csv,E4GsrData.csv,E4TmpData.csv}
  LPE condition/Pxx/{... + EngagementData.csv, GazeData.csv, PerformanceData.csv}
  HPE condition/Pxx/{... + EngagementData.csv, GazeData.csv, PerformanceData.csv}
Subjective questionnaire.xlsx
InterventionData.xlsx
Session Elapsed Time.xlsx
```

> The exact trees and file naming follow the dataset documentation. Only **derived** features and tabular files are referenced by the benchmark.

---

## Minimal field schema (high-level)

- Keys: `sample_id`, `participant_id_global`, `dataset`
- Targets: `movement_intensity_raw`, `movement_intensity_z`, `movement_intensity_bin`
- Demographics for fairness: `sex`, `age_years` (and/or `age_group`)
- Reproducibility: `split_seed`, `split_iid`, `split_lodo`, `group_kfold`
- Provenance: `source_file`, `path_*` to derived features (relative paths)

A full schema with types and examples is provided in `docs/metadata_schema.md` (see below).

---

## Reproducing our results

1. **Harmonize** the datasets → produce `metadata_master.*` using the notebooks in `Code for preparing tables/`.  
2. **Train IID** per dataset and **LODO** cross-dataset using the provided training notebook/script.  
3. **Collect metrics** from `outputs/tables/metrics_{iid,lodo}.csv`.  
4. **Plot** with `Baseline/figures.ipynb` → `figs/roc_lodo.png`, `figs/fairness_deltas.png`.

---

## Intended use & limitations

- Datasets comprise **children with ASD only**; this is **not** an ASD-vs-TD diagnostic pipeline.  
- `movement_intensity_bin` is a **proxy** label designed for cross-dataset comparability and privacy.  
- We intentionally favor simple, transparent tabular models to emphasize **transportability** and **fairness**, not leaderboard SOTA.

---

## Docs (recommended)

Create a `docs/` folder and add:

- `docs/datasets.md` – how to obtain and place the datasets; exact folder trees  
- `docs/metadata_schema.md` – complete field list, types, allowed values, examples  
- `docs/preparation.md` – step-by-step harmonization & label building (`movement_intensity_*`)  
- `docs/training.md` – training configs, GroupKFold details, seeds, optional class/subject weights  
- `docs/results.md` – how to read `metrics_*.csv` and reproduce the figures  
- `docs/reproducibility.md` – versioning, frozen manifests, split manifests

---

## Citing & license

Please cite the **original datasets** (MMASD and Engagnition) and this benchmark if you use it in a paper or poster.  
License: see `LICENSE`.

**Contact.** Questions and suggestions are welcome in GitHub Issues.
