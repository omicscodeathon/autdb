# Preparation Pipeline

This document describes the **end-to-end data preparation** used by AutDB: from building basic tables and features to producing a frozen, split-ready snapshot for training.

> Scope: **Python-only** pipeline; works on Windows(tested on Python 3.10).  
> Data: original datasets are **not distributed** — see `docs/datasets.md` for local placement.

---

## 0) Prerequisites

### Environment setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

python -m pip install -U pip
pip install -r scripts/requirements.txt
```

### Expected local data layout
See `docs/datasets.md` for the exact folder structures of **MMASD** (2D skeleton JSON) and **Engagnition** (Empatica E4 CSV + study spreadsheets).

---

## 1) Build per‑dataset basic tables & features

> Outputs of this section live under `data/Prepared data with features/` and feed the unification stage.

### 1.1 MMASD (2D skeleton → features)
**Input:** OpenPose JSONs (per frame) under `data/MMASD/2D skeleton/2D_openpose_output/`.

Run from `data/Code for preparing tables/`:
See README.txt

### 1.2 Engagnition (ACC → features, + engagement)
**Input:** Empatica E4 CSVs per participant/condition + study spreadsheets.

Run from `data/Code for preparing tables/`:
See README.txt

---

## 2) Unify IDs & rows

Goal: a **single, sample-level view** across datasets with consistent keys for merges and group-aware CV.

- **Row definition**  
  - MMASD: 1 row = 1 **clip**  
  - Engagnition: 1 row = 1 **session**
- **Keys**  
  `dataset ∈ {MMASD, Engagnition}`, `sample_id` (clip/session), `participant_id` (dataset-local), `participant_id_global` (cross‑dataset), plus optional `condition` (Engagnition), `activity_class_basic` (MMASD), and relative **paths** used by scripts.
- **Cleaning & QC**  
  De-duplicate by `(dataset, sample_id)`, ensure valid `participant_id`, keep human‑readable columns only (no PII).

### 2.1 Notebooks (ID unification + cleaning)
From `data/Prepared data with IDglobal/` run:
```bash
jupyter nbconvert --to notebook --execute "Metadata cleaning.ipynb"     --output "Metadata_cleaning_executed.ipynb"
jupyter nbconvert --to notebook --execute "schema.yaml.ipynb"           --output "schema_executed.ipynb"
```
Key outputs:
- `metadata_ml_ready.xlsx` — unified table for modeling
- `schema.yaml` — column definitions and dtypes

### 2.2 Fixed participant-level splits
We ship canonical splits in `splits_manifest.json` with two setups: `split_iid` and `split_lodo` (each with `train/val/test`).  
Use these to derive split-aware tables or consume them directly at training.

---

## 3) Frozen snapshot (immutable)

Freeze all ingredients used in experiments under a versioned folder, e.g.:
```
data/frozen/v1_2025-09-13/
  metadata_ml_ready_splits.xlsx
  schema.yaml
  splits_manifest.json
  add_global_id.py
  metadata_ml_ready_splits_withGlobalID.xlsx
  schema_withGlobalID.yaml
  splits_manifest_withGlobalID.json
```
**Rules**
- Frozen folders are **immutable**. Any changes → create a new `vN_YYYY-MM-DD`.
- All training scripts reference a **specific** frozen snapshot to ensure comparability.

---

## 4) Movement-intensity targets

- Base scalar per sample: robust **median** of frame‑wise speed (MMASD) or **median** SVM (Engagnition ACC).  
- Within‑participant normalization: robust z‑score **per participant** (and per dataset) to remove subject scale.  
- Binary proxy label: `movement_intensity_bin = 1 if z >= 0 else 0` — balanced within each participant.

---

## 5) Reproducibility notes

- Use **GroupKFold(groups=participant_id_global)** to avoid subject leakage in any CV.  
- Perform scaling/fit steps in training with **train-only statistics** (`scaling_mode=train_only` in LODO scripts).  
- After a successful run, export a lock file:  

---

## 6) Troubleshooting

- Paths with spaces on Windows must be quoted (`"path with spaces"`).  
- If a script cannot find inputs, verify **relative** paths in metadata columns.  
- If imports fail, re‑install from `requirements.txt` and ensure Python 3.10 is active.

