<div align="center">

# **AutDB - Privacy‑First Benchmark for Video‑Based ASD Screening**

*A reproducible, privacy‑preserving baseline that harmonizes two datasets - **MMASD** (video‑derived skeleton/flow) and **Engagnition** (E4 wristband + annotations) - into a unified table and evaluates **transportability (LODO)** and **IID** performance on a shared proxy target.*

</div>

---

## 🔎 Overview

- **Goal.** Provide a transparent, reproducible pipeline for training and evaluating portable ML models on derived, **non‑identifiable** features from MMASD and Engagnition.
- **Key deliverables.**
  - A **unified metadata table** (one row = MMASD clip or Engagnition session).
  - A shared proxy target: `movement_intensity_raw → z‑score within participant → movement_intensity_bin (z ≥ 0)`.
  - Ready‑to‑run **IID** (GroupKFold by `participant_id_global`) and **LODO** (train on one dataset, test on the other) experiments.
  - Saved **metrics CSVs** (AUROC, Balanced Accuracy, F1) and optional figures.
- **Privacy notice.** No raw videos are stored here. Only derived features and relative file paths. Obtain original data from the dataset owners (see `docs/datasets.md`).

---

## 🗂️ Repository Structure

autdb/
├─ docs/
│  ├─ preparation.md         # end-to-end data/feature preparation
│  ├─ training.md            # IID/LODO setups, models, metrics, scaling modes
│  ├─ reproducibility.md     # seeds, frozen snapshots, lock files
│  ├─ metadata_schema.md     # column definitions (incl. participant_id_global)
│  └─ datasets.md            # how to place source datasets locally
├─ data/
│  ├─ Code for preparing tables/            # build_* and compute_* scripts
│  ├─ Prepared data with features/          # feature builders & merges
│  ├─ Prepared data with IDglobal/          # cleaning/schema notebooks, splits manifest
│  ├─ frozen/
│  │  └─ v1_2025-09-13/                     # immutable snapshot (tables, schema, splits)
│  └─ Baseline/
│     ├─ Experiment - 1 (General Baseline)/ # multi-input baselines (IID + LODO)
│     ├─ Experiment - 2 (MMASD)/            # MMASD-only tasks
│     ├─ Experiment - 3 (Engagnition)/      # Engagnition-only tasks (+ enrichment)
│     └─ Experiment - 4 (LODO)/             # unified LODO intensity pipeline
├─ scripts/
│  ├─ requirements.txt
│  └─ requirements.train.txt                # optional: training-only
├─ LICENSE
└─ README.md


Installation (Python 3.10)
python -m venv .venv
# Windows:
.venv\Scripts\activate

python -m pip install -U pip
pip install -r requirements.txt

Key dependencies (see requirements.txt for versions):
pandas, numpy, scikit-learn, xgboost (optional), matplotlib, tqdm, openpyxl, XlsxWriter, orjson, pyyaml, jupyter.

## 🔁 End‑to‑End Workflow (6 Steps)

1. **Harmonize IDs & rows** → one row per sample (MMASD clip or Engagnition session).  
2. **Compute movement intensity** per dataset:  
   - MMASD → from skeleton/optical‑flow, produce `movement_intensity_raw`.  
   - Engagnition → from E4 ACC (SVM), produce `movement_intensity_raw`.  
   Then: robust z‑score within participant → `movement_intensity_bin`.
3. **Unify** into `metadata_master.*` with provenance (`path_*` columns).  
4. **Create splits**: GroupKFold for **IID** and **LODO** tags using `participant_id_global`.  
5. **Train & evaluate** interpretable models (e.g., Logistic Regression). Save metrics to CSV.  
6. **Freeze** outputs and schema in `frozen/` to guarantee reproducibility.

![Flowchart (1)](https://github.com/user-attachments/assets/08e82f4a-c25c-4f96-88b8-b1ceade62057)

---

## 🧬 Data Schema (Essentials)

- **Keys:** `sample_id`, `participant_id_global`, `dataset`, `activity/condition`  
- **Targets:** `movement_intensity_raw`, `movement_intensity_z`, `movement_intensity_bin`  
- **Demographics:** `sex`, `age_years` / `age_group`  
- **Splits:** `split_seed`, `split_iid`, `split_lodo`, `group_kfold`  
- **Provenance:** `source_file`, `path_*` to derived artifacts  

See `docs/metadata_schema.md` for the authoritative, typed specification.

---

## 🧊 Frozen Snapshots

**`frozen/v1_2025-09-13/`** is an immutable snapshot (schema, manifests, ML‑ready tables, splits).  
**Rule:** never overwrite it; future revisions go into a new versioned folder.

---

## 🧪 Experiments & Outputs

- **Exp‑1 - General Baseline:** overall baselines and sanity checks.  
- **Exp‑2 - MMASD:** intra‑dataset experiments and feature variants.  
- **Exp‑3 - Engagnition:** same for Engagnition + feature enrichment.  
- **Exp‑4 - LODO:** cross‑dataset transfer (train ↔ test).  

Each script writes **metrics CSVs** under its `outputs/` subfolder.

---

## ⚖️ Intended Use

- **Use this if** you need a transparent, reproducible **baseline** for portability and privacy‑by‑design screening signals.  
- **Do not use as** a clinical ASD vs TD diagnostic tool. The current setup relies on a **proxy outcome** for cross‑dataset comparability.

---

## 📚 Cite & License

Please cite the original datasets when using this benchmark:

- **MMASD** - Li *et al.* *A Multimodal Dataset for Autism Intervention Analysis* (ICMI 2023).  
- **Engagnition** - Kim *et al.* *Engagnition: multi‑dimensional dataset for engagement recognition of children with ASD* (*Scientific Data*, 2024).

Code/text license: see `LICENSE`.

---

## 🤝 Contributing & Support

- Open an issue for bugs/questions.  
- PRs are welcome - follow folder conventions and **do not modify `frozen/`**.  
- For data access questions, start with `docs/datasets.md`.

---

## Reporting Issues
To report an issue please use the issues page (https://github.com/omicscodeathon/autdb/issues). Please check existing issues before submitting a new one.

## Contribute to the Project
You can offer to help with the further development of this project by making pull requests on this repo. To do so, fork this repository and make the proposed changes. Once completed and tested, submit a pull request to this repo.

## Team

Ruslan Kurmashev, Munster Technological University, Dublin, Ireland (Developer - Writer).

Adina Yessimova, City Colleges in Dublin, Dublin, Ireland (Writer).

Denis Traore, Université Nazi Boni, Burkina Faso (Writer).

Olaitan I. Awe, Ph.D., Institute for Genomic Medicine Research (IGMR), United States. (Supervisor)

