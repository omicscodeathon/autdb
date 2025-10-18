AutDB-Video — Frozen snapshot v1_2025-09-13


This folder is a frozen snapshot of the tabular data and support files used in AutDB-Video experiments. A frozen snapshot is stored as-is: do not edit files in place. Any updates should be made by creating a new versioned folder.

Folder purpose

Preserve an exact, reproducible checkpoint for experiments and reports.

Provide a single location for:

the ML-ready metadata table,

the column schema,

the split manifest (how data were partitioned),

and the helper script that adds a cross-dataset participant ID and produces withGlobalID files.


Contents — what each file is for

metadata_ml_ready_splits.xlsx — source curated table for ML.
One row ≈ one clip/sample. Key columns typically include:

dataset — data source (e.g., MMASD, Engagnition);

sample_id — internal sample identifier;

split/fold columns and other engineered features.

schema.yaml — column definitions for the metadata table: name, type, short description. Used for validation and auto-documentation.

splits_manifest.json — canonical split manifest (train/val/test and/or folds) followed by the experiments. Treat this as the source of truth for splits for this version.

add_global_id.py — utility that creates the cross-dataset participant identifier participant_id_global and emits the “enriched” files listed below. Its matching logic can be adapted if ID formats change.

metadata_ml_ready_splits_withGlobalID.xlsx — derived file: metadata_ml_ready_splits.xlsx plus the new participant_id_global column.
Use it when you need to align/merge records across datasets without exposing PII.

schema_withGlobalID.yaml — same schema as above, augmented with the participant_id_global field so tools/validation are aware of it.

splits_manifest_withGlobalID.json — copy of the split manifest with the participant_id_global field appended (if the manifest defines a fields list). Keeps schema and manifest consistent.


What is participant_id_global.

Purpose: a human-readable, cross-dataset key that allows you to track and join the same participant across tables and datasets.

Use cases: joins/merges, stratification, deduplication, participant-level reporting.

Privacy: derived from de-identified IDs; contains no PII.

Versioning & reproducibility

Each vYYYY-MM-DD folder is immutable. Any new columns/fixes → a new versioned folder.

Downstream code should reference a specific version of the frozen snapshot.

For transparent reporting, keep both the original and the withGlobalID files together in the same versioned folder.
