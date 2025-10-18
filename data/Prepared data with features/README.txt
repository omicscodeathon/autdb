Step 1 - Reproduce Engagnition ACC feature table

This folder contains a ready-to-run Python script that builds the Engagnition ACC features table from Empatica E4 accelerometer files (E4AccData.csv) using the session list in Engagnition basic.xlsx.

Folder structure
autdb/
├─ data/
│  ├─ Code for preparing tables/
│  │   ├─ build_eng_features.py
│  │   └─ (output: Engagnition_features.xlsx)
│  └─ Engagnition/
│      ├─ Baseline condition/<Pxx>/E4AccData.csv
│      ├─ LPE condition/<Pxx>/E4AccData.csv
│      └─ HPE condition/<Pxx>/E4AccData.csv
└─ (the file "Engagnition basic.xlsx" is in: autdb/data/Code for preparing tables/)


Reads the session list from Engagnition basic.xlsx (expects columns participant_id and condition).

For each session, loads E4AccData.csv from
Engagnition/{condition} condition/{participant_id}/E4AccData.csv.

Computes ACC magnitude features (from x,y,z):
acc_median, acc_iqr, acc_p75, acc_std, acc_mad, acc_max, acc_var, acc_energy, acc_autocorr_lag1, acc_high_fraction, acc_duration_s (ACC assumed 32 Hz).

Saves the combined table to Engagnition_features.xlsx.

Commands (Windows CMD example)

Run these commands from inside:

autdb\data\Code for preparing tables\

py -3.10 build_eng_features.py ^
  --data "Engagnition basic.xlsx" ^
  --out  "Engagnition_features.xlsx" ^
  --root "..\Engagnition"

Output

After successful execution, the resulting file will appear in the same folder:

autdb\data\Code for preparing tables\Engagnition_features.xlsx

This table contains: all columns from Engagnition basic.xlsx plus the computed ACC feature columns listed above.

Reproducibility note

As long as your folder layout matches the structure shown here, the command will recreate exactly the same Engagnition_features.xlsx used in the experiments—no manual path edits needed.





Step 2 - Merge Engagnition ACC features with engagement labels

This script left-joins engagement_level (0/1/2) from Engagnition basic.xlsx into your features table Engagnition_features.xlsx. It first matches by sample_id, then falls back to a (participant_id, condition) match if needed. Output is an Excel file with the engagement column attached.

Folder structure
autdb/
├─ data/
│  ├─ Code for preparing tables/
│  │   ├─ build_eng_features.py
│  │   ├─ merge_eng_features_with_engagement.py
│  │   ├─ Engagnition basic.xlsx
│  │   └─ Engagnition_features.xlsx
│  └─ Engagnition/...


Reads features from Engagnition_features.xlsx and sessions from Engagnition basic.xlsx (sheet “Engagnition basic”).

Normalizes condition names to Baseline / LPE / HPE; ensures keys sample_id, participant_id, condition exist.

Joins engagement_level via:

sample_id (deduplicated), then

fallback on (participant_id, condition) if still missing.

Coerces engagement_level to Int64 with allowed values {0,1,2}; if a prior engagement_level column exists in features, it is respected as a fallback.

Saves to Engagnition_features_with_engagement.xlsx (or a custom path via --out).

Commands (Windows CMD example)

Run these commands from inside:

autdb\data\Code for preparing tables\

py -3.10 merge_eng_features_with_engagement.py ^
  --features "Engagnition_features.xlsx" ^
  --basic    "Engagnition basic.xlsx"

After successful execution, the resulting file will appear in the same folder.

autdb\data\Code for preparing tables\Engagnition_features_with_engagement.xlsx

This table contains: all columns from Engagnition_features.xlsx plus the appended engagement_level (0/1/2).

Reproducibility note

As long as Engagnition_features.xlsx and Engagnition basic.xlsx are present in this folder and follow the expected keys (sample_id, participant_id, condition), running the command above will deterministically recreate the merged table used in analysis.




Step 3 - Compute MMASD 2D-skeleton features (fast)

This script scans OpenPose 2D skeleton JSONs in MMASD/2D skeleton/output/, picks the best-confidence person per frame, computes per-clip joint-speed statistics, and saves a single feature table to Excel. It uses orjson for fast I/O and parallel workers for speed.

Folder structure
autdb/
├─ data/
│  ├─ Code for preparing tables/
│  │   ├─ build_mmasd_features_fast.py
│  │   └─ (output: MMASD_features.xlsx)
│  └─ MMASD/
│      └─ 2D skeleton/
│          └─ output/
│              ├─ <activity_A>/
│              │   ├─ <clip_id>/frame_000000_keypoints.json
│              │   └─ ...
│              └─ <activity_B>/...



Loads each clip’s *_keypoints.json, selects the person with max summed keypoint confidence, and keeps the (x,y) coordinates.

Computes per-frame joint L2 displacement → mean across 25 joints → a speed series per clip.

Aggregates clip-level features (with default fps=30):
skel_median, skel_iqr, skel_p75, skel_std, skel_mad, skel_max, skel_var, skel_high_fraction (fraction > median), skel_duration_s (N/fps).

Writes one row per clip with activity_class, clip_id, and the features to MMASD_features.xlsx.

Commands (Windows CMD example)

Run these commands from inside:

autdb\data\Code for preparing tables\


Default run

py -3.10 build_mmasd_features_fast.py ^
  --root "..\MMASD\2D skeleton\output" ^
  --out  "MMASD_features.xlsx"

Output

After successful execution, the resulting file will appear in the same folder:

autdb\data\Code for preparing tables\MMASD_features.xlsx


This table contains: activity_class, clip_id, and all skel_* features listed above.




Step 4 - Merge MMASD basic table with skeleton features

This script combines the MMASD subject/activity metadata with the per-clip skeleton features into a single table.
It performs an outer merge on sample_id (from the basic table) and clip_id (from the features table), then writes MMASD_merged.xlsx and prints a short merge report.

Folder structure
autdb/
├─ data/
│  ├─ Code for preparing tables/
│  │   ├─ MMASD_merged.py
│  │   ├─ MMASD basic - clean.xlsx
│  │   └─ MMASD_features.xlsx
│  └─ MMASD/...

What the script does

Loads MMASD basic - clean.xlsx (metadata) and MMASD_features.xlsx (per-clip skeleton features).

Merges them with:

left_on="sample_id" (basic)

right_on="clip_id" (features)

how="outer" (keeps unmatched rows from both).

Applies suffixes ("_basic", "_feat") to any overlapping column names.

Saves the result to MMASD_merged.xlsx and prints counts of matched vs. unmatched rows.

Commands (Windows CMD example)

Run from inside:

autdb\data\Code for preparing tables\

py -3.10 MMASD_merged.py

Output

After successful execution, the resulting file will appear in the same folder:

autdb\data\Code for preparing tables\MMASD_merged.xlsx


This table contains: all columns from the basic metadata and the skeleton feature set, aligned by clip (sample_id ↔ clip_id). Unmatched rows from either side are retained (outer join), enabling you to spot missing features or metadata.

