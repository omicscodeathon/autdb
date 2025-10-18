#Step 1 - Reproduce MMASD preprocessing pipeline

This folder contains ready-to-run Python scripts that reproduce the MMASD preprocessing workflow
as described in the manuscript.

Folder structure:

autdb/
├─ data/
│  ├─ Code for preparing tables/
│  │   ├─ build_mmasd_basic.py
│  │   ├─ compute_mmasd_features.py
│  │   └─ (output: MMASD basic.xlsx)
│  └─ MMASD/
│      ├─ 2D skeleton/
│      │   └─ output/...
│      └─ ADOS_rating.xlsx

Commands (Windows CMD example)

Run these commands from inside
autdb\data\Code for preparing tables\:

Step 1. Build base MMASD table
py -3.10 build_mmasd_basic.py ^
  --root "..\MMASD" ^
  --ados "..\MMASD\ADOS_rating.xlsx" ^
  --out  "."

Step 2. Compute movement_intensity features
py -3.10 compute_mmasd_features.py ^
  --root "..\MMASD" ^
  --meta "MMASD basic.xlsx"

Output

After successful execution, the resulting file will appear in the same folder:

autdb\data\Code for preparing tables\MMASD basic.xlsx


This table contains:

participant metadata (ID, sex, age)

activity annotations extracted from folder names

movement metrics calculated from OpenPose skeletons:

movement_intensity_raw – median joint Euclidean displacement

movement_intensity_z – normalized per participant (median/IQR)

movement_intensity_bin – binary proxy outcome (1 = above-median movement)

Reproducibility note

Running these two commands will recreate exactly the same table (MMASD basic.xlsx)
that was used in the manuscript for downstream modeling and analysis.
No manual intervention or local path changes are required as long as the folder structure matches the one shown above.


#Step 2 – Reproduce Engagnition preprocessing pipeline

This folder contains ready-to-run Python scripts that reproduce the Engagnition preprocessing workflow
as described in the manuscript.

Folder structure:

autdb/
├─ data/
│  ├─ Code for preparing tables/
│  │   ├─ build_engagnition_basic.py
│  │   ├─ compute_eng_features.py
│  │   └─ (output: Engagnition basic.xlsx)
│  └─ Engagnition/
│      ├─ Baseline condition/…/E4AccData.csv
│      ├─ LPE condition/…/E4AccData.csv
│      ├─ HPE condition/…/E4AccData.csv
│      ├─ E4GsrData.csv
│      ├─ E4TmpData.csv
│      ├─ Subjective questionnaire.xlsx
│      ├─ InterventionData.xlsx
│      └─ Session Elapsed Time.xlsx


Commands (Windows CMD example)

Run these commands from inside
autdb\data\Code for preparing tables\:

Step 1. Build base Engagnition table
py -3.10 build_engagnition_basic.py ^
  --data-root ".." ^
  --out-xlsx "Engagnition basic.xlsx"

Step 2. Compute movement_intensity features
py -3.10 compute_eng_features.py ^
  --data-root ".." ^
  --meta "Engagnition basic.xlsx"


Output

After successful execution, the resulting file will appear in the same folder:

autdb\data\Code for preparing tables\Engagnition basic.xlsx


This table contains:

participant/session metadata (from questionnaires & session logs)

aggregated Empatica E4 signals (ACC, GSR, TEMP)

movement metrics derived from ACC:

movement_intensity_raw – median vector magnitude (SVM) over the session

movement_intensity_z – normalized per participant/condition (median/IQR with robust fallback)

movement_intensity_bin – binary proxy outcome (1 = above-median movement)

(optional, if annotations provided in your setup) engagement labels propagated to session/block rows

Reproducibility note

Running these two commands will recreate exactly the same table (Engagnition basic.xlsx)
that was used in the manuscript for downstream modeling and analysis.
No manual intervention or local path changes are required as long as your folder structure matches the one shown above.


#Step 3 – Merge MMASD and Engagnition tables

This step combines the preprocessed feature tables from both datasets into a single unified metadata file used for downstream cleaning, schema generation, and train/test splits.

Folder structure
autdb/
├─ data/
│  ├─ Code for preparing tables/
│  │   ├─ merge_tables.py
│  │   ├─ MMASD basic.xlsx
│  │   ├─ Engagnition basic.xlsx
│  │   └─ (output: metadata_master.xlsx / .csv)

Commands (Windows CMD example)

Run these commands from inside
autdb\data\Code for preparing tables\:

py -3.10 merge_tables.py ^
  --mmasd "MMASD basic.xlsx" ^
  --eng   "Engagnition basic.xlsx" ^
  --out-csv  "metadata_master.csv" ^
  --out-xlsx "metadata_master.xlsx"

Or as a single-line command:

py -3.10 merge_tables.py --mmasd "MMASD basic.xlsx" --eng "Engagnition basic.xlsx" --out-csv "metadata_master.csv" --out-xlsx "metadata_master.xlsx"

Output

After successful execution, two new files will appear in the same folder:

metadata_master.csv
metadata_master.xlsx


The resulting tables contain:

Unified metadata from both datasets
(MMASD + Engagnition)

Dataset origin tags automatically added (dataset column)

Consistent feature names and harmonized column structure

Ready for subsequent cleaning (Metadata cleaning.ipynb)
and split definition (metadata_ml_ready_splits.xlsx)

Reproducibility note

Running this single command will fully reproduce the unified metadata table used in the manuscript and subsequent modeling stages.
All paths are relative, so no manual editing is required if your folder structure matches the one shown above.



