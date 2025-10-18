#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge two Excel bases into a single master table WITHOUT losing any rows or columns.

Input:
  - MMASD basic.xlsx        (sheet: 'MMASD basic')
  - Engagnition basic.xlsx  (sheet: 'Engagnition basic')

Rules:
  - Keep ALL rows from BOTH files (NO deduplication).
  - Keep ALL columns from BOTH files (full union).
  - Preserve values exactly as they are (read as dtype=object; no coercion).
  - If a column is missing in one source, fill with NA for that source.
  - If a 'dataset' column is missing in a source, set it to the corresponding dataset name.

Output:
  - metadata_master.csv   (UTF-8-SIG)
  - metadata_master.xlsx  (sheet: 'metadata_master')
  - Prints a short summary and the final line: metadata_master

Usage (Windows CMD/PowerShell):
  python merge_master.py ^
    --mmasd "C:\\...\\MMASD basic.xlsx" ^
    --eng   "C:\\...\\Engagnition basic.xlsx" ^
    --out-csv  "C:\\...\\metadata_master.csv" ^
    --out-xlsx "C:\\...\\metadata_master.xlsx"

Dependencies:
  pip install pandas openpyxl xlsxwriter
"""

import os
import argparse
import pandas as pd

MM_SHEET  = "MMASD basic"
ENG_SHEET = "Engagnition basic"

# A canonical ordering to make the merged file predictable (all extras are appended at the end)
CANONICAL_ORDER = [
    # identifiers & scope
    "dataset", "sample_id", "unit_level", "modality",
    "participant_id", "condition", "source_dir",
    # MMASD compatibility
    "activity_prefix", "activity_class", "rel_path_openpose",
    # Engagnition paths & presence flags
    "rel_path_acc", "rel_path_gsr", "rel_path_tmp",
    "rel_path_engagement", "rel_path_gaze", "rel_path_performance",
    "has_acc", "has_gsr", "has_tmp", "has_engagement", "has_gaze", "has_performance",
    # blocks
    "block_field", "block_id",
    # targets / features
    "engagement_level",
    "movement_intensity_raw", "movement_intensity_z", "movement_intensity_bin",
    # fairness
    "sex", "age_years", "age_group",
    # questionnaires / extras
    "intervention_type", "intervention_timestamps_raw",
    "elapsed_time_sec_total",
    "sus_total", "nasa_tlx_weighted", "nasa_tlx_unweighted",
    # splits
    "split_seed", "split_iid", "split_lodo",
]

def read_sheet_or_first(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    """Read the requested sheet; if absent, read the first sheet. Always dtype=object."""
    if not os.path.isfile(xlsx_path):
        raise SystemExit(f"[ERR] File not found: {xlsx_path}")
    try:
        return pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype="object")
    except Exception:
        # fall back to first sheet
        return pd.read_excel(xlsx_path, dtype="object")

def ensure_dataset_column(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if "dataset" not in df.columns:
        df = df.copy()
        df["dataset"] = dataset_name
    return df

def aligned_union(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Return concatenation with full union of columns; no row drops; dtype=object preserved."""
    all_cols = list(dict.fromkeys(list(df_a.columns) + list(df_b.columns)))
    df_a2 = df_a.reindex(columns=all_cols)
    df_b2 = df_b.reindex(columns=all_cols)
    out = pd.concat([df_a2, df_b2], ignore_index=True)
    return out, all_cols

def reorder_columns(df: pd.DataFrame, all_cols: list) -> pd.DataFrame:
    """Place canonical columns first; keep every other column afterwards (original order preserved)."""
    canon = [c for c in CANONICAL_ORDER if c in all_cols]
    tail  = [c for c in all_cols if c not in canon]
    return df.reindex(columns=canon + tail)

def main():
    ap = argparse.ArgumentParser(description="Merge 'MMASD basic.xlsx' and 'Engagnition basic.xlsx' into a master table.")
    ap.add_argument("--mmasd", required=True, help="Path to 'MMASD basic.xlsx'.")
    ap.add_argument("--eng",   required=True, help="Path to 'Engagnition basic.xlsx'.")
    ap.add_argument("--out-csv",  required=True, help="Path to save 'metadata_master.csv'.")
    ap.add_argument("--out-xlsx", required=True, help="Path to save 'metadata_master.xlsx'.")
    args = ap.parse_args()

    # 1) Read both sources (dtype=object)
    df_mm  = read_sheet_or_first(args.mmasd, MM_SHEET)
    df_eng = read_sheet_or_first(args.eng,   ENG_SHEET)

    # 2) Ensure dataset labels if missing
    df_mm  = ensure_dataset_column(df_mm,  "MMASD")
    df_eng = ensure_dataset_column(df_eng, "Engagnition")

    # 3) Align columns and concatenate
    merged, all_cols = aligned_union(df_mm, df_eng)

    # 4) Reorder columns for readability
    merged = reorder_columns(merged, all_cols)

    # 5) Save CSV (UTF-8-SIG)
    out_csv  = os.path.abspath(args.out_csv)
    out_xlsx = os.path.abspath(args.out_xlsx)
    os.makedirs(os.path.dirname(out_csv),  exist_ok=True)
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)

    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved CSV:  {out_csv} (rows={len(merged)}, cols={len(merged.columns)})")

    # 6) Save XLSX
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xlw:
        merged.to_excel(xlw, index=False, sheet_name="metadata_master")
    print(f"[OK] Saved XLSX: {out_xlsx} (rows={len(merged)}, cols={len(merged.columns)})")

    # 7) Diagnostics (optional but helpful)
    print("[INFO] Source sizes:")
    print(f"  MMASD rows:        {len(df_mm)}")
    print(f"  Engagnition rows:  {len(df_eng)}")
    only_mm_cols  = [c for c in df_mm.columns  if c not in df_eng.columns]
    only_eng_cols = [c for c in df_eng.columns if c not in df_mm.columns]
    print(f"[INFO] Columns only in MMASD ({len(only_mm_cols)}): {only_mm_cols}")
    print(f"[INFO] Columns only in Engagnition ({len(only_eng_cols)}): {only_eng_cols}")

    # Final marker
    print("metadata_master")

if __name__ == "__main__":
    main()
