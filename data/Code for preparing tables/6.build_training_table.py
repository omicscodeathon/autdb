#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a training-ready table from the frozen metadata.

Inputs (read-only):
  - frozen metadata_ml_ready_splits.(xlsx|csv)
  - optional extra feature files (--feat), each with: sample_id + feature columns
    (parquet|csv|xlsx). They will be left-joined by sample_id.

What it does:
  1) Loads frozen table, validates required columns, keeps only rows with target.
  2) Selects safe feature set (no leakage): categorical + numeric covariates.
     By default movement_intensity_raw is EXCLUDED to avoid leakage.
     You can include it with --include-raw.
  3) Optionally merges any extra feature files by sample_id.
  4) Exports:
      - one unified file with ALL rows and both splits (iid & lodo)
      - three files train/val/test for the chosen split (iid by default)
     in Parquet + CSV (Excel optional).
  5) Prints compact summary.

Usage (Windows):
  py build_training_table.py ^
     --frozen "C:\...\data\frozen\v1_2025-09-13\metadata_ml_ready_splits.xlsx" ^
     --out-dir "C:\...\data\training\v1" ^
     --split iid
  # to include movement_intensity_raw as a feature:
  #   add: --include-raw
  # to merge extra features (can repeat --feat):
  #   --feat "C:\...\features\features_engagnition.parquet" --feat "...\features_mmasd.parquet"

Dependencies:
  pip install pandas numpy pyarrow openpyxl
"""

import os
import argparse
import pandas as pd
import numpy as np

SAFE_NUM_COLS = [
    "age_years",
    "elapsed_time_sec_total",
    "sus_total",
    "nasa_tlx_weighted",
    "nasa_tlx_unweighted",
]

SAFE_CAT_COLS = [
    "dataset",
    "condition",
    "unit_level",
    "modality",
    "sex",
    "age_group",
]

REQUIRED_COLS = [
    "sample_id",
    "participant_id",
    "movement_intensity_bin",
    "split_iid",
    "split_lodo",
]


def _load_any(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".xlsx"):
        return pd.read_excel(path, sheet_name=0, dtype="object")
    if p.endswith(".csv"):
        return pd.read_csv(path, dtype="object")
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    raise SystemExit(f"[ERR] Unsupported file type: {path}")


def build_training_table(frozen_path: str,
                         out_dir: str,
                         split_name: str = "iid",
                         include_raw: bool = False,
                         extra_features: list | None = None,
                         save_excel: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = _load_any(frozen_path)
    # validate
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise SystemExit(f"[ERR] Missing required column in frozen file: {c}")

    # keep rows with valid target
    y = pd.to_numeric(df["movement_intensity_bin"], errors="coerce")
    df = df.loc[y.notna()].copy()
    df["movement_intensity_bin"] = y.astype(int)

    # base feature set (no leakage)
    feat_cols_num = [c for c in SAFE_NUM_COLS if c in df.columns]
    feat_cols_cat = [c for c in SAFE_CAT_COLS if c in df.columns]

    if include_raw:
        if "movement_intensity_raw" in df.columns:
            feat_cols_num = ["movement_intensity_raw"] + feat_cols_num
        else:
            print("[WARN] movement_intensity_raw not found; --include-raw ignored.")

    # reduce to needed columns + splits
    keep_cols = (["sample_id", "participant_id"] +
                 feat_cols_num + feat_cols_cat +
                 ["movement_intensity_bin", "split_iid", "split_lodo"])
    keep_cols = [c for c in keep_cols if c in df.columns]
    base = df[keep_cols].copy()

    # cast numerics
    for c in feat_cols_num:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    # optional: merge extra features
    if extra_features:
        for path in extra_features:
            ft = _load_any(path)
            if "sample_id" not in ft.columns:
                raise SystemExit(f"[ERR] Extra features file has no 'sample_id': {path}")
            # avoid collisions with existing column names
            dup_cols = [c for c in ft.columns if c != "sample_id" and c in base.columns]
            if dup_cols:
                ft = ft.rename(columns={c: f"{os.path.splitext(os.path.basename(path))[0]}__{c}" for c in dup_cols})
            base = base.merge(ft, on="sample_id", how="left")
            print(f"[OK] merged features: {path} -> shape now {base.shape}")

    # export unified table
    unified_parquet = os.path.join(out_dir, "dataset_for_training.parquet")
    unified_csv     = os.path.join(out_dir, "dataset_for_training.csv")
    base.to_parquet(unified_parquet, index=False)
    base.to_csv(unified_csv, index=False, encoding="utf-8-sig")
    if save_excel:
        base.to_excel(os.path.join(out_dir, "dataset_for_training.xlsx"), index=False)

    # export split-specific files
    split_col = f"split_{split_name.lower()}"
    if split_col not in base.columns:
        raise SystemExit(f"[ERR] Split column not found: {split_col} (available: split_iid, split_lodo)")

    for part in ["train", "val", "test"]:
        part_df = base.loc[base[split_col] == part].copy()
        part_df.to_parquet(os.path.join(out_dir, f"trainset_{split_name}_{part}.parquet"), index=False)
        part_df.to_csv(os.path.join(out_dir, f"trainset_{split_name}_{part}.csv"), index=False, encoding="utf-8-sig")
        print(f"[OK] saved {part}: {len(part_df)} rows")

    # small summaries
    print("\n[SUMMARY] rows by split (", split_name, "):", sep="")
    print(base[split_col].value_counts())
    if "dataset" in base.columns:
        print("\n[SUMMARY] dataset × bin per split:")
        print(base.pivot_table(index=[split_col, "dataset"],
                               columns="movement_intensity_bin",
                               values="sample_id",
                               aggfunc="count",
                               fill_value=0))

    print(f"\n[DONE] Saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build training-ready dataset from frozen metadata.")
    ap.add_argument("--frozen", required=True, help="Path to frozen metadata_ml_ready_splits.(xlsx|csv|parquet)")
    ap.add_argument("--out-dir", required=True, help="Output directory for training files")
    ap.add_argument("--split", choices=["iid", "lodo"], default="iid", help="Which split to materialize (default: iid)")
    ap.add_argument("--include-raw", action="store_true",
                    help="Include movement_intensity_raw among numeric features (may introduce leakage).")
    ap.add_argument("--feat", action="append", default=None,
                    help="Path to extra feature file (parquet|csv|xlsx). Can be specified multiple times.")
    ap.add_argument("--excel", action="store_true", help="Also save Excel copies.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_training_table(
        frozen_path=args.frozen,
        out_dir=args.out_dir,
        split_name=args.split,
        include_raw=args.include_raw,
        extra_features=args.feat,
        save_excel=args.excel,
    )
