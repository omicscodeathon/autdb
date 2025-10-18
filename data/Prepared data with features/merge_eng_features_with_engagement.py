#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_eng_features_with_engagement.py

Goal:
Left-join engagement_level (0/1/2) from 'Engagnition basic.xlsx'
into your features table 'Engagnition_features.xlsx'.
"""

import os
import argparse
import numpy as np
import pandas as pd

def norm_cond(x: str) -> str:
    if not isinstance(x, str):
        return x
    n = x.strip().lower()
    if "hpe" in n: return "HPE"
    if "lpe" in n: return "LPE"
    if "base" in n: return "Baseline"
    return x.strip()

def coerce_int012(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    v = v.where(v.isin([0,1,2]), np.nan)
    return v.astype("Int64")

def read_excel_any(path: str, sheet=None) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=(sheet if sheet is not None else 0), dtype="object")

def main(features_path, basic_path, out_path, basic_sheet="Engagnition basic"):
    if not os.path.isfile(features_path):
        raise SystemExit(f"[ERR] Features file not found:\n{features_path}")
    if not os.path.isfile(basic_path):
        raise SystemExit(f"[ERR] Basic file not found:\n{basic_path}")

    print("[INFO] Reading features …")
    feats = read_excel_any(features_path)
    feats.columns = [str(c) for c in feats.columns]

    # ensure keys exist
    for c in ["sample_id","participant_id","condition"]:
        if c not in feats.columns:
            feats[c] = pd.NA

    # normalize keys
    feats["sample_id"]      = feats["sample_id"].astype(str).str.strip()
    feats["participant_id"] = feats["participant_id"].astype(str).str.strip()
    feats["condition"]      = feats["condition"].map(norm_cond)

    # if engagement_level already present -> stash it
    had_prev = False
    if "engagement_level" in feats.columns:
        feats = feats.rename(columns={"engagement_level": "engagement_level_prev"})
        had_prev = True

    print("[INFO] Reading Engagnition basic …")
    base = pd.read_excel(basic_path, sheet_name=basic_sheet, dtype="object")
    base.columns = [str(c) for c in base.columns]
    for c in ["dataset","unit_level","sample_id","participant_id","condition","engagement_level"]:
        if c not in base.columns:
            base[c] = pd.NA

    # only Engagnition sessions
    mask = (base["dataset"] == "Engagnition") & (base["unit_level"] == "session")
    base = base.loc[mask, ["sample_id","participant_id","condition","engagement_level"]].copy()
    base["sample_id"]      = base["sample_id"].astype(str).str.strip()
    base["participant_id"] = base["participant_id"].astype(str).str.strip()
    base["condition"]      = base["condition"].map(norm_cond)
    base["engagement_level"] = coerce_int012(base["engagement_level"])

    print(f"[INFO] Features rows: {len(feats)}, Basic(session) rows: {len(base)}")

    #1) join by sample_id (use drop_duplicates to avoid fanout)
    base_sid = base.dropna(subset=["sample_id"]).copy()
    base_sid = base_sid[["sample_id","engagement_level"]].drop_duplicates(subset=["sample_id"], keep="last")

    merged = feats.merge(
        base_sid,
        how="left", on="sample_id"
    )

    #2) fallback by (participant_id, condition)
    need = merged["engagement_level"].isna()
    if need.any():
        base_pc = base.dropna(subset=["participant_id","condition"]).copy()
        base_pc = base_pc[["participant_id","condition","engagement_level"]].drop_duplicates(
            subset=["participant_id","condition"], keep="last"
        )
        merged = merged.merge(
            base_pc.rename(columns={"engagement_level":"engagement_level_fb"}),
            how="left", on=["participant_id","condition"]
        )
        merged["engagement_level"] = merged["engagement_level"].fillna(merged["engagement_level_fb"])
        merged = merged.drop(columns=["engagement_level_fb"])
        
    if had_prev:
        merged["engagement_level_prev"] = coerce_int012(merged["engagement_level_prev"])
        merged["engagement_level"] = merged["engagement_level"].fillna(merged["engagement_level_prev"])
        merged = merged.drop(columns=["engagement_level_prev"])

    merged["engagement_level"] = coerce_int012(merged["engagement_level"])

    # Summary
    total = len(merged)
    n_ok = merged["engagement_level"].notna().sum()
    n_nan = total - n_ok
    vc = merged["engagement_level"].value_counts(dropna=False).sort_index()

    print(f"[OK] Attached engagement_level. Total: {total} | with values: {n_ok} | NaN: {n_nan}")
    try:
        print("[OK] engagement_level distribution:\n" + vc.to_string())
    except Exception:
        print("[OK] engagement_level distribution:", vc)

    # Save
    out_path = out_path or os.path.join(os.path.dirname(features_path),
                                        "Engagnition_features_with_engagement.xlsx")
    out_path = os.path.abspath(out_path)
    with pd.ExcelWriter(out_path, engine="xlsxwriter", mode="w") as xlw:
        merged.to_excel(xlw, index=False, sheet_name="Engagnition features+eng")
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to 'Engagnition_features.xlsx'")
    ap.add_argument("--basic",    required=True, help="Path to 'Engagnition basic.xlsx'")
    ap.add_argument("--out",      default=None,  help="Output Excel path (optional)")
    ap.add_argument("--basic-sheet", default="Engagnition basic", help="Sheet name in the basic workbook")
    args = ap.parse_args()
    main(args.features, args.basic, args.out, args.basic_sheet)