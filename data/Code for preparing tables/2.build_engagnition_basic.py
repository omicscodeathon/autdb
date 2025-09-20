#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Engagnition basic rows (facts only) and save both CSV and XLSX.

What this script does:
1) Scans Engagnition folders (Baseline/LPE/HPE) and creates:
   - session × modality rows (ACC, GSR, TMP, ENG, GAZE, PERF)
   - block rows for PERF/GAZE in LPE/HPE if a block-like column is found
2) Merges ready-made values from:
   - InterventionData.xlsx        -> intervention_type, intervention_timestamps_raw
   - Session Elapsed Time.xlsx    -> elapsed_time_sec_total
   - Subjective questionnaire.xlsx-> SUS / NASA-TLX
3) Removes previous ENG_* rows from the existing metadata to avoid duplicates.
4) Appends new rows, aligns columns, saves to CSV (args.out) and also to:
      <out_dir>/Engagnition basic.xlsx   (sheet: "Engagnition basic")
5) Prints a summary and the final line "Engagnition basic".

No computations here (no movement intensity, no engagement level).
Dependencies: pandas, openpyxl, xlsxwriter
"""

import os
import argparse
import pandas as pd

CONDITION_DIRS = {
    "Baseline": "Baseline condition",
    "LPE": "LPE condition",
    "HPE": "HPE condition",
}

# (modality, filename, rel_path_col, presence_flag)
MODS = [
    ("ACC",  "E4AccData.csv",        "rel_path_acc",         "has_acc"),
    ("GSR",  "E4GsrData.csv",        "rel_path_gsr",         "has_gsr"),
    ("TMP",  "E4TmpData.csv",        "rel_path_tmp",         "has_tmp"),
    ("ENG",  "EngagementData.csv",   "rel_path_engagement",  "has_engagement"),
    ("GAZE", "GazeData.csv",         "rel_path_gaze",        "has_gaze"),
    ("PERF", "PerformanceData.csv",  "rel_path_performance", "has_performance"),
]

BLOCK_KEYS = ("block", "trial", "round", "task", "stage", "level", "segment", "game")


# helpers

def relp(path: str, start: str) -> str:
    return os.path.relpath(path, start=start).replace("/", "\\")

def norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "")

def normalize_condition(s: str) -> str:
    t = str(s).strip().lower()
    if t in ("hpe", "highphysicalengagement", "high-physical-engagement"):
        return "HPE"
    if t in ("lpe", "lowphysicalengagement", "low-physical-engagement"):
        return "LPE"
    if "base" in t:
        return "Baseline"
    return t.upper()

def find_block_column(df: pd.DataFrame):
    if df is None or df.empty:
        return None, []
    # 1) preferred by name
    for c in df.columns:
        n = norm(c)
        if any(k in n for k in BLOCK_KEYS):
            vals = pd.Series(df[c]).dropna().unique().tolist()
            if 2 <= len(vals) <= 50:
                return c, vals
    # 2) fallback: categorical-ish with 2..50 uniques
    for c in df.columns:
        vals = pd.Series(df[c]).dropna().unique().tolist()
        if 2 <= len(vals) <= 50:
            return c, vals
    return None, []

def try_read_csv(path: str):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


# builders

def build_rows_for_session_and_blocks(data_root, participant_id, cond, pdir):
    base_row = {
        "dataset": "Engagnition",
        "participant_id": participant_id,
        "condition": cond,
        "source_dir": relp(pdir, data_root),
        # placeholders (facts only, no computations here)
        "activity_class": pd.NA,
        "engagement_level": pd.NA,
        "movement_intensity_raw": pd.NA,
        "movement_intensity_z": pd.NA,
        "movement_intensity_bin": pd.NA,
        # compatibility with MMASD
        "activity_prefix": pd.NA,
        "rel_path_openpose": pd.NA,
        # fairness placeholders
        "sex": pd.NA,
        "age_years": pd.NA,
        "age_group": pd.NA,
        # split placeholders
        "split_seed": pd.NA,
        "split_iid": pd.NA,
        "split_lodo": pd.NA,
    }

    rows = []

    # 1) session × modality
    for mod_name, filename, rel_col, has_col in MODS:
        fpath = os.path.join(pdir, filename)
        exists = os.path.isfile(fpath)

        row = base_row.copy()
        row["unit_level"] = "session"
        row["modality"] = mod_name
        row["sample_id"] = f"ENG_{participant_id}_{cond}_{mod_name}"
        row[rel_col] = relp(fpath, data_root) if exists else pd.NA
        row[has_col] = int(exists)
        rows.append(row)

    # 2) block-level rows for PERF/GAZE in LPE/HPE
    if cond in ("LPE", "HPE"):
        for mod_name, filename, rel_col, has_col in MODS:
            if mod_name not in ("PERF", "GAZE"):
                continue
            fpath = os.path.join(pdir, filename)
            if not os.path.isfile(fpath):
                continue
            df = try_read_csv(fpath)
            col, uniq_vals = find_block_column(df)
            if not col:
                continue
            for v in uniq_vals:
                block_id = str(v).strip().replace(" ", "")
                brow = base_row.copy()
                brow["unit_level"] = "block"
                brow["modality"] = mod_name
                brow["block_field"] = col
                brow["block_id"] = str(v)
                brow["sample_id"] = f"ENG_{participant_id}_{cond}_{mod_name}_B{block_id}"
                brow[rel_col] = relp(fpath, data_root)
                brow[has_col] = 1
                rows.append(brow)

    return rows

def collect_rows(data_root):
    eng_root = os.path.join(data_root, "Engagnition")
    rows = []
    for cond, cond_dir in CONDITION_DIRS.items():
        base = os.path.join(eng_root, cond_dir)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if not name.upper().startswith("P"):
                continue
            pdir = os.path.join(base, name)
            if not os.path.isdir(pdir):
                continue
            participant_id = name  # e.g., "P01"
            rows.extend(build_rows_for_session_and_blocks(data_root, participant_id, cond, pdir))
    return pd.DataFrame(rows, dtype="object")


# XLSX supplements

def _find_col(cols_map, predicate):
    for orig, n in cols_map.items():
        if predicate(n):
            return orig
    return None

def load_intervention_df(xlsx_path):
    if not os.path.isfile(xlsx_path):
        return pd.DataFrame()
    df0 = pd.read_excel(xlsx_path)
    if df0.empty:
        return pd.DataFrame()
    cols_map = {c: norm(c) for c in df0.columns}
    pid_col  = _find_col(cols_map, lambda n: n.startswith("p") or "participant" in n or n in ("id","pid"))
    cond_col = _find_col(cols_map, lambda n: n.startswith("condition"))
    type_col = _find_col(cols_map, lambda n: "intervention" in n and "type" in n)
    ts_col   = _find_col(cols_map, lambda n: "timestamp" in n or "timestamps" in n or "time" in n)
    if not pid_col or not cond_col:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["participant_id"] = df0[pid_col].astype(str).str.upper().str.extract(r"(P\d+)")[0]
    out["condition"] = df0[cond_col].apply(normalize_condition)
    if type_col:
        out["intervention_type"] = df0[type_col]
    if ts_col:
        out["intervention_timestamps_raw"] = df0[ts_col].astype(str)
    return out.dropna(subset=["participant_id","condition"]).drop_duplicates(["participant_id","condition"])

def load_elapsed_df(xlsx_path):
    if not os.path.isfile(xlsx_path):
        return pd.DataFrame()
    df0 = pd.read_excel(xlsx_path)
    if df0.empty:
        return pd.DataFrame()
    cols_map = {c: norm(c) for c in df0.columns}
    pid_col  = _find_col(cols_map, lambda n: n.startswith("p") or "participant" in n or n in ("id","pid"))
    cond_col = _find_col(cols_map, lambda n: n.startswith("condition"))
    tot_col  = _find_col(cols_map, lambda n: "totalsec" in n or "elapsed" in n or "durationsec" in n or "totaltime" in n)
    if not pid_col or not cond_col or not tot_col:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["participant_id"] = df0[pid_col].astype(str).str.upper().str.extract(r"(P\d+)")[0]
    out["condition"] = df0[cond_col].apply(normalize_condition)
    out["elapsed_time_sec_total"] = df0[tot_col]
    return out.dropna(subset=["participant_id","condition"]).drop_duplicates(["participant_id","condition"])

def load_questionnaire_df(xlsx_path):
    if not os.path.isfile(xlsx_path):
        return pd.DataFrame()
    df0 = pd.read_excel(xlsx_path)
    if df0.empty:
        return pd.DataFrame()
    cols_map = {c: norm(c) for c in df0.columns}
    pid_col  = _find_col(cols_map, lambda n: n.startswith("p") or "participant" in n or n in ("id","pid"))
    cond_col = _find_col(cols_map, lambda n: n.startswith("condition"))
    sus_col  = _find_col(cols_map, lambda n: "sus" in n and "total" in n)
    nasa_w   = _find_col(cols_map, lambda n: "nasa" in n and "weighted" in n)
    nasa_u   = _find_col(cols_map, lambda n: "nasa" in n and ("unweighted" in n or "raw" in n))
    if not pid_col or not cond_col:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["participant_id"] = df0[pid_col].astype(str).str.upper().str.extract(r"(P\d+)")[0]
    out["condition"] = df0[cond_col].apply(normalize_condition)
    if sus_col:
        out["sus_total"] = df0[sus_col]
    if nasa_w:
        out["nasa_tlx_weighted"] = df0[nasa_w]
    if nasa_u:
        out["nasa_tlx_unweighted"] = df0[nasa_u]
    return out.dropna(subset=["participant_id","condition"]).drop_duplicates(["participant_id","condition"])


# main

def main():
    ap = argparse.ArgumentParser(description="Build Engagnition basic table.")
    ap.add_argument("--data-root", required=True, help="Path to dataset root (contains Engagnition/...).")
    ap.add_argument("--out", required=True, help="CSV file to save output metadata (e.g., metadata_master.csv).")
    args = ap.parse_args()

    # Load existing metadata if present
    if os.path.isfile(args.out):
        meta = pd.read_csv(args.out, dtype="object")
    else:
        meta = pd.DataFrame(dtype="object")

    # Collect Engagnition rows
    eng = collect_rows(args.data_root)
    if eng.empty:
        print("[WARN] Engagnition not found.")
        return

    # Merge XLSX extras
    base = os.path.join(args.data_root, "Engagnition")
    for extra in [
        load_intervention_df(os.path.join(base, "InterventionData.xlsx")),
        load_elapsed_df(os.path.join(base, "Session Elapsed Time.xlsx")),
        load_questionnaire_df(os.path.join(base, "Subjective questionnaire.xlsx")),
    ]:
        if not extra.empty:
            eng = eng.merge(extra, on=["participant_id", "condition"], how="left")

    # Drop old ENG_* rows
    if "sample_id" in meta.columns:
        before = len(meta)
        meta = meta[~meta["sample_id"].astype(str).str.startswith("ENG_")]
        removed = before - len(meta)
        if removed:
            print(f"[INFO] Removed old ENG rows: {removed}")

    # Align columns and concatenate
    all_cols = list(dict.fromkeys(list(meta.columns) + list(eng.columns)))
    meta = meta.reindex(columns=all_cols)
    eng  = eng.reindex(columns=all_cols)
    out_df = pd.concat([meta, eng], ignore_index=True)

    # Save CSV
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] Added ENG rows: {len(eng)}; total rows: {len(out_df)}")
    print(f"[OK] Saved CSV: {args.out}")

    # Save Excel with fixed name "Engagnition basic.xlsx"
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    xlsx_path = os.path.join(out_dir, "Engagnition basic.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xlw:
            out_df.to_excel(xlw, index=False, sheet_name="Engagnition basic")
        print(f"[OK] Saved Excel: {xlsx_path}")
    except Exception as e:
        raise SystemExit(f"[ERR] Failed to write Excel: {xlsx_path}\n{e}")

    # Final marker
    print("Engagnition basic")


if __name__ == "__main__":
    main()
