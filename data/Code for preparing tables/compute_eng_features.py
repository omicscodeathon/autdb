#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_eng_features.py

Engagnition-only enrichment:
1) Compute movement_intensity_raw/z/bin from ACC CSVs into 'Engagnition basic.xlsx'.
2) Add engagement_level (0/1/2) per session by aggregating external annotations (mode).

This script DOES NOT touch MMASD.

Input:
  --meta: path to 'Engagnition basic.xlsx' (sheet default 'Engagnition basic')
  --data-root: root folder to resolve relative ACC paths (rel_path_acc)
  --annots: (optional) path to a single CSV/XLSX file with annotations OR a folder with multiple CSV/XLSX files
  --annots-sheet: (optional) sheet name if annotations are in Excel
  --propagate-blocks: also propagate z/bin and engagement_level to ACC block rows (same participant_id + condition)

How engagement_level is derived:
- We read timewise annotations and aggregate to session level by MODE.
- Matching priority:
    1) by 'sample_id' if present in BOTH metadata and annotations;
    2) else by ('participant_id','condition') if present in BOTH;
    3) else by 'participant_id' only (fallback, warn).
- Accepted engagement column names: ['engagement_level','engagement','level','engage','involvement'].
- Values coerced to integers {0,1,2} (others ignored as NaN).

Dependencies:
  pip install pandas numpy openpyxl xlsxwriter tqdm

Tested on Python 3.10 (Windows).
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# small helpers

def norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

def find_col(cols, predicate):
    for c in cols:
        if predicate(norm(str(c))):
            return c
    return None

def read_acc_csv(csv_path: str) -> pd.DataFrame:
    """Read ACC CSV robustly (engine='python' helps with odd separators)."""
    return pd.read_csv(csv_path, engine="python")

def get_svm_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series with SVM per row.
    If an SVM-like column exists -> use it; otherwise compute from XYZ.
    """
    cols = list(df.columns)

    # 1) Ready SVM-like column
    svm_col = find_col(cols, lambda n: "svm" in n or "vectormagnitude" in n or n.endswith("magnitude"))
    if svm_col:
        return pd.to_numeric(df[svm_col], errors="coerce")

    # 2) XYZ variants
    x_col = find_col(cols, lambda n: n in ("x", "accx") or n.endswith("accx"))
    y_col = find_col(cols, lambda n: n in ("y", "accy") or n.endswith("accy"))
    z_col = find_col(cols, lambda n: n in ("z", "accz") or n.endswith("accz"))

    # fallback: any suffix x/y/z
    if not x_col: x_col = find_col(cols, lambda n: n.endswith("x"))
    if not y_col: y_col = find_col(cols, lambda n: n.endswith("y"))
    if not z_col: z_col = find_col(cols, lambda n: n.endswith("z"))

    if not (x_col and y_col and z_col):
        raise ValueError("ACC CSV has no SVM and cannot locate X/Y/Z columns.")

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    z = pd.to_numeric(df[z_col], errors="coerce")
    return np.sqrt(x**2 + y**2 + z**2)

def robust_center_scale(series: pd.Series):
    """Return (median, IQR) for numeric series (NaNs ignored)."""
    v = pd.to_numeric(series, errors="coerce").dropna().values
    if v.size == 0:
        return np.nan, np.nan
    med = float(np.median(v))
    q25, q75 = np.percentile(v, 25), np.percentile(v, 75)
    return med, float(q75 - q25)

def _save_excel(df: pd.DataFrame, out_path: str, sheet_name: str):
    """Write a single-sheet Excel with the given sheet name."""
    out_path = os.path.abspath(out_path)
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter", mode="w") as xlw:
            df.to_excel(xlw, index=False, sheet_name=sheet_name)
        print(f"[OK] Saved Excel: {out_path}")
    except Exception as e:
        raise SystemExit(f"[ERR] Failed to write Excel: {out_path}\n{e}")

# annotations I/O

ENG_COL_CANDIDATES = ["engagement_level", "engagement", "level", "engage", "involvement"]

def _read_one_table(path: str, sheet: str|None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        df = pd.read_csv(path, engine="python", dtype="object")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=(sheet or 0), dtype="object")
    else:
        raise ValueError(f"Unsupported annotations file type: {path}")
    # Normalize colnames once
    df.columns = [str(c) for c in df.columns]
    return df

def load_annotations(annots_path: str, annots_sheet: str | None = None) -> pd.DataFrame:
    import os, warnings
    import pandas as pd

    def _read_one_table(path: str, sheet):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".txt"):
            return pd.read_csv(path, engine="python", dtype="object")
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path, sheet_name=(sheet or 0), dtype="object")
        else:
            raise ValueError(f"Unsupported annotations file type: {path}")

    if not annots_path or not os.path.exists(annots_path):
        warnings.warn(f"[WARN] Annotations path not found: {annots_path}")
        return pd.DataFrame()

    files = []
    if os.path.isfile(annots_path):
        files = [annots_path]
    else:
        # recursively collect only EngagementData.* files
        for root, _, names in os.walk(annots_path):
            for n in names:
                if n.lower().startswith("engagementdata.") and os.path.splitext(n)[1].lower() in (".csv", ".xlsx", ".xls"):
                    files.append(os.path.join(root, n))

    if not files:
        warnings.warn("[WARN] No EngagementData files found recursively.")
        return pd.DataFrame()

    tabs = []
    for p in files:
        try:
            t = _read_one_table(p, annots_sheet)

            # if the file does not contain participant_id/condition, try to extract from the path
            prts = os.path.normpath(p).split(os.sep)
            # ...\Engagnition\<COND folder>\<PXX>\EngagementData.csv
            cond = next((x for x in prts if x.lower().endswith("condition")), None)  # e.g. 'HPE condition'
            pid  = next((x for x in prts if x.lower().startswith("p") and x[1:].isdigit()), None)

            if "participant_id" not in t.columns and pid:
                t["participant_id"] = pid

            if "condition" not in t.columns and cond:
                c = cond.lower()
                if "hpe" in c:
                    c = "HPE"
                elif "lpe" in c:
                    c = "LPE"
                elif "base" in c:
                    c = "Baseline"
                t["condition"] = c

            t["__source_file"] = p
            tabs.append(t)

        except Exception as e:
            warnings.warn(f"[WARN] Failed to read {p}: {e}")

    if not tabs:
        return pd.DataFrame()

    ann = pd.concat(tabs, ignore_index=True)
    ann.columns = [str(c).strip() for c in ann.columns]
    return ann

def pick_engagement_column(ann: pd.DataFrame) -> str|None:
    cols = list(ann.columns)
    for cand in ENG_COL_CANDIDATES:
        c = find_col(cols, lambda n, c2=cand: n == norm(c2))
        if c:
            return c
    # not found
    return None

def series_mode(values: pd.Series):
    """Return a single mode value, preferring the smallest in case of ties; NaN if empty."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return np.nan
    m = v.mode()
    if m.empty:
        return np.nan
    # Prefer the smallest to keep tie-breaking deterministic
    return float(np.min(m.values))

# core logic

def compute_eng_features(
    data_root: str,
    meta_xlsx: str,
    sheet_name: str = "Engagnition basic",
    annots_path: str | None = None,
    annots_sheet: str | None = None,
    propagate_blocks: bool = False
):
    # Guards
    if not os.path.isfile(meta_xlsx):
        raise SystemExit(f"[ERR] Excel file not found: {meta_xlsx}")

    # Read sheet (fallback to the first sheet if needed)
    try:
        df = pd.read_excel(meta_xlsx, sheet_name=sheet_name, dtype="object")
    except ValueError:
        df = pd.read_excel(meta_xlsx, dtype="object")
        # pandas doesn't give sheet back, so just keep provided name
        print(f"[WARN] Sheet '{sheet_name}' not found. Loaded the first sheet instead.")

    # Ensure required columns exist
    for col in [
        "dataset", "unit_level", "modality",
        "rel_path_acc",
        "movement_intensity_raw", "movement_intensity_z", "movement_intensity_bin",
        "sample_id", "participant_id", "condition",
        "engagement_level"
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # Filter session-level Engagnition × ACC with a valid path
    mask_sess = (
        (df.get("dataset") == "Engagnition") &
        ( (df.get("modality") == "ACC") | df.get("rel_path_acc").notna() ) &
        (df.get("unit_level", "session") == "session") &
        df.get("rel_path_acc").notna()
    )
    idx_sess = df[mask_sess].index.tolist()
    if not idx_sess:
        print("[WARN] No Engagnition session×ACC rows found in the sheet.")
        # Try to still attach engagement if annotations are provided and rows exist
        if (df.get("dataset") == "Engagnition").any() and annots_path:
            df = attach_engagement(df, annots_path, annots_sheet)
        _save_excel(df, meta_xlsx, sheet_name)
        print("Engagnition basic")
        return

    # movement_intensity_*
    raws = pd.Series(index=df.index, dtype="float64")
    ok, fail = 0, 0

    print("[INFO] Computing ACC RAW (median SVM) …")
    for i in tqdm(idx_sess, desc="ACC sessions", unit="row", ncols=96):
        rel = df.at[i, "rel_path_acc"]
        if not isinstance(rel, str) or not rel:
            fail += 1
            continue
        fcsv = os.path.join(data_root, rel)
        if not os.path.isfile(fcsv):
            fail += 1
            continue
        try:
            acc = read_acc_csv(fcsv)
            svm = get_svm_series(acc)
            rv = pd.to_numeric(svm, errors="coerce").dropna().median()
            if pd.notna(rv):
                raws.at[i] = float(rv)
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1

    df.loc[raws.index, "movement_intensity_raw"] = raws

    # Prepare for z-scaling cascade
    work = df.loc[idx_sess, ["participant_id", "condition", "movement_intensity_raw"]].copy()
    work["raw"] = pd.to_numeric(work["movement_intensity_raw"], errors="coerce")
    valid = work["raw"].notna()
    if valid.sum() == 0:
        print("[ERROR] No valid RAW values computed.")
        # Still try to attach engagement if provided
        if annots_path:
            df = attach_engagement(df, annots_path, annots_sheet)
        _save_excel(df, meta_xlsx, sheet_name)
        print("Engagnition basic")
        return

    # Global stats
    g_med, g_iqr = robust_center_scale(work.loc[valid, "raw"])

    # (pid, condition) stats
    pc_stats = {}
    for (pid, cond), sub in work.groupby(["participant_id", "condition"]):
        med, iqr = robust_center_scale(sub["raw"])
        pc_stats[(pid, cond)] = (med, iqr)

    # condition stats
    c_stats = {}
    for cond, sub in work.groupby(["condition"]):
        med, iqr = robust_center_scale(sub["raw"])
        c_stats[cond] = (med, iqr)

    # Compute Z/BIN with cascade
    z_out = pd.Series(index=df.index, dtype="float64")
    bin_out = pd.Series(index=df.index, dtype="Int64")

    for i in idx_sess:
        r = raws.at[i]
        if pd.isna(r):
            continue

        pid = df.at[i, "participant_id"]
        cond = df.at[i, "condition"]

        med, iqr = pc_stats.get((pid, cond), (np.nan, np.nan))
        if not (np.isfinite(iqr) and iqr > 0):
            med, iqr = c_stats.get(cond, (np.nan, np.nan))
        if not (np.isfinite(iqr) and iqr > 0):
            med, iqr = g_med, g_iqr

        if not (np.isfinite(iqr) and iqr > 0):
            z = r - (med if np.isfinite(med) else 0.0)  # last resort
        else:
            z = (r - med) / iqr

        z_out.at[i] = float(z)
        bin_out.at[i] = int(z >= 0.0)

    df.loc[z_out.index, "movement_intensity_z"] = z_out
    df.loc[bin_out.index, "movement_intensity_bin"] = bin_out

    # engagement_level
    if annots_path:
        df = attach_engagement(df, annots_path, annots_sheet)
    else:
        print("[WARN] No --annots provided -> engagement_level will not be added.")

    # Optional: propagate to block rows
    if propagate_blocks:
        # z/bin
        mask_block_acc = (
            (df.get("dataset") == "Engagnition") &
            ( (df.get("modality") == "ACC") | df.get("rel_path_acc").notna() ) &
            (df.get("unit_level") == "block")
        )
        blocks_acc = df[mask_block_acc].index.tolist()
        if blocks_acc:
            sess_tab = df.loc[idx_sess, ["participant_id", "condition",
                                         "movement_intensity_z", "movement_intensity_bin"]].copy()
            sess_tab["z"] = pd.to_numeric(sess_tab["movement_intensity_z"], errors="coerce")
            z_map = sess_tab.groupby(["participant_id", "condition"])["z"].median()
            for j in blocks_acc:
                pid = df.at[j, "participant_id"]
                cond = df.at[j, "condition"]
                if (pid, cond) in z_map.index:
                    z_val = z_map.loc[(pid, cond)]
                    df.at[j, "movement_intensity_z"] = z_val
                    df.at[j, "movement_intensity_bin"] = int(float(z_val) >= 0.0)

        # engagement_level
        if "engagement_level" in df.columns and df["engagement_level"].notna().any():
            blocks_any = df[(df.get("dataset") == "Engagnition") & (df.get("unit_level") == "block")].index.tolist()
            if blocks_any:
                sess_eng = df.loc[idx_sess, ["participant_id", "condition", "engagement_level"]].copy()
                sess_eng["engagement_level"] = pd.to_numeric(sess_eng["engagement_level"], errors="coerce")
                e_map = sess_eng.groupby(["participant_id", "condition"])["engagement_level"].agg(series_mode)
                for j in blocks_any:
                    pid = df.at[j, "participant_id"]
                    cond = df.at[j, "condition"]
                    if (pid, cond) in e_map.index and pd.notna(e_map.loc[(pid, cond)]):
                        df.at[j, "engagement_level"] = int(e_map.loc[(pid, cond)])

    # Save back to the SAME Excel file (overwrite)
    _save_excel(df, meta_xlsx, sheet_name)

    # Summary
    print(f"[OK] RAW computed: {ok}, failed: {fail}")
    eng_rows = df.loc[idx_sess]
    vc = pd.to_numeric(eng_rows["movement_intensity_bin"], errors="coerce").value_counts(dropna=False)
    print("[OK] BIN value counts (session×ACC):")
    print(vc.to_string())
    print("[OK] Z describe (session×ACC):")
    print(pd.to_numeric(eng_rows["movement_intensity_z"], errors="coerce").describe())

    if "engagement_level" in df.columns:
        vce = pd.to_numeric(df.loc[idx_sess, "engagement_level"], errors="coerce").value_counts(dropna=False).sort_index()
        print("[OK] engagement_level counts (session):")
        try:
            print(vce.to_string())
        except Exception:
            print(vce)

    # Final marker (keep exactly this for upstream tooling)
    print("Engagnition basic")


def attach_engagement(df: pd.DataFrame, annots_path: str, annots_sheet: str|None) -> pd.DataFrame:
    """Attach engagement_level to df (Engagnition, session-level) from annotations."""
    ann = load_annotations(annots_path, annots_sheet)
    if ann.empty:
        print("[WARN] No annotations loaded; engagement_level skipped.")
        return df

    # Normalize colnames
    ann.columns = [str(c) for c in ann.columns]

    # Pick engagement column
    e_col = pick_engagement_column(ann)
    if e_col is None:
        print(f"[WARN] No engagement column found in annotations (looked for: {ENG_COL_CANDIDATES}). Skipped.")
        return df

    # Standardize keys
    # Try sample_id first
    has_sample_in_ann = "sample_id" in [c.lower() for c in ann.columns]
    has_sample_in_meta = "sample_id" in [c for c in df.columns]
    # Try participant_id + condition
    has_pid_ann = "participant_id" in [c.lower() for c in ann.columns]
    has_cond_ann = "condition" in [c.lower() for c in ann.columns]
    has_pid_meta = "participant_id" in df.columns
    has_cond_meta = "condition" in df.columns

    # Make lowercase accessors to be safe
    def col_case(df_, name):
        """Return actual column name matching lowercased 'name'."""
        for c in df_.columns:
            if c.lower() == name:
                return c
        return None

    e_vals = pd.to_numeric(ann[e_col], errors="coerce")
    ann = ann.copy()
    ann[e_col] = e_vals

    # Only session-level rows target
    session_mask = (df.get("dataset") == "Engagnition") & (df.get("unit_level") == "session")
    if "engagement_level" not in df.columns:
        df["engagement_level"] = pd.NA

    attached = 0

    if has_sample_in_ann and has_sample_in_meta:
        a_sid = col_case(ann, "sample_id")
        m_sid = "sample_id"  # already present
        grp = ann.groupby(a_sid)[e_col].agg(series_mode)
        # Map by sample_id
        for sid, lvl in grp.items():
            if pd.isna(lvl):
                continue
            mask = session_mask & (df[m_sid].astype(str) == str(sid))
            if mask.any():
                df.loc[mask, "engagement_level"] = int(lvl)
                attached += int(mask.sum())
        print(f"[OK] engagement_level attached by sample_id (rows updated: {attached}).")
        return df

    if has_pid_ann and has_cond_ann and has_pid_meta and has_cond_meta:
        a_pid = col_case(ann, "participant_id")
        a_cond = col_case(ann, "condition")
        grp = ann.groupby([a_pid, a_cond])[e_col].agg(series_mode)
        for (pid, cond), lvl in grp.items():
            if pd.isna(lvl):
                continue
            mask = session_mask & (df["participant_id"].astype(str) == str(pid)) & (df["condition"].astype(str) == str(cond))
            if mask.any():
                df.loc[mask, "engagement_level"] = int(lvl)
                attached += int(mask.sum())
        print(f"[OK] engagement_level attached by (participant_id, condition) (rows updated: {attached}).")
        return df

    if has_pid_ann and has_pid_meta:
        # last resort: by participant_id only
        a_pid = col_case(ann, "participant_id")
        grp = ann.groupby(a_pid)[e_col].agg(series_mode)
        for pid, lvl in grp.items():
            if pd.isna(lvl):
                continue
            mask = session_mask & (df["participant_id"].astype(str) == str(pid))
            if mask.any():
                df.loc[mask, "engagement_level"] = int(lvl)
                attached += int(mask.sum())
        print(f"[WARN] engagement_level attached by participant_id ONLY (rows updated: {attached}). "
              f"Consider adding 'sample_id' or 'condition' for precise mapping.")
        return df

    print("[WARN] Could not match annotations to metadata by sample_id or (participant_id, condition). Skipped.")
    return df

# CLI

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute Engagnition ACC movement_intensity and attach engagement_level.")
    ap.add_argument("--data-root", required=True, help="Path to the data root (rel_path_acc is relative to this).")
    ap.add_argument("--meta", required=True, help="Path to 'Engagnition basic.xlsx' to update in-place.")
    ap.add_argument("--sheet", default="Engagnition basic", help="Sheet name (default: 'Engagnition basic').")
    ap.add_argument("--annots", default=None,
                    help="Path to annotations file (CSV/XLSX) OR folder with multiple files (optional).")
    ap.add_argument("--annots-sheet", default=None,
                    help="Sheet name for Excel annotations (optional).")
    ap.add_argument("--propagate-blocks", action="store_true",
                    help="Copy session z/bin and engagement_level to ACC block rows per (participant_id, condition).")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    compute_eng_features(
        data_root=args.data_root,
        meta_xlsx=args.meta,
        sheet_name=args.sheet,
        annots_path=args.annots,
        annots_sheet=args.annots_sheet,
        propagate_blocks=args.propagate_blocks,
    )
