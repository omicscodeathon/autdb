#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMASD-only enrichment: compute movement_intensity for rows from the 'MMASD basic.xlsx'

1) Reads the Excel file 'MMASD basic.xlsx' (sheet 'MMASD basic').
2) For each unique rel_path_openpose:
   - Recursively collects OpenPose JSON frames under <ROOT>/<rel_path_openpose>.
   - If nothing is found, additionally tries <ROOT>/MMASD/<rel_path_openpose>.
   - Computes movement_intensity_raw as:
       median_over_frames( median_over_joints( sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2) )
         using only joints with confidence >= conf_thr on BOTH frames ).
     Frames may be sub-sampled to 'max_frames' evenly spaced files for speed.
3) Robust normalization per participant_id (fallback → global over MMASD):
     z = (raw - median) / IQR   (if IQR <= 0 → use global; if still <= 0 → z = raw - median_or_0)
   and BIN = 1 if z >= 0 else 0.
4) Writes the enriched table back to the SAME Excel file,
   sheet name stays exactly 'MMASD basic'.

Progress bar:
- Shows percentage and ETA while processing clips, so you can see it’s alive.

CLI example (Windows CMD):
  python compute_mmasd_features.py ^
      --root "C:\\Users\\rusla\\Desktop\\Ruslan\\AutDB-Video\\data\\MMASD" ^
      --meta "C:\\Users\\rusla\\Desktop\\Ruslan\\AutDB-Video\\data\\MMASD basic.xlsx"

Dependencies: pandas, numpy, openpyxl (for reading), xlsxwriter (for writing), tqdm (for progress bar)
  pip install pandas numpy openpyxl xlsxwriter tqdm
"""

import os
import re
import json
import argparse
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm  # <-- progress bar


# helpers

def ensure_base_folder(root: str) -> str:
    """Verify '<root>/2D skeleton/output' exists and return its path."""
    base_dir = os.path.join(root, "2D skeleton", "output")
    if not os.path.isdir(base_dir):
        raise SystemExit(
            f"[ERR] Base folder not found: {base_dir}\n"
            f"Check that --root points to the MMASD root containing '2D skeleton/output'."
        )
    return base_dir


def collect_json_files_anywhere(path: Optional[str]) -> List[str]:
    """Return all JSON files under 'path' (recursively)."""
    if not path:
        return []
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return [path]
    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path, "*.json")))
        if files:
            return files
        return sorted(glob(os.path.join(path, "**", "*.json"), recursive=True))
    return []


def parse_openpose_keypoints(js_obj) -> Optional[np.ndarray]:
    """Extract Nx3 array [x,y,conf] from OpenPose JSON structure."""
    people = js_obj.get("people", [])
    if not people:
        return None
    arr = people[0].get("pose_keypoints_2d") or people[0].get("pose_keypoints")
    if not arr or not isinstance(arr, list):
        return None
    a = np.array(arr, dtype="float64")
    if a.ndim == 1:
        a = a.reshape(-1, 3)
    if a.shape[1] < 2:
        return None
    return a


def median_joint_speed_from_files(files: List[str], conf_thr: float, max_frames: int) -> float:
    """
    Core MMASD metric:
      RAW = median_over_frames( median_over_joints( speed ) ),
      speed = sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2),
      using joints with confidence >= conf_thr in BOTH frames.
    """
    if not files:
        return float("nan")

    # even sub-sample to at most max_frames files (stable selection)
    if max_frames and len(files) > max_frames:
        idx = np.linspace(0, len(files) - 1, max_frames).astype(int)
        files = [files[i] for i in idx]

    seq = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                js = json.load(f)
            kp = parse_openpose_keypoints(js)
        except Exception:
            kp = None
        seq.append(kp)

    speeds = []
    prev = None
    for cur in seq:
        if prev is not None and cur is not None and prev.shape == cur.shape:
            if prev.shape[1] >= 3:
                mask = (prev[:, 2] >= conf_thr) & (cur[:, 2] >= conf_thr)
            else:
                mask = np.ones(prev.shape[0], dtype=bool)
            if mask.any():
                dx = cur[mask, 0] - prev[mask, 0]
                dy = cur[mask, 1] - prev[mask, 1]
                sp = np.sqrt(dx * dx + dy * dy)
                if sp.size > 0:
                    speeds.append(float(np.median(sp)))
        prev = cur

    return float(np.median(speeds)) if speeds else float("nan")


def robust_center_scale(vals: pd.Series) -> Tuple[float, float]:
    """Return (median, IQR) with numeric coercion and NaNs ignored."""
    v = pd.to_numeric(vals, errors="coerce").dropna().values
    if v.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(v))
    q25, q75 = np.percentile(v, 25), np.percentile(v, 75)
    return med, float(q75 - q25)


# main computation

def compute_mmasd_features(root: str,
                           meta_xlsx: str,
                           sheet_name: str = "MMASD basic",
                           conf_thr: float = 0.10,       # default aligned with previous unified script
                           max_frames: int = 100,        # default aligned with previous unified script
                           debug_examples: int = 10) -> None:
    """
    Read 'MMASD basic.xlsx', compute movement_intensity_* for MMASD rows,
    and write back to the same Excel file (same sheet name).
    """

    # 0) guards
    ensure_base_folder(root)
    if not os.path.isfile(meta_xlsx):
        raise SystemExit(f"[ERR] Excel file not found: {meta_xlsx}")

    # 1) read meta
    try:
        df = pd.read_excel(meta_xlsx, sheet_name=sheet_name, dtype="object")
    except ValueError:
        # sheet doesn't exist → read first sheet
        df = pd.read_excel(meta_xlsx, dtype="object")
        sheet_name = df.attrs.get("sheet_name", "MMASD basic")

    # Ensure necessary columns exist
    for col in ["movement_intensity_raw", "movement_intensity_z", "movement_intensity_bin"]:
        if col not in df.columns:
            df[col] = pd.NA

    # 2) select MMASD rows with rel_path_openpose
    mm_mask = (df.get("dataset") == "MMASD") & df.get("rel_path_openpose").notna()
    rel_list = (
        df.loc[mm_mask, "rel_path_openpose"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .unique()
        .tolist()
    )

    computed, missing = 0, 0
    debug_list = []
    cache = {}

    # 3) compute RAW per unique rel_path_openpose with a progress bar (shows ETA)
    for relp in tqdm(rel_list, desc="[MMASD] Processing clips", unit="clip", ncols=100):
        cand1 = os.path.normpath(os.path.join(root, relp))
        files = collect_json_files_anywhere(cand1)

        used_path = cand1
        if not files:
            # fallback to <root>/MMASD/<rel> (mirrors previous behavior)
            cand2 = os.path.normpath(os.path.join(root, "MMASD", relp))
            files = collect_json_files_anywhere(cand2)
            used_path = cand2 if files else cand1

        val = median_joint_speed_from_files(files, conf_thr=conf_thr, max_frames=max_frames) if files else float("nan")
        cache[relp] = val
        if np.isnan(val):
            missing += 1
            if len(debug_list) < debug_examples:
                debug_list.append((used_path, len(files)))
        else:
            computed += 1

    # fill RAW values back to all MMASD rows
    df.loc[mm_mask, "movement_intensity_raw"] = (
        df.loc[mm_mask, "rel_path_openpose"].map(cache)
    )

    # 4) robust Z / BIN per participant with fallback → global over MMASD
    z_out = pd.Series(np.nan, index=df.index, dtype="float64")
    b_out = pd.Series(index=df.index, dtype="Int64")

    raw_all = pd.to_numeric(df.loc[mm_mask, "movement_intensity_raw"], errors="coerce")
    g_med, g_iqr = robust_center_scale(raw_all)

    for pid, grp in df[mm_mask].groupby("participant_id", dropna=False):
        vals = pd.to_numeric(grp["movement_intensity_raw"], errors="coerce")
        med, iqr = robust_center_scale(vals)
        if not (np.isfinite(iqr) and iqr > 0):
            med, iqr = g_med, g_iqr
        if np.isfinite(iqr) and iqr > 0:
            z_vals = (vals - med) / iqr
        else:
            # last-resort: centered values (IQR==0 or NaN)
            z_vals = vals - (med if np.isfinite(med) else 0.0)

        z_out.loc[grp.index] = z_vals.values
        b_out.loc[grp.index] = (z_vals >= 0).astype("Int64").values

    df["movement_intensity_z"] = z_out
    df["movement_intensity_bin"] = b_out

    # 5) write back to the SAME Excel file, same sheet name
    out_path = os.path.abspath(meta_xlsx)
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter", mode="w") as xlw:
            df.to_excel(xlw, index=False, sheet_name=sheet_name)
    except Exception as e:
        raise SystemExit(f"[ERR] Failed to write Excel: {out_path}\n{e}")

    # 6) concise summary
    print("[INFO] MMASD enrichment summary")
    print(f"  unique rel_path_openpose : {len(rel_list)}")
    print(f"  computed RAW             : {computed}")
    print(f"  missing RAW              : {missing}")
    if debug_list:
        print("  examples of missing (path, json_count):")
        for p, n in debug_list:
            print(f"    - {p}  -> {n}")
    print(f"[OK] Saved: {out_path}")
    print("MMASD basic")


# CLI

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute MMASD movement_intensity and write into 'MMASD basic.xlsx'.")
    ap.add_argument("--root", required=True, help="Path to MMASD root (must contain '2D skeleton/output').")
    ap.add_argument("--meta", required=True, help="Path to the Excel file 'MMASD basic.xlsx' to update in-place.")
    ap.add_argument("--conf-thr", type=float, default=0.10, help="Confidence threshold for joints (default: 0.10).")
    ap.add_argument("--max-frames", type=int, default=100, help="Max frames per clip for sub-sampling (default: 100).")
    ap.add_argument("--debug-examples", type=int, default=10, help="How many missing examples to print (default: 10).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_mmasd_features(
        root=args.root,
        meta_xlsx=args.meta,
        conf_thr=args.conf_thr,
        max_frames=args.max_frames,
        debug_examples=args.debug_examples,
    )
