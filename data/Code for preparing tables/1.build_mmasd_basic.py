#!/usr/bin/env python
# coding: utf-8

"""
Parser for MMASD: build 'MMASD basic.xlsx'
- Collects sample_id, participant_id, activity_class from folder structure
- Merges with ADOS_rating.xlsx (sex, age_years)
- No computations, only known data
- Safer: verifies paths, creates folders, prints summary
"""

import os
import re
import argparse
import sys
import pandas as pd

# activity prefix dictionary
ACTIVITY_MAP = {
    "as": "Arm Swing",
    "bs": "Body Swing",
    "ce": "Chest Expansion",
    "sq": "Squat",
    "dr": "Drumming",
    "mfs": "Maracas Forward Shaking",
    "ms": "Maracas Shaking",
    "sac": "Sing and Clap",
    "fg": "Frog Pose",
    "tr": "Tree Pose",
    "tw": "Twist Pose",
}

SAMPLE_ID_RE = re.compile(r"^([a-z]+)_([0-9]+)_", re.IGNORECASE)


def parse_sample(sample_id: str):
    """Extract activity_prefix and participant_id from sample_id"""
    m = SAMPLE_ID_RE.match(sample_id)
    if not m:
        return "", ""
    return m.group(1).lower(), m.group(2)


def scan_openpose(root: str):
    """
    Scan structure '2D skeleton/output/'
    Return DataFrame with sample_id, participant_id, activity_class
    """
    base_dir = os.path.join(root, "2D skeleton", "output")
    if not os.path.isdir(base_dir):
        sys.exit(f"[ERR] Base folder not found: {base_dir}")

    rows = []
    for activity_folder in os.listdir(base_dir):
        activity_dir = os.path.join(base_dir, activity_folder)
        if not os.path.isdir(activity_dir):
            continue

        for sample_id in os.listdir(activity_dir):
            sample_path = os.path.join(activity_dir, sample_id)
            if not os.path.isdir(sample_path):
                continue

            prefix, participant_id = parse_sample(sample_id)
            activity_class = ACTIVITY_MAP.get(prefix, "Unknown")

            rows.append({
                "sample_id": sample_id,
                "participant_id": participant_id,
                "activity_prefix": prefix,
                "activity_class": activity_class,
                "rel_path_openpose": os.path.relpath(sample_path, root),
            })

    return pd.DataFrame(rows)


def read_ados(ados_path: str):
    """Read ADOS_rating.xlsx and extract sex and age_years"""
    if not os.path.isfile(ados_path):
        sys.exit(f"[ERR] ADOS file not found: {ados_path}")

    df = pd.read_excel(ados_path)
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    pid_col = next((cols[c] for c in cols if "id" in c), None)
    sex_col = next((cols[c] for c in cols if "sex" in c or "gender" in c), None)
    age_col = next((cols[c] for c in cols if "age" in c), None)

    out = pd.DataFrame()
    if pid_col:
        out["participant_id"] = df[pid_col].astype(str).str.extract(r"(\d+)")[0]
    if sex_col:
        out["sex"] = df[sex_col]
    if age_col:
        out["age_years"] = df[age_col]

    return out.dropna(subset=["participant_id"]).drop_duplicates("participant_id")


def main():
    ap = argparse.ArgumentParser(description="Build 'MMASD basic.xlsx'.")
    ap.add_argument("--root", required=True, help="Path to MMASD root folder")
    ap.add_argument("--ados", required=True, help="Path to ADOS_rating.xlsx")
    ap.add_argument("--out", required=True, help="Output directory or file path")
    args = ap.parse_args()

    # 1) parse OpenPose structure
    df_openpose = scan_openpose(args.root)

    # 2) add dataset column
    df_openpose["dataset"] = "MMASD"

    # 3) merge ADOS
    ados = read_ados(args.ados)
    df = df_openpose.merge(ados, on="participant_id", how="left")

    # 4) resolve output path
    out_dir = args.out if os.path.isdir(args.out) else os.path.dirname(args.out)
    if out_dir == "":
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "MMASD basic.xlsx")

    # 5) save to Excel with sheet name "MMASD basic"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="MMASD basic")

    # 6) print summary
    print(f"[OK] Saved {len(df)} rows -> {out_path}")
    print("MMASD basic")


if __name__ == "__main__":
    main()
