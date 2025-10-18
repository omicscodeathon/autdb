# build_lodo_intensity_table.py
# Build a unified feature table for LODO (movement_intensity_bin)
# from:
#   - MMASD: mmasd_cleaned.csv  (skeletal derived features)
#   - Engagnition: Engagnition_features_enriched.xlsx (acc_* features)
#
# Output CSV columns:
#   dataset, group_id, sample_id, movement_intensity_bin,
#   feat_median, feat_max, feat_p75, feat_iqr, feat_std, feat_var,
#   feat_mad, feat_cv, feat_range, feat_high_fraction,
#   feat_duration, feat_max_per_s, feat_std_per_s


import os
import argparse
import numpy as np
import pandas as pd

MMASD_MAP = {
    "skel_median": "feat_median",
    "skel_max": "feat_max",
    "skel_p75": "feat_p75",
    "skel_iqr": "feat_iqr",
    "skel_std": "feat_std",
    "skel_var": "feat_var",
    "skel_mad": "feat_mad",
    "skel_cv": "feat_cv",
    "skel_range": "feat_range",
    "skel_high_fraction": "feat_high_fraction",
    "skel_duration_s": "feat_duration",
    "skel_max_per_s": "feat_max_per_s",
    "skel_std_per_s": "feat_std_per_s",
}

ENG_MAP = {
    "acc_median": "feat_median",
    "acc_max": "feat_max",
    "acc_p75": "feat_p75",
    "acc_iqr": "feat_iqr",
    "acc_std": "feat_std",
    "acc_var": "feat_var",
    "acc_mad": "feat_mad",
    "acc_cv": "feat_cv",
    "acc_high_fraction": "feat_high_fraction",
    "acc_duration_s": "feat_duration",
    # feat_range, feat_max_per_s, feat_std_per_s will be derived if needed
}

UNIFIED_FEATS = [
    "feat_median", "feat_max", "feat_p75", "feat_iqr", "feat_std", "feat_var",
    "feat_mad", "feat_cv", "feat_range", "feat_high_fraction",
    "feat_duration", "feat_max_per_s", "feat_std_per_s",
]

GROUP_CANDIDATES = [
    "participant_id", "participant_id_global", "pid_group",
    "group", "group_id", "subject_id", "user_id"
]

ID_CANDIDATES = ["sample_id", "clip_id", "record_id", "session_id"]


def detect_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-8) -> pd.Series:
    return a / (b.clip(lower=eps).abs())


def build_mmasd_block(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    # group
    gcol = detect_col(df, GROUP_CANDIDATES)
    if gcol is None:
        raise ValueError("MMASD: group column not found. Add one of: " + ", ".join(GROUP_CANDIDATES))
    # id
    icol = detect_col(df, ID_CANDIDATES)
    if icol is None:
        # create synthetic id
        df["_tmp_index"] = np.arange(len(df))
        icol = "_tmp_index"

    if "movement_intensity_bin" not in df.columns:
        raise ValueError("MMASD: movement_intensity_bin not found.")

    out = pd.DataFrame({
        "dataset": "MMASD",
        "group_id": df[gcol].astype(str),
        "sample_id": df[icol].astype(str),
        "movement_intensity_bin": df["movement_intensity_bin"].astype("float"),
    })

    # map known features
    for src, dst in MMASD_MAP.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
        else:
            out[dst] = np.nan

    # ensure all unified features exist
    for f in UNIFIED_FEATS:
        if f not in out.columns:
            out[f] = np.nan

    # drop rows without target or group
    out = out.dropna(subset=["movement_intensity_bin", "group_id"])
    return out


def build_eng_block(path_xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(path_xlsx, sheet_name=0, engine="openpyxl")

    # keep Engagnition only if dataset column exists
    if "dataset" in df.columns:
        df = df.loc[df["dataset"].astype(str).str.lower() == "engagnition"].copy()

    # filter to ACC rows to avoid modality duplicates
    if "sample_id" in df.columns:
        mask = df["sample_id"].astype(str).str.contains("_ACC", case=False, na=False)
        if mask.any():
            df = df.loc[mask].copy()

    gcol = detect_col(df, GROUP_CANDIDATES)
    if gcol is None:
        raise ValueError("Engagnition: group column not found. Add one of: " + ", ".join(GROUP_CANDIDATES))

    icol = detect_col(df, ID_CANDIDATES)
    if icol is None:
        df["_tmp_index"] = np.arange(len(df))
        icol = "_tmp_index"

    if "movement_intensity_bin" not in df.columns:
        raise ValueError("Engagnition: movement_intensity_bin not found.")

    out = pd.DataFrame({
        "dataset": "Engagnition",
        "group_id": df[gcol].astype(str),
        "sample_id": df[icol].astype(str),
        "movement_intensity_bin": pd.to_numeric(df["movement_intensity_bin"], errors="coerce"),
    })

    # map acc_* to feat_*
    for src, dst in ENG_MAP.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
        else:
            out[dst] = np.nan

    # derive missing universal features
    # feat_range: prefer explicit acc_range, else acc_max - acc_median
    if "feat_range" not in out.columns or out["feat_range"].isna().all():
        if "acc_range" in df.columns:
            out["feat_range"] = pd.to_numeric(df["acc_range"], errors="coerce")
        else:
            out["feat_range"] = pd.to_numeric(df.get("acc_max", np.nan), errors="coerce") - \
                                pd.to_numeric(df.get("acc_median", np.nan), errors="coerce")

    # feat_max_per_s, feat_std_per_s
    acc_max = pd.to_numeric(df.get("acc_max", np.nan), errors="coerce")
    acc_std = pd.to_numeric(df.get("acc_std", np.nan), errors="coerce")
    duration = pd.to_numeric(df.get("acc_duration_s", np.nan), errors="coerce")

    out["feat_max_per_s"] = safe_div(acc_max, duration)
    out["feat_std_per_s"] = safe_div(acc_std, duration)

    # ensure all unified features exist
    for f in UNIFIED_FEATS:
        if f not in out.columns:
            out[f] = np.nan

    # drop rows without target or group
    out = out.dropna(subset=["movement_intensity_bin", "group_id"])
    return out


def main():
    ap = argparse.ArgumentParser(description="Build unified LODO feature table (movement_intensity_bin)")
    ap.add_argument("--mmasd", required=True, help="Path to mmasd_cleaned.csv")
    ap.add_argument("--eng", required=True, help="Path to Engagnition_features_enriched.xlsx")
    ap.add_argument("--out", default="lodo_intensity_features.csv", help="Output CSV path")
    args = ap.parse_args()

    mmasd_block = build_mmasd_block(args.mmasd)
    eng_block = build_eng_block(args.eng)

    # Select final columns order
    cols = ["dataset", "group_id", "sample_id", "movement_intensity_bin"] + UNIFIED_FEATS

    # Concat and keep only desired columns
    all_df = pd.concat([mmasd_block[cols], eng_block[cols]], ignore_index=True)

    # Optional: drop rows where ALL features are NaN
    feat_only = all_df[UNIFIED_FEATS]
    all_df = all_df.loc[~feat_only.isna().all(axis=1)].copy()

    # Save
    out_path = args.out
    all_df.to_csv(out_path, index=False)
    print(f"[✓] Saved unified table: {out_path}")
    print(all_df["dataset"].value_counts(dropna=False).to_string())
    print("\nColumns:\n", ", ".join(all_df.columns))


if __name__ == "__main__":
    main()
