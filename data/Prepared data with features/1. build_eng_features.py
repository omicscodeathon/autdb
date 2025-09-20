# -*- coding: utf-8 -*-
"""
Build Engagnition features table from raw E4AccData.csv files
and Engagnition basic.xlsx using argparse (universal style + progress bar).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm   # <-- progress bar


# helpers
def compute_acc_features(acc_df: pd.DataFrame) -> dict:
    """Compute statistical features from ACC data (x, y, z)."""
    svm = np.sqrt(acc_df['x']**2 + acc_df['y']**2 + acc_df['z']**2)

    feats = {}
    feats["acc_median"] = np.median(svm)
    feats["acc_iqr"] = np.percentile(svm, 75) - np.percentile(svm, 25)
    feats["acc_p75"] = np.percentile(svm, 75)
    feats["acc_std"] = np.std(svm, ddof=1)
    feats["acc_mad"] = np.median(np.abs(svm - np.median(svm)))
    feats["acc_max"] = np.max(svm)
    feats["acc_var"] = np.var(svm, ddof=1)
    feats["acc_energy"] = np.sum(svm**2)

    if len(svm) > 1:
        feats["acc_autocorr_lag1"] = np.corrcoef(svm[:-1], svm[1:])[0, 1]
    else:
        feats["acc_autocorr_lag1"] = np.nan

    feats["acc_high_fraction"] = np.mean(svm > np.median(svm))
    feats["acc_duration_s"] = len(svm) / 32.0   # Empatica E4 ACC = 32 Hz

    return feats


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="Path to Engagnition basic.xlsx (input metadata)")
    ap.add_argument("--out", required=True,
                    help="Path to save new Engagnition_features.xlsx")
    ap.add_argument("--root", default="Engagnition",
                    help="Root folder with raw data (default: Engagnition/)")
    args = ap.parse_args()

    basic_path = Path(args.data)
    data_root = Path(args.root)
    out_path = Path(args.out)

    # read the base session list
    basic = pd.read_excel(basic_path)

    rows = []
    # add tqdm progress bar
    for _, row in tqdm(basic.iterrows(), total=len(basic), desc="Processing sessions"):
        pid = row["participant_id"]   # e.g. P01
        cond = row["condition"]       # Baseline / LPE / HPE

        # expected file structure: Engagnition/{COND} condition/Pxx/E4AccData.csv
        acc_path = data_root / f"{cond} condition" / pid / "E4AccData.csv"

        if acc_path.exists():
            try:
                acc_df = pd.read_csv(acc_path)
                if not {"x", "y", "z"}.issubset(acc_df.columns):
                    acc_df = pd.read_csv(acc_path, names=["ts", "x", "y", "z"], header=None)
            except Exception:
                acc_df = pd.read_csv(acc_path, names=["ts", "x", "y", "z"], header=None)

            # Force numeric conversion
            for col in ["x", "y", "z"]:
                acc_df[col] = pd.to_numeric(acc_df[col], errors="coerce").fillna(0)

            feats = compute_acc_features(acc_df)
        else:
            feats = {k: np.nan for k in [
                "acc_median","acc_iqr","acc_p75","acc_std","acc_mad",
                "acc_max","acc_var","acc_energy","acc_autocorr_lag1",
                "acc_high_fraction","acc_duration_s"
            ]}

        base = row.to_dict()
        base.update(feats)
        rows.append(base)

    # build final feature table
    eng_feats = pd.DataFrame(rows)

    # save
    eng_feats.to_excel(out_path, index=False)
    print(f"\n✅ Features table saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
