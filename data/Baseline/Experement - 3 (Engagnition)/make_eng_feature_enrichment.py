# -*- coding: utf-8 -*-
"""
make_eng_feature_enrichment.py
Enriches Engagnition_features_with_engagement.xlsx:
- adds new derived features from ACC directly from the table
- (optional) reads raw CSVs from source_dir and adds extended ACC/GSR/TMP features

Run:
py make_eng_feature_enrichment.py --input Engagnition_features_with_engagement.xlsx --output Engagnition_features_enriched.xlsx --data-root "C:\\path\\to\\your\\repo"
"""
import os, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

EPS = 1e-9

def safe_read_csv(path):
    try:
        if os.path.isfile(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None

def spectral_entropy(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return np.nan
    # FFT power
    p = np.abs(np.fft.rfft(x))**2
    if p.sum() <= 0:
        return np.nan
    p = p / p.sum()
    # -sum p log p
    return float(-(p * np.log(p + 1e-12)).sum())

def dominant_freq_hz(x, fs_hz):
    x = np.asarray(x, float)
    if x.size < 8 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.nan
    # rfft freqs
    P = np.abs(np.fft.rfft(x))**2
    f = np.fft.rfftfreq(x.size, d=1.0/fs_hz)
    if P.size < 2:
        return np.nan
    k = int(np.argmax(P[1:])) + 1  # skip DC
    return float(f[k])

def autocorr_at_lag(x, lag):
    x = np.asarray(x, float)
    n = x.size
    if n < lag + 2:
        return np.nan
    x = x - np.nanmean(x)
    num = np.nansum(x[:-lag] * x[lag:])
    den = np.nansum(x * x)
    if den == 0:
        return np.nan
    return float(num / den)

def robust_slope(y):
    # simple slope: linear regression via covariance
    y = np.asarray(y, float)
    n = y.size
    if n < 3:
        return np.nan
    x = np.arange(n, dtype=float)
    x -= x.mean(); y = y - np.nanmean(y)
    den = (x**2).sum()
    if den <= 0:
        return np.nan
    num = np.nansum(x * y)
    return float(num / den)

def peaks_count_simple(x, thr=None):
    # very simple peak counter: counts local maxima
    x = np.asarray(x, float)
    if x.size < 3:
        return 0
    if thr is None:
        thr = np.nanmedian(x) + 0.5*np.nanstd(x)
    c = 0
    for i in range(1, len(x)-1):
        if np.isfinite(x[i-1]) and np.isfinite(x[i]) and np.isfinite(x[i+1]):
            if x[i] > x[i-1] and x[i] > x[i+1] and x[i] >= thr:
                c += 1
    return int(c)

def infer_fs_hz(t):
    # t — either seconds (float) or UNIX timestamp
    t = pd.Series(t).astype(float)
    t = t[np.isfinite(t)]
    if t.size < 3:
        return np.nan
    dt = np.diff(t.values)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return np.nan
    med = np.median(dt)
    if med <= 0:
        return np.nan
    return float(1.0 / med)

def compute_acc_features_from_csv(df_acc):
    # expects columns: (timestamp|time) + (x,y,z) or (ax,ay,az)
    cols = {c.lower(): c for c in df_acc.columns}
    tcol = cols.get("timestamp") or cols.get("time")
    xcol = cols.get("ax") or cols.get("x")
    ycol = cols.get("ay") or cols.get("y")
    zcol = cols.get("az") or cols.get("z")
    if not all([tcol, xcol, ycol, zcol]):
        return {}
    x = df_acc[xcol].astype(float).values
    y = df_acc[ycol].astype(float).values
    z = df_acc[zcol].astype(float).values
    svm = np.sqrt(x*x + y*y + z*z)

    fs = infer_fs_hz(df_acc[tcol])
    feats = {
        "acc_skew": float(pd.Series(svm).skew(skipna=True)),
        "acc_kurtosis": float(pd.Series(svm).kurt(skipna=True)),
        "acc_entropy": spectral_entropy(svm),
        "acc_dom_freq_hz": dominant_freq_hz(svm, fs),
        "acc_autocorr_lag2": autocorr_at_lag(svm, 2),
        "acc_autocorr_lag3": autocorr_at_lag(svm, 3),
    }
    return feats

def compute_gsr_features_from_csv(df_gsr):
    # expected columns: (timestamp|time), (GSR|gsr)
    cols = {c.lower(): c for c in df_gsr.columns}
    tcol = cols.get("timestamp") or cols.get("time")
    gcol = cols.get("gsr") or cols.get("gsr_micro") or cols.get("eda") or cols.get("electrodermal") or None
    if not gcol:
        # try to use the only numeric column
        nums = [c for c in df_gsr.columns if np.issubdtype(df_gsr[c].dtype, np.number)]
        gcol = nums[0] if nums else None
    if not gcol:
        return {}
    g = df_gsr[gcol].astype(float).values
    fs = infer_fs_hz(df_gsr[tcol]) if tcol in df_gsr.columns else np.nan
    n = len(g)
    per_min = (n / fs / 60.0) if (np.isfinite(fs) and fs > 0) else np.nan
    peaks = peaks_count_simple(g)
    feats = {
        "gsr_mean": float(np.nanmean(g)),
        "gsr_std": float(np.nanstd(g)),
        "gsr_mad": float(np.nanmedian(np.abs(g - np.nanmedian(g)))),
        "gsr_slope": robust_slope(g),
        "gsr_entropy": spectral_entropy(g),
        "gsr_peaks_per_min": (float(peaks) / per_min) if (per_min and per_min > 0) else np.nan,
    }
    return feats

def compute_tmp_features_from_csv(df_tmp):
    cols = {c.lower(): c for c in df_tmp.columns}
    tcol = cols.get("timestamp") or cols.get("time")
    vcols = [c for c in df_tmp.columns if np.issubdtype(df_tmp[c].dtype, np.number)]
    if not vcols:
        return {}
    v = df_tmp[vcols[0]].astype(float).values
    feats = {
        "tmp_mean": float(np.nanmean(v)),
        "tmp_std": float(np.nanstd(v)),
        "tmp_slope": robust_slope(v),
    }
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Engagnition_features_with_engagement.xlsx")
    ap.add_argument("--output", required=True, help="Path to save enriched XLSX")
    ap.add_argument("--data-root", default=None, help="Root folder with data (to locate E4AccData.csv/Gsr/Tmp)")
    args = ap.parse_args()

    df = pd.read_excel(args.input)

    # 1) Derived features from already available ACC aggregates
    def add_ratio(col, denom="acc_median"):
        if col in df.columns and denom in df.columns:
            df[f"{col}_over_median"] = df[col].astype(float) / (df[denom].astype(float) + EPS)

    if "acc_energy" in df.columns:
        df["acc_energy_log1p"]   = np.log1p(df["acc_energy"].astype(float))
    if "acc_energy" in df.columns and "acc_duration_s" in df.columns:
        d = df["acc_duration_s"].astype(float).replace(0, np.nan)
        df["acc_energy_per_s"] = df["acc_energy"].astype(float) / (d + EPS)

    if "acc_std" in df.columns and "acc_median" in df.columns:
        df["acc_cv"] = df["acc_std"].astype(float) / (df["acc_median"].astype(float) + EPS)

    for base in ["acc_iqr", "acc_mad", "acc_max", "acc_var"]:
        if base in df.columns:
            add_ratio(base)

    if "acc_high_fraction" in df.columns:
        df["is_high_activity"] = (df["acc_high_fraction"].astype(float) > 0.5).astype(int)

    # 2) (optional) extended features from raw CSVs, if data-root and source_dir are available
    if args.data_root and "source_dir" in df.columns:
        extra_cols = [
            # ACC
            "acc_skew","acc_kurtosis","acc_entropy","acc_dom_freq_hz","acc_autocorr_lag2","acc_autocorr_lag3",
            # GSR
            "gsr_mean","gsr_std","gsr_mad","gsr_slope","gsr_entropy","gsr_peaks_per_min",
            # TMP
            "tmp_mean","tmp_std","tmp_slope"
        ]
        for c in extra_cols:
            if c not in df.columns:
                df[c] = np.nan

        for i, row in df.iterrows():
            base = os.path.join(args.data_root, str(row["source_dir"])) if pd.notna(row.get("source_dir", np.nan)) else None
            if not base or not os.path.isdir(base):
                continue
            sid = str(row.get("sample_id",""))

            # ACC
            acc_path = os.path.join(base, "E4AccData.csv")
            acc_df = safe_read_csv(acc_path)
            if acc_df is not None:
                acc_feats = compute_acc_features_from_csv(acc_df)
                for k, v in acc_feats.items():
                    df.at[i, k] = v

            # GSR
            gsr_path = os.path.join(base, "E4GsrData.csv")
            gsr_df = safe_read_csv(gsr_path)
            if gsr_df is not None:
                gsr_feats = compute_gsr_features_from_csv(gsr_df)
                for k, v in gsr_feats.items():
                    df.at[i, k] = v

            # TMP
            tmp_path = os.path.join(base, "E4TmpData.csv")
            tmp_df = safe_read_csv(tmp_path)
            if tmp_df is not None:
                tmp_feats = compute_tmp_features_from_csv(tmp_df)
                for k, v in tmp_feats.items():
                    df.at[i, k] = v

    # 3) Save the result
    df.to_excel(args.output, index=False)
    print(f"[✓] Saved enriched table to: {args.output}")

if __name__ == "__main__":
    main()
