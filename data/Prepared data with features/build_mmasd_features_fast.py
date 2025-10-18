# build_mmasd_features_fast.py

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import orjson
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def load_keypoints_orjson(json_path: Path):
    """Fast load keypoints with orjson."""
    try:
        with open(json_path, "rb") as f:
            data = orjson.loads(f.read())
    except Exception:
        return None
    people = data.get("people", [])
    if not people:
        return None
    # choose person with max confidence
    best = None
    best_conf = -1
    for p in people:
        arr = np.array(p.get("pose_keypoints_2d", []), dtype=float)
        if arr.size == 0:
            continue
        pts = arr.reshape(-1, 3)
        conf = np.nansum(pts[:, 2])
        if conf > best_conf:
            best_conf = conf
            best = pts[:, :2]
    return best

def compute_clip_speeds(clip_dir: Path) -> np.ndarray:
    jfiles = sorted(clip_dir.glob("*_keypoints.json"))
    if len(jfiles) < 2:
        return np.array([])
    coords = []
    for jf in jfiles:
        pts = load_keypoints_orjson(jf)
        if pts is None:
            coords.append(np.full((25, 2), np.nan))
        else:
            coords.append(pts)
    coords = np.array(coords)  # (frames, joints, 2)
    if coords.shape[0] < 2:
        return np.array([])
    diffs = np.diff(coords, axis=0)
    l2 = np.sqrt(np.nansum(diffs**2, axis=2))
    speeds = np.nanmean(l2, axis=1)
    return speeds[~np.isnan(speeds)]

def features_from_speeds(speeds: np.ndarray, fps: float):
    keys = [
        "skel_median","skel_iqr","skel_p75","skel_std","skel_mad",
        "skel_max","skel_var","skel_high_fraction","skel_duration_s"
    ]
    if speeds.size == 0:
        return {k: np.nan for k in keys}
    q25, q75 = np.percentile(speeds, [25, 75])
    med = np.median(speeds)
    return {
        "skel_median": med,
        "skel_iqr": q75 - q25,
        "skel_p75": q75,
        "skel_std": np.std(speeds, ddof=1),
        "skel_mad": np.median(np.abs(speeds - med)),
        "skel_max": np.max(speeds),
        "skel_var": np.var(speeds, ddof=1),
        "skel_high_fraction": np.mean(speeds > med),
        "skel_duration_s": speeds.size / fps,
    }

def process_clip(clip_dir: Path, activity: str, fps: float):
    speeds = compute_clip_speeds(clip_dir)
    feats = features_from_speeds(speeds, fps)
    return {
        "activity_class": activity,
        "clip_id": clip_dir.name,
        **feats
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root: MMASD/2D skeleton/output/")
    ap.add_argument("--out", default="MMASD_features.xlsx",
                    help="Output Excel file")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers")
    args = ap.parse_args()

    root = Path(args.root)
    act_dirs = [p for p in root.iterdir() if p.is_dir()]

    all_clip_dirs = []
    for act in act_dirs:
        for clip in act.iterdir():
            if clip.is_dir() and any(clip.glob("*.json")):
                all_clip_dirs.append((clip, act.name))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_clip, clip, act, args.fps)
                   for clip, act in all_clip_dirs]
        for f in tqdm(futures, desc="MMASD clips", ncols=100):
            rows.append(f.result())

    df = pd.DataFrame(rows)
    df.to_excel(args.out, index=False)
    print(f"✅ Saved {len(df)} rows → {args.out}")

if __name__ == "__main__":
    main()
