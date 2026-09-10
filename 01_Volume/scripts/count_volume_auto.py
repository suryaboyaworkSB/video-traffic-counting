"""
Auto day/night volume counter.

Samples the video at intervals to classify each minute as 'day' or 'night'
based on mean frame brightness. Day minutes go to the YOLO pipeline; night
minutes go to the headlight-pair pipeline. Results merge into one Excel.

Usage:
    python count_volume_auto.py --config ../config/config.yaml
    python count_volume_auto.py --config ../config/config.yaml --max-seconds 600
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from count_volume import main as run_day
from count_volume_night import (
    detect_headlights, detect_taillights, pair_lights, CentroidTracker,
)
from shared.utils.zones import crossed_line


DEFAULT_NIGHT_THRESHOLD = 25.0


def classify_segments(video_path: Path, fps: float,
                      sample_every_s: float = 60.0,
                      night_threshold: float = DEFAULT_NIGHT_THRESHOLD,
                      ) -> list[tuple[float, float, str]]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_s = total_frames / fps
    sample_pts = np.arange(0, total_s, sample_every_s)

    labels: list[tuple[float, str]] = []
    for t in sample_pts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            break
        mean = float(frame.mean())
        labels.append((float(t), "night" if mean < night_threshold else "day"))
    cap.release()

    segs: list[tuple[float, float, str]] = []
    cur_start, cur_label = labels[0]
    for t, label in labels[1:]:
        if label != cur_label:
            segs.append((cur_start, t, cur_label))
            cur_start, cur_label = t, label
    segs.append((cur_start, total_s, cur_label))
    return segs


def run_night_segment(cfg: dict, video_path: Path, fps: float,
                      start_s: float, end_s: float,
                      direction_names: dict) -> list[dict]:
    """Night detector v5 - motion-based direction + time dedup."""
    night = cfg.get("night", {})
    roi_y_min = int(night.get("roi_y_min", 100))
    pair_max_dx = float(night.get("pair_max_dx", 40))
    pair_max_dy = float(night.get("pair_max_dy", 8))
    track_max_dist = float(night.get("track_max_distance", 45))
    track_max_missed = int(night.get("track_max_missed", 12))
    merge_dist = float(night.get("merge_distance", 80))
    singleton_min_area = int(night.get("singleton_min_area", 120))
    dedup_seconds = float(night.get("dedup_seconds_night", 1.0))
    line = cfg["lines"][0]["points"]
    line_name = cfg["lines"][0]["name"]

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))

    tracker = CentroidTracker(max_distance=track_max_dist, max_missed=track_max_missed)
    last_pos: dict[int, tuple[float, float]] = {}
    counted: set[int] = set()
    last_cross_ts: dict[str, float] = {}
    rows: list[dict] = []

    frame_idx = int(start_s * fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        if ts > end_s:
            break

        heads = detect_headlights(frame, roi_y_min)
        tails = detect_taillights(frame, roi_y_min)
        merged_blobs = list(heads)
        for t in tails:
            close = any(((t["cx"] - m["cx"]) ** 2 + (t["cy"] - m["cy"]) ** 2) ** 0.5 < merge_dist
                        for m in merged_blobs)
            if not close:
                merged_blobs.append(t)
        all_pairs = pair_lights(merged_blobs, pair_max_dx, pair_max_dy,
                                single_min_area=singleton_min_area)
        for p in all_pairs:
            p["kind"] = "vehicle"

        for tid, p in tracker.update(all_pairs):
            curr = (p["cx"], p["cy"])
            prev = last_pos.get(tid)
            last_pos[tid] = curr
            if prev is not None and tid not in counted:
                if crossed_line(prev, curr, line):
                    (x1, y1), (x2, y2) = line[0], line[1]
                    cross = ((x2 - x1) * (prev[1] - y1)
                             - (y2 - y1) * (prev[0] - x1))
                    side = "A" if cross > 0 else "B"
                    last_ts = last_cross_ts.get(side)
                    if last_ts is not None and (ts - last_ts) < dedup_seconds:
                        continue
                    counted.add(tid)
                    last_cross_ts[side] = ts
                    rows.append({
                        "timestamp_s": ts, "line": line_name, "track_id": f"N{tid}",
                        "direction_code": side,
                        "direction": direction_names.get(side, side),
                        "mode": "night",
                    })
        frame_idx += 1
    cap.release()
    return rows


def main(cfg_path: str, max_seconds: float | None = None,
         night_threshold: float = DEFAULT_NIGHT_THRESHOLD) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    direction_names = cfg.get("direction_names", {"A": "Direction A", "B": "Direction B"})
    bin_minutes = cfg.get("bin_minutes", 15)
    video_path = (Path(cfg_path).parent / cfg["video"]).resolve()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    cap.release()

    print("Classifying segments by brightness...")
    segs = classify_segments(video_path, fps,
                             sample_every_s=60.0,
                             night_threshold=night_threshold)
    if max_seconds is not None:
        segs = [(s, min(e, max_seconds), lab) for (s, e, lab) in segs if s < max_seconds]
    for s, e, lab in segs:
        print(f"  {s/3600:5.2f}h - {e/3600:5.2f}h : {lab}")

    all_rows: list[dict] = []

    from count_volume import main as run_day_main
    day_xlsx = Path(cfg["output"]["xlsx"])
    for s, e, lab in segs:
        if lab != "day":
            continue
        existing_rows = None
        if day_xlsx.exists():
            try:
                tmp = pd.read_excel(day_xlsx, sheet_name="Raw crossings (kept)")
                if len(tmp) > 0:
                    existing_rows = tmp
                    print(f"Reusing existing day output ({len(tmp)} rows) from {day_xlsx}")
            except Exception:
                existing_rows = None
        if existing_rows is None:
            print(f"Running YOLO on day segment {s/3600:.2f}-{e/3600:.2f}h...")
            run_day_main(cfg_path, max_seconds=e)
            if day_xlsx.exists():
                existing_rows = pd.read_excel(day_xlsx, sheet_name="Raw crossings (kept)")
        if existing_rows is not None:
            df = existing_rows[(existing_rows["timestamp_s"] >= s) &
                               (existing_rows["timestamp_s"] <= e)].copy()
            df["mode"] = "day"
            df["track_id"] = "D" + df["track_id"].astype(str)
            all_rows.extend(df.to_dict("records"))
        break

    for s, e, lab in segs:
        if lab != "night":
            continue
        print(f"Running night detector on {s/3600:.2f}-{e/3600:.2f}h...")
        rows = run_night_segment(cfg, video_path, fps, s, e, direction_names)
        all_rows.extend(rows)
        print(f"  +{len(rows)} crossings")

    if not all_rows:
        print("No crossings found.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = Path(cfg["output"]["xlsx"]).with_name("volume_counts_auto_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"Checkpoint CSV: {csv_path} ({len(df)} rows)")
    df["bin_min"] = (df["timestamp_s"] // (bin_minutes * 60)).astype(int) * bin_minutes
    by_bin = df.pivot_table(index="bin_min", columns="direction",
                            values="track_id", aggfunc="count", fill_value=0)
    by_bin["Total"] = by_bin.sum(axis=1)
    totals = df.groupby("direction")["track_id"].nunique().to_frame("Vehicles")
    totals.loc["TOTAL"] = totals["Vehicles"].sum()
    by_mode = df.groupby(["mode", "direction"]).size().to_frame("crossings").reset_index()

    out = Path(cfg["output"]["xlsx"]).with_name("volume_counts_auto.xlsx")
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        totals.to_excel(xw, sheet_name="Summary")
        by_bin.to_excel(xw, sheet_name="Volume by 15-min")
        by_mode.to_excel(xw, sheet_name="Day or Night", index=False)
        df.to_excel(xw, sheet_name="Raw crossings", index=False)
    print(f"Wrote {out}")
    print(totals)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--night-threshold", type=float, default=DEFAULT_NIGHT_THRESHOLD,
                   help="Mean frame brightness below this is treated as night (0-255)")
    args = p.parse_args()
    main(args.config, args.max_seconds, args.night_threshold)
