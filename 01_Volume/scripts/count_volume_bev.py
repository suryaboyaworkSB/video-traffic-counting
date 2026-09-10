"""
BEV (bird's-eye-view) volume counter.

Same YOLO+ByteTrack pipeline as count_volume.py, but each vehicle's
foot-of-bbox is projected through the homography matrix to BEV coordinates
BEFORE checking line crossings. In BEV space the road runs straight, so
crossings are accurate and direction is unambiguous regardless of how
oblique the camera is.

Counts are still done against the configured count line, but the line
is also projected to BEV automatically, so you can keep using your
existing `lines:` config (single line, multi-line, doesn't matter).

Usage:
    python count_volume_bev.py --config ../config/config.yaml
    python count_volume_bev.py --config ../config/config.yaml --max-seconds 32400

Prerequisites:
    - Run pick_homography.py first to set the homography in config
    - Have at least one line in config.yaml's `lines:` section
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils.tracker import track_video
from shared.utils.zones import crossed_line


def bbox_foot(xyxy):
    """Bottom-middle of bbox - where wheels meet the road."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, y2)


def bbox_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def project_point(pt, H):
    """Project a single (x, y) point through 3x3 homography H."""
    a = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    b = cv2.perspectiveTransform(a, H)
    return (float(b[0, 0, 0]), float(b[0, 0, 1]))


def line_normal_side(prev, curr, line):
    """'A' if prev was on side A of the line, 'B' if side B."""
    (x1, y1), (x2, y2) = line[0], line[1]
    px, py = prev
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return "A" if cross > 0 else "B"


def main(cfg_path: str, max_seconds: float | None = None,
         start_seconds: float = 0.0) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    bin_minutes = cfg.get("bin_minutes", 15)
    direction_names = cfg.get("direction_names",
                              {"A": "Direction A", "B": "Direction B"})

    if "homography" not in cfg:
        print("ERROR: no 'homography' in config. Run pick_homography.py first.")
        return
    H = np.array(cfg["homography"]["matrix"], dtype=np.float32)
    bev_W, bev_H = cfg["homography"]["bev_size"]

    lines = cfg["lines"]
    # Project each line's endpoints to BEV space once, ahead of time
    lines_bev = []
    for line in lines:
        p1_bev = project_point(line["points"][0], H)
        p2_bev = project_point(line["points"][1], H)
        lines_bev.append({"name": line["name"], "points": [p1_bev, p2_bev]})
        print(f"[BEV] line {line['name']}: image {line['points']} -> BEV {[list(p1_bev), list(p2_bev)]}")

    # v5 filters
    min_track_age = int(cfg.get("min_track_age_frames", 2))
    min_area = float(cfg.get("min_bbox_area_px", 400))
    dedup_seconds = float(cfg.get("dedup_seconds", 1.5))

    print(f"[BEV] Model: {cfg['model']}, conf: {cfg.get('conf', 0.25)}")
    print(f"[BEV] BEV space: {bev_W}x{bev_H}")
    print(f"[BEV] {len(lines)} count line(s)")

    last_bev: dict[int, tuple[float, float]] = {}
    track_age: dict[int, int] = defaultdict(int)
    counted_vehicles: set[int] = set()
    last_cross_ts: dict[str, float] = {}

    rows = []
    raw_rows = []

    for det in track_video(cfg["video"], model_path=cfg["model"],
                           conf=cfg.get("conf", 0.25),
                           classes=cfg.get("classes"),
                           start_seconds=start_seconds):
        if max_seconds is not None and det["timestamp_s"] > max_seconds:
            break

        tid = det["track_id"]
        track_age[tid] += 1
        if track_age[tid] < min_track_age:
            continue

        area = bbox_area(det["xyxy"])
        if area < min_area:
            continue

        # Project foot to BEV
        foot_img = bbox_foot(det["xyxy"])
        curr_bev = project_point(foot_img, H)

        # Skip detections that fall outside the BEV view (way off-road)
        if not (-bev_W <= curr_bev[0] <= 2 * bev_W and
                -bev_H <= curr_bev[1] <= 2 * bev_H):
            continue

        prev_bev = last_bev.get(tid)
        last_bev[tid] = curr_bev
        if prev_bev is None:
            continue

        if tid in counted_vehicles:
            continue

        # Check crossings of each configured line in BEV space
        for line_bev in lines_bev:
            if not crossed_line(prev_bev, curr_bev, line_bev["points"]):
                continue

            side = line_normal_side(prev_bev, curr_bev, line_bev["points"])
            dir_name = direction_names.get(side, side)
            ts = det["timestamp_s"]

            raw_rows.append({
                "timestamp_s": ts, "line": line_bev["name"],
                "track_id": tid,
                "direction_code": side, "direction": dir_name,
                "foot_img_x": foot_img[0], "foot_img_y": foot_img[1],
                "foot_bev_x": curr_bev[0], "foot_bev_y": curr_bev[1],
            })

            last_ts = last_cross_ts.get(side)
            if last_ts is not None and (ts - last_ts) < dedup_seconds:
                break

            counted_vehicles.add(tid)
            last_cross_ts[side] = ts
            rows.append({
                "timestamp_s": ts, "line": line_bev["name"],
                "track_id": tid,
                "direction_code": side, "direction": dir_name,
            })
            break  # count once per vehicle

    df = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw_rows)
    if df.empty:
        print("No BEV crossings counted. Check homography and line placement.")
        return

    df["bin_min"] = (df["timestamp_s"] // (bin_minutes * 60)).astype(int) * bin_minutes
    by_bin = df.pivot_table(index="bin_min", columns="direction",
                            values="track_id", aggfunc="count", fill_value=0)
    by_bin["Total"] = by_bin.sum(axis=1)
    totals = df.groupby("direction")["track_id"].nunique().to_frame("Vehicles")
    totals.loc["TOTAL"] = totals["Vehicles"].sum()

    by_line = df.groupby(["line", "direction"]).size().to_frame("count").reset_index()

    out = Path(cfg["output"]["xlsx"]).with_name("volume_counts_bev.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        totals.to_excel(xw, sheet_name="Summary")
        by_bin.to_excel(xw, sheet_name="Volume by 15-min")
        by_line.to_excel(xw, sheet_name="By line", index=False)
        df.to_excel(xw, sheet_name="Raw crossings", index=False)
        raw_df.to_excel(xw, sheet_name="All crossings (debug)", index=False)
    print(f"Wrote {out}")
    print(totals)
    print()
    print("By line:")
    print(by_line.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--start-seconds", type=float, default=0.0,
                   help="Seek forward to this video time before counting "
                        "(e.g., 57600 to start at hour 16). Saves CPU.")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Stop counting at this video time")
    args = p.parse_args()
    main(args.config, args.max_seconds, args.start_seconds)
