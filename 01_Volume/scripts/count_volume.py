"""
Volume counting — total vehicles crossing one or more count lines,
optionally split by direction inferred from motion vector.

Includes three anti-overcount filters:
  - min_track_age_frames: skip tracks that haven't existed long enough
  - min_bbox_area_px:     skip detections smaller than this
  - dedup_seconds:        merge same-direction crossings closer in time

Usage:
    python count_volume.py --config ../config/config.yaml
    python count_volume.py --config ../config/config.yaml --max-seconds 60
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils.tracker import track_video
from shared.utils.zones import bbox_center, crossed_line


def line_normal_side(prev, curr, line):
    """Returns 'A' or 'B' depending on which side of the line `prev` was on."""
    (x1, y1), (x2, y2) = line[0], line[1]
    px, py = prev
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return "A" if cross > 0 else "B"


def bbox_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def main(cfg_path: str, max_seconds: float | None = None,
         start_seconds: float = 0.0) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    bin_minutes = cfg.get("bin_minutes", 15)
    direction_names = cfg.get("direction_names", {"A": "Direction A", "B": "Direction B"})

    # Anti-overcount filters (configurable)
    min_track_age = int(cfg.get("min_track_age_frames", 5))
    min_area = float(cfg.get("min_bbox_area_px", 500))
    dedup_seconds = float(cfg.get("dedup_seconds", 1.5))

    last_pos: dict[int, tuple[float, float]] = {}
    track_age: dict[int, int] = defaultdict(int)
    counted_by_line: dict[str, set[int]] = defaultdict(set)
    # last accepted crossing per (line, direction) for time-based dedup
    last_cross_ts: dict[tuple[str, str], float] = {}

    raw_rows = []   # every detected crossing, even ones we drop
    rows = []      # accepted crossings (post-filter)

    for det in track_video(cfg["video"], model_path=cfg["model"],
                           conf=cfg["conf"], classes=cfg.get("classes"),
                           start_seconds=start_seconds):
        if max_seconds is not None and det["timestamp_s"] > max_seconds:
            break

        tid = det["track_id"]
        track_age[tid] += 1
        curr = bbox_center(det["xyxy"])
        prev = last_pos.get(tid)
        last_pos[tid] = curr
        if prev is None:
            continue

        for line in cfg["lines"]:
            if tid in counted_by_line[line["name"]]:
                continue
            if not crossed_line(prev, curr, line["points"]):
                continue

            side = line_normal_side(prev, curr, line["points"])
            dir_name = direction_names.get(side, side)
            area = bbox_area(det["xyxy"])
            age = track_age[tid]
            ts = det["timestamp_s"]

            # log all crossings before filtering for diagnostics
            reason = "accepted"
            if age < min_track_age:
                reason = f"track_age<{min_track_age}"
            elif area < min_area:
                reason = f"bbox_area<{min_area:.0f}"
            else:
                key = (line["name"], side)
                last = last_cross_ts.get(key)
                if last is not None and (ts - last) < dedup_seconds:
                    reason = f"dedup<{dedup_seconds}s"

            raw_rows.append({
                "timestamp_s": ts,
                "line": line["name"],
                "track_id": tid,
                "direction_code": side,
                "direction": dir_name,
                "bbox_area": area,
                "track_age_frames": age,
                "filter_result": reason,
            })

            if reason == "accepted":
                counted_by_line[line["name"]].add(tid)
                last_cross_ts[(line["name"], side)] = ts
                rows.append({
                    "timestamp_s": ts,
                    "line": line["name"],
                    "track_id": tid,
                    "direction_code": side,
                    "direction": dir_name,
                })

    df = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw_rows)

    if df.empty:
        print("No crossings accepted after filters. Try lowering min_track_age_frames or min_bbox_area_px.")
        return

    df["bin_min"] = (df["timestamp_s"] // (bin_minutes * 60)).astype(int) * bin_minutes

    by_bin = df.pivot_table(index="bin_min", columns="direction",
                            values="track_id", aggfunc="count", fill_value=0)
    by_bin["Total"] = by_bin.sum(axis=1)

    totals = df.groupby("direction")["track_id"].nunique().to_frame("Vehicles")
    totals.loc["TOTAL"] = totals["Vehicles"].sum()

    # Filter breakdown for diagnostics
    filter_summary = (raw_df.groupby(["direction", "filter_result"])
                      .size().to_frame("count").reset_index())

    out = Path(cfg["output"]["xlsx"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        totals.to_excel(xw, sheet_name="Summary")
        by_bin.to_excel(xw, sheet_name="Volume by 15-min")
        df.to_excel(xw, sheet_name="Raw crossings (kept)", index=False)
        raw_df.to_excel(xw, sheet_name="All crossings (debug)", index=False)
        filter_summary.to_excel(xw, sheet_name="Filter summary", index=False)
    print(f"Wrote {out}")
    print(totals)
    print()
    print("Filter breakdown:")
    print(filter_summary.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--start-seconds", type=float, default=0.0,
                   help="Seek forward to this video time before counting "
                        "(e.g., 57600 to start at hour 16)")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Stop counting at this video time")
    args = p.parse_args()
    main(args.config, args.max_seconds, args.start_seconds)
