"""
v6 day volume counter — kitchen sink: multi-line + foot-of-bbox + configurable model.

Three improvements stacked on top of v5:

1. MULTI-LINE OR-COUNTING
   Config can list multiple count lines. A vehicle is counted ONCE the first
   time its track crosses ANY of them. Helps catch vehicles that miss the
   primary line due to track fragmentation or unusual paths.

2. FOOT-OF-BBOX CROSSING POINT
   Instead of using the bbox center (which drifts with perspective as a
   vehicle gets closer), use the bottom-middle of the bbox — where the
   wheels are. Stable position on the road plane, more accurate line
   crossing detection.

3. BIGGER MODEL (config knob)
   Just set `model: yolov8l.pt` (or yolov8x.pt) in config.yaml. Slower
   per frame on CPU but higher recall, especially on small/dark vehicles.

Other v5 improvements (filters, time-based dedup) carry over unchanged.

Usage:
    python count_volume_v6.py --config ../config/config.yaml
    python count_volume_v6.py --config ../config/config.yaml --max-seconds 3600
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils.tracker import track_video
from shared.utils.zones import crossed_line


def bbox_foot(xyxy):
    """Bottom-middle of the bbox - where the wheels meet the road."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, y2)


def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def line_normal_side(prev, curr, line):
    """'A' if prev was on side A, 'B' if on side B."""
    (x1, y1), (x2, y2) = line[0], line[1]
    px, py = prev
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return "A" if cross > 0 else "B"


def main(cfg_path: str, max_seconds: float | None = None) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    bin_minutes = cfg.get("bin_minutes", 15)
    direction_names = cfg.get("direction_names", {"A": "Direction A", "B": "Direction B"})

    # v6 knobs
    crossing_point_name = cfg.get("crossing_point", "foot")    # 'foot' (default) or 'center'
    crossing_fn = bbox_foot if crossing_point_name == "foot" else bbox_center

    # v5 filters (unchanged)
    min_track_age = int(cfg.get("min_track_age_frames", 2))
    min_area = float(cfg.get("min_bbox_area_px", 400))
    dedup_seconds = float(cfg.get("dedup_seconds", 1.5))

    lines = cfg["lines"]
    if len(lines) > 1:
        print(f"[v6] Multi-line OR-counting enabled ({len(lines)} lines)")
    print(f"[v6] Crossing point: {crossing_point_name}")
    print(f"[v6] Model: {cfg['model']}")
    print(f"[v6] Conf threshold: {cfg.get('conf', 0.25)}")

    last_pos: dict[int, tuple[float, float]] = {}
    track_age: dict[int, int] = defaultdict(int)
    counted_vehicles: set[int] = set()          # each tid counted at most once across ALL lines
    last_cross_ts: dict[str, float] = {}        # time-based dedup per direction code

    raw_rows = []   # everything seen, for diagnostics
    rows = []      # accepted (counted) crossings

    for det in track_video(cfg["video"], model_path=cfg["model"],
                           conf=cfg.get("conf", 0.25),
                           classes=cfg.get("classes")):
        if max_seconds is not None and det["timestamp_s"] > max_seconds:
            break

        tid = det["track_id"]
        track_age[tid] += 1
        if track_age[tid] < min_track_age:
            continue

        area = bbox_area(det["xyxy"])
        if area < min_area:
            continue

        curr = crossing_fn(det["xyxy"])
        prev = last_pos.get(tid)
        last_pos[tid] = curr
        if prev is None:
            continue

        # already counted on a previous line — skip
        if tid in counted_vehicles:
            continue

        # Try each configured line; count on the FIRST one this vehicle crosses
        for line in lines:
            if not crossed_line(prev, curr, line["points"]):
                continue

            side = line_normal_side(prev, curr, line["points"])
            dir_name = direction_names.get(side, side)
            ts = det["timestamp_s"]

            # diagnostic row (recorded regardless of dedup)
            raw_rows.append({
                "timestamp_s": ts,
                "line": line["name"],
                "track_id": tid,
                "direction_code": side,
                "direction": dir_name,
                "bbox_area": area,
                "track_age_frames": track_age[tid],
            })

            # time-based dedup catches tracker fragmentation (new tid, same vehicle)
            last_ts = last_cross_ts.get(side)
            if last_ts is not None and (ts - last_ts) < dedup_seconds:
                break  # rejected, but we already broke from line search

            counted_vehicles.add(tid)
            last_cross_ts[side] = ts
            rows.append({
                "timestamp_s": ts,
                "line": line["name"],
                "track_id": tid,
                "direction_code": side,
                "direction": dir_name,
            })
            break  # one count per vehicle, even if it crosses other lines

    df = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw_rows)

    if df.empty:
        print("No crossings counted. Check lines, model, conf threshold.")
        return

    df["bin_min"] = (df["timestamp_s"] // (bin_minutes * 60)).astype(int) * bin_minutes
    by_bin = df.pivot_table(index="bin_min", columns="direction",
                            values="track_id", aggfunc="count", fill_value=0)
    by_bin["Total"] = by_bin.sum(axis=1)
    totals = df.groupby("direction")["track_id"].nunique().to_frame("Vehicles")
    totals.loc["TOTAL"] = totals["Vehicles"].sum()

    # which line did the counted vehicles get caught on (multi-line diagnostic)
    by_line = df.groupby(["line", "direction"]).size().to_frame("count").reset_index()

    out = Path(cfg["output"]["xlsx"]).with_name("volume_counts_v6.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        totals.to_excel(xw, sheet_name="Summary")
        by_bin.to_excel(xw, sheet_name="Volume by 15-min")
        by_line.to_excel(xw, sheet_name="By line", index=False)
        df.to_excel(xw, sheet_name="Raw crossings (kept)", index=False)
        raw_df.to_excel(xw, sheet_name="All crossings (debug)", index=False)
    print(f"Wrote {out}")
    print(totals)
    print()
    print("Crossings by line:")
    print(by_line.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Process only this many seconds of video (for testing)")
    args = p.parse_args()
    main(args.config, args.max_seconds)
