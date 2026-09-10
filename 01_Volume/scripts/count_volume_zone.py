"""
Zone-based volume counter (night mode).

Instead of one count line, this uses a ZONE bounded by two lines:
  - FAR line  (higher in the frame, away from the camera)
  - NEAR line (lower in the frame, close to the camera)

A vehicle is only counted when it crosses BOTH lines. Direction comes from
the crossing order:
  FAR first, then NEAR  -> moving toward camera   -> Direction B
  NEAR first, then FAR  -> moving away from camera -> Direction A

Why this beats a single line:
  - Direction is unambiguous (crossing order, not a noisy motion vector)
  - False positives are rejected: a reflection / fragment that only touches
    one line never gets counted
  - Crossings are matched LOOSELY (by time + x-position, not track ID), so
    tracker fragmentation between the two lines doesn't lose real vehicles

This script is night-focused: it detects vehicles by their lights using the
same detectors as count_volume_night.py.

Usage:
    python count_volume_zone.py --config ../config/config.yaml
    python count_volume_zone.py --config ../config/config.yaml --start-seconds 68400 --max-seconds 3600
    python count_volume_zone.py --config ../config/config.yaml --start-seconds 68400 --max-seconds 600 --debug-video
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils.zones import crossed_line
from count_volume_night import (
    detect_headlights, detect_taillights, pair_lights, CentroidTracker,
)


def main(cfg_path: str, start_seconds: float = 0,
         max_seconds: float | None = None,
         debug_video: bool = False) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    direction_names = cfg.get("direction_names", {"A": "Direction A", "B": "Direction B"})
    bin_minutes = cfg.get("bin_minutes", 15)

    night = cfg.get("night", {})
    roi_y_min = int(night.get("roi_y_min", 100))
    pair_max_dx = float(night.get("pair_max_dx", 40))
    pair_max_dy = float(night.get("pair_max_dy", 8))
    track_max_dist = float(night.get("track_max_distance", 45))
    track_max_missed = int(night.get("track_max_missed", 12))
    merge_dist = float(night.get("merge_distance", 80))
    singleton_min_area = int(night.get("singleton_min_area", 120))

    if "zone" not in cfg:
        print("ERROR: no 'zone' section in config. Run pick_zone.py first.")
        return
    zone = cfg["zone"]
    far_line = zone["far_line"]
    near_line = zone["near_line"]
    match_window = float(zone.get("match_window_s", 3.0))
    match_x_tol = float(zone.get("match_x_tolerance", 80))
    dedup_seconds = float(zone.get("dedup_seconds", 1.0))
    line_name = cfg["lines"][0]["name"] if cfg.get("lines") else "Zone"

    video_path = (Path(cfg_path).parent / cfg["video"]).resolve()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_seconds * fps))

    writer = None
    if debug_video:
        dbg_path = Path(cfg["output"]["xlsx"]).parent / "debug_zone.mp4"
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(dbg_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = CentroidTracker(max_distance=track_max_dist, max_missed=track_max_missed)
    last_pos: dict[int, tuple[float, float]] = {}

    pending_far: list[dict] = []
    pending_near: list[dict] = []
    last_cross_ts: dict[str, float] = {}
    rows: list[dict] = []
    flashes: list[tuple] = []

    frame_idx = int(start_seconds * fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        if max_seconds is not None and (ts - start_seconds) > max_seconds:
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
            if prev is None:
                continue

            crossed_far = crossed_line(prev, curr, far_line)
            crossed_near = crossed_line(prev, curr, near_line)

            def record(side, cx, cy):
                last = last_cross_ts.get(side)
                if last is not None and (ts - last) < dedup_seconds:
                    return False
                last_cross_ts[side] = ts
                rows.append({
                    "timestamp_s": ts, "line": line_name,
                    "track_id": f"Z{tid}",
                    "direction_code": side,
                    "direction": direction_names.get(side, side),
                    "mode": "night",
                })
                if writer is not None:
                    flashes.append((8, f"{direction_names.get(side, side)}",
                                    int(cx), int(cy)))
                return True

            if crossed_far:
                matched = None
                for ev in pending_near:
                    if (not ev["used"] and (ts - ev["ts"]) <= match_window
                            and abs(ev["x"] - curr[0]) <= match_x_tol):
                        matched = ev
                        break
                if matched is not None:
                    matched["used"] = True
                    record("A", curr[0], curr[1])
                else:
                    pending_far.append({"ts": ts, "x": curr[0], "used": False})

            if crossed_near:
                matched = None
                for ev in pending_far:
                    if (not ev["used"] and (ts - ev["ts"]) <= match_window
                            and abs(ev["x"] - curr[0]) <= match_x_tol):
                        matched = ev
                        break
                if matched is not None:
                    matched["used"] = True
                    record("B", curr[0], curr[1])
                else:
                    pending_near.append({"ts": ts, "x": curr[0], "used": False})

        pending_far = [e for e in pending_far
                       if not e["used"] and (ts - e["ts"]) <= match_window]
        pending_near = [e for e in pending_near
                        if not e["used"] and (ts - e["ts"]) <= match_window]

        if writer is not None:
            disp = frame.copy()
            cv2.line(disp, tuple(far_line[0]), tuple(far_line[1]), (0, 165, 255), 2)
            cv2.line(disp, tuple(near_line[0]), tuple(near_line[1]), (0, 255, 0), 2)
            new_flashes = []
            for fr, txt, x, y in flashes:
                cv2.circle(disp, (x, y), 13, (0, 0, 255), 2)
                cv2.putText(disp, txt, (x + 14, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                if fr > 1:
                    new_flashes.append((fr - 1, txt, x, y))
            flashes = new_flashes
            cv2.putText(disp, f"t={ts:.1f}s  counted={len(rows)}",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(disp)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    if df.empty:
        print("No zone transits detected. Check zone lines and thresholds.")
        return

    df["bin_min"] = (df["timestamp_s"] // (bin_minutes * 60)).astype(int) * bin_minutes
    by_bin = df.pivot_table(index="bin_min", columns="direction",
                            values="track_id", aggfunc="count", fill_value=0)
    by_bin["Total"] = by_bin.sum(axis=1)
    totals = df.groupby("direction")["track_id"].count().to_frame("Vehicles")
    totals.loc["TOTAL"] = totals["Vehicles"].sum()

    out = Path(cfg["output"]["xlsx"]).with_name("volume_counts_zone.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        totals.to_excel(xw, sheet_name="Summary")
        by_bin.to_excel(xw, sheet_name="Volume by 15-min")
        df.to_excel(xw, sheet_name="Raw transits", index=False)
    print(f"Wrote {out}")
    print(totals)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--start-seconds", type=float, default=0)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--debug-video", action="store_true",
                   help="Also write annotated debug_zone.mp4")
    args = p.parse_args()
    main(args.config, args.start_seconds, args.max_seconds, args.debug_video)
    
