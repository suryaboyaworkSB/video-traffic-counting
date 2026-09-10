"""
Zone-based volume counter (DAY mode, using YOLO detection).

Combines YOLO detection (reliable daytime vehicle finder) with the two-line
zone transit logic from count_volume_zone.py.

Why this should beat v5 single-line at day:
  - YOLO catches the vehicles (no detection bottleneck like night)
  - Zone transit gives unambiguous direction (crossing order)
  - Zone transit also filters edge-of-frame false positives that the single
    line might count (vehicle barely entered, crossed line once, but didn't
    travel through the road segment)

Usage:
    python count_volume_zone_day.py --config ../config/config.yaml
    python count_volume_zone_day.py --config ../config/config.yaml --max-seconds 3600
    python count_volume_zone_day.py --config ../config/config.yaml --max-seconds 600 --debug-video
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils.zones import crossed_line, bbox_center
from shared.utils.tracker import track_video


def main(cfg_path: str, max_seconds: float | None = None,
         debug_video: bool = False) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    direction_names = cfg.get("direction_names", {"A": "Direction A", "B": "Direction B"})
    bin_minutes = cfg.get("bin_minutes", 15)

    if "zone" not in cfg:
        print("ERROR: no 'zone' section in config. Run pick_zone.py first.")
        return
    zone = cfg["zone"]
    far_line = zone["far_line"]
    near_line = zone["near_line"]
    match_window = float(zone.get("match_window_s", 3.0))
    match_x_tol = float(zone.get("match_x_tolerance", 200))
    dedup_seconds = float(zone.get("dedup_seconds", 1.0))
    line_name = cfg["lines"][0]["name"] if cfg.get("lines") else "Zone"

    # Optional anti-false-positive filters (same as day single-line script)
    min_track_age = int(cfg.get("min_track_age_frames", 2))
    min_area = float(cfg.get("min_bbox_area_px", 400))

    # Optional debug video
    writer = None
    if debug_video:
        cap_tmp = cv2.VideoCapture(str((Path(cfg_path).parent / cfg["video"]).resolve()))
        fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 10
        w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_tmp.release()
        dbg_path = Path(cfg["output"]["xlsx"]).parent / "debug_zone_day.mp4"
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(dbg_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # State for the matching algorithm
    last_pos: dict[int, tuple[float, float]] = {}
    track_age: dict[int, int] = {}
    pending_far: list[dict] = []
    pending_near: list[dict] = []
    last_cross_ts: dict[str, float] = {}
    rows: list[dict] = []
    flashes: list[tuple] = []

    # We need frames for the debug video. The tracker iterator processes the
    # video for us; we'll re-open the video for frames in parallel.
    if debug_video:
        cap_dbg = cv2.VideoCapture(str((Path(cfg_path).parent / cfg["video"]).resolve()))

    last_frame_idx = -1
    for det in track_video(cfg["video"], model_path=cfg["model"],
                           conf=cfg["conf"], classes=cfg.get("classes")):
        if max_seconds is not None and det["timestamp_s"] > max_seconds:
            break

        tid = det["track_id"]
        track_age[tid] = track_age.get(tid, 0) + 1
        if track_age[tid] < min_track_age:
            continue

        # filter by bbox area
        x1, y1, x2, y2 = det["xyxy"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_area:
            continue

        curr = bbox_center(det["xyxy"])
        prev = last_pos.get(tid)
        last_pos[tid] = curr
        if prev is None:
            continue

        ts = det["timestamp_s"]
        crossed_far = crossed_line(prev, curr, far_line)
        crossed_near = crossed_line(prev, curr, near_line)

        def record(side, cx, cy):
            last = last_cross_ts.get(side)
            if last is not None and (ts - last) < dedup_seconds:
                return False
            last_cross_ts[side] = ts
            rows.append({
                "timestamp_s": ts, "line": line_name,
                "track_id": f"D{tid}",
                "direction_code": side,
                "direction": direction_names.get(side, side),
                "mode": "day",
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

        # Purge stale unmatched events
        pending_far = [e for e in pending_far
                       if not e["used"] and (ts - e["ts"]) <= match_window]
        pending_near = [e for e in pending_near
                        if not e["used"] and (ts - e["ts"]) <= match_window]

        # Debug video: write annotated frame (only when frame_idx advances)
        if debug_video and det["frame_idx"] != last_frame_idx:
            ret, frame = cap_dbg.read()
            if ret:
                disp = frame.copy()
                cv2.line(disp, tuple(far_line[0]), tuple(far_line[1]), (0, 165, 255), 2)
                cv2.line(disp, tuple(near_line[0]), tuple(near_line[1]), (0, 255, 0), 2)
                # draw current bbox
                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 200, 0), 1)
                cv2.putText(disp, f"#{tid}", (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
                new_flashes = []
                for fr, txt, fx, fy in flashes:
                    cv2.circle(disp, (fx, fy), 13, (0, 0, 255), 2)
                    cv2.putText(disp, txt, (fx + 14, fy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    if fr > 1:
                        new_flashes.append((fr - 1, txt, fx, fy))
                flashes = new_flashes
                cv2.putText(disp, f"t={ts:.1f}s  counted={len(rows)}",
                            (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(disp)
            last_frame_idx = det["frame_idx"]

    if writer is not None:
        cap_dbg.release()
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

    out = Path(cfg["output"]["xlsx"]).with_name("volume_counts_zone_day.xlsx")
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
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--debug-video", action="store_true",
                   help="Also write annotated debug_zone_day.mp4")
    args = p.parse_args()
    main(args.config, args.max_seconds, args.debug_video)
