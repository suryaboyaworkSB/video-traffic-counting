"""
Night-mode volume counter.

YOLO cannot see vehicle bodies in pitch-dark footage. We detect vehicles
indirectly via their lights:

  - Oncoming vehicles  ->  pair of bright white/yellow headlights
  - Departing vehicles ->  pair of red tail lights (often dimmer, smaller)

Each candidate blob pair is treated as a vehicle. We track pairs across
frames by nearest-centroid matching and count when the pair's centroid
crosses the configured line.

Output: same Excel layout as count_volume.py so day/night results can be
merged later.

Usage:
    python count_volume_night.py --config ../config/config.yaml --start-seconds 19000 --max-seconds 19060
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

from shared.utils.zones import crossed_line


def _line_normal_side(prev, curr, line):
    """Returns 'A' or 'B' depending on which side of the line `prev` was on."""
    (x1, y1), (x2, y2) = line[0], line[1]
    px, py = prev
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return "A" if cross > 0 else "B"


# ---------- light detection ----------------------------------------------------

def _bloom_suppress(roi_bgr: np.ndarray, clip: int = 200) -> np.ndarray:
    out = roi_bgr.copy()
    mask = (out.max(axis=2) > clip)
    if mask.any():
        out[mask] = (out[mask].astype(np.float32) *
                     (clip / 255.0)).clip(0, 255).astype(np.uint8)
    return out


def detect_headlights(frame_bgr: np.ndarray, roi_y_min: int,
                      use_bloom_suppression: bool = True) -> list[dict]:
    roi = frame_bgr[roi_y_min:, :]
    rois = [roi]
    if use_bloom_suppression:
        rois.append(_bloom_suppress(roi, clip=180))

    out: list[dict] = []
    seen: list[tuple[float, float]] = []
    for r in rois:
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 3 or area > 4000:
                continue
            cx, cy = centroids[i]
            if any((cx - sx) ** 2 + (cy - sy) ** 2 < 25 for sx, sy in seen):
                continue
            seen.append((float(cx), float(cy)))
            out.append({"cx": float(cx), "cy": float(cy) + roi_y_min,
                        "area": int(area), "kind": "head"})
    return out


class MotionBuffer:
    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha
        self.bg: np.ndarray | None = None

    def diff(self, gray: np.ndarray) -> np.ndarray:
        if self.bg is None:
            self.bg = gray.astype(np.float32)
            return np.zeros_like(gray)
        cv2.accumulateWeighted(gray, self.bg, self.alpha)
        bg_u8 = cv2.convertScaleAbs(self.bg)
        return cv2.subtract(gray, bg_u8)


def detect_motion_lights(frame_bgr: np.ndarray, roi_y_min: int,
                         motion_buf: MotionBuffer,
                         diff_threshold: int = 60) -> list[dict]:
    roi = frame_bgr[roi_y_min:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    diff = motion_buf.diff(gray)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4 or area > 4000:
            continue
        cx, cy = centroids[i]
        out.append({"cx": float(cx), "cy": float(cy) + roi_y_min,
                    "area": int(area), "kind": "motion"})
    return out


def detect_taillights(frame_bgr: np.ndarray, roi_y_min: int) -> list[dict]:
    roi = frame_bgr[roi_y_min:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 60, 80]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([168, 60, 80]), np.array([180, 255, 255]))
    red = m1 | m2
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2 or area > 1500:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if max(bw, bh) > 8 * max(1, min(bw, bh)):
            continue
        cx, cy = centroids[i]
        out.append({"cx": float(cx), "cy": float(cy) + roi_y_min,
                    "area": int(area), "kind": "tail"})
    return out


def suppress_tails_near_heads(head_pairs: list[dict], tail_pairs: list[dict],
                              min_separation: float = 50.0) -> list[dict]:
    kept = []
    for tp in tail_pairs:
        too_close = False
        for hp in head_pairs:
            d = ((tp["cx"] - hp["cx"]) ** 2 + (tp["cy"] - hp["cy"]) ** 2) ** 0.5
            if d < min_separation:
                too_close = True
                break
        if not too_close:
            kept.append(tp)
    return kept


# ---------- blob pairing -------------------------------------------------------

def pair_lights(blobs: list[dict], max_dx: float, max_dy: float,
                area_ratio_max: float = 3.0,
                single_min_area: int = 80) -> list[dict]:
    blobs = sorted(blobs, key=lambda b: b["cx"])
    used = set()
    pairs = []
    for i, a in enumerate(blobs):
        if i in used:
            continue
        for j in range(i + 1, len(blobs)):
            if j in used:
                continue
            b = blobs[j]
            dx = b["cx"] - a["cx"]
            dy = abs(b["cy"] - a["cy"])
            if dx <= 0 or dx > max_dx:
                continue
            if dy > max_dy:
                continue
            ratio = max(a["area"], b["area"]) / max(1, min(a["area"], b["area"]))
            if ratio > area_ratio_max:
                continue
            cx = (a["cx"] + b["cx"]) / 2
            cy = (a["cy"] + b["cy"]) / 2
            pairs.append({"cx": cx, "cy": cy, "dx": dx,
                          "kind": a["kind"], "left": a, "right": b})
            used.add(i)
            used.add(j)
            break

    for i, a in enumerate(blobs):
        if i in used:
            continue
        if a["area"] < single_min_area:
            continue
        pairs.append({"cx": a["cx"], "cy": a["cy"], "dx": 0.0,
                      "kind": a["kind"], "left": a, "right": a})
    return pairs


# ---------- simple centroid tracker -------------------------------------------

class CentroidTracker:
    def __init__(self, max_distance: float = 40.0, max_missed: int = 5):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, pairs: list[dict]) -> list[tuple[int, dict]]:
        for t in self.tracks.values():
            t["missed"] += 1

        assignments: list[tuple[int, dict]] = []
        unmatched = list(range(len(pairs)))

        for tid, t in sorted(self.tracks.items(), key=lambda x: x[1]["missed"]):
            best = None
            best_d = self.max_distance + 1
            for k in unmatched:
                p = pairs[k]
                d = ((p["cx"] - t["cx"]) ** 2 + (p["cy"] - t["cy"]) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best = k
            if best is not None:
                p = pairs[best]
                t["cx"], t["cy"] = p["cx"], p["cy"]
                t["missed"] = 0
                t["history"].append((p["cx"], p["cy"]))
                assignments.append((tid, p))
                unmatched.remove(best)

        for k in unmatched:
            p = pairs[k]
            self.tracks[self.next_id] = {
                "cx": p["cx"], "cy": p["cy"], "kind": p["kind"],
                "missed": 0, "history": [(p["cx"], p["cy"])],
            }
            assignments.append((self.next_id, p))
            self.next_id += 1

        for tid in list(self.tracks):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]
        return assignments


# ---------- main ---------------------------------------------------------------

def main(cfg_path: str, start_seconds: float = 0,
         max_seconds: float | None = None,
         debug_video: bool = False) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    bin_minutes = cfg.get("bin_minutes", 15)
    direction_names = cfg.get("direction_names", {"A": "Direction A", "B": "Direction B"})

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

    video_path = (Path(cfg_path).parent / cfg["video"]).resolve()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_seconds * fps))

    writer = None
    if debug_video:
        dbg_path = Path(cfg["output"]["xlsx"]).parent / "debug_night.mp4"
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(dbg_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = CentroidTracker(max_distance=track_max_dist, max_missed=track_max_missed)
    last_pos: dict[int, tuple[float, float]] = {}
    counted: set[int] = set()
    last_cross_ts: dict[str, float] = {}
    rows = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = start_seconds + (frame_idx / fps)
        if max_seconds is not None and (ts - start_seconds) > max_seconds:
            break

        heads = detect_headlights(frame, roi_y_min, use_bloom_suppression=True)
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
        assignments = tracker.update(all_pairs)

        for tid, p in assignments:
            curr = (p["cx"], p["cy"])
            prev = last_pos.get(tid)
            last_pos[tid] = curr
            if prev is not None and tid not in counted:
                if crossed_line(prev, curr, line):
                    side = _line_normal_side(prev, curr, line)
                    last_ts = last_cross_ts.get(side)
                    if last_ts is not None and (ts - last_ts) < dedup_seconds:
                        continue
                    counted.add(tid)
                    last_cross_ts[side] = ts
                    rows.append({
                        "timestamp_s": ts,
                        "line": line_name,
                        "track_id": tid,
                        "direction_code": side,
                        "direction": direction_names.get(side, side),
                        "kind": p.get("kind", "vehicle"),
                    })

        if writer is not None:
            disp = frame.copy()
            cv2.line(disp, tuple(line[0]), tuple(line[1]), (0, 255, 255), 2)
            for tid, p in assignments:
                col = (0, 255, 0) if p["kind"] == "head" else (0, 0, 255)
                cv2.circle(disp, (int(p["cx"]), int(p["cy"])), 8, col, 2)
                cv2.putText(disp, f"#{tid}", (int(p["cx"]) + 10, int(p["cy"])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            cv2.putText(disp, f"t={ts:.1f}s  counted={len(counted)}",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(disp)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    if df.empty:
        print("No night-mode crossings detected. Check ROI and threshold settings.")
        return

    df["bin_min"] = (df["timestamp_s"] // (bin_minutes * 60)).astype(int) * bin_minutes
    by_bin = df.pivot_table(index="bin_min", columns="direction",
                            values="track_id", aggfunc="count", fill_value=0)
    by_bin["Total"] = by_bin.sum(axis=1)
    totals = df.groupby("direction")["track_id"].nunique().to_frame("Vehicles")
    totals.loc["TOTAL"] = totals["Vehicles"].sum()

    out = Path(cfg["output"]["xlsx"]).with_name("volume_counts_night.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        totals.to_excel(xw, sheet_name="Summary")
        by_bin.to_excel(xw, sheet_name="Volume by 15-min")
        df.to_excel(xw, sheet_name="Raw crossings", index=False)
    print(f"Wrote {out}")
    print(totals)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--start-seconds", type=float, default=0)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--debug-video", action="store_true",
                   help="Also write annotated debug_night.mp4")
    args = p.parse_args()
    main(args.config, args.start_seconds, args.max_seconds, args.debug_video)
