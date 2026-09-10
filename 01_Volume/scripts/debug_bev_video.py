"""
Debug visualizer for BEV traffic counting.

Generates a side-by-side mp4 showing exactly what count_volume_bev.py sees:
  LEFT:  original frame with YOLO detection boxes, track IDs, count line,
         red foot-of-bbox dots, count flashes when a vehicle crosses
  RIGHT: bird's-eye-view (BEV) warped view with the same count line
         projected to BEV, and dots marking each tracked vehicle's foot
         position in BEV space

What the colours mean:
  GREEN  detection box   - track is old enough and big enough to count
  GRAY   detection box   - track is filtered out (too new or too small)
  RED    dot             - foot-of-bbox (the point that gets projected to BEV)
  RED    circle + label  - a COUNT happened on this vehicle
  YELLOW dot in BEV      - a vehicle that has already been counted
  GREEN  dot in BEV      - a vehicle still in play (not yet counted)

Use this to visually inspect:
  - Are vehicles being detected reliably?
  - Are tracks stable (low fragmentation)?
  - Are line crossings being registered correctly?
  - Is the BEV transform sensible?

Usage:
    # Daytime, 60s starting at 10am (36000s into the video)
    python debug_bev_video.py --config ../config/config.yaml --start-seconds 36000 --duration 60

    # Night, 30s starting at 2am
    python debug_bev_video.py --config ../config/config.yaml --start-seconds 7200 --duration 30

    # Custom output path
    python debug_bev_video.py --config ../config/config.yaml --output ../outputs/my_debug.mp4
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils.zones import crossed_line


def bbox_foot(xyxy):
    """Bottom-middle of bbox - where wheels meet the road."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, y2)


def bbox_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def project_point(pt, H):
    a = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    b = cv2.perspectiveTransform(a, H)
    return (float(b[0, 0, 0]), float(b[0, 0, 1]))


def line_normal_side(prev, curr, line):
    (x1, y1), (x2, y2) = line[0], line[1]
    px, py = prev
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return "A" if cross > 0 else "B"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--start-seconds", type=float, default=36000,
                   help="Where in the video to start (default 36000 = 10am, daytime)")
    p.add_argument("--duration", type=float, default=60,
                   help="How many seconds of video to render (default 60)")
    p.add_argument("--output", default="../outputs/debug_bev.mp4",
                   help="Output mp4 path (relative to scripts dir)")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())

    if "homography" not in cfg:
        print("ERROR: no 'homography' in config. Run pick_homography.py first.")
        return
    H = np.array(cfg["homography"]["matrix"], dtype=np.float32)
    bev_W, bev_H = cfg["homography"]["bev_size"]

    direction_names = cfg.get("direction_names", {"A": "Direction A", "B": "Direction B"})
    lines = cfg["lines"]

    # Project count lines to BEV once
    lines_bev = []
    for line in lines:
        p1_bev = project_point(line["points"][0], H)
        p2_bev = project_point(line["points"][1], H)
        lines_bev.append({"name": line["name"], "points": [p1_bev, p2_bev]})

    # Filters (same as count_volume_bev.py)
    min_track_age = int(cfg.get("min_track_age_frames", 2))
    min_area = float(cfg.get("min_bbox_area_px", 400))
    dedup_seconds = float(cfg.get("dedup_seconds", 1.5))

    # Open video
    video_path = (cfg_path.parent / cfg["video"]).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {W}x{H_frame} @ {fps:.2f} fps")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_seconds * fps))

    # Output canvas: scaled original next to BEV, both at BEV height
    target_H = bev_H
    scaled_W = int(W * (target_H / H_frame))
    out_W = scaled_W + bev_W + 20
    out_H = target_H
    output_path = (Path(__file__).parent / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_W, out_H))
    print(f"Output canvas: {out_W}x{out_H}")

    # Load YOLO
    from ultralytics import YOLO
    print(f"Loading model: {cfg['model']}")
    model = YOLO(cfg["model"])
    conf = float(cfg.get("conf", 0.25))
    classes = cfg.get("classes")

    # Counter state
    last_bev = {}
    last_seen_frame = {}
    track_age = defaultdict(int)
    counted = set()
    last_cross_ts = {}
    count_A = 0
    count_B = 0
    flashes = []  # (frames_remaining, side, x, y) in original image coords

    n_frames = int(args.duration * fps)
    frame_idx_start = int(args.start_seconds * fps)
    print(f"\nProcessing {n_frames} frames starting at frame {frame_idx_start} "
          f"(t={args.start_seconds:.0f}s)")
    print(f"Writing to {output_path}\n")

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"End of video at frame {i}")
            break
        ts = (frame_idx_start + i) / fps

        # Run YOLO with persistent tracking
        results = model.track(frame, persist=True, conf=conf, classes=classes,
                              tracker="bytetrack.yaml", verbose=False)

        disp = frame.copy()

        if (results and results[0].boxes is not None
                and results[0].boxes.id is not None):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clses = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, clses):
                x1, y1, x2, y2 = box.astype(int)
                track_age[tid] += 1
                last_seen_frame[tid] = i

                area = bbox_area(box)
                is_valid = area >= min_area and track_age[tid] >= min_track_age

                color = (0, 255, 0) if is_valid else (128, 128, 128)
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                cv2.putText(disp, f"ID{tid}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Foot point (red dot on original)
                foot_img = bbox_foot(box)
                cv2.circle(disp, (int(foot_img[0]), int(foot_img[1])),
                           4, (0, 0, 255), -1)

                if not is_valid:
                    continue

                # Project to BEV
                curr_bev = project_point(foot_img, H)
                prev_bev = last_bev.get(tid)
                last_bev[tid] = curr_bev
                if prev_bev is None:
                    continue
                if tid in counted:
                    continue

                # Check crossings
                for line_bev in lines_bev:
                    if not crossed_line(prev_bev, curr_bev, line_bev["points"]):
                        continue
                    side = line_normal_side(prev_bev, curr_bev, line_bev["points"])
                    last_ts = last_cross_ts.get(side)
                    if last_ts is not None and (ts - last_ts) < dedup_seconds:
                        break
                    counted.add(tid)
                    last_cross_ts[side] = ts
                    if side == "A":
                        count_A += 1
                    else:
                        count_B += 1
                    flashes.append((12, side, int(foot_img[0]), int(foot_img[1])))
                    break

        # Draw count line on original (orange)
        for line in lines:
            (lx1, ly1), (lx2, ly2) = line["points"][0], line["points"][1]
            cv2.line(disp, (int(lx1), int(ly1)), (int(lx2), int(ly2)),
                     (0, 165, 255), 2)
            cv2.putText(disp, line["name"], (int(lx1), max(15, int(ly1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        # Draw flashes (count events)
        new_flashes = []
        for fr, side, x, y in flashes:
            cv2.circle(disp, (x, y), 22, (0, 0, 255), 3)
            cv2.putText(disp, direction_names.get(side, side), (x + 24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            if fr > 1:
                new_flashes.append((fr - 1, side, x, y))
        flashes = new_flashes

        # Counter overlay (bottom of original)
        info = f"A: {count_A}   B: {count_B}   T: {count_A + count_B}   t={ts:.1f}s"
        cv2.putText(disp, info, (10, H_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(disp, info, (10, H_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Build BEV view from actual frame
        bev_frame = cv2.warpPerspective(frame, H, (bev_W, bev_H))

        # Count line on BEV
        for line_bev in lines_bev:
            (x1, y1), (x2, y2) = line_bev["points"][0], line_bev["points"][1]
            cv2.line(bev_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 165, 255), 2)

        # Draw recently-tracked foot points on BEV
        for tid, bev_pt in last_bev.items():
            if last_seen_frame.get(tid, -100) < i - 5:
                continue  # stale, hide it
            if not (0 <= bev_pt[0] < bev_W and 0 <= bev_pt[1] < bev_H):
                continue
            color = (0, 255, 255) if tid in counted else (0, 255, 0)
            cv2.circle(bev_frame, (int(bev_pt[0]), int(bev_pt[1])),
                       6, color, -1)
            cv2.putText(bev_frame, f"{tid}",
                        (int(bev_pt[0]) + 8, int(bev_pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Scale original to BEV height
        disp_scaled = cv2.resize(disp, (scaled_W, target_H))

        # Compose side-by-side
        canvas = np.zeros((out_H, out_W, 3), dtype=np.uint8)
        canvas[:, :scaled_W] = disp_scaled
        canvas[:, scaled_W + 20:] = bev_frame

        # Labels at top of each panel
        cv2.putText(canvas, "Original + detections", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Original + detections", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "BEV (top-down)", (scaled_W + 30, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, "BEV (top-down)", (scaled_W + 30, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(canvas)

        if (i + 1) % 50 == 0:
            print(f"  frame {i+1}/{n_frames}  A={count_A}  B={count_B}  T={count_A+count_B}")

    writer.release()
    cap.release()
    print(f"\nDone. Wrote {output_path}")
    print(f"Final counts in this {args.duration:.0f}s segment: "
          f"A={count_A}  B={count_B}  TOTAL={count_A + count_B}")


if __name__ == "__main__":
    main()
