"""
Interactive counting-zone picker.

A zone is two lines: a FAR line (higher in the frame, away from the camera)
and a NEAR line (lower in the frame, close to the camera). A vehicle is
counted only when it crosses both, and direction comes from the crossing
order.

You click 4 points in this order:
    1, 2  -> the FAR line   (the one further up the road)
    3, 4  -> the NEAR line  (the one closer to the camera)

Controls:
    Left-click   - place a point
    R            - reset, start over
    Enter/Space  - save to config (needs all 4 points)
    Esc / Q      - quit without saving

Usage:
    python pick_zone.py --config ../config/config.yaml
    python pick_zone.py --config ../config/config.yaml --frame 600
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def _draw(img, pts):
    disp = img.copy()
    h, w = disp.shape[:2]
    labels = ["FAR line - point 1", "FAR line - point 2",
              "NEAR line - point 1", "NEAR line - point 2"]
    msg = "Click: " + (labels[len(pts)] if len(pts) < 4 else "DONE - press Enter to save")
    for txt, y in [(msg, 22), ("R = reset   Enter = save   Esc = quit", 44)]:
        cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

    for i, p in enumerate(pts):
        col = (0, 200, 255) if i < 2 else (0, 255, 0)
        cv2.circle(disp, p, 6, col, -1)
        cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    if len(pts) >= 2:
        cv2.line(disp, pts[0], pts[1], (0, 165, 255), 2)
    if len(pts) >= 4:
        cv2.line(disp, pts[2], pts[3], (0, 255, 0), 2)
        overlay = disp.copy()
        poly = [pts[0], pts[1], pts[3], pts[2]]
        cv2.fillPoly(overlay, [np.array(poly)], (60, 60, 0))
        disp = cv2.addWeighted(overlay, 0.25, disp, 0.75, 0)
    return disp


def pick(frame):
    window = "Pick counting zone"
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, _draw(frame, pts))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            cv2.destroyWindow(window)
            return None
        if key in (ord('r'), ord('R')):
            pts.clear()
        if key in (13, 10, 32) and len(pts) == 4:
            cv2.destroyWindow(window)
            return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--frame", type=int, default=0,
                   help="Frame index to display (use a daylight frame if the "
                        "video starts at night)")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    video_path = (cfg_path.parent / cfg["video"]).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read frame.")
        return

    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    print("Click FAR line (2 pts), then NEAR line (2 pts), then Enter.")
    pts = pick(frame)
    if pts is None:
        print("Cancelled. Config not changed.")
        return

    cfg.setdefault("zone", {})
    cfg["zone"]["far_line"] = [list(pts[0]), list(pts[1])]
    cfg["zone"]["near_line"] = [list(pts[2]), list(pts[3])]
    cfg["zone"].setdefault("match_window_s", 3.0)
    cfg["zone"].setdefault("match_x_tolerance", 80)
    cfg["zone"].setdefault("dedup_seconds", 1.0)

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Saved zone to {cfg_path}")
    print(f"  far_line  = {cfg['zone']['far_line']}")
    print(f"  near_line = {cfg['zone']['near_line']}")


if __name__ == "__main__":
    main()
