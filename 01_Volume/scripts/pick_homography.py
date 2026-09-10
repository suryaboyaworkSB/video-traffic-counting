"""
Homography picker for bird's-eye-view (BEV) traffic counting.

You click 4 points on a daytime frame that form a known rectangle on the
road surface — typically the 4 corners of a lane segment between two clearly
visible road markings. The script computes a perspective transform that maps
the oblique camera view to a top-down BEV view.

CLICK ORDER (important):
    1. Top-Left      — far end of the rectangle, left edge
    2. Top-Right     — far end of the rectangle, right edge
    3. Bottom-Right  — close end of the rectangle, right edge
    4. Bottom-Left   — close end of the rectangle, left edge

Tip: pick a road segment that's clearly rectangular in reality. A common
choice is two lane stripes (forming the long sides) and your best visual
estimate of two perpendicular lines across the road (the short sides).

After picking, the script shows you a warped preview so you can verify the
homography is good. The rectangle should look straight and rectangular in
the warped view.

Usage:
    python pick_homography.py --config ../config/config.yaml --frame 360000
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def _draw(img, pts):
    disp = img.copy()
    h, w = disp.shape[:2]
    labels = ["Top-Left (far-left)", "Top-Right (far-right)",
              "Bottom-Right (near-right)", "Bottom-Left (near-left)"]
    msg = "Click: " + (labels[len(pts)] if len(pts) < 4
                       else "DONE - press Enter to preview & save")
    for txt, y in [(msg, 22), ("R = reset   Enter = save   Esc = quit", 44)]:
        cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    colors = [(0, 200, 255), (0, 255, 200), (200, 0, 255), (255, 200, 0)]
    for i, p in enumerate(pts):
        cv2.circle(disp, p, 7, colors[i], -1)
        cv2.putText(disp, f"{i+1}", (p[0] + 9, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1, cv2.LINE_AA)
    if len(pts) == 4:
        poly = np.array(pts, dtype=np.int32)
        overlay = disp.copy()
        cv2.fillPoly(overlay, [poly], (60, 100, 60))
        disp = cv2.addWeighted(overlay, 0.25, disp, 0.75, 0)
        cv2.polylines(disp, [poly], True, (0, 255, 0), 2)
    return disp


def pick(frame):
    window = "Pick homography rectangle"
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
                   help="Frame index to display (use daylight, e.g. 360000)")
    p.add_argument("--bev-width", type=int, default=300,
                   help="Width of the BEV destination rectangle in pixels (default 300)")
    p.add_argument("--bev-height", type=int, default=600,
                   help="Height of the BEV destination rectangle in pixels (default 600). "
                        "Aspect ratio should reflect the real road segment.")
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
    print("Click 4 corners in order: TL, TR, BR, BL (clockwise from top-left).")
    pts = pick(frame)
    if pts is None:
        print("Cancelled. Config not changed.")
        return

    src = np.array(pts, dtype=np.float32)
    W, H = args.bev_width, args.bev_height
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)

    # Show preview of warped frame
    warped = cv2.warpPerspective(frame, homography, (W, H))
    preview = "Homography preview (close to continue)"
    cv2.namedWindow(preview, cv2.WINDOW_NORMAL)
    cv2.imshow(preview, warped)
    print("Preview shown. Close the window to confirm and save.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save homography to config
    cfg["homography"] = {
        "src_points": [[int(p[0]), int(p[1])] for p in pts],
        "bev_size": [W, H],
        "matrix": homography.tolist(),  # 3x3 list of lists
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Saved homography to {cfg_path}")
    print(f"  Source points: {cfg['homography']['src_points']}")
    print(f"  BEV size: {W}x{H}")
    print()
    print("Next step: run count_volume_bev.py --config ..\\config\\config.yaml")


if __name__ == "__main__":
    main()
