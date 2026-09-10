"""
Multi-line picker — draw multiple count lines that follow the road perspective.

You click 2 points per line. The default is 3 lines, so 6 clicks total in order:
  line 1 (high - up the road):   point 1, point 2
  line 2 (mid - middle):         point 3, point 4
  line 3 (low - close to camera): point 5, point 6

Each line gets a different colour so you can see them as you go. The saved
config has the lines named line_high, line_mid, line_low (or line_1..N if
you specify --num-lines > 3).

Controls:
    Left-click     - place a point
    R              - reset (start over)
    Enter / Space  - save (only enabled when all points placed)
    Esc / Q        - quit without saving

Usage:
    python pick_lines.py --config ../config/config.yaml --frame 360000
    python pick_lines.py --config ../config/config.yaml --num-lines 2 --frame 360000
"""

import argparse
from pathlib import Path

import cv2
import yaml


# Colour per line index — BGR (orange, yellow, green, cyan, magenta, ...)
LINE_COLORS = [
    (0, 165, 255),   # orange
    (0, 255, 255),   # yellow
    (0, 255, 0),     # green
    (255, 255, 0),   # cyan
    (255, 0, 255),   # magenta
    (200, 200, 200), # white-ish
]


def _line_name(idx: int, num_lines: int) -> str:
    """Use friendly names for 3-line case, generic names otherwise."""
    if num_lines == 3:
        return ["line_high", "line_mid", "line_low"][idx]
    return f"line_{idx + 1}"


def _draw(img, pts, num_lines):
    disp = img.copy()
    n_needed = num_lines * 2
    n_done = len(pts)

    # Instruction text
    if n_done >= n_needed:
        msg = "DONE - press Enter to save, R to redo"
    else:
        line_idx = n_done // 2
        which_pt = (n_done % 2) + 1
        msg = f"Click point {which_pt} of {_line_name(line_idx, num_lines)} (line {line_idx + 1} of {num_lines})"

    for txt, y in [(msg, 22), ("R = reset   Enter = save   Esc = quit", 44)]:
        cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Draw points and complete lines
    for i, p in enumerate(pts):
        line_idx = i // 2
        col = LINE_COLORS[line_idx % len(LINE_COLORS)]
        cv2.circle(disp, p, 6, col, -1)
        cv2.putText(disp, f"{i + 1}", (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # Connect completed pairs
    for line_idx in range(min(num_lines, n_done // 2)):
        p1 = pts[line_idx * 2]
        p2 = pts[line_idx * 2 + 1]
        col = LINE_COLORS[line_idx % len(LINE_COLORS)]
        cv2.line(disp, p1, p2, col, 2)
        cv2.putText(disp, _line_name(line_idx, num_lines), (p1[0], p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    return disp


def pick(frame, num_lines):
    window = "Pick count lines"
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < num_lines * 2:
            pts.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, _draw(frame, pts, num_lines))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            cv2.destroyWindow(window)
            return None
        if key in (ord('r'), ord('R')):
            pts.clear()
        if key in (13, 10, 32) and len(pts) == num_lines * 2:
            cv2.destroyWindow(window)
            return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--frame", type=int, default=0,
                   help="Frame index to display (use daylight frame, e.g. 360000)")
    p.add_argument("--num-lines", type=int, default=3,
                   help="Number of lines to pick (default 3)")
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
    print(f"Click {args.num_lines * 2} points ({args.num_lines} lines, 2 points each).")
    print("Order: highest line first, then mid, then lowest.")
    pts = pick(frame, args.num_lines)
    if pts is None:
        print("Cancelled. Config not changed.")
        return

    # Build the lines list and write to config
    new_lines = []
    for i in range(args.num_lines):
        name = _line_name(i, args.num_lines)
        p1 = list(pts[i * 2])
        p2 = list(pts[i * 2 + 1])
        new_lines.append({"name": name, "points": [p1, p2]})

    cfg["lines"] = new_lines
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Saved {args.num_lines} lines to {cfg_path}:")
    for line in new_lines:
        print(f"  {line['name']}: {line['points']}")


if __name__ == "__main__":
    main()
