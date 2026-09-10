"""
Interactive count-line picker.

Opens the first frame (or a chosen frame) of the video, lets you click two
points to define the count line, then writes the line coordinates back to
config.yaml.

Controls:
    Left-click             - place a point (first click = line start, second = end)
    R                      - reset, start over
    Enter / Space          - confirm and save to config
    Esc / Q                - quit without saving

Usage:
    python pick_count_line.py --config ../config/config.yaml
    python pick_count_line.py --config ../config/config.yaml --frame 0
    python pick_count_line.py --config ../config/config.yaml --frame 600     # use frame 600 instead of first

You can pass --line-name to update a specific named line in a config with
multiple lines (defaults to overwriting the first line).
"""

import argparse
from pathlib import Path

import cv2
import yaml


def _draw_overlay(img, points, line_name):
    """Render the click points, the connecting line, and instructions."""
    h, w = img.shape[:2]
    disp = img.copy()

    # instructions banner
    msgs = [
        "Click 2 points to draw count line",
        "R = reset   Enter = save   Esc = quit",
        f"Line name: {line_name}",
    ]
    if len(points) == 1:
        msgs[0] = "Click the SECOND point (or R to reset)"
    elif len(points) == 2:
        msgs[0] = "OK! Press Enter to save, R to redo"

    y = 22
    for m in msgs:
        cv2.putText(disp, m, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, m, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

    # draw clicked points and line
    for i, p in enumerate(points):
        cv2.circle(disp, p, 7, (0, 0, 255), -1)
        cv2.putText(disp, f"P{i+1} {p}", (p[0] + 10, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    if len(points) == 2:
        cv2.line(disp, points[0], points[1], (0, 255, 255), 3)

    return disp


def pick_line_interactive(frame, line_name: str = "Main") -> list[tuple[int, int]] | None:
    """Show the frame in a window, return the two clicked points or None."""
    window = "Pick count line"
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        disp = _draw_overlay(frame, points, line_name)
        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q'), ord('Q')):       # Esc or Q
            cv2.destroyWindow(window)
            return None
        if key in (ord('r'), ord('R')):
            points.clear()
        if key in (13, 10, 32) and len(points) == 2:   # Enter or Space
            cv2.destroyWindow(window)
            return points
    # unreachable
    cv2.destroyWindow(window)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--frame", type=int, default=0,
                   help="Frame index to display (default 0 = first frame). "
                        "Useful if first frame is blank/dark.")
    p.add_argument("--line-name", default="Main",
                   help="Which line in config to update (default: 'Main' or first line)")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())

    # Resolve video path relative to config
    video_path = (cfg_path.parent / cfg["video"]).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Could not read frame from video.")
        return

    h, w = frame.shape[:2]
    print(f"Video frame size: {w}x{h}")
    print(f"Showing frame index {args.frame}.")
    print("Click two points to define the count line, then press Enter to save.")

    points = pick_line_interactive(frame, line_name=args.line_name)
    if points is None:
        print("Cancelled. Config not modified.")
        return

    # Update config in-place. Either replace the named line or the first one.
    lines = cfg.get("lines", [])
    target_idx = None
    for i, line in enumerate(lines):
        if line.get("name") == args.line_name:
            target_idx = i
            break
    if target_idx is None:
        target_idx = 0 if lines else None

    new_line = {"name": args.line_name,
                "points": [list(points[0]), list(points[1])]}
    if target_idx is None:
        cfg["lines"] = [new_line]
    else:
        cfg["lines"][target_idx] = new_line

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Wrote line {new_line['name']} = {new_line['points']} to {cfg_path}")
    print("Config updated. Re-run count_volume.py to use the new line.")


if __name__ == "__main__":
    main()
