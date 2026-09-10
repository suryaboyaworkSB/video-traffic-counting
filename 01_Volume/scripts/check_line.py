"""
Render the saved count line(s) on top of a chosen video frame and save as PNG.

Use this to visually verify the line is in the right spot BEFORE running a
long count. Faster than launching the interactive picker just to look.

Usage:
    python check_line.py --config ../config/config.yaml
    python check_line.py --config ../config/config.yaml --frame 576000
    python check_line.py --config ../config/config.yaml --frame 576000 --output ../outputs/line_check.png
"""

import argparse
from pathlib import Path

import cv2
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--frame", type=int, default=576000,
                   help="Frame index to render (default 576000 = 4 PM at 10 fps)")
    p.add_argument("--output", default="../outputs/line_check.png",
                   help="Where to save the PNG (relative to scripts dir)")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    video_path = (cfg_path.parent / cfg["video"]).resolve()

    cap = cv2.VideoCapture(str(video_path))
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read frame {args.frame}")
        return

    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    disp = frame.copy()
    lines = cfg.get("lines", [])
    if not lines:
        print("No lines in config — nothing to draw.")
        return

    # Colours for multiple lines (BGR)
    colors = [(0, 165, 255), (0, 255, 255), (0, 255, 0),
              (255, 255, 0), (255, 0, 255)]

    for i, line in enumerate(lines):
        col = colors[i % len(colors)]
        (x1, y1), (x2, y2) = line["points"][0], line["points"][1]
        cv2.line(disp, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.circle(disp, (int(x1), int(y1)), 5, col, -1)
        cv2.circle(disp, (int(x2), int(y2)), 5, col, -1)
        label = f"{line['name']}  ({x1},{y1}) -> ({x2},{y2})"
        cv2.putText(disp, label, (int(x1) + 8, int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        print(f"Drew line {line['name']} from ({x1},{y1}) to ({x2},{y2})")

    out_path = (Path(__file__).parent / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), disp)
    print(f"\nSaved {out_path}")
    print("Open the PNG to verify the line is where you want it.")


if __name__ == "__main__":
    main()
