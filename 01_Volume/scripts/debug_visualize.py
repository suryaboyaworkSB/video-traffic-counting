"""
Debug visualizer — re-runs detection over a short window of video and writes
an annotated .mp4 with:
  - count line in yellow
  - all tracked bboxes with track_id labels
  - a flash + label at the moment a track crosses the line

Usage:
    python debug_visualize.py --config ../config/config.yaml --max-seconds 60

Output: ../outputs/debug_visualization.mp4
"""

import argparse
from pathlib import Path

import cv2
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils.zones import bbox_center, crossed_line
from ultralytics import YOLO


def main(cfg_path: str, max_seconds: float = 60.0) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    video_path = Path(__file__).resolve().parent.parent / "videos" / Path(cfg["video"]).name
    if not video_path.exists():
        # fall back to the path as written in config (relative to script)
        video_path = Path(cfg["video"])
    out_path = Path(cfg["output"]["xlsx"]).parent / "debug_visualization.mp4"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(cfg["model"])
    last_pos: dict[int, tuple[float, float]] = {}
    counted: set[int] = set()
    flashes = []  # list of (frame_remaining, text, xy)

    line = cfg["lines"][0]["points"]
    p1 = tuple(line[0])
    p2 = tuple(line[1])

    stream = model.track(
        source=str(video_path), conf=cfg["conf"], iou=0.5,
        tracker="bytetrack.yaml", classes=cfg.get("classes"),
        persist=True, stream=True, verbose=False,
    )

    for frame_idx, result in enumerate(stream):
        ts = frame_idx / fps
        if ts > max_seconds:
            break
        frame = result.orig_img.copy()

        # draw count line
        cv2.line(frame, p1, p2, (0, 255, 255), 2)

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            xyxy = result.boxes.xyxy.cpu().tolist()
            for tid, box in zip(ids, xyxy):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
                cv2.putText(frame, f"#{tid}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
                curr = bbox_center(box)
                prev = last_pos.get(tid)
                last_pos[tid] = curr
                if prev is not None and tid not in counted:
                    if crossed_line(prev, curr, line):
                        counted.add(tid)
                        flashes.append((6, f"CROSS #{tid}", (int(curr[0]), int(curr[1]))))

        # render flashes
        new_flashes = []
        for fr, text, (x, y) in flashes:
            cv2.circle(frame, (x, y), 12, (0, 0, 255), 2)
            cv2.putText(frame, text, (x + 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            if fr > 1:
                new_flashes.append((fr - 1, text, (x, y)))
        flashes = new_flashes

        cv2.putText(frame, f"t={ts:5.2f}s  counted={len(counted)}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-seconds", type=float, default=60.0)
    args = p.parse_args()
    main(args.config, args.max_seconds)
