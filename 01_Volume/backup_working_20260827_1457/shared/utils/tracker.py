"""
Thin wrapper around Ultralytics' built-in tracker.

Each section (volume, class, TMC) reads detections + persistent IDs from this
module so we count each vehicle exactly once.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
from ultralytics import YOLO


def track_video(
    video_path: str | Path,
    model_path: str = "yolov8n.pt",
    conf: float = 0.35,
    iou: float = 0.5,
    tracker: str = "bytetrack.yaml",
    classes: list[int] | None = None,
    start_seconds: float = 0.0,
) -> Iterator[dict]:
    """
    Yield per-frame tracking results.

    Each yielded dict contains:
      frame_idx, timestamp_s, track_id, class_id, conf, xyxy (bbox)

    NOTE on timestamp: we read the video FPS via OpenCV (CAP_PROP_FPS).
    Ultralytics' `result.speed` reports processing speeds, NOT video FPS,
    so we cannot rely on it for timestamps.

    If start_seconds > 0, the video is seeked forward to that timestamp
    before tracking begins. This avoids burning CPU on frames you don't
    want counted (e.g., testing hour 16 without running hours 0-15 first).
    """
    # Read FPS from video header (Ultralytics' result.speed is processing speed,
    # NOT video FPS — don't use it).
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        fps = 30.0  # last-resort fallback

    start_frame = int(start_seconds * fps) if start_seconds > 0 else 0

    # IMPORTANT: use Ultralytics' streaming mode (matches original v5).
    # Manually iterating frames and calling model.track(frame) per-frame
    # changes tracker behavior — recall dropped vs streaming in testing.
    # If start_seconds > 0 we still let YOLO process the early frames but
    # skip yielding them. This costs CPU but preserves tracking quality.
    model = YOLO(model_path)
    stream = model.track(
        source=str(video_path),
        conf=conf,
        iou=iou,
        tracker=tracker,
        classes=classes,
        persist=True,
        stream=True,
        verbose=False,
    )

    for frame_idx, result in enumerate(stream):
        if frame_idx < start_frame:
            continue
        ts = frame_idx / fps
        if result.boxes is None or result.boxes.id is None:
            continue
        ids = result.boxes.id.int().cpu().tolist()
        cls = result.boxes.cls.int().cpu().tolist()
        conf_ = result.boxes.conf.cpu().tolist()
        xyxy = result.boxes.xyxy.cpu().tolist()

        for tid, c, p, box in zip(ids, cls, conf_, xyxy):
            yield {
                "frame_idx": frame_idx,
                "timestamp_s": ts,
                "track_id": tid,
                "class_id": c,
                "conf": p,
                "xyxy": box,
            }
