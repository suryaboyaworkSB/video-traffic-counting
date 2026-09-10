"""
Trim a section of a video into a new file, using OpenCV only.

This is a pure-Python fallback for when ffmpeg isn't installed. It re-encodes
the chunk via mp4v which is slightly slower than ffmpeg's stream-copy, but
gets the job done in a few minutes for a 1-hour clip.

Usage:
    python trim_video.py --input ../videos/183907b-c.mp4 \
                         --output ../videos/183907_hour15.mp4 \
                         --start-seconds 54000 --duration 3600
"""

import argparse
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-seconds", type=float, required=True)
    p.add_argument("--duration", type=float, required=True,
                   help="Seconds of video to keep")
    args = p.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Could not open input: {src}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input: {w}x{h} @ {fps:.2f} fps")

    start_frame = int(args.start_seconds * fps)
    n_frames = int(args.duration * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Could not open output: {dst}")
        return

    print(f"Trimming {n_frames} frames (frame {start_frame} -> {start_frame + n_frames})")
    print(f"Writing to {dst}")

    written = 0
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"End of input at frame {i}")
            break
        writer.write(frame)
        written += 1
        if (i + 1) % 1000 == 0:
            print(f"  wrote {i+1}/{n_frames} frames")

    writer.release()
    cap.release()
    print(f"\nDone. {written} frames written to {dst}")
    print(f"Duration: {written / fps:.1f} seconds")


if __name__ == "__main__":
    main()
