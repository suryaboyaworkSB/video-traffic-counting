"""
Extract a stratified training set of frames from the 24-hour video.

For YOLO fine-tuning, we need diverse frames covering all conditions that
the deployed model will encounter: full daylight, dawn, dusk, full night
with headlights, light traffic, heavy traffic, every lane configuration.

Stratification: divides 24 hours into N equally-spaced sample times.
This guarantees coverage across all hours, lighting conditions, and
traffic densities (rush hour, mid-day, night).

Frames are saved as numbered JPGs ready for upload to Roboflow / CVAT /
Label Studio.

Usage:
    python extract_training_frames.py --config ../config/config.yaml --num-frames 1000
    python extract_training_frames.py --config ../config/config.yaml --num-frames 500 --output ../training/frames
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--num-frames", type=int, default=1000,
                   help="Total frames to extract (default 1000)")
    p.add_argument("--output", default="../training/frames",
                   help="Output folder for extracted frames")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPG quality 1-100, higher = larger files (default 95)")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    video_path = (cfg_path.parent / cfg["video"]).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    out_dir = (Path(__file__).parent / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps
    print(f"Video: {w}x{h} @ {fps:.2f} fps")
    print(f"Total: {n_frames} frames over {duration:.1f}s ({duration/3600:.2f} hours)")
    print(f"Extracting {args.num_frames} stratified frames...")
    print()

    # Stratified sampling: evenly spaced across the entire video.
    # This guarantees every hour, lighting condition, and traffic density
    # is represented proportionally.
    target_frames = np.linspace(0, n_frames - 1, args.num_frames, dtype=int)

    saved = 0
    metadata_rows = []
    last_pct = -1

    for i, frame_idx in enumerate(target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue

        ts = frame_idx / fps
        hh = int(ts // 3600)
        mm = int((ts % 3600) // 60)
        ss = int(ts % 60)

        # Filename includes the timestamp so labellers can sort or filter
        # by time-of-day if needed
        fname = f"frame_{i:05d}_h{hh:02d}m{mm:02d}s{ss:02d}.jpg"
        out_path = out_dir / fname

        cv2.imwrite(str(out_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

        # Classify the frame by brightness for the metadata
        mean_brightness = float(frame.mean())
        lighting = "night" if mean_brightness < 25 else (
                   "dusk" if mean_brightness < 60 else "day")

        metadata_rows.append({
            "filename": fname,
            "frame_index": int(frame_idx),
            "timestamp_s": float(ts),
            "hour": hh,
            "mean_brightness": round(mean_brightness, 2),
            "lighting": lighting,
        })

        saved += 1

        pct = int(100 * (i + 1) / len(target_frames))
        if pct != last_pct and pct % 10 == 0:
            print(f"  {pct:>3}%  ({saved}/{args.num_frames}) at hour {hh:02d}:{mm:02d}  lighting={lighting}")
            last_pct = pct

    cap.release()

    # Write metadata CSV alongside the frames
    import csv
    meta_path = out_dir / "frames_metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Lighting distribution summary
    print()
    print(f"Done. Wrote {saved} frames to:")
    print(f"  {out_dir}")
    print(f"  {meta_path}")
    print()

    from collections import Counter
    light_counts = Counter(r["lighting"] for r in metadata_rows)
    hour_counts = Counter(r["hour"] for r in metadata_rows)
    print("Lighting distribution:")
    for k in ["day", "dusk", "night"]:
        c = light_counts.get(k, 0)
        print(f"  {k:>6}: {c:>4} frames ({100*c/saved:.1f}%)")
    print()
    print("Frames per hour:")
    for hh in range(24):
        c = hour_counts.get(hh, 0)
        bar = "#" * c
        print(f"  hour {hh:02d}: {c:>3}  {bar}")

    print()
    print("Next step: upload the frames folder to Roboflow (or CVAT/Label Studio)")
    print("and label every vehicle with a bounding box (single class: 'vehicle').")


if __name__ == "__main__":
    main()
