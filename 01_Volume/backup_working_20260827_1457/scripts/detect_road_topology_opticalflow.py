"""
Detect road topology (one-way vs two-way) using dense optical flow instead of
object detection/tracking - an alternative to detect_road_topology.py, kept
as a separate script so the two can be compared side by side on the same
real videos rather than one replacing the other blind.

Why this approach
------------------
detect_road_topology.py fits a line per YOLO-tracked vehicle and classifies
its direction. That works, but needed real patching around two problems
found on a real site: pedestrians on a sidewalk mis-tracked as "bicycle"
polluted the direction split (fixed with a speed-ratio exclusion), and pixel
offsets from the camera don't scale uniformly across the frame (perspective,
worked around with a near-camera zone restriction).

Dense optical flow sidesteps object detection entirely. Instead of asking
"which way did this vehicle go", it asks "which way is EVERY pixel in the
scene moving, right now" - no model, no tracker, no per-object
classification, and (since there's no YOLO inference) noticeably faster to
run. Farneback dense optical flow (cv2.calcOpticalFlowFarneback) computes a
motion vector (dx, dy) for every pixel between two consecutive frames. Real
traffic moving through the scene shows up as coherent motion vectors along
the road's direction(s); this script aggregates those vectors, weighted by
speed, into a 360-degree angle histogram:

  - ONE-WAY: the histogram has one dominant lobe (most flow-weight points
    one way).
  - TWO-WAY: two lobes roughly 180 degrees apart, each carrying a real share
    of the flow-weight.

The one-way/two-way call reuses the exact share-of-movement rule from the
track-based script (--one-way-majority-frac, default 80%): split all
qualifying flow vectors into two halves by which side of the dominant axis
they fall on, and call it ONE-WAY only if one half has more than that share
of the total flow WEIGHT. Weighting by magnitude (not just counting
vectors) means a lot of slow-moving pixels (pedestrians, background noise,
compression artifacts) can't out-vote a smaller number of fast-moving
vehicle pixels - the same "speed matters" insight that came out of
debugging the track-based script's pedestrian problem, but built into the
weighting here instead of needing a separate explicit speed filter.

This script never touches shared/utils/tracker.py, and doesn't run YOLO at
all - it only reuses auto_detect_line.py's get_video_meta/extract_clip for
video-sampling (default to the video's MIDPOINT, ffmpeg-fast-seeked, same
as the other scripts).

Usage
-----
    python detect_road_topology_opticalflow.py \
        --video ../videos/403065_0001_20251118_000002.mp4 --search
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from auto_detect_line import extract_clip, get_video_meta  # noqa: E402


def compute_flow_vectors(clip_path: str, resize_width: int, stride_frames: int,
                          min_velocity: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Run Farneback dense optical flow across a clip and return the
    qualifying motion vectors (magnitude >= min_velocity, in resized-pixel
    units) as parallel (dx, dy) arrays, plus the number of frame pairs
    actually processed (for reporting).
    """
    cap = cv2.VideoCapture(clip_path)
    dxs: list[np.ndarray] = []
    dys: list[np.ndarray] = []
    prev_gray = None
    frame_idx = 0
    n_pairs = 0
    stride = max(stride_frames, 1)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        if resize_width and frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            dx = flow[..., 0].ravel()
            dy = flow[..., 1].ravel()
            mag = np.sqrt(dx * dx + dy * dy)
            keep = mag >= min_velocity
            if np.any(keep):
                dxs.append(dx[keep])
                dys.append(dy[keep])
            n_pairs += 1
        prev_gray = gray
        frame_idx += 1
    cap.release()
    if dxs:
        return np.concatenate(dxs), np.concatenate(dys), n_pairs
    return np.array([]), np.array([]), n_pairs


def dominant_axis(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Weighted (by magnitude) circular mean direction via the doubling-angle
    trick - the same construction used for the road axis in
    detect_road_topology.py (average_direction), but reimplemented here as a
    vectorized numpy computation since there can be hundreds of thousands of
    pixel vectors rather than a few dozen vehicle tracks.
    """
    mag = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    sin2 = np.sum(mag * np.sin(2 * theta))
    cos2 = np.sum(mag * np.cos(2 * theta))
    half_angle = 0.5 * np.arctan2(sin2, cos2)
    return np.array([np.cos(half_angle), np.sin(half_angle)])


def analyze_flow(dx: np.ndarray, dy: np.ndarray, one_way_majority_frac: float = 0.8,
                  n_bins: int = 36) -> dict:
    if len(dx) == 0:
        raise RuntimeError(
            "No qualifying motion found in this window (nothing moved fast enough "
            "above --min-velocity). Try a busier window, a lower --min-velocity, "
            "or a longer --segment-seconds."
        )
    mag = np.sqrt(dx * dx + dy * dy)
    theta_deg = np.degrees(np.arctan2(dy, dx)) % 360.0

    axis = dominant_axis(dx, dy)  # undirected road axis
    # split by sign of dot(vector, axis) - same half-plane test as the track script
    dot = dx * axis[0] + dy * axis[1]
    weight_a = float(np.sum(mag[dot >= 0]))
    weight_b = float(np.sum(mag[dot < 0]))
    total = weight_a + weight_b
    majority = max(weight_a, weight_b)
    majority_frac = (majority / total) if total else 0.0
    is_two_way = majority_frac <= one_way_majority_frac

    bin_edges = np.linspace(0, 360, n_bins + 1)
    hist_weights, _ = np.histogram(theta_deg, bins=bin_edges, weights=mag)

    return {
        "axis": axis, "weight_a": weight_a, "weight_b": weight_b,
        "majority_frac": majority_frac, "is_two_way": is_two_way,
        "hist_weights": hist_weights, "bin_edges": bin_edges,
        "n_vectors": int(len(dx)),
    }


def print_ascii_histogram(hist_weights: np.ndarray, bin_edges: np.ndarray, width: int = 40) -> None:
    max_w = hist_weights.max() if len(hist_weights) else 0.0
    if max_w <= 0:
        return
    print("  Angle histogram (0deg = flow pointing +x/right, 90deg = down, etc.):")
    for i, wgt in enumerate(hist_weights):
        if wgt <= 0:
            continue
        bar_len = int(round((wgt / max_w) * width))
        lo, hi = bin_edges[i], bin_edges[i + 1]
        print(f"    {lo:5.0f}-{hi:3.0f}deg | {'#' * bar_len} ({wgt:.0f})")


def save_preview(video_path: str, start_seconds: float, result: dict, out_path: Path) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_seconds * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("  (could not read a frame for the preview image)")
        return

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    axis = result["axis"]
    scale = min(w, h) * 0.35

    total = result["weight_a"] + result["weight_b"]
    frac_a = result["weight_a"] / total if total else 0.0
    frac_b = result["weight_b"] / total if total else 0.0

    def arrow(direction: np.ndarray, color: tuple, weight_frac: float) -> None:
        length = scale * (0.4 + 0.6 * weight_frac)
        end = (int(cx + direction[0] * length), int(cy + direction[1] * length))
        cv2.arrowedLine(frame, (cx, cy), end, color, 3, tipLength=0.2)

    arrow(axis, (0, 165, 255), frac_a)          # orange: direction A (+axis)
    arrow(-axis, (255, 80, 80), frac_b)         # blue: direction B (-axis)

    # small flow-weight histogram strip along the bottom of the frame
    hist = result["hist_weights"]
    max_w = hist.max() if len(hist) else 0.0
    if max_w > 0:
        strip_h = 60
        bar_w = max(1, w // len(hist))
        for i, wgt in enumerate(hist):
            bar_h = int((wgt / max_w) * (strip_h - 4))
            x0 = i * bar_w
            cv2.rectangle(frame, (x0, h - 4 - bar_h), (x0 + bar_w - 1, h - 4), (0, 200, 0), -1)

    label_lines = [
        f"{'TWO-WAY' if result['is_two_way'] else 'ONE-WAY'}  "
        f"(majority flow-weight share: {result['majority_frac']*100:.0f}%)",
        f"Direction A weight: {result['weight_a']:.0f}  [orange arrow]",
        f"Direction B weight: {result['weight_b']:.0f}  [blue arrow]",
        f"{result['n_vectors']} qualifying motion vectors",
    ]
    for i, text in enumerate(label_lines):
        cv2.putText(frame, text, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"  preview saved: {out_path}")


def find_good_flow_window(video_path: str, segment_seconds: float, resize_width: int,
                           stride_frames: int, min_velocity: float, min_total_weight: float,
                           start_seconds: float, step_seconds: float, max_search_seconds: float,
                           ) -> tuple[float, np.ndarray, np.ndarray]:
    elapsed = 0.0
    with tempfile.TemporaryDirectory(prefix="flow_topology_") as tmpdir:
        while elapsed <= max_search_seconds:
            t = start_seconds + elapsed
            print(f"  trying window {t/3600:.2f}h - {(t+segment_seconds)/3600:.2f}h ...")
            clip_path = Path(tmpdir) / "clip.mp4"
            try:
                print(f"    extracting clip ({segment_seconds:.0f}s, via ffmpeg)...")
                extract_clip(video_path, t, segment_seconds, clip_path)
            except RuntimeError as e:
                print(f"    skip: {e}")
                elapsed += step_seconds
                continue
            print("    computing dense optical flow on the clip...")
            dx, dy, n_pairs = compute_flow_vectors(str(clip_path), resize_width, stride_frames, min_velocity)
            clip_path.unlink(missing_ok=True)
            total_weight = float(np.sum(np.sqrt(dx * dx + dy * dy))) if len(dx) else 0.0
            print(f"    {len(dx)} qualifying motion vectors across {n_pairs} frame pairs "
                  f"(total weight {total_weight:.0f})")
            if total_weight >= min_total_weight:
                return t, dx, dy
            elapsed += step_seconds
    raise RuntimeError(
        f"Could not find a window with enough motion (need total flow-weight >= "
        f"{min_total_weight:.0f}) within {max_search_seconds/3600:.1f}h of searching."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-seconds", type=float, default=None,
                     help="Defaults to the video's midpoint if omitted (same as the other scripts).")
    ap.add_argument("--segment-seconds", type=float, default=60.0,
                     help="How much video to analyze per window. Optical flow doesn't need "
                          "as long a window as vehicle tracking does. Default 60s.")
    ap.add_argument("--resize-width", type=int, default=480,
                     help="Downscale frames to this width before computing flow, for speed. "
                          "Default 480.")
    ap.add_argument("--stride-frames", type=int, default=2,
                     help="Only compute flow at every Nth frame (still consecutive pairs at "
                          "that stride), to reduce compute on long/high-fps clips. Default 2.")
    ap.add_argument("--min-velocity", type=float, default=0.5,
                     help="Minimum per-pixel motion (in resized pixels per frame-pair) to "
                          "count as real movement rather than noise. Default 0.5.")
    ap.add_argument("--min-total-weight", type=float, default=2000.0,
                     help="Minimum summed motion-vector magnitude for a window to be "
                          "considered usable; below this, --search tries the next window.")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--step-seconds", type=float, default=1800.0)
    ap.add_argument("--max-search-seconds", type=float, default=24 * 3600.0)
    ap.add_argument("--one-way-majority-frac", type=float, default=0.8,
                     help="Same rule as the track-based script: if one direction accounts "
                          "for more than this fraction of total flow-weight, call the road "
                          "one-way. Default 0.8 (80%%).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg is required (used to fast-seek into the video) but isn't on PATH.\n"
            "Install it first, e.g. on macOS: brew install ffmpeg\n"
            "Then re-run this command."
        )

    w, h, duration_s = get_video_meta(args.video)

    if args.start_seconds is not None:
        start_seconds = args.start_seconds
    elif duration_s > 0:
        start_seconds = duration_s / 2.0
        print(f"No --start-seconds given: video is {duration_s/3600:.2f}h long, "
              f"sampling from its midpoint ({start_seconds/3600:.2f}h).")
    else:
        start_seconds = 0.0
        print("Could not read video duration - falling back to start at 0s.")

    max_search = args.max_search_seconds if args.search else args.segment_seconds

    print(f"Searching for a usable motion window in {args.video} ...")
    start_ts, dx, dy = find_good_flow_window(
        args.video, args.segment_seconds, args.resize_width, args.stride_frames,
        args.min_velocity, args.min_total_weight, start_seconds, args.step_seconds, max_search,
    )

    result = analyze_flow(dx, dy, one_way_majority_frac=args.one_way_majority_frac)

    print()
    print(f"Used window: {start_ts/3600:.2f}h - {(start_ts+args.segment_seconds)/3600:.2f}h")
    print(f"{result['n_vectors']} qualifying motion vectors")
    print(f"Direction A flow-weight: {result['weight_a']:.0f}")
    print(f"Direction B flow-weight: {result['weight_b']:.0f}")
    print(f"Majority direction share: {result['majority_frac']*100:.1f}%")
    print()
    print_ascii_histogram(result["hist_weights"], result["bin_edges"])
    print()
    verdict = "TWO-WAY" if result["is_two_way"] else "ONE-WAY"
    print(f"VERDICT: {verdict}")

    out_path = args.out or Path(args.video).with_name(Path(args.video).stem + "_opticalflow_topology_preview.png")
    save_preview(args.video, start_ts, result, out_path)


if __name__ == "__main__":
    main()

# Roadmap note: this is a genuinely different algorithm from
# detect_road_topology.py (no detection/tracking at all - pure pixel motion),
# kept as a separate script deliberately so the two can be compared side by
# side on the same real videos rather than one replacing the other blind.
# Known limitations to watch for: (1) camera shake/compression artifacts can
# register as spurious low-magnitude flow - --min-velocity is the guard;
# (2) a completely static scene (red light, no traffic) should correctly
# fail to find a usable window rather than guessing, via --min-total-weight;
# (3) unlike the track-based script, this has no concept of "vehicle" vs
# "pedestrian" at all - it relies entirely on magnitude-weighting (real
# traffic being faster) to keep slow pedestrian/background motion from
# dominating the verdict, which is coarser than the track script's explicit
# speed-ratio exclusion, but needs no detection model and runs much faster.
