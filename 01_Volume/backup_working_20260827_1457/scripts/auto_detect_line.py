"""
Auto-detect a vehicle counting line from a video, instead of drawing one by hand.

How it works (camera-angle-agnostic)
-------------------------------------
The camera's angle relative to the road varies from site to site (sometimes
square-on, sometimes shot from the side) - so we never assume anything about
that angle. Instead we derive the counting line purely from how vehicles
actually moved:

  1. Pick a representative sample window. By default this is the MIDPOINT of
     the video's total duration (a 24h video samples around hour 12, a 14h
     video around hour 7) - this avoids the atypical traffic right at the
     start of a recording and, for day-long videos, tends to land in
     daylight. Pass --start-seconds to override this.
  2. Run the existing detector + tracker (shared.utils.tracker.track_video)
     over that window to get per-vehicle trajectories. If the window is too
     quiet (e.g. it landed at night), automatically step forward and try
     again (--search).
  3. Fit ONE clean line to each vehicle's own path (total least squares
     through all of that vehicle's points - far less noisy than using raw
     scattered points or single-frame deltas).
  4. Drop vehicles that didn't actually travel anywhere (parked/stopped -
     these have no meaningful direction and only add noise).
  5. Geometric fact: the shortest segment connecting two (near-)parallel
     lines is always exactly perpendicular to both of them, regardless of
     the camera's angle onto the road. So the counting-line direction comes
     from the shared direction across all these per-vehicle lines (robustly
     averaged - see average_direction()), which is equivalent to what you'd
     get from that minimum-distance construction between any two of them.
  6. The line is centered on the moving-traffic point cloud (for visibility)
     and, by default, extended all the way to the edges of the frame along
     that perpendicular direction - so it spans the full road cross-section
     (all lanes) even if some lanes happened to be empty during the sampled
     window. Use --no-extend-to-frame to fall back to a line sized only to
     the observed traffic instead.

Usage
-----
    python auto_detect_line.py --video ../videos/403065_0001_20251118_000002.mp4 \
        --model yolo11m.pt --classes 1 2 3 5 7 --search

    # also patch the result into a config file (backs up the original):
    python auto_detect_line.py --video ../videos/403065_0001_20251118_000002.mp4 \
        --model yolo11m.pt --classes 1 2 3 5 7 --search \
        --write-to-config ../config/config_403065.yaml --line-name Main

Notes
-----
- Proposes ONE line (matches how your configs work today: one "Main" line,
  direction A/B decided by which side it was crossed from). For multi-lane
  highways needing per-lane lines, cluster the per-track fitted lines by
  their perpendicular offset before averaging - not needed yet for the
  two-way sites this has been tested on.
- Always check the preview PNG before trusting a new camera angle.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.utils.tracker import track_video  # noqa: E402

MIN_NET_DISPLACEMENT_PX = 30.0  # below this, a track is "parked/stopped", not moving traffic


def get_video_meta(video_path: str) -> tuple[int, int, float]:
    """Returns (width, height, duration_seconds). duration is 0.0 if it can't
    be determined (e.g. a bad fps read from a corrupt/odd container)."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    duration = (frame_count / fps) if fps > 0 else 0.0
    return w, h, duration


def extract_clip(video_path: str, start_seconds: float, duration_seconds: float,
                  out_path: Path) -> None:
    """Fast keyframe-based trim via ffmpeg stream-copy (no re-encode).

    We use this instead of relying on shared.utils.tracker.track_video's own
    start_seconds handling: that function does NOT actually seek the video -
    it runs full YOLO inference on every frame from 0 and just discards the
    results until start_seconds is reached (this preserves ByteTrack's
    behavior, per its own docstring). That's cheap when start_seconds is
    small, but ruinous when it's "hour 12 of a 24h video" - it would run
    YOLO across all 12 hours first. Pre-extracting a short clip with ffmpeg
    sidesteps that: the clip starts at (approximately - ffmpeg snaps to the
    nearest preceding keyframe) start_seconds, so the tracker can be run on
    it with start_seconds=0 and only ever processes the few minutes we
    actually need.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{max(start_seconds, 0.0):.3f}",
        "-i", str(video_path), "-t", f"{duration_seconds:.3f}",
        "-c", "copy", str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. On macOS: brew install ffmpeg "
            "(then re-run this command)."
        )
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg failed to extract a clip at {start_seconds:.1f}s "
            f"(is ffmpeg installed and on PATH?): {result.stderr[-500:]}"
        )


def collect_tracks(video_path: str, model_path: str, conf: float, classes: list[int],
                    tracker: str, start_seconds: float, segment_seconds: float,
                    ) -> dict[int, list[tuple[float, float]]]:
    tracks: dict[int, list[tuple[float, float]]] = {}
    end_ts = start_seconds + segment_seconds
    for det in track_video(video_path, model_path=model_path, conf=conf,
                            tracker=tracker, classes=classes,
                            start_seconds=start_seconds):
        if det["timestamp_s"] > end_ts:
            break
        x1, y1, x2, y2 = det["xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        tracks.setdefault(det["track_id"], []).append((cx, cy))
    return tracks


def find_good_window(video_path: str, model_path: str, conf: float, classes: list[int],
                      tracker: str, segment_seconds: float, min_track_len: int,
                      needed_tracks: int, start_seconds: float, step_seconds: float,
                      max_search_seconds: float,
                      ) -> tuple[float, dict[int, list[tuple[float, float]]]]:
    t = start_seconds
    with tempfile.TemporaryDirectory(prefix="auto_line_") as tmpdir:
        while t < start_seconds + max_search_seconds:
            print(f"  trying window {t/3600:.2f}h - {(t+segment_seconds)/3600:.2f}h ...")
            clip_path = Path(tmpdir) / f"window_{int(t)}.mp4"
            print(f"    extracting clip ({segment_seconds:.0f}s, via ffmpeg)...")
            try:
                extract_clip(video_path, t, segment_seconds, clip_path)
            except RuntimeError as e:
                print(f"    (could not extract this window: {e}; skipping)")
                t += step_seconds
                continue
            print(f"    running {model_path} tracking on the clip "
                  f"(this is the slow part - no per-frame output while it runs)...")
            # start_seconds=0 here on purpose - the clip already starts at t,
            # so there's nothing left to skip (see extract_clip's docstring
            # for why we don't just pass t straight into collect_tracks).
            tracks = collect_tracks(str(clip_path), model_path, conf, classes, tracker,
                                     0.0, segment_seconds)
            clip_path.unlink(missing_ok=True)
            good = {tid: pts for tid, pts in tracks.items() if len(pts) >= min_track_len}
            print(f"    {len(good)} usable tracks (of {len(tracks)} raw)")
            if len(good) >= needed_tracks:
                return t, good
            t += step_seconds
    raise RuntimeError(
        f"Could not find a window with >= {needed_tracks} usable tracks within "
        f"the first {max_search_seconds/3600:.1f}h of video (starting from "
        f"{start_seconds/3600:.2f}h). Try a larger --segment-seconds, a lower "
        f"--needed-tracks, or check the video isn't mostly empty/night-only."
    )


def collect_tracks_with_span(video_path: str, model_path: str, conf: float, classes: list[int],
                              tracker: str, start_seconds: float, segment_seconds: float,
                              ) -> tuple[dict[int, list[tuple[float, float]]], dict[int, tuple[float, float]]]:
    """Same as collect_tracks (same (x, y) points, same track IDs), but also
    returns each track's [first_ts, last_ts] active time span within the
    window. Added as a separate function rather than changing collect_tracks
    itself, so every existing caller (this script's own main(), and anything
    else built on collect_tracks/find_good_window) keeps working exactly as
    before - this is purely additive, opt-in machinery for callers that need
    to know *when* a track was moving, not just where."""
    tracks: dict[int, list[tuple[float, float]]] = {}
    spans: dict[int, tuple[float, float]] = {}
    end_ts = start_seconds + segment_seconds
    for det in track_video(video_path, model_path=model_path, conf=conf,
                            tracker=tracker, classes=classes,
                            start_seconds=start_seconds):
        if det["timestamp_s"] > end_ts:
            break
        x1, y1, x2, y2 = det["xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        tid, ts = det["track_id"], det["timestamp_s"]
        tracks.setdefault(tid, []).append((cx, cy))
        if tid not in spans:
            spans[tid] = (ts, ts)
        else:
            first, last = spans[tid]
            spans[tid] = (min(first, ts), max(last, ts))
    return tracks, spans


def find_good_window_with_span(video_path: str, model_path: str, conf: float, classes: list[int],
                                tracker: str, segment_seconds: float, min_track_len: int,
                                needed_tracks: int, start_seconds: float, step_seconds: float,
                                max_search_seconds: float,
                                ) -> tuple[float, dict[int, list[tuple[float, float]]], dict[int, tuple[float, float]]]:
    """Same search loop as find_good_window (try a window, extract via
    ffmpeg, track it, check for enough usable tracks, else step forward),
    but also returns each kept track's time span - see
    collect_tracks_with_span. find_good_window itself is left untouched."""
    t = start_seconds
    with tempfile.TemporaryDirectory(prefix="auto_line_") as tmpdir:
        while t < start_seconds + max_search_seconds:
            print(f"  trying window {t/3600:.2f}h - {(t+segment_seconds)/3600:.2f}h ...")
            clip_path = Path(tmpdir) / f"window_{int(t)}.mp4"
            print(f"    extracting clip ({segment_seconds:.0f}s, via ffmpeg)...")
            try:
                extract_clip(video_path, t, segment_seconds, clip_path)
            except RuntimeError as e:
                print(f"    (could not extract this window: {e}; skipping)")
                t += step_seconds
                continue
            print(f"    running {model_path} tracking on the clip "
                  f"(this is the slow part - no per-frame output while it runs)...")
            tracks, spans = collect_tracks_with_span(str(clip_path), model_path, conf, classes, tracker,
                                                      0.0, segment_seconds)
            clip_path.unlink(missing_ok=True)
            good = {tid: pts for tid, pts in tracks.items() if len(pts) >= min_track_len}
            print(f"    {len(good)} usable tracks (of {len(tracks)} raw)")
            if len(good) >= needed_tracks:
                good_spans = {tid: spans[tid] for tid in good}
                return t, good, good_spans
            t += step_seconds
    raise RuntimeError(
        f"Could not find a window with >= {needed_tracks} usable tracks within "
        f"the first {max_search_seconds/3600:.1f}h of video (starting from "
        f"{start_seconds/3600:.2f}h). Try a larger --segment-seconds, a lower "
        f"--needed-tracks, or check the video isn't mostly empty/night-only."
    )


def fit_line_tls(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line fit through one vehicle's points.
    Returns (point_on_line, unit_direction)."""
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center)
    direction = vt[0]
    return center, direction / np.linalg.norm(direction)


def average_direction(directions: list[np.ndarray]) -> np.ndarray:
    """Average a set of undirected line directions (each vehicle may be
    heading either way along the road, so plain vector averaging would
    cancel opposite-direction traffic out). Doubling the angle folds theta
    and theta+180 onto the same value before averaging, then halving undoes it."""
    angles = np.array([np.arctan2(d[1], d[0]) for d in directions])
    mean_sin = np.mean(np.sin(2 * angles))
    mean_cos = np.mean(np.cos(2 * angles))
    avg_angle = np.arctan2(mean_sin, mean_cos) / 2.0
    return np.array([np.cos(avg_angle), np.sin(avg_angle)])


def extend_to_frame_edges(center: np.ndarray, direction: np.ndarray,
                           w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Extend the line through `center` in `direction` (a unit vector) out to
    where it exits the frame on each side (standard ray-vs-rectangle "slab"
    clip). This is what guarantees the counting line spans the FULL road
    cross-section - every lane - rather than just however wide the sampled
    traffic happened to be. Assumes `center` is itself inside the frame."""
    cx, cy = float(center[0]), float(center[1])
    dx, dy = float(direction[0]), float(direction[1])
    t_lo, t_hi = -np.inf, np.inf

    if abs(dx) > 1e-9:
        t1, t2 = (0 - cx) / dx, (w - 1 - cx) / dx
        t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))
    elif not (0 <= cx <= w - 1):
        return center, center  # degenerate: center already outside frame in x

    if abs(dy) > 1e-9:
        t1, t2 = (0 - cy) / dy, (h - 1 - cy) / dy
        t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))
    elif not (0 <= cy <= h - 1):
        return center, center  # degenerate: center already outside frame in y

    p1 = center + direction * t_lo
    p2 = center + direction * t_hi
    return p1, p2


def compute_line(tracks: dict[int, list[tuple[float, float]]], frame_wh: tuple[int, int],
                  min_net_displacement: float = MIN_NET_DISPLACEMENT_PX,
                  padding: float = 1.3, min_half_len: float = 60.0,
                  extend_to_frame: bool = True,
                  ) -> dict:
    w, h = frame_wh

    fitted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tid, pts in tracks.items():
        arr = np.array(pts)
        if np.linalg.norm(arr[-1] - arr[0]) < min_net_displacement:
            continue  # parked/stopped - no meaningful direction, would only add noise
        fitted[tid] = fit_line_tls(arr)

    if len(fitted) < 3:
        raise RuntimeError(
            f"Only {len(fitted)} tracks had enough net movement to use "
            f"(need >= 3). Try a longer --segment-seconds or a busier window."
        )

    major_axis = average_direction([d for _, d in fitted.values()])
    minor_axis = np.array([-major_axis[1], major_axis[0]])  # perpendicular = the counting line

    all_points = np.array([p for tid in fitted for p in tracks[tid]])
    center = all_points.mean(axis=0)

    if extend_to_frame:
        # Cover every lane, not just the ones with traffic in this sample.
        p1, p2 = extend_to_frame_edges(center, minor_axis, w, h)
    else:
        proj = (all_points - center) @ minor_axis
        half_len = max((proj.max() - proj.min()) / 2.0 * padding, min_half_len)
        p1 = center - minor_axis * half_len
        p2 = center + minor_axis * half_len

    clip = lambda pt: (float(np.clip(pt[0], 0, w - 1)), float(np.clip(pt[1], 0, h - 1)))

    return {
        "p1": clip(p1), "p2": clip(p2), "center": center,
        "major_axis": major_axis, "minor_axis": minor_axis,
        "fitted": fitted, "n_dropped_stationary": len(tracks) - len(fitted),
    }


def save_preview(video_path: str, start_seconds: float,
                  tracks: dict[int, list[tuple[float, float]]], result: dict,
                  out_path: Path) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_seconds * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("  (could not read a frame for the preview image)")
        return
    fitted = result["fitted"]
    for tid, pts in tracks.items():
        pts_i = np.array(pts, dtype=int)
        color = (0, 180, 255) if tid in fitted else (120, 120, 120)  # gray = dropped
        for i in range(1, len(pts_i)):
            cv2.line(frame, tuple(pts_i[i - 1]), tuple(pts_i[i]), color, 1)
    p1, p2, center = result["p1"], result["p2"], result["center"]
    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 3)
    cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"  preview saved: {out_path}  (gray tracks were dropped as near-stationary)")


def write_to_config(config_path: Path, line_name: str,
                     p1: tuple[float, float], p2: tuple[float, float]) -> None:
    backup = config_path.with_suffix(config_path.suffix + f".bak_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(config_path, backup)
    cfg = yaml.safe_load(config_path.read_text()) or {}
    lines = cfg.setdefault("lines", [])
    pts = [[round(p1[0]), round(p1[1])], [round(p2[0]), round(p2[1])]]
    for line in lines:
        if line.get("name") == line_name:
            line["points"] = pts
            break
    else:
        lines.append({"name": line_name, "points": pts})
    config_path.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"  wrote line into {config_path} (backup saved: {backup})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 5, 7],
                     help="COCO class ids to track (default: bicycle car motorcycle bus truck)")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--start-seconds", type=float, default=None,
                     help="Where to start sampling. If omitted, defaults to the MIDPOINT of the "
                          "video's total duration (e.g. hour 12 of a 24h video, hour 7 of a 14h "
                          "video) so the sample isn't biased by whatever's happening right at "
                          "the start of the recording.")
    ap.add_argument("--segment-seconds", type=float, default=180.0)
    ap.add_argument("--min-track-len", type=int, default=5)
    ap.add_argument("--min-net-displacement", type=float, default=MIN_NET_DISPLACEMENT_PX,
                     help="Tracks moving less than this (px) are treated as parked/stopped and dropped.")
    ap.add_argument("--needed-tracks", type=int, default=8)
    ap.add_argument("--search", action="store_true",
                     help="If the first window is too quiet, keep stepping forward instead of failing.")
    ap.add_argument("--step-seconds", type=float, default=1800.0)
    ap.add_argument("--max-search-seconds", type=float, default=24 * 3600.0)
    ap.add_argument("--line-name", default="Main")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--write-to-config", type=Path, default=None)
    ap.add_argument("--no-extend-to-frame", dest="extend_to_frame", action="store_false",
                     help="By default the line is stretched to the frame edges so it spans every "
                          "lane. Pass this to instead size it tightly around the observed traffic "
                          "(old behavior) - useful if edge-to-edge would cross an unrelated road.")
    ap.set_defaults(extend_to_frame=True)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg is required (used to fast-seek into the video without running "
            "the detector over everything before it) but isn't on PATH.\n"
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
        print("Could not read video duration (bad fps/frame count) - falling back to "
              "start at 0s. Pass --start-seconds explicitly if that's not right.")

    max_search = args.max_search_seconds if args.search else args.segment_seconds

    print(f"Searching for a usable warm-up window in {args.video} ...")
    start_ts, tracks = find_good_window(
        args.video, args.model, args.conf, args.classes, args.tracker,
        args.segment_seconds, args.min_track_len, args.needed_tracks,
        start_seconds, args.step_seconds, max_search,
    )

    result = compute_line(tracks, (w, h), min_net_displacement=args.min_net_displacement,
                           extend_to_frame=args.extend_to_frame)

    print()
    print(f"Used window: {start_ts/3600:.2f}h - {(start_ts+args.segment_seconds)/3600:.2f}h")
    print(f"{len(result['fitted'])} vehicles used for direction "
          f"({result['n_dropped_stationary']} dropped as near-stationary)")
    print(f"Road direction: {result['major_axis']}   Line direction: {result['minor_axis']}")
    print(f"Line sizing: {'extended to frame edges (all lanes)' if args.extend_to_frame else 'fit to observed traffic only'}")
    print()
    print("AUTO-DETECTED LINE (drop into config.yaml):")
    print(f"  - name: {args.line_name}")
    p1, p2 = result["p1"], result["p2"]
    print(f"    points: [[{p1[0]:.0f}, {p1[1]:.0f}], [{p2[0]:.0f}, {p2[1]:.0f}]]")

    out_path = args.out or Path(args.video).with_name(Path(args.video).stem + "_auto_line_preview.png")
    save_preview(args.video, start_ts, tracks, result, out_path)

    if args.write_to_config:
        write_to_config(args.write_to_config, args.line_name, p1, p2)


if __name__ == "__main__":
    main()

# Roadmap note: for multi-lane highways where a single "Main" line isn't
# enough, cluster the per-track fitted lines by their perpendicular offset
# (e.g. k-means on `center @ minor_axis` for each fitted track) before
# averaging, then run compute_line() once per cluster to get one line per
# lane. Not needed yet for the two-way sites this has been tested on.
