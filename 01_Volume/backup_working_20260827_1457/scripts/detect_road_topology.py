"""
Detect road topology from a video: is it one-way or two-way?

This reuses auto_detect_line.py's video-sampling machinery (midpoint
sampling, the ffmpeg fast-seek fix, per-track total-least-squares fitting)
via direct import instead of duplicating it - one source of truth for "how
do we safely grab a representative window of traffic from a long video."
shared/utils/tracker.py itself is never touched by either script.

How it works
------------
  1. Same sampling as the line detector: default to the video's MIDPOINT
     (24h video -> hour 12, 14h video -> hour 7), ffmpeg-fast-seeked so we
     never run the detector over hours of discarded frames.
  2. Fit one TLS line per vehicle track using only its points from the
     near-camera zone (bottom --near-camera-frac of the frame, default the
     bottom half) - dropping near-stationary/parked ones, and dropping any
     track that never enters that zone. Also keep each track's SIGNED
     direction (net displacement start -> end) within that zone, which the
     line detector didn't need (it only cared about the undirected road
     axis). Restricting to near-camera points matters: camera perspective
     maps the same real-world lateral distance to very different pixel
     offsets depending on how far up the frame something is, so comparing
     raw full-track positions across vehicles at different depths isn't
     reliable - a real site tested this way and it correctly excluded
     nothing else against a global "did this happen at all" check.
  3. Find the road's overall axis (doubling-angle average, camera-angle
     agnostic - same math as the counting line).
  4. Classify each track as heading "direction A" or "direction B" along
     that axis, by the sign of its own signed direction relative to it.
  5. Filter out non-traffic by speed only: a pedestrian mis-tracked into a
     vehicle class (e.g. "bicycle") still moves at walking pace, far slower
     than real vehicles, regardless of which side of the road they're on.
     Position/lateral-offset is deliberately NOT used to exclude minority-
     direction tracks - a real opposite lane is, by definition, laterally
     offset from the majority lane, so an offset-based corridor built from
     the majority direction alone would wrongly treat genuine oncoming
     traffic as "off the road." (Confirmed on a real two-way site: this
     filter combo was silently zeroing out the opposing-direction lane.)
  6. One-way vs two-way, by a simple share-of-movement rule on what's left:
     if one direction accounts for more than --one-way-majority-frac
     (default 80%) of all tracks, call it ONE-WAY. Otherwise there's
     meaningful movement in both (opposite) directions, so call it
     TWO-WAY. A small minimum track count on the minority side guards
     against a single wrong-way/turning vehicle flipping the verdict.

Lane counting (how many lanes per direction) is deliberately not part of
this script right now - it's a separate, harder problem and is being
parked until the one-way/two-way call itself is solid across more real
sites.

Usage
-----
    python detect_road_topology.py --video ../videos/403065_0001_20251118_000002.mp4 \
        --model yolo11m.pt --classes 1 2 3 5 7 --search
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from auto_detect_line import (  # noqa: E402
    MIN_NET_DISPLACEMENT_PX,
    average_direction,
    find_good_window_with_span,
    fit_line_tls,
    get_video_meta,
)


def tracks_overlap_in_time(span_a: tuple[float, float], span_b: tuple[float, float]) -> bool:
    """True if two tracks' [first_ts, last_ts] active windows overlap at
    all - i.e. both were actually moving at the same moment, not just
    somewhere in the same multi-minute sample."""
    a_first, a_last = span_a
    b_first, b_last = span_b
    return max(a_first, b_first) <= min(a_last, b_last)


def classify_directions(tracks: dict[int, list[tuple[float, float]]],
                         min_net_displacement: float = MIN_NET_DISPLACEMENT_PX,
                         frame_h: float | None = None,
                         near_camera_frac: float = 0.5,
                         min_axis_alignment_deg: float = 60.0,
                         debug: bool = False,
                         ) -> dict:
    """Fit each track, find the road axis, and split tracks into two
    direction groups by their own travel direction relative to that axis -
    but only if that travel is actually roughly along the axis (forward or
    backward). A track has to be within `min_axis_alignment_deg` of being
    parallel or anti-parallel to the road axis to count as real along-the-
    road traffic at all; anything more sideways than that (e.g. a car
    crossing or turning at a far intersection, a pedestrian/cyclist
    crossing the street) is excluded as "crossing" rather than forced into
    whichever of the two directions its small forward/backward component
    happens to lean toward. Without this, a purely sideways track can end
    up on the "wrong" side of a simple sign check by pure noise (TLS
    fitting wobble, or a couple of near-camera-zone points) and get
    miscounted as opposing traffic on a real one-way street.

    Only each track's points from the near-camera zone (the bottom
    `near_camera_frac` of the frame - closest to the camera, for a typical
    downward-angled street camera) are used. Camera perspective means the
    same real-world lateral distance maps to very different pixel offsets
    depending on how far up the frame a vehicle is; restricting to the
    near-camera band keeps every track's measurements in a consistent,
    close-to-undistorted depth range instead of mixing near and far. A
    track that never enters that band (e.g. only visible far away for the
    whole sampled window) is skipped rather than measured unreliably.
    """
    all_y_all_tracks = [p[1] for pts in tracks.values() for p in pts]
    if frame_h is not None:
        near_y = frame_h * (1.0 - near_camera_frac)
    else:
        near_y = float(np.percentile(all_y_all_tracks, 100.0 * (1.0 - near_camera_frac))) if all_y_all_tracks else 0.0

    if debug:
        y_lo = min(all_y_all_tracks) if all_y_all_tracks else 0.0
        y_hi = max(all_y_all_tracks) if all_y_all_tracks else 0.0
        print(f"  [debug] frame_h={frame_h}  near_camera_frac={near_camera_frac}  "
              f"near_y_threshold={near_y:.1f}  (observed y range: {y_lo:.1f} - {y_hi:.1f})")

    # fitted[tid] = (center, axis, signed_dir, speed_px_per_sample). speed is
    # net displacement divided by number of sample-to-sample steps in the
    # near-camera segment - a relative (not absolute px/sec) figure, but
    # comparable across tracks from the same video since they're all sampled
    # at the same rate. Used to tell real vehicle traffic from pedestrians
    # (misdetected as "bicycle") who move far slower - see analyze_topology.
    fitted: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    n_dropped_never_near = 0
    n_dropped_stationary = 0
    for tid, pts in tracks.items():
        arr = np.array(pts)
        near = arr[arr[:, 1] >= near_y]
        if len(near) < 2:
            n_dropped_never_near += 1
            continue  # never close enough to the camera to measure reliably
        net = near[-1] - near[0]
        net_disp = np.linalg.norm(net)
        if net_disp < min_net_displacement:
            n_dropped_stationary += 1
            continue  # parked/stopped (within the near-camera zone)
        center, axis = fit_line_tls(near)
        signed_dir = net / net_disp  # actual travel direction, unlike axis (undirected)
        speed = net_disp / max(len(near) - 1, 1)
        fitted[tid] = (center, axis, signed_dir, speed)

    if debug:
        print(f"  [debug] tracks: {len(tracks)} total, {n_dropped_never_near} never near camera, "
              f"{n_dropped_stationary} stationary within near-camera zone, {len(fitted)} usable")

    if len(fitted) < 3:
        raise RuntimeError(
            f"Only {len(fitted)} tracks had enough near-camera movement to use "
            f"(need >= 3). Try a longer --segment-seconds, a busier window, or a "
            f"larger --near-camera-frac."
        )

    # Road axis from the UNDIRECTED per-track axes (doubling-angle average -
    # same construction as the counting line, so it agrees with it).
    major_axis = average_direction([axis for _, axis, _, _ in fitted.values()])
    minor_axis = np.array([-major_axis[1], major_axis[0]])  # perpendicular

    align_threshold = float(np.cos(np.radians(min_axis_alignment_deg)))
    group_a: list[int] = []
    group_b: list[int] = []
    crossing: list[int] = []
    for tid, (_, _, signed_dir, _) in fitted.items():
        align = float(np.dot(signed_dir, major_axis))
        if align >= align_threshold:
            group_a.append(tid)
        elif align <= -align_threshold:
            group_b.append(tid)
        else:
            crossing.append(tid)

    if debug:
        angles = {tid: float(np.degrees(np.arccos(np.clip(
            np.dot(fitted[tid][2], major_axis), -1.0, 1.0)))) for tid in fitted}
        print(f"  [debug] axis-alignment threshold: {min_axis_alignment_deg:.0f} deg "
              f"(align_threshold={align_threshold:.2f})")
        print(f"  [debug] angle to major_axis per track (0=forward, 180=backward, "
              f"~90=crossing): {sorted(angles.values())}")
        if crossing:
            print(f"  [debug] excluded as crossing/turning (not aligned with road axis): "
                  f"{len(crossing)} tracks")

    return {
        "fitted": fitted, "major_axis": major_axis, "minor_axis": minor_axis,
        "group_a": group_a, "group_b": group_b, "crossing": crossing,
    }


def analyze_topology(tracks: dict[int, list[tuple[float, float]]],
                      min_net_displacement: float = MIN_NET_DISPLACEMENT_PX,
                      one_way_majority_frac: float = 0.8,
                      min_minority_tracks: int = 2,
                      frame_h: float | None = None,
                      near_camera_frac: float = 0.5,
                      min_speed_frac: float = 0.35,
                      min_axis_alignment_deg: float = 60.0,
                      spans: dict[int, tuple[float, float]] | None = None,
                      require_overlap: bool = False,
                      debug: bool = False,
                      ) -> dict:
    """One-way/two-way call via a simple share-of-movement rule: if one
    direction has more than `one_way_majority_frac` of all tracks, it's
    one-way; otherwise there's real movement both ways, so it's two-way.
    `min_minority_tracks` stops a single outlier (a turning or wrong-way
    vehicle) from being enough to call it two-way on its own.

    Before that count, tracks are cleaned up by speed only: pedestrians
    (walking pace) mis-tracked as "bicycle" move far slower than real
    traffic. The majority direction's median speed is used as "real
    traffic speed"; any track (either direction) slower than
    `min_speed_frac` of that is excluded as "off-road" rather than counted
    as opposing traffic.

    Position/lateral-offset is deliberately NOT used as a filter here. An
    earlier version also built a perpendicular-offset "road corridor" from
    the majority direction and excluded minority-direction tracks outside
    it - that worked for a sidewalk (which genuinely can overlap the
    driving lane's offset range, so speed alone was already doing the real
    work there) but broke on a real two-way road: the opposite lane is, by
    definition, laterally offset from the majority lane, so that corridor
    check was silently reclassifying real oncoming traffic as "off-road"
    and zeroing out the minority direction. Speed alone doesn't have that
    failure mode, since a real vehicle in the opposite lane still moves at
    real vehicle speed.

    Optionally (`require_overlap=True`, `spans` provided), the two-way call
    can additionally require that at least one direction-A and one
    direction-B track were actually active at the same time (a real
    simultaneous pass), not just both present somewhere in the sampled
    window. This is stronger evidence but a stricter bar - off by default
    so it doesn't regress the already-validated majority-share-only
    behavior; `spans` comes from find_good_window_with_span.

    Before any of the above, `classify_directions` already drops tracks
    that aren't reasonably aligned with the road axis at all (see its
    docstring) - a track crossing or turning at a far intersection, tagged
    "crossing" rather than forced into direction A or B by a bare sign
    check. Confirmed on a real one-way site: a handful of tracks near a
    far cross-street, moving roughly perpendicular to the main road, were
    getting counted as "opposing traffic" under the old sign-only split.
    """
    classified = classify_directions(tracks, min_net_displacement, frame_h, near_camera_frac,
                                      min_axis_alignment_deg, debug)
    fitted = classified["fitted"]
    crossing = classified["crossing"]
    major_axis, minor_axis = classified["major_axis"], classified["minor_axis"]
    group_a, group_b = classified["group_a"], classified["group_b"]

    offsets = {tid: float(np.dot(fitted[tid][0], minor_axis)) for tid in fitted}
    speeds = {tid: float(fitted[tid][3]) for tid in fitted}

    if debug:
        print(f"  [debug] pre-filter: group_a={len(group_a)} group_b={len(group_b)}")
        print(f"  [debug] group_a offsets: {sorted(offsets[t] for t in group_a)}")
        print(f"  [debug] group_b offsets: {sorted(offsets[t] for t in group_b)}")
        print(f"  [debug] group_a speeds: {sorted(speeds[t] for t in group_a)}")
        print(f"  [debug] group_b speeds: {sorted(speeds[t] for t in group_b)}")

    off_road: list[int] = []
    if len(group_a) >= len(group_b):
        corridor_ids, to_filter, filtered_is_b = group_a, group_b, True
    else:
        corridor_ids, to_filter, filtered_is_b = group_b, group_a, False

    if corridor_ids and to_filter:
        # Speed cleanup only, applied to BOTH groups (a slow "majority-side"
        # track is just as suspect as a slow minority one). No position/
        # lateral-offset filtering - see analyze_topology's docstring for
        # why that broke real two-way roads.
        ref_speed = float(np.median([speeds[tid] for tid in corridor_ids]))
        speed_threshold = ref_speed * min_speed_frac
        if debug:
            print(f"  [debug] reference speed (median of larger group): {ref_speed:.2f}  "
                  f"speed_threshold: {speed_threshold:.2f}")
        slow_corridor = [tid for tid in corridor_ids if speeds[tid] < speed_threshold]
        fast_corridor = [tid for tid in corridor_ids if tid not in slow_corridor]
        slow_candidates = [tid for tid in to_filter if speeds[tid] < speed_threshold]
        speed_ok_candidates = [tid for tid in to_filter if tid not in slow_candidates]
        if debug and (slow_corridor or slow_candidates):
            print(f"  [debug] excluded for being too slow: "
                  f"{len(slow_corridor)} from larger group, {len(slow_candidates)} from smaller group")

        off_road = slow_corridor + slow_candidates
        if filtered_is_b:
            group_a, group_b = fast_corridor, speed_ok_candidates
        else:
            group_b, group_a = fast_corridor, speed_ok_candidates

    n_a, n_b = len(group_a), len(group_b)
    total = n_a + n_b
    majority = max(n_a, n_b)
    minority = min(n_a, n_b)
    majority_frac = (majority / total) if total else 0.0
    minority_frac = (minority / total) if total else 0.0

    is_two_way = (majority_frac <= one_way_majority_frac) and minority >= min_minority_tracks

    has_overlap: bool | None = None
    n_overlapping_pairs = 0
    if spans is not None:
        overlap_pairs = [
            (a, b) for a in group_a for b in group_b
            if a in spans and b in spans and tracks_overlap_in_time(spans[a], spans[b])
        ]
        n_overlapping_pairs = len(overlap_pairs)
        has_overlap = n_overlapping_pairs > 0
        if debug:
            print(f"  [debug] simultaneous (time-overlapping) opposite-direction pairs: "
                  f"{n_overlapping_pairs} (of {n_a * n_b} possible A x B pairs)")
        if require_overlap and is_two_way and not has_overlap:
            if debug:
                print("  [debug] require_overlap=True and no A/B pair overlapped in time - "
                      "downgrading what would have been TWO-WAY to ONE-WAY")
            is_two_way = False

    return {
        "fitted": fitted, "major_axis": major_axis, "minor_axis": minor_axis,
        "group_a": group_a, "group_b": group_b, "off_road": off_road,
        "crossing": crossing, "n_crossing": len(crossing),
        "n_a": n_a, "n_b": n_b, "n_off_road": len(off_road),
        "majority_frac": majority_frac, "minority_frac": minority_frac,
        "has_overlap": has_overlap, "n_overlapping_pairs": n_overlapping_pairs,
        "is_two_way": is_two_way,
    }


DIR_A_COLOR = (0, 165, 255)   # orange
DIR_B_COLOR = (255, 80, 80)   # blue
OFF_ROAD_COLOR = (200, 0, 200)  # magenta
CROSSING_COLOR = (0, 220, 220)  # yellow-ish cyan


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

    # dim everything dropped-as-stationary first
    for tid, pts in tracks.items():
        if tid not in fitted:
            pts_i = np.array(pts, dtype=int)
            for i in range(1, len(pts_i)):
                cv2.line(frame, tuple(pts_i[i - 1]), tuple(pts_i[i]), (110, 110, 110), 1)

    for group, color in [
        (result["group_a"], DIR_A_COLOR),
        (result["group_b"], DIR_B_COLOR),
        (result.get("off_road", []), OFF_ROAD_COLOR),
        (result.get("crossing", []), CROSSING_COLOR),
    ]:
        for tid in group:
            pts_i = np.array(tracks[int(tid)], dtype=int)
            for i in range(1, len(pts_i)):
                cv2.line(frame, tuple(pts_i[i - 1]), tuple(pts_i[i]), color, 2)

    label_lines = [
        f"{'TWO-WAY' if result['is_two_way'] else 'ONE-WAY'}  "
        f"(majority share: {result['majority_frac']*100:.0f}%)",
        f"Direction A: {result['n_a']} tracks  [orange]",
        f"Direction B: {result['n_b']} tracks  [blue]",
        f"Off-road (sidewalk/etc, excluded): {result.get('n_off_road', 0)}  [magenta]",
        f"Crossing/turning (not axis-aligned, excluded): {result.get('n_crossing', 0)}  [cyan]",
        "gray = dropped as near-stationary",
    ]
    for i, text in enumerate(label_lines):
        cv2.putText(frame, text, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"  preview saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 5, 7])
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--start-seconds", type=float, default=None,
                     help="Defaults to the video's midpoint if omitted (same as auto_detect_line.py).")
    ap.add_argument("--segment-seconds", type=float, default=180.0)
    ap.add_argument("--min-track-len", type=int, default=5)
    ap.add_argument("--min-net-displacement", type=float, default=MIN_NET_DISPLACEMENT_PX)
    ap.add_argument("--needed-tracks", type=int, default=15,
                     help="Higher than auto_detect_line.py's default (8) - a direction split "
                          "needs more samples to be trustworthy.")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--step-seconds", type=float, default=1800.0)
    ap.add_argument("--max-search-seconds", type=float, default=24 * 3600.0)
    ap.add_argument("--one-way-majority-frac", type=float, default=0.8,
                     help="If one direction accounts for more than this fraction of all "
                          "movement, call the road one-way. Default 0.8 (80%%).")
    ap.add_argument("--min-minority-tracks", type=int, default=2,
                     help="The minority direction needs at least this many tracks to count "
                          "as real opposite-direction traffic (guards against one outlier).")
    ap.add_argument("--near-camera-frac", type=float, default=0.5,
                     help="Only use each track's points from the bottom this-fraction of the "
                          "frame - closest to the camera, least perspective-distorted - for "
                          "direction and offset measurements. A track that never enters this "
                          "zone during the sampled window is skipped. Default 0.5 (bottom "
                          "half).")
    ap.add_argument("--min-speed-frac", type=float, default=0.35,
                     help="A track (either direction) moving slower than this fraction of the "
                          "majority direction's median speed is excluded as not real traffic "
                          "(e.g. a pedestrian mis-tracked as a bicycle, moving at walking "
                          "pace). Default 0.35.")
    ap.add_argument("--min-axis-alignment-deg", type=float, default=60.0,
                     help="A track must be within this many degrees of parallel/anti-parallel "
                          "to the road's own axis to count as direction-A/B traffic at all; "
                          "anything more sideways (e.g. a car crossing or turning at a far "
                          "intersection) is excluded as 'crossing' instead of being forced "
                          "into whichever direction its small forward/backward component "
                          "happens to lean toward. Confirmed on a real one-way site: without "
                          "this, near-perpendicular crossing traffic was being miscounted as "
                          "opposing traffic. Default 60 (must be within 60 deg of the axis; "
                          "the middle 60 deg band around perpendicular is excluded).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--debug", action="store_true",
                     help="Print diagnostic info: near-camera y-threshold, how many tracks "
                          "were dropped and why, and the corridor bounds used for the "
                          "off-road check.")
    ap.add_argument("--num-samples", type=int, default=1,
                     help="Sample this many windows spread evenly across the video's full "
                          "duration (instead of just one at the midpoint/--start-seconds), "
                          "and call the road TWO-WAY if ANY sampled window shows real "
                          "two-way traffic - a two-way road only has to prove it once, "
                          "while a ONE-WAY call requires every sample to agree. Useful for "
                          "low-traffic streets where a single short window might miss "
                          "oncoming traffic purely by chance (confirmed on a real, "
                          "known-two-way site: default single-window sampling can call it "
                          "ONE-WAY just because no opposing car happened to pass in that "
                          "window). Not compatible with --start-seconds. Default 1 (old "
                          "single-window behavior).")
    ap.add_argument("--require-overlap", action="store_true",
                     help="Additionally require that a direction-A track and a direction-B "
                          "track were actually active at the same time (a real simultaneous "
                          "pass), not just both present somewhere in the sampled window, "
                          "before calling a window two-way. Off by default: the majority-share "
                          "rule alone is already validated; this is a stricter, opt-in check "
                          "for when you want stronger side-by-side-in-time proof.")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg is required (used to fast-seek into the video) but isn't on PATH.\n"
            "Install it first, e.g. on macOS: brew install ffmpeg\n"
            "Then re-run this command."
        )

    if args.num_samples > 1 and args.start_seconds is not None:
        raise SystemExit(
            "--start-seconds is not compatible with --num-samples > 1 (sample windows are "
            "spread automatically across the video's duration)."
        )

    w, h, duration_s = get_video_meta(args.video)
    max_search = args.max_search_seconds if args.search else args.segment_seconds

    def run_one_window(start_seconds: float, out_path: Path) -> dict | None:
        print(f"Searching for a usable warm-up window near {start_seconds/3600:.2f}h in {args.video} ...")
        try:
            start_ts, tracks, spans = find_good_window_with_span(
                args.video, args.model, args.conf, args.classes, args.tracker,
                args.segment_seconds, args.min_track_len, args.needed_tracks,
                start_seconds, args.step_seconds, max_search,
            )
        except RuntimeError as e:
            print(f"  skipped: {e}")
            return None
        try:
            result = analyze_topology(
                tracks, min_net_displacement=args.min_net_displacement,
                one_way_majority_frac=args.one_way_majority_frac,
                min_minority_tracks=args.min_minority_tracks,
                frame_h=h, near_camera_frac=args.near_camera_frac,
                min_speed_frac=args.min_speed_frac,
                min_axis_alignment_deg=args.min_axis_alignment_deg,
                spans=spans, require_overlap=args.require_overlap,
                debug=args.debug,
            )
        except RuntimeError as e:
            print(f"  skipped: {e}")
            return None
        result["start_ts"] = start_ts
        print()
        print(f"Used window: {start_ts/3600:.2f}h - {(start_ts+args.segment_seconds)/3600:.2f}h")
        print(f"Direction A: {result['n_a']} tracks")
        print(f"Direction B: {result['n_b']} tracks")
        if result.get("n_off_road", 0):
            print(f"Excluded as off-road (e.g. sidewalk pedestrians/bicycles): {result['n_off_road']} tracks")
        if result.get("n_crossing", 0):
            print(f"Excluded as crossing/turning (not axis-aligned): {result['n_crossing']} tracks")
        if result.get("has_overlap") is not None:
            print(f"Simultaneous opposite-direction pairs (same-time A & B): {result['n_overlapping_pairs']}")
        print(f"Majority direction share: {result['majority_frac']*100:.1f}%")
        print(f"-> {'TWO-WAY' if result['is_two_way'] else 'ONE-WAY'}")
        print()
        save_preview(args.video, start_ts, tracks, result, out_path)
        return result

    if args.num_samples <= 1:
        if args.start_seconds is not None:
            start_seconds = args.start_seconds
        elif duration_s > 0:
            start_seconds = duration_s / 2.0
            print(f"No --start-seconds given: video is {duration_s/3600:.2f}h long, "
                  f"sampling from its midpoint ({start_seconds/3600:.2f}h).")
        else:
            start_seconds = 0.0
            print("Could not read video duration - falling back to start at 0s.")

        out_path = args.out or Path(args.video).with_name(Path(args.video).stem + "_topology_preview.png")
        result = run_one_window(start_seconds, out_path)
        if result is None:
            raise SystemExit("Could not find a usable window - try --search, a longer "
                              "--segment-seconds, or a different --start-seconds.")
        verdict = "TWO-WAY" if result["is_two_way"] else "ONE-WAY"
        print(f"VERDICT: {verdict}")
        return

    # Multi-sample mode: spread args.num_samples windows evenly across the
    # video and combine them - TWO-WAY if any sample shows real two-way
    # traffic, ONE-WAY only if every sample agrees.
    if duration_s <= 0:
        raise SystemExit("Could not determine this video's duration, which --num-samples needs "
                          "to spread windows across it.")
    print(f"Multi-sample mode: {args.num_samples} windows spread across the "
          f"{duration_s/3600:.2f}h video.")
    sample_starts = [duration_s * (i + 1) / (args.num_samples + 1) for i in range(args.num_samples)]
    out_base = args.out or Path(args.video).with_name(Path(args.video).stem + "_topology_preview.png")
    results: list[dict] = []
    for idx, s in enumerate(sample_starts):
        print(f"\n--- Sample {idx + 1}/{args.num_samples} ---")
        out_path = out_base.with_name(f"{out_base.stem}_sample{idx + 1}{out_base.suffix}")
        result = run_one_window(s, out_path)
        if result is not None:
            results.append(result)

    if not results:
        raise SystemExit("None of the sampled windows had enough usable traffic to classify. "
                          "Try more --num-samples, a longer --segment-seconds, or --search.")

    print("\n=== Overall (multi-sample) summary ===")
    for i, result in enumerate(results):
        print(f"  sample {i + 1} @ {result['start_ts']/3600:.2f}h: "
              f"{'TWO-WAY' if result['is_two_way'] else 'ONE-WAY'} "
              f"(A={result['n_a']} B={result['n_b']}, majority {result['majority_frac']*100:.1f}%)")
    overall_two_way = any(r["is_two_way"] for r in results)
    verdict = "TWO-WAY" if overall_two_way else "ONE-WAY"
    reason = ("at least one sampled window showed real two-way traffic" if overall_two_way
              else "every sampled window was single-direction")
    print(f"\nVERDICT: {verdict}  ({reason})")


if __name__ == "__main__":
    main()

# Roadmap note: lane counting per direction was tried via a gap-detection
# heuristic on perpendicular pixel offsets and produced clearly wrong
# results on a real site (16 + 4 "lanes" on a city street) - almost
# certainly because pixel offset doesn't scale linearly with real-world
# lateral distance across a frame with camera perspective, and because
# each track's own centroid (sampled wherever it happened to be along the
# road) isn't a fair like-for-like cross-section to compare against other
# tracks. Parked until the one-way/two-way logic above is validated on
# more sites. When it's revisited, two candidate fixes: (1) evaluate each
# track's fitted line at a shared reference cross-section along the major
# axis before comparing perpendicular offsets, instead of using each
# track's raw centroid, and (2) a proper clustering model (1D
# GaussianMixture + BIC, per the project's decided tech stack) instead of
# gap-threshold cutting, with a sanity cap on max plausible lanes.
