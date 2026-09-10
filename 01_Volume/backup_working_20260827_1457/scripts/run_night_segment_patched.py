def run_night_segment(cfg: dict, video_path: Path, fps: float,
                      start_s: float, end_s: float,
                      direction_names: dict) -> list[dict]:
    """Night detector v5 - motion-based direction + time+spatial dedup."""
    night = cfg.get("night", {})
    roi_y_min = int(night.get("roi_y_min", 100))
    pair_max_dx = float(night.get("pair_max_dx", 40))
    pair_max_dy = float(night.get("pair_max_dy", 8))
    track_max_dist = float(night.get("track_max_distance", 45))
    track_max_missed = int(night.get("track_max_missed", 12))
    merge_dist = float(night.get("merge_distance", 80))
    singleton_min_area = int(night.get("singleton_min_area", 120))
    dedup_seconds = float(night.get("dedup_seconds_night", 1.0))
    dedup_buffer_px = float(night.get("dedup_spatial_buffer_px_night", 100))
    line = cfg["lines"][0]["points"]
    line_name = cfg["lines"][0]["name"]

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))

    tracker = CentroidTracker(max_distance=track_max_dist, max_missed=track_max_missed)
    last_pos: dict[int, tuple[float, float]] = {}
    counted: set[int] = set()
    crossing_history: dict[str, list] = {"A": [], "B": []}
    rows: list[dict] = []

    frame_idx = int(start_s * fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        if ts > end_s:
            break

        heads = detect_headlights(frame, roi_y_min)
        tails = detect_taillights(frame, roi_y_min)
        merged_blobs = list(heads)
        for t in tails:
            close = any(((t["cx"] - m["cx"]) ** 2 + (t["cy"] - m["cy"]) ** 2) ** 0.5 < merge_dist
                        for m in merged_blobs)
            if not close:
                merged_blobs.append(t)
        all_pairs = pair_lights(merged_blobs, pair_max_dx, pair_max_dy,
                                single_min_area=singleton_min_area)
        for p in all_pairs:
            p["kind"] = "vehicle"

        for tid, p in tracker.update(all_pairs):
            curr = (p["cx"], p["cy"])
            prev = last_pos.get(tid)
            last_pos[tid] = curr
            if prev is not None and tid not in counted:
                if crossed_line(prev, curr, line):
                    (x1, y1), (x2, y2) = line[0], line[1]
                    cross = ((x2 - x1) * (prev[1] - y1)
                             - (y2 - y1) * (prev[0] - x1))
                    side = "A" if cross > 0 else "B"
                    history = crossing_history[side]
                    history[:] = [(t, x) for t, x in history if (ts - t) < dedup_seconds]
                    curr_x = curr[0]
                    duplicate = any(abs(curr_x - prev_x) < dedup_buffer_px
                                    for _, prev_x in history)
                    if duplicate:
                        continue
                    counted.add(tid)
                    crossing_history[side].append((ts, curr_x))
                    rows.append({
                        "timestamp_s": ts, "line": line_name, "track_id": f"N{tid}",
                        "direction_code": side,
                        "direction": direction_names.get(side, side),
                        "mode": "night",
                    })
        frame_idx += 1
    cap.release()
    return rows
