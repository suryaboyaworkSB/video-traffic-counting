"""
Turning Movement Counts (TMC) with vehicle class.

For every tracked vehicle we record the first approach zone it entered
and the last approach zone it exited through, then look up the movement
(Through / Left / Right / U-turn) from the config matrix.

Usage:
    python count_tmc.py --config ../config/config.example.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils.tracker import track_video
from shared.utils.zones import bbox_center, point_in_zone
from shared.utils.classes import coco_to_category


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    approaches: dict[str, list[list[float]]] = cfg["approaches"]
    movements: dict[str, dict[str, str]] = cfg["movements"]

    # Per-track state: first zone entered, last zone seen, last timestamp, class
    first_zone: dict[int, str] = {}
    last_zone: dict[int, str] = {}
    last_ts: dict[int, float] = {}
    track_class: dict[int, str] = {}

    for det in track_video(
        cfg["video"], model_path=cfg["model"], conf=cfg["conf"],
        classes=cfg.get("classes"),
    ):
        tid = det["track_id"]
        center = bbox_center(det["xyxy"])
        track_class.setdefault(tid, coco_to_category(det["class_id"]) or "Unknown")
        last_ts[tid] = det["timestamp_s"]

        for leg, poly in approaches.items():
            if point_in_zone(center, poly):
                first_zone.setdefault(tid, leg)
                last_zone[tid] = leg
                break

    rows = []
    for tid, entry in first_zone.items():
        exit_ = last_zone.get(tid)
        if exit_ is None or exit_ == entry and last_ts[tid] < 1.0:
            continue
        movement = movements.get(entry, {}).get(exit_, "Unknown")
        rows.append({
            "track_id": tid,
            "entry": entry,
            "exit": exit_,
            "movement": movement,
            "category": track_class[tid],
            "timestamp_s": last_ts[tid],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No completed movements detected.")
        return

    df["bin_min"] = (df["timestamp_s"] // (cfg["bin_minutes"] * 60)).astype(int) * cfg["bin_minutes"]

    by_movement = df.pivot_table(
        index=["bin_min", "entry"], columns="movement",
        values="track_id", aggfunc="count", fill_value=0,
    )
    by_class = df.pivot_table(
        index=["entry", "movement"], columns="category",
        values="track_id", aggfunc="count", fill_value=0,
    )
    by_class["Total"] = by_class.sum(axis=1)

    out = Path(cfg["output"]["xlsx"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        by_movement.to_excel(xw, sheet_name="TMC by bin")
        by_class.to_excel(xw, sheet_name="Movement x Class")
        df.to_excel(xw, sheet_name="Raw movements", index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
