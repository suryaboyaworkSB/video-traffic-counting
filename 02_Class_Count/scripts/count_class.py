"""
Class counting — counts vehicles per category crossing each line.

Usage:
    python count_class.py --config ../config/config.example.yaml
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.utils.tracker import track_video
from shared.utils.zones import bbox_center, crossed_line
from shared.utils.classes import coco_to_category


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    last_pos: dict[int, tuple[float, float]] = {}
    counted_by_line: dict[str, set[int]] = defaultdict(set)
    rows = []

    for det in track_video(
        cfg["video"], model_path=cfg["model"], conf=cfg["conf"],
        classes=cfg.get("classes"),
    ):
        tid = det["track_id"]
        curr = bbox_center(det["xyxy"])
        prev = last_pos.get(tid)
        last_pos[tid] = curr
        if prev is None:
            continue
        category = coco_to_category(det["class_id"]) or "Unknown"
        for line in cfg["lines"]:
            if tid in counted_by_line[line["name"]]:
                continue
            if crossed_line(prev, curr, line["points"]):
                counted_by_line[line["name"]].add(tid)
                rows.append({
                    "timestamp_s": det["timestamp_s"],
                    "line": line["name"],
                    "track_id": tid,
                    "category": category,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No crossings detected.")
        return

    df["bin_min"] = (df["timestamp_s"] // (cfg["bin_minutes"] * 60)).astype(int) * cfg["bin_minutes"]
    pivot = df.pivot_table(index=["bin_min", "line"], columns="category",
                           values="track_id", aggfunc="count", fill_value=0)
    pivot["Total"] = pivot.sum(axis=1)

    out = Path(cfg["output"]["xlsx"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        pivot.to_excel(xw, sheet_name="Class by bin")
        df.to_excel(xw, sheet_name="Raw crossings", index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
