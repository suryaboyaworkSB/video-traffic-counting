"""
Zone / line crossing helpers.

- A 'line' is two (x, y) points; a vehicle is counted when its track
  center crosses the line.
- A 'zone' is a polygon used for entry/exit logic in TMC counts.
"""

from shapely.geometry import LineString, Point, Polygon


def crossed_line(prev_xy: tuple[float, float], curr_xy: tuple[float, float], line: list[tuple[float, float]]) -> bool:
    """True if the segment prev -> curr crosses the count line."""
    seg = LineString([prev_xy, curr_xy])
    return seg.intersects(LineString(line))


def point_in_zone(xy: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    return Polygon(polygon).contains(Point(xy))


def bbox_center(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
