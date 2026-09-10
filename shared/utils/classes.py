"""
Vehicle class mapping used across all three counting sections.

Maps COCO class IDs (from YOLO) to FHWA-style vehicle categories
commonly used in traffic studies. Adjust to match your local agency's
classification scheme (e.g., FHWA 13-class, MUTCD, or custom).
"""

# COCO class IDs from a standard YOLO model
COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Simplified traffic-study categories
TRAFFIC_CATEGORIES = {
    "bicycle": "Bicycle",
    "motorcycle": "Motorcycle",
    "car": "Passenger Car",
    "bus": "Bus",
    "truck": "Truck",
}

# FHWA 13-class scheme (placeholder — refine with axle/length logic)
FHWA_13 = [
    "1 Motorcycles",
    "2 Passenger Cars",
    "3 Pickups, Vans, Other 2-Axle 4-Tire",
    "4 Buses",
    "5 2-Axle, 6-Tire Single Unit",
    "6 3-Axle Single Unit",
    "7 4 or More Axle Single Unit",
    "8 4 or Fewer Axle Single Trailer",
    "9 5-Axle Single Trailer",
    "10 6 or More Axle Single Trailer",
    "11 5 or Fewer Axle Multi-Trailer",
    "12 6-Axle Multi-Trailer",
    "13 7 or More Axle Multi-Trailer",
]


def coco_to_category(coco_id: int) -> str | None:
    """Map a YOLO/COCO class ID to a simplified traffic category."""
    name = COCO_VEHICLE_CLASSES.get(coco_id)
    if name is None:
        return None
    return TRAFFIC_CATEGORIES[name]
