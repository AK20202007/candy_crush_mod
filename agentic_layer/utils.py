from __future__ import annotations

import math
from typing import Optional

from .models import BBox, Detection, Direction


def direction_from_bbox(bbox: BBox, frame_width: float) -> Direction:
    """Convert a bbox center into blind-navigation-friendly coarse direction."""
    if frame_width <= 0:
        return Direction.UNKNOWN
    x = bbox.center_x / frame_width
    if x < 0.20:
        return Direction.LEFT
    if x < 0.42:
        return Direction.SLIGHT_LEFT
    if x <= 0.58:
        return Direction.CENTER
    if x <= 0.80:
        return Direction.SLIGHT_RIGHT
    return Direction.RIGHT


def bbox_area_ratio(bbox: BBox, frame_width: float, frame_height: float) -> float:
    area = max(1.0, frame_width * frame_height)
    return (bbox.width * bbox.height) / area


def distance_inches_to_meters(distance_inches: Optional[float]) -> Optional[float]:
    if distance_inches is None:
        return None
    return max(0.0, float(distance_inches) * 0.0254)


def estimate_distance_inches_from_bbox(bbox: BBox, distance_scale: float = 1.0) -> float:
    """
    Monocular distance heuristic adapted from paul-pias/Object-Detection-and-Distance-Measurement.

    This is not true depth. Treat it as a rough ranking signal until calibrated with
    a known phone camera, object class, and real-world measurements.
    """
    width = max(1.0, bbox.width)
    height = max(1.0, bbox.height)
    raw_inches = ((2 * math.pi * 180.0) / (width + height * 360.0)) * 1000.0 + 3.0
    return max(1.0, raw_inches * max(0.01, distance_scale))


def detection_from_bbox(
    label: str,
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: float,
    frame_height: float,
    distance_scale: float = 1.0,
    source: str = "yolov8-distance",
) -> Detection:
    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
    distance_inches = estimate_distance_inches_from_bbox(bbox, distance_scale=distance_scale)
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        direction=direction_from_bbox(bbox, frame_width),
        distance_m=distance_inches_to_meters(distance_inches),
        source=source,
        attributes={
            "area_ratio": bbox_area_ratio(bbox, frame_width, frame_height),
            "center_x_ratio": bbox.center_x / max(1.0, frame_width),
            "center_y_ratio": bbox.center_y / max(1.0, frame_height),
            "distance_inches_heuristic": distance_inches,
        },
    )
