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


# Estimated real-world widths in meters for common objects.
REAL_WORLD_WIDTHS_M = {
    "person": 0.45,
    "car": 1.8,
    "chair": 0.5,
    "dining table": 1.2,
    "couch": 2.0,
    "bench": 1.5,
    "bus": 2.5,
    "truck": 2.5,
    "bicycle": 0.6,
    "motorcycle": 0.8,
    "door": 0.9,
    "stairs": 1.2,
    "traffic cone": 0.3,
    "fire hydrant": 0.3,
    "stop sign": 0.75,
}

def estimate_distance_m_from_bbox(
    bbox: BBox, 
    label: str, 
    frame_width: float, 
    distance_scale: float = 1.0
) -> float:
    """
    Estimate real-world distance in meters using focal length and known widths.
    Assumes a standard ~70 degree horizontal FOV (focal length ~450px for 640px width).
    """
    # 1. Use label-specific width if available, else default to 0.5m
    real_width_m = REAL_WORLD_WIDTHS_M.get(label.lower(), 0.5)
    
    # 2. Approximate focal length in pixels (assuming 70deg HFOV)
    # focal_length = (width / 2) / tan(hfov / 2)
    # For 640px, focal_length is approx 450.
    focal_length_px = (frame_width / 2.0) / 0.7  # tan(35deg) is ~0.7
    
    # 3. Calculate distance: d = (W * f) / w
    bbox_width_px = max(1.0, bbox.width)
    distance_m = (real_width_m * focal_length_px) / bbox_width_px
    
    # 4. Apply calibration scale
    return max(0.1, distance_m * max(0.01, distance_scale))


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
    distance_m = estimate_distance_m_from_bbox(bbox, label, frame_width, distance_scale)
    
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        direction=direction_from_bbox(bbox, frame_width),
        distance_m=distance_m,
        source=source,
        attributes={
            "area_ratio": bbox_area_ratio(bbox, frame_width, frame_height),
            "center_x_ratio": bbox.center_x / max(1.0, frame_width),
            "center_y_ratio": bbox.center_y / max(1.0, frame_height),
        },
    )
