from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from agentic_layer import BBox, Detection, Direction, SurfaceKind  # noqa: E402
from vision import VisionSystem  # noqa: E402


def _traffic_light_detection() -> Detection:
    return Detection(
        label="traffic light",
        confidence=0.9,
        bbox=BBox(10, 10, 50, 90),
        distance_m=8.0,
        direction=Direction.CENTER,
    )


def test_green_traffic_light_crop_is_categorized() -> None:
    frame = np.zeros((100, 60, 3), dtype=np.uint8)
    cv2.circle(frame, (30, 70), 10, (0, 255, 0), -1)

    state, confidence, scores = VisionSystem._estimate_traffic_light_state(frame, _traffic_light_detection())

    assert state == "green"
    assert confidence > 0.0
    assert scores["green"] > scores["red"]


def test_red_traffic_light_crop_is_categorized() -> None:
    frame = np.zeros((100, 60, 3), dtype=np.uint8)
    cv2.circle(frame, (30, 25), 10, (0, 0, 255), -1)

    state, confidence, scores = VisionSystem._estimate_traffic_light_state(frame, _traffic_light_detection())

    assert state == "red"
    assert confidence > 0.0
    assert scores["red"] > scores["green"]


def test_surface_crop_identifies_bright_gray_as_sidewalk() -> None:
    crop = np.full((120, 160, 3), (165, 165, 165), dtype=np.uint8)

    obs = VisionSystem._classify_surface_crop(object.__new__(VisionSystem), crop, Direction.CENTER)

    assert obs is not None
    assert obs.kind == SurfaceKind.SIDEWALK
    assert obs.confidence >= 0.42
    assert obs.direction == Direction.CENTER


def test_surface_crop_identifies_dark_gray_as_road() -> None:
    crop = np.full((120, 160, 3), (55, 55, 55), dtype=np.uint8)

    obs = VisionSystem._classify_surface_crop(object.__new__(VisionSystem), crop, Direction.CENTER)

    assert obs is not None
    assert obs.kind == SurfaceKind.ROAD
    assert obs.confidence >= 0.42


def main() -> None:
    test_green_traffic_light_crop_is_categorized()
    test_red_traffic_light_crop_is_categorized()
    test_surface_crop_identifies_bright_gray_as_sidewalk()
    test_surface_crop_identifies_dark_gray_as_road()
    print("vision signal state tests passed")


if __name__ == "__main__":
    main()
