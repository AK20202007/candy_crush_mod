from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from agentic_layer import BBox, Detection, Direction, SurfaceKind  # noqa: E402
from agentic_layer.utils import detection_from_bbox  # noqa: E402
from vision import VisionConfig, VisionSystem  # noqa: E402


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


def test_detection_from_bbox_marks_frame_edge_partial() -> None:
    det = detection_from_bbox(
        label="chair",
        confidence=0.81,
        x1=0,
        y1=120,
        x2=90,
        y2=478,
        frame_width=640,
        frame_height=480,
    )

    assert det.attributes["edge_truncated"] is True
    assert det.attributes["partial_visibility"] == "frame_edge"
    assert det.attributes["distance_reliability"] == "low"
    assert "left" in det.attributes["edge_contact"]
    assert "bottom" in det.attributes["edge_contact"]


def test_door_handle_detector_finds_horizontal_lever() -> None:
    frame = np.full((480, 640, 3), (150, 150, 145), dtype=np.uint8)
    cv2.line(frame, (250, 80), (250, 450), (70, 70, 70), 3)
    cv2.line(frame, (520, 80), (520, 450), (70, 70, 70), 3)
    cv2.rectangle(frame, (330, 185), (370, 345), (105, 105, 100), 4)
    cv2.rectangle(frame, (345, 278), (470, 305), (205, 205, 195), -1)
    cv2.rectangle(frame, (345, 278), (470, 305), (90, 90, 85), 2)

    vision = object.__new__(VisionSystem)
    vision._cfg = VisionConfig(camera_mount="hand")

    obs = vision._detect_door(frame, 640, 480)

    assert obs is not None
    assert obs.kind == SurfaceKind.DOOR
    assert obs.source == "vision-door-handle"
    assert obs.attributes["handle_detected"] is True
    assert obs.attributes["handle_orientation"] == "lever_horizontal"
    assert obs.attributes["recommended_hand"] == "right"
    assert obs.attributes["handle_bbox"][0] < obs.attributes["handle_bbox"][2]


def test_door_assist_gate_skips_door_detector() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    vision = object.__new__(VisionSystem)
    vision._cfg = VisionConfig(enable_door_assist=False)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("door detector should not run when door assist is disabled")

    vision._detect_door = fail_if_called

    observations = vision._surface_observations(frame, 320, 240)

    assert all(obs.kind != SurfaceKind.DOOR for obs in observations)


def main() -> None:
    test_green_traffic_light_crop_is_categorized()
    test_red_traffic_light_crop_is_categorized()
    test_surface_crop_identifies_bright_gray_as_sidewalk()
    test_surface_crop_identifies_dark_gray_as_road()
    test_detection_from_bbox_marks_frame_edge_partial()
    test_door_handle_detector_finds_horizontal_lever()
    test_door_assist_gate_skips_door_detector()
    print("vision signal state tests passed")


if __name__ == "__main__":
    main()
