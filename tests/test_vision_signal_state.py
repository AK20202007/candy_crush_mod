from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from agentic_layer import BBox, Detection, Direction, SurfaceKind  # noqa: E402
from agentic_layer.utils import detection_from_bbox  # noqa: E402
from vision import VisionConfig, VisionSystem  # noqa: E402


class _FakeDoorPartsResults:
    def __init__(self, rows):
        self.xyxy = [np.asarray(rows, dtype=np.float32)]


class _FakeDoorPartsModel:
    names = {0: "door_flap", 1: "handle", 2: "hinge"}

    def __call__(self, frame, size=416):
        return _FakeDoorPartsResults(
            [
                [414, 252, 462, 296, 0.63, 1],
                [150, 60, 510, 430, 0.72, 0],
            ]
        )


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


def test_door_handle_like_pattern_is_not_confirmed_doorway() -> None:
    frame = np.full((480, 640, 3), (150, 150, 145), dtype=np.uint8)
    cv2.line(frame, (250, 80), (250, 450), (70, 70, 70), 3)
    cv2.line(frame, (520, 80), (520, 450), (70, 70, 70), 3)
    cv2.rectangle(frame, (330, 185), (370, 345), (105, 105, 100), 4)
    cv2.rectangle(frame, (345, 278), (470, 305), (205, 205, 195), -1)
    cv2.rectangle(frame, (345, 278), (470, 305), (90, 90, 85), 2)

    vision = object.__new__(VisionSystem)
    vision._cfg = VisionConfig(camera_mount="hand")
    vision._consec_door_hits = 0

    # Handle-like lines are too easy to hallucinate, so they must not become a
    # confirmed door surface unless the conservative doorway checks also pass.
    for _ in range(3):
        obs = vision._detect_door(frame, 640, 480)

    assert obs is None


def test_wall_plane_detector_finds_wall_like_obstacle() -> None:
    frame = np.full((480, 640, 3), (145, 145, 142), dtype=np.uint8)
    for x in [190, 260, 330, 405]:
        cv2.line(frame, (x, 130), (x, 430), (70, 70, 70), 4)
    cv2.line(frame, (170, 410), (470, 410), (85, 85, 85), 3)

    vision = object.__new__(VisionSystem)
    vision._cfg = VisionConfig(camera_mount="hand")

    obs = vision._detect_wall_plane(frame, 640, 480)

    assert obs is not None
    assert obs.kind == SurfaceKind.WALL
    assert obs.source == "vision-wall-plane"
    assert obs.confidence >= 0.58
    assert obs.attributes["vertical_lines"] >= 2


def test_round_door_knob_with_frame_is_confirmed() -> None:
    frame = np.full((480, 640, 3), (155, 155, 150), dtype=np.uint8)
    cv2.line(frame, (160, 80), (160, 440), (65, 65, 65), 5)
    cv2.line(frame, (500, 80), (500, 440), (65, 65, 65), 5)
    cv2.line(frame, (160, 82), (500, 82), (75, 75, 75), 4)
    cv2.circle(frame, (438, 275), 18, (210, 210, 200), -1)
    cv2.circle(frame, (438, 275), 18, (70, 70, 65), 3)
    cv2.circle(frame, (432, 269), 5, (245, 245, 235), -1)

    vision = object.__new__(VisionSystem)
    vision._cfg = VisionConfig(camera_mount="hand")
    vision._consec_door_hits = 0
    vision._door_candidate_key = None

    obs = None
    for _ in range(3):
        obs = vision._detect_door(frame, 640, 480)

    assert obs is not None
    assert obs.kind == SurfaceKind.DOOR
    assert obs.source == "vision-door-handle"
    assert obs.attributes["handle_orientation"] == "round_knob"
    assert obs.attributes["clear_handle"] is True
    assert "turn the knob" in obs.attributes["handle_action"]


def test_optional_door_parts_model_promotes_handle_surface() -> None:
    frame = np.full((480, 640, 3), (150, 150, 145), dtype=np.uint8)

    vision = object.__new__(VisionSystem)
    vision._cfg = VisionConfig(camera_mount="hand", door_parts_conf=0.40)
    vision._door_parts_model = _FakeDoorPartsModel()

    observations = vision._detect_door_parts(frame, 640, 480)

    assert len(observations) == 1
    obs = observations[0]
    assert obs.kind == SurfaceKind.DOOR
    assert obs.source == "joechencc-door-parts-handle"
    assert obs.attributes["model_source"] == "Joechencc/Door_detection"
    assert obs.attributes["clear_handle"] is True
    assert obs.attributes["has_frame"] is True
    assert obs.attributes["handle_orientation"] == "model_handle"


def main() -> None:
    test_green_traffic_light_crop_is_categorized()
    test_red_traffic_light_crop_is_categorized()
    test_surface_crop_identifies_bright_gray_as_sidewalk()
    test_surface_crop_identifies_dark_gray_as_road()
    test_detection_from_bbox_marks_frame_edge_partial()
    test_door_handle_like_pattern_is_not_confirmed_doorway()
    test_wall_plane_detector_finds_wall_like_obstacle()
    test_round_door_knob_with_frame_is_confirmed()
    test_optional_door_parts_model_promotes_handle_surface()
    print("vision signal state tests passed")


if __name__ == "__main__":
    main()
