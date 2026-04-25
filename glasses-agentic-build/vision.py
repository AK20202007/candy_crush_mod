"""
Webcam capture + YOLOv8 + object-distance cues + agentic routing.

This replaces the old direct warning callback. The flow is now:

    YOLO detections -> structured Detection/WarningEvent -> AgenticNavigationRouter -> speech/haptic decision

The existing warning behavior is preserved, but it becomes one input to the router
instead of immediately speaking from inside the vision loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from agentic_layer import (
    AgentDecision,
    AgenticNavigationRouter,
    Direction,
    FrameContext,
    MotionState,
    RouteState,
    SceneState,
    UserState,
    WarningEvent,
)
from agentic_layer.models import Detection
from agentic_layer.utils import detection_from_bbox

# Standard COCO labels that matter most for navigation. Door/stairs/etc. are
# optional for custom models because yolov8n.pt will not expose them.
CORE_CLASSES: Set[str] = {
    "person",
    "chair",
    "bench",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "traffic light",
    "stop sign",
}
OPTIONAL_CLASSES: Set[str] = {
    "door",
    "stairs",
    "stair",
    "staircase",
    "curb",
    "elevator",
    "sign",
    "traffic cone",
}

STOP_WARNING_LABELS = {"stairs", "stair", "staircase", "curb", "car", "bus", "truck"}
OBSTACLE_LABELS = {
    "person",
    "chair",
    "bench",
    "door",
    "traffic cone",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
}


def _distance_bucket(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return "unknown"
    if distance_m <= 0.8:
        return "immediate"
    if distance_m <= 1.4:
        return "near"
    return "far"


@dataclass
class VisionConfig:
    """Tunable thresholds for warnings and YOLO inference."""

    model_path: str = "yolov8n.pt"
    obstacle_area_ratio: float = 0.12
    person_center_radius: float = 0.18
    warning_cooldown_s: float = 2.5
    confirm_frames: int = 2
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    half: Optional[bool] = None
    augment: bool = False
    max_det: int = 50

    # Demo defaults. On a phone these should come from IMU/GPS/user state.
    assume_moving: bool = True
    location_type: str = "unknown"
    visual_confidence: float = 0.75

    # Calibration multiplier for the monocular distance heuristic. Increase if
    # distances are consistently too short; decrease if they are too long.
    distance_scale: float = 1.0


class VisionSystem:
    """
    Opens the webcam, runs YOLO each frame, and emits one agent decision.

    The old system emitted strings like "Obstacle ahead" directly from the frame
    loop. This version emits structured context to the router, so safety warnings,
    route guidance, object finding, and orientation compete in one place.
    """

    def __init__(
        self,
        on_decision: Callable[[AgentDecision], None],
        config: Optional[VisionConfig] = None,
        router: Optional[AgenticNavigationRouter] = None,
        route_provider: Optional[Callable[[], RouteState]] = None,
        user_provider: Optional[Callable[[], UserState]] = None,
        motion_provider: Optional[Callable[[], MotionState]] = None,
    ) -> None:
        self._on_decision = on_decision
        self._cfg = config or VisionConfig()
        self._router = router or AgenticNavigationRouter(
            min_urgent_repeat_interval_ms=int(max(0.2, self._cfg.warning_cooldown_s) * 1000)
        )
        self._route_provider = route_provider
        self._user_provider = user_provider
        self._motion_provider = motion_provider

        self._model = YOLO(self._cfg.model_path)
        self._name_to_id: Dict[str, int] = {name.lower(): idx for idx, name in self._model.names.items()}

        self._active_classes = {name for name in CORE_CLASSES if name in self._name_to_id}
        for name in OPTIONAL_CLASSES:
            if name in self._name_to_id:
                self._active_classes.add(name)

        # If a custom model exposes none of these labels, let YOLO run all classes
        # rather than silently doing nothing.
        self._class_ids: Optional[List[int]] = None
        if self._active_classes:
            self._class_ids = sorted(self._name_to_id[n] for n in self._active_classes)

        self._consec_warning_hits: Dict[str, int] = {}
        self._last_spoken: Optional[str] = None

        use_half = self._cfg.half if self._cfg.half is not None else bool(torch.cuda.is_available())
        self._predict_half = bool(use_half and torch.cuda.is_available())

        print(
            f"[vision] Inference: imgsz={self._cfg.imgsz} conf={self._cfg.conf} "
            f"iou={self._cfg.iou} tracked_classes={len(self._active_classes) or 'all'} "
            f"half={self._predict_half} augment={self._cfg.augment} "
            f"confirm_frames={self._cfg.confirm_frames}"
        )

    def active_class_labels(self) -> Set[str]:
        return set(self._active_classes)

    def run_forever(self, camera_index: int = 0, stop_event=None) -> None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}")

        print("[vision] Webcam open. Press 'q' in the preview window or Ctrl+C in the terminal to quit.")
        print(f"[vision] Tracking classes: {sorted(self._active_classes) if self._active_classes else 'all'}")

        fps_frames = 0
        fps_t0 = time.time()

        try:
            while stop_event is None or not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[vision] Frame grab failed; retrying...")
                    time.sleep(0.05)
                    continue

                h, w = frame.shape[:2]
                decision = self._process_frame(frame, w, h)
                if decision.should_speak and decision.message:
                    self._last_spoken = decision.message
                    self._on_decision(decision)

                fps_frames += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 3.0:
                    print(f"[vision] ~{fps_frames / elapsed:.1f} FPS over last {elapsed:.1f}s")
                    fps_frames = 0
                    fps_t0 = time.time()

                cv2.imshow("Assistive Nav — preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[vision] Quit requested from preview window.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray, w: int, h: int) -> AgentDecision:
        now_ms = int(time.time() * 1000)
        detections = self._detect(frame, w, h)
        warnings = self._warnings_from_detections(detections, now_ms)
        ctx = FrameContext(
            timestamp_ms=now_ms,
            frame_id=str(time.time()),
            detections=detections,
            warnings=warnings,
            motion=self._motion_state(),
            route=self._route_state(),
            scene=SceneState(
                location_type=self._cfg.location_type,
                visual_confidence=self._cfg.visual_confidence,
            ),
            user=self._user_state(),
            last_spoken=self._last_spoken,
        )
        decision = self._router.decide(ctx)
        self._draw_decision(frame, decision)
        return decision

    def _detect(self, frame: np.ndarray, w: int, h: int) -> List[Detection]:
        cfg = self._cfg
        results = self._model.predict(
            source=frame,
            verbose=False,
            stream=False,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            half=self._predict_half,
            augment=cfg.augment,
            max_det=cfg.max_det,
            classes=self._class_ids,
        )

        if not results:
            return []
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        boxes = r0.boxes.xyxy.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else np.ones(len(boxes))

        detections: List[Detection] = []
        for i, ((x1, y1, x2, y2), cls_id) in enumerate(zip(boxes, clss)):
            label = self._model.names[int(cls_id)].lower()
            confidence = float(confs[i]) if i < len(confs) else 0.0
            det = detection_from_bbox(
                label=label,
                confidence=confidence,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                frame_width=float(w),
                frame_height=float(h),
                distance_scale=self._cfg.distance_scale,
                source="ultralytics-yolov8-distance",
            )
            detections.append(det)
            self._draw_detection(frame, det)

        return detections

    def _warnings_from_detections(self, detections: List[Detection], observed_at_ms: Optional[int] = None) -> List[WarningEvent]:
        n_confirm = max(1, self._cfg.confirm_frames)
        observed_at_ms = observed_at_ms or int(time.time() * 1000)
        raw: List[WarningEvent] = []

        for det in detections:
            label = det.label.lower()
            area_ratio = float(det.attributes.get("area_ratio", 0.0) or 0.0)
            centered = self._is_centered_for_warning(det)
            close = det.distance_m is not None and det.distance_m <= 1.4
            large = area_ratio >= self._cfg.obstacle_area_ratio

            if label in STOP_WARNING_LABELS and centered and (close or large):
                raw.append(
                    WarningEvent(
                        kind=label,
                        message=f"{label} ahead",
                        severity="critical",
                        confidence=det.confidence,
                        distance_m=det.distance_m,
                        direction=det.direction,
                        source="vision-yolo-distance",
                        observed_at_ms=observed_at_ms,
                    )
                )
            elif label == "person" and centered and (close or large):
                raw.append(
                    WarningEvent(
                        kind="person",
                        message="person ahead",
                        severity="high" if close or large else "medium",
                        confidence=det.confidence,
                        distance_m=det.distance_m,
                        direction=det.direction,
                        source="vision-yolo-distance",
                        observed_at_ms=observed_at_ms,
                    )
                )
            elif label in OBSTACLE_LABELS and centered and (close or large):
                raw.append(
                    WarningEvent(
                        kind="obstacle",
                        message=f"{label} ahead",
                        severity="high",
                        confidence=det.confidence,
                        distance_m=det.distance_m,
                        direction=det.direction,
                        source="vision-yolo-distance",
                        observed_at_ms=observed_at_ms,
                    )
                )

        confirmed: List[WarningEvent] = []
        current_keys = set()
        for warning in raw:
            key = f"{warning.kind}:{warning.message}:{warning.direction.value}:{_distance_bucket(warning.distance_m)}"
            current_keys.add(key)
            self._consec_warning_hits[key] = self._consec_warning_hits.get(key, 0) + 1
            if self._consec_warning_hits[key] >= n_confirm:
                confirmed.append(warning)

        # Decay counters for warnings no longer visible.
        for key in list(self._consec_warning_hits):
            if key not in current_keys:
                self._consec_warning_hits[key] = 0

        return confirmed

    def _is_centered_for_warning(self, det: Detection) -> bool:
        if det.direction not in {Direction.CENTER, Direction.SLIGHT_LEFT, Direction.SLIGHT_RIGHT}:
            return False
        if det.label != "person":
            return True
        center_x = det.attributes.get("center_x_ratio")
        if center_x is None:
            return True
        return abs(float(center_x) - 0.5) <= max(0.05, self._cfg.person_center_radius)

    def _route_state(self) -> RouteState:
        if self._route_provider is None:
            return RouteState(active=False)
        try:
            return self._route_provider()
        except Exception as exc:
            print(f"[vision] route_provider failed: {exc}")
            return RouteState(active=False)

    def _user_state(self) -> UserState:
        if self._user_provider is None:
            return UserState()
        try:
            return self._user_provider()
        except Exception as exc:
            print(f"[vision] user_provider failed: {exc}")
            return UserState()

    def _motion_state(self) -> MotionState:
        if self._motion_provider is None:
            return MotionState(is_moving=self._cfg.assume_moving)
        try:
            return self._motion_provider()
        except Exception as exc:
            print(f"[vision] motion_provider failed: {exc}")
            return MotionState(is_moving=self._cfg.assume_moving)

    @staticmethod
    def _draw_detection(frame: np.ndarray, det: Detection) -> None:
        if det.bbox is None:
            return
        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
        color = (0, 200, 0) if det.label == "person" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        distance = "?"
        if det.distance_m is not None:
            distance = f"{det.distance_m * 3.28084:.1f}ft"
        text = f"{det.label} {det.confidence:.2f} {det.direction.value} {distance}"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_decision(frame: np.ndarray, decision: AgentDecision) -> None:
        if not decision.message:
            return
        text = f"{decision.action.value}: {decision.message[:80]}"
        cv2.putText(
            frame,
            text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
