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

try:
    from CoreMotion import CMAltimeter, CMMotionManager
    from Foundation import NSOperationQueue
    HAS_CORE_MOTION = True
except ImportError:
    HAS_CORE_MOTION = False

from agentic_layer import (
    AgentDecision,
    AgenticNavigationRouter,
    Direction,
    FrameContext,
    MotionState,
    RouteState,
    SceneState,
    SurfaceKind,
    SurfaceObservation,
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
SIGNAL_CLASSES: Set[str] = {
    "pedestrian signal",
    "pedestrian crossing signal",
    "crosswalk signal",
    "walk signal",
    "walk sign",
    "dont walk sign",
    "don't walk sign",
    "do not walk sign",
    "countdown signal",
    "traffic signal",
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
    *SIGNAL_CLASSES,
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


def _signal_state_from_label(label: str) -> str:
    normalized = label.lower().replace("_", " ").replace("-", " ")
    if any(term in normalized for term in {"dont walk", "don't walk", "do not walk", "red hand", "stop hand"}):
        return "dont_walk"
    if "countdown" in normalized:
        return "countdown"
    if "walk" in normalized:
        return "walk"
    return "unknown"


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

    # Lightweight sidewalk/road/curb heuristic. This is intentionally
    # conservative; production sidewalk navigation should replace or supplement
    # it with segmentation + depth.
    enable_surface_heuristic: bool = True


class MotionFallDetector:
    """Uses Apple's CoreMotion (CMAltimeter) to detect rapid altitude drops."""

    def __init__(self, drop_threshold: float = -1.37): # 4.5 feet is roughly 1.37 meters
        self._altimeter = None
        self._drop_threshold = drop_threshold
        self._is_available = False
        self._fall_detected = False
        self._history = deque() # tuples of (timestamp, altitude)
        self._fall_candidate_time = None

        if HAS_CORE_MOTION:
            try:
                self._altimeter = CMAltimeter.alloc().init()
                self._is_available = CMAltimeter.isRelativeAltitudeAvailable()
            except Exception as e:
                print(f"[motion] Error initializing CMAltimeter: {e}")

    def start(self, callback: Callable[[float], None]):
        """Starts listening for altitude changes."""
        if not self._is_available or self._altimeter is None:
            print("[motion] Relative altitude not available on this device.")
            return

        queue = NSOperationQueue.mainQueue()

        def _handler(data, error):
            if error:
                return
            
            now = time.time()
            alt = data.relativeAltitude().floatValue()
            
            self._history.append((now, alt))
            # Keep a 5-second rolling window to find max altitude before the fall
            while self._history and now - self._history[0][0] > 5.0:
                self._history.popleft()
                
            max_recent_alt = max(a for _, a in self._history)
            diff = alt - max_recent_alt # typically negative during a drop
            
            if diff <= self._drop_threshold:
                if self._fall_candidate_time is None:
                    self._fall_candidate_time = now
                elif now - self._fall_candidate_time >= 20.0:
                    if not self._fall_detected:
                        self._fall_detected = True
                        callback(diff)
            else:
                self._fall_candidate_time = None

        self._altimeter.startRelativeAltitudeUpdatesToQueue_withHandler_(queue, _handler)

    def is_fall_detected(self) -> bool:
        return self._fall_detected

    def reset(self):
        self._fall_detected = False
        self._fall_candidate_time = None


class SignalMagnitudeFallDetector:
    """
    Implements fall detection logic from aadithyanr/Fall-Detection.
    Calculates Signal Magnitude Vector (SMV) and checks orientation.
    """

    def __init__(self, smv_threshold: float = 25.0, angle_threshold: float = 30.0):
        self._smv_threshold = smv_threshold
        self._angle_threshold = angle_threshold
        self._manager = None
        self._is_available = False
        self._fall_detected = False

        if HAS_CORE_MOTION:
            try:
                self._manager = CMMotionManager.alloc().init()
                self._is_available = self._manager.isAccelerometerAvailable()
            except Exception as e:
                print(f"[sensor] Error initializing CMMotionManager: {e}")

    def start(self, callback: Callable[[float, float, float], None]):
        if not self._is_available or self._manager is None:
            print("[sensor] Accelerometer not available on this device.")
            return

        queue = NSOperationQueue.mainQueue()

        def _handler(data, error):
            if error:
                return
            accel = data.accelerometerData().acceleration()
            # Acceleration is in Gs on iOS/macOS, convert to m/s^2 if needed
            # The Java code uses m/s^2 (G=9.8), so we multiply by 9.8
            ax, ay, az = accel.x * 9.8, accel.y * 9.8, accel.z * 9.8
            smv = (ax**2 + ay**2 + az**2)**0.5

            if smv > self._smv_threshold:
                # In a real app we'd check deviceMotion for orientation (fused pitch/roll)
                # For this prototype, we'll trigger if SMV is high.
                self._fall_detected = True
                callback(smv, ax, ay)

        self._manager.startAccelerometerUpdatesToQueue_withHandler_(queue, _handler)

    def is_fall_detected(self) -> bool:
        return self._fall_detected

    def reset(self):
        self._fall_detected = False


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
        on_frame_decision: Optional[Callable[[FrameContext, AgentDecision], None]] = None,
    ) -> None:
        self._on_decision = on_decision
        self._cfg = config or VisionConfig()
        self._router = router or AgenticNavigationRouter(
            min_urgent_repeat_interval_ms=int(max(0.2, self._cfg.warning_cooldown_s) * 1000)
        )
        self._route_provider = route_provider
        self._user_provider = user_provider
        self._motion_provider = motion_provider
        self._on_frame_decision = on_frame_decision

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
        
        self._motion_fall_detector = MotionFallDetector()
        self._smv_fall_detector = SignalMagnitudeFallDetector()

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
        surfaces = self._surface_observations(frame, w, h) if self._cfg.enable_surface_heuristic else []
        ctx = FrameContext(
            timestamp_ms=now_ms,
            frame_id=str(time.time()),
            detections=detections,
            warnings=warnings,
            surfaces=surfaces,
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
        if self._on_frame_decision is not None:
            try:
                self._on_frame_decision(ctx, decision)
            except Exception as exc:
                print(f"[vision] decision logger failed: {exc}")
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
            self._augment_signal_attributes(frame, det)
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

    def _augment_signal_attributes(self, frame: np.ndarray, det: Detection) -> None:
        label = det.label.lower()
        if label in {"traffic light", "traffic signal"}:
            state, confidence, scores = self._estimate_traffic_light_state(frame, det)
            det.attributes["signal_category"] = "traffic_light"
            det.attributes["signal_state"] = state
            det.attributes["signal_state_confidence"] = confidence
            det.attributes["traffic_light_color_scores"] = scores
        elif label == "stop sign":
            det.attributes["signal_category"] = "stop_sign"
            det.attributes["signal_state"] = "stop_sign"
        elif label in SIGNAL_CLASSES:
            det.attributes["signal_category"] = "pedestrian_signal"
            det.attributes["signal_state"] = _signal_state_from_label(label)

    def _surface_observations(self, frame: np.ndarray, w: int, h: int) -> List[SurfaceObservation]:
        if w <= 0 or h <= 0:
            return []

        y1 = int(h * 0.58)
        y2 = h
        if y2 - y1 < 16:
            return []

        bands = [
            (Direction.LEFT, 0.0, 0.36),
            (Direction.CENTER, 0.32, 0.68),
            (Direction.RIGHT, 0.64, 1.0),
        ]
        observations: List[SurfaceObservation] = []
        for direction, x_start, x_end in bands:
            x1 = int(w * x_start)
            x2 = int(w * x_end)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            obs = self._classify_surface_crop(crop, direction)
            if obs is not None:
                observations.append(obs)

        curb = self._estimate_curb_edge(frame, w, h)
        if curb is not None:
            observations.append(curb)
        return observations

    def _classify_surface_crop(self, crop: np.ndarray, direction: Direction) -> Optional[SurfaceObservation]:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        low_sat = sat < 55
        gray_ratio = float(np.mean(low_sat))
        bright_ratio = float(np.mean(low_sat & (val >= 105)))
        dark_ratio = float(np.mean(low_sat & (val < 105)))
        near_field_ratio = max(bright_ratio, dark_ratio)

        if near_field_ratio < 0.32:
            return None

        if dark_ratio > bright_ratio * 1.25:
            kind = SurfaceKind.ROAD
            confidence = min(0.72, 0.38 + dark_ratio * 0.45 + gray_ratio * 0.10)
        else:
            kind = SurfaceKind.SIDEWALK
            confidence = min(0.68, 0.34 + bright_ratio * 0.42 + gray_ratio * 0.10)

        if confidence < 0.42:
            return None

        return SurfaceObservation(
            kind=kind,
            confidence=confidence,
            direction=direction,
            near_field_ratio=near_field_ratio,
            distance_m=0.8 if direction == Direction.CENTER else 1.2,
            source="vision-surface-heuristic",
            attributes={
                "gray_ratio": round(gray_ratio, 3),
                "bright_gray_ratio": round(bright_ratio, 3),
                "dark_gray_ratio": round(dark_ratio, 3),
                "note": "color/texture heuristic; use segmentation+depth for deployment",
            },
        )

    @staticmethod
    def _estimate_curb_edge(frame: np.ndarray, w: int, h: int) -> Optional[SurfaceObservation]:
        y1 = int(h * 0.48)
        y2 = int(h * 0.84)
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(18, crop.shape[1] // 8),
            minLineLength=max(24, crop.shape[1] // 4),
            maxLineGap=12,
        )
        if lines is None:
            return None

        best_len = 0.0
        best_y = None
        for line in lines[:, 0]:
            lx1, ly1, lx2, ly2 = [float(v) for v in line]
            dx = lx2 - lx1
            dy = ly2 - ly1
            if abs(dx) < 1.0:
                continue
            slope = abs(dy / dx)
            if slope > 0.18:
                continue
            length = float((dx * dx + dy * dy) ** 0.5)
            if length > best_len:
                best_len = length
                best_y = (ly1 + ly2) / 2.0

        if best_y is None:
            return None

        length_ratio = best_len / max(1.0, crop.shape[1])
        if length_ratio < 0.35:
            return None
        y_ratio = best_y / max(1.0, crop.shape[0])
        distance_m = 0.6 + y_ratio * 1.2
        return SurfaceObservation(
            kind=SurfaceKind.CURB,
            confidence=min(0.78, 0.40 + length_ratio * 0.40),
            direction=Direction.CENTER,
            near_field_ratio=min(1.0, length_ratio),
            distance_m=distance_m,
            source="vision-curb-edge-heuristic",
            attributes={"horizontal_edge_ratio": round(length_ratio, 3), "edge_y_ratio": round(y_ratio, 3)},
        )

    @staticmethod
    def _estimate_traffic_light_state(frame: np.ndarray, det: Detection) -> tuple[str, float, Dict[str, int]]:
        if det.bbox is None:
            return "unknown", 0.0, {}

        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(det.bbox.x1)))
        y1 = max(0, min(h - 1, int(det.bbox.y1)))
        x2 = max(x1 + 1, min(w, int(det.bbox.x2)))
        y2 = max(y1 + 1, min(h, int(det.bbox.y2)))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown", 0.0, {}

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, np.array([0, 70, 80]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 70, 80]), np.array([180, 255, 255]))
        yellow = cv2.inRange(hsv, np.array([15, 70, 80]), np.array([35, 255, 255]))
        green = cv2.inRange(hsv, np.array([40, 50, 60]), np.array([95, 255, 255]))

        scores = {
            "red": int(cv2.countNonZero(red1) + cv2.countNonZero(red2)),
            "yellow": int(cv2.countNonZero(yellow)),
            "green": int(cv2.countNonZero(green)),
        }
        total = max(1, int(crop.shape[0] * crop.shape[1]))
        state, score = max(scores.items(), key=lambda item: item[1])
        min_pixels = max(8, int(total * 0.01))
        if score < min_pixels:
            return "unknown", 0.0, scores
        return state, min(1.0, score / max(min_pixels * 4, 1)), scores

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
