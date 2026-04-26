"""
Webcam capture + YOLOv8 + object-distance cues + agentic routing.

This replaces the old direct warning callback. The flow is now:

    YOLO detections -> structured Detection/WarningEvent -> AgenticNavigationRouter -> speech/haptic decision

The existing warning behavior is preserved, but it becomes one input to the router
instead of immediately speaking from inside the vision loop.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
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
    INDOOR_LOCATION_TYPES,
    MotionState,
    OUTDOOR_LOCATION_TYPES,
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
    "dining table",
    "couch",
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
    "dining table",
    "couch",
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


def _model_label(names: object, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, names.get(str(cls_id), cls_id))).lower().strip()
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id]).lower().strip()
    return str(cls_id)


def _has_nearby_door_part_context(handle_row: np.ndarray, context_rows: List[np.ndarray]) -> bool:
    if not context_rows:
        return False
    hx1, hy1, hx2, hy2 = [float(v) for v in handle_row[:4]]
    hcx = (hx1 + hx2) / 2.0
    hcy = (hy1 + hy2) / 2.0
    handle_w = max(1.0, hx2 - hx1)
    handle_h = max(1.0, hy2 - hy1)
    for row in context_rows:
        x1, y1, x2, y2 = [float(v) for v in row[:4]]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        part_w = max(1.0, x2 - x1)
        part_h = max(1.0, y2 - y1)
        if x1 <= hcx <= x2 and y1 <= hcy <= y2:
            return True
        if abs(cx - hcx) <= max(handle_w * 5.0, part_w * 0.65) and abs(cy - hcy) <= max(handle_h * 6.0, part_h * 0.65):
            return True
    return False


@dataclass
class VisionConfig:
    """Tunable thresholds for warnings and YOLO inference."""

    model_path: str = "yolov8n.pt"
    door_parts_model_path: Optional[str] = None
    door_parts_yolov5_dir: Optional[str] = None
    door_parts_conf: float = 0.40
    door_parts_imgsz: int = 416
    obstacle_area_ratio: float = 0.12
    person_center_radius: float = 0.18
    warning_cooldown_s: float = 2.5
    confirm_frames: int = 2
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 416
    half: Optional[bool] = None
    augment: bool = False
    max_det: int = 50
    target_fps: float = 6.0
    show_preview: bool = False

    # Demo defaults. On a phone these should come from IMU/GPS/user state.
    assume_moving: bool = True
    location_type: str = "indoor"  # Default to indoor to prevent outdoor hallucinations
    visual_confidence: float = 0.75

    # Calibration multiplier for the monocular distance heuristic. Increase if
    # distances are consistently too short; decrease if they are too long.
    distance_scale: float = 1.0

    # Camera mount: "head" for glasses/head-mounted, "hand" for phone-in-hand.
    # Head-mounted shifts the ground plane lower in the frame and adjusts
    # distance/area thresholds for the higher vantage point (~1.6m vs ~1.0m).
    camera_mount: str = "hand"

    # Universal proximity warning: ANY object within this distance triggers
    # a stop warning regardless of category. Higher values give more reaction time
    # but increase false positives. Head-mounted may need adjustment due to
    # distance estimation scaling.
    universal_proximity_distance_m: float = 1.0

    # Lightweight sidewalk/road/curb heuristic. This is intentionally
    # conservative; production sidewalk navigation should replace or supplement
    # it with segmentation + depth.
    enable_surface_heuristic: bool = True
    surface_every_n_frames: int = 3

    @property
    def is_head_mounted(self) -> bool:
        return self.camera_mount == "head"

    @property
    def effective_obstacle_area_ratio(self) -> float:
        # From head height, ground-level objects (chairs, tables) have smaller
        # bounding boxes. Lower the threshold so they still trigger warnings.
        return self.obstacle_area_ratio * 0.55 if self.is_head_mounted else self.obstacle_area_ratio

    @property
    def effective_distance_scale(self) -> float:
        # Head-mounted camera is ~1.6m up vs ~1.0m for phone-in-hand.
        # Objects appear farther (smaller bbox), so scale distances down.
        return self.distance_scale * 0.72 if self.is_head_mounted else self.distance_scale

    @property
    def surface_y_start(self) -> float:
        # Phone: ground starts at ~58% of frame height.
        # Head:  ground starts at ~72% (looking more forward, ground is lower).
        return 0.72 if self.is_head_mounted else 0.58

    @property
    def curb_y_range(self) -> tuple[float, float]:
        # Phone: curb scan 48%-84% of frame.
        # Head:  curb scan 62%-92% (shifted down).
        return (0.62, 0.92) if self.is_head_mounted else (0.48, 0.84)


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
        on_decision: Optional[Callable[[AgentDecision], None]] = None,
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
        self._door_parts_model = self._load_door_parts_model()

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
        self._consec_door_hits = 0
        self._door_candidate_key: Optional[str] = None
        self._last_spoken: Optional[str] = None
        self._last_frame_context: Optional[FrameContext] = None
        self._processed_frames = 0
        self._cached_surfaces: List[SurfaceObservation] = []
        self._motion_fall_detector = MotionFallDetector()
        self._smv_fall_detector = SignalMagnitudeFallDetector()

        use_half = self._cfg.half if self._cfg.half is not None else bool(torch.cuda.is_available())
        self._predict_half = bool(use_half and torch.cuda.is_available())

        print(
            f"[vision] Inference: imgsz={self._cfg.imgsz} conf={self._cfg.conf} "
            f"iou={self._cfg.iou} tracked_classes={len(self._active_classes) or 'all'} "
            f"half={self._predict_half} augment={self._cfg.augment} "
            f"confirm_frames={self._cfg.confirm_frames} target_fps={self._cfg.target_fps} "
            f"preview={self._cfg.show_preview}"
        )
        if self._door_parts_model is not None:
            names = getattr(self._door_parts_model, "names", {})
            model_path = getattr(self, "_door_parts_model_path", self._cfg.door_parts_model_path)
            print(f"[vision] Door-parts model active: {model_path} names={names}")

    def active_class_labels(self) -> Set[str]:
        return set(self._active_classes)

    def _load_door_parts_model(self):
        """Optionally load a local YOLOv5 door-parts model.

        Joechencc/Door_detection provides YOLOv5 weights with an explicit
        "handle" class. Those weights are not YOLOv8-forward-compatible, so we
        only load them through a user-provided local YOLOv5 checkout.
        """
        raw_model_path = (self._cfg.door_parts_model_path or os.environ.get("DOOR_PARTS_MODEL_PATH") or "").strip()
        if not raw_model_path:
            return None

        model_path = Path(raw_model_path).expanduser()
        if not model_path.exists():
            print(f"[vision] Door-parts model not found: {model_path}")
            return None

        raw_yolov5_dir = (self._cfg.door_parts_yolov5_dir or os.environ.get("DOOR_PARTS_YOLOV5_DIR") or "").strip()
        if not raw_yolov5_dir:
            print("[vision] DOOR_PARTS_YOLOV5_DIR is required for YOLOv5 door-parts weights.")
            return None

        yolov5_dir = Path(raw_yolov5_dir).expanduser()
        if not (yolov5_dir / "hubconf.py").exists():
            print(f"[vision] YOLOv5 hubconf.py not found in: {yolov5_dir}")
            return None

        try:
            if str(yolov5_dir) not in sys.path:
                sys.path.insert(0, str(yolov5_dir))
            original_torch_load = torch.load

            def _trusted_checkpoint_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                try:
                    return original_torch_load(*args, **kwargs)
                except TypeError as exc:
                    if "weights_only" not in str(exc):
                        raise
                    kwargs.pop("weights_only", None)
                    return original_torch_load(*args, **kwargs)

            # YOLOv5 checkpoints are pickle-based. Reaching this code already
            # requires an explicit local YOLOv5 checkout and local weights path.
            torch.load = _trusted_checkpoint_load
            try:
                model = torch.hub.load(
                    str(yolov5_dir),
                    "custom",
                    path_or_model=str(model_path),
                    source="local",
                    autoshape=True,
                    verbose=False,
                )
            finally:
                torch.load = original_torch_load
            model.conf = float(self._cfg.door_parts_conf)
            model.iou = float(self._cfg.iou)
            model.max_det = int(self._cfg.max_det)
            self._door_parts_model_path = str(model_path)
            return model
        except Exception as exc:
            print(f"[vision] Door-parts model disabled: {exc}")
            return None

    def run_forever(self, camera_index: int = 0, stop_event: Optional[threading.Event] = None) -> None:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self._cfg.show_preview:
            print("[vision] Webcam open. Press 'q' in the preview window or Ctrl+C in the terminal to quit.")
        else:
            print("[vision] Webcam open. Preview disabled for lower lag; rerun with --preview to show the window.")

        frame_count = 0
        last_fps_time = time.time()
        fps_frames = 0
        last_health_check = time.time()

        try:
            while stop_event is None or not stop_event.is_set():
                loop_started = time.time()
                try:
                    # Flush buffer to ensure real-time frame
                    for _ in range(2):
                        cap.grab()
                    
                    ret, frame = cap.read()
                    if not ret:
                        print("[vision] Failed to read frame from webcam")
                        time.sleep(0.1)
                        continue

                    w, h = frame.shape[1], frame.shape[0]
                    decision = self._process_frame(frame, w, h)
                    
                    if self._on_decision is not None and decision.should_speak and decision.message:
                        self._last_spoken = decision.message
                        if self._on_decision is not None:
                            self._on_decision(decision)

                    frame_count += 1
                    fps_frames += 1
                    now = time.time()
                    if now - last_fps_time >= 3.0:
                        fps = fps_frames / (now - last_fps_time)
                        print(f"[vision] ~{fps:.1f} FPS over last {now - last_fps_time:.1f}s")
                        fps_frames = 0
                        last_fps_time = now

                    # Health check every 10 seconds
                    if now - last_health_check >= 10.0:
                        print(f"[vision] Health check: processed {frame_count} frames, system stable")
                        last_health_check = now

                    if self._cfg.show_preview:
                        cv2.imshow("Assistive Nav — preview", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("[vision] Quit requested from preview window.")
                            break

                    if self._cfg.target_fps > 0:
                        min_interval = 1.0 / self._cfg.target_fps
                        elapsed = time.time() - loop_started
                        if elapsed < min_interval:
                            time.sleep(min_interval - elapsed)
                except Exception as e:
                    print(f"[vision] Frame processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue processing next frame
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("[vision] Interrupted by user")
        except Exception as e:
            print(f"[vision] CRITICAL ERROR in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[vision] Shutting down...")
            cap.release()
            if self._cfg.show_preview:
                cv2.destroyAllWindows()
            print(f"[vision] Processed {frame_count} frames total")
        return frame_count

    def _infer_location_type(self, detections: List[Detection]) -> str:
        """Use YOLO detections to infer indoor/outdoor when GPS is unavailable."""
        _INDOOR_INDICATOR_LABELS = {
            "chair", "couch", "bed", "dining table", "toilet", "sink",
            "refrigerator", "oven", "microwave", "toaster", "tv", "laptop",
            "mouse", "keyboard", "book", "vase", "potted plant", "clock",
        }
        _OUTDOOR_INDICATOR_LABELS = {
            "car", "truck", "bus", "motorcycle", "bicycle", "traffic light",
            "stop sign", "parking meter", "fire hydrant", "bench",
        }
        indoor_score = 0.0
        outdoor_score = 0.0
        for det in detections:
            lbl = det.label.lower()
            if lbl in _INDOOR_INDICATOR_LABELS:
                indoor_score += det.confidence
            elif lbl in _OUTDOOR_INDICATOR_LABELS:
                outdoor_score += det.confidence
        if indoor_score >= 0.8 and indoor_score > 2 * outdoor_score:
            return "indoor"
        if outdoor_score >= 1.0 and outdoor_score > 2 * indoor_score:
            return "outdoor"
        return self._cfg.location_type

    def _process_frame(self, frame: np.ndarray, w: int, h: int) -> AgentDecision:
        try:
            now_ms = int(time.time() * 1000)
            self._processed_frames += 1
            detections = self._detect(frame, w, h)
            location_type = self._infer_location_type(detections)
            warnings = self._warnings_from_detections(detections, now_ms)
            surfaces = self._surface_observations_for_frame(frame, w, h) if self._cfg.enable_surface_heuristic else []
            if location_type in INDOOR_LOCATION_TYPES:
                surfaces = [s for s in surfaces if s.kind not in {SurfaceKind.ROAD, SurfaceKind.SIDEWALK, SurfaceKind.CROSSWALK}]
            if self._cfg.show_preview:
                for surface in surfaces:
                    self._draw_surface_observation(frame, surface)
            ctx = FrameContext(
                timestamp_ms=now_ms,
                frame_id=str(time.time()),
                detections=detections,
                warnings=warnings,
                surfaces=surfaces,
                motion=self._motion_state(),
                route=self._route_state(),
                scene=SceneState(
                    location_type=location_type,
                    visual_confidence=self._cfg.visual_confidence,
                ),
                user=self._user_state(),
                last_spoken=self._last_spoken,
            )
            decision = self._router.decide(ctx)
            if self._cfg.show_preview:
                self._draw_decision(frame, decision)
            if self._on_frame_decision is not None:
                try:
                    spoken = self._on_frame_decision(ctx, decision)
                    if spoken and decision.message:
                        self._last_spoken = decision.message
                except Exception as e:
                    print(f"[vision] Frame decision callback error: {e}")
            self._last_frame_context = ctx
            return decision
        except Exception as e:
            print(f"[vision] CRITICAL: Frame processing failed: {e}")
            import traceback
            traceback.print_exc()
            self._last_frame_context = None
            # Return a safe fallback decision to prevent complete system failure
            from agentic_layer.models import AgentDecision, AgentAction, HapticPattern
            return AgentDecision(
                action=AgentAction.SILENT,
                priority=0,
                message="System error encountered",
                haptic=HapticPattern.NONE,
                agents_consulted=["vision_error"],
                debug={"error": str(e)},
            )

    def process_frame_numpy(
        self,
        frame_bgr: np.ndarray,
        route: Optional[RouteState] = None,
    ) -> AgentDecision:
        """
        Run a single YOLO + router pass on a BGR image (e.g. JPEG decoded from a phone).

        ``frame_bgr`` is copied internally so annotation overlays do not mutate the input.
        If ``route`` is given, it is used for this frame only (session destination from client).
        """
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a BGR uint8 array with shape HxWx3")
        h, w = frame_bgr.shape[:2]
        work = frame_bgr.copy()
        prev_provider = self._route_provider
        if route is not None:
            self._route_provider = lambda r=route: r
        try:
            return self._process_frame(work, w, h)
        finally:
            self._route_provider = prev_provider

    def _surface_observations_for_frame(self, frame: np.ndarray, w: int, h: int) -> List[SurfaceObservation]:
        n = max(1, int(self._cfg.surface_every_n_frames))
        route = self._route_state()
        scan_needs_surface = getattr(route, "mapping_state", "done") != "done" or getattr(route, "exit_seeking", False)
        if self._processed_frames == 1 or scan_needs_surface or self._processed_frames % n == 0:
            self._cached_surfaces = self._surface_observations(frame, w, h)
        return list(self._cached_surfaces)

    def handle_sensor_fall(self, reason: str, priority: int = 10) -> None:
        """Triggered by hardware sensors; starts the warning/alarm cycle."""
        print(f"[vision] Sensor-based fall sensed: {reason}")
        from agentic_layer import AgentAction
        decision = AgentDecision(
            should_speak=True,
            action=AgentAction.WARN,
            message=f"Fall detected. {reason}",
            priority=priority,
            haptic=True
        )
        self._on_decision(decision)

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
            if label in {"door", "door handle", "handle"}:
                # Door/handle guidance is intentionally disabled for mobile flow.
                continue
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
                distance_scale=self._cfg.effective_distance_scale,
                source="ultralytics-yolov8-distance",
            )
            self._augment_signal_attributes(frame, det)
            detections.append(det)
            if self._cfg.show_preview:
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
            large = area_ratio >= self._cfg.effective_obstacle_area_ratio

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
            immediate_contact = warning.distance_m is not None and warning.distance_m <= 0.8
            if immediate_contact or self._consec_warning_hits[key] >= n_confirm:
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

        y1 = int(h * self._cfg.surface_y_start)
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

        # Edge density detection for unknown obstacles (walls, pillars, boxes)
        try:
            edge_obstacle = self._detect_edge_density_obstacle(frame, w, h)
            if edge_obstacle is not None:
                observations.append(edge_obstacle)
        except Exception as e:
            # Log error but don't crash the entire system
            print(f"[vision] Edge density detection error: {e}")
            import traceback
            traceback.print_exc()

        # Optional trained door-parts detector with an explicit handle class.
        try:
            observations.extend(self._detect_door_parts(frame, w, h))
        except Exception as e:
            print(f"[vision] Door-parts detection error: {e}")
            import traceback
            traceback.print_exc()

        # Door detection using vertical edge patterns
        try:
            door = self._detect_door(frame, w, h)
            if door is not None:
                observations.append(door)
        except Exception as e:
            print(f"[vision] Door detection error: {e}")
            import traceback
            traceback.print_exc()

        # Wall plane detection: close wall/large vertical surface ahead.
        try:
            wall = self._detect_wall_plane(frame, w, h)
            if wall is not None:
                observations.append(wall)
        except Exception as e:
            print(f"[vision] Wall detection error: {e}")
            import traceback
            traceback.print_exc()

        # White object detection for low-contrast obstacles (white tables, boxes)
        try:
            white_obj = self._detect_white_object(frame, w, h)
            if white_obj is not None:
                observations.append(white_obj)
        except Exception as e:
            print(f"[vision] White object detection error: {e}")
            import traceback
            traceback.print_exc()

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

    def _detect_edge_density_obstacles(self, frame: np.ndarray, w: int, h: int) -> List[SurfaceObservation]:
        """Detect unknown obstacles via edge density in center and side regions.

        Catches walls, pillars, boxes, and other obstacles that YOLO might miss
        due to being partially visible at the frame edge or not in its training
        classes.
        """
        y0 = self._cfg.surface_y_start
        y1 = min(0.95, y0 + 0.20)  # 20% vertical slice from surface start
        y_start = int(h * y0)
        y_end = int(h * y1)

        bands = [
            (Direction.LEFT, 0.00, 0.22),
            (Direction.CENTER, 0.35, 0.65),
            (Direction.RIGHT, 0.78, 1.00),
        ]

        observations: List[SurfaceObservation] = []
        for direction, x0, x1 in bands:
            x_start = int(w * x0)
            x_end = int(w * x1)
            if y_end - y_start < 16 or x_end - x_start < 16:
                continue
            crop = frame[y_start:y_end, x_start:x_end]
            if crop.size == 0:
                continue
            obs = self._edge_density_observation_from_crop(crop, direction, y_start, y_end, h)
            if obs is not None:
                observations.append(obs)
        return observations

    def _detect_edge_density_obstacle(self, frame: np.ndarray, w: int, h: int) -> Optional[SurfaceObservation]:
        """Backward-compatible single-observation wrapper."""
        observations = self._detect_edge_density_obstacles(frame, w, h)
        if not observations:
            return None
        center = [obs for obs in observations if obs.direction == Direction.CENTER]
        return center[0] if center else observations[0]

    def _edge_density_observation_from_crop(
        self,
        crop: np.ndarray,
        direction: Direction,
        y_start: int,
        y_end: int,
        frame_height: int,
    ) -> Optional[SurfaceObservation]:
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Use lower thresholds for white-on-white (low contrast)
        edges = cv2.Canny(gray, 30, 80)

        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        if total_pixels == 0:
            return None
        edge_density = edge_pixels / total_pixels

        # Analyze edge orientations to distinguish vertical obstacles from ground texture
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(8, crop.shape[1] // 16),
            minLineLength=max(12, crop.shape[1] // 8),
            maxLineGap=8,
        )

        if lines is None:
            return None

        # Count vertical vs horizontal lines
        vertical_count = 0
        horizontal_count = 0
        total_line_length = 0

        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = np.sqrt(dx*dx + dy*dy)
            total_line_length += length

            if dx < 2:  # Nearly vertical
                vertical_count += 1
            elif dy < 2:  # Nearly horizontal
                horizontal_count += 1

        # Require significant edge density with vertical dominance
        base_density = 0.08 if self._cfg.is_head_mounted else 0.06
        min_density = base_density + (0.015 if direction in {Direction.LEFT, Direction.RIGHT} else 0.0)
        if edge_density < min_density:
            return None

        # Vertical lines should dominate for obstacles (walls, pillars)
        if vertical_count <= max(2, horizontal_count):
            return None

        # Estimate distance based on vertical position in frame
        # Lower in frame = closer
        y_center = (y_start + y_end) / 2 / frame_height
        if y_center > 0.85:
            estimated_distance = 0.6  # Very close
        elif y_center > 0.75:
            estimated_distance = 1.0  # Close
        else:
            estimated_distance = 1.5  # Near

        # Confidence based on edge density and vertical dominance
        confidence = min(0.75, edge_density * 2 + (vertical_count / max(1, len(lines))) * 0.3)

        return SurfaceObservation(
            kind=SurfaceKind.OBSTACLE_EDGE,
            confidence=confidence,
            direction=direction,
            near_field_ratio=edge_density,
            distance_m=estimated_distance,
            source="vision-edge-density",
            attributes={
                "edge_density": round(edge_density, 3),
                "vertical_lines": vertical_count,
                "horizontal_lines": horizontal_count,
                "total_lines": len(lines),
                "frame_edge": direction.value if direction in {Direction.LEFT, Direction.RIGHT} else None,
                "partial_visibility": "frame_edge" if direction in {Direction.LEFT, Direction.RIGHT} else "unknown",
                "note": "edge-based obstacle detection for unknown or partially visible objects",
            },
        )

    def _detect_door(self, frame: np.ndarray, w: int, h: int) -> Optional[SurfaceObservation]:
        """Door and handle detection are disabled for mobile/web flow."""
        return None

        This is deliberately conservative. Handle-like lines are not enough on
        their own; they must appear inside a door-frame context and persist for
        several calls before a possible doorway surface is returned.
        """
        handle = self._detect_door_handle(frame, w, h)
        frame_context = self._detect_door_frame_context(frame, w, h)
        wall_context = self._detect_wall_plane(frame, w, h)
        candidate: Optional[SurfaceObservation] = None

        if handle is not None and frame_context is not None:
            confidence = float(handle["confidence"])
            confidence = min(0.86, confidence + 0.08)

            attrs = dict(handle)
            attrs["has_frame"] = True
            attrs["door_frame"] = frame_context
            attrs["clear_handle"] = float(handle["confidence"]) >= 0.72
            attrs["note"] = "possible doorway with handle-like feature; verify by touch"

            candidate = SurfaceObservation(
                kind=SurfaceKind.DOOR,
                confidence=confidence,
                direction=self._direction_from_x_ratio(float(handle["handle_x_ratio"])),
                near_field_ratio=float(handle.get("near_field_ratio", 0.0)),
                distance_m=float(handle["distance_m"]),
                source="vision-door-handle",
                attributes=attrs,
            )

        elif handle is not None and wall_context is not None and wall_context.confidence >= 0.62:
            confidence = min(0.78, float(handle["confidence"]) * 0.55 + wall_context.confidence * 0.45)
            attrs = dict(handle)
            attrs["has_frame"] = False
            attrs["wall_context"] = wall_context.model_dump()
            attrs["clear_handle"] = float(handle["confidence"]) >= 0.74 and wall_context.confidence >= 0.66
            attrs["note"] = "wall-like surface with handle-like feature; possible door, verify by touch"

            candidate = SurfaceObservation(
                kind=SurfaceKind.DOOR,
                confidence=confidence,
                direction=self._direction_from_x_ratio(float(handle["handle_x_ratio"])),
                near_field_ratio=max(float(handle.get("near_field_ratio", 0.0)), wall_context.near_field_ratio),
                distance_m=min(float(handle["distance_m"]), wall_context.distance_m or float(handle["distance_m"])),
                source="vision-wall-handle-candidate",
                attributes=attrs,
            )

        elif frame_context is not None and float(frame_context["confidence"]) >= 0.72:
            candidate = SurfaceObservation(
                kind=SurfaceKind.DOOR,
                confidence=float(frame_context["confidence"]),
                direction=Direction.CENTER,
                near_field_ratio=float(frame_context["near_field_ratio"]),
                distance_m=float(frame_context["distance_m"]),
                source="vision-door-frame",
                attributes={
                    **frame_context,
                    "handle_detected": False,
                    "handle_side": "unknown",
                    "recommended_hand": "unknown",
                    "handle_action": "confirm the doorway by touch before using it",
                    "note": "possible door frame; verify by touch",
                },
            )

        if candidate is None:
            self._consec_door_hits = 0
            self._door_candidate_key = None
            return None

        key = f"{candidate.source}:{candidate.direction.value}:{round(candidate.distance_m or 0.0, 1)}"
        if getattr(self, "_door_candidate_key", None) != key:
            self._door_candidate_key = key
            self._consec_door_hits = 0
        self._consec_door_hits = getattr(self, "_consec_door_hits", 0) + 1
        required_hits = 2 if bool(candidate.attributes.get("clear_handle")) and candidate.confidence >= 0.82 else 3
        if self._consec_door_hits < required_hits:
            return None
        return candidate

    def _detect_door_parts(self, frame: np.ndarray, w: int, h: int) -> List[SurfaceObservation]:
        if self._door_parts_model is None:
            return []

        model_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._door_parts_model(model_frame, size=int(self._cfg.door_parts_imgsz))
        if not hasattr(results, "xyxy") or not results.xyxy:
            return []

        raw_rows = results.xyxy[0]
        if hasattr(raw_rows, "detach"):
            rows = raw_rows.detach().cpu().numpy()
        else:
            rows = np.asarray(raw_rows)
        if rows.size == 0:
            return []

        names = getattr(self._door_parts_model, "names", {})
        context_rows = []
        handle_rows = []
        for row in rows:
            if len(row) < 6:
                continue
            cls_id = int(row[5])
            label = _model_label(names, cls_id)
            confidence = float(row[4])
            if confidence < float(self._cfg.door_parts_conf):
                continue
            if label != "handle":
                if label.startswith("door") or label.startswith("frame") or label == "hinge":
                    context_rows.append(row)
                continue
            handle_rows.append(row)

        observations: List[SurfaceObservation] = []
        for row in handle_rows:
            confidence = float(row[4])
            x1, y1, x2, y2 = [int(round(float(v))) for v in row[:4]]
            x1 = max(0, min(w - 1, x1))
            x2 = max(x1 + 1, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(y1 + 1, min(h, y2))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            has_context = _has_nearby_door_part_context(row, context_rows)
            handle = self._door_handle_attributes(
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                frame_width=w,
                frame_height=h,
                orientation="model_handle",
                support_side="model",
                near_field_ratio=min(1.0, max((x2 - x1) / max(1.0, w * 0.12), (y2 - y1) / max(1.0, h * 0.14))),
            )
            handle["clear_handle"] = True
            handle["has_frame"] = has_context
            handle["model_label"] = "handle"
            handle["model_source"] = "Joechencc/Door_detection"
            handle["detector_confidence_threshold"] = round(float(self._cfg.door_parts_conf), 3)

            observations.append(
                SurfaceObservation(
                    kind=SurfaceKind.DOOR,
                    confidence=min(0.96, confidence + (0.08 if has_context else 0.03)),
                    direction=self._direction_from_x_ratio(float(handle["handle_x_ratio"])),
                    near_field_ratio=float(handle.get("near_field_ratio", 0.0)),
                    distance_m=float(handle["distance_m"]),
                    source="joechencc-door-parts-handle",
                    attributes=handle,
                )
            )

        return observations

    def _detect_wall_plane(self, frame: np.ndarray, w: int, h: int) -> Optional[SurfaceObservation]:
        """Detect a close wall-like vertical plane in the walking path."""
        if w <= 0 or h <= 0:
            return None

        y_start = int(h * 0.28)
        y_end = int(h * 0.92)
        x_start = int(w * 0.25)
        x_end = int(w * 0.75)
        if y_end - y_start < 80 or x_end - x_start < 80:
            return None

        crop = frame[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        low_sat_ratio = float(np.mean(sat < 65))
        value_std = float(np.std(val))

        edges = cv2.Canny(gray, 30, 90)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(18, crop.shape[1] // 10),
            minLineLength=max(32, crop.shape[0] // 5),
            maxLineGap=14,
        )
        if lines is None:
            return None

        vertical_count = 0
        horizontal_count = 0
        vertical_xs = []
        for line in lines[:, 0]:
            lx1, ly1, lx2, ly2 = line
            dx = abs(lx2 - lx1)
            dy = abs(ly2 - ly1)
            if dy >= max(24, dx * 3.0):
                vertical_count += 1
                vertical_xs.append((lx1 + lx2) / 2.0)
            elif dx >= max(24, dy * 3.0):
                horizontal_count += 1

        if vertical_count < 2:
            return None

        x_spread = (max(vertical_xs) - min(vertical_xs)) / max(1.0, float(crop.shape[1])) if vertical_xs else 0.0
        edge_density = float(np.mean(edges > 0))
        wall_texture = low_sat_ratio >= 0.35 and value_std <= 65.0
        vertical_dominance = vertical_count >= max(2, horizontal_count)
        if not (wall_texture and vertical_dominance and x_spread >= 0.28 and edge_density >= 0.025):
            return None

        lower_half_edges = float(np.mean(edges[crop.shape[0] // 2 :, :] > 0))
        if lower_half_edges >= 0.055:
            distance_m = 0.8
        elif lower_half_edges >= 0.035:
            distance_m = 1.1
        else:
            distance_m = 1.5

        confidence = min(
            0.84,
            0.38
            + min(0.22, low_sat_ratio * 0.18)
            + min(0.18, edge_density * 2.0)
            + min(0.16, x_spread * 0.20)
            + min(0.10, vertical_count * 0.015),
        )
        if confidence < 0.58:
            return None

        return SurfaceObservation(
            kind=SurfaceKind.WALL,
            confidence=confidence,
            direction=Direction.CENTER,
            near_field_ratio=edge_density,
            distance_m=distance_m,
            source="vision-wall-plane",
            attributes={
                "low_saturation_ratio": round(low_sat_ratio, 3),
                "value_std": round(value_std, 2),
                "edge_density": round(edge_density, 3),
                "lower_half_edge_density": round(lower_half_edges, 3),
                "vertical_lines": vertical_count,
                "horizontal_lines": horizontal_count,
                "vertical_spread": round(float(x_spread), 3),
                "note": "wall-like vertical plane ahead; verify with depth sensor for deployment",
            },
        )

    def _detect_door_handle(self, frame: np.ndarray, w: int, h: int) -> Optional[Dict[str, object]]:
        y0 = max(0.18, self._cfg.surface_y_start - 0.38)
        y1 = 0.94
        x0, x1 = 0.04, 0.96

        y_start = int(h * y0)
        y_end = int(h * y1)
        x_start = int(w * x0)
        x_end = int(w * x1)
        if y_end - y_start < 48 or x_end - x_start < 64:
            return None

        crop = frame[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 45, 140)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(18, crop.shape[1] // 28),
            minLineLength=max(24, crop.shape[1] // 18),
            maxLineGap=14,
        )
        if lines is None:
            horizontal = self._detect_horizontal_lever_handle(frame, w, h, crop, x_start, y_start)
            if horizontal is not None:
                return horizontal
            vertical = self._detect_vertical_pull_handle(frame, w, h, crop, x_start, y_start)
            if vertical is not None:
                return vertical
            return self._detect_round_knob_handle(frame, w, h, crop, x_start, y_start)

        global_lines = []
        for raw in lines[:, 0]:
            lx1, ly1, lx2, ly2 = [float(v) for v in raw]
            global_lines.append((lx1 + x_start, ly1 + y_start, lx2 + x_start, ly2 + y_start))

        best: Optional[Dict[str, object]] = None
        best_score = 0.0
        min_len = max(32.0, w * 0.035)
        max_len = max(min_len + 1.0, w * 0.30)

        for line in global_lines:
            lx1, ly1, lx2, ly2 = line
            dx = lx2 - lx1
            dy = ly2 - ly1
            length = float((dx * dx + dy * dy) ** 0.5)
            if length < min_len or length > max_len:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            angle = min(angle, 180.0 - angle)
            if angle > 18.0:
                continue

            cx = (lx1 + lx2) / 2.0
            cy = (ly1 + ly2) / 2.0
            x_ratio = cx / max(1.0, float(w))
            y_ratio = cy / max(1.0, float(h))
            if not (0.28 <= y_ratio <= 0.92):
                continue

            color_score = self._door_handle_color_score(frame, line)
            support_score, support_side = self._door_handle_support_score(global_lines, line, w, h)
            length_score = min(1.0, length / max(1.0, w * 0.14))
            mid_height_bonus = 0.10 if 0.45 <= y_ratio <= 0.82 else 0.0
            score = 0.22 + length_score * 0.26 + color_score * 0.24 + support_score * 0.24 + mid_height_bonus
            if score < 0.52:
                continue

            x_min = max(0, int(min(lx1, lx2) - w * 0.025))
            x_max = min(w, int(max(lx1, lx2) + w * 0.025))
            y_min = max(0, int(min(ly1, ly2) - h * 0.035))
            y_max = min(h, int(max(ly1, ly2) + h * 0.035))
            if y_max <= y_min or x_max <= x_min:
                continue

            handle = self._door_handle_attributes(
                confidence=min(0.90, score),
                bbox=(x_min, y_min, x_max, y_max),
                center=(cx, cy),
                frame_width=w,
                frame_height=h,
                orientation="lever_horizontal",
                support_side=support_side,
                near_field_ratio=min(1.0, length / max(1.0, w * 0.22)),
            )
            if score > best_score:
                best = handle
                best_score = score

        if best is not None:
            return best
        horizontal = self._detect_horizontal_lever_handle(frame, w, h, crop, x_start, y_start)
        if horizontal is not None:
            return horizontal
        vertical = self._detect_vertical_pull_handle(frame, w, h, crop, x_start, y_start)
        if vertical is not None:
            return vertical
        return self._detect_round_knob_handle(frame, w, h, crop, x_start, y_start)

    def _detect_horizontal_lever_handle(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        crop: np.ndarray,
        x_start: int,
        y_start: int,
    ) -> Optional[Dict[str, object]]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 35, 115)
        bright_low_sat = ((hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 120)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(bright_low_sat, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[Dict[str, object]] = None
        best_score = 0.0
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if cw <= 0 or ch <= 0:
                continue
            aspect = cw / max(1.0, float(ch))
            width_ratio = cw / max(1.0, float(w))
            height_ratio = ch / max(1.0, float(h))
            if aspect < 2.4 or not (0.035 <= width_ratio <= 0.28) or not (0.006 <= height_ratio <= 0.09):
                continue

            gx1, gy1 = x + x_start, y + y_start
            gx2, gy2 = gx1 + cw, gy1 + ch
            cx = (gx1 + gx2) / 2.0
            cy = (gy1 + gy2) / 2.0
            y_ratio = cy / max(1.0, float(h))
            if not (0.34 <= y_ratio <= 0.88):
                continue

            roi = frame[max(0, gy1):min(h, gy2), max(0, gx1):min(w, gx2)]
            if roi.size == 0:
                continue
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            low_sat = float(np.mean(roi_hsv[:, :, 1] < 110))
            useful_value = float(np.mean(roi_hsv[:, :, 2] > 105))
            contour_fill = cv2.contourArea(contour) / max(1.0, float(cw * ch))
            score = min(0.86, 0.24 + min(1.0, aspect / 8.0) * 0.22 + low_sat * 0.18 + useful_value * 0.12 + min(1.0, contour_fill) * 0.18)
            if score < 0.53:
                continue

            handle = self._door_handle_attributes(
                confidence=score,
                bbox=(gx1, gy1, gx2, gy2),
                center=(cx, cy),
                frame_width=w,
                frame_height=h,
                orientation="lever_horizontal",
                support_side="unknown",
                near_field_ratio=min(1.0, width_ratio / 0.20),
            )
            if score > best_score:
                best = handle
                best_score = score
        return best

    def _detect_vertical_pull_handle(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        crop: np.ndarray,
        x_start: int,
        y_start: int,
    ) -> Optional[Dict[str, object]]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[Dict[str, object]] = None
        best_score = 0.0
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if cw <= 0 or ch <= 0:
                continue
            aspect = ch / max(1.0, float(cw))
            height_ratio = ch / max(1.0, float(h))
            width_ratio = cw / max(1.0, float(w))
            if aspect < 2.2 or not (0.05 <= height_ratio <= 0.34) or not (0.008 <= width_ratio <= 0.08):
                continue

            gx1, gy1 = x + x_start, y + y_start
            gx2, gy2 = gx1 + cw, gy1 + ch
            cx = (gx1 + gx2) / 2.0
            cy = (gy1 + gy2) / 2.0
            if not (0.25 <= cy / max(1.0, h) <= 0.92):
                continue

            roi = frame[max(0, gy1):min(h, gy2), max(0, gx1):min(w, gx2)]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            low_sat = float(np.mean(hsv[:, :, 1] < 85))
            contrast = float(np.std(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))) / 64.0
            score = min(0.82, 0.28 + min(1.0, aspect / 5.0) * 0.25 + low_sat * 0.20 + min(1.0, contrast) * 0.20)
            if score < 0.54:
                continue

            handle = self._door_handle_attributes(
                confidence=score,
                bbox=(gx1, gy1, gx2, gy2),
                center=(cx, cy),
                frame_width=w,
                frame_height=h,
                orientation="vertical_pull",
                support_side="unknown",
                near_field_ratio=min(1.0, height_ratio / 0.22),
            )
            if score > best_score:
                best = handle
                best_score = score
        return best

    def _detect_round_knob_handle(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        crop: np.ndarray,
        x_start: int,
        y_start: int,
    ) -> Optional[Dict[str, object]]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        edges = cv2.Canny(gray, 45, 135)
        low_sat = hsv[:, :, 1] < 115
        visible_value = (hsv[:, :, 2] > 65) & (hsv[:, :, 2] < 250)
        material_mask = (low_sat & visible_value).astype(np.uint8) * 255
        material_mask = cv2.bitwise_and(material_mask, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
        material_mask = cv2.morphologyEx(material_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(material_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[Dict[str, object]] = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            if cw <= 0 or ch <= 0:
                continue
            width_ratio = cw / max(1.0, float(w))
            height_ratio = ch / max(1.0, float(h))
            if not (0.018 <= width_ratio <= 0.11 and 0.018 <= height_ratio <= 0.13):
                continue

            aspect = cw / max(1.0, float(ch))
            if not (0.62 <= aspect <= 1.45):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < 0.48:
                continue

            gx1, gy1 = x + x_start, y + y_start
            gx2, gy2 = gx1 + cw, gy1 + ch
            cx = (gx1 + gx2) / 2.0
            cy = (gy1 + gy2) / 2.0
            x_ratio = cx / max(1.0, float(w))
            y_ratio = cy / max(1.0, float(h))
            if not (0.34 <= y_ratio <= 0.86):
                continue

            roi = frame[max(0, gy1):min(h, gy2), max(0, gx1):min(w, gx2)]
            if roi.size == 0:
                continue
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            contrast = min(1.0, float(np.std(roi_gray)) / 45.0)
            metal_like = float(np.mean(roi_hsv[:, :, 1] < 115))
            fill_ratio = min(1.0, area / max(1.0, float(cw * ch)))
            size_score = min(1.0, max(width_ratio / 0.06, height_ratio / 0.08))
            score = min(0.88, 0.24 + circularity * 0.24 + contrast * 0.18 + metal_like * 0.12 + fill_ratio * 0.14 + size_score * 0.10)
            if score < 0.58:
                continue

            handle = self._door_handle_attributes(
                confidence=score,
                bbox=(gx1, gy1, gx2, gy2),
                center=(cx, cy),
                frame_width=w,
                frame_height=h,
                orientation="round_knob",
                support_side="unknown",
                near_field_ratio=min(1.0, max(width_ratio / 0.08, height_ratio / 0.09)),
            )
            if score > best_score:
                best = handle
                best_score = score
        return best

    @staticmethod
    def _door_handle_color_score(frame: np.ndarray, line: tuple[float, float, float, float]) -> float:
        h, w = frame.shape[:2]
        lx1, ly1, lx2, ly2 = line
        pad_x = max(6, int(w * 0.012))
        pad_y = max(6, int(h * 0.018))
        x1 = max(0, int(min(lx1, lx2) - pad_x))
        x2 = min(w, int(max(lx1, lx2) + pad_x))
        y1 = max(0, int(min(ly1, ly2) - pad_y))
        y2 = min(h, int(max(ly1, ly2) + pad_y))
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        low_sat = float(np.mean(hsv[:, :, 1] < 95))
        usable_value = float(np.mean((hsv[:, :, 2] > 75) & (hsv[:, :, 2] < 255)))
        contrast = min(1.0, float(np.std(gray)) / 56.0)
        return min(1.0, low_sat * 0.45 + usable_value * 0.25 + contrast * 0.30)

    @staticmethod
    def _door_handle_support_score(
        lines: List[tuple[float, float, float, float]],
        handle_line: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, str]:
        hx1, hy1, hx2, hy2 = handle_line
        left_x = min(hx1, hx2)
        right_x = max(hx1, hx2)
        cy = (hy1 + hy2) / 2.0
        support_window_x = max(18.0, frame_width * 0.045)
        support_window_y = max(32.0, frame_height * 0.12)
        min_vertical_len = max(24.0, frame_height * 0.045)

        left_hits = 0
        right_hits = 0
        for lx1, ly1, lx2, ly2 in lines:
            dx = abs(lx2 - lx1)
            dy = abs(ly2 - ly1)
            if dy < min_vertical_len or dx > max(5.0, dy * 0.22):
                continue
            line_x = (lx1 + lx2) / 2.0
            line_y = (ly1 + ly2) / 2.0
            if abs(line_y - cy) > support_window_y:
                continue
            if abs(line_x - left_x) <= support_window_x:
                left_hits += 1
            if abs(line_x - right_x) <= support_window_x:
                right_hits += 1

        hits = left_hits + right_hits
        support_side = "left" if left_hits > right_hits else "right" if right_hits > left_hits else "unknown"
        return min(1.0, hits / 3.0), support_side

    @staticmethod
    def _door_handle_attributes(
        confidence: float,
        bbox: tuple[int, int, int, int],
        center: tuple[float, float],
        frame_width: int,
        frame_height: int,
        orientation: str,
        support_side: str,
        near_field_ratio: float,
    ) -> Dict[str, object]:
        x1, y1, x2, y2 = bbox
        cx, cy = center
        x_ratio = cx / max(1.0, float(frame_width))
        y_ratio = cy / max(1.0, float(frame_height))
        side = "left" if x_ratio < 0.44 else "right" if x_ratio > 0.56 else "center"
        recommended_hand = "left" if x_ratio < 0.48 else "right" if x_ratio > 0.52 else "either hand"
        if y_ratio < 0.46:
            height_zone = "chest height"
        elif y_ratio <= 0.78:
            height_zone = "waist height"
        else:
            height_zone = "low"

        width_ratio = (x2 - x1) / max(1.0, float(frame_width))
        height_ratio = (y2 - y1) / max(1.0, float(frame_height))
        size_signal = max(width_ratio / 0.18, height_ratio / 0.24)
        if size_signal >= 0.8 or y_ratio >= 0.76:
            distance_m = 0.8
        elif size_signal >= 0.45 or y_ratio >= 0.60:
            distance_m = 1.2
        else:
            distance_m = 1.8

        if orientation == "lever_horizontal":
            handle_action = "press the lever down, then gently test whether the door pushes or pulls"
        elif orientation == "round_knob":
            handle_action = "turn the knob, then gently test whether the door pushes or pulls"
        elif orientation == "model_handle":
            handle_action = "locate the handle by touch, then gently test whether the door pushes or pulls"
        else:
            handle_action = "feel along the handle plate for the lever or pull, then gently test whether the door pushes or pulls"

        return {
            "confidence": round(float(confidence), 3),
            "handle_detected": True,
            "handle_confidence": round(float(confidence), 3),
            "handle_bbox": [int(x1), int(y1), int(x2), int(y2)],
            "handle_x_ratio": round(float(x_ratio), 3),
            "handle_y_ratio": round(float(y_ratio), 3),
            "handle_side": side,
            "recommended_hand": recommended_hand,
            "handle_height_zone": height_zone,
            "handle_orientation": orientation,
            "handle_support_side": support_side,
            "handle_action": handle_action,
            "distance_m": round(float(distance_m), 2),
            "near_field_ratio": round(float(near_field_ratio), 3),
        }

    def _detect_door_frame_context(self, frame: np.ndarray, w: int, h: int) -> Optional[Dict[str, object]]:
        # Focus on center region where doors are most relevant.
        y0 = max(0.18, self._cfg.surface_y_start - 0.35)
        y1 = 0.95
        x0, x1 = 0.15, 0.90

        y_start = int(h * y0)
        y_end = int(h * y1)
        x_start = int(w * x0)
        x_end = int(w * x1)

        if y_end - y_start < 48 or x_end - x_start < 64:
            return None

        crop = frame[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            return None

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(12, crop.shape[1] // 12),
            minLineLength=max(20, crop.shape[1] // 6),
            maxLineGap=10,
        )

        if lines is None or len(lines) < 3:
            return None

        # Analyze line orientations
        vertical_lines = []
        horizontal_lines = []

        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx < 3:  # Nearly vertical
                vertical_lines.append(line)
            elif dy < 3:  # Nearly horizontal
                horizontal_lines.append(line)

        # Door signature: at least two strong vertical lines. A horizontal line
        # helps, but many close-up camera angles omit the door top.
        if len(vertical_lines) < 2:
            return None

        # Check for vertical lines on left and right sides (door frame)
        crop_width = crop.shape[1]
        left_lines = [l for l in vertical_lines if l[0] < crop_width * 0.3]
        right_lines = [l for l in vertical_lines if l[0] > crop_width * 0.7]

        has_frame = len(left_lines) > 0 and len(right_lines) > 0

        if not has_frame:
            return None

        # Estimate distance based on vertical position
        y_center = (y_start + y_end) / 2 / h
        if y_center > 0.85:
            estimated_distance = 0.8
        elif y_center > 0.75:
            estimated_distance = 1.2
        else:
            estimated_distance = 1.8

        confidence = min(0.84, 0.42 + len(vertical_lines) * 0.045 + len(horizontal_lines) * 0.025 + 0.10)
        return {
            "vertical_lines": len(vertical_lines),
            "horizontal_lines": len(horizontal_lines),
            "confidence": round(float(confidence), 3),
            "near_field_ratio": round(float(len(vertical_lines) / max(1, len(lines))), 3),
            "distance_m": round(float(estimated_distance), 2),
        }

    @staticmethod
    def _direction_from_x_ratio(x_ratio: float) -> Direction:
        if x_ratio < 0.20:
            return Direction.LEFT
        if x_ratio < 0.42:
            return Direction.SLIGHT_LEFT
        if x_ratio <= 0.58:
            return Direction.CENTER
        if x_ratio <= 0.80:
            return Direction.SLIGHT_RIGHT
        return Direction.RIGHT

    def _detect_white_object(self, frame: np.ndarray, w: int, h: int) -> Optional[SurfaceObservation]:
        """Detect white/low-contrast obstacles using brightness and saturation analysis.

        Catches white tables, boxes, and other light-colored obstacles that YOLO misses
        due to low contrast or partial visibility.
        """
        # Focus on center-bottom region where ground-level obstacles appear
        y0 = self._cfg.surface_y_start
        y1 = 0.95
        x0, x1 = 0.30, 0.70  # Center 40% horizontally

        y_start = int(h * y0)
        y_end = int(h * y1)
        x_start = int(w * x0)
        x_end = int(w * x1)

        if y_end - y_start < 32 or x_end - x_start < 32:
            return None

        crop = frame[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            return None

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # White detection: low saturation, high brightness
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        # Find white/light regions
        white_mask = (sat < 40) & (val > 180)
        white_ratio = float(np.mean(white_mask))

        # Need significant white region to be an obstacle
        if white_ratio < 0.15:
            return None

        # Check if white region has structure (not just uniform white background)
        # Use variance in brightness to detect edges within white region
        val_variance = np.var(val[white_mask]) if np.any(white_mask) else 0
        if val_variance < 100:  # Too uniform, likely background
            return None

        # Check for spatial clustering (white object should be concentrated, not scattered)
        white_pixels = np.sum(white_mask)
        total_pixels = white_mask.size
        if white_pixels / total_pixels < 0.10:
            return None

        # Estimate distance based on vertical position
        y_center = (y_start + y_end) / 2 / h
        if y_center > 0.85:
            estimated_distance = 0.8
        elif y_center > 0.75:
            estimated_distance = 1.2
        else:
            estimated_distance = 1.8

        # Confidence based on white ratio and variance
        confidence = min(0.80, white_ratio * 1.5 + (val_variance / 500))

        return SurfaceObservation(
            kind=SurfaceKind.OBSTACLE_EDGE,  # Reuse obstacle edge type
            confidence=confidence,
            direction=Direction.CENTER,
            near_field_ratio=white_ratio,
            distance_m=estimated_distance,
            source="vision-white-object",
            attributes={
                "white_ratio": round(white_ratio, 3),
                "brightness_variance": round(val_variance, 2),
                "note": "white/low-contrast obstacle detected via brightness analysis",
            },
        )

    def _estimate_curb_edge(self, frame: np.ndarray, w: int, h: int) -> Optional[SurfaceObservation]:
        curb_y0, curb_y1 = self._cfg.curb_y_range
        y1 = int(h * curb_y0)
        y2 = int(h * curb_y1)
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
    def _draw_surface_observation(frame: np.ndarray, surface: SurfaceObservation) -> None:
        if surface.kind == SurfaceKind.DOOR and surface.attributes.get("handle_bbox"):
            try:
                x1, y1, x2, y2 = [int(v) for v in surface.attributes["handle_bbox"]]
            except Exception:
                return
            h, w = frame.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(x1 + 1, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(y1 + 1, min(h, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 220, 0), 2)
            label = f"handle {surface.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 220, 0),
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
