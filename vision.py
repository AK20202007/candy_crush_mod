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
    "dining table",
    "couch",
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
        self._consec_door_hits: int = 0
        self._last_spoken: Optional[str] = None
        self._frame_counter: int = 0
        self._cached_surfaces: List[SurfaceObservation] = []
        
        # Vehicle tracking for speed and direction estimation
        self._vehicle_tracks: Dict[str, List[Dict]] = {}  # track_id -> list of detections
        self._last_track_cleanup = time.time()

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

    def run_forever(self, camera_index: int = 0, stop_event: Optional[threading.Event] = None) -> None:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}")

        print("[vision] Webcam open. Press 'q' in the preview window or Ctrl+C in the terminal to quit.")

        frame_count = 0
        last_fps_time = time.time()
        fps_frames = 0
        last_health_check = time.time()

        # Reduce OpenCV buffer to 1 frame to minimize lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    # Drain stale buffered frames — only process the latest.
                    # Grab+discard a few frames to clear the internal buffer,
                    # then do a full read for the freshest frame.
                    for _ in range(2):
                        cap.grab()
                    ret, frame = cap.read()
                    if not ret:
                        print("[vision] Failed to read frame from webcam")
                        time.sleep(0.1)
                        continue

                    # Flip horizontally: webcams produce mirrored (selfie)
                    # images, so left in the real world appears as right on
                    # screen.  Flipping corrects directions so "left" in our
                    # detection matches the user's actual left.
                    frame = cv2.flip(frame, 1)

                    w, h = frame.shape[1], frame.shape[0]
                    decision = self._process_frame(frame, w, h)

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

                    cv2.imshow("Assistive Nav — preview", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[vision] Quit requested from preview window.")
                        break
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
            cv2.destroyAllWindows()
            print(f"[vision] Processed {frame_count} frames total")

    def _process_frame(self, frame: np.ndarray, w: int, h: int) -> AgentDecision:
        try:
            now_ms = int(time.time() * 1000)
            self._frame_counter += 1
            detections = self._detect(frame, w, h)
            # Track vehicles to estimate speed and direction
            detections = self._track_vehicles(detections, now_ms)
            warnings = self._warnings_from_detections(detections, now_ms)
            # Run expensive surface heuristics every other frame; reuse cache otherwise
            if self._cfg.enable_surface_heuristic and (self._frame_counter % 2 == 0):
                surfaces = self._surface_observations(frame, w, h, detections)
                self._cached_surfaces = surfaces
            elif self._cfg.enable_surface_heuristic:
                surfaces = self._cached_surfaces
            else:
                surfaces = []
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
                except Exception as e:
                    print(f"[vision] Frame decision callback error: {e}")
            return decision
        except Exception as e:
            print(f"[vision] CRITICAL: Frame processing failed: {e}")
            import traceback
            traceback.print_exc()
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

    def _track_vehicles(self, detections: List[Detection], now_ms: int) -> List[Detection]:
        """Track vehicles across frames to estimate speed and direction."""
        vehicle_labels = {"car", "bus", "truck", "motorcycle", "bicycle"}
        tracked_detections = []
        
        # Clean up old tracks (older than 2 seconds)
        if now_ms - self._last_track_cleanup > 2000:
            self._last_track_cleanup = now_ms
            cutoff = now_ms - 2000
            self._vehicle_tracks = {
                k: v for k, v in self._vehicle_tracks.items()
                if v[-1]["timestamp_ms"] > cutoff
            }
        
        for det in detections:
            if det.label.lower() not in vehicle_labels:
                tracked_detections.append(det)
                continue
            
            # Simple tracking based on bounding box center
            bbox = det.attributes.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                tracked_detections.append(det)
                continue
            
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Find matching track (closest center position from previous frame)
            track_id = None
            best_distance = 50  # pixels
            for tid, history in self._vehicle_tracks.items():
                last = history[-1]
                last_center_x = last["center_x"]
                last_center_y = last["center_y"]
                distance = ((center_x - last_center_x)**2 + (center_y - last_center_y)**2)**0.5
                if distance < best_distance:
                    best_distance = distance
                    track_id = tid
            
            if track_id is None:
                # Create new track
                track_id = f"{det.label}_{len(self._vehicle_tracks)}_{now_ms}"
                self._vehicle_tracks[track_id] = []
            
            # Add to track
            self._vehicle_tracks[track_id].append({
                "timestamp_ms": now_ms,
                "center_x": center_x,
                "center_y": center_y,
                "distance_m": det.distance_m,
            })
            
            # Calculate speed and direction if we have enough history
            history = self._vehicle_tracks[track_id]
            if len(history) >= 2:
                # Get positions from last 2 frames
                prev = history[-2]
                curr = history[-1]
                dt = (curr["timestamp_ms"] - prev["timestamp_ms"]) / 1000.0  # seconds
                
                if dt > 0:
                    # Calculate movement in pixels
                    dx = curr["center_x"] - prev["center_x"]
                    dy = curr["center_y"] - prev["center_y"]
                    
                    # Estimate speed (pixels/second, roughly m/s at typical distances)
                    pixel_speed = ((dx**2 + dy**2)**0.5) / dt
                    
                    # Convert to approximate m/s (heuristic: 100 pixels ≈ 1m at 5m distance)
                    if det.distance_m is not None:
                        m_per_pixel = det.distance_m / (bbox[2] - bbox[0])  # rough scale
                        speed_mps = pixel_speed * m_per_pixel * 10  # adjusted heuristic
                    else:
                        speed_mps = pixel_speed / 50  # fallback heuristic
                    
                    # Determine direction
                    if abs(dx) > abs(dy):
                        # Horizontal movement
                        if dx > 0:
                            direction = "right"
                        else:
                            direction = "left"
                    else:
                        # Vertical movement
                        if dy > 0:
                            direction = "down"  # getting closer/larger
                        else:
                            direction = "up"  # getting farther/smaller
                    
                    # Add motion attributes to detection
                    motion_attrs = det.attributes.copy()
                    motion_attrs.update({
                        "speed_mps": round(speed_mps, 2),
                        "movement_direction": direction,
                        "is_moving": speed_mps > 0.5,  # threshold for "moving"
                        "track_id": track_id,
                    })
                    
                    tracked_det = Detection(
                        label=det.label,
                        confidence=det.confidence,
                        distance_m=det.distance_m,
                        direction=det.direction,
                        attributes=motion_attrs,
                    )
                    tracked_detections.append(tracked_det)
                else:
                    tracked_detections.append(det)
            else:
                tracked_detections.append(det)
        
        return tracked_detections

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
                distance_scale=self._cfg.effective_distance_scale,
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

    def _surface_observations(self, frame: np.ndarray, w: int, h: int,
                              detections: Optional[List[Detection]] = None) -> List[SurfaceObservation]:
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

        # Suppress door handle detection when YOLO sees furniture (chair, bench, couch)
        # — their handles/edges produce false-positive door handle scores.
        _furniture_labels = {"chair", "bench", "couch", "dining table"}
        has_furniture = detections and any(
            d.label.lower() in _furniture_labels and d.confidence >= 0.40
            for d in detections
        )

        is_outdoor = self._cfg.location_type in {"sidewalk", "street", "street_crossing", "outdoor"}
        is_indoor = self._cfg.location_type in {"hallway", "room", "corridor", "building", "indoor"}

        # Door detection — skip outdoors and when furniture is present
        if not has_furniture and not is_outdoor:
            try:
                door = self._detect_door(frame, w, h)
                if door is not None:
                    observations.append(door)
            except Exception as e:
                print(f"[vision] Door detection error: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Reset consecutive door counter when suppressed
            self._consec_door_hits = 0

        # White object detection for low-contrast obstacles (white tables, boxes)
        try:
            white_obj = self._detect_white_object(frame, w, h)
            if white_obj is not None:
                observations.append(white_obj)
        except Exception as e:
            print(f"[vision] White object detection error: {e}")
            import traceback
            traceback.print_exc()
        
        # Outdoor-only detections: skip indoors to avoid false positives and save CPU
        if not is_indoor:
            # Crosswalk detection using zebra stripe patterns
            try:
                crosswalk = self._detect_crosswalk(frame, w, h)
                if crosswalk is not None:
                    observations.append(crosswalk)
            except Exception as e:
                print(f"[vision] Crosswalk detection error: {e}")
                import traceback
                traceback.print_exc()
            
            # Sidewalk boundary detection using edge density
            try:
                sidewalk_boundary = self._detect_sidewalk_boundary(frame, w, h)
                if sidewalk_boundary is not None:
                    observations.append(sidewalk_boundary)
            except Exception as e:
                print(f"[vision] Sidewalk boundary detection error: {e}")
                import traceback
                traceback.print_exc()
            
            # Sidewalk obstacle detection (poles, bollards)
            try:
                sidewalk_obstacles = self._detect_sidewalk_obstacles(frame, w, h)
                for obs in sidewalk_obstacles:
                    observations.append(obs)
            except Exception as e:
                print(f"[vision] Sidewalk obstacle detection error: {e}")
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
        
        # Enhanced texture analysis for road vs sidewalk distinction
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        texture_variance = np.var(gray)
        
        # Edge density for texture analysis
        edges = cv2.Canny(gray, 40, 100)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
        
        # Road: typically dark gray with moderate saturation, higher edge density (asphalt texture)
        # Sidewalk: lighter gray with lower saturation, lower edge density (smoother concrete)
        
        # Road indicators
        is_road = (gray_ratio > 0.4 and dark_ratio > 0.3 and edge_density > 0.08 and texture_variance > 100)
        
        # Sidewalk indicators
        is_sidewalk = (gray_ratio > 0.3 and bright_ratio > 0.2 and edge_density < 0.15 and 50 < texture_variance < 400)

        if is_road:
            return SurfaceObservation(
                kind=SurfaceKind.ROAD,
                confidence=min(0.85, gray_ratio + dark_ratio + edge_density),
                direction=direction,
                near_field_ratio=gray_ratio,
                distance_m=0.8 if direction == Direction.CENTER else 1.2,
                source="vision-surface-heuristic-enhanced",
                attributes={
                    "gray_ratio": round(gray_ratio, 3),
                    "bright_gray_ratio": round(bright_ratio, 3),
                    "dark_gray_ratio": round(dark_ratio, 3),
                    "edge_density": round(edge_density, 3),
                    "texture_variance": round(texture_variance, 2),
                    "note": "enhanced road detection via texture and edge analysis",
                },
            )
        
        if is_sidewalk:
            return SurfaceObservation(
                kind=SurfaceKind.SIDEWALK,
                confidence=min(0.85, gray_ratio + bright_ratio + (1.0 - edge_density)),
                direction=direction,
                near_field_ratio=gray_ratio,
                distance_m=2.0 if direction == Direction.CENTER else 2.5,
                source="vision-surface-heuristic-enhanced",
                attributes={
                    "gray_ratio": round(gray_ratio, 3),
                    "bright_gray_ratio": round(bright_ratio, 3),
                    "dark_gray_ratio": round(dark_ratio, 3),
                    "edge_density": round(edge_density, 3),
                    "texture_variance": round(texture_variance, 2),
                    "note": "enhanced sidewalk detection via texture and edge analysis",
                },
            )
        
        # Fallback to original logic for backward compatibility
        if gray_ratio > 0.4 and dark_ratio > 0.3:
            return SurfaceObservation(
                kind=SurfaceKind.ROAD,
                confidence=min(0.75, gray_ratio + dark_ratio),
                direction=direction,
                near_field_ratio=gray_ratio,
                distance_m=0.8 if direction == Direction.CENTER else 1.2,
                source="vision-surface-heuristic",
                attributes={
                    "gray_ratio": round(gray_ratio, 3),
                    "bright_gray_ratio": round(bright_ratio, 3),
                    "dark_gray_ratio": round(dark_ratio, 3),
                    "note": "color/texture heuristic; use segmentation+depth for deployment",
                },
            )
        
        # Fallback sidewalk: bright gray surface that didn't match enhanced thresholds
        if gray_ratio > 0.3 and bright_ratio > 0.2:
            return SurfaceObservation(
                kind=SurfaceKind.SIDEWALK,
                confidence=min(0.70, gray_ratio + bright_ratio),
                direction=direction,
                near_field_ratio=gray_ratio,
                distance_m=2.0 if direction == Direction.CENTER else 2.5,
                source="vision-surface-heuristic",
                attributes={
                    "gray_ratio": round(gray_ratio, 3),
                    "bright_gray_ratio": round(bright_ratio, 3),
                    "dark_gray_ratio": round(dark_ratio, 3),
                    "note": "color/texture heuristic; use segmentation+depth for deployment",
                },
            )

    def _detect_edge_density_obstacles(self, frame: np.ndarray, w: int, h: int) -> List[SurfaceObservation]:
        """Detect unknown obstacles via edge density in left/center/right bands.
        
        Catches obstacles that YOLO might miss entirely (walls, pillars, boxes)
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
        # Allow walls with fewer distinct edges by using ratio instead of absolute count
        if vertical_count <= horizontal_count * 1.5:
            return None

        # Estimate distance based on vertical position in frame
        # Lower in frame = closer
        y_center = (y_start + y_end) / 2 / h
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
        """Detect a door interaction target.

        The preferred signal is the handle itself. Door frames are useful
        context, but they are less actionable than "handle at 2 o'clock; reach
        right hand; press the lever down."
        """
        frame_context = self._detect_door_frame_context(frame, w, h)
        handle = self._detect_door_handle(frame, w, h)

        if handle is not None:
            handle_conf = float(handle["confidence"])
            # Low-confidence handles require door frame context to avoid
            # false positives.  High-confidence handles (≥0.85) are trusted
            # on their own since the edge/color/geometry criteria are strong.
            if handle_conf < 0.85 and frame_context is None:
                handle = None   # discard likely false positive

        # Consecutive-frame gate: require 3 consecutive detections to suppress
        # single-frame hallucinations (random edges scored as handles).
        if handle is not None or frame_context is not None:
            self._consec_door_hits += 1
        else:
            self._consec_door_hits = 0
            return None

        if self._consec_door_hits < 3:
            return None

        if handle is not None:
            confidence = float(handle["confidence"])
            if frame_context is not None:
                confidence = min(0.94, confidence + 0.08)

            attrs = dict(handle)
            attrs["has_frame"] = frame_context is not None
            if frame_context is not None:
                attrs["door_frame"] = frame_context
            attrs["note"] = "door handle detected; verify push or pull gently"

            return SurfaceObservation(
                kind=SurfaceKind.DOOR,
                confidence=confidence,
                direction=self._direction_from_x_ratio(float(handle["handle_x_ratio"])),
                near_field_ratio=float(handle.get("near_field_ratio", 0.0)),
                distance_m=float(handle["distance_m"]),
                source="vision-door-handle",
                attributes=attrs,
            )

        if frame_context is None:
            return None

        return SurfaceObservation(
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
                "handle_action": "find the handle by touch, then test push or pull gently",
                "note": "door frame detected; handle not visible",
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
            return None

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
            if score < 0.78:
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

        return best

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

    def _detect_crosswalk(self, frame: np.ndarray, w: int, h: int) -> Optional[SurfaceObservation]:
        """Detect crosswalk zebra stripe patterns for pedestrian crossing guidance.
        
        Crosswalks have alternating white/dark stripes perpendicular to the walking direction.
        """
        # Focus on center-bottom region where crosswalks appear
        y0 = self._cfg.surface_y_start
        y1 = 0.95
        x0, x1 = 0.20, 0.80  # Center 60% horizontally
        
        y_start = int(h * y0)
        y_end = int(h * y1)
        x_start = int(w * x0)
        x_end = int(w * x1)
        
        if y_end - y_start < 32 or x_end - x_start < 32:
            return None
            
        crop = frame[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal lines (zebra stripes are horizontal)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(15, crop.shape[1] // 10),
            minLineLength=max(30, crop.shape[1] // 4),
            maxLineGap=15,
        )
        
        if lines is None or len(lines) < 3:
            return None
        
        # Filter for horizontal lines (zebra stripes)
        horizontal_lines = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy < 3:  # Nearly horizontal
                horizontal_lines.append(line)
        
        # Need multiple horizontal lines to be a crosswalk
        if len(horizontal_lines) < 3:
            return None
        
        # Check for alternating pattern (zebra stripes)
        # Group lines by y-position
        line_y_positions = sorted([l[1] for l in horizontal_lines])
        
        # Check if lines are roughly evenly spaced
        if len(line_y_positions) >= 3:
            gaps = [line_y_positions[i+1] - line_y_positions[i] for i in range(len(line_y_positions)-1)]
            avg_gap = np.mean(gaps) if gaps else 0
            gap_variance = np.var(gaps) if gaps else 0
            # Crosswalk stripes should have relatively even spacing
            if gap_variance > (avg_gap * 2) ** 2:
                return None
        
        # Estimate distance based on vertical position
        y_center = (y_start + y_end) / 2 / h
        if y_center > 0.85:
            estimated_distance = 1.0
        elif y_center > 0.75:
            estimated_distance = 2.0
        else:
            estimated_distance = 3.5
        
        # Confidence based on number of horizontal lines
        confidence = min(0.85, 0.4 + len(horizontal_lines) * 0.05)
        
        return SurfaceObservation(
            kind=SurfaceKind.CROSSWALK,
            confidence=confidence,
            direction=Direction.CENTER,
            near_field_ratio=len(horizontal_lines) / max(1, len(lines)),
            distance_m=estimated_distance,
            source="vision-crosswalk-detection",
            attributes={
                "horizontal_lines": len(horizontal_lines),
                "line_spacing_avg": round(np.mean(gaps) if gaps else 0, 1),
                "note": "zebra stripe pattern detected",
            },
        )

    def _detect_sidewalk_boundary(self, frame: np.ndarray, w: int, h: int) -> Optional[SurfaceObservation]:
        """Detect sidewalk boundaries using edge density and texture analysis.
        
        Sidewalks typically have:
        - Clear boundary edges (curb or grass)
        - More uniform texture than roads
        - Less visual noise than asphalt
        """
        # Analyze left, center, and right regions for sidewalk boundaries
        directions = [
            (Direction.LEFT, 0.0, 0.35),
            (Direction.CENTER, 0.35, 0.65),
            (Direction.RIGHT, 0.65, 1.0),
        ]
        
        best_boundary = None
        best_confidence = 0.0
        
        for direction, x_start_ratio, x_end_ratio in directions:
            y0 = self._cfg.surface_y_start
            y1 = 0.95
            
            y_start = int(h * y0)
            y_end = int(h * y1)
            x_start = int(w * x_start_ratio)
            x_end = int(w * x_end_ratio)
            
            if y_end - y_start < 32 or x_end - x_start < 32:
                continue
                
            crop = frame[y_start:y_end, x_start:x_end]
            if crop.size == 0:
                continue
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Edge density analysis
            edges = cv2.Canny(gray, 40, 100)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0
            
            # Texture uniformity (variance in grayscale)
            texture_variance = np.var(gray)
            
            # Sidewalks have moderate edge density (not too smooth like road, not too noisy like grass)
            # and moderate texture variance
            sidewalk_edge_density = (0.05 <= edge_density <= 0.20)
            sidewalk_texture = (50 <= texture_variance <= 500)
            
            if sidewalk_edge_density and sidewalk_texture:
                confidence = min(0.80, edge_density * 3 + (texture_variance / 1000))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_boundary = SurfaceObservation(
                        kind=SurfaceKind.SIDEWALK,
                        confidence=confidence,
                        direction=direction,
                        near_field_ratio=edge_density,
                        distance_m=2.0,  # Sidewalk typically extends ahead
                        source="vision-sidewalk-boundary",
                        attributes={
                            "edge_density": round(edge_density, 3),
                            "texture_variance": round(texture_variance, 2),
                            "note": "sidewalk detected via boundary and texture analysis",
                        },
                    )
        
        return best_boundary

    def _detect_sidewalk_obstacles(self, frame: np.ndarray, w: int, h: int) -> List[SurfaceObservation]:
        """Detect sidewalk obstacles like poles, signs, and bollards.
        
        These obstacles are typically:
        - Vertical thin objects (poles, bollards)
        - Rectangular shapes (signs)
        - Located along sidewalk edges
        - Have distinctive edge patterns
        """
        observations = []
        
        # Analyze left, center, and right regions
        regions = [
            (Direction.LEFT, 0.0, 0.35),
            (Direction.CENTER, 0.35, 0.65),
            (Direction.RIGHT, 0.65, 1.0),
        ]
        
        for direction, x_start_ratio, x_end_ratio in regions:
            y0 = self._cfg.surface_y_start
            y1 = 0.95
            
            y_start = int(h * y0)
            y_end = int(h * y1)
            x_start = int(w * x_start_ratio)
            x_end = int(w * x_end_ratio)
            
            if y_end - y_start < 32 or x_end - x_start < 32:
                continue
                
            crop = frame[y_start:y_end, x_start:x_end]
            if crop.size == 0:
                continue
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Detect vertical edges (poles, bollards)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=max(10, crop.shape[1] // 12),
                minLineLength=max(20, crop.shape[0] // 3),
                maxLineGap=10,
            )
            
            if lines is None or len(lines) < 2:
                continue
            
            # Count vertical vs horizontal lines
            vertical_count = 0
            horizontal_count = 0
            for line in lines[:, 0]:
                x1, y1, x2, y2 = line
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx < 3:  # Nearly vertical
                    vertical_count += 1
                elif dy < 3:  # Nearly horizontal
                    horizontal_count += 1
            
            # Poles/bollards: strong vertical dominance
            is_pole = vertical_count >= 3 and vertical_count > horizontal_count * 2
            
            # Sign detection disabled — the mixed-edge heuristic produces too
            # many false positives (bookshelves, window frames, table edges).
            # YOLO already detects stop signs and traffic lights reliably.
            
            if is_pole:
                # Estimate distance based on vertical position
                y_center = (y_start + y_end) / 2 / h
                if y_center > 0.85:
                    estimated_distance = 0.8
                elif y_center > 0.75:
                    estimated_distance = 1.5
                else:
                    estimated_distance = 2.5
                
                obstacle_type = "pole or bollard"
                confidence = min(0.80, 0.5 + vertical_count * 0.05)
                
                observations.append(SurfaceObservation(
                    kind=SurfaceKind.OBSTACLE_EDGE,  # Reuse obstacle edge type
                    confidence=confidence,
                    direction=direction,
                    near_field_ratio=vertical_count / max(1, len(lines)),
                    distance_m=estimated_distance,
                    source="vision-sidewalk-obstacle",
                    attributes={
                        "obstacle_type": obstacle_type,
                        "vertical_lines": vertical_count,
                        "horizontal_lines": horizontal_count,
                        "is_pole": is_pole,
                        "note": f"{obstacle_type} detected via edge pattern analysis",
                    },
                ))
        
        return observations

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
        
        # Estimate curb height based on vertical position and edge characteristics
        # Lower in frame = closer = taller curb appears
        if y_ratio > 0.7:
            estimated_height_cm = 15  # Standard curb ~15cm
        elif y_ratio > 0.5:
            estimated_height_cm = 10  # Lower curb or ramp
        else:
            estimated_height_cm = 5  # Very low, possibly a ramp
        
        # Detect ramps by checking for gradual slope instead of sharp edge
        # Ramps have more gradual transitions in the edge profile
        is_ramp = False
        if length_ratio > 0.5 and y_ratio < 0.6:
            # Check if edge is more gradual (ramp-like)
            # Ramps typically have longer, less distinct edges
            is_ramp = True
            estimated_height_cm = 5  # Ramps are low
        
        return SurfaceObservation(
            kind=SurfaceKind.CURB,
            confidence=min(0.85, 0.40 + length_ratio * 0.40),
            direction=Direction.CENTER,
            near_field_ratio=min(1.0, length_ratio),
            distance_m=distance_m,
            source="vision-curb-edge-enhanced",
            attributes={
                "horizontal_edge_ratio": round(length_ratio, 3),
                "edge_y_ratio": round(y_ratio, 3),
                "estimated_height_cm": estimated_height_cm,
                "is_ramp": is_ramp,
                "note": "curb with height estimation and ramp detection",
            },
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
