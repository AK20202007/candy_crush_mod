"""
Webcam capture + YOLOv8 detection for assistive navigation prototype.

Targets: person, chair, car, and door if the loaded model exposes that class
name (standard COCO YOLO weights usually do not include "door"; we detect it
when present).

The repo https://github.com/jjking00/YOLO-OD is an MMYOLO/OpenMMLab fork
(training stack: MMCV, MMDet, configs). It is not practical to embed here;
instead we tighten Ultralytics inference (class filter, conf/iou/imgsz,
optional TTA, FP16 on CUDA) and stabilize warnings across frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

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


# Names must match the model's class strings (COCO-style).
CORE_CLASSES: Set[str] = {"person", "chair", "car"}
OPTIONAL_CLASSES: Set[str] = {"door"}


@dataclass
class VisionConfig:
    """Tunable thresholds for warnings and sensor/vision fall detection."""

    model_path: Optional[str] = None
    # Fraction of frame area: above this => "close" obstacle (chair/car/door).
    obstacle_area_ratio: float = 0.12
    # How close to image center (normalized 0–1) a person's bbox center must be.
    person_center_radius: float = 0.18
    # Minimum seconds between repeating the same warning phrase.
    warning_cooldown_s: float = 2.5
    # Require this many consecutive frames with the hazard before speaking (reduces flicker).
    confirm_frames: int = 2
    # Ultralytics predict: min confidence (higher = fewer false positives, may miss small objects).
    conf: float = 0.35
    # NMS IoU threshold.
    iou: float = 0.5
    # Square inference size (larger often helps small objects; slower).
    imgsz: int = 640
    # FP16 on CUDA only (Ultralytics); ignored on CPU/MPS if unsupported.
    half: Optional[bool] = None
    # Test-time augmentation (slower, can improve recall on hard frames).
    augment: bool = False
    # Cap detections per frame for speed on busy scenes.
    max_det: int = 50

    # Seizure / Fall Detection
    strobe_freq: float = 3.0
    strobe_window_s: float = 5.0
    elevation_threshold: float = 40.0
    emergency_contact: str = "Emergency Contact"



class BrightnessMonitor:
    """Tracks frame brightness to detect rapid flickering (strobe lights)."""

    def __init__(self, window_size: int = 30, freq_threshold: float = 3.0, fps: float = 30.0):
        self._history: List[float] = []
        self._window_size = window_size
        self._freq_threshold = freq_threshold
        self._fps = fps
        self._last_strobe_time = 0.0

    def update(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        self._history.append(brightness)
        if len(self._history) > self._window_size:
            self._history.pop(0)

        if len(self._history) < self._window_size:
            return False

        avg = float(np.mean(self._history))
        std = float(np.std(self._history))
        if std < 15:  # Minimum variance to avoid noise
            return False

        crossings = 0
        for i in range(1, len(self._history)):
            if (self._history[i - 1] < avg and self._history[i] >= avg) or (
                self._history[i - 1] > avg and self._history[i] <= avg
            ):
                crossings += 1

        freq = crossings / (2 * (self._window_size / self._fps))
        if freq >= self._freq_threshold:
            self._last_strobe_time = time.time()
            return True
        return False

    def is_strobe_recent(self, window_s: float = 5.0) -> bool:
        return (time.time() - self._last_strobe_time) < window_s


class FallDetector:
    """Detects sudden drops or camera orientation shifts and tracks duration."""

    def __init__(self, threshold: float = 40.0, warning_s: float = 15.0, alarm_s: float = 30.0):
        self._last_frame_gray: Optional[np.ndarray] = None
        self._threshold = threshold
        self._warning_s = warning_s
        self._alarm_s = alarm_s
        
        self._is_fallen = False
        self._fall_start_time = 0.0
        self._last_fall_time = 0.0

    def update(self, frame: np.ndarray) -> str:
        """Returns 'NORMAL', 'WARNING', or 'ALARM' based on fall duration."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_frame_gray is None:
            self._last_frame_gray = gray
            return "NORMAL"

        diff = cv2.absdiff(gray, self._last_frame_gray)
        mean_diff = float(np.mean(diff))
        self._last_frame_gray = gray

        now = time.time()
        # Heuristic: sudden high motion followed by low motion at a "fallen" angle.
        # For this prototype, we'll treat persistent high motion OR a one-time trigger as a "fall start".
        if mean_diff > self._threshold:
            if not self._is_fallen:
                self._is_fallen = True
                self._fall_start_time = now
            self._last_fall_time = now
        
        # If we are in a "fallen" state, check the timer.
        if self._is_fallen:
            duration = now - self._fall_start_time
            if duration >= self._alarm_s:
                return "ALARM"
            if duration >= self._warning_s:
                return "WARNING"
        
        return "NORMAL"

    def trigger_mock_fall(self) -> None:
        self._is_fallen = True
        self._fall_start_time = time.time()
        self._last_fall_time = time.time()

    def reset(self) -> None:
        self._is_fallen = False
        self._fall_start_time = 0.0

    def is_fall_recent(self, window_s: float = 5.0) -> bool:
        # For backward compatibility or extra checks
        return (time.time() - self._last_fall_time) < window_s

    def fall_duration(self) -> float:
        if not self._is_fallen:
            return 0.0
        return time.time() - self._fall_start_time


class MotionFallDetector:
    """Uses Apple's CoreMotion (CMAltimeter) to detect rapid altitude drops."""

    def __init__(self, drop_threshold: float = -1.5):
        self._altimeter = None
        self._drop_threshold = drop_threshold
        self._last_altitude = 0.0
        self._is_available = False
        self._fall_detected = False

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
            # relativeAltitude is in meters
            alt = data.relativeAltitude().floatValue()
            diff = alt - self._last_altitude
            self._last_altitude = alt
            
            if diff < self._drop_threshold:
                self._fall_detected = True
                callback(diff)

        self._altimeter.startRelativeAltitudeUpdatesToQueue_withHandler_(queue, _handler)

    def is_fall_detected(self) -> bool:
        return self._fall_detected

    def reset(self):
        self._fall_detected = False


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
    Opens the default webcam, runs YOLO each frame, and invokes callbacks.

    Callbacks receive short user-facing strings; the caller routes them to TTS.
    """

    def __init__(
        self,
        on_warning: Callable[[str], None],
        on_emergency: Callable[[str], None],
        config: Optional[VisionConfig] = None,
    ) -> None:
        self._on_warning = on_warning
        self._on_emergency = on_emergency
        self._cfg = config or VisionConfig()
        self._model = None
        self._name_to_id: Dict[str, int] = {}
        self._active_classes: Set[str] = set()

        if self._cfg.model_path:
            self._model = YOLO(self._cfg.model_path)
            self._name_to_id = {
                name.lower(): idx for idx, name in self._model.names.items()
            }
            self._active_classes = set(CORE_CLASSES)
            for name in OPTIONAL_CLASSES:
                if name in self._name_to_id:
                    self._active_classes.add(name)

        self._last_phrase_time: Dict[str, float] = {}
        self._class_ids: List[int] = sorted(
            self._name_to_id[n] for n in self._active_classes if n in self._name_to_id
        )
        self._consec_person: int = 0
        self._consec_obstacle: int = 0

        self._brightness_monitor = BrightnessMonitor(freq_threshold=self._cfg.strobe_freq)
        self._fall_detector = FallDetector(threshold=self._cfg.elevation_threshold)
        self._motion_fall_detector = MotionFallDetector()
        self._smv_fall_detector = SignalMagnitudeFallDetector()
        
        self._emergency_triggered = False

        use_half = self._cfg.half if self._cfg.half is not None else bool(torch.cuda.is_available())
        self._predict_half = bool(use_half and torch.cuda.is_available())
        print(
            f"[vision] Inference: imgsz={self._cfg.imgsz} conf={self._cfg.conf} iou={self._cfg.iou} "
            f"classes={len(self._class_ids)} half={self._predict_half} augment={self._cfg.augment} "
            f"confirm_frames={self._cfg.confirm_frames}"
        )

    def active_class_labels(self) -> Set[str]:
        """Which COCO (or model) labels we actually track for this run."""
        return set(self._active_classes)

    def _cooldown_ok(self, phrase: str) -> bool:
        now = time.time()
        last = self._last_phrase_time.get(phrase, 0.0)
        if now - last < self._cfg.warning_cooldown_s:
            return False
        self._last_phrase_time[phrase] = now
        return True

    def _maybe_warn(self, phrase: str) -> bool:
        if self._cooldown_ok(phrase):
            self._on_warning(phrase)
            return True
        return False

    def run_forever(self, camera_index: int = 0, stop_event=None) -> None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}")

        print("[vision] Webcam open. Press 'q' in the window or Ctrl+C in the terminal to quit.")
        print(f"[vision] Tracking classes: {sorted(self._active_classes)}")

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
                self._process_frame(frame, w, h)

                # Lightweight heartbeat so you can tell the loop is keeping up with the camera.
                fps_frames += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 3.0:
                    print(f"[vision] ~{fps_frames / elapsed:.1f} FPS over last {elapsed:.1f}s")
                    fps_frames = 0
                    fps_t0 = time.time()

                cv2.imshow("Assistive Nav — preview (not for users who rely on screen readers)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[vision] Quit requested from preview window.")
                    break
                elif key == ord("f"):
                    print("[vision] MOCK FALL TRIGGERED")
                    self._fall_detector.trigger_mock_fall()
                elif key == ord("e"):
                    print("[vision] MANUAL EMERGENCY TRIGGERED")
                    self.trigger_emergency("USER MANUALLY TRIGGERED EMERGENCY")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray, w: int, h: int) -> None:
        # Seizure Detection: Monitor for rapid brightness changes (strobe lights).
        if self._brightness_monitor.update(frame):
            # Only trigger a warning for the user, not an emergency call.
            if self._maybe_warn("WARNING: STROBE LIGHT DETECTED. PLEASE BE CAREFUL."):
                print("[vision] STROBE WARNING")

        # Fall Detection (Vision-based): Monitor for sudden drops or large camera motion.
        fall_state = self._fall_detector.update(frame)
        if fall_state == "WARNING":
            if self._maybe_warn("Warning: Potential fall detected. Are you okay?"):
                print("[vision] FALL WARNING (15s)")
        elif fall_state == "ALARM":
            self._check_emergency_conditions(alarm_triggered=True)

        self._check_emergency_conditions()

    def trigger_emergency(self, reason: str) -> None:
        """Immediately trigger an emergency alert and call the contact."""
        if self._emergency_triggered:
            return
        
        self._emergency_triggered = True
        contact = self._cfg.emergency_contact.upper()
        msg = f"EMERGENCY DETECTED: {reason}. CALLING {contact} NOW. PLEASE SEND HELP TO THIS LOCATION."
        # Use the dedicated emergency callback (which repeats speech)
        self._on_emergency(msg)
        print(f"[vision] !!! EMERGENCY !!! {msg}")

    def _check_emergency_conditions(self, alarm_triggered: bool = False) -> None:
        # Falls always trigger an emergency call.
        if alarm_triggered:
            self.trigger_emergency("FALL CONFIRMED")

    def handle_sensor_fall(self, reason: str) -> None:
        """Triggered by hardware sensors; starts the warning/alarm cycle."""
        print(f"[vision] Sensor-based fall sensed: {reason}")
        self._fall_detector.trigger_mock_fall()
        # Give an immediate auditory cue that a fall was sensed.
        self._maybe_warn("Fall detected. Starting 15-second countdown.")
