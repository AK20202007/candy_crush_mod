"""
Webcam capture + YOLOv8 detection for assistive navigation prototype.

Targets: person, chair, car, and door if the loaded model exposes that class
name (standard COCO YOLO weights usually do not include "door"; we detect it
when present).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# Names must match the model's class strings (COCO-style).
CORE_CLASSES: Set[str] = {"person", "chair", "car"}
OPTIONAL_CLASSES: Set[str] = {"door"}


@dataclass
class VisionConfig:
    """Tunable thresholds for warnings."""

    model_path: str = "yolov8n.pt"
    # Fraction of frame area: above this => "close" obstacle (chair/car/door).
    obstacle_area_ratio: float = 0.12
    # How close to image center (normalized 0–1) a person's bbox center must be.
    person_center_radius: float = 0.18
    # Minimum seconds between repeating the same warning phrase.
    warning_cooldown_s: float = 2.5


class VisionSystem:
    """
    Opens the default webcam, runs YOLO each frame, and invokes callbacks.

    Callbacks receive short user-facing strings; the caller routes them to TTS.
    """

    def __init__(
        self,
        on_warning: Callable[[str], None],
        config: Optional[VisionConfig] = None,
    ) -> None:
        self._on_warning = on_warning
        self._cfg = config or VisionConfig()
        self._model = YOLO(self._cfg.model_path)
        self._name_to_id: Dict[str, int] = {
            name.lower(): idx for idx, name in self._model.names.items()
        }
        self._active_classes = set(CORE_CLASSES)
        for name in OPTIONAL_CLASSES:
            if name in self._name_to_id:
                self._active_classes.add(name)

        self._last_phrase_time: Dict[str, float] = {}

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

    def _maybe_warn(self, phrase: str) -> None:
        if self._cooldown_ok(phrase):
            self._on_warning(phrase)

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
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray, w: int, h: int) -> None:
        results = self._model.predict(source=frame, verbose=False, stream=False)
        if not results:
            return
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return

        boxes = r0.boxes.xyxy.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)

        frame_area = float(w * h)
        cx_img, cy_img = w / 2.0, h / 2.0

        person_center_hit = False
        obstacle_close = False

        for (x1, y1, x2, y2), cls_id in zip(boxes, clss):
            name = self._model.names[int(cls_id)].lower()
            if name not in self._active_classes:
                continue

            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            area_ratio = (bw * bh) / frame_area

            if name == "person":
                bx_c = (x1 + x2) / 2.0
                by_c = (y1 + y2) / 2.0
                dx = (bx_c - cx_img) / w
                dy = (by_c - cy_img) / h
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < self._cfg.person_center_radius:
                    person_center_hit = True
            else:
                # chair / car / door — treat large bbox as "close" obstacle.
                if area_ratio >= self._cfg.obstacle_area_ratio:
                    obstacle_close = True

            # Draw lightweight debug overlay.
            color = (0, 200, 0) if name == "person" else (0, 165, 255)
            p1: Tuple[int, int] = (int(x1), int(y1))
            p2: Tuple[int, int] = (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(
                frame,
                f"{name} {area_ratio:.2f}",
                (int(x1), max(20, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if person_center_hit:
            self._maybe_warn("Watch out, person ahead")
        if obstacle_close:
            self._maybe_warn("Obstacle ahead")
