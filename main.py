#!/usr/bin/env python3
"""
Assistive navigation prototype: webcam + YOLO warnings + mock directions.

Combines:
  - vision.py  : person / obstacle cues
  - speech.py  : pyttsx3 with urgent vs normal priority
  - navigation.py : scripted spoken directions (or real steps via routing.py)

Optional maps: set OPENROUTESERVICE_API_KEY and pass --origin lon,lat (see routing.py).

Run from this folder:
  python main.py
"""

from __future__ import annotations

import argparse
import os
import threading

from navigation import run_navigation_loop
from routing import build_maps_route
from speech import SpeechController
from vision import VisionConfig, VisionSystem


def _parse_origin_lon_lat(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError('Expected two numbers as "longitude,latitude".')
    lon, lat = float(parts[0]), float(parts[1])
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError("Longitude or latitude out of range.")
    return lon, lat


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistive navigation prototype (webcam + YOLO + TTS).")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--nav-interval",
        type=float,
        default=5.0,
        help="Seconds between spoken navigation prompts (default: 5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics weights path or name (default: yolov8n.pt). Try yolov8s.pt for higher accuracy.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum detection confidence (default: 0.35). Lower recalls more; higher reduces false positives.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold for predict() (default: 0.5).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square inference size (default: 640). 960 or 1280 can help small objects at a FPS cost.",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=2,
        help="Consecutive frames with a hazard before speaking (default: 2). Increase to reduce flicker.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable test-time augmentation (slower, may improve hard frames).",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 inference on CUDA (default: FP16 on CUDA when available).",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        help='Walking start as "longitude,latitude" (use with OPENROUTESERVICE_API_KEY for real directions).',
    )
    args = parser.parse_args()

    print("=== Assistive Navigation Prototype ===")
    print("Loading speech engine...")
    speech = SpeechController()
    speech.start()

    destination = input("Enter destination (place name or address; mock text if not using maps): ").strip()
    if not destination:
        destination = "the lobby"
        print(f"[main] No destination entered; using default: {destination}")

    nav_route: list[str] | None = None
    repeat_nav = True
    ors_key = (os.environ.get("OPENROUTESERVICE_API_KEY") or "").strip()
    if args.origin:
        if not ors_key:
            print("[main] --origin set but OPENROUTESERVICE_API_KEY is unset; using mock navigation.")
        else:
            try:
                start = _parse_origin_lon_lat(args.origin)
                nav_route = build_maps_route(ors_key, start, destination)
                repeat_nav = False
                print(f"[main] OpenRouteService: {len(nav_route)} spoken steps loaded.")
            except Exception as exc:
                print(f"[main] Maps routing failed ({exc}); using mock navigation.")
    elif ors_key:
        print("[main] OPENROUTESERVICE_API_KEY is set; add --origin lon,lat for walking directions from ORS.")

    stop_event = threading.Event()

    def urgent_say(msg: str) -> None:
        speech.speak_urgent(msg)
        print(f"[vision->speech] URGENT: {msg}")

    vcfg = VisionConfig(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        confirm_frames=args.confirm_frames,
        augment=args.augment,
        half=False if args.no_half else None,
    )
    vision = VisionSystem(on_warning=urgent_say, config=vcfg)

    print(f"[main] Model will track labels: {sorted(vision.active_class_labels())}")
    print("[main] Navigation thread starting (" + ("map route" if nav_route else "mock directions") + ").")

    def nav_speak(msg: str) -> None:
        speech.speak_normal(msg)
        print(f"[nav->speech] {msg}")

    nav_thread = threading.Thread(
        target=run_navigation_loop,
        kwargs={
            "destination": destination,
            "speak": nav_speak,
            "interval_seconds": args.nav_interval,
            "stop_event": stop_event,
            "route": nav_route,
            "repeat_route": repeat_nav,
        },
        name="Navigation",
        daemon=True,
    )
    nav_thread.start()

    print("[main] Starting vision loop (opens a small preview window).")
    try:
        vision.run_forever(camera_index=args.camera, stop_event=stop_event)
    except KeyboardInterrupt:
        print("[main] Keyboard interrupt — shutting down.")
    finally:
        stop_event.set()
        speech.stop()
        nav_thread.join(timeout=2.0)
        print("[main] Goodbye.")


if __name__ == "__main__":
    main()
