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
    parser = argparse.ArgumentParser(description="Seizure and Fall Detection (Strobe + Sensor Fusion).")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--strobe-freq",
        type=float,
        default=3.0,
        help="Minimum frequency (Hz) to trigger strobe warning (default: 3.0).",
    )
    parser.add_argument(
        "--emergency-contact",
        type=str,
        default="Emergency Services",
        help="Name of the contact to alert in an emergency.",
    )
    args = parser.parse_args()

    print("=== Seizure & Fall Detection System ===")
    print("Loading speech engine (Emergency Only)...")
    speech = SpeechController()
    speech.start()

    stop_event = threading.Event()

    def alert_callback(msg: str) -> None:
        # We only speak critical warnings now.
        speech.speak_urgent(msg)
        print(f"[ALERT] {msg}")

    vcfg = VisionConfig(
        strobe_freq=args.strobe_freq,
        emergency_contact=args.emergency_contact,
        # We can keep vision-based fall detection active as a supplement
        confirm_frames=2,
    )
    vision = VisionSystem(
        on_warning=alert_callback,
        on_emergency=speech.speak_emergency,
        config=vcfg
    )

    # Start hardware sensor listeners if available
    vision._motion_fall_detector.start(lambda d: vision.handle_sensor_fall(f"Altitude drop of {d:.1f}m"))
    vision._smv_fall_detector.start(lambda s, x, y: vision.handle_sensor_fall(f"Impact force of {s:.1f} m/s^2"))

    print("[main] System active. Monitoring for strobe lights and falls...")
    try:
        vision.run_forever(camera_index=args.camera, stop_event=stop_event)
    except KeyboardInterrupt:
        print("[main] Shutdown requested.")
    finally:
        stop_event.set()
        speech.stop()
        print("[main] System stopped.")


if __name__ == "__main__":
    main()
