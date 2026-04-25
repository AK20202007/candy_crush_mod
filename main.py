#!/usr/bin/env python3
"""
Assistive navigation prototype: webcam + YOLO warnings + mock directions.

Combines:
  - vision.py  : person / obstacle cues
  - speech.py  : pyttsx3 with urgent vs normal priority
  - navigation.py : scripted spoken directions

Run from this folder:
  python main.py
"""

from __future__ import annotations

import argparse
import threading
import time

from navigation import run_navigation_loop
from speech import SpeechController
from vision import VisionSystem


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
    args = parser.parse_args()

    print("=== Assistive Navigation Prototype ===")
    print("Loading speech engine...")
    speech = SpeechController()
    speech.start()

    destination = input("Enter a destination (text, mock directions only): ").strip()
    if not destination:
        destination = "the lobby"
        print(f"[main] No destination entered; using default: {destination}")

    stop_event = threading.Event()

    def urgent_say(msg: str) -> None:
        speech.speak_urgent(msg)
        print(f"[vision->speech] URGENT: {msg}")

    vision = VisionSystem(on_warning=urgent_say)

    print(f"[main] Model will track labels: {sorted(vision.active_class_labels())}")
    print("[main] Navigation thread starting (mock directions).")

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
