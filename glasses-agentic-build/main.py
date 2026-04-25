#!/usr/bin/env python3
"""
Assistive navigation prototype with an agentic routing layer.

Flow:
  webcam + YOLOv8 + distance heuristic
      -> detections/warnings/context
      -> AgenticNavigationRouter
      -> urgent/normal speech

Optional maps: set OPENROUTESERVICE_API_KEY and pass --origin lon,lat.
"""
from __future__ import annotations

import argparse
import os
import threading
from typing import Optional

from agentic_layer import AgentAction, AgentDecision, AgenticNavigationRouter, MotionState, RouteState, UserState
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


class SharedRouteContext:
    """Thread-safe route state shared between navigation thread and vision agent."""

    def __init__(self, destination: str) -> None:
        self._lock = threading.Lock()
        self._state = RouteState(active=False, destination=destination)

    def update_instruction(self, message: str) -> None:
        with self._lock:
            self._state = RouteState(
                active=True,
                destination=self._state.destination,
                next_instruction=message.strip(),
                off_route=False,
            )

    def get(self) -> RouteState:
        with self._lock:
            return RouteState(
                active=self._state.active,
                destination=self._state.destination,
                next_instruction=self._state.next_instruction,
                next_turn_distance_m=self._state.next_turn_distance_m,
                off_route=self._state.off_route,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assistive navigation prototype (agentic webcam + YOLO + TTS).")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0).")
    parser.add_argument("--nav-interval", type=float, default=5.0, help="Seconds between navigation prompts (default: 5).")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics weights path or name.")
    parser.add_argument("--conf", type=float, default=0.35, help="Minimum detection confidence.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold for predict().")
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size.")
    parser.add_argument("--confirm-frames", type=int, default=2, help="Consecutive frames before warning.")
    parser.add_argument("--augment", action="store_true", help="Enable test-time augmentation.")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 inference on CUDA.")
    parser.add_argument("--origin", type=str, default=None, help='Walking start as "longitude,latitude" for ORS routing.')

    # Agentic layer knobs.
    parser.add_argument("--target", type=str, default=None, help="Object/search target, e.g. door, chair, elevator.")
    parser.add_argument("--query", type=str, default=None, help='User-style query, e.g. "where is the door?"')
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "orientation", "wayfinding", "object_search", "ocr"],
        help="Agentic mode hint.",
    )
    parser.add_argument("--standing-still", action="store_true", help="Tell the agent the user is not walking.")
    parser.add_argument("--location-type", type=str, default="unknown", help="Scene hint: hallway, room, sidewalk, street_crossing, etc.")
    parser.add_argument("--visual-confidence", type=float, default=0.75, help="Demo visual confidence hint.")
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=1.0,
        help="Calibration multiplier for bbox distance heuristic. Increase if estimated distances are too short.",
    )
    parser.add_argument(
        "--agent-repeat-ms",
        type=int,
        default=4500,
        help="Minimum repeat interval for non-critical agent speech.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("=== Assistive Navigation Prototype + Agentic Layer ===")
    print("Loading speech engine...")
    speech = SpeechController()
    speech.start()

    destination = input("Enter destination (place name or address; mock text if not using maps): ").strip()
    if not destination:
        destination = "the lobby"
        print(f"[main] No destination entered; using default: {destination}")

    nav_route: Optional[list[str]] = None
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
    route_context = SharedRouteContext(destination=destination)

    def route_update(msg: str) -> None:
        # Do not speak immediately. The agent decides whether route guidance or
        # a safety/object warning should be spoken in the current frame.
        route_context.update_instruction(msg)
        print(f"[nav->agent] {msg}")

    nav_thread = threading.Thread(
        target=run_navigation_loop,
        kwargs={
            "destination": destination,
            "speak": route_update,
            "interval_seconds": args.nav_interval,
            "stop_event": stop_event,
            "route": nav_route,
            "repeat_route": repeat_nav,
        },
        name="Navigation",
        daemon=True,
    )
    nav_thread.start()

    router = AgenticNavigationRouter(min_repeat_interval_ms=args.agent_repeat_ms)

    def user_state() -> UserState:
        return UserState(query=args.query, target=args.target, mode=args.mode, verbosity="minimal")

    def motion_state() -> MotionState:
        return MotionState(is_moving=not args.standing_still)

    def handle_decision(decision: AgentDecision) -> None:
        print(
            f"[agent->speech] action={decision.action.value} priority={decision.priority} "
            f"haptic={decision.haptic.value}: {decision.message}"
        )
        if decision.action == AgentAction.WARN or decision.priority >= 80:
            speech.speak_urgent(decision.message)
        else:
            speech.speak_normal(decision.message)

    vcfg = VisionConfig(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        confirm_frames=args.confirm_frames,
        augment=args.augment,
        half=False if args.no_half else None,
        assume_moving=not args.standing_still,
        location_type=args.location_type,
        visual_confidence=args.visual_confidence,
        distance_scale=args.distance_scale,
    )

    vision = VisionSystem(
        on_decision=handle_decision,
        config=vcfg,
        router=router,
        route_provider=route_context.get,
        user_provider=user_state,
        motion_provider=motion_state,
    )

    print(f"[main] Model will track labels: {sorted(vision.active_class_labels()) or 'all'}")
    print("[main] Agentic router is active. Safety warnings override route guidance.")
    if args.target or args.query:
        print(f"[main] User intent hint: target={args.target!r}, query={args.query!r}, mode={args.mode!r}")

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
