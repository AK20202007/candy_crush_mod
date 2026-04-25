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
from agentic_layer.config import DEFAULT_PROFILE_NAME, choose, load_profile, load_profiles
from agentic_layer.database import MongoTelemetryStore
from indoor_routing import IndoorGraph, find_node_by_name, load_graph_from_json
from navigation import run_navigation_loop
from routing import build_maps_route
from speech import SpeechController
from voice_input import DestinationCaptureConfig, VoiceInputError, capture_destination_by_voice, capture_destination_with_codeword
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
    profile_names = sorted(load_profiles())
    parser = argparse.ArgumentParser(description="Assistive navigation prototype (agentic webcam + YOLO + TTS).")
    parser.add_argument("--profile", choices=profile_names, default=DEFAULT_PROFILE_NAME, help="Runtime profile from config/agentic_profiles.json.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0).")
    parser.add_argument("--nav-interval", type=float, default=5.0, help="Seconds between navigation prompts (default: 5).")
    parser.add_argument("--model", type=str, default=None, help="Override profile Ultralytics weights path or name.")
    parser.add_argument("--conf", type=float, default=None, help="Override profile minimum detection confidence.")
    parser.add_argument("--iou", type=float, default=None, help="Override profile NMS IoU threshold for predict().")
    parser.add_argument("--imgsz", type=int, default=None, help="Override profile square inference size.")
    parser.add_argument("--confirm-frames", type=int, default=None, help="Override profile consecutive frames before warning.")
    parser.add_argument("--augment", action="store_true", help="Enable test-time augmentation.")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 inference on CUDA.")
    parser.add_argument("--origin", type=str, default=None, help='Walking start as "longitude,latitude" for ORS routing.')
    parser.add_argument("--destination", type=str, default=None, help="Destination text for non-interactive runs; skips voice capture.")
    parser.add_argument("--typed-destination", action="store_true", help="Use the old terminal destination prompt instead of voice capture.")
    parser.add_argument("--indoor-layout", type=str, default=None, help="Path to indoor layout JSON file for indoor routing.")
    parser.add_argument("--indoor-start-node", type=str, default="entrance", help="Starting node ID for indoor routing (default: entrance).")
    parser.add_argument("--voice-timeout-s", type=float, default=8.0, help="Seconds to listen for spoken destination.")
    parser.add_argument("--voice-attempts", type=int, default=2, help="How many times to retry spoken destination capture.")
    parser.add_argument("--voice-locale", type=str, default="en_US", help="Speech recognizer locale for destination capture.")
    parser.add_argument("--codeword", type=str, default="navigate", help="Word that starts continuous destination capture.")
    parser.add_argument("--stop-word", type=str, default="stop", help="Word that ends continuous destination capture.")
    parser.add_argument("--no-codeword", action="store_true", help="Use one-shot spoken destination capture instead of codeword start/stop capture.")
    parser.add_argument("--allow-network-speech", action="store_true", help="Allow Apple Speech to use network recognition if on-device recognition is unavailable.")

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
    parser.add_argument("--location-type", type=str, default=None, help="Override profile scene hint: hallway, room, sidewalk, street_crossing, etc.")
    parser.add_argument("--visual-confidence", type=float, default=None, help="Override profile visual confidence hint.")
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=None,
        help="Override profile bbox distance multiplier. Increase if estimated distances are too short.",
    )
    parser.add_argument(
        "--obstacle-area-ratio",
        type=float,
        default=None,
        help="Override profile area threshold for close obstacle warnings.",
    )
    parser.add_argument(
        "--person-center-radius",
        type=float,
        default=None,
        help="Override profile normalized center radius for person warnings.",
    )
    parser.add_argument(
        "--warning-cooldown-s",
        type=float,
        default=None,
        help="Override profile urgent warning cooldown used by the vision-owned router.",
    )
    parser.add_argument(
        "--agent-repeat-ms",
        type=int,
        default=None,
        help="Override profile minimum repeat interval for non-critical agent speech.",
    )
    parser.add_argument(
        "--agent-urgent-repeat-ms",
        type=int,
        default=None,
        help="Override profile minimum repeat interval for repeated urgent safety speech.",
    )
    return parser


def _get_destination(args: argparse.Namespace, speech: SpeechController) -> str:
    if args.destination:
        destination = args.destination.strip()
        print(f"[main] Destination from --destination: {destination}")
        return destination

    if args.typed_destination:
        destination = input("Enter destination (place name or address; mock text if not using maps): ").strip()
        if destination:
            return destination
        destination = "the lobby"
        print(f"[main] No destination entered; using default: {destination}")
        return destination

    print("[main] Listening for destination. Speak after the prompt.")
    try:
        voice_config = DestinationCaptureConfig(
            timeout_s=args.voice_timeout_s,
            locale=args.voice_locale,
            prefer_on_device=not args.allow_network_speech,
            attempts=args.voice_attempts,
            codeword=args.codeword,
            stop_word=args.stop_word,
        )
        if args.no_codeword:
            destination = capture_destination_by_voice(speak=speech.speak_normal, config=voice_config)
        else:
            destination = capture_destination_with_codeword(speak=speech.speak_normal, config=voice_config)
        print(f"[main] Heard destination: {destination}")
        return destination
    except VoiceInputError as exc:
        destination = "the lobby"
        print(f"[main] Voice destination capture failed ({exc}); using default: {destination}")
        speech.speak_normal(f"I could not hear a destination. Using {destination}.")
        return destination


def main() -> None:
    args = build_parser().parse_args()
    profile = load_profile(args.profile)

    print("=== Assistive Navigation Prototype + Agentic Layer ===")
    print(f"[main] Runtime profile: {profile.name} — {profile.description}")
    print("Loading speech engine...")
    speech = SpeechController()
    speech.start()

    destination = _get_destination(args, speech)

    nav_route: Optional[list[str]] = None
    repeat_nav = True
    ors_key = (os.environ.get("OPENROUTESERVICE_API_KEY") or "").strip()
    
    # Indoor routing
    indoor_graph: Optional[IndoorGraph] = None
    indoor_route_steps: Optional[list[str]] = None
    if args.indoor_layout:
        try:
            layout_path = Path(args.indoor_layout)
            if not layout_path.exists():
                layout_path = Path(__file__).resolve().parents[0] / "config" / args.indoor_layout
            indoor_graph = load_graph_from_json(layout_path)
            print(f"[main] Loaded indoor layout from {layout_path}")
            
            # Find destination node
            dest_node_id = find_node_by_name(indoor_graph, destination)
            if not dest_node_id:
                print(f"[main] Could not find node matching '{destination}' in indoor layout")
                dest_node_id = find_node_by_name(indoor_graph, "lobby") or args.indoor_start_node
            
            start_node_id = args.indoor_start_node
            if dest_node_id:
                route = indoor_graph.find_route(start_node_id, dest_node_id)
                indoor_route_steps = [step.instruction for step in route.steps]
                print(f"[main] Indoor route: {len(indoor_route_steps)} steps, {route.total_distance_m:.1f}m total")
                for i, step in enumerate(indoor_route_steps, 1):
                    print(f"  {i}. {step}")
            else:
                print("[main] Using mock navigation (no destination node found)")
        except Exception as exc:
            print(f"[main] Indoor routing failed ({exc}); using mock navigation.")
    
    # Outdoor routing (ORS)
    if args.origin and not indoor_route_steps:
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
    telemetry_store = MongoTelemetryStore.from_env()
    if telemetry_store.enabled:
        print("[main] MongoDB telemetry enabled.")
    else:
        print(f"[main] MongoDB telemetry disabled: {telemetry_store.unavailable_reason}")

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
            "route": indoor_route_steps if indoor_route_steps else nav_route,
            "repeat_route": repeat_nav,
        },
        name="Navigation",
        daemon=True,
    )
    nav_thread.start()

    router = AgenticNavigationRouter(
        min_repeat_interval_ms=choose(args.agent_repeat_ms, profile.agent_repeat_ms),
        min_urgent_repeat_interval_ms=choose(args.agent_urgent_repeat_ms, profile.agent_urgent_repeat_ms),
    )

    def user_state() -> UserState:
        return UserState(query=args.query, target=args.target, mode=args.mode, verbosity="minimal")

    def motion_state() -> MotionState:
        return MotionState(is_moving=not args.standing_still)

    def handle_decision(decision: AgentDecision) -> None:
        print(
            f"[agent->speech] action={decision.action.value} priority={decision.priority} "
            f"haptic={decision.haptic.value}: {decision.message}"
        )
        # Print all agent candidates for debugging
        if "candidates" in decision.debug:
            print(f"[debug] All candidates:")
            for i, c in enumerate(decision.debug["candidates"], 1):
                print(f"  {i}. {c['action']} (priority={c['priority']}) by {c['agents']}: {c['message']}")
        if decision.action == AgentAction.WARN or decision.priority >= 80:
            speech.speak_urgent(decision.message)
        else:
            speech.speak_normal(decision.message)

    vcfg = VisionConfig(
        model_path=choose(args.model, profile.model_path),
        obstacle_area_ratio=choose(args.obstacle_area_ratio, profile.obstacle_area_ratio),
        person_center_radius=choose(args.person_center_radius, profile.person_center_radius),
        warning_cooldown_s=choose(args.warning_cooldown_s, profile.warning_cooldown_s),
        conf=choose(args.conf, profile.conf),
        iou=choose(args.iou, profile.iou),
        imgsz=choose(args.imgsz, profile.imgsz),
        confirm_frames=choose(args.confirm_frames, profile.confirm_frames),
        augment=args.augment,
        half=False if args.no_half else None,
        assume_moving=not args.standing_still,
        location_type=choose(args.location_type, profile.location_type),
        visual_confidence=choose(args.visual_confidence, profile.visual_confidence),
        distance_scale=choose(args.distance_scale, profile.distance_scale),
    )

    vision = VisionSystem(
        on_decision=handle_decision,
        config=vcfg,
        router=router,
        route_provider=route_context.get,
        user_provider=user_state,
        motion_provider=motion_state,
        on_frame_decision=telemetry_store.record_decision if telemetry_store.enabled else None,
    )

    # Start hardware sensor listeners if available
    vision._motion_fall_detector.start(lambda d: vision.handle_sensor_fall(f"DRAMATIC elevation change of {d:.1f}m", priority=99))
    vision._smv_fall_detector.start(lambda s, x, y: vision.handle_sensor_fall(f"Impact force of {s:.1f} m/s^2", priority=10))

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
