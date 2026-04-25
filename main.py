#!/usr/bin/env python3
"""
Assistive Navigation System

A user-friendly interface for blind/low-vision navigation with:
- Intelligent speech timing (no spam)
- Context-aware feedback
- Clean status indicators
- User-configurable preferences
- Conservative safety policy enforcement

Usage:
    python main.py                                  # Interactive mode
    python main.py --destination "Library"          # Direct navigation
    python main.py --origin "-118.448,34.070"       # Map-backed route state
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from agentic_layer import AgenticNavigationRouter, MotionState, RouteState, UserState
from agentic_layer.config import DEFAULT_PROFILE_NAME, choose, load_profiles
from agentic_layer.models import FrameContext, SceneState
from navigation import run_navigation_loop
from navigation_interface import NavigationInterface
from routing import build_maps_route, google_walking_directions
from user_interface import UserPreferences
from vision import VisionConfig, VisionSystem

# Try to import optional components
try:
    from voice_input import DestinationCaptureConfig, VoiceInputError, capture_destination_by_voice
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False
    print("[system] Voice input not available, using text input only")


def _parse_origin_lon_lat(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError('Expected two numbers as "longitude,latitude".')
    lon, lat = float(parts[0]), float(parts[1])
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError("Longitude or latitude out of range.")
    return lon, lat


class SharedRouteContext:
    """Thread-safe route state shared between navigation prompts and vision."""

    def __init__(self, destination: str) -> None:
        self._lock = threading.Lock()
        self._state = RouteState(active=False, destination=destination)

    def update_instruction(self, message: str) -> None:
        with self._lock:
            self._state = RouteState(
                active=True,
                destination=self._state.destination,
                next_instruction=message.strip(),
                next_turn_distance_m=self._state.next_turn_distance_m,
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


class NavigationApp:
    """Main application class for assistive navigation."""

    def __init__(self):
        self.interface: Optional[NavigationInterface] = None
        self.vision: Optional[VisionSystem] = None
        self.stop_event = threading.Event()
        self.route_context: Optional[SharedRouteContext] = None
        self._nav_stop_event: Optional[threading.Event] = None
        self._nav_thread: Optional[threading.Thread] = None
        self._last_frame_context: Optional[FrameContext] = None
        self._active_args: Optional[argparse.Namespace] = None
        self.is_running = False

        # Status tracking
        self._start_time: Optional[float] = None
        self._frames_processed = 0
        self._last_fps_print = 0

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            print("\n[system] Shutdown signal received, stopping gracefully...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def load_preferences(self, args) -> UserPreferences:
        """Load user preferences from args and config file."""
        prefs = UserPreferences()

        # Override from command line
        if args.verbosity:
            prefs.verbosity = args.verbosity
        if args.speech_rate:
            prefs.speech_rate = max(0.5, min(1.5, args.speech_rate))
        if args.sensitivity:
            prefs.warning_sensitivity = args.sensitivity

        # Try to load from config file
        config_path = Path.home() / ".assistive_nav" / "preferences.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    for key, value in config.items():
                        if hasattr(prefs, key):
                            setattr(prefs, key, value)
                print(f"[config] Loaded preferences from {config_path}")
            except Exception as e:
                print(f"[config] Error loading preferences: {e}")

        return prefs

    def save_preferences(self, prefs: UserPreferences) -> None:
        """Save user preferences to config file."""
        config_dir = Path.home() / ".assistive_nav"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "preferences.json"

        try:
            with open(config_path, 'w') as f:
                json.dump(prefs.__dict__, f, indent=2)
            print(f"[config] Saved preferences to {config_path}")
        except Exception as e:
            print(f"[config] Error saving preferences: {e}")

    def initialize(self, args) -> bool:
        """Initialize all components."""
        try:
            print("=" * 60)
            print("ASSISTIVE NAVIGATION SYSTEM")
            print("Controlled Prototype Interface")
            print("=" * 60)
            self._active_args = args

            # Load preferences
            prefs = self.load_preferences(args)
            print(f"[config] Verbosity: {prefs.verbosity}")
            print(f"[config] Sensitivity: {prefs.warning_sensitivity}")
            print(f"[config] Speech rate: {prefs.speech_rate}x")

            # Create router and interface
            profiles = load_profiles()
            profile = profiles[args.profile]
            router = AgenticNavigationRouter(
                min_repeat_interval_ms=profile.agent_repeat_ms,
                min_urgent_repeat_interval_ms=profile.agent_urgent_repeat_ms,
            )
            self.interface = NavigationInterface(router, prefs)

            # Setup vision system
            vision_config = VisionConfig(
                model_path=choose(args.model, profile.model_path),
                obstacle_area_ratio=profile.obstacle_area_ratio,
                person_center_radius=profile.person_center_radius,
                warning_cooldown_s=profile.warning_cooldown_s,
                confirm_frames=choose(args.confirm_frames, profile.confirm_frames),
                imgsz=choose(args.imgsz, profile.imgsz),
                conf=choose(args.conf, profile.conf),
                iou=choose(args.iou, profile.iou),
                augment=args.augment,
                half=False if args.no_half else None,
                assume_moving=not args.standing_still,
                location_type=choose(args.location_type, profile.location_type),
                visual_confidence=choose(args.visual_confidence, profile.visual_confidence),
                distance_scale=choose(args.distance_scale, profile.distance_scale),
                camera_mount=args.camera_mount,
                enable_door_assist=not args.disable_door_assist,
            )

            print(f"[config] Runtime profile: {profile.name} - {profile.description}")
            if vision_config.is_head_mounted:
                print(
                    f"[config] Head-mounted camera mode: surface_y={vision_config.surface_y_start:.2f}, "
                    f"distance_scale={vision_config.effective_distance_scale:.2f}"
                )
            if not vision_config.enable_door_assist:
                print("[config] Door assist disabled.")

            def handle_decision(decision):
                """Callback for vision system decisions."""
                if self.interface and not self.stop_event.is_set():
                    ctx = self._last_frame_context or self._fallback_frame_context()
                    self.interface.process_decision(decision, ctx)

            def handle_frame_decision(ctx: FrameContext, _decision) -> None:
                self._last_frame_context = ctx
                self._frames_processed += 1

            self.vision = VisionSystem(
                config=vision_config,
                on_decision=handle_decision,
                router=router,
                route_provider=self._route_state,
                user_provider=self._user_state,
                motion_provider=self._motion_state,
                on_frame_decision=handle_frame_decision,
            )

            # Start hardware sensor listeners if available
            self.vision._motion_fall_detector.start(lambda d: self.vision.handle_sensor_fall(f"DRAMATIC elevation change of {d:.1f}m", priority=99))
            self.vision._smv_fall_detector.start(lambda s, x, y: self.vision.handle_sensor_fall(f"Impact force of {s:.1f} m/s^2", priority=10))

            # Start the interface
            self.interface.start()

            # Welcome message
            self.interface.speak_info(
                "Welcome to assistive navigation. "
                "System is ready. Set a destination to begin navigation."
            )

            return True

        except Exception as e:
            print(f"[error] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _route_state(self) -> RouteState:
        if self.route_context is None:
            return RouteState(active=False)
        return self.route_context.get()

    def _user_state(self) -> UserState:
        args = self._active_args
        if args is None:
            return UserState()
        return UserState(query=args.query, target=args.target, mode=args.mode, verbosity=args.verbosity or "normal")

    def _motion_state(self) -> MotionState:
        args = self._active_args
        return MotionState(is_moving=not bool(args and args.standing_still))

    def _fallback_frame_context(self) -> FrameContext:
        args = self._active_args
        location_type = args.location_type if args and args.location_type else "unknown"
        return FrameContext(
            timestamp_ms=int(time.time() * 1000),
            frame_id=str(time.time()),
            route=self._route_state(),
            motion=self._motion_state(),
            scene=SceneState(location_type=location_type),
            user=self._user_state(),
        )

    def get_destination(self, args) -> Optional[str]:
        """Get destination from user via voice or text input."""
        if args.destination:
            return args.destination

        if args.typed_destination or not VOICE_INPUT_AVAILABLE:
            # Text input
            print("\n[system] Enter destination (or 'quit' to exit):")
            try:
                destination = input("> ").strip()
                if destination.lower() in ['quit', 'exit', 'q']:
                    return None
                return destination if destination else None
            except EOFError:
                return None
        else:
            # Voice input
            print("\n[system] Listening for destination...")
            print("[system] Say your destination clearly after the beep")

            try:
                config = DestinationCaptureConfig(
                    timeout_s=args.voice_timeout,
                    locale=args.voice_locale
                )
                destination = capture_destination_by_voice(config)
                return destination
            except VoiceInputError as e:
                print(f"[system] Voice input failed: {e}")
                print("[system] Falling back to text input")
                return self.get_destination(argparse.Namespace(
                    destination=None,
                    typed_destination=True,
                    voice_timeout=args.voice_timeout,
                    voice_locale=args.voice_locale
                ))

    def _build_route(self, args, destination: str) -> tuple[Optional[list[str]], bool, str]:
        provider = args.route_provider
        if provider == "mock":
            return None, True, "mock"

        if not args.origin:
            print("[route] No --origin provided; using mock route prompts.")
            return None, True, "mock"

        start = _parse_origin_lon_lat(args.origin)
        google_key = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
        ors_key = (os.environ.get("OPENROUTESERVICE_API_KEY") or "").strip()

        if provider in {"auto", "google"} and google_key:
            steps = google_walking_directions(google_key, start, destination)
            return steps, False, "google"

        if provider == "google":
            print("[route] GOOGLE_MAPS_API_KEY is unset; falling back to mock route prompts.")
            return None, True, "mock"

        if provider in {"auto", "ors"} and ors_key:
            steps = build_maps_route(ors_key, start, destination)
            return steps, False, "openrouteservice"

        if provider == "ors":
            print("[route] OPENROUTESERVICE_API_KEY is unset; falling back to mock route prompts.")
        else:
            print("[route] No map API key found; using mock route prompts.")
        return None, True, "mock"

    def _start_route(self, args, destination: str) -> None:
        self._stop_route()
        self.route_context = SharedRouteContext(destination)
        self._nav_stop_event = threading.Event()

        try:
            route, repeat_route, source = self._build_route(args, destination)
        except Exception as exc:
            print(f"[route] Route lookup failed ({exc}); using mock route prompts.")
            route, repeat_route, source = None, True, "mock"

        def route_update(message: str) -> None:
            if self.route_context is None:
                return
            self.route_context.update_instruction(message)
            print(f"[route->{source}] {message}")

        self._nav_thread = threading.Thread(
            target=run_navigation_loop,
            kwargs={
                "destination": destination,
                "speak": route_update,
                "interval_seconds": args.nav_interval,
                "stop_event": self._nav_stop_event,
                "route": route,
                "repeat_route": repeat_route,
            },
            name="NavigationRoute",
            daemon=True,
        )
        self._nav_thread.start()

    def _stop_route(self) -> None:
        if self._nav_stop_event is not None:
            self._nav_stop_event.set()
        if self._nav_thread is not None:
            self._nav_thread.join(timeout=2.0)
        self._nav_stop_event = None
        self._nav_thread = None
        self.route_context = None

    def run(self, args) -> int:
        """Main application loop."""
        if not self.initialize(args):
            return 1

        self.setup_signal_handlers()
        self.is_running = True
        self._start_time = time.time()

        try:
            # Main loop
            while not self.stop_event.is_set():
                # Get destination
                destination = self.get_destination(args)

                if destination is None:
                    print("[system] No destination provided, exiting")
                    break

                # Set destination and start navigation
                print(f"\n[system] Starting navigation to: {destination}")
                self._start_route(args, destination)
                self.interface.set_destination(destination)

                # Run vision system
                print("[system] Starting vision system...")
                print("[system] Press Ctrl+C to stop, or say 'stop'")

                try:
                    self.vision.run_forever(
                        camera_index=args.camera,
                        stop_event=self.stop_event
                    )
                except KeyboardInterrupt:
                    print("\n[system] Interrupted by user")
                    break

                self._stop_route()

                # Clear destination and ask for new one
                self.interface.clear_destination()
                print("\n[system] Navigation ended. Set a new destination or Ctrl+C to quit")

                # Reset args.destination so we prompt again
                args.destination = None

        except Exception as e:
            print(f"\n[error] Runtime error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        finally:
            self.stop()

        return 0

    def stop(self):
        """Stop all components gracefully."""
        if not self.is_running:
            return

        print("\n[system] Stopping components...")
        self.stop_event.set()
        self._stop_route()
        self.is_running = False

        if self.interface:
            self.interface.speak_info("Shutting down. Goodbye.")
            self.interface.stop()

        print("[system] Shutdown complete")

        # Print statistics
        if self._start_time:
            runtime = time.time() - self._start_time
            print(f"\n[stats] Runtime: {runtime:.1f} seconds")
            if self.interface:
                status = self.interface.get_status()
                print(f"[stats] Alerts spoken: {status['alert_stats']['total_alerts']}")
                print(f"[stats] Frames processed: {self._frames_processed}")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    profile_names = sorted(load_profiles())

    parser = argparse.ArgumentParser(
        description="Controlled assistive navigation prototype for blind/low-vision users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode with voice input
  %(prog)s --destination "Library"  # Navigate to Library
  %(prog)s --origin "-118.448,34.070" --destination "Rieber Hall UCLA"
  %(prog)s --typed-destination      # Use text input instead of voice
  %(prog)s --verbosity minimal      # Reduce speech feedback
  %(prog)s --sensitivity high       # More frequent warnings
        """
    )

    # Basic options
    parser.add_argument(
        "--destination", "-d",
        type=str,
        default=None,
        help="Destination to navigate to (skips voice prompt)"
    )

    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        help='Walking start as "longitude,latitude" for Google/ORS route prompts.',
    )
    parser.add_argument(
        "--route-provider",
        choices=["auto", "google", "ors", "mock"],
        default="auto",
        help="Route source when --origin is present. auto prefers Google, then ORS.",
    )
    parser.add_argument(
        "--nav-interval",
        type=float,
        default=5.0,
        help="Seconds between route prompt updates.",
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--typed-destination", "-t",
        action="store_true",
        help="Use text input instead of voice capture"
    )
    input_group.add_argument(
        "--voice-timeout",
        type=float,
        default=8.0,
        help="Seconds to listen for voice input (default: 8)"
    )
    input_group.add_argument(
        "--voice-locale",
        type=str,
        default="en_US",
        help="Speech recognition locale (default: en_US)"
    )

    # User preferences
    pref_group = parser.add_argument_group("User Preferences")
    pref_group.add_argument(
        "--verbosity", "-v",
        choices=["minimal", "normal", "verbose"],
        default=None,
        help="Speech feedback level"
    )
    pref_group.add_argument(
        "--sensitivity", "-s",
        choices=["low", "normal", "high"],
        default=None,
        help="Warning frequency (high = more warnings)"
    )
    pref_group.add_argument(
        "--speech-rate", "-r",
        type=float,
        default=None,
        help="Speech speed multiplier (0.5 to 1.5)"
    )

    agent_group = parser.add_argument_group("Agent Context")
    agent_group.add_argument("--target", type=str, default=None, help="Object/search target, e.g. door, chair, elevator.")
    agent_group.add_argument("--query", type=str, default=None, help='User-style query, e.g. "where is the door?"')
    agent_group.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "orientation", "wayfinding", "object_search", "ocr"],
        help="Agentic mode hint.",
    )
    agent_group.add_argument("--standing-still", action="store_true", help="Tell the agent the user is not walking.")
    agent_group.add_argument("--location-type", type=str, default=None, help="Scene hint: hallway, room, sidewalk, street_crossing, etc.")
    agent_group.add_argument("--visual-confidence", type=float, default=None, help="Override profile visual confidence hint.")
    agent_group.add_argument("--distance-scale", type=float, default=None, help="Override profile bbox distance multiplier.")

    # Vision options
    vision_group = parser.add_argument_group("Vision Options")
    vision_group.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    vision_group.add_argument(
        "--profile", "-p",
        choices=profile_names,
        default=DEFAULT_PROFILE_NAME,
        help="Detection profile"
    )
    vision_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override profile Ultralytics weights path or model name.",
    )
    vision_group.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold"
    )
    vision_group.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold"
    )
    vision_group.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size"
    )
    vision_group.add_argument(
        "--confirm-frames",
        type=int,
        default=None,
        help="Override profile consecutive frames before warning.",
    )
    vision_group.add_argument(
        "--no-half",
        action="store_true",
        help="Disable half-precision inference"
    )
    vision_group.add_argument(
        "--augment",
        action="store_true",
        help="Enable test-time augmentation"
    )
    vision_group.add_argument(
        "--camera-mount",
        type=str,
        default="hand",
        choices=["head", "hand"],
        help="Camera mount position: 'head' for glasses/head-mounted, 'hand' for phone (default: hand)."
    )
    vision_group.add_argument(
        "--disable-door-assist",
        action="store_true",
        help="Disable prototype door-handle detection and door-use guidance.",
    )

    return parser


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    app = NavigationApp()
    return app.run(args)


if __name__ == "__main__":
    sys.exit(main())
