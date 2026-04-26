#!/usr/bin/env python3
"""
Production-Ready Assistive Navigation System

A user-friendly interface for blind/low-vision navigation with:
- Intelligent speech timing (no spam)
- Context-aware feedback
- Clean status indicators
- User-configurable preferences
- Professional-grade error handling

Usage:
    python main_improved.py                    # Interactive mode
    python main_improved.py --destination "Library"  # Direct navigation
    python main_improved.py --verbosity minimal    # Minimal feedback
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
import urllib3

# Suppress unverified HTTPS warnings caused by the macOS SSL workaround
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load .env file if it exists (API keys)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# Import our improved components
from agentic_layer import AgenticNavigationRouter
from agentic_layer.config import DEFAULT_PROFILE_NAME, load_profiles
from agentic_layer.models import RouteState
from navigation_interface import NavigationInterface
from speech_controller import IntelligentSpeechController
from user_interface import UserMode, UserPreferences
from vision import VisionConfig, VisionSystem

from destination_verifier import search_destination, format_confirmation_message, get_text_confirmation, get_voice_confirmation
from laptop_route_session import LaptopRouteSession

# Try to import optional components
try:
    from voice_input import DestinationCaptureConfig, VoiceInputError, capture_destination_by_voice
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False
    print("[system] Voice input not available, using text input only")


class NavigationApp:
    """Main application class for assistive navigation."""
    
    def __init__(self):
        self.interface: Optional[NavigationInterface] = None
        self.vision: Optional[VisionSystem] = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.route_state = RouteState(active=False)
        self.route_session: Optional[LaptopRouteSession] = None
        
        # Status tracking
        self._start_time: Optional[float] = None
        self._frames_processed = 0
        self._last_fps_print = 0
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            print("\n[system] Shutdown signal received, stopping gracefully...")
            self.stop_event.set()
        
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
            print("Production-Ready Interface v2.0")
            print("=" * 60)
            
            # Load preferences
            prefs = self.load_preferences(args)
            print(f"[config] Verbosity: {prefs.verbosity}")
            print(f"[config] Sensitivity: {prefs.warning_sensitivity}")
            print(f"[config] Speech rate: {prefs.speech_rate}x")
            print(f"[config] Speech engine: {args.speech_engine}")
            
            # Create router and interface
            router = AgenticNavigationRouter()
            self.interface = NavigationInterface(router, prefs, speech_engine=args.speech_engine)
            self.route_session = LaptopRouteSession(
                self.route_state,
                Path(__file__).parent,
                arrival_radius_m=args.arrival_radius_m,
                poll_seconds=args.route_poll_seconds,
                on_instruction=self._speak_route_instruction,
            )
            
            # Setup vision system
            vision_config = VisionConfig(
                imgsz=args.imgsz or 416,
                conf=args.conf or 0.35,
                iou=args.iou or 0.5,
                augment=args.augment,
                half=not args.no_half,
                camera_mount=args.camera_mount,
                target_fps=args.target_fps,
                show_preview=args.preview,
                surface_every_n_frames=args.surface_every_n,
                door_parts_model_path=args.door_parts_model,
                door_parts_yolov5_dir=args.door_parts_yolov5_dir,
                door_parts_conf=args.door_parts_conf,
            )
            
            def handle_frame_decision(ctx, decision):
                """Callback for per-frame decisions with the real route context."""
                if self.interface and not self.stop_event.is_set():
                    return self.interface.process_decision(decision, ctx)
                return False
            
            self.vision = VisionSystem(
                config=vision_config,
                on_decision=None,
                route_provider=lambda: self.route_state,
                on_frame_decision=handle_frame_decision,
            )
            
            # Start hardware sensor listeners if available
            self.vision._motion_fall_detector.start(lambda d: self.vision.handle_sensor_fall(f"DRAMATIC elevation change of {d:.1f}m", priority=99))
            self.vision._smv_fall_detector.start(lambda s, x, y: self.vision.handle_sensor_fall(f"Impact force of {s:.1f} m/s^2", priority=10))
            
            # Start the interface (no speech yet — wait for destination confirmation)
            self.interface.start()
            
            return True
            
        except Exception as e:
            print(f"[error] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _listen_for_destination(self, prompt: str = "Where would you like to go?", timeout: float = 6.0) -> Optional[str]:
        """
        Speak a prompt then record audio and transcribe via ElevenLabs STT.
        Returns the transcribed text or None on failure.
        """
        import requests
        try:
            import speech_recognition as sr
        except ImportError:
            print("[system] SpeechRecognition module not available, using text input")
            print(f"[system] {prompt}")
            try:
                return input("> ").strip() or None
            except EOFError:
                return None

        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not api_key:
            print("[system] No ElevenLabs API key — falling back to text input")
            print(f"[system] {prompt}")
            try:
                return input("> ").strip() or None
            except EOFError:
                return None

        # Speak the prompt
        if self.interface:
            self.interface.speak_info(prompt, force=True)
            timeout = time.time() + 15.0
            if hasattr(self.interface.speech, 'is_idle'):
                while not self.interface.speech.is_idle() and time.time() < timeout:
                    time.sleep(0.1)
            else:
                time.sleep(4)
            self.interface.speech.clear_queues()

        time.sleep(0.5)
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 150
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.0

        print(f"\n[system] 🎤 Listening... (speak now)")

        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                wav_data = audio.get_wav_data()
        except Exception as e:
            print(f"[system] Microphone error: {e}")
            return None

        try:
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": api_key},
                data={"model_id": "scribe_v1"},
                files={"file": ("audio.wav", wav_data, "audio/wav")},
                timeout=15,
                verify=False,  # macOS SSL workaround
            )
            import re
            text = response.json().get("text", "").strip() if response.status_code == 200 else ""
            
            # Clean up ElevenLabs STT hallucinations like "(music)" or "[coughing]"
            cleaned_text = re.sub(r'\([^)]*\)', '', text)
            cleaned_text = re.sub(r'\[[^\]]*\]', '', cleaned_text)
            cleaned_text = cleaned_text.strip()
            
            if text != cleaned_text:
                print(f"[system] Raw Heard: '{text}'")
            print(f"[system] Heard: '{cleaned_text}'")
            return cleaned_text if cleaned_text else None
        except Exception as e:
            print(f"[system] STT error: {e}")
            return None

    def get_destination(self, args) -> Optional[str]:
        """Get raw destination text from CLI arg or via voice/text prompt."""
        if args.destination:
            return args.destination

        if args.typed_destination:
            print("\n[system] Enter destination (or 'quit' to exit):")
            self._speak_status("Enter your destination in the terminal.")
            try:
                text = input("> ").strip()
                return None if text.lower() in ('quit', 'exit', 'q', '') else text
            except EOFError:
                return None

        # No CLI destination — ask via voice
        result = self._listen_for_destination("Please give a destination.")
        if result:
            return result

        # Voice failed — fall back to text
        print("\n[system] Enter destination (or 'quit' to exit):")
        try:
            text = input("> ").strip()
            return None if text.lower() in ('quit', 'exit', 'q', '') else text
        except EOFError:
            return None

    def verify_destination(
        self,
        raw_destination: str,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        *,
        text_confirmation: bool = False,
    ) -> Optional[dict]:
        """
        Look up the destination on Google Maps, announce it via audio,
        listen for spoken yes/no confirmation.

        Returns a dict with:
            name       - verified place name
            address    - full formatted address
            lat        - latitude (float or None)
            lng        - longitude (float or None)
        or None if the user rejected.
        """
        print(f"[system] Looking up '{raw_destination}' on Google Maps...")

        result = search_destination(raw_destination, lat=lat, lng=lng)
        if not result:
            msg = f"Sorry, I could not find {raw_destination}. Please try again."
            print(f"[system] {msg}")
            if self.interface:
                self.interface.speak_warning(msg, force=True)
                time.sleep(3)
            return None

        # Build and speak confirmation message
        confirm_msg = f"Navigating to {result['name']}. Is this correct?"
        print(f"[system] {confirm_msg}")

        if self.interface:
            self.interface.speak_info(confirm_msg, force=True)
            timeout = time.time() + 15.0
            if hasattr(self.interface.speech, 'is_idle'):
                while not self.interface.speech.is_idle() and time.time() < timeout:
                    time.sleep(0.1)
            else:
                time.sleep(4)
            self.interface.speech.clear_queues()

        # Listen for yes/no using the robust sounddevice verifier
        confirmed = get_text_confirmation() if text_confirmation else get_voice_confirmation()

        if confirmed:
            print(f"[system] ✓ Destination confirmed: {result['name']} | {result['address']} | lat={result['lat']} lng={result['lng']}")
            if self.interface:
                self.interface.speak_info(f"Great. Starting navigation to {result['name']}.", force=True)
                time.sleep(2)
            return result
        else:
            print("[system] Destination rejected. Please try again.")
            if self.interface:
                self.interface.speak_info("OK, let me know where you would like to go.", force=True)
                time.sleep(2)
            return None

    def preflight_camera(self, camera_index: int) -> bool:
        """Check camera availability before starting route state and navigation."""
        try:
            import cv2
        except Exception as exc:
            print(f"[camera] OpenCV is not available: {exc}")
            return False

        cap = None
        try:
            if sys.platform == "darwin":
                cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            else:
                cap = cv2.VideoCapture(camera_index)

            if not cap.isOpened():
                self._print_camera_help(camera_index)
                return False

            ok, _frame = cap.read()
            if not ok:
                print(f"[camera] Webcam index {camera_index} opened, but no frame could be read.")
                self._print_camera_help(camera_index)
                return False
            return True
        finally:
            if cap is not None:
                cap.release()

    def _print_camera_help(self, camera_index: int) -> None:
        print(f"[camera] Could not open webcam index {camera_index}.")
        if sys.platform == "darwin":
            print(
                "[camera] macOS is blocking camera access. Open System Settings > "
                "Privacy & Security > Camera, then enable the app that launched this "
                "command, such as Terminal, iTerm, VS Code, or Codex."
            )
            print("[camera] If the app never appears there, run: tccutil reset Camera")
        else:
            print("[camera] Check that the camera is connected, not in use, and that this process has permission.")

    def _speak_route_instruction(self, message: str) -> None:
        print(f"[spoken-route] {message}")
        if self.interface:
            self.interface.speak_guidance(message, force=True)

    def _speak_status(self, message: str) -> None:
        print(f"[spoken-status] {message}")
        if self.interface:
            self.interface.speak_info(message, force=True)

    def _wait_for_speech_idle(self, timeout_s: float = 12.0) -> None:
        if not self.interface or not hasattr(self.interface.speech, "is_idle"):
            time.sleep(min(timeout_s, 2.0))
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline and not self.interface.speech.is_idle():
            time.sleep(0.1)

    @staticmethod
    def _initial_route_instruction(destination_name: str, indoor_start: str) -> str:
        if str(indoor_start).lower().strip() in {"yes", "true", "1"}:
            return "Leave the room first. Stand still, turn 360 degrees slowly, and scan for a door or exit sign."
        return f"Head toward {destination_name}."

    def _start_route_guidance(
        self,
        args,
        destination_name: str,
        destination_address: str,
        route_start_fix,
    ) -> None:
        """Start route state and speak the first route instruction before camera preflight."""
        if self.interface:
            self.interface.set_destination(destination_name)

        if args.laptop_route and self.route_session:
            route_query = destination_address or destination_name
            try:
                self.route_session.start(
                    destination_name,
                    route_query,
                    indoor_start=args.indoor_start,
                    start_fix=route_start_fix,
                )
                return
            except Exception as exc:
                print(f"[route] Laptop route setup failed: {exc}")

        starts_indoor = str(args.indoor_start).lower().strip() in {"yes", "true", "1"}
        self.route_state.active = True
        self.route_state.destination = destination_name
        self.route_state.mapping_state = "pending" if starts_indoor else "done"
        self.route_state.exit_seeking = starts_indoor
        self.route_state.off_route = False
        self.route_state.next_turn_distance_m = None
        self.route_state.next_instruction = self._initial_route_instruction(destination_name, args.indoor_start)
        self._speak_route_instruction(self.route_state.next_instruction)

    def run(self, args) -> int:
        """Main application loop."""
        if not self.initialize(args):
            return 1

        self.setup_signal_handlers()
        self.is_running = True
        self._start_time = time.time()

        # Final confirmed destination data
        self.confirmed_destination: Optional[dict] = None

        try:
            while not self.stop_event.is_set():
                # Try to fetch GPS location to bias the search and seed laptop routing.
                current_lat = None
                current_lng = None
                route_start_fix = None
                if args.laptop_route:
                    try:
                        import location_service
                        loc = location_service.get_current_location(timeout_s=5.0)
                        route_start_fix = loc
                        current_lat = loc.latitude
                        current_lng = loc.longitude
                        print(f"[system] GPS bias acquired: {current_lat}, {current_lng}")
                    except Exception as e:
                        print(f"[system] Could not fetch GPS location for search bias: {e}")

                # Get raw destination input (voice or CLI)
                raw_destination = self.get_destination(args)

                if raw_destination is None:
                    print("[system] No destination provided, exiting")
                    break

                # Keep asking until user verbally confirms
                destination_data = None
                while destination_data is None and not self.stop_event.is_set():
                    destination_data = self.verify_destination(
                        raw_destination,
                        lat=current_lat,
                        lng=current_lng,
                        text_confirmation=args.typed_destination,
                    )
                    if destination_data is None:
                        # Ask for a new destination
                        if args.typed_destination:
                            print("\n[system] Enter destination (or 'quit' to exit):")
                            try:
                                typed = input("> ").strip()
                                raw_destination = None if typed.lower() in ("quit", "exit", "q", "") else typed
                            except EOFError:
                                raw_destination = None
                        else:
                            raw_destination = self._listen_for_destination("Please give a destination.")
                        if raw_destination is None:
                            # Voice failed — text fallback
                            print("\n[system] Enter destination (or 'quit' to exit):")
                            try:
                                raw_destination = input("> ").strip() or None
                            except EOFError:
                                raw_destination = None
                        if raw_destination is None:
                            break

                if destination_data is None:
                    print("[system] No destination confirmed, exiting")
                    break

                # Store confirmed destination with all data
                self.confirmed_destination = destination_data
                destination_name = destination_data["name"]
                destination_address = destination_data["address"]
                destination_lat = destination_data["lat"]
                destination_lng = destination_data["lng"]

                print(f"\n[system] Starting navigation to: {destination_name}")
                print(f"[system]   Address : {destination_address}")
                print(f"[system]   Coords  : lat={destination_lat}, lng={destination_lng}")
                self._speak_status(f"Starting navigation to {destination_name}.")

                # Initialize and speak route guidance before camera preflight so
                # blocked camera permission cannot skip the route announcement.
                self._start_route_guidance(args, destination_name, destination_address, route_start_fix)
                self._wait_for_speech_idle(timeout_s=14.0)

                if not self.preflight_camera(args.camera):
                    blocked_msg = "Camera is blocked. Enable camera permission for Terminal, then run again."
                    print("[system] Camera preflight failed. Fix camera permission and run the command again.")
                    if self.interface:
                        self.interface.speak_warning(blocked_msg, force=True)
                        time.sleep(2)
                    if self.route_session:
                        self.route_session.stop()
                    return 1

                # NOW start vision + obstacle detection
                print("[system] Starting vision system...")
                print("[system] Press Ctrl+C to stop")
                self._speak_status("Camera guidance is starting. I will speak immediate alerts first, otherwise route directions.")

                try:
                    frames = self.vision.run_forever(
                        camera_index=args.camera,
                        stop_event=self.stop_event
                    )
                    if frames:
                        self._frames_processed += frames
                except KeyboardInterrupt:
                    print("\n[system] Interrupted by user")
                    break

                self.interface.clear_destination()
                if self.route_session:
                    self.route_session.stop()
                self.confirmed_destination = None
                if args.continuous:
                    print("\n[system] Navigation ended. Set a new destination or Ctrl+C to quit")
                    args.destination = None
                else:
                    print("\n[system] Navigation ended. Exiting.")
                    break
                
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
        self.is_running = False
        
        if self.interface:
            self.interface.speak_info("Shutting down. Goodbye.", force=True)
            self.interface.stop()
        if self.route_session:
            self.route_session.stop()
        
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
        description="Production-ready assistive navigation for blind/low-vision users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode with voice input
  %(prog)s --destination "Library"  # Navigate to Library
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
    input_group.add_argument(
        "--continuous",
        action="store_true",
        help="After navigation ends, prompt for another destination instead of exiting.",
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
    pref_group.add_argument(
        "--speech-engine",
        choices=["system", "elevenlabs"],
        default="system",
        help="Speech engine for live guidance. 'system' is low-latency and single-voice; 'elevenlabs' is higher quality but slower.",
    )
    
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
        "--target-fps",
        type=float,
        default=6.0,
        help="Maximum processed camera frames per second (default: 6).",
    )
    vision_group.add_argument(
        "--surface-every-n",
        type=int,
        default=3,
        help="Run expensive surface/door heuristics every N frames, except during indoor exit scans.",
    )
    vision_group.add_argument(
        "--preview",
        action="store_true",
        help="Show the OpenCV preview window. Disabled by default to reduce laptop lag.",
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
        "--door-parts-model",
        type=str,
        default=os.environ.get("DOOR_PARTS_MODEL_PATH"),
        help="Optional YOLOv5 door-parts .pt model with a 'handle' class, such as Joechencc/Door_detection yolov5/best.pt.",
    )
    vision_group.add_argument(
        "--door-parts-yolov5-dir",
        type=str,
        default=os.environ.get("DOOR_PARTS_YOLOV5_DIR"),
        help="Local YOLOv5 checkout directory required for --door-parts-model weights.",
    )
    vision_group.add_argument(
        "--door-parts-conf",
        type=float,
        default=float(os.environ.get("DOOR_PARTS_CONF", "0.40")),
        help="Confidence threshold for the optional YOLOv5 door-parts model.",
    )

    # Address-to-address navigation
    nav_group = parser.add_argument_group("Address Navigation")
    nav_group.add_argument(
        "--from-address",
        type=str,
        default=None,
        help="Starting address for address-to-address navigation",
    )
    nav_group.add_argument(
        "--to-address",
        type=str,
        default=None,
        help="Destination address for address-to-address navigation",
    )
    nav_group.add_argument(
        "--no-laptop-route",
        dest="laptop_route",
        action="store_false",
        help="Disable laptop live route loading; keep camera obstacle guidance only.",
    )
    nav_group.add_argument(
        "--indoor-start",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether to begin with 360-degree indoor door/exit seeking (default: auto).",
    )
    nav_group.add_argument(
        "--arrival-radius-m",
        type=float,
        default=14.0,
        help="Meters from a route step endpoint before advancing to the next step.",
    )
    nav_group.add_argument(
        "--route-poll-seconds",
        type=float,
        default=3.0,
        help="Seconds between laptop location polls while navigating.",
    )
    
    return parser


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Address-to-address navigation planning
    if getattr(args, "from_address", None) and getattr(args, "to_address", None):
        try:
            from address_navigation import NavigationPlanner, LegType

            ors_key = (os.environ.get("OPENROUTESERVICE_API_KEY") or "").strip() or None
            indoor_layout = os.environ.get("INDOOR_LAYOUT_JSON")
            planner = NavigationPlanner(
                ors_key=ors_key,
                indoor_graph_path=indoor_layout,
            )
            plan = planner.plan(args.from_address, args.to_address)
            print(f"\n{'=' * 60}")
            print(f"NAVIGATION PLAN: {plan.origin} -> {plan.destination}")
            print(f"  Origin indoor : {plan.origin_is_indoor}")
            print(f"  Dest indoor   : {plan.destination_is_indoor}")
            print(f"  Total steps   : {plan.total_steps}")
            print(f"{'=' * 60}")
            for i, leg in enumerate(plan.legs, 1):
                print(f"\nLeg {i} ({leg.leg_type.value}):")
                if leg.exit_strategy:
                    print(f"  Exit strategy: {leg.exit_strategy.value}")
                for step in leg.steps:
                    print(f"    - {step}")
        except Exception as exc:
            print(f"[address-nav] Planning failed: {exc}")
            import traceback
            traceback.print_exc()

    app = NavigationApp()
    return app.run(args)


if __name__ == "__main__":
    sys.exit(main())
