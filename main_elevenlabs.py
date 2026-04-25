#!/usr/bin/env python3
"""
Assistive Navigation System with ElevenLabs Speech

High-quality voice interaction using ElevenLabs for both TTS and STT.

Setup:
    1. Get API key from https://elevenlabs.io
    2. Set environment variable: export ELEVENLABS_API_KEY="your_key"
    3. Install dependencies: pip install -r requirements.txt
    4. Run: python main_elevenlabs.py

Features:
    - Natural, human-like text-to-speech
    - Accurate speech-to-text for voice commands
    - Multiple voice options (default, calm, professional, warm)
    - Fallback to system TTS if ElevenLabs unavailable
    - All existing navigation features (vision, routing, safety)

Usage:
    python main_elevenlabs.py                          # Interactive mode
    python main_elevenlabs.py -d "Library"            # Direct destination
    python main_elevenlabs.py --voice calm           # Use calm voice
    python main_elevenlabs.py --voice-id "CUSTOM_ID" # Use specific voice
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

from agentic_layer import AgenticNavigationRouter
from agentic_layer.config import DEFAULT_PROFILE_NAME, load_profiles
from elevenlabs_speech import ElevenLabsSpeechController, create_elevenlabs_controller
from navigation_interface import NavigationInterface
from user_interface import UserMode, UserPreferences
from vision import VisionConfig, VisionSystem


class ElevenLabsNavigationApp:
    """Navigation app with ElevenLabs speech integration."""
    
    def __init__(self):
        self.interface: Optional[NavigationInterface] = None
        self.speech: Optional[ElevenLabsSpeechController] = None
        self.vision: Optional[VisionSystem] = None
        self.stop_event = threading.Event()
        self.is_running = False
        self._start_time: Optional[float] = None
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown."""
        def signal_handler(signum, frame):
            print("\n[system] Shutdown signal received...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def load_preferences(self, args) -> UserPreferences:
        """Load user preferences."""
        prefs = UserPreferences()
        
        if args.verbosity:
            prefs.verbosity = args.verbosity
        if args.speech_rate:
            prefs.speech_rate = max(0.5, min(1.5, args.speech_rate))
        if args.sensitivity:
            prefs.warning_sensitivity = args.sensitivity
        
        # Load from config file
        config_path = Path.home() / ".assistive_nav" / "preferences.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    for key, value in config.items():
                        if hasattr(prefs, key):
                            setattr(prefs, key, value)
            except Exception as e:
                print(f"[config] Error loading: {e}")
        
        return prefs
    
    def initialize(self, args) -> bool:
        """Initialize all components with ElevenLabs."""
        try:
            print("=" * 70)
            print("ASSISTIVE NAVIGATION SYSTEM with ElevenLabs Speech")
            print("=" * 70)
            
            # Check API key
            api_key = args.api_key or os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                print("\n⚠️  Warning: No ELEVENLABS_API_KEY found!")
                print("   Set it with: export ELEVENLABS_API_KEY='your_key'")
                print("   Get a key at: https://elevenlabs.io\n")
                print("   Continuing with fallback system TTS...\n")
            
            # Load preferences
            prefs = self.load_preferences(args)
            print(f"[config] Verbosity: {prefs.verbosity}")
            print(f"[config] Sensitivity: {prefs.warning_sensitivity}")
            print(f"[config] Speech rate: {prefs.speech_rate}x")
            
            # Initialize ElevenLabs speech
            print(f"[elevenlabs] Initializing with voice: {args.voice}")
            self.speech = create_elevenlabs_controller(
                api_key=api_key,
                voice=args.voice,
                voice_id=args.voice_id,
                user_interface=NavigationInterface(AgenticNavigationRouter(), prefs).ui,
                fallback_to_system=True
            )
            self.speech.start()
            
            # Create interface with ElevenLabs speech
            router = AgenticNavigationRouter()
            self.interface = NavigationInterface(router, prefs)
            # Replace speech controller with ElevenLabs
            self.interface.speech = self.speech
            
            # Setup vision
            vision_config = VisionConfig(
                imgsz=args.imgsz or 640,
                conf=args.conf or 0.35,
                iou=args.iou or 0.5,
                augment=args.augment,
                half=not args.no_half,
            )
            
            def handle_decision(decision):
                """Callback for vision decisions."""
                if self.interface and not self.stop_event.is_set():
                    from agentic_layer.models import FrameContext, SceneState
                    ctx = FrameContext(
                        timestamp_ms=int(time.time() * 1000),
                        frame_id=str(time.time()),
                        scene=SceneState(location_type="sidewalk")
                    )
                    self.interface.process_decision(decision, ctx)
            
            self.vision = VisionSystem(
                config=vision_config,
                on_decision=handle_decision,
            )
            
            # Welcome message with ElevenLabs
            self.speech.speak_info(
                "Welcome to assistive navigation with ElevenLabs. "
                "Say 'navigate' followed by your destination, or type it in."
            )
            
            return True
            
        except Exception as e:
            print(f"[error] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_destination_voice(self, timeout: float = 10.0) -> Optional[str]:
        """Get destination using ElevenLabs speech-to-text."""
        if not self.speech:
            return None
        
        self.speech.speak_info("Listening for destination. Please speak clearly.")
        
        try:
            result = self.speech.recognize_speech(timeout=timeout)
            return result
        except Exception as e:
            print(f"[elevenlabs] Voice recognition failed: {e}")
            return None
    
    def get_destination_text(self) -> Optional[str]:
        """Get destination via text input."""
        print("\n[system] Enter destination (or 'quit' to exit):")
        try:
            destination = input("> ").strip()
            if destination.lower() in ['quit', 'exit', 'q']:
                return None
            return destination if destination else None
        except EOFError:
            return None
    
    def run(self, args) -> int:
        """Main application loop."""
        if not self.initialize(args):
            return 1
        
        self.setup_signal_handlers()
        self.is_running = True
        self._start_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                # Get destination
                destination = None
                
                if args.destination:
                    destination = args.destination
                elif args.use_voice and self.speech and self.speech.client:
                    destination = self.get_destination_voice(timeout=args.voice_timeout)
                    if not destination:
                        print("[system] Voice input failed, switching to text...")
                        destination = self.get_destination_text()
                else:
                    destination = self.get_destination_text()
                
                if destination is None:
                    print("[system] No destination provided, exiting")
                    break
                
                # Confirm destination
                self.speech.speak_info(f"Starting navigation to {destination}")
                print(f"\n[system] Starting navigation to: {destination}")
                
                # Set destination
                self.interface.set_destination(destination)
                
                # Run vision
                print("[system] Starting vision system...")
                print("[system] Press Ctrl+C to stop, or 'q' in camera window")
                
                try:
                    self.vision.run_forever(
                        camera_index=args.camera,
                        stop_event=self.stop_event
                    )
                except KeyboardInterrupt:
                    print("\n[system] Interrupted by user")
                    break
                
                # Navigation ended
                self.interface.clear_destination()
                self.speech.speak_info("Navigation ended. Where would you like to go next?")
                print("\n[system] Navigation ended. Enter new destination or Ctrl+C to quit")
                
                # Reset for next iteration
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
        """Stop all components."""
        if not self.is_running:
            return
        
        print("\n[system] Stopping components...")
        self.stop_event.set()
        self.is_running = False
        
        if self.speech:
            self.speech.speak_info("Shutting down. Goodbye.")
            self.speech.stop()
        
        if self.interface:
            self.interface.stop()
        
        # Print stats
        if self._start_time:
            runtime = time.time() - self._start_time
            print(f"\n[stats] Runtime: {runtime:.1f} seconds")
            if self.speech:
                stats = self.speech.get_statistics()
                print(f"[stats] Messages spoken: {stats['messages_spoken']}")
                print(f"[stats] ElevenLabs API calls: {stats['api_calls']}")
                print(f"[stats] Using ElevenLabs: {stats['elevenlabs_available']}")


def build_parser() -> argparse.ArgumentParser:
    """Build command line parser."""
    profile_names = sorted(load_profiles())
    
    parser = argparse.ArgumentParser(
        description="Assistive navigation with ElevenLabs speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive with ElevenLabs
  %(prog)s -d "Library"            # Direct destination
  %(prog)s --voice calm            # Use calm voice preset
  %(prog)s --voice-id "CUSTOM"     # Use specific voice ID
  %(prog)s --use-voice             # Use voice input
  %(prog)s --api-key "KEY"         # Provide API key directly

Voice Presets:
  default      - Natural, balanced tone
  calm         - Relaxed, soothing voice
  professional - Clear, authoritative tone
  warm         - Friendly, approachable voice

Setup:
  1. Get API key from https://elevenlabs.io
  2. export ELEVENLABS_API_KEY="your_key"
  3. python %(prog)s
        """
    )
    
    # ElevenLabs options
    el_group = parser.add_argument_group("ElevenLabs Options")
    el_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)"
    )
    el_group.add_argument(
        "--voice",
        choices=["default", "calm", "professional", "warm"],
        default="default",
        help="Voice preset (default: default)"
    )
    el_group.add_argument(
        "--voice-id",
        type=str,
        default=None,
        help="Specific ElevenLabs voice ID (overrides --voice)"
    )
    el_group.add_argument(
        "--use-voice",
        action="store_true",
        help="Use voice input (ElevenLabs STT) for destination"
    )
    el_group.add_argument(
        "--voice-timeout",
        type=float,
        default=10.0,
        help="Seconds to listen for voice input (default: 10)"
    )
    
    # Navigation options
    nav_group = parser.add_argument_group("Navigation Options")
    nav_group.add_argument(
        "-d", "--destination",
        type=str,
        default=None,
        help="Direct destination (skips prompt)"
    )
    nav_group.add_argument(
        "-t", "--typed-destination",
        action="store_true",
        help="Force text input (not voice)"
    )
    
    # User preferences
    pref_group = parser.add_argument_group("User Preferences")
    pref_group.add_argument(
        "-v", "--verbosity",
        choices=["minimal", "normal", "verbose"],
        default=None,
        help="Speech feedback level"
    )
    pref_group.add_argument(
        "-s", "--sensitivity",
        choices=["low", "normal", "high"],
        default=None,
        help="Warning frequency"
    )
    pref_group.add_argument(
        "-r", "--speech-rate",
        type=float,
        default=None,
        help="Speech speed multiplier (0.5 to 1.5)"
    )
    
    # Vision options
    vision_group = parser.add_argument_group("Vision Options")
    vision_group.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    vision_group.add_argument(
        "-p", "--profile",
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
        "--no-half",
        action="store_true",
        help="Disable half-precision inference"
    )
    vision_group.add_argument(
        "--augment",
        action="store_true",
        help="Enable test-time augmentation"
    )
    
    return parser


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    # If --typed-destination is set, disable voice input
    if args.typed_destination:
        args.use_voice = False
    
    app = ElevenLabsNavigationApp()
    return app.run(args)


if __name__ == "__main__":
    sys.exit(main())
