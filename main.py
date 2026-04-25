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

# Import our improved components
from agentic_layer import AgenticNavigationRouter
from agentic_layer.config import DEFAULT_PROFILE_NAME, load_profiles
from navigation_interface import NavigationInterface
from speech_controller import IntelligentSpeechController
from user_interface import UserMode, UserPreferences
from vision import VisionConfig, VisionSystem

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
            print("Production-Ready Interface v2.0")
            print("=" * 60)
            
            # Load preferences
            prefs = self.load_preferences(args)
            print(f"[config] Verbosity: {prefs.verbosity}")
            print(f"[config] Sensitivity: {prefs.warning_sensitivity}")
            print(f"[config] Speech rate: {prefs.speech_rate}x")
            
            # Create router and interface
            router = AgenticNavigationRouter()
            self.interface = NavigationInterface(router, prefs)
            
            # Setup vision system
            vision_config = VisionConfig(
                imgsz=args.imgsz or 640,
                conf=args.conf or 0.35,
                iou=args.iou or 0.5,
                augment=args.augment,
                half=not args.no_half,
            )
            
            def handle_decision(decision):
                """Callback for vision system decisions."""
                if self.interface and not self.stop_event.is_set():
                    # Create a minimal frame context
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
    
    app = NavigationApp()
    return app.run(args)


if __name__ == "__main__":
    sys.exit(main())
