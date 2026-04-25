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
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics weights path or name (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--emergency-contact",
        type=str,
        default="Emergency Services",
        help="Name of the contact to alert in an emergency.",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default=None,
        help="Target destination name (skips interactive prompt if provided).",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        help='Walking start as "longitude,latitude" (use with OPENROUTESERVICE_API_KEY for real directions).',
    )
    parser.add_argument(
        "--nav-interval",
        type=float,
        default=5.0,
        help="Seconds between spoken navigation prompts (default: 5).",
    )
    args = parser.parse_args()

    print("=== Assistive Navigation & Safety System ===")
    print("Loading speech engine...")
    speech = SpeechController()
    speech.start()

    if args.destination:
        destination = args.destination.strip()
    else:
        destination = input("Enter destination (place name or address): ").strip()

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
                # Helper to parse origin
                parts = [p.strip() for p in args.origin.split(",")]
                start = (float(parts[0]), float(parts[1]))
                nav_route = build_maps_route(ors_key, start, destination)
                repeat_nav = False
                print(f"[main] OpenRouteService: {len(nav_route)} spoken steps loaded.")
            except Exception as exc:
                print(f"[main] Maps routing failed ({exc}); using mock navigation.")

    stop_event = threading.Event()

    def alert_callback(msg: str) -> None:
        # We only speak critical warnings now.
        speech.speak_urgent(msg)
        print(f"[ALERT] {msg}")

    vcfg = VisionConfig(
        model_path=args.model,
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

    if args.origin:
        try:
            parts = [p.strip() for p in args.origin.split(",")]
            vision.set_location(float(parts[0]), float(parts[1]))
        except Exception:
            pass

    # Start hardware sensor listeners if available
    vision._motion_fall_detector.start(lambda d: vision.handle_sensor_fall(f"Altitude drop of {d:.1f}m"))
    vision._smv_fall_detector.start(lambda s, x, y: vision.handle_sensor_fall(f"Impact force of {s:.1f} m/s^2"))

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

    print("[main] System active. Monitoring for strobe lights, falls, and providing navigation...")
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
