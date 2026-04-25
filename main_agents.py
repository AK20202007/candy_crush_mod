import argparse
import os
import threading

from navigation import RouteStep, run_live_navigation_loop, run_navigation_loop
from routing import build_maps_route, google_walking_directions, google_walking_route_steps
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


def _parse_lat_lon(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError('Expected "lat,lon".')
    lat, lon = float(parts[0]), float(parts[1])
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError("Latitude or longitude out of range.")
    return lat, lon


class LiveLocationStore:
    """Thread-safe holder for latest simulated/real location sample."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: tuple[float, float] | None = None

    def set(self, lat: float, lon: float) -> None:
        with self._lock:
            self._latest = (lat, lon)

    def get(self) -> tuple[float, float] | None:
        with self._lock:
            return self._latest


def _simulate_location_input_loop(store: LiveLocationStore, stop_event: threading.Event) -> None:
    print('[sim] Live location simulation enabled. Type coordinates as "lat,lon" then Enter.')
    print('[sim] Example: 37.4275,-122.1697  (Ctrl+C or q to quit app)')
    while not stop_event.is_set():
        try:
            raw = input().strip()
        except EOFError:
            return
        if not raw:
            continue
        if raw.lower() in {"q", "quit", "exit"}:
            stop_event.set()
            return
        try:
            lat, lon = _parse_lat_lon(raw)
            store.set(lat, lon)
            print(f"[sim] location updated: lat={lat:.6f}, lon={lon:.6f}")
        except ValueError as exc:
            print(f"[sim] invalid coordinate ({exc}); expected: lat,lon")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seizure and Fall Detection (Strobe + Sensor Fusion).")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
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
        help='Walking start as "longitude,latitude" (required for real map directions).',
    )
    parser.add_argument(
        "--maps-provider",
        type=str,
        default="auto",
        choices=("auto", "ors", "google"),
        help="Routing backend: auto, ors, or google (default: auto).",
    )
    parser.add_argument(
        "--live-nav",
        action="store_true",
        help="Advance map instructions by live location progress instead of time intervals.",
    )
    parser.add_argument(
        "--arrival-radius-m",
        type=float,
        default=14.0,
        help="Meters from a maneuver endpoint to count the step as completed (live-nav mode).",
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
    nav_live_steps: list[RouteStep] | None = None
    repeat_nav = True
    ors_key = (os.environ.get("OPENROUTESERVICE_API_KEY") or "").strip()
    google_key = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
    if args.origin:
        try:
            start = _parse_origin_lon_lat(args.origin)
            provider = args.maps_provider
            if provider == "auto":
                if google_key:
                    provider = "google"
                elif ors_key:
                    provider = "ors"
                else:
                    provider = "none"

            if provider == "google":
                if not google_key:
                    raise RuntimeError("GOOGLE_MAPS_API_KEY is not set.")
                if args.live_nav:
                    nav_live_steps = google_walking_route_steps(google_key, start, destination)
                    nav_route = [f"Starting map navigation toward {destination.strip()}."]
                    print(f"[main] Google Maps live-nav: {len(nav_live_steps)} maneuver steps loaded.")
                else:
                    nav_route = google_walking_directions(google_key, start, destination)
                    print(f"[main] Google Maps: {len(nav_route)} spoken steps loaded.")
            elif provider == "ors":
                if not ors_key:
                    raise RuntimeError("OPENROUTESERVICE_API_KEY is not set.")
                nav_route = build_maps_route(ors_key, start, destination)
                print(f"[main] OpenRouteService: {len(nav_route)} spoken steps loaded.")
            else:
                print("[main] --origin set but no map API key found; using mock navigation.")

            if nav_route:
                repeat_nav = False
        except Exception as exc:
            print(f"[main] Maps routing failed ({exc}); using mock navigation.")
    else:
        if args.live_nav:
            print("[main] WARNING: --live-nav requires --origin; falling back to mock navigation.")
        if google_key or ors_key:
            print("[main] Map API key detected; add --origin=lon,lat to enable turn-by-turn directions.")

    stop_event = threading.Event()
    location_store = LiveLocationStore()

    def alert_callback(msg: str, priority: int = 10) -> None:
        # We only speak critical warnings now.
        speech.speak_urgent(msg, priority=priority)
        print(f"[ALERT] {msg}")

    vcfg = VisionConfig(
        model_path=args.model,
        emergency_contact=args.emergency_contact,
        # We can keep vision-based fall detection active as a supplement
        confirm_frames=2,
    )
    vision = VisionSystem(
        on_warning=alert_callback,
        on_emergency=speech.speak_emergency,
        config=vcfg
    )

    print(f"[main] Model will track labels: {sorted(vision.active_class_labels())}")
    nav_mode = "mock directions"
    if nav_live_steps:
        nav_mode = "map route (live progress)"
    elif nav_route:
        nav_mode = "map route"
    print(f"[main] Navigation thread starting ({nav_mode}).")

    if args.origin:
        try:
            start = _parse_origin_lon_lat(args.origin)
            vision.set_location(start[0], start[1])
        except Exception:
            pass

    # Start hardware sensor listeners if available
    vision._motion_fall_detector.start(lambda d: vision.handle_sensor_fall(f"DRAMATIC elevation change of {d:.1f}m", priority=99))
    vision._smv_fall_detector.start(lambda s, x, y: vision.handle_sensor_fall(f"Impact force of {s:.1f} m/s^2", priority=10))

    def nav_speak(msg: str) -> None:
        speech.speak_normal(msg)
        print(f"[nav->speech] {msg}")

    nav_target = run_navigation_loop
    nav_kwargs = {
        "destination": destination,
        "speak": nav_speak,
        "interval_seconds": args.nav_interval,
        "stop_event": stop_event,
        "route": nav_route,
        "repeat_route": repeat_nav,
    }
    if nav_live_steps:
        nav_target = run_live_navigation_loop
        nav_kwargs = {
            "route_steps": nav_live_steps,
            "speak": nav_speak,
            "get_location": location_store.get,
            "stop_event": stop_event,
            "arrival_radius_m": args.arrival_radius_m,
        }

    nav_thread = threading.Thread(
        target=nav_target,
        kwargs=nav_kwargs,
        name="Navigation",
        daemon=True,
    )
    nav_thread.start()

    sim_thread = None
    if nav_live_steps:
        sim_thread = threading.Thread(
            target=_simulate_location_input_loop,
            kwargs={"store": location_store, "stop_event": stop_event},
            name="LocationSimulator",
            daemon=True,
        )
        sim_thread.start()

    print("[main] System active. Monitoring for strobe lights, falls, and providing navigation (opens preview window).")
    try:
        vision.run_forever(camera_index=args.camera, stop_event=stop_event)
    except KeyboardInterrupt:
        print("[main] Shutdown requested.")
    finally:
        stop_event.set()
        speech.stop()
        nav_thread.join(timeout=2.0)
        if sim_thread is not None:
            sim_thread.join(timeout=0.5)
        print("[main] System stopped.")


if __name__ == "__main__":
    main()
