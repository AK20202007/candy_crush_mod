"""Laptop route-state driver for webcam navigation.

This module keeps address routing separate from the camera loop. The vision
system reads ``RouteState`` every frame, while this session updates the next
route instruction from laptop location fixes.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import location_service
from agentic_layer.models import RouteState
from navigation import RouteStep
from routing import google_walking_route_steps

WALKING_STEP_M = 0.75


def google_maps_key_from_env_or_app(project_root: Path) -> Optional[str]:
    """Return Google Maps API key from env, then Expo app config."""
    env_key = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
    if env_key:
        return env_key

    app_json = project_root / "mobile" / "app.json"
    try:
        data = json.loads(app_json.read_text())
        key = str(((data.get("expo") or {}).get("extra") or {}).get("googleMapsApiKey") or "").strip()
        return key or None
    except Exception:
        return None


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _walking_steps_phrase(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return ""
    steps = max(1, int(round(distance_m / WALKING_STEP_M)))
    unit = "step" if steps == 1 else "steps"
    return f"in about {steps} {unit}"


class LaptopRouteSession:
    """Updates a shared RouteState from address routing and laptop location."""

    def __init__(
        self,
        route_state: RouteState,
        project_root: Path,
        arrival_radius_m: float = 14.0,
        poll_seconds: float = 3.0,
        confirm_hits: int = 2,
        on_instruction: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.route_state = route_state
        self.project_root = project_root
        self.arrival_radius_m = max(2.0, float(arrival_radius_m))
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.confirm_hits = max(1, int(confirm_hits))
        self.on_instruction = on_instruction
        self._steps: list[RouteStep] = []
        self._idx = 0
        self._hits = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._outdoor_start_announced = False

    def start(
        self,
        destination_name: str,
        destination_query: str,
        *,
        indoor_start: str = "auto",
        start_fix: Optional[location_service.LocationFix] = None,
    ) -> None:
        """Load route steps and start live progression.

        ``indoor_start`` accepts ``auto``, ``yes``, or ``no``. In auto mode the
        laptop's current coordinate is checked against OSM building footprints.
        """
        self.stop(clear_state=False)
        self._stop_event.clear()
        self._idx = 0
        self._hits = 0
        self._steps = []
        self._outdoor_start_announced = False

        fix = start_fix or location_service.get_current_location(timeout_s=6.0)
        source_note = f" via {fix.source}" if fix.source else ""
        print(
            f"[route] Current location{source_note}: "
            f"lat={fix.latitude:.6f}, lon={fix.longitude:.6f}, accuracy={fix.horizontal_accuracy_m}"
        )

        key = google_maps_key_from_env_or_app(self.project_root)
        if key:
            try:
                self._steps = google_walking_route_steps(key, fix.as_lon_lat(), destination_query)
                print(f"[route] Loaded {len(self._steps)} Google walking steps.")
            except Exception as exc:
                print(f"[route] Google walking route failed: {exc}")
        else:
            print("[route] No Google Maps API key found; using destination heading only.")

        starts_indoor = self._detect_indoor_start(fix, indoor_start)
        outdoor_instruction = self._steps[0].instruction if self._steps else f"Head toward {destination_name}."
        first_instruction = (
            "Leave the room first. Stand still, turn 360 degrees slowly, and scan for a door or exit sign."
            if starts_indoor
            else outdoor_instruction
        )
        self._outdoor_start_announced = not starts_indoor

        self.route_state.active = True
        self.route_state.destination = destination_name
        self.route_state.next_instruction = first_instruction
        self.route_state.next_turn_distance_m = None if starts_indoor else self._distance_to_current_step(fix)
        self.route_state.off_route = False
        self.route_state.exit_seeking = starts_indoor
        self.route_state.mapping_state = "pending" if starts_indoor else "done"

        if starts_indoor:
            print("[route] Start appears indoors; camera router will ask for a 360-degree scan and seek a door.")
            self._announce(
                "Route loaded. You appear to be indoors, so I will guide you out of this room first. "
                "Stand still, turn 360 degrees slowly, and scan for a door or exit sign."
            )
        else:
            self._announce(
                self._instruction_message(
                    prefix=f"Route loaded to {destination_name}. First direction:",
                    fix=fix,
                )
            )
        if self._steps:
            self._thread = threading.Thread(target=self._progress_loop, name="LaptopRouteSession", daemon=True)
            self._thread.start()

    def stop(self, *, clear_state: bool = True) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if clear_state:
            self.route_state.active = False
            self.route_state.destination = None
            self.route_state.next_instruction = None
            self.route_state.next_turn_distance_m = None
            self.route_state.off_route = False
            self.route_state.exit_seeking = False
            self.route_state.mapping_state = "done"

    def _detect_indoor_start(self, fix: location_service.LocationFix, indoor_start: str) -> bool:
        mode = indoor_start.lower().strip()
        if mode in {"yes", "true", "1"}:
            return True
        if mode in {"no", "false", "0"}:
            return False
        try:
            return location_service.is_indoors(fix.latitude, fix.longitude, timeout_s=6.0)
        except Exception as exc:
            print(f"[route] Indoor check failed: {exc}")
            return False

    def _distance_to_current_step(self, fix: location_service.LocationFix) -> Optional[float]:
        if not self._steps or self._idx >= len(self._steps):
            return None
        step = self._steps[self._idx]
        return _haversine_meters(fix.latitude, fix.longitude, step.end_lat, step.end_lon)

    def _progress_loop(self) -> None:
        while not self._stop_event.is_set() and self.route_state.active and self._idx < len(self._steps):
            if self.route_state.exit_seeking:
                time.sleep(self.poll_seconds)
                continue

            try:
                fix = location_service.get_current_location(timeout_s=max(2.0, self.poll_seconds))
            except Exception as exc:
                print(f"[route] Location update failed: {exc}")
                time.sleep(self.poll_seconds)
                continue

            dist_m = self._distance_to_current_step(fix)
            self.route_state.next_turn_distance_m = dist_m
            if dist_m is None:
                time.sleep(self.poll_seconds)
                continue

            if not self._outdoor_start_announced:
                self.route_state.next_instruction = self._steps[self._idx].instruction
                self._announce(self._instruction_message(prefix="Outdoor route starting. First direction:", fix=fix))
                self._outdoor_start_announced = True

            if dist_m <= self.arrival_radius_m:
                self._hits += 1
            else:
                self._hits = 0

            if self._hits >= self.confirm_hits:
                self._idx += 1
                self._hits = 0
                if self._idx >= len(self._steps):
                    self.route_state.next_instruction = "You have reached the destination area"
                    self.route_state.next_turn_distance_m = None
                    self.route_state.active = False
                    print("[route] Destination area reached.")
                    self._announce("You have reached the destination area.")
                    return
                self.route_state.next_instruction = self._steps[self._idx].instruction
                self.route_state.next_turn_distance_m = self._distance_to_current_step(fix)
                print(f"[route] Next step {self._idx + 1}/{len(self._steps)}: {self.route_state.next_instruction}")
                self._announce(self._instruction_message(prefix="Next direction:", fix=fix))
            elif not self.route_state.next_instruction:
                self.route_state.next_instruction = self._steps[self._idx].instruction

            time.sleep(self.poll_seconds)

    def _instruction_message(self, *, prefix: str, fix: Optional[location_service.LocationFix] = None) -> str:
        instruction = self.route_state.next_instruction or "Continue toward your destination"
        distance = self.route_state.next_turn_distance_m
        if distance is None and fix is not None:
            distance = self._distance_to_current_step(fix)
        distance_text = ""
        if distance is not None:
            distance_text = f" {_walking_steps_phrase(distance)}"
        return f"{prefix} {instruction}{distance_text}."

    def _announce(self, message: str) -> None:
        if not self.on_instruction:
            return
        try:
            self.on_instruction(message)
        except Exception as exc:
            print(f"[route] Spoken route update failed: {exc}")
