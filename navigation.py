"""
Mock indoor-style directions for a destination.

No maps or APIs — just a small scripted route that repeats so the demo
keeps running until you stop the app.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


# Generic steps we cycle through to simulate ongoing guidance.
_DEFAULT_STEPS = [
    "Walk straight",
    "Continue for about ten meters",
    "Turn left in ten meters",
    "Walk straight",
    "Turn right at the next opening",
    "You are approaching your destination area",
]

# Destination-aware mock steps for common location types
_BUILDING_KEYWORDS = {"hall", "building", "center", "library", "gym", "office", "lab", "store", "shop", "restaurant", "cafe"}
_ROOM_KEYWORDS = {"room", "suite", "unit", "apartment", "floor"}


def _destination_aware_steps(destination: str) -> List[str]:
    """Generate context-aware mock steps based on the destination name."""
    dest_lower = destination.lower()
    steps = []
    
    # Always start with the destination context
    steps.append(f"Head toward {destination}.")
    
    # Check if destination sounds like a building
    is_building = any(kw in dest_lower for kw in _BUILDING_KEYWORDS)
    is_room = any(kw in dest_lower for kw in _ROOM_KEYWORDS)
    is_outdoor = any(kw in dest_lower for kw in {"park", "field", "lot", "street", "avenue", "road", "plaza"})
    
    if is_building:
        steps.extend([
            "Continue walking straight ahead.",
            f"Look for signs pointing to {destination}.",
            "Keep walking, the building should be coming up.",
            f"You are getting closer to {destination}.",
            f"{destination} should be nearby. Look for the entrance.",
        ])
    elif is_room:
        steps.extend([
            "Continue down the hallway.",
            f"Look for signs or room numbers for {destination}.",
            "Keep walking, check the doors on your left and right.",
            f"You are getting closer to {destination}.",
            f"{destination} should be nearby. Check the next few doors.",
        ])
    elif is_outdoor:
        steps.extend([
            "Continue walking along the sidewalk.",
            f"Keep heading toward {destination}.",
            "Watch for crossings and intersections ahead.",
            f"You are getting closer to {destination}.",
            f"{destination} should be in this area.",
        ])
    else:
        steps.extend([
            "Continue walking straight ahead.",
            f"Keep heading toward {destination}.",
            "Stay on the current path.",
            f"You are getting closer to {destination}.",
            f"{destination} should be nearby.",
        ])
    
    return steps


@dataclass
class RouteStep:
    """Single maneuver with target point where it is considered completed."""

    instruction: str
    end_lat: float
    end_lon: float


def build_route(destination: str) -> List[str]:
    """
    Build a spoken route for the given destination label.

    The destination text is echoed once for context, then destination-aware
    mock steps loop to keep reminding the user where they are going.
    """
    dest = destination.strip() or "your destination"
    intro = f"Starting navigation to {dest}."
    steps = _destination_aware_steps(dest)
    return [intro, *steps]


def _wait_interval(stop_event, interval_seconds: float) -> bool:
    """Block for ``interval_seconds`` (in short sleeps). Return True if caller should exit."""
    end = time.time() + interval_seconds
    while time.time() < end:
        if stop_event is not None and stop_event.is_set():
            return True
        time.sleep(0.2)
    return False


def run_navigation_loop(
    destination: str,
    speak: Callable[[str], None],
    interval_seconds: float = 5.0,
    stop_event=None,
    route: Optional[List[str]] = None,
    repeat_route: bool = True,
    startup_delay_s: float = 25.0,
    speak_urgent: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Speak one direction every ``interval_seconds`` until ``stop_event`` is set.

    ``speak`` should enqueue low-priority speech (e.g. ``SpeechController.speak_normal``).
    ``speak_urgent`` is used for the startup confirmation (spoken immediately).

    If ``route`` is provided (e.g. from a maps API), those strings are spoken instead of
    ``build_route(destination)``. When ``repeat_route`` is False, the list is spoken once
    then the thread idles (suitable for real turn-by-turn lists).
    
    ``startup_delay_s`` adds a delay before navigation starts so the system
    can confirm the destination and let the user orient themselves first.
    """
    dest = destination.strip() or "your destination"
    _speak_now = speak_urgent or speak

    # Phase 1: Announce destination and let obstacle detection settle
    _speak_now(f"Destination set to {dest}. Scanning your surroundings for 25 seconds.")
    print(f"[nav] Waiting {startup_delay_s}s before starting navigation prompts...")
    if _wait_interval(stop_event, startup_delay_s):
        return

    # Phase 2: Confirm and begin navigation
    _speak_now(f"Starting navigation to {dest}. I will guide you.")
    if _wait_interval(stop_event, 3.0):
        return

    steps = list(route) if route is not None else build_route(destination)
    if not steps:
        steps = build_route(destination)

    # Skip the intro step (we already announced it)
    if steps and steps[0].lower().startswith("starting navigation"):
        steps = steps[1:]

    if repeat_route:
        cycle = itertools.cycle(steps)
        while stop_event is None or not stop_event.is_set():
            line = next(cycle)
            speak(line)
            if _wait_interval(stop_event, interval_seconds):
                return
    else:
        for line in steps:
            if stop_event is not None and stop_event.is_set():
                return
            speak(line)
            if _wait_interval(stop_event, interval_seconds):
                return
        speak("You should be near your destination. Obstacle detection is still active.")
        while stop_event is None or not stop_event.is_set():
            time.sleep(0.3)


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def run_live_navigation_loop(
    route_steps: List[RouteStep],
    speak: Callable[[str], None],
    get_location: Callable[[], Optional[Tuple[float, float]]],
    stop_event=None,
    arrival_radius_m: float = 14.0,
    poll_seconds: float = 1.0,
    confirm_hits: int = 2,
) -> None:
    """
    Advance instructions only when user reaches each maneuver target.

    ``get_location`` must return (lat, lon) or None when no new location exists.
    """
    if not route_steps:
        speak("No map route steps available. Using obstacle warnings only.")
        while stop_event is None or not stop_event.is_set():
            time.sleep(0.3)
        return

    idx = 0
    hits = 0
    arrival_radius_m = max(2.0, float(arrival_radius_m))
    need_hits = max(1, int(confirm_hits))

    speak(f"Live navigation active. Step 1: {route_steps[0].instruction}")
    while stop_event is None or not stop_event.is_set():
        loc = get_location()
        if loc is None:
            time.sleep(poll_seconds)
            continue

        lat, lon = loc
        target = route_steps[idx]
        dist_m = _haversine_meters(lat, lon, target.end_lat, target.end_lon)
        if dist_m <= arrival_radius_m:
            hits += 1
        else:
            hits = 0

        if hits < need_hits:
            time.sleep(poll_seconds)
            continue

        idx += 1
        hits = 0
        if idx >= len(route_steps):
            speak("You have reached the destination area. Obstacle detection is still active.")
            while stop_event is None or not stop_event.is_set():
                time.sleep(0.3)
            return

        next_step = route_steps[idx]
        speak(f"Next: {next_step.instruction}")
        time.sleep(poll_seconds)
