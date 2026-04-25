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


@dataclass
class RouteStep:
    """Single maneuver with target point where it is considered completed."""

    instruction: str
    end_lat: float
    end_lon: float


def build_route(destination: str) -> List[str]:
    """
    Build a spoken route for the given destination label.

    The destination text is echoed once for context, then mock steps loop.
    """
    dest = destination.strip() or "your destination"
    intro = f"Starting navigation to {dest}."
    return [intro, *_DEFAULT_STEPS]


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
) -> None:
    """
    Speak one direction every ``interval_seconds`` until ``stop_event`` is set.

    ``speak`` should enqueue low-priority speech (e.g. ``SpeechController.speak_normal``).

    If ``route`` is provided (e.g. from a maps API), those strings are spoken instead of
    ``build_route(destination)``. When ``repeat_route`` is False, the list is spoken once
    then the thread idles (suitable for real turn-by-turn lists).
    """
    steps = list(route) if route is not None else build_route(destination)
    if not steps:
        steps = build_route(destination)

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
