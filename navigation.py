"""
Mock indoor-style directions for a destination.

No maps or APIs — just a small scripted route that repeats so the demo
keeps running until you stop the app.
"""

from __future__ import annotations

import itertools
import time
from typing import Callable, List, Optional


# Generic steps we cycle through to simulate ongoing guidance.
_DEFAULT_STEPS = [
    "Walk straight",
    "Continue for about ten meters",
    "Turn left in ten meters",
    "Walk straight",
    "Turn right at the next opening",
    "You are approaching your destination area",
]


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
