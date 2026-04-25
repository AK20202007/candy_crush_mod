"""
Mock indoor-style directions for a destination.

No maps or APIs — just a small scripted route that repeats so the demo
keeps running until you stop the app.
"""

from __future__ import annotations

import itertools
import time
from typing import Callable, List


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


def run_navigation_loop(
    destination: str,
    speak: Callable[[str], None],
    interval_seconds: float = 5.0,
    stop_event=None,
) -> None:
    """
    Speak one direction every ``interval_seconds`` until ``stop_event`` is set.

    ``speak`` should enqueue low-priority speech (e.g. ``SpeechController.speak_normal``).
    """
    route = build_route(destination)
    cycle = itertools.cycle(route)
    while stop_event is None or not stop_event.is_set():
        line = next(cycle)
        speak(line)
        # Wait in slices so we exit promptly when the app stops.
        end = time.time() + interval_seconds
        while time.time() < end:
            if stop_event is not None and stop_event.is_set():
                return
            time.sleep(0.2)
