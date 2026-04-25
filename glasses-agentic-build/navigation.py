from __future__ import annotations

import itertools
import time
from typing import Callable, Iterable, Optional


DEFAULT_ROUTE = (
    "Continue toward your destination",
    "Keep moving carefully and listen for safety warnings",
)


def run_navigation_loop(
    destination: str,
    speak: Callable[[str], None],
    interval_seconds: float = 5.0,
    stop_event=None,
    route: Optional[Iterable[str]] = None,
    repeat_route: bool = True,
) -> None:
    """Minimal route prompt loop used by the agentic demo.

    The agentic router decides whether these prompts are spoken, printed, or
    suppressed in favor of safety warnings.
    """
    steps = list(route or DEFAULT_ROUTE)
    if destination:
        steps[0] = f"Continue toward {destination}"

    iterator = itertools.cycle(steps) if repeat_route else iter(steps)
    for step in iterator:
        if stop_event is not None and stop_event.is_set():
            return
        speak(step)
        deadline = time.time() + max(0.25, float(interval_seconds))
        while time.time() < deadline:
            if stop_event is not None and stop_event.is_set():
                return
            time.sleep(0.1)
