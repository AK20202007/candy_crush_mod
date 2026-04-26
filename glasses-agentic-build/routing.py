from __future__ import annotations

import html
from typing import Any


import requests


ORS_BASE = "https://api.openrouteservice.org"
WALKING_STEP_M = 0.75


def build_maps_route(api_key: str, start: tuple[float, float], destination: str) -> list[str]:
    """Build walking directions from OpenRouteService.

    Args:
        api_key: OpenRouteService API key.
        start: Current location as (longitude, latitude).
        destination: Human-entered destination/address.
    """
    if not api_key:
        raise RuntimeError("OPENROUTESERVICE_API_KEY is unset.")
    if not destination.strip():
        raise RuntimeError("Destination is empty.")

    dest_lon_lat = _geocode_destination(api_key, destination)
    steps = _walking_steps(api_key, start, dest_lon_lat)
    if not steps:
        raise RuntimeError("OpenRouteService returned no walking steps.")
    return steps


def _geocode_destination(api_key: str, destination: str) -> tuple[float, float]:
    response = requests.get(
        f"{ORS_BASE}/geocode/search",
        params={"api_key": api_key, "text": destination, "size": 1},
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    features = data.get("features") or []
    if not features:
        raise RuntimeError(f"Could not geocode destination: {destination}")
    coords = features[0].get("geometry", {}).get("coordinates")
    if not coords or len(coords) < 2:
        raise RuntimeError(f"Geocoder returned invalid coordinates for: {destination}")
    return (float(coords[0]), float(coords[1]))


def _walking_steps(api_key: str, start: tuple[float, float], end: tuple[float, float]) -> list[str]:
    response = requests.post(
        f"{ORS_BASE}/v2/directions/foot-walking/geojson",
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        json={
            "coordinates": [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]],
            "instructions": True,
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    segments = _first_route_segments(data)

    steps: list[str] = []
    for segment in segments:
        for step in segment.get("steps") or []:
            instruction = _clean_instruction(str(step.get("instruction") or ""))
            distance_m = step.get("distance")
            if not instruction:
                continue
            if isinstance(distance_m, (int, float)) and distance_m > 0:
                step_count = max(1, int(round(float(distance_m) / WALKING_STEP_M)))
                unit = "step" if step_count == 1 else "steps"
                steps.append(f"{instruction} for about {step_count} {unit}")
            else:
                steps.append(instruction)
    return steps


def _first_route_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    features = data.get("features") or []
    if not features:
        return []
    properties = features[0].get("properties") or {}
    return list(properties.get("segments") or [])


def _clean_instruction(value: str) -> str:
    text = html.unescape(value).replace("<b>", "").replace("</b>", "")
    return " ".join(text.split()).rstrip(".")
