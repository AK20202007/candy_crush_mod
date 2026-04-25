"""
Optional real walking directions via OpenRouteService (ORS).

Sign up for a free API key: https://openrouteservice.org/dev/#/signup
Set environment variable: OPENROUTESERVICE_API_KEY

ORS uses coordinates as [longitude, latitude] (not lat, lon).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

ORS_GEOCODE = "https://api.openrouteservice.org/geocode/search"
ORS_DIRECTIONS = "https://api.openrouteservice.org/v2/directions/foot-walking/json"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def geocode(api_key: str, query: str, timeout_s: float = 15.0) -> Tuple[float, float]:
    """Return (longitude, latitude) for the best match for ``query``."""
    q = quote(query.strip(), safe="")
    url = f"{ORS_GEOCODE}?text={q}&size=1"
    req = urllib.request.Request(
        url,
        headers={"Authorization": api_key, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Geocoding HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e

    feats = data.get("features") or []
    if not feats:
        raise RuntimeError(f"No geocoding results for: {query!r}")
    geom = feats[0].get("geometry") or {}
    coords = geom.get("coordinates")
    if not coords or len(coords) < 2:
        raise RuntimeError("Geocoding response missing coordinates")
    lon, lat = float(coords[0]), float(coords[1])
    return lon, lat


def walking_directions(api_key: str, start_lon_lat: Tuple[float, float], end_lon_lat: Tuple[float, float], timeout_s: float = 30.0) -> List[str]:
    """
    Return spoken-style step strings from ORS foot-walking directions.

    ``start_lon_lat`` and ``end_lon_lat`` are (longitude, latitude).
    """
    body = json.dumps({"coordinates": [list(start_lon_lat), list(end_lon_lat)]}).encode()
    req = urllib.request.Request(
        ORS_DIRECTIONS,
        data=body,
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Directions HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e

    segments = _directions_segments(data)
    if not segments:
        raise RuntimeError("No route returned (check coordinates and profile).")

    lines: List[str] = []
    for seg in segments:
        for step in seg.get("steps") or []:
            instr = step.get("instruction")
            if not instr:
                continue
            line = _strip_html(str(instr))
            if line:
                lines.append(line)
    if not lines:
        raise RuntimeError("Route had no step instructions to speak.")
    return lines


def _directions_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ORS JSON may expose segments under ``routes[0]`` or GeoJSON ``features[0].properties``."""
    routes = data.get("routes")
    if isinstance(routes, list) and routes:
        return routes[0].get("segments") or []
    feats = data.get("features")
    if isinstance(feats, list) and feats:
        props = feats[0].get("properties") or {}
        return props.get("segments") or []
    return []


def build_maps_route(api_key: str, start_lon_lat: Tuple[float, float], destination_query: str) -> List[str]:
    """Geocode destination, fetch walking steps, prepend a short intro."""
    if not destination_query.strip():
        raise ValueError("Destination query is empty.")
    end = geocode(api_key, destination_query)
    steps = walking_directions(api_key, start_lon_lat, end)
    intro = f"Starting map navigation toward {destination_query.strip()}."
    return [intro, *steps]
