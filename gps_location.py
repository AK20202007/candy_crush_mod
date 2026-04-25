"""GPS-based indoor/outdoor location detection.

Uses the Overpass API (OpenStreetMap) to check if the current GPS coordinates
fall inside a building. No API key required.

Usage:
    from gps_location import LocationDetector
    detector = LocationDetector()
    loc_type = detector.detect()  # "indoor" or "outdoor"

GPS coordinates can be provided via:
    --gps "latitude,longitude"       command-line argument
    GPS_COORDINATES env var          e.g. "37.7749,-122.4194"

If GPS is unavailable, defaults to "unknown".
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPSCoords:
    latitude: float
    longitude: float


def parse_gps(raw: str) -> Optional[GPSCoords]:
    """Parse 'latitude,longitude' string into GPSCoords."""
    try:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            return None
        lat, lon = float(parts[0]), float(parts[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return GPSCoords(latitude=lat, longitude=lon)
    except (ValueError, TypeError):
        pass
    return None


def _query_overpass(lat: float, lon: float, timeout_s: int = 5) -> bool:
    """Query Overpass API to check if coordinates are inside a building.
    
    Returns True if inside a building, False otherwise.
    """
    query = (
        f'[out:json][timeout:{timeout_s}];'
        f'is_in({lat},{lon})->.a;'
        f'area.a[building];'
        f'out count;'
    )
    url = f"https://overpass-api.de/api/interpreter?data={urllib.request.quote(query)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AssistiveNav/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # If any building areas contain this point, count > 0
            elements = data.get("elements", [])
            if elements:
                count = elements[0].get("tags", {}).get("total", 0)
                return int(count) > 0
    except Exception as exc:
        print(f"[gps] Overpass query failed: {exc}")
    return False


class LocationDetector:
    """Detect indoor/outdoor location using GPS + OpenStreetMap building data."""

    def __init__(self, gps: Optional[GPSCoords] = None, cache_ttl_s: float = 60.0) -> None:
        self._gps = gps
        self._cache_ttl_s = cache_ttl_s
        self._cached_result: Optional[str] = None
        self._cache_time: float = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_args(cls, gps_arg: Optional[str] = None) -> "LocationDetector":
        """Create from CLI arg or environment variable."""
        raw = gps_arg or os.environ.get("GPS_COORDINATES", "")
        coords = parse_gps(raw) if raw else None
        if coords:
            print(f"[gps] Coordinates: {coords.latitude}, {coords.longitude}")
        else:
            print("[gps] No GPS coordinates provided. Use --gps 'lat,lon' or GPS_COORDINATES env var.")
        return cls(gps=coords)

    def detect(self) -> str:
        """Return 'indoor', 'outdoor', or 'unknown'.
        
        Results are cached for cache_ttl_s seconds to avoid spamming the API.
        """
        if self._gps is None:
            return "unknown"

        with self._lock:
            now = time.time()
            if self._cached_result and (now - self._cache_time) < self._cache_ttl_s:
                return self._cached_result

            try:
                inside_building = _query_overpass(self._gps.latitude, self._gps.longitude)
                result = "indoor" if inside_building else "outdoor"
                self._cached_result = result
                self._cache_time = now
                print(f"[gps] Location detected: {result}")
                return result
            except Exception as exc:
                print(f"[gps] Detection failed: {exc}")
                return self._cached_result or "unknown"

    def update_gps(self, coords: GPSCoords) -> None:
        """Update GPS coordinates (e.g. from a phone sensor feed)."""
        with self._lock:
            self._gps = coords
            self._cached_result = None  # Invalidate cache

    @property
    def has_gps(self) -> bool:
        return self._gps is not None

    @property
    def location_type_for_config(self) -> str:
        """Map GPS detection to VisionConfig/agent location_type values."""
        result = self.detect()
        if result == "indoor":
            return "hallway"  # Conservative indoor default
        elif result == "outdoor":
            return "sidewalk"  # Conservative outdoor default
        return "unknown"
