from dataclasses import dataclass
import json
import urllib.request
import urllib.parse
from typing import Optional

@dataclass
class GPSCoords:
    latitude: float
    longitude: float

def parse_gps(coords_str: str) -> Optional[GPSCoords]:
    """Parse a string like '34.05,-118.24' into GPSCoords."""
    try:
        lat, lon = map(float, coords_str.split(","))
        return GPSCoords(latitude=lat, longitude=lon)
    except Exception:
        return None

def _query_overpass(lat: float, lon: float, timeout_s: float = 10.0) -> bool:
    """Check if coordinates fall inside a building via Overpass API."""
    # Simple query to check if the point is within a building
    query = f"""
    [out:json][timeout:{int(timeout_s)}];
    is_in({lat},{lon})->.a;
    (
      way(pivot.a)[building];
      relation(pivot.a)[building];
    );
    out count;
    """
    url = f"https://overpass-api.de/api/interpreter?data={urllib.parse.quote(query.strip())}"
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "AssistiveNav/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
            # Overpass returns elements if found. Count > 0 means inside a building.
            elements = data.get("elements", [])
            for el in elements:
                if el.get("type") == "count":
                    return int(el.get("tags", {}).get("total", 0)) > 0
            return len(elements) > 0
    except Exception as e:
        print(f"[gps_location] Overpass query failed: {e}")
        return False
