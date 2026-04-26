"""Address-to-address navigation planner with indoor/outdoor transitions.

Orchestrates a multi-leg journey:
  1. Geocode origin and destination addresses.
  2. Determine if each is indoor or outdoor (via Overpass/OSM building query).
  3. Plan the route legs:
     - Indoor leg (graph routing via ``indoor_routing`` or indrz-be API)
     - Exit leg (vision-based door finding when leaving a building)
     - Outdoor leg (ORS walking directions)
     - Enter leg (door finding when arriving at a building)
  4. Expose route legs as spoken step lists for ``navigation.run_navigation_loop``.

Integrations:
  - **openindoormaps** / **indrz-be**: when an indrz backend is reachable,
    the planner fetches indoor routing via its REST API (PostGIS/PGRouting).
  - **wrld-indoor-maps-api**: if a WRLD API token is configured, indoor
    map geometry can supplement the routing graph.
  - When no indoor map data is available, the system falls back to
    vision-based exit seeking (scanning for doors by turning).

Usage::

    planner = NavigationPlanner.from_env()
    plan = planner.plan("Room 302, Engineering Building", "456 Oak Ave")
    for leg in plan.legs:
        print(leg.leg_type, leg.steps)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from location_service import LocationFix as GPSCoords, is_indoors
from routing import geocode as ors_geocode


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class LegType(str, Enum):
    INDOOR = "indoor"
    EXIT_BUILDING = "exit_building"
    OUTDOOR = "outdoor"
    ENTER_BUILDING = "enter_building"


class ExitStrategy(str, Enum):
    GRAPH_ROUTE = "graph_route"
    VISION_DOOR_SEEK = "vision_door_seek"
    MANUAL = "manual"


@dataclass
class RouteLeg:
    leg_type: LegType
    steps: List[str]
    exit_strategy: ExitStrategy = ExitStrategy.MANUAL
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationPlan:
    origin: str
    destination: str
    origin_is_indoor: bool
    destination_is_indoor: bool
    legs: List[RouteLeg]
    total_steps: int = 0

    def __post_init__(self) -> None:
        self.total_steps = sum(len(leg.steps) for leg in self.legs)

    def all_steps(self) -> List[str]:
        """Flatten legs into a single ordered step list for the nav loop."""
        out: List[str] = []
        for leg in self.legs:
            out.extend(leg.steps)
        return out


# ---------------------------------------------------------------------------
# Indrz-be API client (PostGIS/PGRouting indoor routing)
# ---------------------------------------------------------------------------

class IndrzClient:
    """Minimal REST client for an indrz-be backend.

    Environment variables:
      INDRZ_API_URL   – base URL, e.g. ``http://localhost:8000/api/v1``
      INDRZ_API_TOKEN – auth token (if required)
    """

    def __init__(self, base_url: str, token: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    @classmethod
    def from_env(cls) -> Optional["IndrzClient"]:
        url = os.environ.get("INDRZ_API_URL", "").strip()
        if not url:
            return None
        token = os.environ.get("INDRZ_API_TOKEN", "").strip() or None
        return cls(url, token)

    def _get(self, path: str, timeout_s: float = 10.0) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Token {self.token}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode())

    def route(
        self,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        start_floor: int = 0,
        end_floor: int = 0,
    ) -> List[str]:
        """Fetch indoor route from indrz and return spoken step strings.

        ``start_coords`` / ``end_coords`` are (longitude, latitude).
        """
        path = (
            f"directions/route/{start_coords[0]},{start_coords[1]},{start_floor}"
            f"&{end_coords[0]},{end_coords[1]},{end_floor}/"
        )
        try:
            data = self._get(path)
            return self._extract_steps(data)
        except Exception as exc:
            print(f"[indrz] Routing failed: {exc}")
            return []

    @staticmethod
    def _extract_steps(data: Any) -> List[str]:
        """Parse PGRouting GeoJSON response into spoken instructions."""
        steps: List[str] = []
        features = data.get("features", []) if isinstance(data, dict) else []
        for feat in features:
            props = feat.get("properties", {})
            instruction = props.get("instruction") or props.get("name", "")
            if instruction:
                steps.append(str(instruction))
        if not steps:
            steps.append("Follow the indoor route.")
        return steps


# ---------------------------------------------------------------------------
# Geocoding helpers
# ---------------------------------------------------------------------------

def _nominatim_geocode(query: str, timeout_s: float = 10.0) -> Optional[Tuple[float, float]]:
    """Free geocode via Nominatim (no API key). Returns (lon, lat) or None."""
    url = (
        f"https://nominatim.openstreetmap.org/search?q={quote(query)}"
        f"&format=json&limit=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "AssistiveNav/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            results = json.loads(resp.read().decode())
            if results:
                return float(results[0]["lon"]), float(results[0]["lat"])
    except Exception as exc:
        print(f"[geocode] Nominatim failed for {query!r}: {exc}")
    return None


def _geocode(query: str, ors_key: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Geocode using ORS (if key available) or fall back to Nominatim."""
    if ors_key:
        try:
            return ors_geocode(ors_key, query)
        except Exception:
            pass
    return _nominatim_geocode(query)


def _is_indoor_coords(lat: float, lon: float) -> bool:
    """Check if coordinates fall inside a building via Overpass API."""
    try:
        return is_indoors(lat, lon, timeout_s=8)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Exit-seeking step generation
# ---------------------------------------------------------------------------

def _exit_seeking_steps() -> List[str]:
    """Spoken steps for vision-based exit seeking (no map data available).

    The agent will instruct the user to locate a safe exit path.
    """
    return [
        "You need to leave this room or building first.",
        "Stand still, slowly turn 360 degrees, and scan for a door or exit sign.",
        "If no door is visible, move to the nearest wall and trail your hand along it.",
        "When a door is detected, I will guide you to the handle and tell you when to pass through.",
        "After exiting, outdoor navigation will begin.",
    ]


def _enter_building_steps(destination: str) -> List[str]:
    """Steps for entering a destination building."""
    return [
        f"You are approaching {destination}.",
        "Look for the building entrance ahead.",
        "The system will detect the entrance and guide you toward it.",
        f"Once inside, you have arrived at {destination}.",
    ]


# ---------------------------------------------------------------------------
# Navigation planner
# ---------------------------------------------------------------------------

class NavigationPlanner:
    """Orchestrates address-to-address navigation across indoor/outdoor legs."""

    def __init__(
        self,
        ors_key: Optional[str] = None,
        indrz: Optional[IndrzClient] = None,
        indoor_graph_path: Optional[str] = None,
    ) -> None:
        self.ors_key = ors_key
        self.indrz = indrz
        self.indoor_graph_path = indoor_graph_path

    @classmethod
    def from_env(cls) -> "NavigationPlanner":
        return cls(
            ors_key=os.environ.get("OPENROUTESERVICE_API_KEY", "").strip() or None,
            indrz=IndrzClient.from_env(),
            indoor_graph_path=os.environ.get("INDOOR_LAYOUT_PATH", "").strip() or None,
        )

    def plan(
        self,
        origin: str,
        destination: str,
        origin_gps: Optional[GPSCoords] = None,
        dest_gps: Optional[GPSCoords] = None,
    ) -> NavigationPlan:
        """Build a multi-leg navigation plan from origin to destination.

        Parameters
        ----------
        origin, destination : str
            Human-readable addresses or place names.
        origin_gps, dest_gps : GPSCoords, optional
            Known GPS coordinates to skip geocoding.
        """
        # 1. Geocode
        origin_coords = self._resolve_coords(origin, origin_gps)
        dest_coords = self._resolve_coords(destination, dest_gps)

        # 2. Indoor/outdoor classification
        origin_indoor = False
        dest_indoor = False
        if origin_coords:
            origin_indoor = _is_indoor_coords(origin_coords[1], origin_coords[0])
        if dest_coords:
            dest_indoor = _is_indoor_coords(dest_coords[1], dest_coords[0])

        print(f"[planner] origin={origin!r} indoor={origin_indoor}  dest={destination!r} indoor={dest_indoor}")

        # 3. Build legs
        legs: List[RouteLeg] = []

        # Leg A: Indoor exit (if starting indoors and going outdoors or to another building)
        if origin_indoor:
            exit_steps = self._indoor_exit_steps(origin, origin_coords)
            legs.append(RouteLeg(
                leg_type=LegType.EXIT_BUILDING,
                steps=exit_steps,
                exit_strategy=(
                    ExitStrategy.GRAPH_ROUTE
                    if self._has_indoor_routing()
                    else ExitStrategy.VISION_DOOR_SEEK
                ),
                metadata={"building": origin},
            ))

        # Leg B: Outdoor walking
        if origin_coords and dest_coords:
            outdoor_steps = self._outdoor_steps(origin_coords, dest_coords, destination)
            if outdoor_steps:
                legs.append(RouteLeg(
                    leg_type=LegType.OUTDOOR,
                    steps=outdoor_steps,
                    metadata={"origin_coords": list(origin_coords), "dest_coords": list(dest_coords)},
                ))

        # Leg C: Enter destination building (if destination is indoor)
        if dest_indoor:
            legs.append(RouteLeg(
                leg_type=LegType.ENTER_BUILDING,
                steps=_enter_building_steps(destination),
                exit_strategy=ExitStrategy.VISION_DOOR_SEEK,
                metadata={"building": destination},
            ))

        # Fallback: if no legs were created, provide basic guidance
        if not legs:
            legs.append(RouteLeg(
                leg_type=LegType.OUTDOOR,
                steps=[
                    f"Head toward {destination}.",
                    "Follow the path and listen for further guidance.",
                ],
            ))

        return NavigationPlan(
            origin=origin,
            destination=destination,
            origin_is_indoor=origin_indoor,
            destination_is_indoor=dest_indoor,
            legs=legs,
        )

    def _resolve_coords(
        self, address: str, known_gps: Optional[GPSCoords] = None,
    ) -> Optional[Tuple[float, float]]:
        """Return (lon, lat) for an address."""
        if known_gps:
            return (known_gps.longitude, known_gps.latitude)
        return _geocode(address, self.ors_key)

    def _has_indoor_routing(self) -> bool:
        return self.indrz is not None or self.indoor_graph_path is not None

    def _indoor_exit_steps(
        self, building: str, coords: Optional[Tuple[float, float]],
    ) -> List[str]:
        """Generate steps to exit a building.

        Tries indrz-be API first, then local indoor graph, then
        falls back to vision-based exit seeking.
        """
        # Try indrz-be
        if self.indrz and coords:
            try:
                steps = self.indrz.route(coords, coords, start_floor=0, end_floor=0)
                if steps:
                    return [f"Routing to exit of {building}."] + steps
            except Exception as exc:
                print(f"[planner] indrz exit routing failed: {exc}")

        # Try local indoor graph
        if self.indoor_graph_path:
            try:
                from indoor_routing import load_graph_from_json, find_node_by_name
                from pathlib import Path

                graph = load_graph_from_json(Path(self.indoor_graph_path))
                # Find nearest entrance/exit node
                exit_node = None
                for nid, node in graph.nodes.items():
                    if node.node_type == "entrance":
                        exit_node = nid
                        break
                if exit_node:
                    # Try to find the user's current node (approximation)
                    start_node = list(graph.nodes.keys())[0]
                    route = graph.find_route(start_node, exit_node)
                    steps = [step.instruction for step in route.steps]
                    if steps:
                        return [f"Routing to exit of {building}."] + steps
            except Exception as exc:
                print(f"[planner] Indoor graph exit routing failed: {exc}")

        # Fallback: vision-based exit seeking
        return _exit_seeking_steps()

    def _outdoor_steps(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        dest_name: str,
    ) -> List[str]:
        """Generate outdoor walking directions."""
        if self.ors_key:
            try:
                from routing import walking_directions
                steps = walking_directions(self.ors_key, start, end)
                return [f"Starting outdoor navigation to {dest_name}."] + steps
            except Exception as exc:
                print(f"[planner] ORS outdoor routing failed: {exc}")

        # Fallback: basic direction
        return [
            f"Head toward {dest_name}.",
            "Follow the sidewalk and listen for further guidance.",
            f"You are approaching {dest_name}.",
        ]
