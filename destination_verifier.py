"""
Destination Verifier — Google Maps lookup + audio confirmation.

Flow:
  1. User types (or speaks) a destination name
  2. We search Google Maps Places/Geocoding for the best match
  3. We announce the full result via TTS and ask for confirmation
  4. User types 'yes'/'no' (or says it)
  5. Only after 'yes' do we start obstacle detection + navigation
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional, Tuple
from urllib.parse import quote


GOOGLE_PLACES_TEXTSEARCH = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_GEOCODE = "https://maps.googleapis.com/maps/api/geocode/json"


def _get_google_key() -> Optional[str]:
    return (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip() or None


def search_destination(
    query: str,
    api_key: Optional[str] = None,
    timeout_s: float = 10.0,
) -> Optional[dict]:
    """
    Search Google Maps for a destination.

    Returns dict with keys: name, address, lat, lng
    or None if nothing found / no API key.
    """
    key = api_key or _get_google_key()
    if not key:
        # No API key — return the raw query as-is
        return {"name": query, "address": "", "lat": None, "lng": None}

    # Try Places Text Search first (richer results)
    result = _places_text_search(key, query, timeout_s)
    if result:
        return result

    # Fallback to Geocoding API
    result = _geocode_search(key, query, timeout_s)
    if result:
        return result

    # Nothing found
    return None


def _places_text_search(api_key: str, query: str, timeout_s: float) -> Optional[dict]:
    """Search using Google Places Text Search API."""
    q = quote(query.strip(), safe="")
    k = quote(api_key.strip(), safe="")
    url = f"{GOOGLE_PLACES_TEXTSEARCH}?query={q}&key={k}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[destination] Places API error: {e}")
        return None

    if data.get("status") != "OK":
        return None

    results = data.get("results") or []
    if not results:
        return None

    best = results[0]
    loc = (best.get("geometry") or {}).get("location") or {}
    return {
        "name": best.get("name", query),
        "address": best.get("formatted_address", ""),
        "lat": loc.get("lat"),
        "lng": loc.get("lng"),
    }


def _geocode_search(api_key: str, query: str, timeout_s: float) -> Optional[dict]:
    """Fallback using Google Geocoding API."""
    q = quote(query.strip(), safe="")
    k = quote(api_key.strip(), safe="")
    url = f"{GOOGLE_GEOCODE}?address={q}&key={k}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[destination] Geocoding API error: {e}")
        return None

    if data.get("status") != "OK":
        return None

    results = data.get("results") or []
    if not results:
        return None

    best = results[0]
    loc = ((best.get("geometry") or {}).get("location") or {})
    return {
        "name": query,
        "address": best.get("formatted_address", ""),
        "lat": loc.get("lat"),
        "lng": loc.get("lng"),
    }


def format_confirmation_message(result: dict) -> str:
    """Build a spoken confirmation message from the search result."""
    name = result.get("name", "")
    address = result.get("address", "")

    if address:
        return f"I found {name} at {address}. Is this correct?"
    else:
        return f"Navigating to {name}. Is this correct?"


def get_text_confirmation() -> bool:
    """Ask the user to confirm via terminal input. Returns True for yes."""
    print("\n[system] Type 'yes' to confirm or 'no' to re-enter destination:")
    try:
        response = input("> ").strip().lower()
        return response in ("yes", "y", "yeah", "yep", "correct", "confirm", "ok")
    except (EOFError, KeyboardInterrupt):
        return False
