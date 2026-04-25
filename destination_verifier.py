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
import ssl
import urllib.error
import urllib.request
from typing import Optional, Tuple
from urllib.parse import quote

# macOS Python often lacks root certificates — use unverified context for geocoding
_SSL_CTX = ssl._create_unverified_context()


GOOGLE_PLACES_TEXTSEARCH = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_GEOCODE = "https://maps.googleapis.com/maps/api/geocode/json"
ORS_GEOCODE = "https://api.openrouteservice.org/geocode/search"

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImQ4ODJkOGM4MGViNTRkODU5NDdjYTljMjI3MTkwMDkxIiwiaCI6Im11cm11cjY0In0="


GOOGLE_MAPS_KEY = "AIzaSyCaKUj75iDUS8tM6yxIGi5IJJMtPZqrUDo"


def _get_google_key() -> Optional[str]:
    return (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip() or GOOGLE_MAPS_KEY


def _get_ors_key() -> str:
    return (os.environ.get("OPENROUTESERVICE_API_KEY") or "").strip() or ORS_API_KEY


def search_destination(
    query: str,
    api_key: Optional[str] = None,
    timeout_s: float = 10.0,
) -> Optional[dict]:
    """
    Search for a destination using available APIs.

    Priority: Google Places -> Google Geocoding -> ORS Geocoding -> raw query
    Returns dict with keys: name, address, lat, lng
    """
    google_key = api_key or _get_google_key()
    
    # Try Google Places first (richest results)
    if google_key:
        result = _places_text_search(google_key, query, timeout_s)
        if result:
            return result
        # Try Google Geocoding
        result = _geocode_search(google_key, query, timeout_s)
        if result:
            return result

    # Try OpenRouteService geocoding
    ors_key = _get_ors_key()
    if ors_key:
        result = _ors_geocode_search(ors_key, query, timeout_s)
        if result:
            return result

    # No API worked — return the raw query as-is
    return {"name": query, "address": "", "lat": None, "lng": None}


def _places_text_search(api_key: str, query: str, timeout_s: float) -> Optional[dict]:
    """Search using Google Places Text Search API."""
    q = quote(query.strip(), safe="")
    k = quote(api_key.strip(), safe="")
    url = f"{GOOGLE_PLACES_TEXTSEARCH}?query={q}&key={k}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s, context=_SSL_CTX) as resp:
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
        with urllib.request.urlopen(req, timeout=timeout_s, context=_SSL_CTX) as resp:
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


def _ors_geocode_search(api_key: str, query: str, timeout_s: float) -> Optional[dict]:
    """Search using OpenRouteService Geocoding API."""
    q = quote(query.strip(), safe="")
    url = f"{ORS_GEOCODE}?text={q}&size=1"

    try:
        req = urllib.request.Request(
            url,
            headers={"Authorization": api_key, "Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_s, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[destination] ORS geocoding error: {e}")
        return None

    feats = data.get("features") or []
    if not feats:
        return None

    best = feats[0]
    props = best.get("properties") or {}
    geom = best.get("geometry") or {}
    coords = geom.get("coordinates") or []

    name = props.get("name") or props.get("label") or query
    address = props.get("label") or ""
    lat = coords[1] if len(coords) >= 2 else None
    lng = coords[0] if len(coords) >= 2 else None

    return {"name": name, "address": address, "lat": lat, "lng": lng}


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


def get_voice_confirmation(timeout: float = 5.0) -> bool:
    """
    Listen for a spoken 'yes' or 'no' via the microphone.
    Falls back to text input if voice recognition is unavailable.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        print("[system] Speech recognition not available, using text input")
        return get_text_confirmation()

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as source:
            print("\n[system] Say 'yes' to confirm or 'no' to change destination...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=3)

        # Use Google's free speech recognition
        text = recognizer.recognize_google(audio).strip().lower()
        print(f"[system] Heard: '{text}'")

        yes_words = {"yes", "yeah", "yep", "correct", "confirm", "ok", "okay", "sure", "right", "affirmative"}
        no_words = {"no", "nope", "nah", "wrong", "incorrect", "change", "cancel"}

        for word in yes_words:
            if word in text:
                return True
        for word in no_words:
            if word in text:
                return False

        # Couldn't determine — ask again via text
        print(f"[system] Didn't catch that. Falling back to text input.")
        return get_text_confirmation()

    except Exception as e:
        print(f"[system] Voice recognition error: {e}")
        print("[system] Falling back to text input")
        return get_text_confirmation()
