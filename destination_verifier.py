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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import quote

# macOS Python often lacks root certificates — use unverified context for geocoding
_SSL_CTX = ssl._create_unverified_context()


GOOGLE_PLACES_TEXTSEARCH = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_GEOCODE = "https://maps.googleapis.com/maps/api/geocode/json"
ORS_GEOCODE = "https://api.openrouteservice.org/geocode/search"

# API keys loaded from environment variables (set in .env file)
# Never hardcode keys here — they will be pushed to git


def _get_google_key() -> Optional[str]:
    env_key = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
    if env_key:
        return env_key

    app_json = Path(__file__).parent / "mobile" / "app.json"
    try:
        data = json.loads(app_json.read_text())
        key = str(((data.get("expo") or {}).get("extra") or {}).get("googleMapsApiKey") or "").strip()
        return key or None
    except Exception:
        return None


def _get_ors_key() -> Optional[str]:
    return (os.environ.get("OPENROUTESERVICE_API_KEY") or "").strip() or None


def search_destination(
    query: str,
    api_key: Optional[str] = None,
    timeout_s: float = 10.0,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
) -> Optional[dict]:
    """
    Search for a destination using available APIs.

    Priority: Google Places -> Google Geocoding -> ORS Geocoding -> raw query
    Returns dict with keys: name, address, lat, lng
    """
    google_key = api_key or _get_google_key()
    
    # Try Google Places first (richest results)
    if google_key:
        result = _places_text_search(google_key, query, timeout_s, lat, lng)
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


def _places_text_search(api_key: str, query: str, timeout_s: float, lat: Optional[float] = None, lng: Optional[float] = None) -> Optional[dict]:
    """Search using Google Places Text Search API."""
    q = quote(query.strip(), safe="")
    k = quote(api_key.strip(), safe="")
    url = f"{GOOGLE_PLACES_TEXTSEARCH}?query={q}&key={k}"
    if lat is not None and lng is not None:
        url += f"&location={lat},{lng}&radius=8047"

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
    Listen for a spoken 'yes' or 'no' via the microphone using ElevenLabs STT.
    Falls back to text input if voice recognition is unavailable.
    """
    ELEVENLABS_API_KEY = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    
    try:
        import speech_recognition as sr
    except ImportError:
        print("[system] Audio recording not available, using text input")
        return get_text_confirmation()

    try:
        time.sleep(0.3)
        
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5  # Snappier cut-off when user stops speaking

        print("\n[system] 🎤 Say 'correct', 'confirm', or 'yes' to proceed, or 'no' to change destination...")
        print('\a', end='', flush=True)  # Terminal beep to cue the user

        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5.0)
                wav_data = audio.get_wav_data()
        except sr.WaitTimeoutError:
            print("[system] Confirmation timed out (no speech detected).")
            return False
        except Exception as e:
            print(f"[system] Microphone error: {e}")
            return False
        
        # Send to ElevenLabs STT
        import requests
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            data={"model_id": "scribe_v1"},
            files={"file": ("audio.wav", wav_data, "audio/wav")},
            timeout=15,
            verify=False,  # macOS SSL workaround
        )
        
        if response.status_code == 200:
            import re
            text = response.json().get("text", "").strip().lower()
            
            # Clean up ElevenLabs STT hallucinations like "(music)" or "[coughing]"
            cleaned_text = re.sub(r'\([^)]*\)', '', text)
            cleaned_text = re.sub(r'\[[^\]]*\]', '', cleaned_text)
            cleaned_text = cleaned_text.strip()
            
            if text != cleaned_text:
                print(f"[system] Raw Heard: '{text}'")
            print(f"[system] Heard: '{cleaned_text}'")
            
            if not cleaned_text:
                print("[system] No speech detected. Falling back to text input.")
                return get_text_confirmation()
            
            yes_words = {"yes", "yeah", "yep", "yup", "ya", "correct", "confirm", "ok", "okay", "sure", "right", "affirmative", "absolutely", "indeed"}
            no_words = {"no", "nope", "nah", "wrong", "incorrect", "change", "cancel", "negative"}
            
            # Use word boundaries so "no" doesn't match "now" or "nothing"
            for word in yes_words:
                if re.search(rf'\b{word}\b', cleaned_text):
                    return True
            for word in no_words:
                if re.search(rf'\b{word}\b', cleaned_text):
                    return False
            
            print(f"[system] Didn't understand '{cleaned_text}'. Falling back to text input.")
            return get_text_confirmation()
        else:
            print(f"[system] ElevenLabs STT error: {response.status_code}")
            return get_text_confirmation()

    except Exception as e:
        print(f"[system] Voice recognition error: {e}")
        print("[system] Falling back to text input")
        return get_text_confirmation()
