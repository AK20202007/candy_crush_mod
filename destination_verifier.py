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
    Listen for a spoken 'yes' or 'no' via the microphone using ElevenLabs STT.
    Falls back to text input if voice recognition is unavailable.
    """
    ELEVENLABS_API_KEY = "sk_b7ad54c02663e087df7e214908660e306cc0f1604ccc750c"
    
    try:
        import sounddevice as sd
        import numpy as np
        import wave
        import tempfile
    except ImportError:
        print("[system] Audio recording not available, using text input")
        return get_text_confirmation()

    try:
        # Make sure no audio is playing before recording
        try:
            sd.stop()
        except Exception:
            pass
        time.sleep(0.3)
        
        print("\n[system] 🎤 Say 'yes' to confirm or 'no' to change destination...")
        
        # Record 3 seconds of audio (enough for "yes" or "no")
        samplerate = 16000
        recording = sd.rec(
            int(samplerate * 3),
            samplerate=samplerate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        
        # Save to temp wav file
        temp_path = tempfile.mktemp(suffix=".wav")
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(samplerate)
            wav_file.writeframes(recording.tobytes())
        
        # Send to ElevenLabs STT
        import requests
        with open(temp_path, 'rb') as f:
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                data={"model_id": "scribe_v1"},
                files={"file": f},
                timeout=15
            )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass
        
        if response.status_code == 200:
            text = response.json().get("text", "").strip().lower()
            print(f"[system] Heard: '{text}'")
            
            if not text:
                print("[system] No speech detected. Falling back to text input.")
                return get_text_confirmation()
            
            yes_words = {"yes", "yeah", "yep", "correct", "confirm", "ok", "okay", "sure", "right", "affirmative"}
            no_words = {"no", "nope", "nah", "wrong", "incorrect", "change", "cancel"}
            
            for word in yes_words:
                if word in text:
                    return True
            for word in no_words:
                if word in text:
                    return False
            
            print(f"[system] Didn't understand '{text}'. Falling back to text input.")
            return get_text_confirmation()
        else:
            print(f"[system] ElevenLabs STT error: {response.status_code}")
            return get_text_confirmation()

    except Exception as e:
        print(f"[system] Voice recognition error: {e}")
        print("[system] Falling back to text input")
        return get_text_confirmation()
