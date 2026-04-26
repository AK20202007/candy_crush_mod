#!/usr/bin/env python3
"""
Thin-client streaming API: phones send JPEG frames; this machine runs YOLO + router.

Run (after pip install -r requirements.txt):

  python3 stream_server.py

Open http://localhost:8765 on this machine, or http://<LAN-IP>:8765 from another device.
Mobile browsers usually require HTTPS for camera access — use ``ngrok http 8765`` or
``uvicorn`` with TLS key/cert if testing from a phone.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from agentic_layer.models import RouteState

# Import vision (torch + ultralytics) only on first frame — otherwise the server
# binds very late and the phone appears to "stall" on the first page load.

# --- Global engine (one YOLO instance; sessions only change route text) ---

_engine_lock = threading.Lock()
_engine: Optional[Any] = None  # VisionSystem, typed lazily to avoid importing vision at startup


def _get_engine() -> Any:
    """Lazy-load YOLO (slow first time). Do not call while holding ``_engine_lock``."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            from vision import VisionConfig, VisionSystem

            def _noop(_decision):
                pass

            cfg = VisionConfig()
            _engine = VisionSystem(config=cfg, on_decision=_noop)
        return _engine


@dataclass
class _Session:
    destination: str
    destination_address: Optional[str]
    destination_lat: Optional[float]
    destination_lng: Optional[float]
    created: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    horizontal_accuracy_m: Optional[float] = None
    location_type: Optional[str] = None
    route_steps: Optional[list[str]] = None
    route_step_targets: Optional[list[tuple[float, float]]] = None  # (lat, lon) per step if available
    route_index: int = 0
    mapping_state: str = "done"
    last_progress_lat: Optional[float] = None
    last_progress_lng: Optional[float] = None
    last_route_prompt_s: float = 0.0
    last_route_refresh_s: float = 0.0
    exit_seeking: bool = False


_sessions: Dict[str, _Session] = {}
_sessions_lock = threading.Lock()

app = FastAPI(title="Assistive Nav — frame stream")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # ``credentials=True`` is invalid with wildcard origins and can stall Safari.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIENT_DIR = Path(__file__).resolve().parent / "stream_client"

# Keep behavior consistent with main.py: load API keys from local .env if present.
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())


def _build_frame_response(engine: Any, decision) -> Dict[str, Any]:
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [_json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    ctx = engine._last_frame_context
    detections_out: list = []
    if ctx and ctx.detections:
        detections_out = [_json_safe(d.to_dict()) for d in ctx.detections[:32]]
    return {
        "decision": _json_safe(decision.to_dict()),
        "detections": detections_out,
        "speak": bool(getattr(decision, "should_speak", False) and decision.message),
    }


def _synthesize_elevenlabs_mp3(text: str) -> bytes:
    """
    Synthesize speech using ElevenLabs and return MP3 bytes.
    Uses the same env keys as main.py / elevenlabs_speech.py.
    """
    api_key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    voice_id = (os.environ.get("ELEVENLABS_VOICE_ID") or "").strip() or "21m00Tcm4TlvDq8ikWAM"
    model_id = (os.environ.get("ELEVENLABS_MODEL") or "").strip() or "eleven_flash_v2_5"
    try:
        from elevenlabs import ElevenLabs
    except Exception as exc:
        raise RuntimeError(f"elevenlabs package unavailable: {exc}") from exc

    client = ElevenLabs(api_key=api_key)
    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format="mp3_44100_128",
    )
    audio_data = b""
    for chunk in audio_stream:
        audio_data += chunk
    if not audio_data:
        raise RuntimeError("ElevenLabs returned empty audio")
    return audio_data


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    r = 6371000.0
    p1 = radians(lat1)
    p2 = radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _to_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _build_route_steps_for_session(session: _Session) -> tuple[list[str], str]:
    """Google Maps turn-by-turn only for mobile navigation."""
    google_key = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
    if not google_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is required for Google turn-by-turn navigation")
    if session.latitude is None or session.longitude is None:
        raise RuntimeError("Current GPS location is required before creating a route")
    if not session.destination:
        raise RuntimeError("Destination is required before creating a route")

    from routing import google_walking_route_steps

    raw_steps = google_walking_route_steps(
        google_key,
        (session.longitude, session.latitude),
        session.destination,
    )
    if not raw_steps:
        raise RuntimeError("Google Maps returned no walking route steps")

    session.route_step_targets = [(s.end_lat, s.end_lon) for s in raw_steps]
    steps = [f"Starting outdoor navigation to {session.destination}."] + [
        s.instruction for s in raw_steps
    ]
    return (steps, "done")


def _refresh_route_if_needed(session: _Session, force: bool = False) -> None:
    now = time.time()
    if not force and now - session.last_route_refresh_s < 20.0:
        return
    if session.latitude is None or session.longitude is None or not session.destination:
        return
    google_key = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
    if not google_key:
        return
    try:
        steps, mapping_state = _build_route_steps_for_session(session)
        if steps:
            session.route_steps = steps
            session.mapping_state = mapping_state
            # Step 0 is intro; keep active guidance on first actionable step.
            if session.route_step_targets and len(steps) > 1:
                session.route_index = 1
            elif session.route_index >= len(steps):
                session.route_index = max(0, len(steps) - 1)
        session.last_route_refresh_s = now
    except Exception as exc:
        print(f"[stream/route] refresh failed: {exc}")


def _transcribe_with_elevenlabs(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """
    Transcribe user speech with ElevenLabs STT.
    Returns recognized text (possibly empty string).
    """
    api_key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    if not audio_bytes:
        raise RuntimeError("empty audio")

    files = {
        "file": (filename, io.BytesIO(audio_bytes), "application/octet-stream"),
    }
    data = {"model_id": "scribe_v1"}
    resp = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": api_key},
        data=data,
        files=files,
        timeout=45,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"STT HTTP {resp.status_code}: {resp.text[:200]}")
    payload = resp.json()
    return str(payload.get("text") or "").strip()


def _process_jpeg_bytes(jpeg: bytes, session: _Session, engine: Any) -> Dict[str, Any]:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode JPEG")

    route: Optional[RouteState] = None
    if session.destination:
        route_msg = None
        if session.exit_seeking:
            route_msg = "Leave the room first. Stand still, turn 360 degrees slowly, and scan for a door or exit sign."
        elif session.route_steps and 0 <= session.route_index < len(session.route_steps):
            route_msg = session.route_steps[session.route_index]
        elif session.destination_address:
            route_msg = f"Navigating to {session.destination} ({session.destination_address})."
        else:
            route_msg = f"Head toward {session.destination}."
        route = RouteState(
            active=True,
            destination=session.destination,
            next_instruction=route_msg,
            next_turn_distance_m=None,
            off_route=False,
            exit_seeking=session.exit_seeking,
            mapping_state=session.mapping_state,
        )
    original_location_type = getattr(engine._cfg, "location_type", "indoor")
    if session.location_type:
        engine._cfg.location_type = session.location_type
    try:
        decision = engine.process_frame_numpy(frame, route=route)
        payload = _build_frame_response(engine, decision)
        ctx = getattr(engine, "_last_frame_context", None)
        if ctx and ctx.route:
            session.exit_seeking = bool(getattr(ctx.route, "exit_seeking", session.exit_seeking))
            session.mapping_state = str(getattr(ctx.route, "mapping_state", session.mapping_state))
        payload["session"] = {
            "destination": session.destination,
            "destination_address": session.destination_address,
            "destination_lat": session.destination_lat,
            "destination_lng": session.destination_lng,
            "route_index": session.route_index,
            "route_steps_total": len(session.route_steps or []),
            "route_current_instruction": route_msg,
            "route_provider": "google_directions" if session.route_step_targets else "planner_fallback",
            "exit_seeking": session.exit_seeking,
            "mapping_state": session.mapping_state,
            "location": (
                {
                    "latitude": session.latitude,
                    "longitude": session.longitude,
                    "horizontal_accuracy_m": session.horizontal_accuracy_m,
                    "location_type": session.location_type,
                }
                if session.latitude is not None and session.longitude is not None
                else None
            ),
        }
        return payload
    finally:
        engine._cfg.location_type = original_location_type


@app.get("/ping")
def ping() -> Dict[str, str]:
    """Tiny response for “is the laptop reachable?” — no ML imports."""
    return {"status": "ok"}


@app.get("/health")
def health() -> Dict[str, Any]:
    with _engine_lock:
        loaded = _engine is not None
    return {"ok": True, "model_loaded": loaded}


@app.post("/stream/session")
def create_session(
    destination: str = Form(...),
    latitude: Optional[float] = Form(default=None),
    longitude: Optional[float] = Form(default=None),
    indoor_start: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    _get_engine()

    dest = (destination or "").strip()
    if not dest:
        raise HTTPException(status_code=400, detail="destination required")

    resolved = {"name": dest, "address": "", "lat": None, "lng": None}
    try:
        from destination_verifier import search_destination

        resolved = search_destination(dest, lat=latitude, lng=longitude) or resolved
        # Guard against "destination snapped to where I already am" when the
        # GPS-biased lookup picks a nearby entrance/POI.
        if latitude is not None and longitude is not None:
            rlat = _to_float_or_none(resolved.get("lat"))
            rlng = _to_float_or_none(resolved.get("lng"))
            if rlat is not None and rlng is not None:
                snapped_m = _haversine_m(float(latitude), float(longitude), rlat, rlng)
                if snapped_m < 120.0:
                    unbiased = search_destination(dest, lat=None, lng=None) or resolved
                    ulat = _to_float_or_none(unbiased.get("lat"))
                    ulng = _to_float_or_none(unbiased.get("lng"))
                    if ulat is not None and ulng is not None:
                        unbiased_m = _haversine_m(float(latitude), float(longitude), ulat, ulng)
                        # Prefer unbiased result when it is meaningfully farther
                        # than the snapped local result.
                        if unbiased_m - snapped_m > 120.0:
                            resolved = unbiased
    except Exception as exc:
        print(f"[stream/session] destination lookup failed: {exc}")

    loc_type = None
    if latitude is not None and longitude is not None:
        try:
            from location_service import is_indoors

            loc_type = "indoor" if is_indoors(float(latitude), float(longitude), timeout_s=6.0) else "sidewalk"
        except Exception as exc:
            print(f"[stream/session] indoor check failed: {exc}")

    sid = str(uuid.uuid4())
    destination_name = str(resolved.get("name") or dest)
    destination_address = str(resolved.get("address") or "")
    destination_lat = _to_float_or_none(resolved.get("lat"))
    destination_lng = _to_float_or_none(resolved.get("lng"))

    start_indoors = str(indoor_start or "").strip().lower() in {"yes", "true", "1", "indoor"}
    draft = _Session(
        destination=destination_name,
        destination_address=destination_address,
        destination_lat=destination_lat,
        destination_lng=destination_lng,
        created=time.time(),
        latitude=float(latitude) if latitude is not None else None,
        longitude=float(longitude) if longitude is not None else None,
        location_type=loc_type,
        exit_seeking=start_indoors,
    )
    try:
        steps, mapping_state = _build_route_steps_for_session(draft)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    draft.route_steps = steps
    draft.mapping_state = "pending" if start_indoors else mapping_state
    if draft.route_step_targets and len(draft.route_steps or []) > 1:
        draft.route_index = 1
    draft.last_route_refresh_s = time.time()
    if draft.latitude is not None and draft.longitude is not None:
        draft.last_progress_lat = draft.latitude
        draft.last_progress_lng = draft.longitude

    with _sessions_lock:
        _sessions[sid] = draft
    return {
        "session_id": sid,
        "destination": draft.destination,
        "destination_address": draft.destination_address,
        "destination_lat": draft.destination_lat,
        "destination_lng": draft.destination_lng,
        "destination_distance_m": (
            _haversine_m(float(latitude), float(longitude), draft.destination_lat, draft.destination_lng)
            if latitude is not None
            and longitude is not None
            and draft.destination_lat is not None
            and draft.destination_lng is not None
            else None
        ),
        "gps_bias_used": latitude is not None and longitude is not None,
        "route_steps_total": len(draft.route_steps or []),
        "route_first_instruction": (draft.route_steps or [f"Head toward {draft.destination}."])[0],
        "mapping_state": draft.mapping_state,
        "exit_seeking": draft.exit_seeking,
    }


@app.post("/stream/location")
def update_session_location(
    session_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    horizontal_accuracy_m: Optional[float] = Form(default=None),
    location_type: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session_id")
        session.latitude = float(latitude)
        session.longitude = float(longitude)
        session.horizontal_accuracy_m = (
            float(horizontal_accuracy_m) if horizontal_accuracy_m is not None else None
        )
        normalized = (location_type or "").strip().lower()
        if normalized in {"indoor", "hallway", "room", "corridor", "building"}:
            session.location_type = "indoor"
        elif normalized in {"outdoor", "sidewalk", "street", "street_crossing"}:
            session.location_type = "sidewalk"
        elif normalized:
            session.location_type = normalized

        # Keep route instructions fresh from current GPS, similar to main.py's
        # current-instruction model but with mobile dynamic updates.
        _refresh_route_if_needed(session)

        # Route progression:
        # Advance only when Google step targets exist and user reaches the
        # current target point. This prevents drifting into generic fallback
        # terminal prompts ("approaching ...") when no concrete targets exist.
        if session.route_steps and session.route_index < len(session.route_steps) - 1:
            # Skip intro line once GPS updates start.
            if session.route_index == 0:
                session.route_index = 1
            if (
                session.route_step_targets
                and (session.route_index - 1) < len(session.route_step_targets)
            ):
                target_lat, target_lon = session.route_step_targets[session.route_index - 1]
                dist_to_target = _haversine_m(
                    session.latitude,
                    session.longitude,
                    target_lat,
                    target_lon,
                )
                if dist_to_target <= 22.0:
                    session.route_index += 1
                    session.last_progress_lat = session.latitude
                    session.last_progress_lng = session.longitude
                    session.last_route_prompt_s = 0.0
        elif session.last_progress_lat is None or session.last_progress_lng is None:
            session.last_progress_lat = session.latitude
            session.last_progress_lng = session.longitude
    return {
        "ok": True,
        "session_id": session_id,
        "location": {
            "latitude": session.latitude,
            "longitude": session.longitude,
            "horizontal_accuracy_m": session.horizontal_accuracy_m,
            "location_type": session.location_type,
        },
        "route_index": session.route_index,
        "route_steps_total": len(session.route_steps or []),
        "route_current_instruction": (
            (session.route_steps or [f"Head toward {session.destination}."])[session.route_index]
            if (session.route_steps and 0 <= session.route_index < len(session.route_steps))
            else f"Head toward {session.destination}."
        ),
        "route_provider": "google_directions" if session.route_step_targets else "planner_fallback",
        "exit_seeking": session.exit_seeking,
        "mapping_state": session.mapping_state,
    }


@app.post("/stream/frame")
async def ingest_frame(
    session_id: str = Form(...),
    image: UploadFile = File(...),
) -> JSONResponse:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session_id")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")

    loop = asyncio.get_running_loop()

    def _run() -> Dict[str, Any]:
        engine = _get_engine()
        with _engine_lock:
            return _process_jpeg_bytes(data, session, engine)

    try:
        payload = await loop.run_in_executor(None, _run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(payload)


@app.post("/stream/tts")
async def tts(text: str = Form(...)) -> Response:
    msg = (text or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="text required")
    if len(msg) > 600:
        raise HTTPException(status_code=400, detail="text too long")
    loop = asyncio.get_running_loop()
    try:
        audio_data = await loop.run_in_executor(None, _synthesize_elevenlabs_mp3, msg)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"tts unavailable: {exc}") from exc
    return Response(content=audio_data, media_type="audio/mpeg")


@app.post("/stream/stt")
async def stt(audio: UploadFile = File(...)) -> JSONResponse:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio")
    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(None, _transcribe_with_elevenlabs, data, audio.filename or "audio.webm")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"stt unavailable: {exc}") from exc
    return JSONResponse({"text": text})


@app.get("/")
def root() -> FileResponse:
    index = CLIENT_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail="stream_client/index.html missing")
    return FileResponse(index, headers={"Cache-Control": "no-store"})


if CLIENT_DIR.is_dir():
    app.mount("/stream_client", StaticFiles(directory=str(CLIENT_DIR)), name="stream_client")


def main() -> None:
    parser = argparse.ArgumentParser(description="JPEG frame streaming server for assistive nav")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument("--ssl-keyfile", default=None, help="Optional TLS key for HTTPS (mobile camera)")
    parser.add_argument("--ssl-certfile", default=None, help="Optional TLS cert for HTTPS")
    args = parser.parse_args()

    import uvicorn

    kwargs: Dict[str, Any] = {"app": app, "host": args.host, "port": args.port, "log_level": "info"}
    if args.ssl_keyfile and args.ssl_certfile:
        kwargs["ssl_keyfile"] = args.ssl_keyfile
        kwargs["ssl_certfile"] = args.ssl_certfile

    uvicorn.run(**kwargs)


if __name__ == "__main__":
    main()
