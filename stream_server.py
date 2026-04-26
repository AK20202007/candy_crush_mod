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
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
    created: float


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


def _build_frame_response(engine: Any, decision) -> Dict[str, Any]:
    ctx = engine._last_frame_context
    detections_out: list = []
    if ctx and ctx.detections:
        detections_out = [d.to_dict() for d in ctx.detections[:32]]
    return {
        "decision": decision.to_dict(),
        "detections": detections_out,
        "speak": bool(getattr(decision, "should_speak", False) and decision.message),
    }


def _process_jpeg_bytes(jpeg: bytes, destination: Optional[str], engine: Any) -> Dict[str, Any]:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode JPEG")

    route: Optional[RouteState] = None
    if destination:
        route = RouteState(
            active=True,
            destination=destination,
            next_instruction=None,
            next_turn_distance_m=None,
            off_route=False,
        )
    decision = engine.process_frame_numpy(frame, route=route)
    return _build_frame_response(engine, decision)


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
def create_session(destination: str = Form(...)) -> Dict[str, str]:
    dest = (destination or "").strip()
    if not dest:
        raise HTTPException(status_code=400, detail="destination required")
    sid = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[sid] = _Session(destination=dest, created=time.time())
    return {"session_id": sid, "destination": dest}


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
            return _process_jpeg_bytes(data, session.destination, engine)

    try:
        payload = await loop.run_in_executor(None, _run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(payload)


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
