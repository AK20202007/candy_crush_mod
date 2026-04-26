"""HTTP API wrapper for the local door-handle detector.

Run this locally, then expose it with Cloudflare Tunnel:

    python -B cloudflare_vision_api.py --door-parts-model /path/to/best.pt --door-parts-yolov5-dir /path/to/yolov5
    cloudflared tunnel --url http://localhost:8000

The mobile app can use the generated trycloudflare.com URL as its navigation
vision API base URL. Cloudflare only transports requests; PyTorch inference
still runs on the local machine.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import cv2
import numpy as np

from agentic_layer.models import MotionState, RouteState, UserState
from vision import VisionConfig, VisionSystem

MAX_BODY_BYTES = 8 * 1024 * 1024


class SharedRequestState:
    def __init__(self) -> None:
        self.route = RouteState(active=True, exit_seeking=True, mapping_state="pending")
        self.motion = MotionState(is_moving=False, speed_mps=0.0)
        self.user = UserState()
        self.lock = threading.Lock()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _decode_frame(payload: Dict[str, Any]) -> np.ndarray:
    image = str(payload.get("image_base64") or payload.get("image") or "")
    if "," in image and image.lower().startswith("data:"):
        image = image.split(",", 1)[1]
    if not image:
        raise ValueError("Missing image_base64")
    try:
        raw = base64.b64decode(image, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image: {exc}") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


def _route_from_payload(payload: Dict[str, Any]) -> RouteState:
    route = RouteState.from_dict(payload.get("route") or {})
    indoor_start = str(payload.get("indoor_start", "")).lower().strip()
    if indoor_start in {"yes", "true", "1"}:
        route.active = True
        route.exit_seeking = True
        route.mapping_state = route.mapping_state if route.mapping_state != "done" else "pending"
        route.next_instruction = route.next_instruction or (
            "Leave the room first. Stand still, turn 360 degrees slowly, and scan for a door or exit sign."
        )
    return route


def build_handler(vision: VisionSystem, state: SharedRequestState):
    class CloudflareVisionHandler(BaseHTTPRequestHandler):
        server_version = "LAHacksCloudflareVision/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[cloudflare-api] {self.address_string()} - {fmt % args}")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/health"}:
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "service": "lahacks-cloudflare-vision",
                        "endpoints": ["/health", "/api/vision/frame"],
                        "door_parts_model": bool(getattr(vision, "_door_parts_model", None)),
                    },
                )
                return
            _json_response(self, 404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/vision/frame":
                _json_response(self, 404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                _json_response(self, 400, {"ok": False, "error": "Invalid Content-Length"})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                _json_response(self, 413, {"ok": False, "error": "Request body too large or empty"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                frame = _decode_frame(payload)
            except Exception as exc:
                _json_response(self, 400, {"ok": False, "error": str(exc)})
                return

            with state.lock:
                state.route = _route_from_payload(payload)
                state.motion = MotionState.from_dict(payload.get("motion"))
                state.user = UserState.from_dict(payload.get("user"))
                scene = payload.get("scene") or {}
                location_type = str(scene.get("location_type") or payload.get("location_type") or "").strip()
                old_location_type = vision._cfg.location_type
                if location_type:
                    vision._cfg.location_type = location_type
                try:
                    h, w = frame.shape[:2]
                    decision = vision._process_frame(frame, w, h)
                finally:
                    vision._cfg.location_type = old_location_type

                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "decision": decision.model_dump(),
                        "route": state.route.model_dump(),
                    },
                )

    return CloudflareVisionHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloudflare-tunnel-friendly local vision API")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--model", default=os.environ.get("YOLO_MODEL", "yolov8n.pt"))
    parser.add_argument("--door-parts-model", default=os.environ.get("DOOR_PARTS_MODEL_PATH"))
    parser.add_argument("--door-parts-yolov5-dir", default=os.environ.get("DOOR_PARTS_YOLOV5_DIR"))
    parser.add_argument("--door-parts-conf", type=float, default=float(os.environ.get("DOOR_PARTS_CONF", "0.40")))
    parser.add_argument("--camera-mount", choices=["hand", "head"], default=os.environ.get("CAMERA_MOUNT", "hand"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = SharedRequestState()
    config = VisionConfig(
        model_path=args.model,
        show_preview=False,
        camera_mount=args.camera_mount,
        door_parts_model_path=args.door_parts_model,
        door_parts_yolov5_dir=args.door_parts_yolov5_dir,
        door_parts_conf=args.door_parts_conf,
        assume_moving=False,
        location_type="room",
    )
    vision = VisionSystem(
        on_decision=None,
        config=config,
        route_provider=lambda: state.route,
        motion_provider=lambda: state.motion,
        user_provider=lambda: state.user,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(vision, state))
    print(f"[cloudflare-api] Listening on http://{args.host}:{args.port}")
    print("[cloudflare-api] Expose with: cloudflared tunnel --url http://localhost:%d" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[cloudflare-api] Stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
