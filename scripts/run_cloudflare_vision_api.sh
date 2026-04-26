#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/maheshk/Documents/New project 2/candy_crush_mod_main_backup/.venv/bin/python}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
DOOR_PARTS_MODEL_PATH="${DOOR_PARTS_MODEL_PATH:-/tmp/joechencc_door_detection_review/yolov5/best.pt}"
DOOR_PARTS_YOLOV5_DIR="${DOOR_PARTS_YOLOV5_DIR:-/tmp/joechencc_door_detection_review/yolov5}"

cd "$ROOT_DIR"
DISABLE_CORE_MOTION="${DISABLE_CORE_MOTION:-1}" MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" "$PYTHON_BIN" -u -B cloudflare_vision_api.py \
  --host "$HOST" \
  --port "$PORT" \
  --door-parts-model "$DOOR_PARTS_MODEL_PATH" \
  --door-parts-yolov5-dir "$DOOR_PARTS_YOLOV5_DIR"
