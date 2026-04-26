#!/usr/bin/env bash
# Public HTTPS URL -> local stream_server.py (default port 8765).
# Requires: brew install cloudflared
# Run stream_server first:  python3 stream_server.py
#
# Usage:
#   ./scripts/tunnel_stream_server.sh
#   PORT=9000 ./scripts/tunnel_stream_server.sh

set -euo pipefail
PORT="${PORT:-8765}"
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "Install cloudflared:  brew install cloudflared" >&2
  exit 1
fi
exec cloudflared tunnel --url "http://127.0.0.1:${PORT}"
