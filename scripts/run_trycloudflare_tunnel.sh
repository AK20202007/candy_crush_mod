#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "Install cloudflared:  brew install cloudflared" >&2
  exit 1
fi

cloudflared tunnel --url "http://localhost:${PORT}"
