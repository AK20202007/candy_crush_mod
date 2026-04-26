#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${EXPO_PUBLIC_NAV_API_BASE_URL:-}" ]]; then
  echo "Set EXPO_PUBLIC_NAV_API_BASE_URL to your trycloudflare URL first." >&2
  echo "Example: EXPO_PUBLIC_NAV_API_BASE_URL=https://basis-lisa-acc-korean.trycloudflare.com bash scripts/run_mobile_with_cloudflare_api.sh" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/mobile"

npm start
