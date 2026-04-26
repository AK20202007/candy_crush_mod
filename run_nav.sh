#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-Powell Library}"
CAMERA="${CAMERA:-0}"
PYTHON_BIN="${PYTHON_BIN:-/Users/maheshk/Documents/New project 2/candy_crush_mod_main_backup/.venv/bin/python}"

cd "$(dirname "$0")"
"$PYTHON_BIN" -B main.py --destination "$DESTINATION" --typed-destination --indoor-start yes --camera "$CAMERA" --preview
